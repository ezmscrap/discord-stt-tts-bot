import io, os, wave, asyncio, tempfile, traceback, struct, math
import discord
import typing as T

from dotenv import load_dotenv
from discord import StageChannel, TextChannel, Thread
from discord.abc import Messageable
from discord.ext import commands, tasks
from gtts import gTTS
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_LANG = os.getenv("TTS_LANG", "ja")

if OPENAI_API_KEY:
    print(f"[STT] OPENAI_API_KEY detected: ****{OPENAI_API_KEY[-6:]}")
else:
    print("[STT] OPENAI_API_KEY NOT found (Whisperは動きません)")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True
intents.messages = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

guild_state = {}  # guild_id -> dict( read_channel_id, stt_on, record_window )

DEFAULT_WINDOW = 10  # 秒ごとに録音を区切って字幕化

async def resolve_display_name(guild: discord.Guild, user_id: int, data=None) -> str:
    # 1) Sink が user を持っていれば最優先
    u = getattr(data, "user", None)
    if u:
        if isinstance(u, discord.Member):
            return u.display_name
        # discord.User
        return getattr(u, "global_name", None) or u.name

    # 2) キャッシュ（Members Intent 有効ならここで取れる）
    m = guild.get_member(user_id)
    if m:
        return m.display_name

    # 3) API フェッチ（Guild Member）
    try:
        m = await guild.fetch_member(user_id)
        return m.display_name
    except Exception:
        pass

    # 4) グローバルユーザー
    u = bot.get_user(user_id)
    if u:
        return getattr(u, "global_name", None) or u.name
    try:
        u = await bot.fetch_user(user_id)
        return getattr(u, "global_name", None) or u.name
    except Exception:
        pass

    # 5) 最後の手段：IDの末尾だけ見せる
    return f"不明ユーザー({str(user_id)[-6:]})"

def _rms_from_frames(frames: bytes, sampwidth: int) -> float:
    """PCMフレーム（リトルエンディアン）からRMSを返す。戻り値は0.0〜1.0に正規化。"""
    if not frames:
        return 0.0

    if sampwidth == 1:
        # 8bitは unsigned。0..255 を -128..127 に変換してRMS
        n = len(frames)
        if n == 0: return 0.0
        acc = 0
        for b in frames:
            s = b - 128
            acc += s * s
        mean_sq = acc / n
        return math.sqrt(mean_sq) / 127.0

    elif sampwidth == 2:
        # 16bit signed little-endian
        cnt = len(frames) // 2
        if cnt == 0: return 0.0
        vals = struct.unpack("<%dh" % cnt, frames[:cnt*2])
        acc = 0
        for v in vals:
            acc += v * v
        mean_sq = acc / cnt
        return math.sqrt(mean_sq) / 32767.0

    elif sampwidth == 3:
        # 24bit signed little-endian（3バイトごとに読み取り）
        cnt = len(frames) // 3
        if cnt == 0: return 0.0
        acc = 0
        for i in range(cnt):
            b0 = frames[3*i]
            b1 = frames[3*i+1]
            b2 = frames[3*i+2]
            # 24bit符号拡張
            u = b0 | (b1 << 8) | (b2 << 16)
            if u & 0x800000:
                u = u - 0x1000000
            acc += u * u
        mean_sq = acc / cnt
        return math.sqrt(mean_sq) / 8388607.0  # 2^23-1

    elif sampwidth == 4:
        # 32bit signed little-endian
        cnt = len(frames) // 4
        if cnt == 0: return 0.0
        vals = struct.unpack("<%di" % cnt, frames[:cnt*4])
        acc = 0
        for v in vals:
            acc += v * v
        mean_sq = acc / cnt
        return math.sqrt(mean_sq) / 2147483647.0  # 2^31-1

    else:
        # 未対応の幅は0扱い
        return 0.0

def _pick_fallback_text_channel(g: discord.Guild) -> T.Optional[discord.TextChannel]:
    """そのギルドでBotが送信できる適当なTextChannelを返す"""
    if not g: 
        return None
    # 1) システムチャンネルが送信可なら優先
    sysch = g.system_channel
    if sysch:
        perms = sysch.permissions_for(g.me)
        if perms.view_channel and perms.send_messages:
            return sysch
    # 2) 他のテキストチャンネルで送信可なもの
    for ch in g.text_channels:
        perms = ch.permissions_for(g.me)
        if perms.view_channel and perms.send_messages:
            return ch
    return None

async def resolve_message_channel(channel_id: int, guild_id: int) -> T.Optional[discord.abc.Messageable]:
    """channel_id から「送信可能なMessageable」を返す。スレッドは必要ならunarchiveする。"""
    ch = bot.get_channel(channel_id)  # キャッシュ
    src = "cache"
    if ch is None:
        try:
            ch = await bot.fetch_channel(channel_id)  # API
            src = "api"
        except discord.Forbidden:
            print(f"[STT] fetch_channel forbidden for {channel_id}")
            ch = None
        except discord.NotFound:
            print(f"[STT] fetch_channel notfound for {channel_id}")
            ch = None
        except Exception as e:
            print("[STT] fetch_channel failed:", repr(e))
            ch = None

    if ch is not None:
        print(f"[STT] resolve hit ({src}): type={type(ch).__name__} id={getattr(ch,'id',None)}")
        # スレッドなら必要に応じてunarchive
        if isinstance(ch, discord.Thread) and ch.archived:
            try:
                await ch.edit(archived=False, locked=False)
                print("[STT] thread unarchived")
            except Exception as e:
                print("[STT] thread unarchive failed:", repr(e))
        # Messageable（.sendできる）ならOK
        if isinstance(ch, Messageable) or hasattr(ch, "send"):
            return ch

    # ここまでで解決できないなら、同ギルドでフォールバック
    g = bot.get_guild(guild_id)
    fb = _pick_fallback_text_channel(g)
    if fb:
        print(f"[STT] fallback to #{fb.name} ({fb.id})")
        return fb

    print("[STT] no messageable channel available")
    return None

def wav_stats(src):
    """
    src: パス/bytes/BytesIO/file-like を受け取り、
    (duration_sec, normalized_rms) を返す。rmsは0.0〜1.0。
    """
    need_close = False
    if isinstance(src, (str, os.PathLike)):
        f = open(src, "rb"); need_close = True
    elif isinstance(src, (bytes, bytearray, memoryview)):
        f = io.BytesIO(src)
    else:
        f = src  # file-like
    try:
        try: f.seek(0)
        except: pass
        with wave.open(f, "rb") as wf:
            nframes = wf.getnframes()
            fr = wf.getframerate()
            sw = wf.getsampwidth()
            dur = (nframes / fr) if fr else 0.0
            wf.rewind()
            # 先頭10秒分だけでRMSを計算
            frames = wf.readframes(min(nframes, fr * 10))
            rms_norm = _rms_from_frames(frames, sw)
        return dur, rms_norm
    finally:
        if need_close:
            f.close()

async def transcribe_and_post(src, channel, username: str):
    if not openai:
        print("[STT] OpenAI client is None"); return
    tmp = None; fh = None
    try:
        # デバッグ: 区切りの長さ・音量
        try:
            dur, rms = wav_stats(src)
            print(f"[STT] segment stats: dur={dur:.2f}s rms={rms}")
        except Exception:
            traceback.print_exc()

        # Whisperへはファイルで渡す（BytesIOは一時ファイル化）
        if isinstance(src, (str, os.PathLike)):
            fh = open(src, "rb")
        else:
            if hasattr(src, "read"):
                try: src.seek(0)
                except: pass
                buf = src.read()
            else:
                buf = bytes(src)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp = tf.name; tf.write(buf); tf.close()
            fh = open(tmp, "rb")

        resp = openai.audio.transcriptions.create(
            file=fh, model="whisper-1", language="ja"
        )
        text = (getattr(resp, "text", "") or "").strip()
        print(f"[STT] Whisper result: {text!r}")
        if text:
            await channel.send(f"🎤 **{username}**: {text}")
    except Exception as e:
        print("[STT] Transcription failed:", repr(e))
        traceback.print_exc()
    finally:
        try:
            if fh: fh.close()
        finally:
            if tmp:
                try: os.remove(tmp)
                except: pass

@bot.command(name="rectest", aliases=["録音テスト"])
async def rectest(ctx: commands.Context, seconds: int = 5):
    """現在のボイスCHで seconds 秒だけ録音し、WAVを添付して返す"""
    vc = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        return await ctx.reply("先に `!join` してください。")

    if seconds < 2 or seconds > 30:
        return await ctx.reply("録音秒数は 2〜30 秒の範囲で指定してください。 例: `!rectest 5`")

    # WaveSink でユーザー別にWAVを生成
    sink = discord.sinks.WaveSink()
    done = asyncio.Event()

    async def finished_callback(sink, *args):
        try:
            files_info = []
            attachments = []

            for user_id, data in sink.audio_data.items():
                fileobj = data.file  # これがパス or BytesIO
                name = await resolve_display_name(g, int(user_id), data)
                fname = f"{name.replace(' ', '_')}.wav"

                # 分岐：パス or メモリ
                if isinstance(fileobj, (str, os.PathLike)):
                    dur, rms = wav_stats(fileobj)
                    size = os.path.getsize(fileobj)
                    attachments.append(discord.File(fileobj, filename=fname))
                else:
                    # BytesIO など file-like
                    try:
                        fileobj.seek(0)
                    except Exception:
                        pass
                    buf = fileobj.read() if hasattr(fileobj, "read") else bytes(fileobj)
                    size = len(buf)
                    dur, rms = wav_stats(buf)
                    bio = io.BytesIO(buf); bio.seek(0)
                    attachments.append(discord.File(bio, filename=fname))

                files_info.append((name, dur, rms, size))

            if not files_info:
                await ctx.reply("⚠️ 録音できませんでした。誰も話していない/ボットが聴覚遮断の可能性があります。")
            else:
                lines = ["🎧 **録音テスト結果**"]
                for name, dur, rms, size in files_info:
                    lines.append(f"- {name}: {dur:.2f}s, RMS={rms}, {size/1024:.1f}KB")
                await ctx.reply("\n".join(lines))
                # 添付送信（サイズが大きいと失敗する場合あり）
                for f in attachments:
                    try:
                        await ctx.send(file=f)
                    except Exception as e:
                        await ctx.send(f"（添付失敗: {getattr(f, 'filename', 'file')} / {e!r}）")
        finally:
            try:
                vc.stop_recording()
            except:
                pass
            done.set()

    try:
        vc.start_recording(sink, finished_callback)
    except Exception as e:
        traceback.print_exc()
        return await ctx.reply(f"録音開始に失敗しました: {e!r}")

    await ctx.reply(f"🎙️ {seconds} 秒だけ録音します。話しかけてください…")
    await asyncio.sleep(seconds)

    try:
        vc.stop_recording()
    except:
        pass

    await done.wait()

def get_state(guild_id):
    if guild_id not in guild_state:
        guild_state[guild_id] = dict(read_channel_id=None, stt_on=False, record_window=DEFAULT_WINDOW, stt_task=None)
    return guild_state[guild_id]

def sanitize_for_tts(text: str) -> str:
    import re
    text = re.sub(r"<@!?\d+>", "メンション", text)
    text = re.sub(r"<@&\d+>", "ロールメンション", text)
    text = re.sub(r"<#\d+>", "チャンネル", text)
    text = re.sub(r"https?://\S+", "リンク", text)
    return text[:400]

async def tts_play(guild: discord.Guild, text: str):
    vc: discord.VoiceClient = guild.voice_client
    if not vc or not vc.is_connected():
        return
    # gTTS → mp3 → FFmpegPCMAudio で再生
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    try:
        gTTS(text=sanitize_for_tts(text), lang=TTS_LANG).save(tmp_path)
        audio = discord.FFmpegPCMAudio(tmp_path)
        vc.play(audio)
        while vc.is_playing():
            await asyncio.sleep(0.2)
    finally:
        try: os.remove(tmp_path)
        except: pass

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (py-cord)")
    for g in bot.guilds:
        get_state(g.id)

@bot.command(name="join", aliases=["執事参加", "執事入室", "執事召喚"])
async def join(ctx: commands.Context):
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.reply("先にボイスチャンネルへ入室してください。")
    channel = ctx.author.voice.channel
    vc = ctx.guild.voice_client
    if vc and vc.is_connected():
        await vc.move_to(channel)
    else:
        vc = await channel.connect()

    # 🔧 Stage だったら話者化を試みる（失敗しても続行）
    if isinstance(channel, StageChannel):
        try:
            # 話者に昇格（権限が必要。無い場合は except へ）
            await ctx.guild.change_voice_state(channel=channel, suppress=False)
            # うまくいかない環境では「話させてください」リクエスト
            await ctx.guild.change_voice_state(channel=channel, request_to_speak=True)
            await ctx.reply("Stage で話者化を試みました。（必要ならモデレーターが承認してください）")
        except Exception as e:
            print("[join] Stage unsuppress/request_to_speak failed:", repr(e))
            await ctx.reply("このチャンネルは Stage のようです。録音には通常のボイスチャンネル推奨です。")

    st = get_state(ctx.guild.id)
    st["read_channel_id"] = ctx.channel.id
    # 診断上の見栄え用（ワーカーはこれに依存しませんが True にしておく）
    st["stt_on"] = False
    await ctx.reply(f"Joined **{channel.name}**。このチャンネルを読み上げ対象に設定しました。")

@bot.command(name="diag", aliases=["診断"])
async def diag(ctx: commands.Context):
    import shutil, platform
    vc = ctx.guild.voice_client
    mevoice = getattr(ctx.guild.me, "voice", None)
    ch_type = type(vc.channel).__name__ if (vc and vc.channel) else "None"
    try:
        pynacl_ok = True
        import nacl  # PyNaCl
    except Exception:
        pynacl_ok = False

    lines = [
        f"py-cord: {discord.__version__}",
        f"ffmpeg: {'OK' if shutil.which('ffmpeg') else 'NG'}",
        f"PyNaCl import: {pynacl_ok}",
        f"Opus loaded: {discord.opus.is_loaded()}",
        f"OPENAI_API_KEY: {'FOUND' if OPENAI_API_KEY else 'MISSING'}",
        f"Voice connected: {bool(vc and vc.is_connected())}",
        f"Bot self_deaf: {getattr(mevoice, 'self_deaf', None)}",
        f"STT on: {get_state(ctx.guild.id)['stt_on']}",
        f"Voice channel type: {ch_type}",
        f"Record window: {get_state(ctx.guild.id)['record_window']}s",
        f"OS: {platform.platform()}",
    ]
    await ctx.reply("🔎 **診断**\n" + "\n".join(f"- {x}" for x in lines))

@bot.command(name="whereami")
async def whereami(ctx: commands.Context):
    ch = ctx.channel
    parent = getattr(ch, "parent", None)
    await ctx.reply(
        "📌 **ここは？**\n"
        f"- type={type(ch).__name__}\n"
        f"- id={getattr(ch, 'id', None)}\n"
        f"- name={getattr(ch, 'name', None)}\n"
        f"- parent={getattr(parent, 'name', None)} ({getattr(parent, 'id', None)})"
    )

@bot.command(name="stttest", aliases=["文字起こしテスト"])
async def stttest(ctx: commands.Context):
    if not openai:
        return await ctx.reply("OPENAI_API_KEY が未設定です。`.env` に設定し、再起動してください。")
    import tempfile
    from gtts import gTTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    gTTS(text="これはテストです。音声認識の確認をしています。", lang=TTS_LANG).save(tmp)
    try:
        with open(tmp, "rb") as audio:
            resp = openai.audio.transcriptions.create(
                file=audio, model="whisper-1", language="ja"
            )
        await ctx.reply(f"✅ Whisper応答: `{resp.text}`")
    except Exception as e:
        await ctx.reply(f"❌ Whisper失敗: {e!r}")
    finally:
        try: os.remove(tmp)
        except: pass

@bot.command(name="leave", aliases=["執事退出", "執事離脱"])
async def leave(ctx: commands.Context):
    vc = ctx.guild.voice_client
    if vc and vc.is_connected():
        await vc.disconnect(force=True)
    st = get_state(ctx.guild.id)
    st["read_channel_id"] = None
    st["stt_on"] = False
    await ctx.reply("Left the voice channel.")

@bot.command(name="readon", aliases=["読み上げコマンド", "読み上げ", "読み上げ開始", "読み上げオン", "このチャンネルを読み上げ"])
async def readon(ctx: commands.Context):
    if not ctx.guild.voice_client or not ctx.guild.voice_client.is_connected():
        return await ctx.reply("先に `!join` してください。")
    st = get_state(ctx.guild.id)
    st["read_channel_id"] = ctx.channel.id
    await ctx.reply("このチャンネルの新規投稿を読み上げます。`!readoff` で停止。")

@bot.command(name="readoff", aliases=["読み上げ停止", "読み上げオフ"])
async def readoff(ctx: commands.Context):
    st = get_state(ctx.guild.id)
    st["read_channel_id"] = None
    await ctx.reply("読み上げを停止しました。")

@bot.command(name="stton", aliases=["字幕開始","文字起こし開始","字幕オン","音声認識開始"])
async def stton(ctx: commands.Context, window: int | None = None):
    vc = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        return await ctx.reply("先に `!join` してください。")
    st = get_state(ctx.guild.id)
    if window and 3 <= window <= 60:
        st["record_window"] = window

    # 既存タスク停止
    if st.get("stt_task") and not st["stt_task"].done():
        st["stt_task"].cancel()
        try: await st["stt_task"]
        except: pass

    # 🔎 送信先チャンネルを“今ここ”から解決（VoiceChannelならフォールバック）
    dest = await resolve_message_channel(ctx.channel.id, ctx.guild.id)
    if dest is None:
        return await ctx.reply("送信可能なテキストチャンネルが見つかりませんでした。権限をご確認ください。")

    print(f"[STT] stton from channel: id={ctx.channel.id} type={type(ctx.channel).__name__} -> post to id={dest.id} type={type(dest).__name__}")

    # ワーカーには “解決済みの送信先ID” を渡す
    st["stt_task"] = asyncio.create_task(stt_worker(ctx.guild.id, dest.id))
    st["stt_on"] = True
    await ctx.reply(f"🎧 音声認識を開始（{st['record_window']}秒区切り）。投稿先: <#{dest.id}> / OpenAI鍵: {'あり' if openai else 'なし'}")


@bot.command(name="sttoff", aliases=["字幕停止","文字起こし停止","字幕オフ","音声認識停止"])
async def sttoff(ctx: commands.Context):
    st = get_state(ctx.guild.id)
    if st.get("stt_task") and not st["stt_task"].done():
        st["stt_task"].cancel()
        try: await st["stt_task"]
        except: pass
    st["stt_task"] = None
    await ctx.reply("音声認識を停止しました。")

@bot.command(name="readhere", aliases=["ここを読み上げ"])
async def readhere(ctx: commands.Context):
    if not ctx.guild.voice_client or not ctx.guild.voice_client.is_connected():
        return await ctx.reply("先に `!join` してください。")
    st = get_state(ctx.guild.id)
    st["read_channel_id"] = ctx.channel.id
    await ctx.reply("このチャンネルを読み上げ対象に設定しました。`!readoff` で停止。")

@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)
    if not message.guild or message.author.bot:
        return
    
    # コマンドは読まないようにする
    text = (message.content or "").strip()
    if text.startswith(("!", "！")):
        return
    
    st = get_state(message.guild.id)
    if st["read_channel_id"] == message.channel.id and text:
        display = message.author.display_name if isinstance(message.author, discord.Member) else message.author.name
        to_say = f"{display}：{text}"
        await tts_play(message.guild, to_say)
    


async def stt_worker(guild_id: int, channel_id: int):
    g = bot.get_guild(guild_id)
    if not g:
        return
    print("[STT] worker start", guild_id, channel_id)
    try:
        while True:
            vc = g.voice_client
            if not vc or not vc.is_connected():
                print("[STT] no voice connection; retry")
                await asyncio.sleep(1.0)
                continue

            ch = await resolve_message_channel(channel_id, guild_id)
            if ch is None:
                print("[STT] message channel not found; retry")
                await asyncio.sleep(2.0)
                continue

            # どのタイプのVCにいるかを可視化
            try:
                ch_type = type(vc.channel).__name__
                print(f"[STT] in voice channel type = {ch_type}")
            except Exception:
                pass

            sink = discord.sinks.WaveSink()
            done = asyncio.Event()
            captured = []

            def _collect_filelike(fileobj) -> bytes:
                if isinstance(fileobj, (str, os.PathLike)):
                    with open(fileobj, "rb") as rf:
                        return rf.read()
                try:
                    fileobj.seek(0)
                except Exception:
                    pass
                return fileobj.read() if hasattr(fileobj, "read") else bytes(fileobj)

            async def finished_callback(sink, *args):
                try:
                    for user_id, data in sink.audio_data.items():
                        name = await resolve_display_name(g, int(user_id), data)
                        buf = _collect_filelike(data.file)
                        captured.append((name, buf))
                finally:
                    done.set()

            try:
                print("[STT] start_recording()")
                vc.start_recording(sink, finished_callback)
            except Exception as e:
                print("[STT] start_recording failed:", repr(e))
                await asyncio.sleep(2.0)
                continue

            window = get_state(guild_id)["record_window"]
            await asyncio.sleep(window)

            # 停止は同期関数
            try:
                print("[STT] stop_recording()")
                vc.stop_recording()
            except Exception as e:
                print("[STT] stop_recording failed:", repr(e))

            await done.wait()

            if not captured:
                print("[STT] no audio captured in this window")
                await asyncio.sleep(0.3)
                continue

            # Whisper へ
            jobs = [transcribe_and_post_from_bytes(buf, ch, username) for (username, buf) in captured]
            await asyncio.gather(*jobs, return_exceptions=True)

    except asyncio.CancelledError:
        print("[STT] worker cancelled", guild_id)
    except Exception as e:
        print("[STT] worker crashed:", repr(e))
        traceback.print_exc()
    finally:
        print("[STT] worker end", guild_id)

@bot.command(name="intentcheck")
async def intentcheck(ctx):
    # コード側の意図（bool）
    code_flag = bot.intents.members

    # キャッシュ/フェッチの実挙動
    cache_hit = ctx.guild.get_member(ctx.author.id) is not None
    try:
        fetched = await ctx.guild.fetch_member(ctx.author.id)
        fetch_ok = fetched is not None
        err = None
    except Exception as e:
        fetch_ok = False
        err = repr(e)

    await ctx.reply(
        "🧪 intents.members(check)\n"
        f"- code_flag: {code_flag}\n"
        f"- cache_has_author: {cache_hit}\n"
        f"- fetch_member_ok: {fetch_ok}\n"
        f"- fetch_error: {err}"
    )

async def record_once(guild: discord.Guild, seconds: int):
    """
    seconds 秒だけ録音して、[(username, bytes)] を返す。
    """
    vc: discord.VoiceClient = guild.voice_client
    if not vc or not vc.is_connected():
        return []

    sink = discord.sinks.WaveSink()
    done = asyncio.Event()
    results: list[tuple[str, bytes]] = []

    async def finished_callback(sink, *args):
        try:
            for user_id, data in sink.audio_data.items():
                name = await resolve_display_name(g, int(user_id), data)
                fileobj = data.file  # path or BytesIO

                # bytes へ落とす
                if isinstance(fileobj, (str, os.PathLike)):
                    with open(fileobj, "rb") as rf:
                        buf = rf.read()
                else:
                    try: fileobj.seek(0)
                    except: pass
                    buf = fileobj.read() if hasattr(fileobj, "read") else bytes(fileobj)

                results.append((name, buf))
        finally:
            try: vc.stop_recording()
            except: pass
            done.set()

    try:
        vc.start_recording(sink, finished_callback)
    except Exception as e:
        print("[STT] start_recording failed:", repr(e))
        return []

    await asyncio.sleep(seconds)
    try: 
        vc.stop_recording()
    except: 
        pass
    await done.wait()
    return results

async def transcribe_and_post_from_bytes(buf: bytes, channel, username: str):
    print(f"start: transcribe_and_post_from_bytes")
    if not openai:
        print("[STT] OpenAI client is None"); return
    tmp = None; fh = None
    try:
        # デバッグ（出なくても動作には影響しない）
        try:
            dur, rms = wav_stats(buf)
            print(f"[STT] segment stats: dur={dur:.2f}s rms={rms:.3f}")
        except Exception:
            traceback.print_exc()

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp = tf.name; tf.write(buf); tf.close()
        fh = open(tmp, "rb")

        resp = openai.audio.transcriptions.create(
            file=fh, model="whisper-1", language="ja"
        )
        text = (getattr(resp, "text", "") or "").strip()
        print(f"[STT] Whisper result: {text!r}")
        if text:
            try:
                await channel.send(f"🎤 **{username}**: {text}")
            except Exception as e:
                print("[STT] send failed:", repr(e))
    except Exception as e:
        print("[STT] Transcription failed:", repr(e))
        traceback.print_exc()
    finally:
        try:
            if fh: fh.close()
        finally:
            if tmp:
                try: os.remove(tmp)
                except: pass

bot.run(TOKEN)
