import io, os, wave, asyncio, tempfile, traceback, struct, math,re, time,csv
import discord
import typing as T

from datetime import datetime, timezone
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

LOG_DIR = BASE_DIR / "logs"
TTS_LOG_PATH = LOG_DIR / "tts_logs.csv"
STT_LOG_PATH = LOG_DIR / "stt_logs.csv"
_log_lock = asyncio.Lock()  # 複数タスクからの同時書き込みを保護

def _ensure_csv_with_header(path: Path, headers: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)

# 初期化（ヘッダ行を用意）
_ensure_csv_with_header(
    TTS_LOG_PATH,
    ["timestamp_iso", "guild_id", "channel_id", "message_id", "author_id", "author_display", "text"],
)
_ensure_csv_with_header(
    STT_LOG_PATH,
    ["timestamp_iso", "guild_id", "dest_channel_id", "user_id", "user_display", "text", "duration_sec", "rms", "dbfs"],
)

def _norm_text_for_csv(text: str) -> str:
    return (text or "").replace("\r", " ").replace("\n", " ").strip()

async def _append_csv(path: Path, row: list):
    async with _log_lock:
        # 失敗しても bot 全体を止めない
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row)
        except Exception as e:
            print(f"[LOG] write failed for {path.name}:", repr(e))

async def log_tts_event(message: discord.Message, spoken_text: str):
    """読み上げたテキストのログ（入力者・入力時刻付き）"""
    ts = message.created_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    row = [
        ts.astimezone(timezone.utc).isoformat(),
        message.guild.id if message.guild else 0,
        message.channel.id if hasattr(message, "channel") else 0,
        message.id,
        message.author.id if message.author else 0,
        (message.author.display_name if isinstance(message.author, discord.Member) else getattr(message.author, "name", "unknown")),
        _norm_text_for_csv(spoken_text),
    ]
    await _append_csv(TTS_LOG_PATH, row)

async def log_stt_event(
    guild_id: int,
    dest_channel_id: int | None,
    user_id: int,
    user_display: str,
    text: str,
    duration: float | None,
    rms: float | None,
    dbfs: float | None,
):
    """音声認識結果のログ（発言者・発言時間（記録時刻）付き）"""
    ts = datetime.now(timezone.utc).isoformat()
    row = [
        ts,
        guild_id or 0,
        dest_channel_id or 0,
        user_id,
        user_display,
        _norm_text_for_csv(text),
        f"{duration:.3f}" if isinstance(duration, (int, float)) else "",
        f"{rms:.6f}" if isinstance(rms, (int, float)) else "",
        f"{dbfs:.2f}" if isinstance(dbfs, (int, float)) else "",
    ]
    await _append_csv(STT_LOG_PATH, row)


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

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

guild_state = {}  # guild_id -> dict( read_channel_id, stt_on, record_window )

DEFAULT_WINDOW = 10  # 秒ごとに録音を区切って字幕化

# === 追加：話速・声色プロファイルとFFmpegフィルタ生成 ===
VOICE_PROFILES = [
    {"name": "alto",     "semitones": -2, "tempo": 1.15},  # ちょい低め・やや速い
    {"name": "neutral",  "semitones":  0, "tempo": 1.25},  # 標準ピッチ・速い
    {"name": "bright",   "semitones": +4, "tempo": 1.20},  # 高め・少し速い
    {"name": "deep",     "semitones": -5, "tempo": 1.12},  # 低め・少し速い
]

def _atempo_chain(x: float) -> list[str]:
    # FFmpegの atempo は 0.5〜2.0 の範囲なので分割
    chain = []
    while x > 2.0:
        chain.append("atempo=2.0")
        x /= 2.0
    while x < 0.5:
        chain.append("atempo=0.5")
        x /= 0.5
    chain.append(f"atempo={x:.4f}")
    return chain

def _build_ffmpeg_afilter(semitones: float, final_tempo: float) -> str:
    """
    semitones: ピッチ上下（+で高く）
    final_tempo: 最終的な話速倍率（>1で速い）
    ピッチは asetrate で上げ下げ → atempo で速度を調整。
    """
    # ピッチ係数（半音×12 → 2^(n/12)）
    pitch_factor = 2.0 ** (semitones / 12.0)
    # asetrate で速度も pitch_factor 倍になるので、atempo で目標話速へ補正
    # つまり total_atempo = final_tempo / pitch_factor
    total_atempo = final_tempo / max(pitch_factor, 1e-6)

    # サンプルレートは 48kHz に統一（Discord向けに安定）
    parts = [f"asetrate=48000*{pitch_factor:.6f}", "aresample=48000"]
    parts += _atempo_chain(total_atempo)
    # カンマで連結・スペース不要（Windowsのffmpegでのクォート回避）
    return ",".join(parts)

def _dbfs_from_rms(rms: float) -> float:
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)

def jp_cleanup(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return t
    # 末尾に句読点が無ければ「。」を付ける（英数で終わるなら付けない）
    if not re.search(r"[。！？!?]$", t) and re.search(r"[ぁ-んァ-ン一-龥]", t):
        t += "。"
    return t

async def post_caption(guild_id: int, channel, user_id: int, username: str, new_text: str):
    st = get_state(guild_id)
    now = time.monotonic()
    ch_id = str(getattr(channel, "id", 0))
    key_u = str(user_id)

    ch_map = st["last_msgs"].setdefault(ch_id, {})
    entry = ch_map.get(key_u)

    # 直近メッセージがあって merge_window 内なら編集で追記
    if entry and entry.get("message") and (now - entry.get("ts", 0)) < st["merge_window"]:
        try:
            base = entry["message"].content
            # 先頭の「🎤 **名前**: 」を保ったまま後ろに文章を足す
            # baseが空でない前提で半角スペース区切り
            merged = (base + " " + new_text).strip()
            await entry["message"].edit(content=merged)
            entry["ts"] = now
            return
        except Exception as e:
            print("[STT] edit failed; fallback send:", repr(e))

    # 新規投稿
    m = await channel.send(f"🎤 **{username}**: {new_text}")
    ch_map[key_u] = {"message": m, "ts": now}

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
            # だれのトラックが生成されたかを一覧表示
            print("[STT] users in window:", list(sink.audio_data.keys()))
            for user_id, data in sink.audio_data.items():
                uid = int(user_id)

                # どのくらい録れたか（py-cordのAudioDataはbyte_countを持っているはず）
                byte_count = getattr(data, "byte_count", None)

                # fileサイズ（WAVならヘッダ込みサイズ）
                size = None
                f = data.file
                if isinstance(f, (str, os.PathLike)):
                    try: size = os.path.getsize(f)
                    except: size = None
                else:
                    try:
                        pos = f.tell()
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(0)
                    except Exception:
                        size = None

                print(f"[STT] capture stat uid={uid} byte_count={byte_count} size={size}")

                # 典型的な「空WAV」（ヘッダだけ ≒ 44バイト）や byte_count==0 は弾く
                if (byte_count is not None and byte_count == 0) or (size is not None and size <= 44):
                    continue

                # 実データだけ追加
                buf = _collect_filelike(data.file)
                captured.append((uid, data, buf))
        finally:
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

def _pick_voice_profile_for_user(guild_id: int, user_id: int | None) -> dict:
    """ギルド設定の override を最優先。なければVOICESを user_id で安定割当。"""
    st = get_state(guild_id)
    if user_id is not None:
        ov = st["tts_overrides"].get(int(user_id))
        if ov:  # 明示オーバーライド
            return {"name": "custom", "semitones": ov.get("semitones", 0.0), "tempo": ov.get("tempo", 1.0)}
        # 自動割当
        base = VOICE_PROFILES[user_id % len(VOICE_PROFILES)]
        return base
    return {"name": "neutral", "semitones": 0.0, "tempo": 1.0}

def get_state(guild_id):
    if guild_id not in guild_state:
        guild_state[guild_id] = dict(
            read_channel_id=None,
            stt_on=False,
            record_window=DEFAULT_WINDOW,
            stt_task=None,
            vad_rms=0.02,
            min_dur=0.8,
            merge_window=6.0,
            lang="ja",
            use_thread=False,
            caption_dest_id=None,
            last_msgs={},
            rec_lock=asyncio.Lock(),
            tts_base_tempo=float(os.getenv("TTS_TEMPO", "0.7")),  # サーバー全体の基準話速
            tts_overrides={},   # { user_id: {"semitones": float, "tempo": float} }
        )
    return guild_state[guild_id]


async def ensure_stopped(vc: discord.VoiceClient, why: str = ""):
    """録音が残っていれば強制停止して、少し待つ"""
    try:
        rec_flag = getattr(vc, "recording", False)
        print(f"[STT] ensure_stopped({why}) recording={rec_flag}")
        if rec_flag:
            try:
                vc.stop_recording()
                print("[STT] forced stop_recording()")
            except Exception as e:
                print("[STT] forced stop failed:", repr(e))
        await asyncio.sleep(0.25)  # フラッシュ待ち
    except Exception as e:
        print("[STT] ensure_stopped error:", repr(e))

def sanitize_for_tts(text: str) -> str:
    import re
    text = re.sub(r"<@!?\d+>", "メンション", text)
    text = re.sub(r"<@&\d+>", "ロールメンション", text)
    text = re.sub(r"<#\d+>", "チャンネル", text)
    text = re.sub(r"https?://\S+", "リンク", text)
    return text[:400]

async def tts_play(guild: discord.Guild, text: str, speaker_id: int | None = None):
    vc: discord.VoiceClient = guild.voice_client
    if not vc or not vc.is_connected():
        return

    st = get_state(guild.id)

    prof = _pick_voice_profile_for_user(guild.id, speaker_id)
    # サーバー基準 × 各話者のテンポ（安全にクリップ）
    final_tempo = st["tts_base_tempo"] * prof.get("tempo", 1.0)
    final_tempo = max(0.5, min(2.5, final_tempo))
    semitones = float(prof.get("semitones", 0.0))

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    try:
        gTTS(text=sanitize_for_tts(text), lang=TTS_LANG).save(tmp_path)
        af = _build_ffmpeg_afilter(semitones=semitones, final_tempo=final_tempo)
        audio = discord.FFmpegPCMAudio(tmp_path, options=f"-vn -af {af}")
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
        if st.get("merge_auto", True):
            # 認識窓より少し長く（1.25倍 + 余裕）
            st["merge_window"] = max(st["merge_window"], round(window * 1.25, 2))

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
    # ★ 念のため停止
    vc = ctx.guild.voice_client
    if vc and vc.is_connected():
        await ensure_stopped(vc, "manual off")
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
        await tts_play(message.guild, to_say, speaker_id=message.author.id)

        # ★ ログ: 読み上げたテキスト（元入力）・投稿者・入力時間
        # 「読み上げたテキスト」は message.content（TTS前の生テキスト）を残すのが要件に忠実
        await log_tts_event(message, text)

@bot.command(name="ttsspeed", aliases=["読み上げ速度"])
async def ttsspeed(ctx: commands.Context, ratio: str = None):
    if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
        return await ctx.reply("このコマンドはサーバー管理者のみ実行できます。")
    if not ratio:
        return await ctx.reply("使い方: `!ttsspeed 1.35`  （推奨: 0.6〜2.0）")

    try:
        r = float(ratio)
        if not (0.4 <= r <= 3.0):
            return await ctx.reply("値が広すぎます。0.4〜3.0 の範囲で指定してください（推奨 0.6〜2.0）。")
    except Exception:
        return await ctx.reply("数値で指定してください。例: `!ttsspeed 1.25`")

    st = get_state(ctx.guild.id)
    st["tts_base_tempo"] = r
    await ctx.reply(f"✅ サーバー基準の読み上げ話速を **{r:.2f}倍** に設定しました。")

@bot.command(name="ttsvoice", aliases=["声色"])
async def ttsvoice(ctx: commands.Context, member: discord.Member = None, semitones: str = None, tempo: str = None):
    if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
        return await ctx.reply("このコマンドはサーバー管理者のみ実行できます。")

    if member is None or semitones is None:
        return await ctx.reply(
            "使い方:\n"
            "- `!ttsvoice @ユーザー +3 1.15`  … 半音+3 / テンポ1.15倍\n"
            "- `!ttsvoice @ユーザー reset`   … 個別設定を解除\n"
            "  ※テンポは省略可（省略時は1.0）"
        )

    st = get_state(ctx.guild.id)

    if semitones.lower() == "reset":
        st["tts_overrides"].pop(member.id, None)
        return await ctx.reply(f"🔄 {member.display_name} の個別声設定をリセットしました。")

    # "+3" や "-5" などに対応
    try:
        if semitones.startswith(("+", "-")):
            semi = float(semitones)
        else:
            semi = float(semitones)  # "3" も許可
    except Exception:
        return await ctx.reply("半音は数値で指定してください（例: +3, -2, 0）。")

    try:
        t = 1.0 if tempo is None else float(tempo)
        if not (0.5 <= (t * st["tts_base_tempo"]) <= 2.5):
            # 実効話速（サーバー基準×個別）の安全範囲をざっくりチェック
            pass
    except Exception:
        return await ctx.reply("テンポは数値で指定してください（例: 1.10）。")

    st["tts_overrides"][member.id] = {"semitones": semi, "tempo": t}
    await ctx.reply(
        f"✅ {member.display_name} の声色を設定しました： 半音 **{semi:+.1f}**, テンポ係数 **{t:.2f}**"
    )

@bot.command(name="ttsconfig", aliases=["読み上げ設定"])
async def ttsconfig(ctx: commands.Context):
    st = get_state(ctx.guild.id)
    lines = [
        f"🔧 **TTS設定**",
        f"- サーバー基準話速: x{st['tts_base_tempo']:.2f}",
        f"- 個別設定数: {len(st['tts_overrides'])}",
    ]
    if st["tts_overrides"]:
        lines.append("- 個別設定（最大10件表示）:")
        for uid, ov in list(st["tts_overrides"].items())[:10]:
            m = ctx.guild.get_member(uid)
            name = m.display_name if m else f"User {uid}"
            lines.append(f"  • {name}: semitones={ov.get('semitones',0):+.1f}, tempo={ov.get('tempo',1.0):.2f}")
        if len(st["tts_overrides"]) > 10:
            lines.append(f"  …ほか {len(st['tts_overrides']) - 10} 件")
    await ctx.reply("\n".join(lines))

@bot.command(name="sttset")
async def sttset(ctx, key: str=None, value: str=None):
    """
      !sttset vad 0.008
      !sttset vaddb -46
      !sttset mindur 0.4
      !sttset merge 14
      !sttset mergeauto on/off
      !sttset lang auto
      !sttset thread on
    """
    st = get_state(ctx.guild.id)
    if not key:
        return await ctx.reply(
            ("設定: vad={vad_rms} vaddb={vad_db} mindur={min_dur}s "
             "merge={merge_window}s mergeauto={merge_auto} lang={lang} thread={use_thread}").format(**st)
        )

    try:
        k = key.lower()
        if k == "vad":
            st["vad_rms"] = float(value)
        elif k in ("vaddb","db"):
            st["vad_db"] = float(value)
        elif k in ("mindur","min"):
            st["min_dur"] = float(value)
        elif k in ("merge","mw"):
            st["merge_window"] = float(value)
        elif k in ("mergeauto","ma"):
            st["merge_auto"] = (value.lower() in ("on","true","1","yes","y"))
        elif k == "lang":
            st["lang"] = value.lower()
        elif k in ("thread","th"):
            st["use_thread"] = (value.lower() in ("on","true","1","yes","y"))
            st["caption_dest_id"] = None
        else:
            return await ctx.reply("未知のキー: vad / vaddb / mindur / merge / mergeauto / lang / thread")
    except Exception as e:
        return await ctx.reply(f"設定失敗: {e!r}")

    await ctx.reply(
        ("OK: vad={vad_rms} vaddb={vad_db} mindur={min_dur}s "
         "merge={merge_window}s mergeauto={merge_auto} lang={lang} thread={use_thread}").format(**st)
    )


async def stt_worker(guild_id: int, channel_id: int):
    guild_obj = bot.get_guild(guild_id)
    if not guild_obj:
        return
    print("[STT] worker start", guild_id, channel_id)
    st = get_state(guild_id)

    try:
        while True:
            vc = guild_obj.voice_client
            if not vc or not vc.is_connected():
                print("[STT] no voice connection; retry")
                await asyncio.sleep(1.0)
                continue

            # 投稿先解決
            base = await resolve_message_channel(channel_id, guild_id)
            if base is None:
                print("[STT] message channel not found; retry")
                await asyncio.sleep(2.0)
                continue

            # スレッド（必要なら）
            dest = base
            if st["use_thread"]:
                try:
                    if st.get("caption_dest_id"):
                        t = await resolve_message_channel(st["caption_dest_id"], guild_id)
                        if isinstance(t, discord.Thread):
                            dest = t
                        else:
                            st["caption_dest_id"] = None
                    if st.get("caption_dest_id") is None and isinstance(base, discord.TextChannel):
                        th = await base.create_thread(name="🎤字幕", auto_archive_duration=60)
                        st["caption_dest_id"] = th.id
                        dest = th
                except Exception as e:
                    print("[STT] thread create/resolve failed:", repr(e))
                    dest = base

            # ===== 録音 1 サイクル =====
            async with st["rec_lock"]:  # ★ 同時実行をブロック
                # もし取り残しがあれば止める
                await ensure_stopped(vc, "before start")

                sink = discord.sinks.WaveSink()
                done = asyncio.Event()
                captured: list[tuple[int, object, bytes]] = []

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
                        # デバッグ：どのユーザーが来たか
                        print("[STT] users in window:", list(sink.audio_data.keys()))
                        for user_id, data in sink.audio_data.items():
                            uid = int(user_id)
                            f = data.file
                            # 空WAVは弾く
                            size = None
                            if isinstance(f, (str, os.PathLike)):
                                try: size = os.path.getsize(f)
                                except: size = None
                            else:
                                try:
                                    p = f.tell(); f.seek(0, os.SEEK_END)
                                    size = f.tell(); f.seek(p)
                                except: size = None
                            if size is not None and size <= 44:
                                continue
                            buf = _collect_filelike(f)
                            captured.append((uid, data, buf))
                    finally:
                        done.set()

                # start_recording（取り残しがあると例外になる）
                try:
                    print(f"[STT] start_recording() rec={getattr(vc,'recording',None)}")
                    vc.start_recording(sink, finished_callback)
                except Exception as e:
                    print("[STT] start_recording failed:", repr(e))
                    # すでに録音中なら止めて次ループ
                    if "Already recording" in str(e):
                        await ensure_stopped(vc, "after start fail")
                        await asyncio.sleep(0.3)
                        continue
                    await asyncio.sleep(1.0)
                    continue

                window = st["record_window"]
                await asyncio.sleep(window)

                # 停止（同期）
                try:
                    print("[STT] stop_recording()")
                    vc.stop_recording()
                except Exception as e:
                    print("[STT] stop_recording failed:", repr(e))

                await done.wait()
                await ensure_stopped(vc, "after stop")  # 念のため

            # ここまでが1サイクル（ロック解放）

            if not captured:
                print("[STT] no audio captured in this window")
                await asyncio.sleep(0.3)
                continue

            # Whisper へ
            jobs = []
            for (uid, data, buf) in captured:
                name = await resolve_display_name(guild_obj, uid, data)
                jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, buf, dest))
            await asyncio.gather(*jobs, return_exceptions=True)

    except asyncio.CancelledError:
        print("[STT] worker cancelled", guild_id)
        # キャンセル時も録音残ってたら止める
        vc = guild_obj.voice_client
        if vc and vc.is_connected():
            await ensure_stopped(vc, "on cancel")
    except Exception as e:
        print("[STT] worker crashed:", repr(e))
        traceback.print_exc()
        # クラッシュ時も安全弁
        vc = guild_obj.voice_client
        if vc and vc.is_connected():
            await ensure_stopped(vc, "on crash")
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
                name = await resolve_display_name(guild, int(user_id), data)
                fileobj = data.file

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

async def transcribe_and_post_from_bytes(guild_id: int, user_id: int, username: str, buf: bytes, channel):
    if not openai:
        print("[STT] OpenAI client is None"); return
    st = get_state(guild_id)

    # --- VAD（無音スキップの条件を緩める）---
    dur = None
    rms = None
    db = None
    try:
        dur, rms = wav_stats(buf)
        # WAVメタ不整合対策：概算長（48kHz/16bit/2ch ≒ 192kB/s）
        if (dur == 0.0 or dur is None) and len(buf) > 44:
            dur = len(buf) / 192000.0
        db = _dbfs_from_rms(rms or 0.0)
        print(f"[STT] segment stats: dur={dur:.2f}s rms={rms:.4f} ({db:.1f} dBFS)")

        # 「短い かつ 小さい かつ 静か」ならスキップ（AND）
        should_skip = (dur < st["min_dur"]) and (rms < st["vad_rms"]) and (db < st["vad_db"])
        if should_skip:
            print("[STT] skip by VAD")
            return
    except Exception:
        traceback.print_exc()

    # --- Whisper ---
    tmp = None; fh = None
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp = tf.name; tf.write(buf); tf.close()
        fh = open(tmp, "rb")

        kwargs = {}
        if st["lang"] != "auto":
            kwargs["language"] = st["lang"]

        resp = openai.audio.transcriptions.create(file=fh, model="whisper-1", **kwargs)
        text = (getattr(resp, "text", "") or "").strip()
        print(f"[STT] Whisper result: {text!r}")

        if text:
            # ★ ログ: 音声認識テキスト・発言者・記録時刻（近似）
            dest_id = getattr(channel, "id", 0)
            await log_stt_event(
                guild_id=guild_id,
                dest_channel_id=dest_id,
                user_id=user_id,
                user_display=username,
                text=text,
                duration=float(dur) if dur is not None else None,
                rms=float(rms) if rms is not None else None,
                dbfs=float(db) if db is not None else None,
            )

            # キャプション投稿（連投マージ対応）
            await post_caption(guild_id, channel, user_id, username, jp_cleanup(text))

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


# =========================
# Help コマンド（カスタム）
# =========================

def _is_admin_ctx(ctx: commands.Context) -> bool:
    perms = getattr(ctx.author, "guild_permissions", None)
    return bool(perms and (perms.manage_guild or perms.administrator))

# コマンド定義（書式と説明）
_HELP_ITEMS = [
    {
        "name": "join", "aliases": ["執事参加","執事入室","執事召喚"],
        "usage": "{p}join",
        "desc": "今いるボイスチャンネルへボットを参加させます（Stage では話者化を試みます）。",
    },
    {
        "name": "leave", "aliases": ["執事退出","執事離脱"],
        "usage": "{p}leave",
        "desc": "ボイスチャンネルから退出します。",
    },
    {
        "name": "readon", "aliases": ["読み上げコマンド","読み上げ","読み上げ開始","読み上げオン","このチャンネルを読み上げ"],
        "usage": "{p}readon",
        "desc": "このテキストチャンネルの新規メッセージをボイスチャンネルで読み上げます。",
    },
    {
        "name": "readoff", "aliases": ["読み上げ停止","読み上げオフ"],
        "usage": "{p}readoff",
        "desc": "読み上げを停止します。",
    },
    {
        "name": "readhere", "aliases": ["ここを読み上げ"],
        "usage": "{p}readhere",
        "desc": "読み上げ対象チャンネルを“今ここ”に変更します。",
    },
    {
        "name": "stton", "aliases": ["字幕開始","文字起こし開始","字幕オン","音声認識開始"],
        "usage": "{p}stton [区切り秒数(3-60)]",
        "desc": "ボイスチャンネルの音声を区切って文字起こしし、ここ（またはスレッド）に投稿します。",
    },
    {
        "name": "sttoff", "aliases": ["字幕停止","文字起こし停止","字幕オフ","音声認識停止"],
        "usage": "{p}sttoff",
        "desc": "音声認識ワーカーを停止します。",
    },
    {
        "name": "stttest", "aliases": ["文字起こしテスト"],
        "usage": "{p}stttest",
        "desc": "gTTS→Whisper の疎通テストを行います（日本語固定）。",
    },
    {
        "name": "rectest", "aliases": ["録音テスト"],
        "usage": "{p}rectest [秒数(2-30)]",
        "desc": "現在のボイスチャンネルを一時録音し、結果を返信します（デバッグ用）。",
    },
    {
        "name": "diag", "aliases": ["診断"],
        "usage": "{p}diag",
        "desc": "py-cord のバージョンや ffmpeg/PyNaCl などの診断情報を表示します。",
    },
    {
        "name": "whereami", "aliases": [],
        "usage": "{p}whereami",
        "desc": "このテキストチャンネル（またはスレッド）の情報を表示します。",
    },
    {
        "name": "intentcheck", "aliases": [],
        "usage": "{p}intentcheck",
        "desc": "Members Intent 等の実際の挙動を簡易チェックします。",
    },
    {
        "name": "sttset", "aliases": [],
        "usage": (
            "{p}sttset vad <rms> | vaddb <dB> | mindur <秒> | merge <秒> | "
            "mergeauto on/off | lang <auto/ja/en> | thread on/off"
        ),
        "desc": (
            "VADしきい値・最小長・マージ時間・言語・スレッド運用などを調整します。"
            " 例: `{p}sttset vad 0.008`, `{p}sttset lang auto`, `{p}sttset thread on`"
        ),
    },
    # ==== 管理者向け（表示制御） ====
    {
        "name": "ttsspeed", "aliases": ["読み上げ速度"],
        "usage": "{p}ttsspeed <倍率>",
        "desc": "サーバー全体の基準話速を設定します。例: `1.35`（推奨 0.6〜2.0）",
        "admin_only": True,
    },
    {
        "name": "ttsvoice", "aliases": ["声色"],
        "usage": "{p}ttsvoice @ユーザー (<半音> [テンポ] | reset)",
        "desc": "特定ユーザーの声色（半音）とテンポ係数を上書きします。例: `@太郎 +3 1.10` / `reset`",
        "admin_only": True,
    },
    {
        "name": "ttsconfig", "aliases": ["読み上げ設定"],
        "usage": "{p}ttsconfig",
        "desc": "現在の話速・個別声色オーバーライドの一覧を表示します。",
        "admin_only": True,
    },
]

def _find_help_item(name: str):
    n = name.lower()
    for item in _HELP_ITEMS:
        if item["name"].lower() == n or n in [a.lower() for a in item.get("aliases", [])]:
            return item
    return None

def _format_cmd_line(item: dict, prefix: str) -> tuple[str, str]:
    """Embed のフィールド (name, value) を返す"""
    aliases = item.get("aliases") or []
    alias_str = (" / " + " / ".join(aliases)) if aliases else ""
    admin_tag = " 🔒" if item.get("admin_only") else ""
    name = f"{prefix}{item['name']}{alias_str}{admin_tag}"
    usage = (item["usage"] or "").format(p=prefix)
    desc = (item["desc"] or "").format(p=prefix)
    value = f"**書式**: `{usage}`\n{desc}"
    return name, value

@bot.command(name="help", aliases=["h"])
async def help_command(ctx: commands.Context, *, command_name: str = None):
    """カスタムヘルプ: !help / !help <コマンド名>"""
    prefix = ctx.prefix or "!"
    is_admin = _is_admin_ctx(ctx)

    # 個別ヘルプ（!help stton のように指定された場合）
    if command_name:
        item = _find_help_item(command_name)
        if not item:
            return await ctx.reply(f"`{command_name}` のヘルプは見つかりませんでした。`{prefix}help` で一覧を表示します。")
        if item.get("admin_only") and not is_admin:
            return await ctx.reply("このコマンドはサーバー管理者向けです。")
        name, value = _format_cmd_line(item, prefix)
        emb = discord.Embed(
            title="📖 コマンドヘルプ",
            description=f"`{prefix}{item['name']}` の説明です。",
            color=discord.Color.blurple(),
        )
        emb.add_field(name=name, value=value, inline=False)
        return await ctx.reply(embed=emb)

    # 一覧ヘルプ
    emb = discord.Embed(
        title="📖 ヘルプ — ボイス字幕ボット",
        description=(
            f"プレフィックス: `{prefix}`\n"
            f"詳細は `{prefix}help <コマンド名>` で確認できます。"
        ),
        color=discord.Color.blurple(),
    )

    # 実行者が使えるコマンドのみ表示
    visible_items = [
        x for x in _HELP_ITEMS
        if (not x.get("admin_only") or is_admin)
    ]

    # 見やすい順に並べ替え（お好みで）
    order = ["join","leave","readon","readoff","readhere","stton","sttoff",
             "stttest","rectest","diag","whereami","intentcheck","sttset",
             "ttsspeed","ttsvoice","ttsconfig"]
    sort_key = {name:i for i,name in enumerate(order)}
    visible_items.sort(key=lambda x: sort_key.get(x["name"], 999))

    for item in visible_items:
        name, value = _format_cmd_line(item, prefix)
        emb.add_field(name=name, value=value, inline=False)

    # フッター
    if not is_admin:
        emb.set_footer(text="🔒 が付いた項目はサーバー管理者向けです。")
    else:
        emb.set_footer(text="管理者向けのコマンドも表示しています。")

    await ctx.reply(embed=emb)


bot.run(TOKEN)
