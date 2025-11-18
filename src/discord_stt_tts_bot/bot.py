import io, os, wave, asyncio, tempfile, traceback, struct, math,re, time,csv, json, threading
import audioop
import collections
import shutil
import discord
import typing as T
from types import MethodType

from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv
from discord import StageChannel, TextChannel, Thread
from discord.abc import Messageable
from discord.ext import commands, tasks
from gtts import gTTS
from openai import OpenAI
from pathlib import Path
import requests

from discord.voice_client import VoiceClient as _PycordVoiceClient

try:
    import webrtcvad
except ImportError:  # pragma: no cover - optional dependency guard
    webrtcvad = None

# === CCFOLIA BRIDGE START: import ===
from aiohttp import web
# === CCFOLIA BRIDGE END: import ===

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
if SRC_DIR.name == "src":
    PROJECT_ROOT = SRC_DIR.parent
else:
    PROJECT_ROOT = PACKAGE_DIR.parent

# 優先してリポジトリ直下の .env を読み込む。存在しない環境では既定の検索にフォールバック。
dotenv_loaded = False
candidate_env = PROJECT_ROOT / ".env"
if candidate_env.exists():
    dotenv_loaded = load_dotenv(candidate_env)
if not dotenv_loaded:
    load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_LANG = os.getenv("TTS_LANG", "ja")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "gtts").strip().lower() or "gtts"
VOICEVOX_BASE_URL = (os.getenv("VOICEVOX_BASE_URL", "http://127.0.0.1:50021").strip().rstrip("/"))
VOICEVOX_TIMEOUT = float(os.getenv("VOICEVOX_TIMEOUT", "15"))
VOICEVOX_DEFAULT_SPEAKER = int(os.getenv("VOICEVOX_DEFAULT_SPEAKER", "2"))
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
ARNNDN_MODEL = os.getenv("ARNNDN_MODEL_PATH", "").strip()
DEFAULT_PRIMARY_STT_MODEL = os.getenv("STT_PRIMARY_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
DEFAULT_FALLBACK_STT_MODEL = os.getenv("STT_FALLBACK_MODEL", "gpt-4o-transcribe").strip() or "gpt-4o-transcribe"
_DEFAULT_STT_COSTS = {
    "gpt-4o-mini-transcribe": 0.0006,  # USD per audio minute (近似)
    "gpt-4o-transcribe": 0.0015,
    "whisper-1": 0.0006,
}
try:
    _STT_COST_OVERRIDES = json.loads(os.getenv("STT_MODEL_COSTS_JSON", "{}"))
    if not isinstance(_STT_COST_OVERRIDES, dict):
        _STT_COST_OVERRIDES = {}
except Exception:
    _STT_COST_OVERRIDES = {}
# Force py-cord to fall back to legacy XSalsa20-based modes. Discord started
# advertising AEAD/XChaCha modes first, but py-cord fails to decrypt them in
# some environments, resulting in `CryptoError: Decryption failed`. By limiting
# supported_modes to XSalsa variants we ensure the voice gateway negotiates a
# mode we can handle reliably.
_PycordVoiceClient.supported_modes = (
    "xsalsa20_poly1305_lite",
    "xsalsa20_poly1305_suffix",
    "xsalsa20_poly1305",
)

# === CCFOLIA BRIDGE START: env ===
CCFO_HOST = os.getenv("CCFOLIA_BRIDGE_HOST", "127.0.0.1")
CCFO_PORT = int(os.getenv("CCFOLIA_BRIDGE_PORT", "8800"))
CCFO_SECRET = os.getenv("CCFOLIA_POST_SECRET", "")
CCFO_ACCEPT_FROM = {x.strip() for x in os.getenv("CCFOLIA_ACCEPT_FROM", "127.0.0.1,::1").split(",")}
CCFO_MIRROR_CH_ID = int(os.getenv("CCFOLIA_MIRROR_CHANNEL_ID", "0") or "0")
CCFO_TTS_MODE = os.getenv("CCFOLIA_TTS_MODE", "file").strip().lower()  # file/voice/off
CCFO_SPK_MAP = json.loads(os.getenv("CCFOLIA_SPEAKER_MAP_JSON", '{"（未指定）":2}'))
CCFO_DEFAULT_SPK = int(os.getenv("CCFOLIA_DEFAULT_SPEAKER", "2"))
# イベントキュー
ccfo_queue: "asyncio.Queue[dict]" = asyncio.Queue()
# ココフォリアの重複対策用に、直近のイベント履歴を保持する件数
CCFO_EVENT_HISTORY_LIMIT: int = 300
# ココフォリア連携で「重複」とみなす時間幅（秒）
CCFO_EVENT_DUPLICATE_WINDOW_SEC: float = 10.0
# ココフォリア連携の重複判定に用いるイベント履歴
_ccfo_event_history: collections.deque["CCFoliaEventRecord"] = collections.deque()
# イベントごとの最終記録時刻を管理する辞書
_ccfo_event_last_seen_map: dict[tuple[str, str, str, str], datetime] = {}
# ログから履歴を読み込む処理が完了したかどうかのフラグ
_ccfo_event_history_loaded: bool = False
# イベント履歴を読み書きする際の同期用ロック
_ccfo_event_history_lock = asyncio.Lock()
# CCFOLIA Webサーバーのランナー参照（クリーンアップ用）
_ccfo_web_runner: web.AppRunner | None = None
# CCFOLIA Webサーバーのサイト参照（クリーンアップ用）
_ccfo_web_site: web.TCPSite | None = None
# バックグラウンドタスク集合（キャンセル管理用）
_ccfo_background_tasks: set[asyncio.Task[object]] = set()
# デバッグログ出力制御用ロック
_ccfo_debug_log_lock = threading.Lock()
# === CCFOLIA BRIDGE END: env ===

# === GUI 簡易設定 ===
GUI_ADMIN_TOKEN = os.getenv("GUI_ADMIN_TOKEN", "").strip()
GUI_USER_CACHE_TTL = float(os.getenv("GUI_USER_CACHE_TTL", "15"))
GUI_VOICEVOX_CACHE_TTL = float(os.getenv("GUI_VOICEVOX_CACHE_TTL", "60"))
GUI_STATIC_ROOT = PACKAGE_DIR / "webui"

# GUI 用のキャッシュは asyncio.Lock で保護する
_gui_user_cache: dict[str, T.Any] = {"timestamp": 0.0, "entries": []}
_gui_user_cache_lock = asyncio.Lock()
_gui_voicevox_cache: dict[str, T.Any] = {"timestamp": 0.0, "items": []}
_gui_voicevox_cache_lock = asyncio.Lock()
# === GUI 簡易設定 終了 ===

# STT字幕用の基本16色（視認性の高い色を選択）
STT_COLOR_PALETTE: list[int] = [
    0xF44336, 0xE91E63, 0x9C27B0, 0x673AB7,
    0x3F51B5, 0x2196F3, 0x03A9F4, 0x00BCD4,
    0x009688, 0x4CAF50, 0x8BC34A, 0xCDDC39,
    0xFFC107, 0xFF9800, 0xFF5722, 0x795548,
]


def _heuristic_punctuate(text: str, duration: float | None) -> str:
    text = (text or "").strip()
    if not text:
        return text

    repl_map = {
        "?": "？",
        "!": "！",
        "。": "。",
        "！": "！",
        "？": "？",
    }
    for ascii_punct, jp_punct in (("?", "？"), ("!", "！")):
        text = text.replace(ascii_punct, jp_punct)

    has_period = any(ch in text for ch in ("。", "！", "？"))
    if not has_period:
        allow_len = max(6, min(40, len(text) + 5))
        if len(text) < allow_len:
            text = f"{text}。"

    text = text.replace("。。", "。")
    return text

def _resolve_log_dir(base_dir: Path, env_value: str | None) -> Path:
    # 空 or 未設定 → デフォルト "logs"
    if not env_value or not env_value.strip():
        return base_dir / "logs"
    # ~ と 環境変数 を展開
    expanded = os.path.expanduser(os.path.expandvars(env_value.strip()))
    p = Path(expanded)
    # 相対パスなら PROJECT_ROOT 基準に
    if not p.is_absolute():
        p = base_dir / p
    return p

def _resolve_data_dir(base_dir: Path, env_value: str | None, default_name: str) -> Path:
    if not env_value or not env_value.strip():
        return base_dir / default_name
    expanded = os.path.expanduser(os.path.expandvars(env_value.strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path

LOG_DIR = _resolve_log_dir(PROJECT_ROOT, os.getenv("LOG_DIR"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TTS_LOG_PATH = LOG_DIR / "tts_logs.csv"
STT_LOG_PATH = LOG_DIR / "stt_logs.csv"
STT_METRICS_PATH = LOG_DIR / "stt_metrics.csv"
CCFO_LOG_PATH = LOG_DIR / "ccfolia_event_logs.csv"
CCFO_DEBUG_LOG_PATH = LOG_DIR / "ccfolia_debug.log"
_log_lock = asyncio.Lock()  # 複数タスクからの同時書き込みを保護
print(f"[LOG] output directory: {LOG_DIR}")  # 起動時に出力先を表示
DATA_DIR = _resolve_data_dir(PROJECT_ROOT, os.getenv("BOT_STATE_DIR"), "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TTS_PERSIST_PATH = DATA_DIR / "tts_profiles.json"
_tts_persist_lock = threading.Lock()
_tts_persist_cache: dict[int, dict] = {}


def _load_tts_persist_cache():
    global _tts_persist_cache
    if not TTS_PERSIST_PATH.exists():
        _tts_persist_cache = {}
        return
    try:
        with open(TTS_PERSIST_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        print(f"[TTS] failed to load persisted speaker settings: {exc!r}")
        _tts_persist_cache = {}
        return
    if isinstance(raw, dict) and "guilds" in raw and isinstance(raw["guilds"], dict):
        raw = raw["guilds"]
    new_cache: dict[int, dict] = {}
    if isinstance(raw, dict):
        for gid, payload in raw.items():
            try:
                gid_int = int(gid)
            except Exception:
                continue
            if isinstance(payload, dict):
                new_cache[gid_int] = payload
    _tts_persist_cache = new_cache


def _hydrate_tts_state_from_disk(state: dict, guild_id: int):
    with _tts_persist_lock:
        payload = _tts_persist_cache.get(guild_id)
    if not payload:
        return
    try:
        state["tts_base_tempo"] = float(payload.get("tts_base_tempo", state.get("tts_base_tempo", 0.7)))
    except Exception:
        pass
    try:
        state["tts_default_speaker"] = int(payload.get("tts_default_speaker", state.get("tts_default_speaker", VOICEVOX_DEFAULT_SPEAKER)))
    except Exception:
        pass
    overrides_raw = payload.get("tts_overrides", {})
    overrides: dict[int, dict[str, float]] = {}
    if isinstance(overrides_raw, dict):
        for uid, cfg in overrides_raw.items():
            try:
                uid_int = int(uid)
            except Exception:
                continue
            if not isinstance(cfg, dict):
                continue
            try:
                semi = float(cfg.get("semitones", 0.0))
            except Exception:
                semi = 0.0
            try:
                tempo = float(cfg.get("tempo", 1.0))
            except Exception:
                tempo = 1.0
            overrides[uid_int] = {"semitones": semi, "tempo": tempo}
    state["tts_overrides"] = overrides

    speakers_raw = payload.get("tts_speakers", {})
    speakers: dict[int, int] = {}
    if isinstance(speakers_raw, dict):
        for uid, sid in speakers_raw.items():
            try:
                speakers[int(uid)] = int(sid)
            except Exception:
                continue
    state["tts_speakers"] = speakers


def _persist_tts_preferences(guild_id: int, state: dict):
    payload = {
        "tts_base_tempo": float(state.get("tts_base_tempo", 0.7)),
        "tts_default_speaker": int(state.get("tts_default_speaker", VOICEVOX_DEFAULT_SPEAKER)),
        "tts_speakers": {},
        "tts_overrides": {},
    }
    for uid, sid in (state.get("tts_speakers") or {}).items():
        try:
            payload["tts_speakers"][str(int(uid))] = int(sid)
        except Exception:
            continue
    for uid, cfg in (state.get("tts_overrides") or {}).items():
        if not isinstance(cfg, dict):
            continue
        try:
            uid_key = str(int(uid))
        except Exception:
            continue
        try:
            semi = float(cfg.get("semitones", 0.0))
        except Exception:
            semi = 0.0
        try:
            tempo = float(cfg.get("tempo", 1.0))
        except Exception:
            tempo = 1.0
        payload["tts_overrides"][uid_key] = {"semitones": semi, "tempo": tempo}

    serializable = {}
    try:
        with _tts_persist_lock:
            serializable = {
                "guilds": {
                    str(gid): data
                    for gid, data in {**_tts_persist_cache, guild_id: payload}.items()
                }
            }
            _tts_persist_cache[guild_id] = payload
        tmp_path = TTS_PERSIST_PATH.with_suffix(".tmp")
        TTS_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, TTS_PERSIST_PATH)
    except Exception as exc:
        print(f"[TTS] failed to persist speaker settings for guild {guild_id}: {exc!r}")


_load_tts_persist_cache()


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
_ensure_csv_with_header(
    STT_METRICS_PATH,
    [
        "timestamp_iso",
        "guild_id",
        "user_id",
        "model",
        "fallback_used",
        "duration_sec",
        "avg_logprob",
        "no_speech_prob",
        "compression_ratio",
        "token_count",
        "estimated_cost_usd",
        "text_length",
        "gain_factor",
        "gate_frames",
    ],
)
_ensure_csv_with_header(
    CCFO_LOG_PATH,
    ["timestamp_iso", "user_display", "text", "room", "ts_client"],
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


async def log_stt_metrics(
    guild_id: int,
    user_id: int,
    model: str,
    fallback_used: bool,
    duration: float | None,
    avg_logprob: float | None,
    no_speech_prob: float | None,
    compression_ratio: float | None,
    token_count: int | None,
    estimated_cost: float | None,
    text_length: int,
    gain_factor: float | None,
    gate_frames: int | None,
):
    ts = datetime.now(timezone.utc).isoformat()
    row = [
        ts,
        guild_id or 0,
        user_id,
        model or "",
        int(bool(fallback_used)),
        f"{duration:.3f}" if isinstance(duration, (int, float)) else "",
        f"{avg_logprob:.3f}" if isinstance(avg_logprob, (int, float)) else "",
        f"{no_speech_prob:.3f}" if isinstance(no_speech_prob, (int, float)) else "",
        f"{compression_ratio:.3f}" if isinstance(compression_ratio, (int, float)) else "",
        str(int(token_count)) if isinstance(token_count, (int, float)) else "",
        f"{estimated_cost:.6f}" if isinstance(estimated_cost, (int, float)) else "",
        str(int(text_length)) if isinstance(text_length, (int, float)) else "",
        f"{gain_factor:.3f}" if isinstance(gain_factor, (int, float)) else "",
        str(int(gate_frames)) if isinstance(gate_frames, (int, float)) else "",
    ]
    await _append_csv(STT_METRICS_PATH, row)


def _evaluate_transcription_response(resp, text: str, duration: float | None) -> tuple[dict, bool]:
    metrics: dict[str, float | int | None] = {
        "avg_logprob": None,
        "no_speech_prob": None,
        "compression_ratio": None,
        "token_count": None,
    }
    segments = getattr(resp, "segments", None)
    avg_logprob_acc: list[float] = []
    token_total = 0
    if isinstance(segments, (list, tuple)):
        for seg in segments:
            if isinstance(seg, dict):
                val = seg.get("avg_logprob")
                if isinstance(val, (int, float)):
                    avg_logprob_acc.append(float(val))
                ns = seg.get("no_speech_prob")
                if metrics["no_speech_prob"] is None and isinstance(ns, (int, float)):
                    metrics["no_speech_prob"] = float(ns)
                tokens = seg.get("tokens")
                if isinstance(tokens, (list, tuple)):
                    token_total += len(tokens)
    if avg_logprob_acc:
        metrics["avg_logprob"] = sum(avg_logprob_acc) / len(avg_logprob_acc)
    comp = getattr(resp, "compression_ratio", None)
    if isinstance(comp, (int, float)):
        metrics["compression_ratio"] = float(comp)
    if token_total > 0:
        metrics["token_count"] = token_total
    else:
        tokens_attr = getattr(resp, "tokens", None)
        if isinstance(tokens_attr, (list, tuple)):
            metrics["token_count"] = len(tokens_attr)

    should_retry = False
    text_len = len(text or "")
    dur = duration or 0.0
    avg_logprob = metrics["avg_logprob"]
    no_speech_prob = metrics["no_speech_prob"]
    compression_ratio = metrics["compression_ratio"]

    if isinstance(avg_logprob, (int, float)) and avg_logprob < -0.7:
        should_retry = True
    if isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.8:
        should_retry = True
    if isinstance(compression_ratio, (int, float)) and compression_ratio > 2.4:
        should_retry = True
    if text_len <= 4 and dur >= 2.0:
        should_retry = True

    return metrics, should_retry

@dataclass(frozen=True)
class CCFoliaEventRecord:
    """ココフォリア連携の重複判定に利用するイベント情報.

    Attributes:
        speaker: 発言者名（正規化済み）。
        text: 発言本文（CSV向けに正規化済み）。
        room: ココフォリアの部屋名。
        ts_client: クライアントが送信したタイムスタンプや識別子。
        timestamp_utc: 取得時刻（UTC）。
    """

    speaker: str
    text: str
    room: str
    ts_client: str
    timestamp_utc: datetime

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """イベントを一意に識別するためのキーを返す.

        Returns:
            発言内容を特定するためのタプル。
        """
        return (self.speaker, self.text, self.room, self.ts_client)


def _append_ccfo_event_history(entry: CCFoliaEventRecord) -> None:
    """重複判定用の履歴へイベントを追加し、古い情報を整理する.

    Args:
        entry: 追加対象のイベント。
    """
    _ccfo_event_history.append(entry)
    _ccfo_event_last_seen_map[entry.identity] = entry.timestamp_utc

    # 直近のイベントのみを保持する
    while len(_ccfo_event_history) > CCFO_EVENT_HISTORY_LIMIT:
        removed_entry = _ccfo_event_history.popleft()
        last_seen = _ccfo_event_last_seen_map.get(removed_entry.identity)
        if last_seen is not None and last_seen <= removed_entry.timestamp_utc:
            _ccfo_event_last_seen_map.pop(removed_entry.identity, None)

    # 時間窓から外れたイベントを除去する
    cutoff = entry.timestamp_utc - timedelta(seconds=CCFO_EVENT_DUPLICATE_WINDOW_SEC)
    while _ccfo_event_history:
        oldest_entry = _ccfo_event_history[0]
        if oldest_entry.timestamp_utc >= cutoff:
            break
        removed_entry = _ccfo_event_history.popleft()
        last_seen = _ccfo_event_last_seen_map.get(removed_entry.identity)
        if last_seen is not None and last_seen <= removed_entry.timestamp_utc:
            _ccfo_event_last_seen_map.pop(removed_entry.identity, None)


def _ccfo_row_to_event_record(row: list[str]) -> CCFoliaEventRecord | None:
    """CSVの1行からイベント情報を復元する.

    Args:
        row: CSVの行データ。

    Returns:
        復元したイベント。復元できない場合はNone。
    """
    if len(row) < 3:
        return None
    timestamp_raw = (row[0] if len(row) > 0 else "").strip()
    try:
        timestamp_utc = datetime.fromisoformat(timestamp_raw)
        if timestamp_utc.tzinfo is None:
            timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
        else:
            timestamp_utc = timestamp_utc.astimezone(timezone.utc)
    except Exception:
        timestamp_utc = datetime.now(timezone.utc)
    speaker = (row[1] if len(row) > 1 else "").strip()
    text = _norm_text_for_csv(row[2] if len(row) > 2 else "")
    room = (row[3] if len(row) > 3 else "").strip()
    ts_client = (row[4] if len(row) > 4 else "").strip()
    return CCFoliaEventRecord(
        speaker=speaker,
        text=text,
        room=room,
        ts_client=ts_client,
        timestamp_utc=timestamp_utc,
    )


def _ccfo_debug_log(stage: str, payload: dict[str, T.Any] | None = None) -> None:
    """CCFOLIA関連のデバッグ情報を標準出力とファイルへ残す。"""
    ts = datetime.now(timezone.utc).isoformat()
    record: dict[str, T.Any] = {"ts": ts, "stage": stage}
    if payload:
        record["payload"] = payload
    try:
        text_payload = json.dumps(payload, ensure_ascii=False) if payload else ""
    except Exception:
        text_payload = str(payload)
    print(f"[CCFOLIA][DEBUG][{stage}] {text_payload}")
    log_line = json.dumps(record, ensure_ascii=False)
    try:
        with _ccfo_debug_log_lock:
            with CCFO_DEBUG_LOG_PATH.open("a", encoding="utf-8") as fp:
                fp.write(log_line + "\n")
    except Exception as exc:
        print(f"[CCFOLIA][DEBUG] failed to write debug log: {exc!r}")


async def _load_ccfo_event_history_from_log() -> None:
    """ログファイルから重複判定用の履歴を読み込む.

    Returns:
        None.
    """
    global _ccfo_event_history_loaded
    if _ccfo_event_history_loaded:
        return
    async with _ccfo_event_history_lock:
        if _ccfo_event_history_loaded:
            return
        if not CCFO_LOG_PATH.exists():
            _ccfo_event_history_loaded = True
            return
        try:
            async with _log_lock:
                with open(CCFO_LOG_PATH, newline="", encoding="utf-8") as log_file:
                    reader = csv.reader(log_file)
                    next(reader, None)
                    for row in reader:
                        entry = _ccfo_row_to_event_record(row)
                        if entry is not None:
                            _append_ccfo_event_history(entry)
        except Exception as exc:
            print(f"[CCFOLIA] failed to load ccfo history: {exc!r}")
        finally:
            _ccfo_event_history_loaded = True


async def log_ccfolia_event(
    event: CCFoliaEventRecord,
) -> bool:
    """ココフォリアのイベントをログへ記録し、重複を抑止する.

    Args:
        event: 記録したいイベント。

    Returns:
        Trueなら新規イベントを記録したことを示し、Falseなら既存のイベントだったことを示す。
    """
    await _load_ccfo_event_history_from_log()
    async with _ccfo_event_history_lock:
        last_seen = _ccfo_event_last_seen_map.get(event.identity)
        if last_seen is not None:
            delta = event.timestamp_utc - last_seen
            if delta.total_seconds() <= CCFO_EVENT_DUPLICATE_WINDOW_SEC:
                print(f"[CCFOLIA] duplicated event skipped: {event.speaker} / {event.text}")
                return False
        _append_ccfo_event_history(event)
        row = [
            event.timestamp_utc.isoformat(),
            event.speaker,
            event.text,
            event.room,
            event.ts_client,
        ]
    await _append_csv(CCFO_LOG_PATH, row)
    print(["log:", CCFO_LOG_PATH, ",", event.speaker, ",", event.text])
    return True

if OPENAI_API_KEY:
    print(f"[STT] OPENAI_API_KEY detected.")
else:
    print("[STT] OPENAI_API_KEY NOT found (Whisperは動きません)")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True
intents.messages = True
intents.members = True

bot = commands.Bot(command_prefix=("!", "！"), intents=intents, help_command=None)
openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

guild_state = {}  # guild_id -> dict( read_channel_id, stt_on, record_window )

_original_bot_close = bot.close

async def _close_with_cleanup(self: commands.Bot) -> None:
    """Bot 終了時に追加のクリーンアップ処理を挟む。"""
    await _cancel_background_tasks()
    await _stop_ccfo_web_server()
    await _disconnect_all_voice_clients(self)
    await _original_bot_close()

bot.close = MethodType(_close_with_cleanup, bot)

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
    color = _resolve_caption_color(guild_id, user_id)

    # 直近メッセージがあって merge_window 内なら編集で追記
    if entry and entry.get("message") and (now - entry.get("ts", 0)) < st["merge_window"]:
        try:
            base_text = entry.get("text", "")
            merged = (base_text + " " + new_text).strip()
            embed = _build_caption_embed(username, merged, color)
            await entry["message"].edit(embed=embed)
            entry["text"] = merged
            entry["ts"] = now
            return
        except Exception as e:
            print("[STT] edit failed; fallback send:", repr(e))

    # 新規投稿
    embed = _build_caption_embed(username, new_text, color)
    m = await channel.send(embed=embed)
    ch_map[key_u] = {"message": m, "ts": now, "text": new_text}

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

class _TrackedAudioData(discord.sinks.AudioData):
    """discord.AudioData + byte_count tracking for reliable size checks."""

    def __init__(self, file_obj: io.BytesIO):
        super().__init__(file_obj)
        self.byte_count = 0

    def write(self, data):
        self.byte_count += len(data)
        super().write(data)


class PatchedWaveSink(discord.sinks.Sink):
    """discord.sinks.WaveSink 互換の録音シンク（WAVヘッダ書き込みの欠落を補完する）。"""

    def __init__(self, *, filters=None):
        super().__init__(filters=filters)
        self.encoding = "wav"

    @discord.sinks.Filters.container
    def write(self, data, user):
        if user not in self.audio_data:
            self.audio_data[user] = _TrackedAudioData(io.BytesIO())
        self.audio_data[user].write(data)

    def cleanup(self):
        self.finished = True
        decoder = getattr(self.vc, "decoder", None)
        channels = getattr(decoder, "CHANNELS", 2)
        sample_rate = getattr(decoder, "SAMPLING_RATE", 48000)
        sample_size = getattr(decoder, "SAMPLE_SIZE", channels * 2)
        sample_width = max(1, sample_size // max(1, channels))

        for audio in self.audio_data.values():
            audio.cleanup()  # rewind + mark finished
            pcm = audio.file.read()
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                if pcm:
                    wf.writeframes(pcm)
            wav_buf.seek(0)
            # replace underlying file with WAV payload
            try:
                audio.file.close()
            except Exception:
                pass
            audio.file = wav_buf
            audio.byte_count = len(pcm)
            audio.on_format(self.encoding)

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


def _wav_bytes_to_pcm16k(src: bytes) -> bytes | None:
    """WaveSinkが生成したWAVを16kHz/mono/16bit PCMへ変換する。"""
    try:
        with wave.open(io.BytesIO(src), "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
    except Exception:
        traceback.print_exc()
        return None

    if not frames:
        return None

    try:
        if channels > 1:
            frames = audioop.tomono(frames, width, 1.0, 1.0)
        if width != 2:
            frames = audioop.lin2lin(frames, width, 2)
            width = 2
        if rate != 16000:
            frames, _ = audioop.ratecv(frames, 2, 1, rate, 16000, None)
    except Exception:
        traceback.print_exc()
        return None

    return frames


def _pcm16k_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buff.getvalue()


class _VadUserStream:
    """ユーザー単位で音声フレームを受け取り、VAD区切りを行う。"""

    FRAME_MS = 20
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2

    def __init__(self):
        self._vad: webrtcvad.Vad | None = None if webrtcvad is None else webrtcvad.Vad(2)
        self._config: dict[str, int] = {}
        self._pre_buffer: collections.deque[bytes] = collections.deque()
        self._active: list[bytes] = []
        self._prev_tail: list[bytes] = []
        self._in_speech = False
        self._silence_counter = 0
        self._post_counter = -1
        self._silence_frames = 8
        self._post_frames = 0
        self._overlap_frames = 0
        self._max_segment_frames = 1200
        self._min_frames = 1

    def configure(self, st: dict):
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is not available")

        cfg = dict(
            aggressiveness=int(max(0, min(3, st.get("vad_aggressiveness", 2))))
        )
        frame_ms = self.FRAME_MS
        cfg["pre_frames"] = max(0, int(st.get("vad_pre_ms", 200) // frame_ms))
        cfg["silence_frames"] = max(1, int(st.get("vad_silence_ms", 450) // frame_ms))
        cfg["post_frames"] = max(0, int(st.get("vad_post_ms", 400) // frame_ms))
        cfg["overlap_frames"] = max(0, int(st.get("vad_overlap_ms", 320) // frame_ms))
        cfg["min_frames"] = max(1, int(st.get("min_dur", 0.8) * 1000 // frame_ms))
        cfg["max_segment_frames"] = max(
            cfg["overlap_frames"] + cfg["silence_frames"] + 1,
            int(st.get("vad_max_segment_ms", 20000) // frame_ms)
        )

        if cfg != self._config:
            self._config = cfg
            self._vad = webrtcvad.Vad(cfg["aggressiveness"])
            self._pre_buffer = collections.deque(maxlen=cfg["pre_frames"] or 1)
            self._silence_frames = cfg["silence_frames"]
            self._post_frames = cfg["post_frames"]
            self._overlap_frames = cfg["overlap_frames"]
            self._max_segment_frames = max(cfg["max_segment_frames"], cfg["pre_frames"] + cfg["overlap_frames"] + 4)
            self._min_frames = cfg["min_frames"]

    def feed(self, wav_bytes: bytes, st: dict) -> list[bytes]:
        if webrtcvad is None:
            return []
        self.configure(st)
        pcm = _wav_bytes_to_pcm16k(wav_bytes)
        if not pcm:
            return []

        frame_bytes = int(self.SAMPLE_RATE * self.FRAME_MS / 1000) * self.SAMPLE_WIDTH
        outputs: list[bytes] = []

        for offset in range(0, len(pcm), frame_bytes):
            frame = pcm[offset:offset + frame_bytes]
            if len(frame) < frame_bytes:
                # 端数は次サイクルへ回す（捨てる）
                break

            is_speech = self._vad.is_speech(frame, self.SAMPLE_RATE) if self._vad else False

            if is_speech:
                if not self._in_speech:
                    starter: list[bytes] = []
                    if self._prev_tail and self._overlap_frames:
                        starter.extend(self._prev_tail)
                    if self._config.get("pre_frames"):
                        starter.extend(self._pre_buffer)
                    if starter:
                        self._active.extend(starter)
                    self._prev_tail = []
                    self._in_speech = True
                self._active.append(frame)
                self._silence_counter = 0
                self._post_counter = -1
            else:
                if self._in_speech:
                    self._active.append(frame)
                    self._silence_counter += 1
                    if self._silence_counter >= self._silence_frames:
                        if self._post_counter < 0:
                            self._post_counter = self._post_frames
                        self._post_counter -= 1
                        if self._post_counter <= 0:
                            seg = self._finalize_segment()
                            if seg:
                                outputs.append(_pcm16k_to_wav_bytes(seg))
                # サイレント状態が続く場合もプリバッファには保持する

            # プリロール用リングバッファは常に最新状態に更新
            if self._pre_buffer.maxlen:
                self._pre_buffer.append(frame)

            if self._in_speech and len(self._active) >= self._max_segment_frames:
                seg = self._finalize_segment(force=True)
                if seg:
                    outputs.append(_pcm16k_to_wav_bytes(seg))

        return outputs

    def drain(self) -> list[bytes]:
        if not self._active:
            return []
        seg = self._finalize_segment(force=True)
        if not seg:
            return []
        return [_pcm16k_to_wav_bytes(seg)]

    def _finalize_segment(self, force: bool = False) -> bytes | None:
        if not self._active:
            self._reset_state()
            return None
        segment_frames = list(self._active)
        if not force and len(segment_frames) < self._min_frames:
            # 短すぎるセグメントはノイズとして破棄
            self._reset_state()
            return None

        tail_count = min(self._overlap_frames, len(segment_frames))
        if tail_count:
            self._prev_tail = segment_frames[-tail_count:]
        else:
            self._prev_tail = []

        pcm = b"".join(segment_frames)
        self._reset_state()
        return pcm

    def _reset_state(self):
        self._active = []
        self._in_speech = False
        self._silence_counter = 0
        self._post_counter = -1


def _build_denoise_filter_chain(st: dict) -> str:
    filters: list[str] = []
    hp = int(st.get("denoise_highpass", 0) or 0)
    lp = int(st.get("denoise_lowpass", 0) or 0)
    if hp > 0:
        filters.append(f"highpass=f={hp}")
    if lp > 0:
        filters.append(f"lowpass=f={lp}")

    mode = (st.get("denoise_mode") or "").lower()
    if mode == "arnndn":
        model = (st.get("denoise_model") or "").strip()
        if model:
            filters.append(f"arnndn=m={model}")
        else:
            filters.append("arnndn")
    elif mode == "afftdn":
        strength = float(st.get("denoise_strength", 12.0) or 0.0)
        filters.append(f"afftdn=nr={strength:.1f}")

    gain = float(st.get("denoise_gain", 0.0) or 0.0)
    if gain > 0.0:
        filters.append(f"dynaudnorm=f=150:g={gain:.1f}")

    comp = (st.get("denoise_compand") or "").strip()
    if comp:
        filters.append(f"compand={comp}")

    return ",".join(filters)


async def _maybe_denoise_wav(raw: bytes, st: dict) -> bytes:
    if not raw or len(raw) <= 44:
        return raw
    if not st.get("denoise_enable", False):
        return raw
    if st.get("denoise_ffmpeg_failed"):
        return raw

    ffmpeg_path = shutil.which(FFMPEG_BIN)
    if not ffmpeg_path:
        st["denoise_ffmpeg_failed"] = True
        print("[STT] denoise skipped: ffmpeg not found")
        return raw

    filters = _build_denoise_filter_chain(st)
    if not filters:
        return raw

    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-af", filters,
            "-f", "wav",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        st["denoise_ffmpeg_failed"] = True
        print("[STT] denoise skipped: ffmpeg execution failed")
        return raw

    try:
        stdout_data, stderr_data = await proc.communicate(raw)
    except Exception as exc:
        st["denoise_ffmpeg_failed"] = True
        print("[STT] denoise failed during execution:", repr(exc))
        return raw

    if proc.returncode != 0 or not stdout_data:
        st["denoise_ffmpeg_failed"] = True
        err_excerpt = (stderr_data or b"").decode("utf-8", errors="ignore").strip()
        if err_excerpt:
            print("[STT] denoise ffmpeg error:", err_excerpt)
        else:
            print("[STT] denoise ffmpeg returned empty output")
        return raw

    return stdout_data


def _estimate_stt_cost(model: str, duration_sec: float) -> float:
    rate = _STT_COST_OVERRIDES.get(model)
    if not isinstance(rate, (int, float)):
        rate = _DEFAULT_STT_COSTS.get(model, 0.0)
    try:
        minutes = max(0.0, float(duration_sec)) / 60.0
    except Exception:
        minutes = 0.0
    return minutes * float(rate)


def _apply_gain_and_gate(wav_bytes: bytes, st: dict) -> tuple[bytes, dict | None]:
    gain_enabled = st.get("gain_enable", False)
    gate_enabled = st.get("gate_enable", False)
    if not (gain_enabled or gate_enabled):
        return wav_bytes, None

    pcm = _wav_bytes_to_pcm16k(wav_bytes)
    if not pcm:
        return wav_bytes, None

    pre_rms = audioop.rms(pcm, 2) / 32767.0 if pcm else 0.0

    pcm_array = bytearray(pcm)
    frame_samples = 320  # 20ms @16kHz
    frame_bytes = frame_samples * 2
    gate_frames_zeroed = 0

    if gate_enabled:
        threshold_db = float(st.get("gate_threshold_db", -55.0))
        linear = 10.0 ** (threshold_db / 20.0)
        amp_threshold = max(0, min(32767, int(linear * 32767)))
        hold_ms = max(40.0, float(st.get("gate_hold_ms", 160.0)))
        hold_frames = max(1, int(round(hold_ms / 20.0)))
        silence_run = 0
        gate_active = False
        run_positions: list[int] = []

        for start in range(0, len(pcm_array), frame_bytes):
            frame = pcm_array[start:start + frame_bytes]
            if len(frame) < frame_bytes:
                break
            rms = audioop.rms(frame, 2)
            if rms < amp_threshold:
                silence_run += 1
                run_positions.append(start)
            else:
                silence_run = 0
                run_positions.clear()
                gate_active = False

            if not gate_active and silence_run >= hold_frames:
                gate_active = True
                for pos in run_positions:
                    segment = pcm_array[pos:pos + frame_bytes]
                    if any(segment):
                        pcm_array[pos:pos + frame_bytes] = b"\x00" * len(segment)
                        gate_frames_zeroed += 1
            elif gate_active and rms < amp_threshold:
                if any(frame):
                    pcm_array[start:start + frame_bytes] = b"\x00" * len(frame)
                    gate_frames_zeroed += 1
            elif gate_active and rms >= amp_threshold:
                gate_active = False
                run_positions.clear()

    pcm_processed = bytes(pcm_array)
    gain_applied = False
    gain_factor = 1.0
    if gain_enabled:
        target_rms = max(1e-4, float(st.get("gain_target_rms", 0.06)))
        max_gain = max(1.0, float(st.get("gain_max_gain", 6.0)))
        current_rms = audioop.rms(pcm_processed, 2) / 32767.0 if pcm_processed else 0.0
        if current_rms > 1e-4:
            gain_factor = target_rms / current_rms
            gain_factor = max(1.0, min(max_gain, gain_factor))
            if gain_factor > 1.02:
                pcm_processed = audioop.mul(pcm_processed, 2, gain_factor)
                gain_applied = True

    post_rms = audioop.rms(pcm_processed, 2) / 32767.0 if pcm_processed else 0.0

    if gain_applied or gate_frames_zeroed > 0:
        wav_bytes = _pcm16k_to_wav_bytes(pcm_processed)

    info = dict(
        pre_rms=pre_rms,
        post_rms=post_rms,
        gain_applied=gain_applied,
        gain_factor=gain_factor,
        gate_frames=gate_frames_zeroed,
    )
    return wav_bytes, info


def _append_pcm_buffer(st: dict, user_id: int, pcm_bytes: bytes) -> list[bytes]:
    buffers: dict[int, bytearray] = st.setdefault("pcm_buffers", {})
    entry = buffers.setdefault(int(user_id), bytearray())
    entry.extend(pcm_bytes)

    max_sec = float(st.get("buffer_max_sec", 20.0))
    max_samples = int(max(1.0, max_sec) * 16000)
    if len(entry) > max_samples * 2:
        entry[:] = entry[-max_samples * 2:]

    last_seen: dict[int, float] = st.setdefault("pcm_last_seen", {})
    last_seen[int(user_id)] = time.time()

    min_sec = max(0.2, float(st.get("min_dur", 0.8)))
    min_samples = int(min_sec * 16000)
    segments: list[bytes] = []
    if len(entry) >= min_samples * 2:
        segments.append(bytes(entry))
        entry.clear()
    return segments


def _flush_stale_pcm_buffers(st: dict, captured_users: set[int]) -> list[tuple[int, bytes]]:
    now = time.time()
    last_seen: dict[int, float] = st.get("pcm_last_seen") or {}
    buffers: dict[int, bytearray] = st.get("pcm_buffers") or {}
    flush_sec = max(0.6, float(st.get("record_window", DEFAULT_WINDOW)))
    outputs: list[tuple[int, bytes]] = []
    for uid, ts in list(last_seen.items()):
        if uid in captured_users:
            continue
        if now - ts >= flush_sec:
            data = buffers.get(uid)
            if data:
                outputs.append((uid, bytes(data)))
                data.clear()
            last_seen.pop(uid, None)
    return outputs


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
    sink = PatchedWaveSink()
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
            vad_db=-45.0,
            min_dur=0.8,
            merge_window=6.0,
            merge_auto=True,
            stt_mode="vad",  # vad | fixed
            vad_aggressiveness=2,
            vad_silence_ms=450,
            vad_pre_ms=200,
            vad_post_ms=400,
            vad_overlap_ms=320,
            vad_max_segment_ms=20000,
            denoise_enable=True,
            denoise_mode=("arnndn" if ARNNDN_MODEL else "afftdn"),
            denoise_model=ARNNDN_MODEL,
            denoise_highpass=120,
            denoise_lowpass=7000,
            denoise_strength=12.0,
            denoise_gain=15.0,
            denoise_compand="attacks=10:decays=100:points=-70/-90|-40/-20|0/-2",
            denoise_ffmpeg_failed=False,
            lang="ja",
            stt_primary_model=DEFAULT_PRIMARY_STT_MODEL,
            stt_fallback_model=DEFAULT_FALLBACK_STT_MODEL,
            gain_enable=True,
            gain_target_rms=0.06,
            gain_max_gain=6.0,
            gate_enable=True,
            gate_threshold_db=-55.0,
            gate_hold_ms=160.0,
            use_thread=False,
            caption_dest_id=None,
            last_msgs={},
            rec_lock=asyncio.Lock(),
            tts_base_tempo=float(os.getenv("TTS_TEMPO", "0.7")),  # サーバー全体の基準話速
            tts_overrides={},   # { user_id: {"semitones": float, "tempo": float} }
            tts_default_speaker=VOICEVOX_DEFAULT_SPEAKER,
            tts_speakers={},    # { user_id: speaker_id }
            stt_color_overrides={},  # { user_id: palette_index }
            stt_metrics=dict(
                total_calls=0,
                fallback_calls=0,
                total_duration=0.0,
                total_cost=0.0,
                model_usage=collections.Counter(),
                gain_events=0,
                gain_factor_sum=0.0,
                gate_frames=0,
            ),
            pcm_buffers={},
            pcm_last_seen={},
        )
        _hydrate_tts_state_from_disk(guild_state[guild_id], guild_id)
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


def _resolve_caption_color(guild_id: int, user_id: int | None) -> int:
    """字幕用のEmbedカラーを決定する。"""
    st = get_state(guild_id)
    if user_id is not None:
        override = st["stt_color_overrides"].get(int(user_id))
        if isinstance(override, int) and 0 <= override < len(STT_COLOR_PALETTE):
            return STT_COLOR_PALETTE[override]
        idx = abs(int(user_id)) % len(STT_COLOR_PALETTE)
        return STT_COLOR_PALETTE[idx]
    return STT_COLOR_PALETTE[0]


def _build_caption_embed(username: str, text: str, color: int) -> discord.Embed:
    """字幕表示用のEmbedを生成する。"""
    embed = discord.Embed(description=text, color=color)
    embed.set_author(name=username)
    return embed


def _voicevox_request(text: str, speaker_id: int) -> bytes:
    """VOICEVOX エンジンへ音声合成を依頼し、音声データ（WAV）のバイナリを返す。"""
    params = {"text": text, "speaker": speaker_id}
    query_url = f"{VOICEVOX_BASE_URL}/audio_query"
    synth_url = f"{VOICEVOX_BASE_URL}/synthesis"

    try:
        query_resp = requests.post(query_url, params=params, timeout=VOICEVOX_TIMEOUT)
        query_resp.raise_for_status()
        query_payload = query_resp.json()
        synth_resp = requests.post(
            synth_url,
            params={"speaker": speaker_id},
            json=query_payload,
            timeout=VOICEVOX_TIMEOUT,
        )
        synth_resp.raise_for_status()
        return synth_resp.content
    except Exception as exc:
        print(f"[TTS] VOICEVOX synthesis failed: {exc!r}")
        raise


async def _voicevox_synthesize(text: str, speaker_id: int) -> bytes:
    """VOICEVOX への同期リクエストをスレッドで実行し、音声バイト列を取得する。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _voicevox_request(text, speaker_id))

async def tts_play(guild: discord.Guild, text: str, speaker_id: int | None = None):
    vc: discord.VoiceClient = guild.voice_client
    if not vc or not vc.is_connected():
        return

    st = get_state(guild.id)
    if TTS_PROVIDER == "voicevox":
        fallback = await _tts_play_voicevox(vc, guild, text, speaker_id)
        if not fallback:
            return
        # VOICEVOX が失敗した場合は fallback により gTTS 呼び出しへ
    await _tts_play_gtts(vc, guild.id, text, speaker_id, st)


async def _play_vc_audio(vc: discord.VoiceClient, path: str):
    """指定パスの音声ファイルを ffmpeg 経由で再生する。"""
    audio = discord.FFmpegPCMAudio(
        path,
        before_options="-loglevel quiet -nostdin",
        options="-vn"
    )
    vc.play(audio)
    while vc.is_playing():
        await asyncio.sleep(0.2)


async def _tts_play_gtts(vc: discord.VoiceClient, guild_id: int, text: str, speaker_id: int | None, st: dict):
    """gTTS を用いた従来の読み上げを実行する。"""
    prof = _pick_voice_profile_for_user(guild_id, speaker_id)
    final_tempo = st["tts_base_tempo"] * prof.get("tempo", 1.0)
    final_tempo = max(0.5, min(2.5, final_tempo))
    semitones = float(prof.get("semitones", 0.0))

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    try:
        gTTS(text=sanitize_for_tts(text), lang=TTS_LANG).save(tmp_path)
        af = _build_ffmpeg_afilter(semitones=semitones, final_tempo=final_tempo)
        audio = discord.FFmpegPCMAudio(
            tmp_path,
            before_options="-loglevel quiet -nostdin",
            options=f"-vn -af {af}"
        )
        vc.play(audio)
        while vc.is_playing():
            await asyncio.sleep(0.2)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


async def _tts_play_voicevox(vc: discord.VoiceClient, guild: discord.Guild, text: str, speaker_id: int | None) -> bool:
    """VOICEVOX を用いた読み上げを実行する。成功したら True、失敗した場合は False を返す。"""
    resolved = _resolve_voicevox_speaker(guild.id, speaker_id)
    sanitized = sanitize_for_tts(text)
    try:
        audio_bytes = await _voicevox_synthesize(sanitized, resolved)
    except Exception as exc:
        print("[TTS] VOICEVOX fallback to gTTS due to:", repr(exc))
        try:
            await _notify_voicevox_failure(guild, resolved)
        except Exception as notify_exc:
            print("[TTS] Failed to notify VOICEVOX fallback:", repr(notify_exc))
        return True  # fallback: gTTS を続ける

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        f.write(audio_bytes)

    try:
        await _play_vc_audio(vc, tmp_path)
        return False
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _resolve_voicevox_speaker(guild_id: int, user_id: int | None) -> int:
    """ユーザー向けの VOICEVOX 話者 ID を決定する。"""
    st = get_state(guild_id)
    if user_id is not None:
        sid = st["tts_speakers"].get(int(user_id))
        if isinstance(sid, int):
            return sid
    return int(st.get("tts_default_speaker", VOICEVOX_DEFAULT_SPEAKER))


async def _notify_voicevox_failure(guild: discord.Guild, speaker_id: int):
    """VOICEVOX 失敗時に読み上げ対象チャンネルへ通知する。"""
    st = get_state(guild.id)
    channel_id = st.get("read_channel_id")
    channel: T.Optional[discord.abc.Messageable] = None
    if channel_id:
        channel = guild.get_channel(channel_id)
    if channel is None:
        channel = _pick_fallback_text_channel(guild)
    if channel is None:
        return
    note = (
        "🔄 VOICEVOX への接続に失敗したため、読み上げを gTTS に切り替えました。\n"
        f"speaker_id={speaker_id}"
    )
    await channel.send(note)


async def _voicevox_fetch_json(method: str, path: str, *, params=None, json_payload=None):
    """VOICEVOX との同期HTTP通信をスレッドで行うヘルパー。"""
    if TTS_PROVIDER != "voicevox":
        raise RuntimeError("VOICEVOX provider is disabled")

    def _request():
        url = f"{VOICEVOX_BASE_URL}{path}"
        resp = requests.request(method, url, params=params, json=json_payload, timeout=VOICEVOX_TIMEOUT)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            detail = None
            try:
                detail_json = resp.json()
                detail = json.dumps(detail_json, ensure_ascii=False)
            except Exception:
                detail = resp.text
            msg = f"HTTP {resp.status_code}: {detail or str(exc)}"
            raise requests.HTTPError(msg, response=resp) from exc
        try:
            return resp.json()
        except ValueError:
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _request)


async def _voicevox_list_dictionary() -> list[dict]:
    data = await _voicevox_fetch_json("GET", "/user_dict")
    items = []
    if isinstance(data, dict):
        items = [{"id": key, **value} for key, value in data.items()]
    elif isinstance(data, list):
        items = data
    items.sort(key=lambda x: (x.get("pronunciation") or "", x.get("surface") or ""))
    return items


async def _voicevox_add_dictionary_word(surface: str, pronunciation: str, accent_type: int | None = None, word_type: str = "PROPER_NOUN"):
    surface = (surface or "").strip()
    pronunciation = _normalize_pronunciation(pronunciation)
    accent = _sanitize_accent_type(pronunciation, accent_type)
    payload = {
        "surface": surface,
        "pronunciation": pronunciation,
        "word_type": word_type or "PROPER_NOUN",
        "accent_type": accent,
    }
    try:
        return await _voicevox_fetch_json("POST", "/user_dict_word", params=payload)
    except requests.HTTPError as exc:
        resp = getattr(exc, "response", None)
        detail = resp.text if resp is not None else str(exc)
        raise RuntimeError(f"VOICEVOX 辞書登録に失敗しました: {detail}") from exc


def _sanitize_accent_type(pronunciation: str, accent_type: int | str | None) -> int:
    try:
        accent = int(accent_type) if accent_type is not None else 1
    except (TypeError, ValueError):
        accent = 1

    accent = max(0, accent)
    pron = (pronunciation or "").strip()
    if not pron:
        return accent

    max_len = max(1, len(pron))
    if accent > max_len:
        accent = max_len
    return accent


_HIRA_TO_KATA = str.maketrans({
    **{chr(h): chr(h + 0x60) for h in range(0x3041, 0x3097)},
    "ゔ": "ヴ",
    "ゐ": "ヰ",
    "ゑ": "ヱ",
})


def _normalize_pronunciation(src: str | None) -> str:
    pron = (src or "").strip()
    if not pron:
        return ""
    pron = pron.translate(_HIRA_TO_KATA)
    pron = pron.replace("　", " ").replace(" ", "")
    return pron


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

    if vc and vc.channel and vc.channel.id == channel.id:
        return await ctx.reply(f"既に **{channel.name}** に接続済みです。")

    if vc:
        try:
            await ensure_stopped(vc, "before rejoin")
        except Exception:
            pass
        try:
            await vc.disconnect(force=True)
        except Exception as exc:
            print("[join] disconnect failed:", repr(exc))

    try:
        vc = await channel.connect()
    except discord.ClientException as exc:
        if "Already connected" in str(exc):
            existing = ctx.guild.voice_client
            if existing:
                try:
                    await existing.move_to(channel)
                    vc = existing
                except Exception as move_exc:
                    print("[join] move_to after Already connected failed:", repr(move_exc))
                    raise
            else:
                raise
        else:
            raise

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
async def stton(ctx: commands.Context, *args: str):
    vc = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        return await ctx.reply("先に `!join` してください。")
    st = get_state(ctx.guild.id)

    desired_mode = st.get("stt_mode", "vad")
    window_override: int | None = None

    for raw in args:
        if raw is None:
            continue
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in ("vad", "auto", "adaptive"):
            desired_mode = "vad"
        elif lowered in ("fixed", "timer", "window", "legacy"):
            desired_mode = "fixed"
        else:
            try:
                win = int(token)
            except ValueError:
                return await ctx.reply("`vad` / `fixed` / 区切り秒数(3-60) のいずれかで指定してください。")
            if not (3 <= win <= 60):
                return await ctx.reply("区切り秒数は 3〜60 の範囲で指定してください。")
            window_override = win
            desired_mode = "fixed"

    if window_override is not None:
        st["record_window"] = window_override
        if st.get("merge_auto", True):
            st["merge_window"] = max(st["merge_window"], round(window_override * 1.25, 2))

    st["stt_mode"] = desired_mode

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

    mode_text = "VADモード" if desired_mode == "vad" else f"固定{st['record_window']}秒区切り"
    await ctx.reply(
        f"🎧 音声認識を開始（{mode_text}）。投稿先: <#{dest.id}> / OpenAI鍵: {'あり' if openai else 'なし'}"
    )


@bot.command(name="sttoff", aliases=["字幕停止","字幕終了","文字起こし停止","字幕オフ","音声認識停止"])
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


@bot.command(name="sttstats", aliases=["stt統計", "sttmetrics"])
async def sttstats(ctx: commands.Context):
    st = get_state(ctx.guild.id)
    metrics = st.get("stt_metrics") or {}
    total = metrics.get("total_calls", 0)
    fallback = metrics.get("fallback_calls", 0)
    duration = metrics.get("total_duration", 0.0)
    cost = metrics.get("total_cost", 0.0)
    usage = metrics.get("model_usage")
    lines = []
    lines.append(f"総トランスクリプト数: {total}")
    if total:
        lines.append(f"フォールバック発生: {fallback} ({(fallback/total)*100:.1f}%)")
    else:
        lines.append("フォールバック発生: 0")
    lines.append(f"累計音声長: {duration:.1f} 秒")
    lines.append(f"推定コスト: ${cost:.4f}")
    if isinstance(usage, collections.Counter) and usage:
        top_models = usage.most_common(5)
        model_text = ", ".join(f"{name}:{count}" for name, count in top_models)
        lines.append(f"モデル使用回数: {model_text}")
    gain_events = metrics.get("gain_events", 0)
    if gain_events:
        avg_gain = metrics.get("gain_factor_sum", 0.0) / max(1, gain_events)
        lines.append(f"平均ゲイン係数: {avg_gain:.2f} （適用 {gain_events} 回）")
    gate_frames = metrics.get("gate_frames", 0)
    if gate_frames:
        gate_seconds = gate_frames * 0.02
        lines.append(f"ゲート無音化: {gate_frames}フレーム（約 {gate_seconds:.1f} 秒）")
    await ctx.reply("STT指標概要:\n" + "\n".join(lines))


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
    if TTS_PROVIDER != "gtts":
        return await ctx.reply("現在の読み上げエンジンでは `ttsspeed` は利用できません (gTTS 専用機能)。")
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
    _persist_tts_preferences(ctx.guild.id, st)
    await ctx.reply(f"✅ サーバー基準の読み上げ話速を **{r:.2f}倍** に設定しました。")

@bot.command(name="ttsvoice", aliases=["声色"])
async def ttsvoice(ctx: commands.Context, member: discord.Member = None, semitones: str = None, tempo: str = None):
    if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
        return await ctx.reply("このコマンドはサーバー管理者のみ実行できます。")
    if TTS_PROVIDER != "gtts":
        return await ctx.reply("現在の読み上げエンジンでは `ttsvoice` は利用できません (gTTS 専用機能)。")

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
        _persist_tts_preferences(ctx.guild.id, st)
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
    _persist_tts_preferences(ctx.guild.id, st)
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
    lines.append(f"- VOICEVOX デフォルト話者ID: {st['tts_default_speaker']}")
    if st["tts_speakers"]:
        lines.append("- VOICEVOX 個別話者（最大10件表示）:")
        for uid, sid in list(st["tts_speakers"].items())[:10]:
            m = ctx.guild.get_member(uid)
            name = m.display_name if m else f"User {uid}"
            lines.append(f"  • {name}: speaker_id={sid}")
        if len(st["tts_speakers"]) > 10:
            lines.append(f"  …ほか {len(st['tts_speakers']) - 10} 件")
    await ctx.reply("\n".join(lines))


@bot.command(name="ttsspeaker", aliases=["スピーカー", "speaker"])
async def ttsspeaker(ctx: commands.Context, *args):
    """VOICEVOX の話者 ID を管理するコマンド。"""
    if TTS_PROVIDER != "voicevox":
        return await ctx.reply("現在の読み上げエンジンでは VOICEVOX 話者設定は利用できません。")

    st = get_state(ctx.guild.id)

    if not args:
        current = st["tts_default_speaker"]
        count = len(st["tts_speakers"])
        return await ctx.reply(f"VOICEVOX デフォルト話者IDは {current}、個別設定は {count} 件です。")

    keyword = args[0].lower()

    if keyword == "export":
        if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
            return await ctx.reply("この操作はサーバー管理者のみ実行できます。")
        payload = {
            "default_speaker": st["tts_default_speaker"],
            "user_speakers": {str(k): v for k, v in st["tts_speakers"].items()},
        }
        blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        fp = io.BytesIO(blob)
        fp.seek(0)
        return await ctx.reply(
            "VOICEVOX の話者設定ファイルです。",
            file=discord.File(fp, filename="voicevox_speakers.json"),
        )

    if keyword == "import":
        if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
            return await ctx.reply("この操作はサーバー管理者のみ実行できます。")
        if not ctx.message.attachments:
            return await ctx.reply("JSON ファイルを添付してください。")
        try:
            data = await ctx.message.attachments[0].read()
            payload = json.loads(data.decode("utf-8"))
        except Exception as exc:
            return await ctx.reply(f"JSON の読み込みに失敗しました: {exc!r}")

        if "default_speaker" in payload:
            try:
                st["tts_default_speaker"] = int(payload["default_speaker"])
            except Exception:
                return await ctx.reply("default_speaker は整数で指定してください。")

        mapping = payload.get("user_speakers", {})
        new_map: dict[int, int] = {}
        try:
            for k, v in mapping.items():
                new_map[int(k)] = int(v)
        except Exception:
            return await ctx.reply("user_speakers 内のキーと値は整数で指定してください。")

        st["tts_speakers"] = new_map
        _persist_tts_preferences(ctx.guild.id, st)
        return await ctx.reply(f"VOICEVOX 話者設定を {len(new_map)} 件読み込みました。")

    if keyword == "default":
        if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
            return await ctx.reply("この操作はサーバー管理者のみ実行できます。")
        if len(args) < 2:
            return await ctx.reply("`!ttsspeaker default <speaker_id>` の形式で指定してください。")
        try:
            sid = int(args[1])
        except Exception:
            return await ctx.reply("speaker_id は整数で指定してください。")
        st["tts_default_speaker"] = sid
        _persist_tts_preferences(ctx.guild.id, st)
        return await ctx.reply(f"VOICEVOX デフォルト話者IDを {sid} に設定しました。")

    # 個別ユーザー設定
    member = ctx.message.mentions[0] if ctx.message.mentions else None
    if member is None:
        try:
            uid = int(args[0])
            member = ctx.guild.get_member(uid)
            if member is None:
                member = await ctx.guild.fetch_member(uid)
        except Exception:
            member = None
    if member is None or (member != ctx.author and not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator)):
        # 管理権限が無い場合は自身のみ設定可能
        member = ctx.author

    if member is None:
        return await ctx.reply("ユーザーを特定できませんでした。メンションまたはユーザーIDで指定してください。")

    if len(args) < 2:
        current = st["tts_speakers"].get(member.id)
        return await ctx.reply(f"{member.display_name} の話者IDは {current if current is not None else '未設定'} です。")

    value = args[1].lower()
    if value in ("reset", "clear"):
        st["tts_speakers"].pop(member.id, None)
        _persist_tts_preferences(ctx.guild.id, st)
        return await ctx.reply(f"{member.display_name} の VOICEVOX 話者設定を削除しました。")

    try:
        sid = int(value)
    except Exception:
        return await ctx.reply("speaker_id は整数で指定してください。")

    st["tts_speakers"][member.id] = sid
    _persist_tts_preferences(ctx.guild.id, st)
    if member == ctx.author:
        return await ctx.reply(f"あなたの VOICEVOX 話者IDを {sid} に設定しました。")
    return await ctx.reply(f"{member.display_name} の VOICEVOX 話者IDを {sid} に設定しました。")


@bot.command(name="sttcolor", aliases=["字幕色", "color"])
async def sttcolor(ctx: commands.Context, *args):
    """字幕カラー設定を管理する。"""

    st = get_state(ctx.guild.id)

    if not args:
        count = len(st["stt_color_overrides"])
        example = ", ".join(
            f"{idx}:{hex(color)[2:]}" for idx, color in enumerate(STT_COLOR_PALETTE)
        )
        return await ctx.reply(
            "字幕カラー設定の概要です。\n"
            f"- 個別設定数: {count} 件\n"
            f"- カラーパレット (0-15): {example}"
        )

    keyword = args[0].lower()

    if keyword == "export":
        if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
            return await ctx.reply("この操作はサーバー管理者のみ実行できます。")
        payload = {
            "user_colors": {str(k): v for k, v in st["stt_color_overrides"].items()}
        }
        blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        fp = io.BytesIO(blob)
        fp.seek(0)
        return await ctx.reply(
            "字幕カラー設定ファイルです。",
            file=discord.File(fp, filename="stt_colors.json"),
        )

    if keyword == "import":
        if not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator):
            return await ctx.reply("この操作はサーバー管理者のみ実行できます。")
        if not ctx.message.attachments:
            return await ctx.reply("JSON ファイルを添付してください。")
        try:
            data = await ctx.message.attachments[0].read()
            payload = json.loads(data.decode("utf-8"))
        except Exception as exc:
            return await ctx.reply(f"JSON の読み込みに失敗しました: {exc!r}")

        mapping = payload.get("user_colors", {})
        new_map: dict[int, int] = {}
        try:
            for k, v in mapping.items():
                idx = int(v)
                if 0 <= idx < len(STT_COLOR_PALETTE):
                    new_map[int(k)] = idx
                else:
                    return await ctx.reply("color index は 0-15 の範囲で指定してください。")
        except Exception:
            return await ctx.reply("user_colors 内のキーと値は整数で指定してください。")

        st["stt_color_overrides"] = new_map
        return await ctx.reply(f"字幕カラー設定を {len(new_map)} 件読み込みました。")

    member = ctx.message.mentions[0] if ctx.message.mentions else None
    if member is None:
        try:
            uid = int(args[0])
            member = ctx.guild.get_member(uid)
            if member is None:
                member = await ctx.guild.fetch_member(uid)
        except Exception:
            member = None
    if member is None or (member != ctx.author and not (ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator)):
        member = ctx.author

    if member is None:
        return await ctx.reply("ユーザーを特定できませんでした。メンションまたはユーザーIDで指定してください。")

    if len(args) < 2:
        current = st["stt_color_overrides"].get(member.id)
        return await ctx.reply(
            f"{member.display_name} の字幕カラーは {current if current is not None else '自動割り当て'} です。"
        )

    value = args[1].lower()
    if value in ("reset", "clear"):
        st["stt_color_overrides"].pop(member.id, None)
        return await ctx.reply(f"{member.display_name} の字幕カラー設定を削除しました。")

    try:
        idx = int(value)
    except Exception:
        return await ctx.reply("カラー番号は 0-15 の整数で指定してください。")

    if not (0 <= idx < len(STT_COLOR_PALETTE)):
        return await ctx.reply("カラー番号は 0-15 の範囲で指定してください。")

    st["stt_color_overrides"][member.id] = idx
    if member == ctx.author:
        return await ctx.reply(f"あなたの字幕カラーを {idx} に設定しました。")
    return await ctx.reply(f"{member.display_name} の字幕カラーを {idx} に設定しました。")


@bot.command(name="voicevoxstyles", aliases=["voxstyles", "voxlist"])
async def voicevoxstyles(ctx: commands.Context):
    """VOICEVOX の話者IDとスタイル一覧を表示する。"""
    if TTS_PROVIDER != "voicevox":
        return await ctx.reply("現在の読み上げエンジンは VOICEVOX ではありません。")
    try:
        resp = requests.get(f"{VOICEVOX_BASE_URL}/speakers", timeout=VOICEVOX_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return await ctx.reply(f"VOICEVOX のスタイル取得に失敗しました: {exc!r}")

    lines: list[str] = []
    for speaker in payload or []:
        base_name = speaker.get("name", "unknown")
        for style in speaker.get("styles", []):
            sid = style.get("id")
            style_name = style.get("name", "default")
            if sid is None:
                continue
            lines.append(f"{sid:>4}: {base_name} / {style_name}")

    if not lines:
        return await ctx.reply("スタイル情報が取得できませんでした。")

    chunk = []
    header = "VOICEVOX 話者IDとスタイル一覧"
    for line in lines:
        chunk.append(line)
        if len("\n".join(chunk)) > 1700:
            text = "\n".join(chunk[:-1])
            await ctx.reply(f"{header}\n```\n{text}\n```")
            header = ""
            chunk = [chunk[-1]]
    if chunk:
        text = "\n".join(chunk)
        await ctx.reply((f"{header}\n" if header else "") + f"```\n{text}\n```")


@bot.command(name="sttpalette", aliases=["colorpalette", "palette"])
async def sttpalette(ctx: commands.Context):
    """字幕カラーのパレット番号とカラーコードを表示する。"""
    embeds: list[discord.Embed] = []
    for idx, color in enumerate(STT_COLOR_PALETTE):
        embed = discord.Embed(
            title=f"パレット {idx}",
            description=f"カラーコード: `#{color:06X}`\nこの色で字幕を設定するには `!sttcolor @ユーザー {idx}`" ,
            color=color,
        )
        embed.set_footer(text="字幕カラーのプレビューです。サイドバーが該当色になります。")
        embeds.append(embed)

    if not embeds:
        return await ctx.reply("パレットが見つかりませんでした。")

    for i in range(0, len(embeds), 10):
        await ctx.reply(embeds=embeds[i:i+10])


@bot.command(name="voxdict", aliases=["辞書管理"])
async def voxdict(ctx: commands.Context, action: str | None = None, *args):
    """VOICEVOX のユーザー辞書を管理する。"""
    if TTS_PROVIDER != "voicevox":
        return await ctx.reply("現在の読み上げエンジンでは VOICEVOX 辞書を管理できません。")

    is_admin = ctx.author.guild_permissions.manage_guild or ctx.author.guild_permissions.administrator

    if not action:
        return await ctx.reply(
            "使い方: `!voxdict export` / `!voxdict import` (JSON添付) / "
            "`!voxdict add <表層形> <発音> [アクセント番号]`"
        )

    key = action.lower()

    if key == "export":
        if not is_admin:
            return await ctx.reply("辞書のエクスポートはサーバー管理者のみ実行できます。")
        try:
            items = await _voicevox_list_dictionary()
        except Exception as exc:
            return await ctx.reply(f"辞書取得に失敗しました: {exc!r}")

        payload = [
            {
                "surface": item.get("surface"),
                "pronunciation": item.get("pronunciation"),
                "accent_type": item.get("accent_type"),
                "word_type": item.get("word_type"),
            }
            for item in items
        ]
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        fp = io.BytesIO(data)
        fp.seek(0)
        return await ctx.reply("VOICEVOX 辞書ファイルです。", file=discord.File(fp, filename="voicevox_dictionary.json"))

    if key == "import":
        if not is_admin:
            return await ctx.reply("辞書のインポートはサーバー管理者のみ実行できます。")
        if not ctx.message.attachments:
            return await ctx.reply("辞書JSONファイルを添付してください。")
        try:
            blob = await ctx.message.attachments[0].read()
            entries = json.loads(blob.decode("utf-8"))
        except Exception as exc:
            return await ctx.reply(f"JSON の読み込みに失敗しました: {exc!r}")

        added = 0
        errors = 0
        last_error = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            surface = entry.get("surface")
            pron = entry.get("pronunciation")
            accent = entry.get("accent_type")
            wtype = entry.get("word_type", "PROPER_NOUN")
            if not surface or not pron:
                continue
            try:
                await _voicevox_add_dictionary_word(surface, pron, accent, wtype)
                added += 1
            except Exception as exc:
                errors += 1
                last_error = str(exc)
                print("[VOICEVOX] import error:", repr(exc))
        message = f"辞書をインポートしました。追加 {added} 件 / 失敗 {errors} 件。"
        if last_error:
            message += f"\n直近のエラー: {last_error}"
        return await ctx.reply(message)

    if key == "add":
        if len(args) < 2:
            return await ctx.reply("`!voxdict add <表層形> <発音> [アクセント番号]` の形式で指定してください。")
        surface = args[0]
        pronunciation = args[1]
        accent = None
        if len(args) >= 3:
            try:
                accent = int(args[2])
            except Exception:
                return await ctx.reply("アクセント番号は整数で指定してください。")
        try:
            await _voicevox_add_dictionary_word(surface, pronunciation, accent)
            return await ctx.reply(f"辞書に `{surface}` ({pronunciation}) を追加しました。")
        except Exception as exc:
            return await ctx.reply(f"辞書への追加に失敗しました: {exc!r}")

    return await ctx.reply("未知の操作です。`export` / `import` / `add` を指定してください。")


@bot.command(name="logs", aliases=["ログ取得", "getlogs"])
async def download_logs(ctx: commands.Context):
    """音声関連ログ（TTS/STT）を取得して送信する。"""
    files: list[discord.File] = []
    async with _log_lock:
        for path in (TTS_LOG_PATH, STT_LOG_PATH, STT_METRICS_PATH, CCFO_LOG_PATH):
            if path.exists():
                data = path.read_bytes()
                buff = io.BytesIO(data)
                buff.seek(0)
                files.append(discord.File(buff, filename=path.name))

    if not files:
        return await ctx.reply("まだログファイルが存在しません。")

    await ctx.reply("最新のログファイルです。", files=files)

@bot.command(name="sttset")
async def sttset(ctx, key: str=None, value: str=None):
    """
      !sttset vad 0.008
      !sttset vaddb -46
      !sttset mindur 0.4
      !sttset merge 14
      !sttset mergeauto on/off
      !sttset lang ja
      !sttset thread on
      !sttset denoise on/off
      !sttset denoisemode arnndn
      !sttset gaintarget 0.07
      !sttset gate off
      !sttset sttmodel gpt-4o-mini-transcribe
      !sttset sttmodel2 gpt-4o-transcribe
    """
    st = get_state(ctx.guild.id)
    if not key:
        return await ctx.reply(
            (
                "設定: mode={stt_mode} window={record_window}s vad={vad_rms} vaddb={vad_db} "
                "mindur={min_dur}s merge={merge_window}s mergeauto={merge_auto} lang={lang} thread={use_thread} "
                "vadlevel={vad_aggressiveness} silence={vad_silence_ms}ms pre={vad_pre_ms}ms post={vad_post_ms}ms overlap={vad_overlap_ms}ms "
                "denoise={denoise_enable} dmode={denoise_mode} hp={denoise_highpass} lp={denoise_lowpass} strength={denoise_strength} gain={denoise_gain} "
                "gain={gain_enable} gtarget={gain_target_rms} gmax={gain_max_gain} gate={gate_enable} gthreshold={gate_threshold_db} ghold={gate_hold_ms} "
                "model={stt_primary_model} fallback={stt_fallback_model}"
            ).format(**st)
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
            st["lang"] = "ja"
            if value and value.lower() not in ("ja", "japanese", "日本語", "jp"):
                return await ctx.reply("言語は日本語固定です（lang=ja）。")
        elif k in ("thread","th"):
            st["use_thread"] = (value.lower() in ("on","true","1","yes","y"))
            st["caption_dest_id"] = None
        elif k in ("mode", "sttmode"):
            lv = value.lower()
            if lv not in ("vad", "fixed"):
                return await ctx.reply("mode は `vad` または `fixed` を指定してください。")
            st["stt_mode"] = lv
        elif k in ("window", "win"):
            win = int(value)
            if not (3 <= win <= 60):
                return await ctx.reply("window は 3〜60 の整数で指定してください。")
            st["record_window"] = win
        elif k in ("vadlevel", "vadmode", "vadaggr"):
            lvl = int(value)
            if not (0 <= lvl <= 3):
                return await ctx.reply("vadlevel は 0〜3 の整数で指定してください。")
            st["vad_aggressiveness"] = lvl
        elif k in ("vadsilence", "silence"):
            st["vad_silence_ms"] = max(120, min(2000, int(value)))
        elif k in ("vadpre", "pre"):
            st["vad_pre_ms"] = max(0, min(1000, int(value)))
        elif k in ("vadpost", "post"):
            st["vad_post_ms"] = max(0, min(1500, int(value)))
        elif k in ("vadoverlap", "overlap"):
            st["vad_overlap_ms"] = max(0, min(1000, int(value)))
        elif k in ("denoise", "dn"):
            st["denoise_enable"] = value.lower() in ("on", "true", "1", "yes", "y")
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoisemode", "dnmode"):
            lv = value.lower()
            if lv not in ("arnndn", "afftdn"):
                return await ctx.reply("denoisemode は `arnndn` または `afftdn` を指定してください。")
            st["denoise_mode"] = lv
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoisemodel", "dnmodel"):
            st["denoise_model"] = value
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoisehp", "dnhp"):
            st["denoise_highpass"] = max(0, min(2000, int(value)))
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoiselp", "dnlp"):
            st["denoise_lowpass"] = max(1000, min(20000, int(value)))
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoisestr", "dnstr", "dnstrength"):
            st["denoise_strength"] = max(0.0, min(30.0, float(value)))
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoisegain", "dngain"):
            st["denoise_gain"] = max(0.0, min(30.0, float(value)))
            st["denoise_ffmpeg_failed"] = False
        elif k in ("denoisecomp", "dncomp"):
            st["denoise_compand"] = value
            st["denoise_ffmpeg_failed"] = False
        elif k in ("gain", "agc"):
            st["gain_enable"] = value.lower() in ("on", "true", "1", "yes", "y")
        elif k in ("gaintarget", "gtarget"):
            st["gain_target_rms"] = max(1e-4, float(value))
        elif k in ("gainmax", "gmax"):
            st["gain_max_gain"] = max(1.0, float(value))
        elif k in ("gate", "noisegate"):
            st["gate_enable"] = value.lower() in ("on", "true", "1", "yes", "y")
        elif k in ("gatethresh", "gatethreshold", "gthreshold"):
            st["gate_threshold_db"] = float(value)
        elif k in ("gatehold", "ghold"):
            st["gate_hold_ms"] = max(40.0, float(value))
        elif k in ("sttmodel", "model", "primarymodel"):
            if not value:
                return await ctx.reply("モデル名を指定してください。例: gpt-4o-mini-transcribe")
            st["stt_primary_model"] = value.strip()
        elif k in ("sttmodel2", "fallback", "secondarymodel"):
            if not value:
                return await ctx.reply("フォールバックモデル名を指定してください。例: gpt-4o-transcribe")
            st["stt_fallback_model"] = value.strip()
        else:
            return await ctx.reply(
                "未知のキー: vad / vaddb / mindur / merge / mergeauto / lang / thread / "
                "mode / window / vadlevel / vadsilence / vadpre / vadpost / vadoverlap / "
                "denoise / denoisemode / denoisemodel / denoisehp / denoiselp / denoisestr / denoisegain / denoisecomp / "
                "gain / gaintarget / gainmax / gate / gatethresh / gatehold / sttmodel / sttmodel2"
            )
    except Exception as e:
        return await ctx.reply(f"設定失敗: {e!r}")

    await ctx.reply(
        (
            "OK: mode={stt_mode} window={record_window}s vad={vad_rms} vaddb={vad_db} "
            "mindur={min_dur}s merge={merge_window}s mergeauto={merge_auto} lang={lang} thread={use_thread} "
            "vadlevel={vad_aggressiveness} silence={vad_silence_ms}ms pre={vad_pre_ms}ms post={vad_post_ms}ms overlap={vad_overlap_ms}ms "
            "denoise={denoise_enable} dmode={denoise_mode} hp={denoise_highpass} lp={denoise_lowpass} strength={denoise_strength} gain={denoise_gain} "
            "gain={gain_enable} gtarget={gain_target_rms} gmax={gain_max_gain} gate={gate_enable} gthreshold={gate_threshold_db} ghold={gate_hold_ms} "
            "model={stt_primary_model} fallback={stt_fallback_model}"
        ).format(**st)
    )


async def stt_worker(guild_id: int, channel_id: int):
    guild_obj = bot.get_guild(guild_id)
    if not guild_obj:
        return
    print("[STT] worker start", guild_id, channel_id)
    st = get_state(guild_id)
    vad_streams: dict[int, _VadUserStream] = {}
    vad_last_seen: dict[int, float] = {}
    warned_vad_missing = False

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

                sink = PatchedWaveSink()
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

                window = float(st.get("record_window", DEFAULT_WINDOW))
                if window < 1.0:
                    window = 1.0
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

            mode = st.get("stt_mode", "vad")
            use_vad = mode == "vad" and webrtcvad is not None
            now_ts = time.time()

            if mode == "vad" and webrtcvad is None and not warned_vad_missing:
                print("[STT] webrtcvad is not available; falling back to fixed segmentation")
                warned_vad_missing = True

            jobs: list[T.Awaitable] = []
            if use_vad:
                captured_users: set[int] = set()
                for (uid, data, buf) in captured:
                    captured_users.add(uid)
                    stream = vad_streams.get(uid)
                    if stream is None:
                        stream = _VadUserStream()
                        vad_streams[uid] = stream
                    vad_last_seen[uid] = now_ts
                    name = await resolve_display_name(guild_obj, uid, data)
                    segments = stream.feed(buf, st)
                    for segment_wav in segments:
                        jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, segment_wav, dest))

                silence_sec = (st.get("vad_silence_ms", 450) + st.get("vad_post_ms", 400)) / 1000.0
                idle_threshold = max(st.get("record_window", DEFAULT_WINDOW), silence_sec + 0.5)

                for uid in list(vad_streams.keys()):
                    if uid in captured_users:
                        continue
                    last_seen = vad_last_seen.get(uid, 0.0)
                    if (now_ts - last_seen) > idle_threshold:
                        name = await resolve_display_name(guild_obj, uid, None)
                        drained = vad_streams[uid].drain()
                        if drained:
                            for segment_wav in drained:
                                jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, segment_wav, dest))
                        vad_streams.pop(uid, None)
                        vad_last_seen.pop(uid, None)

                if jobs:
                    await asyncio.gather(*jobs, return_exceptions=True)
                elif not captured:
                    print("[STT] no audio captured in this window (VAD mode)")
                    await asyncio.sleep(0.3)
                continue

            # VADを通さない固定区切りモード
            if vad_streams:
                for uid, stream in list(vad_streams.items()):
                    name = await resolve_display_name(guild_obj, uid, None)
                    drained = stream.drain()
                    for segment_wav in drained:
                        jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, segment_wav, dest))
                vad_streams.clear()
                vad_last_seen.clear()

            captured_users: set[int] = set()
            for (uid, data, buf) in captured:
                captured_users.add(uid)
                name = await resolve_display_name(guild_obj, uid, data)
                pcm = _wav_bytes_to_pcm16k(buf)
                if not pcm:
                    continue
                for segment_pcm in _append_pcm_buffer(st, uid, pcm):
                    wav_seg = _pcm16k_to_wav_bytes(segment_pcm)
                    jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, wav_seg, dest))

            stale_segments = _flush_stale_pcm_buffers(st, captured_users)
            for uid, pcm_segment in stale_segments:
                name = await resolve_display_name(guild_obj, uid, None)
                wav_segment = _pcm16k_to_wav_bytes(pcm_segment)
                jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, wav_segment, dest))

            if not jobs:
                print("[STT] no audio captured in this window")
                await asyncio.sleep(0.3)
                continue

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
        if vad_streams:
            flush_dest = await resolve_message_channel(channel_id, guild_id)
            if flush_dest:
                flush_jobs = []
                for uid, stream in list(vad_streams.items()):
                    drained = stream.drain()
                    if not drained:
                        continue
                    name = await resolve_display_name(guild_obj, uid, None)
                    for segment_wav in drained:
                        flush_jobs.append(transcribe_and_post_from_bytes(guild_id, uid, name, segment_wav, flush_dest))
                if flush_jobs:
                    await asyncio.gather(*flush_jobs, return_exceptions=True)
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

    sink = PatchedWaveSink()
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

    gain_info = None
    try:
        buf, gain_info = _apply_gain_and_gate(buf, st)
    except Exception:
        traceback.print_exc()

    try:
        processed = await _maybe_denoise_wav(buf, st)
        if processed:
            buf = processed
    except Exception:
        traceback.print_exc()

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

    # --- STT モデル適用 ---
    models_to_try: list[str] = []
    primary_model = (st.get("stt_primary_model") or DEFAULT_PRIMARY_STT_MODEL).strip()
    fallback_model = (st.get("stt_fallback_model") or DEFAULT_FALLBACK_STT_MODEL).strip()

    def _append_model(name: str):
        n = (name or "").strip()
        if n and n not in models_to_try:
            models_to_try.append(n)

    _append_model(primary_model)
    _append_model(fallback_model)
    _append_model("whisper-1")  # 最終フォールバック

    text = ""
    used_model = None
    used_metrics: dict[str, float | int | None] = {}
    last_error: Exception | None = None

    for idx, model_name in enumerate(models_to_try):
        try:
            bio = io.BytesIO(buf)
            bio.name = "audio.wav"
            resp = openai.audio.transcriptions.create(
                file=bio,
                model=model_name,
                language="ja",
            )
            candidate = (getattr(resp, "text", "") or "").strip()
            metrics_candidate, retry_flag = _evaluate_transcription_response(resp, candidate, dur)
            print(f"[STT] {model_name} result: {candidate!r} retry={retry_flag}")
            if candidate:
                text = candidate
                used_model = model_name
                used_metrics = metrics_candidate
                if not retry_flag or idx == len(models_to_try) - 1:
                    break
                else:
                    print(f"[STT] retry requested, moving to next model after {model_name}")
                    continue
        except Exception as exc:
            last_error = exc
            print(f"[STT] transcription via {model_name} failed:", repr(exc))
            traceback.print_exc()

    if not text:
        if last_error:
            print("[STT] all models failed; last error:", repr(last_error))
        else:
            print("[STT] transcription produced empty text even after fallbacks")
        return

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

    if used_model and used_model != st.get("stt_primary_model"):
        print(f"[STT] fallback model used: {used_model}")

    fallback_used = bool(used_model and used_model != (st.get("stt_primary_model") or DEFAULT_PRIMARY_STT_MODEL))
    estimated_cost = _estimate_stt_cost(used_model or "", dur or 0.0)
    token_count = used_metrics.get("token_count") if used_metrics else None
    await log_stt_metrics(
        guild_id=guild_id,
        user_id=user_id,
        model=used_model or "unknown",
        fallback_used=fallback_used,
        duration=float(dur) if isinstance(dur, (int, float)) else None,
        avg_logprob=used_metrics.get("avg_logprob") if used_metrics else None,
        no_speech_prob=used_metrics.get("no_speech_prob") if used_metrics else None,
        compression_ratio=used_metrics.get("compression_ratio") if used_metrics else None,
        token_count=token_count if isinstance(token_count, (int, float)) else None,
        estimated_cost=estimated_cost,
        text_length=len(text or ""),
        gain_factor=(gain_info or {}).get("gain_factor"),
        gate_frames=(gain_info or {}).get("gate_frames"),
    )

    metrics_state = st.get("stt_metrics")
    if isinstance(metrics_state, dict):
        metrics_state["total_calls"] = metrics_state.get("total_calls", 0) + 1
        if fallback_used:
            metrics_state["fallback_calls"] = metrics_state.get("fallback_calls", 0) + 1
        metrics_state["total_duration"] = metrics_state.get("total_duration", 0.0) + (float(dur) if isinstance(dur, (int, float)) else 0.0)
        metrics_state["total_cost"] = metrics_state.get("total_cost", 0.0) + estimated_cost
        usage = metrics_state.get("model_usage")
        if isinstance(usage, collections.Counter):
            usage[used_model or "unknown"] += 1
        else:
            metrics_state["model_usage"] = collections.Counter({used_model or "unknown": 1})
        if gain_info:
            if gain_info.get("gain_applied"):
                metrics_state["gain_events"] = metrics_state.get("gain_events", 0) + 1
                metrics_state["gain_factor_sum"] = metrics_state.get("gain_factor_sum", 0.0) + float(gain_info.get("gain_factor", 1.0))
            metrics_state["gate_frames"] = metrics_state.get("gate_frames", 0) + int(gain_info.get("gate_frames", 0))

    # キャプション投稿（連投マージ対応）
    formatted = _heuristic_punctuate(text, dur)
    await post_caption(guild_id, channel, user_id, username, jp_cleanup(formatted))


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
        "desc": "今いるボイスチャンネルへボットを参加させます（Stage では話者化を試みます）。例: `{p}join`",
    },
    {
        "name": "leave", "aliases": ["執事退出","執事離脱"],
        "usage": "{p}leave",
        "desc": "ボイスチャンネルから退出します。例: `{p}leave`",
    },
    {
        "name": "readon", "aliases": ["読み上げコマンド","読み上げ","読み上げ開始","読み上げオン","このチャンネルを読み上げ"],
        "usage": "{p}readon",
        "desc": "このテキストチャンネルの新規メッセージをボイスチャンネルで読み上げます。例: `{p}readon`",
    },
    {
        "name": "readoff", "aliases": ["読み上げ停止","読み上げオフ"],
        "usage": "{p}readoff",
        "desc": "読み上げを停止します。例: `{p}readoff`",
    },
    {
        "name": "readhere", "aliases": ["ここを読み上げ"],
        "usage": "{p}readhere",
        "desc": "読み上げ対象チャンネルを“今ここ”に変更します。例: `{p}readhere`",
    },
    {
        "name": "stton", "aliases": ["字幕開始","文字起こし開始","字幕オン","音声認識開始"],
        "usage": "{p}stton [vad|fixed] [区切り秒数(3-60)]",
        "desc": "ボイスチャンネルの音声を文字起こしします。既定はVADモード。例: `{p}stton`, `{p}stton fixed 8`",
    },
    {
        "name": "sttoff", "aliases": ["字幕停止","文字起こし停止","字幕オフ","音声認識停止"],
        "usage": "{p}sttoff",
        "desc": "音声認識ワーカーを停止します。例: `{p}sttoff`",
    },
    {
        "name": "stttest", "aliases": ["文字起こしテスト"],
        "usage": "{p}stttest",
        "desc": "gTTS→Whisper の疎通テストを行います（日本語固定）。例: `{p}stttest`",
    },
    {
        "name": "rectest", "aliases": ["録音テスト"],
        "usage": "{p}rectest [秒数(2-30)]",
        "desc": "現在のボイスチャンネルを一時録音し、結果を返信します（デバッグ用）。例: `{p}rectest 5`",
    },
    {
        "name": "diag", "aliases": ["診断"],
        "usage": "{p}diag",
        "desc": "py-cord のバージョンや ffmpeg/PyNaCl などの診断情報を表示します。例: `{p}diag`",
    },
    {
        "name": "whereami", "aliases": [],
        "usage": "{p}whereami",
        "desc": "このテキストチャンネル（またはスレッド）の情報を表示します。例: `{p}whereami`",
    },
    {
        "name": "intentcheck", "aliases": [],
        "usage": "{p}intentcheck",
        "desc": "Members Intent 等の実際の挙動を簡易チェックします。例: `{p}intentcheck`",
    },
    {
        "name": "sttset", "aliases": [],
        "usage": (
            "{p}sttset <key> <value> / key: vad | vaddb | mindur | merge | mergeauto | lang | thread | "
            "mode | window | vadlevel | vadsilence | vadpre | vadpost | vadoverlap | denoise | denoisemode | denoisemodel | denoisehp | denoiselp | denoisestr | denoisegain | sttmodel | sttmodel2"
        ),
        "desc": (
            "VAD・ノイズ抑圧・利用モデルなど認識設定を調整します（言語は日本語固定）。"
            " 例: `{p}sttset vad 0.008`, `{p}sttset vadlevel 3`, `{p}sttset sttmodel gpt-4o-mini-transcribe`"
        ),
    },
    {
        "name": "sttstats", "aliases": ["stt統計", "sttmetrics"],
        "usage": "{p}sttstats",
        "desc": "モデル使用回数やフォールバック率、想定コストなど、直近の音声認識指標を表示します。例: `{p}sttstats`",
    },
    {
        "name": "sttcolor", "aliases": ["字幕色", "color"],
        "usage": "{p}sttcolor [export/import/ユーザー]",
        "desc": "字幕の色を管理します。0-15 のパレット指定や設定ファイルの入出力に対応します。例: `{p}sttcolor @自分 3`",
    },
    {
        "name": "voicevoxstyles", "aliases": ["voxstyles", "voxlist"],
        "usage": "{p}voicevoxstyles",
        "desc": "VOICEVOX の話者IDとスタイル名の一覧を表示します。例: `{p}voicevoxstyles`",
    },
    {
        "name": "sttpalette", "aliases": ["colorpalette", "palette"],
        "usage": "{p}sttpalette",
        "desc": "字幕カラーのパレット番号とカラーコードを確認します。例: `{p}sttpalette`",
    },
    {
        "name": "voxdict", "aliases": ["辞書管理"],
        "usage": "{p}voxdict <export|import|add>",
        "desc": "VOICEVOX のユーザー辞書をエクスポート/インポート/追加します。例: `{p}voxdict add テスト テスト`",
    },
    # ==== 管理者向け（表示制御） ====
    {
        "name": "ttsspeed", "aliases": ["読み上げ速度"],
        "usage": "{p}ttsspeed <倍率>",
        "desc": "サーバー全体の基準話速を設定します。例: `{p}ttsspeed 1.35`（推奨 0.6〜2.0）",
        "admin_only": True,
    },
    {
        "name": "ttsvoice", "aliases": ["声色"],
        "usage": "{p}ttsvoice @ユーザー (<半音> [テンポ] | reset)",
        "desc": "特定ユーザーの声色（半音）とテンポ係数を上書きします（gTTS 利用時のみ）。例: `{p}ttsvoice @太郎 +3 1.10` / `{p}ttsvoice @太郎 reset`",
        "admin_only": True,
    },
    {
        "name": "ttsconfig", "aliases": ["読み上げ設定"],
        "usage": "{p}ttsconfig",
        "desc": "現在の話速・個別声色オーバーライドの一覧を表示します。例: `{p}ttsconfig`",
        "admin_only": True,
    },
    {
        "name": "ttsspeaker", "aliases": ["スピーカー", "speaker"],
        "usage": "{p}ttsspeaker [default/export/import/ユーザー]",
        "desc": "VOICEVOX のデフォルト話者やユーザー別話者IDを管理します（VOICEVOX 利用時のみ）。例: `{p}ttsspeaker default 2`",
        "admin_only": True,
    },
    {
        "name": "logs", "aliases": ["ログ取得", "getlogs"],
        "usage": "{p}logs",
        "desc": "TTS/STT のログファイル（CSV）をダウンロードします。例: `{p}logs`",
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
             "stttest","rectest","diag","logs","whereami","intentcheck","sttset","sttstats",
             "sttcolor","sttpalette","voicevoxstyles","voxdict","ttsspeed","ttsvoice","ttsconfig","ttsspeaker"]
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

# === CCFOLIA BRIDGE START: web server & pump ===
def _ip_allowed(remote: str) -> bool:
    if not CCFO_ACCEPT_FROM:
        return True
    host = (remote or "").split(":")[0]
    return (remote in CCFO_ACCEPT_FROM) or (host in CCFO_ACCEPT_FROM)

# CORS ヘッダ（必要に応じて Origin を制限したければ "*" を "https://ccfolia.com" に）
def _cors_headers(origin: str | None) -> dict:
    allow_origin = origin if origin else "*"
    return {
        "Access-Control-Allow-Origin": allow_origin,
        "Vary": "Origin",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-CCF-Token",
        "Access-Control-Max-Age": "600",
        # PNA（ローカル宛て）を許可（Chrome系）
        "Access-Control-Allow-Private-Network": "true",
    }


def _build_openapi_spec(request: web.Request | None = None) -> dict[str, T.Any]:
    if request is not None and request.host:
        server_url = f"{request.scheme}://{request.host}"
    else:
        server_url = f"http://{CCFO_HOST}:{CCFO_PORT}"
    post_security = [{"CCFToken": []}] if CCFO_SECRET else []
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Discord STT/TTS Bot Bridge API",
            "version": "0.1.0",
            "description": (
                "Discord STT/TTS Bot が提供する CCFOLIA ブリッジ向けエンドポイントの仕様です。\n"
                "C.C.FOLIA などの外部ツールから Discord へイベントを転送するために利用します。"
            ),
        },
        "servers": [{"url": server_url}],
        "paths": {
            "/ccfolia_event": {
                "post": {
                    "tags": ["CCFOLIA Bridge"],
                    "summary": "テキストイベントの送信",
                    "description": (
                        "C.C.FOLIA などから取得したイベントを Discord へ転送するためのキューに積みます。"
                        " `CCFOLIA_POST_SECRET` が設定されている場合は `X-CCF-Token` ヘッダで一致するトークンを送信してください。"
                        " 未設定の場合、トークンヘッダは省略できます。"
                    ),
                    "security": post_security,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CCFoliaEventRequest"},
                                "example": {
                                    "speaker": "PL1",
                                    "text": "ダイス 1D100 → 27",
                                    "room": "卓名",
                                    "ts_client": "ccfolia-sync",
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "キュー投入に成功しました。",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GenericOkResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "入力値が不正です（JSON 解析失敗や text の未指定など）。",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                    "examples": {
                                        "invalid_json": {
                                            "summary": "JSON フォーマット誤り",
                                            "value": {"ok": False, "error": "invalid_json"},
                                        },
                                        "empty_text": {
                                            "summary": "text が空",
                                            "value": {"ok": False, "error": "empty_text"},
                                        },
                                    },
                                }
                            },
                        },
                        "401": {
                            "description": "認証に失敗しました。",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                    "example": {"ok": False, "error": "bad_token"},
                                }
                            },
                        },
                        "403": {
                            "description": "許可されていない IP アドレスからのアクセスです。",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                    "example": {"ok": False, "error": "forbidden_ip"},
                                }
                            },
                        },
                    },
                },
                "options": {
                    "tags": ["CCFOLIA Bridge"],
                    "summary": "CORS プリフライト",
                    "description": "ブラウザからのアクセス時に必要な CORS プリフライト要求へ応答します。",
                    "responses": {
                        "200": {
                            "description": "CORS 設定を示すレスポンスを返します。"
                        }
                    },
                },
            }
            ,
            "/gui": {
                "get": {
                    "tags": ["GUI"],
                    "summary": "GUI のトップページ",
                    "description": "読み上げ設定を管理するブラウザ用 GUI のエントリポイントを返します。",
                    "responses": {
                        "200": {
                            "description": "HTML 形式の GUI",
                            "content": {"text/html": {}},
                        }
                    },
                }
            },
            "/api/gui/config": {
                "get": {
                    "tags": ["GUI API"],
                    "summary": "GUI 初期設定の取得",
                    "description": "利用可能なギルドや現在の TTS プロバイダなど、GUI 表示に必要な初期情報を返します。",
                    "parameters": [
                        {
                            "in": "query",
                            "name": "guild_id",
                            "schema": {"type": "integer"},
                            "description": "対象とするギルド ID。省略時は最初のギルドを使用します。",
                        }
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "responses": {
                        "200": {
                            "description": "設定情報",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GuiConfigResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/api/gui/users": {
                "get": {
                    "tags": ["GUI API"],
                    "summary": "ユーザ設定一覧の取得",
                    "description": "ログから抽出したユーザリストと gTTS / VOICEVOX 個別設定を返します。",
                    "parameters": [
                        {
                            "in": "query",
                            "name": "guild_id",
                            "schema": {"type": "integer"},
                            "required": True,
                            "description": "対象ギルド ID。",
                        },
                        {
                            "in": "query",
                            "name": "q",
                            "schema": {"type": "string"},
                            "description": "部分一致によるユーザ名・ID 検索文字列。",
                        },
                        {
                            "in": "query",
                            "name": "refresh",
                            "schema": {"type": "string", "enum": ["0", "1"]},
                            "description": "\"1\" を指定するとキャッシュを無視します。",
                        },
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "responses": {
                        "200": {
                            "description": "ユーザ一覧",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GuiUserListResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/api/gui/gtts/{guild_id}/{user_id}": {
                "put": {
                    "tags": ["GUI API"],
                    "summary": "gTTS 個別設定の更新",
                    "description": "指定したユーザ ID に対する gTTS の半音・テンポ設定を更新します。",
                    "parameters": [
                        {
                            "in": "path",
                            "name": "guild_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                        {
                            "in": "path",
                            "name": "user_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GttsUpdateRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "更新成功",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GttsUpdateResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "リクエストが不正",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                },
                "delete": {
                    "tags": ["GUI API"],
                    "summary": "gTTS 個別設定の削除",
                    "description": "指定したユーザ ID に設定された gTTS 個別設定を削除します。",
                    "parameters": [
                        {
                            "in": "path",
                            "name": "guild_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                        {
                            "in": "path",
                            "name": "user_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "responses": {
                        "200": {
                            "description": "削除成功",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GttsUpdateResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                },
            },
            "/api/gui/voicevox/{guild_id}/{user_id}": {
                "put": {
                    "tags": ["GUI API"],
                    "summary": "VOICEVOX 個別話者設定の更新",
                    "description": "指定したユーザ ID に VOICEVOX 話者 ID を割り当てます。",
                    "parameters": [
                        {
                            "in": "path",
                            "name": "guild_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                        {
                            "in": "path",
                            "name": "user_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/VoicevoxUpdateRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "更新成功",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VoicevoxUpdateResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                },
                "delete": {
                    "tags": ["GUI API"],
                    "summary": "VOICEVOX 個別話者設定の削除",
                    "description": "指定したユーザ ID に設定された VOICEVOX 話者設定を削除します。",
                    "parameters": [
                        {
                            "in": "path",
                            "name": "guild_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                        {
                            "in": "path",
                            "name": "user_id",
                            "schema": {"type": "integer"},
                            "required": True,
                        },
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "responses": {
                        "200": {
                            "description": "削除成功",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VoicevoxUpdateResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                },
            },
            "/api/gui/voicevox/speakers": {
                "get": {
                    "tags": ["GUI API"],
                    "summary": "VOICEVOX 話者一覧の取得",
                    "description": "VOICEVOX エンジンから取得した話者名、アイコン、サンプル音声の一覧を返します。",
                    "parameters": [
                        {
                            "in": "query",
                            "name": "q",
                            "schema": {"type": "string"},
                            "description": "部分一致検索用の文字列。",
                        },
                        {
                            "in": "query",
                            "name": "refresh",
                            "schema": {"type": "string", "enum": ["0", "1"]},
                            "description": "\"1\" を指定すると VOICEVOX から再取得します。",
                        },
                    ],
                    "security": [{"GUIAdminToken": []}],
                    "responses": {
                        "200": {
                            "description": "話者一覧",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VoicevoxSpeakerListResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "認証失敗",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
        },
        "components": {
            "securitySchemes": {
                "CCFToken": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-CCF-Token",
                    "description": "CCFOLIA_POST_SECRET の値と一致するトークン。未設定時は不要です。",
                },
                "GUIAdminToken": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-Admin-Token",
                    "description": "GUI_ADMIN_TOKEN の値。GUI を保護するための簡易トークンです。",
                },
            },
            "schemas": {
                "CCFoliaEventRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "description": "Discord に送信する表示名。省略時は（未指定）。",
                        },
                        "text": {
                            "type": "string",
                            "description": "Discord に送信する本文。",
                        },
                        "room": {
                            "type": "string",
                            "description": "任意のルーム名やセッション名。",
                        },
                        "ts_client": {
                            "type": "string",
                            "description": "送信元クライアント識別子。",
                        },
                    },
                    "required": ["text"],
                },
                "GenericOkResponse": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ok": {"type": "boolean", "example": True},
                    },
                    "required": ["ok"],
                },
                "ErrorResponse": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ok": {"type": "boolean", "example": False},
                        "error": {
                            "type": "string",
                            "description": "エラー種別コード。",
                        },
                    },
                    "required": ["ok", "error"],
                },
                "GuiConfigResponse": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ok": {"type": "boolean", "example": True},
                        "provider": {"type": "string", "example": "voicevox"},
                        "guild_id": {"type": "integer", "example": 1234567890},
                        "requires_token": {"type": "boolean"},
                        "base_tempo": {"type": "number", "format": "float", "example": 0.7},
                        "default_voicevox_speaker": {"type": "integer", "example": 2},
                        "available_guilds": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "guild_id": {"type": "integer"},
                                    "name": {"type": "string"},
                                },
                                "required": ["guild_id", "name"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["ok", "provider", "guild_id", "requires_token", "available_guilds"],
                },
                "GuiUserListResponse": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "example": True},
                        "total": {"type": "integer", "example": 12},
                        "query": {"type": "string"},
                        "users": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/GuiUserEntry"},
                        },
                    },
                    "required": ["ok", "total", "users"],
                },
                "GuiUserEntry": {
                    "type": "object",
                    "properties": {
                        "user_name": {"type": ["string", "null"]},
                        "user_id": {"type": "integer"},
                        "author_id": {"type": "integer"},
                        "user_display": {"type": "string"},
                        "author_display": {"type": "string"},
                        "speaker_id": {"type": ["integer", "null"]},
                        "gtts_override": {
                            "oneOf": [
                                {"type": "null"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "semitones": {"type": "number", "format": "float"},
                                        "tempo": {"type": "number", "format": "float"},
                                    },
                                    "required": ["semitones", "tempo"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                        "user_ids": {"type": "array", "items": {"type": "integer"}},
                        "author_ids": {"type": "array", "items": {"type": "integer"}},
                        "candidate_user_ids": {"type": "array", "items": {"type": "integer"}},
                        "guild_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["user_id", "author_id", "user_display", "author_display"],
                    "additionalProperties": False,
                },
                "VoicevoxSpeakerListResponse": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "example": True},
                        "total": {"type": "integer"},
                        "query": {"type": "string"},
                        "speakers": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/VoicevoxSpeakerEntry"},
                        },
                    },
                    "required": ["ok", "total", "speakers"],
                },
                "VoicevoxSpeakerEntry": {
                    "type": "object",
                    "properties": {
                        "speaker_name": {"type": "string"},
                        "speaker_id": {"type": "integer"},
                        "speaker_uuid": {"type": "string"},
                        "style_name": {"type": "string"},
                        "icon": {"type": "string", "description": "data URL 形式のアイコン画像"},
                        "voice_samples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {"type": "integer"},
                                    "url": {"type": "string", "description": "data URL 形式の音声サンプル"},
                                },
                                "required": ["index", "url"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["speaker_name", "speaker_id", "speaker_uuid"],
                    "additionalProperties": False,
                },
                "GttsUpdateRequest": {
                    "type": "object",
                    "properties": {
                        "semitones": {"type": "number", "format": "float"},
                        "tempo": {"type": "number", "format": "float"},
                        "reset": {"type": "boolean", "description": "True を指定するとリセットを実施します。"},
                    },
                    "additionalProperties": False,
                },
                "GttsUpdateResponse": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "example": True},
                        "override": {
                            "type": ["object", "null"],
                            "properties": {
                                "semitones": {"type": "number"},
                                "tempo": {"type": "number"},
                            },
                        },
                    },
                    "required": ["ok"],
                },
                "VoicevoxUpdateRequest": {
                    "type": "object",
                    "properties": {
                        "speaker_id": {"type": "integer"},
                        "reset": {"type": "boolean"},
                        "author_id": {"type": "integer"},
                        "user_id": {"type": "integer"},
                        "author_display": {"type": "string"},
                        "user_display": {"type": "string"},
                    },
                    "required": ["author_id"],
                    "additionalProperties": False,
                },
                "VoicevoxUpdateResponse": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "example": True},
                        "speaker_id": {"type": ["integer", "null"]},
                        "author_id": {"type": ["integer", "null"]},
                        "user_id": {"type": ["integer", "null"]},
                        "author_display": {"type": ["string", "null"]},
                        "user_display": {"type": ["string", "null"]},
                    },
                    "required": ["ok"],
                },
            },
        },
    }


async def openapi_handler(request: web.Request):
    return web.json_response(_build_openapi_spec(request))


async def docs_handler(request: web.Request):
    spec_url = f"{request.scheme}://{request.host}/openapi.json"
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Discord STT/TTS Bot API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {{
      SwaggerUIBundle({{
        url: "{spec_url}",
        dom_id: '#swagger-ui',
        presets: [SwaggerUIBundle.presets.apis],
        layout: "BaseLayout"
      }});
    }};
  </script>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


# === GUI Web API START ===

async def gui_index_handler(request: web.Request):
    """GUI のトップページを返す。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        index.html を指す FileResponse。
    """
    index_path = GUI_STATIC_ROOT / "index.html"
    if not index_path.exists():
        raise web.HTTPNotFound(text="GUI assets are not available.")
    response = web.FileResponse(index_path)
    response.headers["Cache-Control"] = "no-store"
    return response

def _ensure_gui_auth(request: web.Request) -> None:
    """GUI API 用の簡易認証を行う。

    Args:
        request: 認証対象のリクエスト。
    """
    if not GUI_ADMIN_TOKEN:
        return

    header_token = (request.headers.get("X-Admin-Token") or "").strip()
    query_token = (request.query.get("token") or "").strip()
    token = header_token or query_token
    if token != GUI_ADMIN_TOKEN:
        payload = {"ok": False, "error": "unauthorized", "message": "GUI 管理トークンが一致しません。"}
        raise web.HTTPUnauthorized(
            text=json.dumps(payload, ensure_ascii=False),
            content_type="application/json",
            headers={"WWW-Authenticate": 'Token realm="GUI"'},
        )


def _gui_error(status: int, code: str, message: str | None = None) -> web.Response:
    """GUI API 向けのエラーレスポンスを生成する。

    Args:
        status: HTTP ステータスコード。
        code: エラー識別子。
        message: 追加の説明文。省略可。

    Returns:
        エラーメッセージを格納した JSON レスポンス。
    """
    payload: dict[str, T.Any] = {"ok": False, "error": code}
    if message:
        payload["message"] = message
    return web.json_response(payload, status=status, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))


def _extract_numeric_id(raw: T.Any) -> int | None:
    """CSV などから取得した値を Discord ユーザーIDとして解釈する。

    Args:
        raw: 解析対象の値。

    Returns:
        整数IDに変換できた場合はその値。失敗した場合は None。
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if math.isnan(raw) or math.isinf(raw):
            return None
        return int(raw)
    try:
        text = str(raw).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _wrap_data_url(raw: str | None, mime: str) -> str | None:
    """VOICEVOX から受け取った Base64 を data URL 形式へ変換する。

    Args:
        raw: Base64 文字列または data URL。
        mime: data URL に付与する MIME タイプ。

    Returns:
        data URL 形式の文字列。入力が空の場合は None。
    """
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.startswith("data:"):
        return value
    return f"data:{mime};base64,{value}"


def _load_gui_user_entries_sync() -> list[dict[str, T.Any]]:
    """GUI 向けにログファイルからユーザ情報を抽出する同期処理。

    Returns:
        ユーザ情報辞書のリスト。
    """
    entries: dict[str, dict[str, T.Any]] = {}

    def ensure_entry(name: str) -> dict[str, T.Any]:
        """名前に対応するエントリを作成または取得する。"""
        return entries.setdefault(
            name,
            {
                "user_name": name,
                "user_displays": set(),
                "user_ids": set(),
                "author_displays": set(),
                "author_ids": set(),
                "guild_ids": set(),
            },
        )

    if STT_LOG_PATH.exists():
        try:
            with open(STT_LOG_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    display = (row.get("user_display") or "").strip()
                    if not display:
                        continue
                    entry = ensure_entry(display)
                    entry["user_displays"].add(display)
                    uid = _extract_numeric_id(row.get("user_id"))
                    if uid is not None:
                        entry["user_ids"].add(uid)
                    gid = _extract_numeric_id(row.get("guild_id"))
                    if gid is not None:
                        entry["guild_ids"].add(gid)
        except Exception as exc:
            print(f"[GUI] failed to parse STT logs: {exc!r}")

    if TTS_LOG_PATH.exists():
        try:
            with open(TTS_LOG_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    display = (row.get("author_display") or "").strip()
                    if not display:
                        continue
                    entry = ensure_entry(display)
                    entry["author_displays"].add(display)
                    aid = _extract_numeric_id(row.get("author_id"))
                    if aid is not None:
                        entry["author_ids"].add(aid)
                    gid = _extract_numeric_id(row.get("guild_id"))
                    if gid is not None:
                        entry["guild_ids"].add(gid)
        except Exception as exc:
            print(f"[GUI] failed to parse TTS logs: {exc!r}")

    result: list[dict[str, T.Any]] = []
    for name, entry in entries.items():
        user_ids = sorted(entry["user_ids"])
        author_ids = sorted(entry["author_ids"])
        candidates = sorted({*user_ids, *author_ids})
        result.append(
            {
                "user_name": name,
                "user_displays": sorted(entry["user_displays"]),
                "user_ids": user_ids,
                "author_displays": sorted(entry["author_displays"]),
                "author_ids": author_ids,
                "candidate_user_ids": candidates,
                "guild_ids": sorted(entry["guild_ids"]),
            }
        )
    result.sort(key=lambda x: x["user_name"])
    return result


async def _gui_get_user_entries(force_refresh: bool = False) -> list[dict[str, T.Any]]:
    """GUI ユーザ一覧のキャッシュ管理を行いながらデータを取得する。

    Args:
        force_refresh: True の場合はキャッシュを無視して再読込する。

    Returns:
        ユーザ情報辞書のリスト。
    """
    now = time.time()
    async with _gui_user_cache_lock:
        cached_entries = _gui_user_cache.get("entries") or []
        cached_at = float(_gui_user_cache.get("timestamp") or 0.0)
        if cached_entries and not force_refresh and (now - cached_at) < GUI_USER_CACHE_TTL:
            return cached_entries

    loop = asyncio.get_running_loop()
    entries = await loop.run_in_executor(None, _load_gui_user_entries_sync)

    async with _gui_user_cache_lock:
        _gui_user_cache["timestamp"] = time.time()
        _gui_user_cache["entries"] = entries
    return entries


def _gui_entry_matches_keyword(entry: dict[str, T.Any], keyword: str) -> bool:
    """ユーザエントリが検索キーワードに一致するか判定する。

    Args:
        entry: 判定対象のユーザエントリ。
        keyword: 文字列の部分一致に利用するキーワード。

    Returns:
        一致する場合は True。
    """
    q = (keyword or "").strip().lower()
    if not q:
        return True
    if q in (entry.get("user_name") or "").lower():
        return True
    for field in ("user_displays", "author_displays"):
        for value in entry.get(field, []):
            if q in str(value).lower():
                return True
    for field in ("user_ids", "author_ids", "candidate_user_ids"):
        for value in entry.get(field, []):
            if q in str(value).lower():
                return True
    return False


def _gui_enrich_entry_with_state(entry: dict[str, T.Any], state: dict | None) -> dict[str, T.Any]:
    """ギルド状態を付加して GUI へ返すユーザ情報を構築する。

    Args:
        entry: ログから抽出したユーザエントリ。
        state: ギルド固有の TTS 設定。

    Returns:
        状態情報付きのユーザエントリ。
    """
    payload = {
        "user_name": entry.get("user_name"),
        "user_displays": list(entry.get("user_displays", [])),
        "user_ids": list(entry.get("user_ids", [])),
        "author_displays": list(entry.get("author_displays", [])),
        "author_ids": list(entry.get("author_ids", [])),
        "candidate_user_ids": list(entry.get("candidate_user_ids", [])),
        "guild_ids": list(entry.get("guild_ids", [])),
    }

    gtts_overrides: dict[str, dict[str, float]] = {}
    voicevox_speakers: dict[str, int] = {}
    voicevox_primary_user_id: int | None = None
    voicevox_primary_speaker_id: int | None = None

    if state:
        overrides = state.get("tts_overrides", {})
        speakers = state.get("tts_speakers", {})
        for raw_id in entry.get("candidate_user_ids", []):
            try:
                user_id = int(raw_id)
            except Exception:
                continue
            ov = overrides.get(user_id)
            if isinstance(ov, dict):
                gtts_overrides[str(user_id)] = {
                    "semitones": float(ov.get("semitones", 0.0)),
                    "tempo": float(ov.get("tempo", 1.0)),
                }
            speaker_id = speakers.get(user_id)
            if speaker_id is not None:
                try:
                    voicevox_speakers[str(user_id)] = int(speaker_id)
                    if voicevox_primary_user_id is None:
                        voicevox_primary_user_id = int(user_id)
                        voicevox_primary_speaker_id = int(speaker_id)
                except Exception:
                    pass

    payload["gtts_overrides"] = gtts_overrides
    payload["voicevox_speakers"] = voicevox_speakers
    payload["voicevox_primary_user_id"] = voicevox_primary_user_id
    payload["voicevox_primary_speaker_id"] = voicevox_primary_speaker_id
    return payload


def _select_default_guild_id() -> int | None:
    """Bot が所属している最初のギルド ID を返す。

    Returns:
        最初のギルド ID。所属ギルドが無い場合は None。
    """
    if not bot.guilds:
        return None
    return bot.guilds[0].id


async def gui_config_handler(request: web.Request):
    """GUI 初期化時に必要な基本設定を返す。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        設定情報を含む JSON レスポンス。
    """
    _ensure_gui_auth(request)
    guild_id_param = (request.query.get("guild_id") or "").strip()
    guild_id: int | None = None

    if guild_id_param:
        try:
            guild_id = int(guild_id_param)
        except ValueError:
            return _gui_error(400, "invalid_guild_id", "guild_id は整数で指定してください。")

    if guild_id is None:
        guild_id = _select_default_guild_id() or 0

    state = get_state(guild_id) if guild_id else None
    payload: dict[str, T.Any] = {
        "ok": True,
        "provider": TTS_PROVIDER,
        "guild_id": guild_id,
        "requires_token": bool(GUI_ADMIN_TOKEN),
        "available_guilds": [
            {"guild_id": g.id, "name": g.name}
            for g in bot.guilds
        ],
    }
    if state:
        payload["base_tempo"] = float(state.get("tts_base_tempo", 0.7))
        payload["default_voicevox_speaker"] = int(
            state.get("tts_default_speaker", VOICEVOX_DEFAULT_SPEAKER)
        )
    return web.json_response(payload, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))


async def gui_users_handler(request: web.Request):
    """GUI 用ユーザ一覧と個別設定を返す。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        ユーザリストを含む JSON レスポンス。
    """
    _ensure_gui_auth(request)

    guild_id_param = (request.query.get("guild_id") or "").strip()
    guild_id = 0
    if guild_id_param:
        try:
            guild_id = int(guild_id_param)
        except ValueError:
            return _gui_error(400, "invalid_guild_id", "guild_id は整数で指定してください。")
    elif bot.guilds:
        guild_id = bot.guilds[0].id

    entries = await _gui_get_user_entries(force_refresh=request.query.get("refresh") == "1")
    keyword = (request.query.get("q") or "").strip()
    filtered = [entry for entry in entries if _gui_entry_matches_keyword(entry, keyword)]

    def _to_int_list(values: T.Iterable[T.Any] | None) -> list[int]:
        result: list[int] = []
        if not values:
            return result
        for raw in values:
            try:
                result.append(int(raw))
            except Exception:
                continue
        return result

    aggregated_speakers: dict[int, int] = {}
    aggregated_gtts: dict[int, dict[str, float]] = {}
    for guild in bot.guilds:
        st = get_state(guild.id)
        for raw_uid, speaker in (st.get("tts_speakers") or {}).items():
            try:
                aggregated_speakers[int(raw_uid)] = int(speaker)
            except Exception:
                continue
        for raw_uid, cfg in (st.get("tts_overrides") or {}).items():
            try:
                uid = int(raw_uid)
            except Exception:
                continue
            if isinstance(cfg, dict):
                try:
                    aggregated_gtts[uid] = {
                        "semitones": float(cfg.get("semitones", 0.0)),
                        "tempo": float(cfg.get("tempo", 1.0)),
                    }
                except Exception:
                    aggregated_gtts[uid] = {"semitones": 0.0, "tempo": 1.0}

    users: list[dict[str, T.Any]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for entry in filtered:
        user_name = entry.get("user_name")
        user_displays = list(entry.get("user_displays", []))
        author_displays = list(entry.get("author_displays", []))
        user_ids = _to_int_list(entry.get("user_ids"))
        author_ids = _to_int_list(entry.get("author_ids"))
        candidate_ids = _to_int_list(entry.get("candidate_user_ids"))

        base_user_display = user_displays[0] if user_displays else (user_name or "")
        base_author_display = (
            author_displays[0] if author_displays else base_user_display
        )

        if not author_ids:
            # 何らかの ID が取得できなければスキップ
            continue

        for author_id in author_ids:
            user_id = user_ids[0] if user_ids else None
            pair_key = (author_id, user_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            record = {
                "user_name": user_name,
                "user_id": str(user_id) if user_id is not None else None,
                "author_id": str(author_id),
                "user_display": base_user_display,
                "author_display": base_author_display,
                "speaker_id": aggregated_speakers.get(author_id),
                "gtts_override": aggregated_gtts.get(user_id) if user_id is not None else None,
                "user_ids": [str(x) for x in user_ids],
                "author_ids": [str(x) for x in author_ids],
                "candidate_user_ids": [str(x) for x in candidate_ids],
                "guild_ids": sorted(set(entry.get("guild_ids", []))),
            }
            users.append(record)

    users.sort(
        key=lambda item: (
            str(item.get("user_display") or item.get("user_name") or ""),
            int(item.get("author_id") or 0),
        )
    )

    payload = {"ok": True, "users": users, "total": len(users)}
    if keyword:
        payload["query"] = keyword
    return web.json_response(payload, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))


def _set_voicevox_speaker_global(author_id: int, speaker_id: int | None) -> None:
    """VOICEVOX 話者設定を全ギルドへ適用するヘルパー。"""

    for guild in bot.guilds:
        st = get_state(guild.id)
        if speaker_id is None:
            st["tts_speakers"].pop(author_id, None)
        else:
            st["tts_speakers"][author_id] = speaker_id
        _persist_tts_preferences(guild.id, st)


async def gui_update_gtts_handler(request: web.Request):
    """gTTS 向けの個別パラメータを更新する。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        更新後の設定を含む JSON レスポンス。
    """
    _ensure_gui_auth(request)
    if TTS_PROVIDER != "gtts":
        return _gui_error(400, "provider_mismatch", "現在の TTS プロバイダでは gTTS 設定を変更できません。")

    try:
        guild_id = int(request.match_info.get("guild_id"))
        user_id = int(request.match_info.get("user_id"))
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_path", "guild_id と user_id は整数で指定してください。")

    try:
        body = await request.json()
    except Exception:
        return _gui_error(400, "invalid_json", "JSON ボディを解析できませんでした。")

    if body.get("reset"):
        state = get_state(guild_id)
        state["tts_overrides"].pop(user_id, None)
        _persist_tts_preferences(guild_id, state)
        return web.json_response({"ok": True, "override": None})

    if "semitones" not in body:
        return _gui_error(400, "missing_field", "`semitones` を指定してください。")
    try:
        semitones = float(body.get("semitones"))
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_value", "`semitones` には数値を指定してください。")

    tempo_raw = body.get("tempo", 1.0)
    try:
        tempo = float(tempo_raw)
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_value", "`tempo` には数値を指定してください。")
    if tempo <= 0.0:
        return _gui_error(400, "invalid_value", "`tempo` は正の値で指定してください。")

    state = get_state(guild_id)
    state["tts_overrides"][user_id] = {"semitones": semitones, "tempo": tempo}
    _persist_tts_preferences(guild_id, state)
    return web.json_response({"ok": True, "override": {"semitones": semitones, "tempo": tempo}})


async def gui_delete_gtts_handler(request: web.Request):
    """gTTS の個別設定を削除する。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        削除結果を示す JSON レスポンス。
    """
    _ensure_gui_auth(request)
    if TTS_PROVIDER != "gtts":
        return _gui_error(400, "provider_mismatch", "現在の TTS プロバイダでは gTTS 設定を変更できません。")

    try:
        guild_id = int(request.match_info.get("guild_id"))
        user_id = int(request.match_info.get("user_id"))
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_path", "guild_id と user_id は整数で指定してください。")

    state = get_state(guild_id)
    state["tts_overrides"].pop(user_id, None)
    _persist_tts_preferences(guild_id, state)
    return web.json_response({"ok": True, "override": None})


async def gui_update_voicevox_handler(request: web.Request):
    """VOICEVOX 向けの話者設定を更新する。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        更新後の話者設定を含む JSON レスポンス。
    """
    _ensure_gui_auth(request)
    if TTS_PROVIDER != "voicevox":
        return _gui_error(400, "provider_mismatch", "VOICEVOX を利用していません。")

    try:
        _ = int(request.match_info.get("guild_id"))
        path_author_id = int(request.match_info.get("user_id"))
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_path", "guild_id と user_id は整数で指定してください。")

    try:
        body = await request.json()
    except Exception:
        return _gui_error(400, "invalid_json", "JSON ボディを解析できませんでした。")

    if body.get("reset"):
        author_raw = body.get("author_id", path_author_id)
        try:
            author_id = int(author_raw)
        except (TypeError, ValueError):
            return _gui_error(400, "invalid_value", "`author_id` には整数を指定してください。")
        _set_voicevox_speaker_global(author_id, None)
        return web.json_response({"ok": True, "speaker_id": None, "author_id": author_id})

    if "speaker_id" not in body:
        return _gui_error(400, "missing_field", "`speaker_id` を指定してください。")
    try:
        speaker_id = int(body.get("speaker_id"))
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_value", "`speaker_id` には整数を指定してください。")

    author_raw = body.get("author_id", path_author_id)
    user_raw = body.get("user_id")
    try:
        author_id = int(author_raw)
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_value", "`author_id` には整数を指定してください。")
    try:
        user_id = int(user_raw)
    except (TypeError, ValueError):
        user_id = None

    _set_voicevox_speaker_global(author_id, speaker_id)

    response_payload = {
        "ok": True,
        "speaker_id": speaker_id,
        "author_id": str(author_id),
        "user_id": str(user_id) if user_id is not None else None,
        "author_display": body.get("author_display"),
        "user_display": body.get("user_display"),
    }
    return web.json_response(response_payload)


async def gui_delete_voicevox_handler(request: web.Request):
    """VOICEVOX の個別話者設定を削除する。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        削除結果を示す JSON レスポンス。
    """
    _ensure_gui_auth(request)
    if TTS_PROVIDER != "voicevox":
        return _gui_error(400, "provider_mismatch", "VOICEVOX を利用していません。")

    try:
        _ = int(request.match_info.get("guild_id"))
        author_id = int(request.match_info.get("user_id"))
    except (TypeError, ValueError):
        return _gui_error(400, "invalid_path", "guild_id と user_id は整数で指定してください。")

    _set_voicevox_speaker_global(author_id, None)
    return web.json_response({"ok": True, "speaker_id": None, "author_id": str(author_id)})


async def _gui_get_voicevox_speakers(force_refresh: bool = False) -> list[dict[str, T.Any]]:
    """VOICEVOX から取得した話者情報をキャッシュしつつ整形して返す。

    Args:
        force_refresh: True の場合はキャッシュを破棄して再取得する。

    Returns:
        話者情報辞書のリスト。
    """
    if TTS_PROVIDER != "voicevox":
        return []

    now = time.time()
    async with _gui_voicevox_cache_lock:
        cached_items = _gui_voicevox_cache.get("items") or []
        cached_at = float(_gui_voicevox_cache.get("timestamp") or 0.0)
        if cached_items and not force_refresh and (now - cached_at) < GUI_VOICEVOX_CACHE_TTL:
            return cached_items

    speakers_data = await _voicevox_fetch_json("GET", "/speakers")
    if not isinstance(speakers_data, list):
        speakers_data = []

    compiled: list[dict[str, T.Any]] = []

    for speaker in speakers_data:
        uuid = speaker.get("speaker_uuid")
        if not uuid:
            continue
        try:
            info = await _voicevox_fetch_json("GET", "/speaker_info", params={"speaker_uuid": uuid})
        except Exception as exc:
            print(f"[GUI] failed to fetch speaker_info for {uuid}: {exc!r}")
            info = {}

        style_infos: dict[int, dict[str, T.Any]] = {}
        if isinstance(info, dict):
            for style_info in info.get("style_infos", []):
                sid = style_info.get("id")
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                style_infos[sid_int] = style_info

        styles = speaker.get("styles") or []
        if not isinstance(styles, list):
            styles = []

        for style in styles:
            sid = style.get("id")
            if sid is None:
                continue
            try:
                style_id = int(sid)
            except Exception:
                continue
            style_name = style.get("name") or ""
            speaker_name = speaker.get("name") or ""
            combined_name = f"{speaker_name}({style_name})" if style_name else speaker_name
            style_info = style_infos.get(style_id, {})
            icon = _wrap_data_url(style_info.get("icon"), "image/png")
            samples_raw = style_info.get("voice_samples") or []
            sample_urls: list[dict[str, T.Any]] = []
            if isinstance(samples_raw, list):
                for idx, sample in enumerate(samples_raw):
                    sample_url = _wrap_data_url(sample, "audio/wav")
                    if sample_url:
                        sample_urls.append({"index": idx, "url": sample_url})

            compiled.append(
                {
                    "speaker_name": combined_name,
                    "speaker_id": style_id,
                    "speaker_uuid": uuid,
                    "style_name": style_name,
                    "icon": icon,
                    "voice_samples": sample_urls,
                }
            )

    compiled.sort(key=lambda x: (x["speaker_name"], x["speaker_id"]))

    async with _gui_voicevox_cache_lock:
        _gui_voicevox_cache["timestamp"] = time.time()
        _gui_voicevox_cache["items"] = compiled
    return compiled


async def gui_voicevox_speakers_handler(request: web.Request):
    """VOICEVOX の話者リストを返す。

    Args:
        request: aiohttp のリクエスト。

    Returns:
        話者情報を含む JSON レスポンス。
    """
    _ensure_gui_auth(request)
    if TTS_PROVIDER != "voicevox":
        return _gui_error(400, "provider_mismatch", "VOICEVOX を利用していません。")

    keyword = (request.query.get("q") or "").strip().lower()
    force_refresh = request.query.get("refresh") == "1"

    speakers = await _gui_get_voicevox_speakers(force_refresh=force_refresh)
    if keyword:
        filtered: list[dict[str, T.Any]] = []
        for item in speakers:
            haystacks = [
                item.get("speaker_name") or "",
                item.get("style_name") or "",
                item.get("speaker_uuid") or "",
                str(item.get("speaker_id", "")),
            ]
            if any(keyword in str(h).lower() for h in haystacks):
                filtered.append(item)
        speakers = filtered

    payload: dict[str, T.Any] = {"ok": True, "speakers": speakers, "total": len(speakers)}
    if keyword:
        payload["query"] = keyword
    return web.json_response(payload, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))


# === GUI Web API END ===

async def ccfo_options_handler(request: web.Request):
    # プリフライトへ 200 応答 + CORS ヘッダ
    origin = request.headers.get("Origin")
    return web.Response(status=200, headers=_cors_headers(origin))

async def ccfo_post_handler(request: web.Request):
    origin = request.headers.get("Origin")
    # IP/Token チェックはこれまで通り
    if not _ip_allowed(request.remote or ""):
        return web.json_response({"ok": False, "error": "forbidden_ip"}, status=403, headers=_cors_headers(origin))
    token = request.headers.get("X-CCF-Token", "")
    if CCFO_SECRET and token != CCFO_SECRET:
        return web.json_response({"ok": False, "error": "bad_token"}, status=401, headers=_cors_headers(origin))
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400, headers=_cors_headers(origin))

    speaker_name = (data.get("speaker") or "（未指定）").strip()
    raw_text = (data.get("text") or "").strip()
    room_name = (data.get("room") or "").strip()
    client_event_tag = (data.get("ts_client") or "").strip()
    normalized_text_for_log = _norm_text_for_csv(raw_text)  # Discord連携用に正規化したテキスト
    post_debug_payload = {
        "remote": request.remote,
        "speaker": speaker_name,
        "text": raw_text,
        "room": room_name,
        "ts_client": client_event_tag,
    }
    _ccfo_debug_log("post_received", post_debug_payload)
    if not normalized_text_for_log:
        _ccfo_debug_log("post_rejected_empty_text", post_debug_payload)
        return web.json_response({"ok": False, "error": "empty_text"}, status=400, headers=_cors_headers(origin))

    event_timestamp = datetime.now(timezone.utc)
    event_record = CCFoliaEventRecord(
        speaker=speaker_name,
        text=normalized_text_for_log,
        room=room_name,
        ts_client=client_event_tag,
        timestamp_utc=event_timestamp,
    )
    is_new_event = await log_ccfolia_event(event=event_record)  # 重複を避けつつログへ記録
    _ccfo_debug_log("event_logged", {**post_debug_payload, "is_new": is_new_event})
    if not is_new_event:
        _ccfo_debug_log("event_skipped_duplicate", post_debug_payload)
        return web.json_response({"ok": True}, headers=_cors_headers(origin))

    await ccfo_queue.put({"speaker": speaker_name, "text": raw_text, "room": room_name, "ts_client": client_event_tag})
    _ccfo_debug_log(
        "event_enqueued",
        {**post_debug_payload, "queue_size": ccfo_queue.qsize()},
    )
    return web.json_response({"ok": True}, headers=_cors_headers(origin))


async def _start_ccfo_web_server():
    """CCFOLIA 向け Web サーバーを起動する。"""
    global _ccfo_web_runner, _ccfo_web_site
    if _ccfo_web_runner is not None:
        return
    app = web.Application()
    app.add_routes([
        web.post("/ccfolia_event", ccfo_post_handler),
        web.options("/ccfolia_event", ccfo_options_handler),  # ← 追加
        web.get("/openapi.json", openapi_handler),
        web.get("/docs", docs_handler),
        web.get("/gui", gui_index_handler),
        web.get("/gui/", gui_index_handler),
        web.get("/api/gui/config", gui_config_handler),
        web.get("/api/gui/users", gui_users_handler),
        web.put("/api/gui/gtts/{guild_id}/{user_id}", gui_update_gtts_handler),
        web.delete("/api/gui/gtts/{guild_id}/{user_id}", gui_delete_gtts_handler),
        web.put("/api/gui/voicevox/{guild_id}/{user_id}", gui_update_voicevox_handler),
        web.delete("/api/gui/voicevox/{guild_id}/{user_id}", gui_delete_voicevox_handler),
        web.get("/api/gui/voicevox/speakers", gui_voicevox_speakers_handler),
    ])
    if GUI_STATIC_ROOT.exists():
        app.router.add_static("/gui/static", str(GUI_STATIC_ROOT), show_index=False)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=CCFO_HOST, port=CCFO_PORT)
    await site.start()
    _ccfo_web_runner = runner
    _ccfo_web_site = site
    print(f"[CCFOLIA] bridge server started at http://{CCFO_HOST}:{CCFO_PORT}")

def _resolve_ccfo_speaker_id(name: str) -> int:
    return int(CCFO_SPK_MAP.get(name, CCFO_SPK_MAP.get("（未指定）", CCFO_DEFAULT_SPK)))

async def _ccfo_send_text(ch: discord.TextChannel, speaker: str, text: str, room: str, ts_client: str):
    header = f"[{room}] {speaker}" if room else speaker
    stamp = f" `({ts_client})`" if ts_client else ""
    content = f"**{header}**{stamp}\n{text}"
    print(['content',content]) # for debug
    print(ch) # for debug
    await ch.send(content)

async def _ccfo_generate_voicevox_wav(text: str, spk_id: int) -> bytes:
    """VOICEVOX で合成した WAV データを返す。"""
    return await _voicevox_synthesize(sanitize_for_tts(text), spk_id)

async def _ccfo_send_voicevox_file(ch: discord.TextChannel, speaker: str, text: str, spk_id: int, wav: bytes | None = None):
    # 既存の VOICEVOX 同期関数を流用
    try:
        wav_bytes = wav if wav is not None else await _ccfo_generate_voicevox_wav(text, spk_id)
    except Exception as e:
        await ch.send(f"【TTS失敗:{speaker}】{e!r}")
        return
    bio = io.BytesIO(wav_bytes); bio.seek(0)
    await ch.send(file=discord.File(bio, filename=f"{speaker}.wav"))

def _register_background_task(task: asyncio.Task[object]) -> None:
    """バックグラウンドタスクを登録して後段のキャンセルに備える。

    Args:
        task: 追跡対象のタスク。
    """
    _ccfo_background_tasks.add(task)
    task.add_done_callback(lambda finished: _ccfo_background_tasks.discard(finished))  # type: ignore[arg-type]


async def _cancel_background_tasks() -> None:
    """起動中のバックグラウンドタスクをすべてキャンセルする。"""
    if not _ccfo_background_tasks:
        return
    for background_task in list(_ccfo_background_tasks):
        background_task.cancel()
    await asyncio.gather(*_ccfo_background_tasks, return_exceptions=True)
    _ccfo_background_tasks.clear()


async def _stop_ccfo_web_server() -> None:
    """CCFOLIA ブリッジ用 Web サーバーを停止する。"""
    global _ccfo_web_site, _ccfo_web_runner
    if _ccfo_web_site is not None:
        await _ccfo_web_site.stop()
        _ccfo_web_site = None
    if _ccfo_web_runner is not None:
        await _ccfo_web_runner.cleanup()
        _ccfo_web_runner = None


async def _disconnect_all_voice_clients(bot_instance: commands.Bot) -> None:
    """ボットが接続中の全ボイスチャンネルから切断する。

    Args:
        bot_instance: 対象の Bot インスタンス。
    """
    voice_clients = list(bot_instance.voice_clients)
    if not voice_clients:
        return
    disconnect_coroutines: list[asyncio.Task[object]] = []
    for voice_client in voice_clients:
        try:
            await ensure_stopped(voice_client, "shutdown")
        except Exception:
            traceback.print_exc()
        try:
            disconnect_coroutines.append(asyncio.create_task(voice_client.disconnect(force=True)))
        except Exception:
            traceback.print_exc()
    if disconnect_coroutines:
        await asyncio.gather(*disconnect_coroutines, return_exceptions=True)


async def _ccfo_play_in_vc(guild: discord.Guild, text: str, spk_id: int, wav: bytes | None = None):
    vc = guild.voice_client
    if not vc or not vc.is_connected():
        return False
    # 直接 VOICEVOX で合成 → 一時WAV → VC 再生
    try:
        wav_bytes = wav if wav is not None else await _ccfo_generate_voicevox_wav(text, spk_id)
    except Exception as e:
        print("[CCFOLIA] VC TTS failed:", repr(e))
        return False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name; f.write(wav_bytes)
    try:
        await _play_vc_audio(vc, tmp)
        return True
    finally:
        try: os.remove(tmp)
        except: pass

async def ccfo_pump_worker():
    await bot.wait_until_ready()
    if not CCFO_MIRROR_CH_ID:
        print("[CCFOLIA] WARNING: CCFOLIA_MIRROR_CHANNEL_ID is not set; events will be dropped.")
    _ccfo_debug_log("pump_worker_started", {"mirror_channel_id": CCFO_MIRROR_CH_ID})
    text_ch = bot.get_channel(CCFO_MIRROR_CH_ID) if CCFO_MIRROR_CH_ID else None

    try:
        while True:
            event_payload = await ccfo_queue.get()
            speaker_name = event_payload.get("speaker") or "（未指定）"
            text_for_dispatch = event_payload.get("text") or ""
            room_name = event_payload.get("room") or ""
            client_event_tag = event_payload.get("ts_client") or ""
            resolved_speaker_id = _resolve_ccfo_speaker_id(speaker_name)
            queue_size = max(ccfo_queue.qsize(), 0)
            event_info = {
                "speaker": speaker_name,
                "text": text_for_dispatch,
                "room": room_name,
                "ts_client": client_event_tag,
                "spk_id": resolved_speaker_id,
                "queue_size": queue_size,
            }
            _ccfo_debug_log("event_dequeued", event_info)

            if not text_ch:
                text_ch = bot.get_channel(CCFO_MIRROR_CH_ID) if CCFO_MIRROR_CH_ID else None

            # 音声生成フェーズ
            tts_bytes: bytes | None = None
            tts_input_text = ""
            if CCFO_TTS_MODE in ("file", "voice") and text_for_dispatch:
                tts_input_text = (
                    text_for_dispatch if CCFO_TTS_MODE == "file" else f"{speaker_name}：{text_for_dispatch}"
                )
                _ccfo_debug_log(
                    "tts_generation_start",
                    {
                        "mode": CCFO_TTS_MODE,
                        "spk_id": resolved_speaker_id,
                        "text": tts_input_text,
                    },
                )
                try:
                    tts_bytes = await _ccfo_generate_voicevox_wav(tts_input_text, resolved_speaker_id)
                    _ccfo_debug_log(
                        "tts_generation_complete",
                        {
                            "mode": CCFO_TTS_MODE,
                            "spk_id": resolved_speaker_id,
                            "bytes": len(tts_bytes),
                        },
                    )
                except Exception as exc:
                    _ccfo_debug_log(
                        "tts_generation_failed",
                        {
                            "mode": CCFO_TTS_MODE,
                            "spk_id": resolved_speaker_id,
                            "error": repr(exc),
                        },
                    )
                    tts_bytes = None
            else:
                _ccfo_debug_log(
                    "tts_generation_skipped",
                    {"reason": "mode_off_or_empty_text", "mode": CCFO_TTS_MODE},
                )

            # Discord 送信フェーズ
            text_sent = False
            attachment_sent = False
            vc_played = False
            _ccfo_debug_log(
                "discord_dispatch_start",
                {
                    "channel_id": getattr(text_ch, "id", None),
                    "tts_mode": CCFO_TTS_MODE,
                },
            )

            if isinstance(text_ch, discord.TextChannel):
                try:
                    await _ccfo_send_text(text_ch, speaker_name, text_for_dispatch, room_name, client_event_tag)
                    text_sent = True
                except Exception as exc:
                    _ccfo_debug_log(
                        "discord_text_send_failed",
                        {"channel_id": text_ch.id, "error": repr(exc)},
                    )

                if CCFO_TTS_MODE == "file" and tts_bytes:
                    _ccfo_debug_log(
                        "discord_file_send_start",
                        {"channel_id": text_ch.id, "bytes": len(tts_bytes)},
                    )
                    try:
                        await _ccfo_send_voicevox_file(text_ch, speaker_name, text_for_dispatch, resolved_speaker_id, wav=tts_bytes)
                        attachment_sent = True
                    except Exception as exc:
                        _ccfo_debug_log(
                            "discord_file_send_failed",
                            {"channel_id": text_ch.id, "error": repr(exc)},
                        )
            else:
                if CCFO_MIRROR_CH_ID:
                    _ccfo_debug_log(
                        "discord_channel_not_found",
                        {"channel_id": CCFO_MIRROR_CH_ID},
                    )

            if CCFO_TTS_MODE == "voice" and text_for_dispatch:
                speech_text = f"{speaker_name}：{text_for_dispatch}"
                _ccfo_debug_log(
                    "vc_playback_start",
                    {"guilds": [g.id for g in bot.guilds], "text": speech_text},
                )
                guilds = bot.guilds
                ok = False
                for g in guilds:
                    ok = await _ccfo_play_in_vc(g, speech_text, resolved_speaker_id, wav=tts_bytes)
                    if ok:
                        vc_played = True
                        break
                if not ok and isinstance(text_ch, discord.TextChannel):
                    await text_ch.send("（VC未接続のためTTSは再生できませんでした）")
                _ccfo_debug_log(
                    "vc_playback_complete",
                    {"success": ok, "guild_ids": [g.id for g in guilds]},
                )

            _ccfo_debug_log(
                "discord_dispatch_complete",
                {
                    "text_sent": text_sent,
                    "attachment_sent": attachment_sent,
                    "vc_played": vc_played,
                },
            )
    except asyncio.CancelledError:
        print("[CCFOLIA] pump worker cancelled")
        raise

# 起動時に webサーバ と ポンプを起動
_original_on_ready = bot.on_ready

@bot.event
async def on_ready():
    # 既存の on_ready ロジックを維持
    if _original_on_ready:
        try:
            await _original_on_ready()
        except TypeError:
            # 既存が同期関数の可能性にも一応配慮
            pass
    # 一度だけ起動
    if not getattr(bot, "_ccfo_server_started", False):
        bot._ccfo_server_started = True
        server_task = bot.loop.create_task(_start_ccfo_web_server(), name="ccfo_web_server")
        _register_background_task(server_task)
        pump_task = bot.loop.create_task(ccfo_pump_worker(), name="ccfo_pump_worker")
        _register_background_task(pump_task)
        print("[CCFOLIA] pump & server tasks started")
# === CCFOLIA BRIDGE END ===


def main() -> None:
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN is not set. Update your environment or .env file.")
    bot.run(TOKEN)


if __name__ == "__main__":
    main()
