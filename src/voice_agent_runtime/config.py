"""Runtime configuration from environment variables.

The runtime is operated by the service provider: LLM endpoints and keys, models, thresholds and limits are
service settings, not per-PBX data. Per-PBX data (agent bundles) arrives from the PBXs at run time and is
kept in memory only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_LLM_URL = "https://foundation-models.api.cloud.ru/v1/chat/completions"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_LLM_EXTRA = {"reasoning_effort": "low"}


@dataclass(frozen=True)
class Voice:
    """A voice the runtime can speak with. ``speaker`` is the model's own name for it."""
    id: str
    engine: str  # "piper" or "silero"
    name: str
    gender: str  # "male" / "female"
    speaker: str


# Only voices that may be used commercially: Piper dmitri and denis (CC0 datasets; irina has no clear license,
# ruslan is CC BY-NC-SA) and the Russian speakers of Silero v5_cis_base (MIT; the v5_ru voices are CC BY-NC).
VOICES: dict[str, Voice] = {v.id: v for v in (
    Voice("ru_RU-dmitri-medium", "piper", "Дмитрий", "male", "dmitri"),
    Voice("ru_RU-denis-medium", "piper", "Денис", "male", "denis"),
    Voice("silero-ekaterina", "silero", "Екатерина", "female", "ru_ekaterina"),
    Voice("silero-oksana", "silero", "Оксана", "female", "ru_oksana"),
    Voice("silero-zinaida", "silero", "Зинаида", "female", "ru_zinaida"),
    Voice("silero-roman", "silero", "Роман", "male", "ru_roman"),
    Voice("silero-eduard", "silero", "Эдуард", "male", "ru_eduard"),
)}
SUPPORTED_VOICES = tuple(VOICES)
DEFAULT_VOICE = "ru_RU-dmitri-medium"


class ConfigError(ValueError):
    """The environment does not describe a usable runtime."""


@dataclass(frozen=True)
class LlmProvider:
    name: str
    url: str
    model: str
    api_key: str
    extra: dict = field(default_factory=dict)
    timeout: float = 30.0
    stream_usage: bool = True  # ask for token usage in the stream (stream_options.include_usage)


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8443
    tls_cert: str = ""
    tls_key: str = ""
    trusted_proxy: bool = False
    tenants_file: Path = Path("/config/tenants.json")
    models_dir: Path = Path("/models")
    primary: LlmProvider | None = None
    secondary: LlmProvider | None = None
    hedge_ms: int = 600
    # Failover to the secondary for every call after this many slow or failed turns of the primary in a row
    # (0: off); while on the secondary, the primary is probed every recovery_probe_s with recovery_probes
    # requests, all answered within the hedge time, before it gets the turns back.
    failover_after: int = 3
    failover_window_s: int = 300
    recovery_probe_s: int = 28800
    recovery_probes: int = 3
    temperature: float = 0.2
    max_tokens: int = 400
    speculate_ms: int = 120
    turn_end_ms: int = 600
    question_end_ms: int = 300
    barge_in_frames: int = 4
    vad_threshold: float = 0.5
    max_calls: int = 4
    max_calls_per_tenant: int = 0  # 0 = only the global limit applies
    max_agents_per_tenant: int = 20
    max_results_per_tenant: int = 1000
    asr_threads: int = 1
    result_ttl_s: int = 900
    max_call_s: int = 1800
    ws_heartbeat_s: int = 20  # 0 disables the WebSocket ping on silent links
    drain_s: int = 30
    phrase_cache_mb: int = 256
    admin_token: str = ""
    default_voice: str = DEFAULT_VOICE
    voices: tuple[str, ...] = SUPPORTED_VOICES  # loaded voices; a bundle with another voice speaks with the default
    log_level: str = "INFO"
    log_transcripts: bool = False
    stt_chunks: bool = True
    stt_chunk_pause_ms: int = 160
    stt_chunk_min_ms: int = 1000
    hotwords: bool = True
    aec: bool = True  # echo cancellation of the caller audio (needs livekit; off without it)
    topic_filter: bool = True  # a side LLM request per caller phrase refuses topics foreign to the company
    timezone: str = "Europe/Moscow"  # the date and time the agent knows (VA_TIMEZONE)
    hotword_boost: float = 0.45
    usage_db: Path | None = None
    usage_flush_s: int = 30
    # The other instance of a blue-green pair (http://runtime-green:8443): bundles are taken over from it at
    # start, and results and acknowledgements of calls it served are forwarded to it. Needs VA_ADMIN_TOKEN.
    peer_url: str = ""
    # Bundles saved by the instance before a restart without a peer (scripts/deploy.sh --restart), loaded and
    # deleted at start; None disables. On the data volume both colors share.
    handoff_file: Path | None = Path("/data/handoff.json")

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert and self.tls_key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        primary = _provider(env, "LLM", "primary", required=True)
        secondary = _provider(env, "LLM2", "secondary", required=False)
        tls_cert, tls_key = env.get("VA_TLS_CERT", "").strip(), env.get("VA_TLS_KEY", "").strip()
        if bool(tls_cert) != bool(tls_key):
            raise ConfigError("VA_TLS_CERT and VA_TLS_KEY must be set together")
        usage_db = env.get("VA_USAGE_DB", "/data/usage.sqlite3").strip()  # empty: usage counted in RAM only
        voices = voices_from_env(env)
        peer_url = env.get("VA_PEER_URL", "").strip().rstrip("/")
        handoff_file = env.get("VA_HANDOFF_FILE", "/data/handoff.json").strip()
        if peer_url and not peer_url.startswith(("http://", "https://")):
            raise ConfigError("VA_PEER_URL must be an http(s) URL")
        admin_token = _secret(env, "VA_ADMIN_TOKEN")
        if peer_url and not admin_token:
            raise ConfigError("VA_PEER_URL needs VA_ADMIN_TOKEN (the same in both instances)")
        voice = env.get("VA_DEFAULT_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
        if voice not in voices:
            raise ConfigError(f"VA_DEFAULT_VOICE must be one of the loaded voices: {', '.join(voices)}")
        return cls(
            host=env.get("VA_LISTEN_HOST", "0.0.0.0"),
            port=_int(env, "VA_LISTEN_PORT", 8443, 1, 65535),
            tls_cert=tls_cert,
            tls_key=tls_key,
            trusted_proxy=_bool(env.get("VA_TRUSTED_PROXY", "")),
            tenants_file=Path(env.get("VA_TENANTS_FILE", "/config/tenants.json")),
            models_dir=Path(env.get("VA_MODELS_DIR", "/models")),
            primary=primary,
            secondary=secondary,
            hedge_ms=_int(env, "LLM_HEDGE_MS", 600, 0, 10000),
            failover_after=_int(env, "LLM_FAILOVER_AFTER", 3, 0, 100),
            failover_window_s=_int(env, "LLM_FAILOVER_WINDOW_S", 300, 10, 86400),
            recovery_probe_s=_int(env, "LLM_RECOVERY_PROBE_S", 28800, 60, 604800),
            recovery_probes=_int(env, "LLM_RECOVERY_PROBES", 3, 1, 10),
            temperature=_float(env, "LLM_TEMPERATURE", 0.2, 0.0, 2.0),
            max_tokens=_int(env, "LLM_MAX_TOKENS", 400, 50, 4000),
            speculate_ms=_int(env, "VA_SPECULATE_MS", 120, 32, 1000),
            turn_end_ms=_int(env, "VA_TURN_END_MS", 600, 200, 3000),
            question_end_ms=_int(env, "VA_QUESTION_END_MS", 300, 100, 3000),
            barge_in_frames=_int(env, "VA_BARGE_IN_FRAMES", 4, 1, 50),
            vad_threshold=_float(env, "VA_VAD_THRESHOLD", 0.5, 0.05, 0.95),
            max_calls=_max_calls(env),
            max_calls_per_tenant=_int(env, "VA_MAX_CALLS_PER_TENANT", 0, 0, 1000),
            max_agents_per_tenant=_int(env, "VA_MAX_AGENTS_PER_TENANT", 20, 1, 10000),
            max_results_per_tenant=_int(env, "VA_MAX_RESULTS_PER_TENANT", 1000, 10, 100000),
            asr_threads=_int(env, "VA_ASR_THREADS", 1, 1, 64),
            result_ttl_s=_int(env, "VA_RESULT_TTL", 900, 60, 86400),
            max_call_s=_int(env, "VA_MAX_CALL_SECONDS", 1800, 60, 14400),
            ws_heartbeat_s=_int(env, "VA_WS_HEARTBEAT", 20, 0, 300),
            drain_s=_int(env, "VA_DRAIN_SECONDS", 30, 0, 1800),
            phrase_cache_mb=_int(env, "VA_PHRASE_CACHE_MB", 256, 8, 8192),
            admin_token=admin_token,
            default_voice=voice,
            voices=voices,
            log_level=env.get("VA_LOG_LEVEL", "INFO").upper(),
            log_transcripts=_bool(env.get("VA_LOG_TRANSCRIPTS", "")),
            stt_chunks=_bool(env.get("VA_STT_CHUNKS", "true")),
            stt_chunk_pause_ms=_int(env, "VA_STT_CHUNK_PAUSE_MS", 160, 64, 2000),
            stt_chunk_min_ms=_int(env, "VA_STT_CHUNK_MIN_MS", 1000, 300, 30000),
            hotwords=_bool(env.get("VA_HOTWORDS", "true")),
            aec=_bool(env.get("VA_AEC", "true")),
            topic_filter=_bool(env.get("VA_TOPIC_FILTER", "true")),
            timezone=_timezone(env),
            hotword_boost=_float(env, "VA_HOTWORD_BOOST", 0.45, 0.0, 2.0),
            usage_db=Path(usage_db) if usage_db else None,
            usage_flush_s=_int(env, "VA_USAGE_FLUSH_SECONDS", 30, 1, 3600),
            peer_url=peer_url,
            handoff_file=Path(handoff_file) if handoff_file else None,
        )


def voices_from_env(env: Mapping[str, str]) -> tuple[str, ...]:
    """``VA_VOICES``: comma-separated voice ids to load (default: all). Without Silero voices torch is not loaded."""
    raw = env.get("VA_VOICES", "").strip()
    if not raw:
        return SUPPORTED_VOICES
    voices = tuple(dict.fromkeys(v.strip() for v in raw.split(",") if v.strip()))
    unknown = [v for v in voices if v not in VOICES]
    if unknown or not voices:
        raise ConfigError(f"VA_VOICES: unknown voice {', '.join(unknown) or '(empty)'}; known: {', '.join(SUPPORTED_VOICES)}")
    return voices


def _timezone(env: Mapping[str, str]) -> str:
    name = env.get("VA_TIMEZONE", "").strip() or "Europe/Moscow"
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"VA_TIMEZONE: unknown time zone {name!r}") from exc
    return name


def _secret(env: Mapping[str, str], key: str) -> str:
    """A secret from ``KEY`` or, when that is empty, from the file named by ``KEY_FILE``."""
    value = env.get(key, "").strip()
    path = env.get(f"{key}_FILE", "").strip()
    if not value and path:
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"{key}_FILE cannot be read: {exc}") from exc
    return value


def _max_calls(env: Mapping[str, str]) -> int:
    """``VA_MAX_CALLS``: a number, or ``auto`` for two concurrent calls per CPU available to the process."""
    if env.get("VA_MAX_CALLS", "").strip().lower() == "auto":
        try:
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:  # not Linux
            cpus = os.cpu_count() or 1
        return max(1, 2 * cpus)
    return _int(env, "VA_MAX_CALLS", 4, 1, 1000)


def _provider(env: Mapping[str, str], prefix: str, name: str, required: bool) -> LlmProvider | None:
    url = env.get(f"{prefix}_URL", "").strip()
    if not url and not required:
        return None
    url = url or DEFAULT_LLM_URL
    key = _secret(env, f"{prefix}_API_KEY")
    if not key:
        raise ConfigError(f"{prefix}_API_KEY (or {prefix}_API_KEY_FILE) is required")
    if not url.startswith(("https://", "http://")):
        raise ConfigError(f"{prefix}_URL must be an http(s) URL")
    raw_extra = env.get(f"{prefix}_EXTRA_JSON", "").strip()
    if raw_extra:
        try:
            extra = json.loads(raw_extra)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{prefix}_EXTRA_JSON is not valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise ConfigError(f"{prefix}_EXTRA_JSON must be a JSON object")
    else:
        extra = dict(DEFAULT_LLM_EXTRA) if url == DEFAULT_LLM_URL else {}
    model = env.get(f"{prefix}_MODEL", "").strip() or (DEFAULT_LLM_MODEL if url == DEFAULT_LLM_URL else "")
    if not model:
        raise ConfigError(f"{prefix}_MODEL is required for a custom {prefix}_URL")
    return LlmProvider(name=name, url=url, model=model, api_key=key, extra=extra,
                       timeout=_float(env, f"{prefix}_TIMEOUT", 30.0, 1.0, 300.0),
                       stream_usage=_bool(env.get(f"{prefix}_STREAM_USAGE", "true")))


def _int(env: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer") from exc
    if not low <= value <= high:
        raise ConfigError(f"{key} must be between {low} and {high}")
    return value


def _float(env: Mapping[str, str], key: str, default: float, low: float, high: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number") from exc
    if not low <= value <= high:
        raise ConfigError(f"{key} must be between {low} and {high}")
    return value


def _bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}
