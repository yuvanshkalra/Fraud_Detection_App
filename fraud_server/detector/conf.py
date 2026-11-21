"""Central runtime configuration, read from environment / .env."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

BASE_DIR = Path(settings.BASE_DIR)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


# ── Auth ─────────────────────────────────────────────────────────────────────
# Empty string (the default) means "no auth required" — convenient for local dev.
# When set, both `Authorization: Bearer <key>` and `X-Api-Key: <key>` are accepted.
API_KEY = os.getenv("FRAUD_API_KEY", "").strip()

# ── Fraud model ──────────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = BASE_DIR.parent / "fraud_detection_package" / "fraud_rf_model.joblib"
MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH") or _DEFAULT_MODEL_PATH).expanduser()

MODEL_VERSION = os.getenv("MODEL_VERSION", "rf-cuml-tfidf").strip() or "rf-cuml-tfidf"

# Binary decision threshold on the model's fraud probability.
DECISION_THRESHOLD = _float("FRAUD_DECISION_THRESHOLD", 0.5)

# Transcripts shorter than this are classified SAFE without invoking the model.
MIN_TRANSCRIPT_CHARS = _int("FRAUD_MIN_TRANSCRIPT_CHARS", 5)

# ── Risk-level banding ───────────────────────────────────────────────────────
# PROBABILITY below the bound => that level. Everything above the last bound is
# CRITICAL. The app compares RiskLevel ordinally, so bounds must be increasing.
_DEFAULT_BANDS = [
    ("SAFE", _float("FRAUD_BAND_SAFE_MAX", 0.30)),
    ("LOW", _float("FRAUD_BAND_LOW_MAX", 0.50)),
    ("MEDIUM", _float("FRAUD_BAND_MEDIUM_MAX", 0.70)),
    ("HIGH", _float("FRAUD_BAND_HIGH_MAX", 0.90)),
]


def _valid_bands(bands: list[tuple[str, float]]) -> bool:
    values = [bound for _, bound in bands]
    return (
        all(0.0 < value < 1.0 for value in values)
        and all(a < b for a, b in zip(values, values[1:]))
    )


BANDS = _DEFAULT_BANDS
if not _valid_bands(BANDS):
    logger.warning("Invalid FRAUD_BAND_* configuration; falling back to 0.30/0.50/0.70/0.90")
    BANDS = [("SAFE", 0.30), ("LOW", 0.50), ("MEDIUM", 0.70), ("HIGH", 0.90)]


def risk_level_for(score: float) -> str:
    """Map a 0..1 fraud probability onto the app's RiskLevel enum."""
    for name, upper_bound in BANDS:
        if score < upper_bound:
            return name
    return "CRITICAL"


# ── Signals ──────────────────────────────────────────────────────────────────
MAX_SIGNALS = _int("FRAUD_MAX_SIGNALS", 5)

# ── STT (Whisper) ────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny").strip() or "tiny"
# "cuda" is the default because CPU inference on this class of machine is too
# slow for the app's 30 s request timeout (a 15 s clip takes a few seconds on GPU
# vs tens of seconds on CPU). If CUDA is unavailable the server falls back to CPU
# automatically.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda").strip() or "cuda"
WHISPER_DOWNLOAD_ROOT = Path(
    os.getenv("WHISPER_DOWNLOAD_ROOT") or (BASE_DIR / "models" / "whisper")
).expanduser()
WHISPER_BEAM_SIZE = _int("WHISPER_BEAM_SIZE", 1)

# Torch defaults to 2x the visible cores on some builds and then thrashes:
# measured on the reference laptop, a 3 s clip took 8.9 s with 1 thread,
# 5.5 s with 2, 11.5 s with 4 and 353 s with 12. Only used on the CPU fallback.
# 0 = auto (capped at 2).
WHISPER_CPU_THREADS = _int("WHISPER_CPU_THREADS", 0)

MAX_AUDIO_MB = _int("MAX_AUDIO_MB", 25)
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024

# ── Misc ─────────────────────────────────────────────────────────────────────
# Store raw transcript text in the DB. Off by default (PII).
LOG_TRANSCRIPTS = _bool("LOG_TRANSCRIPTS", False)

# Delete DetectionLog/TranscriptionLog rows older than this many days via
# `manage.py prune_logs`. 0 disables retention pruning.
LOG_RETENTION_DAYS = _int("LOG_RETENTION_DAYS", 30)

# Load both models in a background thread when the process starts.
PRELOAD_MODELS = _bool("PRELOAD_MODELS", True)
