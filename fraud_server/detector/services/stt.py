"""Speech-to-text via openai-whisper ("tiny" by default).

The Android client actually reads camelCase field names from the response
despite the docs advertising snake_case, so `build_payload` emits a superset
containing both spellings. Gson ignores keys it does not know, which makes the
same payload valid for the documented contract and the shipped client.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

from .. import conf

logger = logging.getLogger(__name__)

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$", re.I)


class SttUnavailable(RuntimeError):
    """Whisper could not be loaded or transcription failed hard."""


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize_language(value: str | None) -> str | None:
    """`"en-US"` → `"en"`. Returns None to let Whisper auto-detect."""
    if not value:
        return None
    primary = value.strip().replace("_", "-").split("-")[0].strip().lower()
    if not primary or primary in {"auto", "und", "unknown"}:
        return None
    return primary if _LANGUAGE_RE.match(primary) else None


def ms(seconds: float | int | None) -> int:
    return int(round(float(seconds or 0.0) * 1000))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _segment_confidence(segment: dict[str, Any]) -> float:
    """Whisper gives avg_logprob; exp() is a reasonable 0..1 proxy."""
    avg_logprob = segment.get("avg_logprob")
    try:
        return round(_clip(math.exp(float(avg_logprob))), 4)
    except (TypeError, ValueError):
        return 0.0


# ── Engine ───────────────────────────────────────────────────────────────────
class WhisperEngine:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._device = conf.WHISPER_DEVICE

        self.loaded = False
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.loaded_at: float | None = None

    @property
    def device(self) -> str:
        return self._device

    def load(self):
        if self.loaded and self._model is not None:
            return self._model

        with self._lock:
            if self.loaded and self._model is not None:
                return self._model

            conf.WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            logger.info(
                "Loading Whisper model %r on %s (download root %s)",
                conf.WHISPER_MODEL,
                self._device,
                conf.WHISPER_DOWNLOAD_ROOT,
            )
            try:
                import whisper

                self._configure_torch_threads()

                try:
                    model = whisper.load_model(
                        conf.WHISPER_MODEL,
                        device=self._device,
                        download_root=str(conf.WHISPER_DOWNLOAD_ROOT),
                    )
                except Exception as exc:  # noqa: BLE001
                    if not self._device.lower().startswith("cuda"):
                        raise
                    logger.warning("CUDA load failed (%s); falling back to CPU", exc)
                    self._device = "cpu"
                    model = whisper.load_model(
                        conf.WHISPER_MODEL,
                        device="cpu",
                        download_root=str(conf.WHISPER_DOWNLOAD_ROOT),
                    )
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                raise SttUnavailable(self.error) from exc

            self._model = model
            self.loaded = True
            self.error = None
            self.load_seconds = round(time.perf_counter() - started, 3)
            self.loaded_at = time.time()
            logger.info("Whisper ready on %s in %.2fs", self._device, self.load_seconds)
            self._warm_up(model)
            return model

    @staticmethod
    def _configure_torch_threads() -> None:
        import os

        import torch

        try:
            visible = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            visible = os.cpu_count() or 4

        # CPU inference degrades sharply past 2 threads on the reference
        # machine; keep the fallback conservative. Override with
        # WHISPER_CPU_THREADS on a machine with real headroom.
        requested = conf.WHISPER_CPU_THREADS or min(visible, 2)
        threads = max(1, min(requested, visible))
        torch.set_num_threads(threads)
        logger.info("Torch CPU threads set to %d (visible cores: %d)", threads, visible)

    def _warm_up(self, model) -> None:
        """Pay torch/kernel init + numba DTW compilation on an inaudible clip."""
        try:
            import numpy as np

            started = time.perf_counter()
            model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                language="en",
                beam_size=conf.WHISPER_BEAM_SIZE,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
                fp16=(self._device == "cuda"),
                verbose=False,
            )
            logger.info("Whisper warm-up inference finished in %.2fs", time.perf_counter() - started)
        except Exception:  # noqa: BLE001 - warm-up is best effort
            logger.warning("Whisper warm-up inference failed", exc_info=True)

    def transcribe(self, audio_path: str, language: str | None) -> dict[str, Any]:
        model = self.load()
        whisper_language = normalize_language(language)

        started = time.perf_counter()
        with self._lock:
            result = model.transcribe(
                audio_path,
                language=whisper_language,
                task="transcribe",
                beam_size=conf.WHISPER_BEAM_SIZE,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
                fp16=(self._device == "cuda"),
                verbose=False,
            )
        logger.info(
            "Whisper transcribed %s in %.2fs (%d segments)",
            Path(audio_path).name,
            time.perf_counter() - started,
            len(result.get("segments") or []),
        )
        return result


_engine = WhisperEngine()


def get_engine() -> WhisperEngine:
    return _engine


# ── Response mapping ─────────────────────────────────────────────────────────
def build_payload(result: dict[str, Any], *, call_id: str, requested_language: str | None) -> dict:
    raw_segments = result.get("segments") or []

    segments: list[dict] = []
    words: list[dict] = []

    for index, raw in enumerate(raw_segments):
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or start)
        segment_confidence = _segment_confidence(raw)
        text = (raw.get("text") or "").strip()

        segments.append(
            {
                "id": f"seg-{index + 1:03d}",
                "text": text,
                "speaker": "unknown",
                "startMs": ms(start),
                "endMs": ms(end),
                "start_ms": ms(start),
                "end_ms": ms(end),
                "confidence": segment_confidence,
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

        for raw_word in raw.get("words") or []:
            word_start = float(raw_word.get("start") or start)
            word_end = float(raw_word.get("end") or word_start)
            try:
                word_confidence = round(_clip(float(raw_word.get("probability"))), 4)
            except (TypeError, ValueError):
                word_confidence = segment_confidence

            words.append(
                {
                    "word": (raw_word.get("word") or "").strip(),
                    "startMs": ms(word_start),
                    "endMs": ms(word_end),
                    "start_ms": ms(word_start),
                    "end_ms": ms(word_end),
                    "confidence": word_confidence,
                    "speaker": "unknown",
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                }
            )

    full_text = (result.get("text") or "").strip()
    if not full_text:
        full_text = " ".join(segment["text"] for segment in segments).strip()

    if words:
        overall_confidence = round(
            _clip(sum(word["confidence"] for word in words) / len(words)), 4
        )
    elif segments:
        overall_confidence = round(
            _clip(sum(segment["confidence"] for segment in segments) / len(segments)), 4
        )
    else:
        overall_confidence = 0.0

    duration_seconds = round(segments[-1]["end"], 3) if segments else 0.0
    detected_language = (result.get("language") or "").strip() or (
        normalize_language(requested_language) or "en"
    )
    language = (requested_language or "").strip() or detected_language

    # NOTE: both key spellings are intentional — see the module docstring.
    return {
        "callId": call_id,
        "call_id": call_id,
        "fullText": full_text,
        "full_text": full_text,
        "text": full_text,
        "language": language,
        "detected_language": detected_language,
        "detectedLanguage": detected_language,
        "durationSeconds": duration_seconds,
        "duration_seconds": duration_seconds,
        "audio_duration_ms": ms(duration_seconds),
        "confidence": overall_confidence,
        "overall_confidence": overall_confidence,
        "overallConfidence": overall_confidence,
        "segments": segments,
        "words": words,
    }


def transcribe_file(audio_path: str, *, call_id: str, language: str | None) -> dict:
    try:
        result = get_engine().transcribe(audio_path, language)
    except SttUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Whisper transcription failed for %s", audio_path)
        raise SttUnavailable(f"{type(exc).__name__}: {exc}") from exc

    return build_payload(result, call_id=call_id, requested_language=language)
