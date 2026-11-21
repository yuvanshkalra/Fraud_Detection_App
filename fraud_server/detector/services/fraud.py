"""Fraud inference: TF-IDF (word + char) → cuML Random Forest.

The artifact is loaded once per process and guarded by a lock, because cuML
estimators are not documented as thread-safe.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from scipy.sparse import hstack

from .. import conf
from . import keywords

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """The joblib artifact could not be loaded."""


class FraudModel:
    def __init__(self) -> None:
        self._artifact: dict[str, Any] | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

        self.loaded = False
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.loaded_at: float | None = None
        self.cuml_version: str | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def load(self) -> dict[str, Any]:
        if self.loaded and self._artifact is not None:
            return self._artifact

        with self._load_lock:
            if self.loaded and self._artifact is not None:
                return self._artifact

            path = conf.MODEL_PATH
            if not path.exists():
                raise ModelUnavailable(f"Model artifact not found at {path}")

            started = time.perf_counter()
            logger.info("Loading fraud model from %s", path)
            try:
                import joblib

                artifact = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                raise ModelUnavailable(self.error) from exc

            missing = [key for key in ("word_vectorizer", "char_vectorizer", "model") if key not in artifact]
            if missing:
                self.error = f"Artifact is missing keys: {', '.join(missing)}"
                raise ModelUnavailable(self.error)

            self._artifact = artifact
            self.cuml_version = str(artifact.get("cuml_version") or "").strip() or None
            self.load_seconds = round(time.perf_counter() - started, 3)
            self.loaded_at = time.time()
            self.loaded = True
            self.error = None

            logger.info("Fraud model ready in %.2fs (cuML %s)", self.load_seconds, self.cuml_version or "?")
            self._warm_up(artifact)
            return artifact

    def _warm_up(self, artifact: dict[str, Any]) -> None:
        """Initialise cuML kernels so the first real call is not 20x slower."""
        started = time.perf_counter()
        try:
            with self._predict_lock:
                matrix = self._matrix(artifact, "warm up the cuML random forest")
                np.asarray(artifact["model"].predict_proba(matrix))
            logger.info("cuML warm-up inference finished in %.2fs", time.perf_counter() - started)
        except Exception:  # noqa: BLE001 - warm-up is best effort
            logger.warning("cuML warm-up inference failed", exc_info=True)

    @staticmethod
    def _matrix(artifact: dict[str, Any], text: str):
        return hstack(
            [
                artifact["word_vectorizer"].transform([text]),
                artifact["char_vectorizer"].transform([text]),
            ]
        ).astype(np.float32).toarray()

    # ── Inference ───────────────────────────────────────────────────────────
    def predict_probability(self, transcript: str) -> float:
        text = (transcript or "").strip()
        if len(text) < conf.MIN_TRANSCRIPT_CHARS:
            return 0.0

        artifact = self.load()

        # Build exactly what the training pipeline used: word + char TF-IDF,
        # hstacked and densified (cuML requires dense input).
        with self._predict_lock:
            probabilities = np.asarray(
                artifact["model"].predict_proba(self._matrix(artifact, text))
            )[:, 1]

        return float(probabilities[0])

    def version(self) -> str:
        if conf.MODEL_VERSION:
            return conf.MODEL_VERSION
        return "rf-cuml-tfidf"


_model = FraudModel()


def get_model() -> FraudModel:
    return _model


def analyze(
    *,
    call_id: str,
    transcript: str,
    language: str = "en-US",
    caller_number: str | None = None,
    duration_ms: int = 0,
    chunk_index: int = 0,
    is_final: bool = True,
) -> dict:
    """Run one detection pass and build the app-facing response object."""
    score = round(get_model().predict_probability(transcript), 6)
    risk_level = conf.risk_level_for(score)
    is_fraud = score >= conf.DECISION_THRESHOLD

    signals = keywords.detect_signals(transcript)
    reason = keywords.build_reason(risk_level, is_fraud, score, signals)

    return {
        "is_fraud": is_fraud,
        "risk_score": score,
        "risk_level": risk_level,
        "reason": reason,
        "signals": signals,
        "model_version": get_model().version(),
    }
