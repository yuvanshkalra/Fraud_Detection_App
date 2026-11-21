from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from rest_framework import status
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from . import conf
from .models import DetectionLog, TranscriptionLog
from .serializers import FraudDetectionRequestSerializer
from .services import fraud, stt

logger = logging.getLogger(__name__)

STARTED_AT = time.time()

# call_<callId>_chunk_<n>.<ext>  ← AudioCaptureManager.segmentPath()
_AUDIO_NAME_RE = re.compile(r"^call_(?P<call_id>.+?)_chunk_(?P<chunk_index>\d+)$")
_ALLOWED_AUDIO_SUFFIXES = {
    ".mp4",
    ".m4a",
    ".aac",
    ".3gp",
    ".amr",
    ".wav",
    ".mp3",
    ".ogg",
    ".opus",
    ".webm",
    ".flac",
}


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: object, default: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _split_call_id(raw: str) -> tuple[str, int | None]:
    """Turn `call_<uuid>_chunk_3` into (`<uuid>`, 3). Pass through otherwise."""
    match = _AUDIO_NAME_RE.match(raw or "")
    if match:
        return match.group("call_id"), int(match.group("chunk_index"))
    return raw, None


# ─────────────────────────────────────────────────────────────────────────────
# Index + health
# ─────────────────────────────────────────────────────────────────────────────
@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def index(request):
    return Response(
        {
            "service": "FraudCall Detector backend",
            "endpoints": {
                "detect": "POST /v1/detect (application/json)",
                "transcribe": "POST /v1/transcribe (multipart/form-data)",
                "health": "GET /healthz",
            },
            "docs": "see fraud_server/README.md",
        }
    )


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def healthz(request):
    model = fraud.get_model()
    whisper = stt.get_engine()
    ready = model.loaded and whisper.loaded

    payload = {
        "status": "ok" if ready else ("starting" if not (model.error or whisper.error) else "degraded"),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "fraud_model": {
            "loaded": model.loaded,
            "error": model.error,
            "load_seconds": model.load_seconds,
            "cuml_version": model.cuml_version,
            "version": model.version(),
            "artifact": str(conf.MODEL_PATH),
            "decision_threshold": conf.DECISION_THRESHOLD,
            "bands": {name: bound for name, bound in conf.BANDS},
        },
        "whisper": {
            "loaded": whisper.loaded,
            "error": whisper.error,
            "load_seconds": whisper.load_seconds,
            "model": conf.WHISPER_MODEL,
            "device": whisper.device,
        },
        "auth_required": bool(conf.API_KEY),
    }
    return Response(payload, status=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE)


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/detect
# ─────────────────────────────────────────────────────────────────────────────
class DetectView(APIView):
    def post(self, request):
        serializer = FraudDetectionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        call_id: str = data["call_id"]
        transcript: str = data["transcript"]
        chunk_index: int = data["chunk_index"]
        is_final: bool = data["is_final"]

        started = time.perf_counter()
        try:
            result = fraud.analyze(
                call_id=call_id,
                transcript=transcript,
                language=data.get("language") or "en-US",
                caller_number=data.get("caller_number"),
                duration_ms=data.get("duration_ms") or 0,
                chunk_index=chunk_index,
                is_final=is_final,
            )
        except fraud.ModelUnavailable as exc:
            logger.error("Fraud model unavailable: %s", exc)
            self._log(data, transcript, latency_ms=(time.perf_counter() - started) * 1000, error=str(exc))
            return Response(
                {"detail": f"Fraud model unavailable: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fraud inference failed for call_id=%s", call_id)
            self._log(
                data,
                transcript,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
            return Response(
                {"detail": "Fraud inference failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        self._log(data, transcript, latency_ms=latency_ms, result=result)

        logger.info(
            "detect call_id=%s chunk=%s final=%s score=%.4f level=%s chars=%d latency_ms=%.1f",
            call_id,
            chunk_index,
            is_final,
            result["risk_score"],
            result["risk_level"],
            len(transcript),
            latency_ms,
        )
        return Response(result, status=status.HTTP_200_OK)

    @staticmethod
    def _log(
        data: dict,
        transcript: str,
        *,
        latency_ms: float,
        result: dict | None = None,
        error: str = "",
    ) -> None:
        try:
            DetectionLog.objects.create(
                call_id=data["call_id"],
                chunk_index=data.get("chunk_index") or 0,
                is_final=bool(data.get("is_final")),
                language=(data.get("language") or "")[:35],
                caller_number=(data.get("caller_number") or "")[:64],
                duration_ms=data.get("duration_ms") or 0,
                transcript_chars=len(transcript or ""),
                transcript_sha256=hashlib.sha256((transcript or "").encode("utf-8")).hexdigest()
                if transcript
                else "",
                transcript=(transcript or "") if conf.LOG_TRANSCRIPTS else "",
                is_fraud=bool(result and result.get("is_fraud")),
                risk_score=float(result.get("risk_score") or 0.0) if result else 0.0,
                risk_level=(result or {}).get("risk_level", ""),
                model_version=(result or {}).get("model_version", ""),
                latency_ms=round(latency_ms, 3),
                error=error,
            )
        except Exception:  # noqa: BLE001 - logging must never break the response
            logger.exception("Failed to write DetectionLog row")


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/transcribe
# ─────────────────────────────────────────────────────────────────────────────
class TranscribeView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        upload = request.FILES.get("audio")
        if upload is None:
            return Response(
                {"detail": 'Missing multipart field "audio".'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if upload.size == 0:
            return Response(
                {"detail": "Uploaded audio is empty."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if upload.size > conf.MAX_AUDIO_BYTES:
            return Response(
                {"detail": f"Audio exceeds the {conf.MAX_AUDIO_MB} MB limit."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        audio_name = Path(upload.name or "audio.m4a").name
        raw_call_id = (request.data.get("call_id") or Path(audio_name).stem or "unknown").strip()
        call_id, parsed_chunk_index = _split_call_id(raw_call_id)

        chunk_index = _as_int(request.data.get("chunk_index"), parsed_chunk_index or 0)
        is_final = _as_bool(request.data.get("is_final"), False)
        language = (request.data.get("language") or "en-US").strip() or "en-US"

        suffix = Path(audio_name).suffix.lower()
        if suffix not in _ALLOWED_AUDIO_SUFFIXES:
            suffix = ".bin"

        handle = tempfile.NamedTemporaryFile(
            prefix="fraudcall_", suffix=suffix, delete=False
        )
        temp_path = handle.name
        started = time.perf_counter()

        try:
            for block in upload.chunks():
                handle.write(block)
            handle.close()

            payload = stt.transcribe_file(temp_path, call_id=call_id, language=language)
        except stt.SttUnavailable as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.error("STT unavailable: %s", exc)
            self._log(
                call_id=call_id,
                chunk_index=chunk_index,
                is_final=is_final,
                language=language,
                audio_name=audio_name,
                audio_bytes=upload.size,
                latency_ms=latency_ms,
                error=str(exc),
            )
            return Response(
                {"detail": f"Transcription unavailable: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000
            logger.exception("Transcription failed for call_id=%s", call_id)
            self._log(
                call_id=call_id,
                chunk_index=chunk_index,
                is_final=is_final,
                language=language,
                audio_name=audio_name,
                audio_bytes=upload.size,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
            return Response(
                {"detail": "Transcription failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        latency_ms = (time.perf_counter() - started) * 1000
        self._log(
            call_id=call_id,
            chunk_index=chunk_index,
            is_final=is_final,
            language=language,
            audio_name=audio_name,
            audio_bytes=upload.size,
            duration_ms=payload.get("audio_duration_ms") or 0,
            confidence=payload.get("confidence") or 0.0,
            text_chars=len(payload.get("fullText") or ""),
            latency_ms=latency_ms,
        )

        logger.info(
            "transcribe call_id=%s chunk=%s bytes=%d chars=%d latency_ms=%.1f",
            call_id,
            chunk_index,
            upload.size,
            len(payload.get("fullText") or ""),
            latency_ms,
        )
        return Response(payload, status=status.HTTP_200_OK)

    @staticmethod
    def _log(
        *,
        call_id: str,
        chunk_index: int,
        is_final: bool,
        language: str,
        audio_name: str,
        audio_bytes: int,
        latency_ms: float,
        duration_ms: int = 0,
        confidence: float = 0.0,
        text_chars: int = 0,
        error: str = "",
    ) -> None:
        try:
            TranscriptionLog.objects.create(
                call_id=call_id[:200],
                chunk_index=chunk_index,
                is_final=is_final,
                language=(language or "")[:35],
                audio_name=(audio_name or "")[:255],
                audio_bytes=audio_bytes,
                audio_duration_ms=duration_ms,
                text_chars=text_chars,
                confidence=confidence,
                model=conf.WHISPER_MODEL,
                latency_ms=round(latency_ms, 3),
                error=error,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write TranscriptionLog row")
