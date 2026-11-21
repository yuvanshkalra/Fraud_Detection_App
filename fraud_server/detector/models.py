from __future__ import annotations

from django.db import models


class DetectionLog(models.Model):
    """One `/v1/detect` request. Useful for tuning and for tracing alerts."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    call_id = models.CharField(max_length=200, db_index=True)
    chunk_index = models.IntegerField(default=0)
    is_final = models.BooleanField(default=False)
    language = models.CharField(max_length=35, blank=True, default="")

    caller_number = models.CharField(max_length=64, blank=True, default="")
    duration_ms = models.BigIntegerField(default=0)

    transcript_chars = models.IntegerField(default=0)
    transcript_sha256 = models.CharField(max_length=64, blank=True, default="")
    transcript = models.TextField(blank=True, default="")

    is_fraud = models.BooleanField(default=False)
    risk_score = models.FloatField(default=0.0)
    risk_level = models.CharField(max_length=16, blank=True, default="")

    model_version = models.CharField(max_length=100, blank=True, default="")
    latency_ms = models.FloatField(default=0.0)
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["call_id", "chunk_index"]),
            models.Index(fields=["risk_level"]),
        ]

    def __str__(self) -> str:
        return f"{self.call_id} #{self.chunk_index} {self.risk_level} ({self.risk_score:.3f})"


class TranscriptionLog(models.Model):
    """One `/v1/transcribe` request."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    call_id = models.CharField(max_length=200, db_index=True)
    chunk_index = models.IntegerField(default=0)
    is_final = models.BooleanField(default=False)
    language = models.CharField(max_length=35, blank=True, default="")

    audio_name = models.CharField(max_length=255, blank=True, default="")
    audio_bytes = models.BigIntegerField(default=0)
    audio_duration_ms = models.BigIntegerField(default=0)

    text_chars = models.IntegerField(default=0)
    confidence = models.FloatField(default=0.0)

    model = models.CharField(max_length=50, blank=True, default="")
    latency_ms = models.FloatField(default=0.0)
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["call_id", "chunk_index"])]

    def __str__(self) -> str:
        return f"{self.call_id} #{self.chunk_index} ({self.audio_bytes} bytes, {self.latency_ms:.0f} ms)"
