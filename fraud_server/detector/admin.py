from django.contrib import admin

from .models import DetectionLog, TranscriptionLog


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "call_id",
        "chunk_index",
        "is_final",
        "risk_level",
        "risk_score",
        "is_fraud",
        "latency_ms",
    )
    list_filter = ("risk_level", "is_fraud", "is_final")
    search_fields = ("call_id", "caller_number", "transcript")
    readonly_fields = [field.name for field in DetectionLog._meta.fields]


@admin.register(TranscriptionLog)
class TranscriptionLogAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "call_id",
        "chunk_index",
        "audio_bytes",
        "text_chars",
        "confidence",
        "latency_ms",
    )
    list_filter = ("is_final", "model")
    search_fields = ("call_id", "audio_name")
    readonly_fields = [field.name for field in TranscriptionLog._meta.fields]
