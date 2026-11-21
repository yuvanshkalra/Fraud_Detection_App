"""Request parsing.

`/v1/detect` accepts BOTH shapes that appear in the project docs:

  # documented in INTEGRATION_GUIDE.md (what the shipped app actually sends)
  { "call_id", "transcript", "language",
    "metadata": { "caller_number", "duration_ms", "chunk_index", "is_final" } }

  # documented in README.md (older flat shape — still accepted)
  { "call_id", "transcript", "caller_number", "duration_ms", "is_final_pass" }
"""
from __future__ import annotations

from rest_framework import serializers


class FraudMetadataSerializer(serializers.Serializer):
    caller_number = serializers.CharField(
        required=False, allow_null=True, allow_blank=True, max_length=64
    )
    duration_ms = serializers.IntegerField(required=False, min_value=0)
    chunk_index = serializers.IntegerField(required=False, min_value=0)
    is_final = serializers.BooleanField(required=False)
    is_final_pass = serializers.BooleanField(required=False)


class FraudDetectionRequestSerializer(serializers.Serializer):
    call_id = serializers.CharField(max_length=200, trim_whitespace=True)
    transcript = serializers.CharField(
        required=False, allow_blank=True, default="", trim_whitespace=False
    )
    # Some clients use "text" instead of "transcript"; accept it as an alias.
    text = serializers.CharField(
        required=False, allow_blank=True, trim_whitespace=False, write_only=True
    )
    language = serializers.CharField(required=False, allow_blank=True, max_length=35)

    metadata = FraudMetadataSerializer(required=False)

    # Flat fallbacks (README.md contract).
    caller_number = serializers.CharField(
        required=False, allow_null=True, allow_blank=True, max_length=64
    )
    duration_ms = serializers.IntegerField(required=False, min_value=0)
    chunk_index = serializers.IntegerField(required=False, min_value=0)
    is_final = serializers.BooleanField(required=False)
    is_final_pass = serializers.BooleanField(required=False)

    def validate_call_id(self, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise serializers.ValidationError("call_id must not be empty.")
        return value

    def validate(self, attrs: dict) -> dict:
        metadata = attrs.pop("metadata", None) or {}

        transcript = attrs.get("transcript")
        if transcript is None or transcript == "":
            transcript = attrs.get("text") or ""
        attrs.pop("text", None)
        attrs["transcript"] = transcript

        def pick(meta_key: str) -> object | None:
            if meta_key in metadata and metadata[meta_key] is not None:
                return metadata[meta_key]
            if meta_key in attrs and attrs[meta_key] is not None:
                return attrs[meta_key]
            return None

        def pick_final() -> bool:
            for key in ("is_final", "is_final_pass"):
                value = metadata.get(key)
                if value is None:
                    value = attrs.get(key)
                if value is not None:
                    return bool(value)
            return True

        caller_number = pick("caller_number")
        duration_ms = pick("duration_ms")
        chunk_index = pick("chunk_index")

        attrs["caller_number"] = (str(caller_number).strip() if caller_number else "") or None
        attrs["duration_ms"] = int(duration_ms or 0)
        attrs["chunk_index"] = int(chunk_index or 0)
        attrs["is_final"] = pick_final()
        attrs["language"] = (attrs.get("language") or "en-US").strip() or "en-US"

        # Remove flat duplicates so the validated_data surface is predictable.
        for key in ("is_final_pass",):
            attrs.pop(key, None)
        return attrs
