"""Contract tests — no GPU / model loading required."""
from __future__ import annotations

from unittest import mock

from django.test import TestCase
from rest_framework.test import APIClient

from detector import conf
from detector.serializers import FraudDetectionRequestSerializer
from detector.services import keywords, stt

DETECT_RESULT = {
    "is_fraud": True,
    "risk_score": 0.91,
    "risk_level": "HIGH",
    "reason": "Fraud pattern detected by the AI model (91% risk score). OTP request.",
    "signals": [
        {"type": "SUSPICIOUS_KEYWORDS", "description": "One-time password (OTP) request", "severity": 0.62}
    ],
    "model_version": "rf-cuml-tfidf",
}


class DetectRequestSerializerTests(TestCase):
    def test_nested_metadata_shape(self):
        serializer = FraudDetectionRequestSerializer(
            data={
                "call_id": "call-1",
                "transcript": "share the otp",
                "language": "en-US",
                "metadata": {
                    "caller_number": "+15551234567",
                    "duration_ms": 42000,
                    "chunk_index": 2,
                    "is_final": False,
                },
            }
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        data = serializer.validated_data
        self.assertEqual(data["call_id"], "call-1")
        self.assertEqual(data["language"], "en-US")
        self.assertEqual(data["caller_number"], "+15551234567")
        self.assertEqual(data["duration_ms"], 42000)
        self.assertEqual(data["chunk_index"], 2)
        self.assertFalse(data["is_final"])

    def test_flat_legacy_shape(self):
        serializer = FraudDetectionRequestSerializer(
            data={
                "call_id": "call-2",
                "transcript": "hello",
                "caller_number": "+15550001111",
                "duration_ms": 12000,
                "is_final_pass": True,
            }
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        data = serializer.validated_data
        self.assertEqual(data["caller_number"], "+15550001111")
        self.assertTrue(data["is_final"])
        self.assertEqual(data["language"], "en-US")  # default
        self.assertEqual(data["chunk_index"], 0)

    def test_nested_metadata_absent(self):
        serializer = FraudDetectionRequestSerializer(data={"call_id": "c", "transcript": "x"})
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertTrue(serializer.validated_data["is_final"])
        self.assertIsNone(serializer.validated_data["caller_number"])

    def test_transcript_alias(self):
        serializer = FraudDetectionRequestSerializer(data={"call_id": "c", "text": "via alias"})
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["transcript"], "via alias")

    def test_call_id_required(self):
        serializer = FraudDetectionRequestSerializer(data={"transcript": "x"})
        self.assertFalse(serializer.is_valid())
        self.assertIn("call_id", serializer.errors)


class RiskBandingTests(TestCase):
    def test_bands_are_monotonic(self):
        bounds = [bound for _, bound in conf.BANDS]
        self.assertEqual(bounds, sorted(bounds))
        self.assertEqual(len(bounds), len(set(bounds)))

    def test_banding_matches_app_alert_threshold(self):
        # The app's DEFAULT config alerts at risk_score >= 0.60 AND level >= MEDIUM,
        # so 0.60 must already be at least MEDIUM for in-call alerts to fire.
        order = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        self.assertEqual(conf.risk_level_for(0.0), "SAFE")
        self.assertGreaterEqual(order.index(conf.risk_level_for(0.60)), order.index("MEDIUM"))
        self.assertEqual(conf.risk_level_for(0.99), "CRITICAL")
        scores = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
        indices = [order.index(conf.risk_level_for(score)) for score in scores]
        self.assertEqual(indices, sorted(indices))


class KeywordSignalTests(TestCase):
    def test_otp_scam_produces_signals(self):
        signals = keywords.detect_signals(
            "Your card is blocked. Share the OTP immediately or your account will be suspended."
        )
        descriptions = [signal["description"] for signal in signals]
        self.assertIn("One-time password (OTP) request", descriptions)
        self.assertLessEqual(len(signals), conf.MAX_SIGNALS)
        for signal in signals:
            self.assertIn(
                signal["type"],
                {
                    "SUSPICIOUS_KEYWORDS",
                    "AI_DETECTED_MANIPULATION",
                    "UNUSUAL_PATTERNS",
                    "KNOWN_SCAM_NUMBER",
                    "RAPID_SPEECH",
                    "VOICE_SPOOFING",
                },
            )
            self.assertGreaterEqual(signal["severity"], 0.0)
            self.assertLessEqual(signal["severity"], 1.0)

    def test_benign_text_has_no_signals(self):
        self.assertEqual(
            keywords.detect_signals("Hi, this is your pharmacy calling about a refill."),
            [],
        )

    def test_reason_never_blank(self):
        for level in ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"):
            reason = keywords.build_reason(level, level in {"HIGH", "CRITICAL"}, 0.75, [])
            self.assertTrue(reason.strip())


class SttPayloadTests(TestCase):
    def test_dual_spelling_and_camel_case_fields(self):
        fake_result = {
            "text": " Please share the OTP. ",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.2,
                    "text": " Please share the OTP.",
                    "avg_logprob": -0.1,
                    "words": [
                        {"word": " Please", "start": 0.0, "end": 0.4, "probability": 0.99},
                        {"word": " OTP.", "start": 3.0, "end": 3.2, "probability": 0.95},
                    ],
                }
            ],
        }
        payload = stt.build_payload(fake_result, call_id="abc", requested_language="en-US")

        # The Android DTO reads exactly these camelCase keys.
        for key in ("callId", "fullText", "segments", "language", "durationSeconds"):
            self.assertIsNotNone(payload.get(key), key)
        # Documented snake_case aliases must also be present.
        for key in ("call_id", "full_text", "duration_seconds"):
            self.assertIsNotNone(payload.get(key), key)

        self.assertEqual(payload["fullText"], "Please share the OTP.")
        self.assertEqual(payload["durationSeconds"], 3.2)
        self.assertEqual(payload["language"], "en-US")
        self.assertEqual(payload["detected_language"], "en")

        segment = payload["segments"][0]
        self.assertEqual(segment["id"], "seg-001")
        self.assertEqual(segment["speaker"], "unknown")
        self.assertEqual(segment["startMs"], 0)
        self.assertEqual(segment["endMs"], 3200)
        self.assertIn("confidence", segment)

        self.assertEqual(len(payload["words"]), 2)
        self.assertEqual(payload["words"][1]["endMs"], 3200)

    def test_empty_result_is_still_well_formed(self):
        payload = stt.build_payload(
            {"text": "", "segments": []}, call_id="abc", requested_language="en-US"
        )
        self.assertEqual(payload["fullText"], "")
        self.assertEqual(payload["segments"], [])
        self.assertEqual(payload["durationSeconds"], 0.0)
        self.assertEqual(payload["confidence"], 0.0)

    def test_language_normalisation(self):
        self.assertEqual(stt.normalize_language("en-US"), "en")
        self.assertEqual(stt.normalize_language("EN_us"), "en")
        self.assertIsNone(stt.normalize_language("auto"))
        self.assertIsNone(stt.normalize_language(""))


@mock.patch("detector.views.fraud.analyze", return_value=DETECT_RESULT)
class DetectEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_nested_shape(self, _analyze):
        response = self.client.post(
            "/v1/detect",
            {
                "call_id": "call-1",
                "transcript": "share the otp",
                "metadata": {"chunk_index": 1, "is_final": False},
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), DETECT_RESULT)

    def test_trailing_slash(self, _analyze):
        response = self.client.post(
            "/v1/detect/", {"call_id": "c", "transcript": "hi"}, format="json"
        )
        self.assertEqual(response.status_code, 200)

    def test_missing_call_id_is_400(self, _analyze):
        response = self.client.post("/v1/detect", {"transcript": "hi"}, format="json")
        self.assertEqual(response.status_code, 400)

    def test_model_failure_is_503(self, analyze):
        from detector.services.fraud import ModelUnavailable

        analyze.side_effect = ModelUnavailable("boom")
        response = self.client.post("/v1/detect", {"call_id": "c", "transcript": "hi"}, format="json")
        self.assertEqual(response.status_code, 503)


class HealthEndpointTests(TestCase):
    def test_health_reports_not_ready(self):
        response = self.client.get("/healthz")
        self.assertIn(response.status_code, (200, 503))
        body = response.json()
        self.assertIn("fraud_model", body)
        self.assertIn("whisper", body)


@mock.patch("detector.auth.conf.API_KEY", "s3cret")
class AuthTests(TestCase):
    """FRAUD_API_KEY empty => open; set => Bearer or X-Api-Key required."""

    def setUp(self):
        self.client = APIClient()
        self.payload = {"call_id": "c", "transcript": "hello"}
        patcher = mock.patch("detector.views.fraud.analyze", return_value=DETECT_RESULT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_open_when_no_key_configured(self):
        with mock.patch("detector.auth.conf.API_KEY", ""):
            response = self.client.post("/v1/detect", self.payload, format="json")
        self.assertEqual(response.status_code, 200)

    def test_missing_key_rejected(self):
        response = self.client.post("/v1/detect", self.payload, format="json")
        self.assertIn(response.status_code, (401, 403))

    def test_wrong_key_rejected(self):
        response = self.client.post(
            "/v1/detect", self.payload, format="json", HTTP_AUTHORIZATION="Bearer nope"
        )
        self.assertIn(response.status_code, (401, 403))

    def test_bearer_key_accepted(self):
        response = self.client.post(
            "/v1/detect", self.payload, format="json", HTTP_AUTHORIZATION="Bearer s3cret"
        )
        self.assertEqual(response.status_code, 200)

    def test_x_api_key_accepted(self):
        response = self.client.post("/v1/detect", self.payload, format="json", HTTP_X_API_KEY="s3cret")
        self.assertEqual(response.status_code, 200)


