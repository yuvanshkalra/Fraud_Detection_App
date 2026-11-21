"""Load both models synchronously and verify they can actually run.

    .venv/bin/python manage.py warmup

Useful as a pre-flight step before gunicorn: the app's OkHttp client times out
after 30 s, so a cold model must not eat the first request.
"""
from __future__ import annotations

import time

from django.core.management.base import BaseCommand, CommandError

from detector import conf
from detector.services import fraud, stt


class Command(BaseCommand):
    help = "Preload and smoke-test the fraud model and Whisper."

    def handle(self, *args, **options):
        failures: list[str] = []

        started = time.perf_counter()
        try:
            model = fraud.get_model()
            model.load()
            probability = model.predict_probability(
                "Your debit card is about to expire, share the OTP to renew it."
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"fraud model OK  ({conf.MODEL_PATH.name}, "
                    f"cuML {model.cuml_version or '?'}, "
                    f"sample probability={probability:.4f})"
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"fraud model: {exc}")
            self.stderr.write(self.style.ERROR(f"fraud model FAILED: {exc}"))

        try:
            engine = stt.get_engine()
            engine.load()
            self.stdout.write(
                self.style.SUCCESS(
                    f"whisper OK      (model={conf.WHISPER_MODEL}, device={engine.device})"
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"whisper: {exc}")
            self.stderr.write(self.style.ERROR(f"whisper FAILED: {exc}"))

        self.stdout.write(f"total={time.perf_counter() - started:.2f}s")

        if failures:
            raise CommandError("; ".join(failures))

        self.stdout.write(self.style.SUCCESS("All models ready."))
