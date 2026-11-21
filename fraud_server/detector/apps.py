from __future__ import annotations

import logging
import os
import sys
import threading

from django.apps import AppConfig

logger = logging.getLogger(__name__)

# Management commands that must not spin up models.
_SKIP_PRELOAD_COMMANDS = {
    "migrate",
    "makemigrations",
    "showmigrations",
    "collectstatic",
    "test",
    "shell",
    "warmup",
    "prune_logs",
    "dumpdata",
    "loaddata",
}


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self) -> None:
        from . import conf

        if not conf.PRELOAD_MODELS:
            return

        command = sys.argv[1] if len(sys.argv) > 1 else ""
        if command in _SKIP_PRELOAD_COMMANDS:
            return

        # `runserver` with autoreload imports settings in the parent watcher
        # too; only the child (RUN_MAIN=true) should hold GPU memory. With
        # --noreload there is no child, so preload in this process.
        autoreload_active = command == "runserver" and "--noreload" not in sys.argv
        if autoreload_active and os.environ.get("RUN_MAIN") != "true":
            return

        threading.Thread(
            target=self._preload,
            name="detector-model-preload",
            daemon=True,
        ).start()

    @staticmethod
    def _preload() -> None:
        from .services import fraud, stt

        for name, loader in (
            ("fraud model", fraud.get_model().load),
            ("whisper", stt.get_engine().load),
        ):
            try:
                loader()
            except Exception:  # noqa: BLE001 - startup must not crash the server
                logger.exception("Failed to preload %s", name)
