"""Delete old request-log rows so SQLite does not grow forever.

Usage:
    .venv/bin/python manage.py prune_logs
    .venv/bin/python manage.py prune_logs --days 7

Retention defaults to LOG_RETENTION_DAYS (30). Set it to 0 to disable.
Intended for a cron job / systemd timer; it never touches the AI models.
"""
from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from detector import conf
from detector.models import DetectionLog, TranscriptionLog


class Command(BaseCommand):
    help = "Delete DetectionLog/TranscriptionLog rows older than LOG_RETENTION_DAYS."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=None,
            help="Override LOG_RETENTION_DAYS for this run (0 disables pruning).",
        )

    def handle(self, *args, **options):
        days = options["days"]
        if days is None:
            days = conf.LOG_RETENTION_DAYS
        if days <= 0:
            self.stdout.write("Retention disabled (days <= 0): nothing deleted.")
            return

        cutoff = timezone.now() - timedelta(days=days)
        detected, _ = DetectionLog.objects.filter(created_at__lt=cutoff).delete()
        transcribed, _ = TranscriptionLog.objects.filter(created_at__lt=cutoff).delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {detected} DetectionLog and {transcribed} TranscriptionLog "
                f"rows older than {cutoff:%Y-%m-%d %H:%M UTC}."
            )
        )
