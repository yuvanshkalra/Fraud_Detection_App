"""Normalise DRF error payloads into a shape the app can log cleanly."""
from __future__ import annotations

from rest_framework.views import exception_handler as drf_exception_handler


def api_exception_handler(exc, context):
    response = drf_exception_handler(exc, context)
    if response is None:
        return None

    # Retrofit only reports the HTTP status, but keep the body predictable for
    # curl / log inspection.
    if not isinstance(response.data, dict) or "detail" not in response.data:
        response.data = {"detail": response.data}
    return response
