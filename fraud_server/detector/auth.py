"""Optional API-key authentication.

The Android client's `AuthInterceptor` can send either
`Authorization: Bearer <key>` or `X-Api-Key: <key>`. Both are accepted.

When FRAUD_API_KEY is unset the API is open — intentional for local device
testing, since the shipped app currently sends no auth header at all.
"""
from __future__ import annotations

import secrets

from django.contrib.auth.models import AnonymousUser
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed, NotAuthenticated
from rest_framework.permissions import BasePermission

from . import conf


def _extract_key(request) -> str:
    header = (request.headers.get("Authorization") or "").strip()
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return (request.headers.get("X-Api-Key") or "").strip()


class ApiKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        if not conf.API_KEY:
            return None

        key = _extract_key(request)
        if not key:
            raise NotAuthenticated("API key required (Authorization: Bearer … or X-Api-Key).")
        if not secrets.compare_digest(key, conf.API_KEY):
            raise AuthenticationFailed("Invalid API key.")

        # No user accounts: a valid key is the whole identity.
        return (AnonymousUser(), key)


class HasApiKey(BasePermission):
    def has_permission(self, request, view) -> bool:
        if not conf.API_KEY:
            return True
        return request.auth is not None
