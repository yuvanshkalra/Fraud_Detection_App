"""Explainability layer for the fraud response.

The Random Forest emits a single probability — no per-signal breakdown. These
regex rules produce the `signals[]` array and a human-readable `reason` for the
notification UI. They are advisory only and never change `risk_score`.

`type` values must match the app's `FraudFlagType` enum where possible:
  SUSPICIOUS_KEYWORDS | AI_DETECTED_MANIPULATION | UNUSUAL_PATTERNS
  KNOWN_SCAM_NUMBER   | RAPID_SPEECH | VOICE_SPOOFING
(The app falls back to SUSPICIOUS_KEYWORDS for anything else.)
"""
from __future__ import annotations

import re

from .. import conf

# (regex, description, severity, type)
_RULES: list[tuple[re.Pattern[str], str, float, str]] = [
    # ── Credential / payment harvesting ─────────────────────────────────────
    (
        re.compile(r"\b(otp|one[\s-]?time (?:password|passcode|code|pin)|verification code)\b", re.I),
        "One-time password (OTP) request",
        0.62,
        "SUSPICIOUS_KEYWORDS",
    ),
    (
        re.compile(r"\b(cvv|cvc|card number|credit card|debit card|expiry date|card details)\b", re.I),
        "Card details requested",
        0.58,
        "SUSPICIOUS_KEYWORDS",
    ),
    (
        re.compile(
            r"\b(account number|routing number|sort code|ssn|social security|"
            r"aadhaar|net ?banking|internet banking|online banking|username and password)\b",
            re.I,
        ),
        "Banking / identity credentials requested",
        0.60,
        "SUSPICIOUS_KEYWORDS",
    ),
    (
        re.compile(
            r"\b(gift card|itunes card|google play card|amazon card|steam card|"
            r"bitcoin|crypto(?:currency)?|usdt|wire transfer|western union|moneygram)\b",
            re.I,
        ),
        "Untraceable payment method mentioned",
        0.66,
        "SUSPICIOUS_KEYWORDS",
    ),
    (
        re.compile(r"\b(anydesk|teamviewer|remote access|screen shar(?:e|ing)|quick ?support)\b", re.I),
        "Remote-access tool mentioned",
        0.58,
        "SUSPICIOUS_KEYWORDS",
    ),
    # ── Social engineering ──────────────────────────────────────────────────
    (
        re.compile(
            r"\b(immediately|urgent(?:ly)?|right now|within \d+ (?:minutes|hours)|"
            r"last warning|final notice|act now|before it(?:'s| is) too late)\b",
            re.I,
        ),
        "Urgency / time pressure language",
        0.42,
        "AI_DETECTED_MANIPULATION",
    ),
    (
        re.compile(
            r"\b(arrest|police|court|legal action|law ?suit|suspend(?:ed)?|blocked|"
            r"deactivat(?:e|ed)|penalty|fine)\b",
            re.I,
        ),
        "Threat-based pressure",
        0.48,
        "AI_DETECTED_MANIPULATION",
    ),
    (
        re.compile(
            r"\b(don'?t tell|do not tell|keep this (?:confidential|secret)|"
            r"between us|do not inform|don'?t hang up|stay on the line)\b",
            re.I,
        ),
        "Secrecy / isolation tactic",
        0.52,
        "AI_DETECTED_MANIPULATION",
    ),
    (
        re.compile(
            r"\b(you(?:'ve| have)? won|lottery|jackpot|prize|free gift|"
            r"selected as (?:a )?winner|claim your reward)\b",
            re.I,
        ),
        "Unsolicited prize / lottery claim",
        0.50,
        "UNUSUAL_PATTERNS",
    ),
    (
        re.compile(r"\b(refund|reimbursement|cashback|overcharged|double charged)\b", re.I),
        "Refund pretext",
        0.34,
        "UNUSUAL_PATTERNS",
    ),
    (
        re.compile(
            r"\b(kyc|verify your (?:identity|account)|account (?:is )?(?:locked|suspended)|"
            r"confirm your details)\b",
            re.I,
        ),
        "Account-verification pretext",
        0.44,
        "SUSPICIOUS_KEYWORDS",
    ),
    (
        re.compile(r"\b(guaranteed returns?|double your (?:money|investment)|trading signals?|forex)\b", re.I),
        "Investment / get-rich-quick pitch",
        0.42,
        "UNUSUAL_PATTERNS",
    ),
]


def detect_signals(transcript: str) -> list[dict]:
    """Return up to MAX_SIGNALS distinct signals, strongest first."""
    if not transcript:
        return []

    found: list[dict] = []
    seen: set[str] = set()

    for pattern, description, severity, signal_type in _RULES:
        if description in seen:
            continue
        if pattern.search(transcript):
            seen.add(description)
            found.append(
                {
                    "type": signal_type,
                    "description": description,
                    "severity": severity,
                }
            )

    found.sort(key=lambda item: item["severity"], reverse=True)
    return found[: conf.MAX_SIGNALS]


def build_reason(risk_level: str, is_fraud: bool, score: float, signals: list[dict]) -> str:
    """A short, always non-blank sentence for the notification body."""
    top = signals[0]["description"] if signals else None
    confidence = f"{score * 100:.0f}%"

    if risk_level == "CRITICAL":
        reason = f"Very high fraud risk ({confidence} risk score)."
    elif risk_level == "HIGH" or (is_fraud and risk_level in {"MEDIUM", "HIGH", "CRITICAL"}):
        reason = f"Fraud pattern detected by the AI model ({confidence} risk score)."
    elif risk_level == "MEDIUM":
        reason = f"Some fraud indicators present ({confidence} risk score)."
    elif risk_level == "LOW":
        reason = f"Faint fraud indicators only ({confidence} risk score)."
    else:
        reason = "No strong fraud indicators detected."

    if top:
        reason = f"{reason} {top}."
    return reason
