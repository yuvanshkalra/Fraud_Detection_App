#!/usr/bin/env bash
# End-to-end contract check against a running server.
#
#   ./smoke_test.sh                       # http://127.0.0.1:8000
#   BASE_URL=http://192.168.1.20:8000 ./smoke_test.sh
#
# Verifies:
#   1. GET  /healthz          — both models loaded
#   2. POST /v1/detect        — nested `metadata` shape (what the app sends)
#   3. POST /v1/detect        — flat legacy shape (README.md)
#   4. POST /v1/transcribe    — multipart, and that the camelCase response
#                               fields the Android client actually reads exist
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
AUTH_HEADER=()
if [[ -n "${FRAUD_API_KEY:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${FRAUD_API_KEY}")
fi

PY="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "health (waiting up to 120 s for models)"
HEALTH=""
for _ in $(seq 1 60); do
  HEALTH=$(curl -sS "${AUTH_HEADER[@]}" "$BASE_URL/healthz" || true)
  if printf '%s' "$HEALTH" | grep -q '"status": *"ok"'; then
    break
  fi
  sleep 2
done
printf '%s' "$HEALTH" | "$PY" -m json.tool
if ! printf '%s' "$HEALTH" | grep -q '"status": *"ok"'; then
  echo "server is not ready (run: .venv/bin/python manage.py warmup)" >&2
  exit 1
fi

say "detect (nested metadata — shipped app shape)"
DETECT_NESTED=$(curl -fsS -X POST "$BASE_URL/v1/detect" \
  "${AUTH_HEADER[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
        "call_id": "smoke-nested-1",
        "transcript": "This is your bank. Your card is blocked. Share the OTP immediately or your account will be suspended.",
        "language": "en-US",
        "metadata": {
          "caller_number": "+15551234567",
          "duration_ms": 42000,
          "chunk_index": 2,
          "is_final": true
        }
      }')
echo "$DETECT_NESTED" | "$PY" -m json.tool

say "detect (flat legacy shape)"
DETECT_FLAT=$(curl -fsS -X POST "$BASE_URL/v1/detect" \
  "${AUTH_HEADER[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
        "call_id": "smoke-flat-1",
        "transcript": "Hi, this is your pharmacy calling about a prescription refill.",
        "caller_number": "+15550001111",
        "duration_ms": 12000,
        "is_final_pass": true
      }')
echo "$DETECT_FLAT" | "$PY" -m json.tool

say "detect contract assertions"
DETECT_NESTED="$DETECT_NESTED" DETECT_FLAT="$DETECT_FLAT" "$PY" - <<'PYCODE'
import json, os, sys

nested = json.loads(os.environ["DETECT_NESTED"])
flat = json.loads(os.environ["DETECT_FLAT"])

LEVELS = {"SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
TYPES = {
    "SUSPICIOUS_KEYWORDS", "AI_DETECTED_MANIPULATION", "UNUSUAL_PATTERNS",
    "KNOWN_SCAM_NUMBER", "RAPID_SPEECH", "VOICE_SPOOFING",
}

problems = []
for name, body in (("nested", nested), ("flat", flat)):
    for field in ("is_fraud", "risk_score", "risk_level", "reason"):
        if field not in body or body[field] is None:
            problems.append(f"{name}: missing required field {field!r}")
    if body.get("risk_level") not in LEVELS:
        problems.append(f"{name}: bad risk_level {body.get('risk_level')!r}")
    if not isinstance(body.get("risk_score"), (int, float)) or not 0 <= body["risk_score"] <= 1:
        problems.append(f"{name}: risk_score out of range: {body.get('risk_score')!r}")
    if not str(body.get("reason") or "").strip():
        problems.append(f"{name}: reason is blank")
    for signal in body.get("signals") or []:
        if signal.get("type") not in TYPES:
            problems.append(f"{name}: unknown signal type {signal.get('type')!r}")
        for key in ("description", "severity"):
            if key not in signal:
                problems.append(f"{name}: signal missing {key!r}")

if nested.get("risk_score", 0) <= 0.5:
    problems.append(f"nested: scam transcript scored only {nested.get('risk_score')} — model may not be loaded")
if flat.get("risk_level") not in {"SAFE", "LOW"}:
    problems.append(f"flat: benign transcript classified {flat.get('risk_level')}")

if problems:
    print("FAIL")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print("OK — both request shapes produce a valid /v1/detect response")
PYCODE

say "transcribe (multipart, generated audio)"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
if command -v ffmpeg >/dev/null 2>&1; then
  # 3 s of tone. Whisper may return an empty transcript for non-speech, which is
  # fine — this test is about the shape of the response, not the words.
  ffmpeg -loglevel error -y -f lavfi -i "sine=frequency=440:duration=3" \
    -c:a aac -b:a 64k "$TMP_DIR/call_smoke-transcribe_chunk_0.mp4"
else
  echo "ffmpeg not found — writing a dummy file (server will 503/500 if it cannot decode)"
  head -c 20000 /dev/urandom > "$TMP_DIR/call_smoke-transcribe_chunk_0.mp4"
fi

TRANSCRIBE=$(curl -fsS -X POST "$BASE_URL/v1/transcribe" \
  "${AUTH_HEADER[@]}" \
  -F "audio=@$TMP_DIR/call_smoke-transcribe_chunk_0.mp4;type=audio/mp4" \
  -F "call_id=call_smoke-transcribe_chunk_0" \
  -F "language=en-US")
echo "$TRANSCRIBE" | "$PY" -m json.tool

say "transcribe contract assertions"
TRANSCRIBE="$TRANSCRIBE" "$PY" - <<'PYCODE'
import json, os, sys

body = json.loads(os.environ["TRANSCRIBE"])

# The Android DTO has NO @SerializedName and Gson uses literal field names, so
# these camelCase keys are the ones that actually matter.
required = ("callId", "fullText", "segments", "language", "durationSeconds")
missing = [key for key in required if key not in body or body[key] is None]
problems = [f"missing {key!r} (the app reads this exact key)" for key in missing]

if not isinstance(body.get("segments"), list):
    problems.append("segments must be a list")
for segment in body.get("segments") or []:
    for key in ("id", "text", "speaker", "startMs", "endMs", "confidence"):
        if segment.get(key) is None:
            problems.append(f"segment missing {key!r}")
    if str(segment.get("speaker", "")).lower() not in {"caller", "user", "unknown"}:
        problems.append(f"bad speaker {segment.get('speaker')!r}")
for key in ("full_text", "call_id", "duration_seconds"):
    if key not in body:
        problems.append(f"missing legacy alias {key!r}")

if problems:
    print("FAIL")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print("OK — /v1/transcribe returns both camelCase and snake_case keys")
PYCODE

say "logs"
curl -fsS "${AUTH_HEADER[@]}" "$BASE_URL/healthz" >/dev/null
echo "all checks passed"
