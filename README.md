# Fraud App — AI Call Fraud Detection

An Android app that listens to phone calls, transcribes them, and scores them for
fraud, plus the Python backend that does the AI work.

Three pieces in this repository:

| Path | What it is |
|---|---|
| `AI_Fraud_Call_Detection_Application/` | Android client (Kotlin, Jetpack Compose, Hilt, Room) |
| `fraud_server/` | Django REST API — `/v1/detect`, `/v1/transcribe`, `/healthz` |
| `fraud_detection_package/` | Trained Random Forest + TF-IDF artifact and its inference/training scripts |

The app works offline on its own: both AI engines have mock implementations. The
server replaces those mocks with the real models.

```
 Phone call
    │
    ▼
AudioCaptureManager ──► 15 s .mp4 chunk ──► POST /v1/transcribe ──► Whisper tiny
    │                                                              │ text
    │                                                              ▼
    └────────────── accumulated transcript ──► POST /v1/detect ──► TF-IDF + cuML RF
                                                                   │ is_fraud, risk_level
                                                                   ▼
                                                        notification + call history
```

---

## 1. What you need

- **Android**: Android Studio, JDK 17, SDK 34. A **physical device** for live call
  recording (the emulator's `MediaRecorder` cannot capture call audio).
- **Backend**: Linux x86_64, Python 3.13, an NVIDIA GPU/driver supporting CUDA 13
  (the model is a cuML GPU estimator), `ffmpeg` on `PATH`.
- **Network**: phone and laptop on the same Wi-Fi, or USB with `adb reverse`.

Reference machine: i5-12450H, 16 GB RAM, RTX 2050 4 GB, driver 580.178.04.

---

## 2. Backend quick start

```bash
cd fraud_server

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r ../requirements.txt \
    --index-strategy unsafe-best-match

cp .env.example .env            # optional — defaults work for local testing
./run.sh                        # migrate (first run) → warmup → serve on :8000
```

`run.sh` loads both models before serving. That matters: the app's OkHttp client
times out after **30 s**, and a cold cuML/Whisper start-up should not land on a
live call.

Verify:

```bash
curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool   # status: ok
./smoke_test.sh                                                # contract tests
cd .. && cd fraud_server && .venv/bin/python manage.py test detector
```

Measured on the reference machine:

| Request | Latency |
|---|---|
| `POST /v1/detect` (warm) | ~13 ms |
| `POST /v1/transcribe`, 15 s chunk (GPU) | ~0.2–1.5 s |
| `POST /v1/transcribe`, 15 s chunk (CPU fallback) | tens of seconds — too slow, see below |

Fine points that cost time if missed:

- Whisper defaults to **CUDA** here (`WHISPER_DEVICE`). On the CPU fallback, torch's
  default thread count made a 3 s clip take 353 s; the server caps it
  (`WHISPER_CPU_THREADS`) and falls back automatically if CUDA fails.
- Keep gunicorn at **one worker** — each worker loads its own copy of the model
  into VRAM.
- Full server documentation: [`fraud_server/README.md`](fraud_server/README.md).
- Model details and training: [`fraud_detection_package/README.md`](fraud_detection_package/README.md).

### On a Windows laptop

The cuML/CUDA stack is Linux-only, so on Windows 11 run the backend inside WSL2
with Ubuntu 24.04 (the NVIDIA driver must support CUDA 13 — WSL exposes the GPU
automatically). The bundled `fraud_server/.venv` is a Linux venv from another
machine and is not reusable on Windows. See
[`SETUP_WINDOWS_WSL.md`](SETUP_WINDOWS_WSL.md) for the full laptop setup, including
the `adb reverse` Android wiring.

The Android client is now bound to the remote engines with both base URLs set to
`http://127.0.0.1:8000/` for USB testing; run `adb reverse tcp:8000 tcp:8000`
before launching the app.

---

## 3. Android app

### Build

```bash
cd AI_Fraud_Call_Detection_Application
./gradlew :app:installDebug          # debug applicationId: com.fraudcalldetector.debug
```

### Point it at the server

The two URLs are `private const val` fields:

| File | Constant |
|---|---|
| `feature/frauddetection/src/main/java/com/fraudcalldetector/feature/frauddetection/di/FraudDetectionModule.kt` | `FRAUD_API_BASE_URL` |
| `feature/stt/src/main/java/com/fraudcalldetector/feature/stt/di/SttModule.kt` | `STT_BASE_URL` |

```kotlin
private const val FRAUD_API_BASE_URL = "http://192.168.1.14:8000/"
private const val STT_BASE_URL       = "http://192.168.1.14:8000/"
```

- The trailing `/` is required by Retrofit.
- Emulator → `http://10.0.2.2:8000/`. USB with `adb reverse tcp:8000 tcp:8000` →
  `http://127.0.0.1:8000/` (this survives an IP change).

### Switch the engines from mock to remote

Same two files, one word each:

```kotlin
// FraudDetectionModule.kt
abstract fun bindFraudDetectionEngine(impl: RemoteFraudDetectionEngine): FraudDetectionEngine

// SttModule.kt
abstract fun bindSttEngine(impl: RemoteSttEngine): SttEngine
```

### Allow plain HTTP in debug builds

Android 9+ blocks cleartext. Create `app/src/debug/AndroidManifest.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application android:usesCleartextTraffic="true" />
</manifest>
```

Release builds are untouched, so ship over HTTPS.

### Permissions

Runtime prompts: `READ_PHONE_STATE`, `READ_CALL_LOG`, `RECORD_AUDIO`,
`POST_NOTIFICATIONS` (API 33+), `CALL_PHONE`. `INTERNET` and the foreground-service
permissions are install-time. Outgoing calls are detected via
`CALL_STATE_OFFHOOK` + `CallLog` — `PROCESS_OUTGOING_CALLS` is intentionally absent.

---

## 4. API contract (short version)

The docs in the app repo and its code disagree in places. This server follows the
**code** and accepts the documented variants too.

### `POST /v1/detect` — JSON

```json
{
  "call_id": "abc-123",
  "transcript": "Share the OTP immediately.",
  "language": "en-US",
  "metadata": {
    "caller_number": "+15551234567",
    "duration_ms": 42000,
    "chunk_index": 2,
    "is_final": true
  }
}
```

The flat shape from the old README (`caller_number`, `duration_ms`,
`is_final_pass` at top level) is also accepted.

```json
{
  "is_fraud": true,
  "risk_score": 0.696,
  "risk_level": "MEDIUM",
  "reason": "Fraud pattern detected by the AI model (70% risk score). One-time password (OTP) request.",
  "signals": [
    { "type": "SUSPICIOUS_KEYWORDS", "description": "One-time password (OTP) request", "severity": 0.62 }
  ],
  "model_version": "rf-cuml-tfidf"
}
```

- `risk_score` is the model's fraud probability; `risk_level` bands it
  (`SAFE < 0.30`, `LOW < 0.50`, `MEDIUM < 0.70`, `HIGH < 0.90`, else `CRITICAL`).
- `signals` come from a regex lexicon, not the model — they explain *why*, and
  never change the score.
- The app calls this **once per 15 s chunk and again at call end**, because
  `FraudDetectionManagerImpl` ignores `supportsIncrementalAnalysis`.
- The app's default alert rule is `risk_score >= 0.60` **and** `risk_level >= MEDIUM`.
- Any error/timeout makes the app treat the call as `SAFE` **silently**.

### `POST /v1/transcribe` — multipart

Parts sent: `audio` (`.mp4`/`.m4a`), `language`, `call_id` — where `call_id` is
actually the filename `call_<callId>_chunk_<n>`. `chunk_index`/`is_final` are
optional and parsed from the name when absent.

Response contains **both** key spellings, because `SttResponse` in
`SttApiService.kt` has no `@SerializedName` and Gson matches literal Kotlin names:

```json
{
  "callId": "abc", "call_id": "abc",
  "fullText": "…", "full_text": "…",
  "durationSeconds": 3.0, "duration_seconds": 3.0,
  "confidence": 0.91, "segments": [ { "id": "seg-001", "text": "…",
    "speaker": "unknown", "startMs": 0, "endMs": 3200, "confidence": 0.91 } ],
  "words": [ … ]
}
```

Following only the snake_case docs would make the app's `fullText` null and break
every transcription.

---

## 5. Configuration

All server settings are environment variables; see
[`fraud_server/.env.example`](fraud_server/.env.example).

| Variable | Default | Note |
|---|---|---|
| `FRAUD_API_KEY` | *(empty)* | Empty = open API. Set to require `Bearer`/`X-Api-Key` |
| `FRAUD_DECISION_THRESHOLD` | `0.5` | `is_fraud` boundary |
| `FRAUD_BAND_*_MAX` | `0.30/0.50/0.70/0.90` | Must be strictly increasing |
| `WHISPER_MODEL` / `WHISPER_DEVICE` | `tiny` / `cuda` | CPU fallback is automatic but slow |
| `LOG_TRANSCRIPTS` | `false` | Stores call text in SQLite (PII) |
| `MAX_AUDIO_MB` | `25` | Upload cap |

Request logs land in `fraud_server/db.sqlite3` (`DetectionLog`,
`TranscriptionLog`) and in `/admin/` after `manage.py createsuperuser`.

---

## 6. Remaining work

See [`final_touches.md`](final_touches.md) for the checklist: app wiring, auth,
TLS/reverse proxy, production service setup, calibration, and known gaps.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Phone can't reach the server | `curl http://<lan-ip>:8000/healthz` from the laptop first; check `DJANGO_ALLOWED_HOSTS=*`, the bind address (`0.0.0.0`), and `sudo ufw allow 8000/tcp` |
| App shows nothing / every call SAFE | Engines may still be bound to the mocks, or cleartext is blocked, or the server returned a 4xx/5xx (the app hides it) |
| `/healthz` stays 503 | Models still loading; run `manage.py warmup` and read the server log |
| `No module named 'cuml'` | Wrong interpreter — always `fraud_server/.venv/bin/python` |
| `cuml-cu13==0.0.0a0` | Installed without the NVIDIA index; reinstall using `../requirements.txt` |
| Whisper timeouts | `WHISPER_DEVICE=cuda`; on CPU lower `WHISPER_BEAM_SIZE` and keep `WHISPER_CPU_THREADS=2` |
| APK installs but old code runs | Debug id is `com.fraudcalldetector.debug`; uninstall the stale package or `./gradlew :app:installDebug` again |

---

## 8. Limitations

- The 98% accuracy figure is measured on clean **text** transcripts. Whisper `tiny`
  on phone audio is much weaker; treat the result as a screening signal.
- No speaker diarisation: `speaker` is always `unknown`.
- No retry on the app side: a dropped request is a missed analysis window, and
  failures look identical to "safe".
- SQLite + one gunicorn worker is the intended single-host deployment.
