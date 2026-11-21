# FraudCall Detector — Django backend

Backend for the `AI_Fraud_Call_Detection_Application` Android client. It serves
two endpoints that the app's Retrofit clients call:

| Endpoint | Used by | Model |
|---|---|---|
| `POST /v1/detect` | `RemoteFraudDetectionEngine` | cuML Random Forest + TF-IDF (from `../fraud_detection_package`) |
| `POST /v1/transcribe` | `RemoteSttEngine` | Whisper `tiny` (openai-whisper) |
| `GET /healthz` | ops / you | reports model readiness |

Logs go to SQLite; the API is open unless `FRAUD_API_KEY` is set. Nothing in the
Android project is modified by this server.

---

## 1. Why the contract section matters

The docs and the shipped code disagree. This server is built against **the code**,
and accepts the documented variants too.

### `/v1/detect` — request

`FraudDetectionRequestDto` uses `@SerializedName`, so the server receives the
nested shape from `INTEGRATION_GUIDE.md`:

```json
{
  "call_id": "a1b2c3d4",
  "transcript": "Please share your one-time password.",
  "language": "en-US",
  "metadata": {
    "caller_number": "+15551234567",
    "duration_ms": 42000,
    "chunk_index": 2,
    "is_final": true
  }
}
```

The flat shape in the top-level `README.md` (`caller_number`, `duration_ms`,
`is_final_pass`) is **also accepted** and merged with the same defaults.

Behaviour that follows from reading `FraudDetectionManagerImpl`:

- The manager calls `analyse()` on **every** STT chunk, even though
  `RemoteFraudDetectionEngine.supportsIncrementalAnalysis = false` (the flag is
  never consulted). Default chunk length is **15 s**, so expect a request roughly
  every 15 s plus one final pass.
- `risk_score` becomes `FraudAnalysis.confidence`; the app's default alert rule
  is `confidence >= 0.60` **and** `riskLevel >= MEDIUM`. That is why `MEDIUM`
  starts at `0.50` here — otherwise the default config would never fire.
- Any HTTP error or timeout makes the manager fall back to `SAFE` **silently**.
  Treat a 5xx as a false negative, not as a loud failure.

### `/v1/detect` — response

```json
{
  "is_fraud": true,
  "risk_score": 0.92,
  "risk_level": "HIGH",
  "reason": "Fraud pattern detected by the AI model (92% risk score). One-time password (OTP) request.",
  "signals": [
    { "type": "SUSPICIOUS_KEYWORDS", "description": "One-time password (OTP) request", "severity": 0.62 }
  ],
  "model_version": "rf-cuml-tfidf"
}
```

All of `is_fraud`, `risk_score`, `risk_level`, `reason` are **required**: the DTO
has no defaults, and Gson's `Unsafe` instantiation leaves missing fields `null`,
which later NPEs inside the app (`reason.isNotBlank()`, `RiskLevel.valueOf`).
`risk_level` must be an exact enum name (`SAFE|LOW|MEDIUM|HIGH|CRITICAL`).
Unknown `signals[].type` values are remapped to `SUSPICIOUS_KEYWORDS` by the app.

### `/v1/transcribe` — the camelCase trap

`SttResponse` in `SttApiService.kt` has **no `@SerializedName`**, and
`NetworkClient` uses a default `GsonConverterFactory`. Gson therefore matches the
literal Kotlin property names — *not* the snake_case shown in the docs:

| Docs advertise | The app actually reads |
|---|---|
| `call_id` | `callId` |
| `full_text` | `fullText` |
| `duration_seconds` | `durationSeconds` |
| `start_ms` / `end_ms` | `startMs` / `endMs` |

A server that follows the docs alone makes `fullText` null and the engine throws
on every single transcription. This server returns **both spellings in one
payload** (Gson ignores unknown keys), so both contracts hold:

```json
{
  "callId": "abc", "call_id": "abc",
  "fullText": "…", "full_text": "…", "text": "…",
  "language": "en-US", "detected_language": "en",
  "durationSeconds": 3.0, "duration_seconds": 3.0, "audio_duration_ms": 3000,
  "confidence": 0.91, "overall_confidence": 0.91,
  "segments": [{
    "id": "seg-001", "text": "…", "speaker": "unknown",
    "startMs": 0, "endMs": 3200, "start_ms": 0, "end_ms": 3200,
    "confidence": 0.91
  }],
  "words": [{ "word": "…", "startMs": 0, "endMs": 320, "confidence": 0.9, "speaker": "unknown" }]
}
```

The request has its own quirk: `RemoteSttEngine` only sends `audio`, `language`
and `call_id`, and that `call_id` is really the filename stem:

```
call_<callId>_chunk_<n>     // AudioCaptureManager.segmentPath()
```

The server extracts the real call ID and chunk index from it, and treats
`chunk_index`/`is_final` as optional.

---

## 2. Setup

Prerequisites: Linux x86_64, Python 3.13, an NVIDIA driver supporting CUDA 13
(the model is a cuML GPU estimator), and `ffmpeg` on `PATH` (Whisper decodes
audio with it).

```bash
cd "fraud_server"

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r ../requirements.txt \
    --index-strategy unsafe-best-match

cp .env.example .env          # optional; defaults work
.venv/bin/python manage.py migrate
.venv/bin/python manage.py warmup      # loads cuML + downloads Whisper tiny
```

> The pinned dependency list lives at the repository root
> ([`../requirements.txt`](../requirements.txt)) and covers both this server and
> `fraud_detection_package/`. `fraud_server/requirements.txt` just includes it.

`warmup` is worth running once: cuML's first CUDA context and the Whisper
download together can exceed the app's **30 s** OkHttp timeout, so you do not
want that happening during a live call.

Sanity-check the whole contract against a running server:

```bash
./run.sh                                             # migrate + warmup + runserver
./smoke_test.sh                                      # terminal 2
```

`run.sh` applies migrations on first run, warms both models (so the first live
call is not slow), then starts the dev server. `WARMUP=0 ./run.sh` skips warmup;
`MODE=gunicorn ./run.sh` starts gunicorn instead.

`smoke_test.sh` asserts both `/v1/detect` request shapes and — importantly — the
camelCase `/v1/transcribe` keys the Android DTO actually reads.

---

## 3. Running for a device

```bash
# Emulator: 10.0.2.2 maps to the host
# Physical device: use the host's LAN IP, or `adb reverse tcp:8000 tcp:8000`
.venv/bin/python manage.py runserver 0.0.0.0:8000

# Or, closer to production:
.venv/bin/gunicorn config.wsgi \
    --bind 0.0.0.0:8000 \
    --workers 1 --threads 4 --timeout 120
```

Keep **one worker**: each worker would load its own copy of the Random Forest
into VRAM. Inference is fast enough that one process with four threads handles a
call comfortably.

### Android-side changes (3 lines, in the app project)

1. `feature/frauddetection/.../di/FraudDetectionModule.kt`
   ```kotlin
   private const val FRAUD_API_BASE_URL = "http://10.0.2.2:8000/"   // trailing slash required
   ```
   and bind `RemoteFraudDetectionEngine` instead of `MockFraudDetectionEngine`.

2. `feature/stt/.../di/SttModule.kt`
   ```kotlin
   private const val STT_BASE_URL = "http://10.0.2.2:8000/"
   ```
   and bind `RemoteSttEngine` instead of `MockSttEngine`.

3. Plain HTTP is blocked by default on Android 9+. Add to the **debug** manifest:
   ```xml
   <application android:usesCleartextTraffic="true" …>
   ```
   …or use `adb reverse tcp:8000 tcp:8000` and `http://127.0.0.1:8000/`.

4. Only if you set `FRAUD_API_KEY`, uncomment the header in
   `core/.../network/AuthInterceptor.kt`:
   ```kotlin
   .header("Authorization", "Bearer $apiKey")
   ```

> The app must be run on a **physical device** for live call audio; the emulator's
> `MediaRecorder` does not capture call audio.

---

## 4. Configuration

Everything lives in `.env` (see `.env.example`). Most useful knobs:

| Variable | Default | Why |
|---|---|---|
| `FRAUD_API_KEY` | *(empty)* | Empty = open API, matching the shipped app which sends no header |
| `FRAUD_DECISION_THRESHOLD` | `0.5` | `is_fraud` boundary |
| `FRAUD_BAND_*_MAX` | `0.30 / 0.50 / 0.70 / 0.90` | Probability → `risk_level`; must be increasing |
| `WHISPER_MODEL` | `tiny` | `base`/`small` trade latency for accuracy |
| `WHISPER_DEVICE` | `cuda` | GPU is the practical choice for the app's 30 s request budget (seconds per 15 s chunk on GPU vs tens of seconds on CPU); auto-falls back to CPU |
| `WHISPER_BEAM_SIZE` | `1` | Greedy; raise for accuracy |
| `WHISPER_CPU_THREADS` | `0` (auto → 2) | Only for the CPU fallback; torch's default oversubscribes cores badly |
| `LOG_TRANSCRIPTS` | `false` | Stores transcript text in SQLite (PII) |
| `MAX_AUDIO_MB` | `25` | Upload cap |
| `PRELOAD_MODELS` | `true` | Background load on startup |

Tuning tips:

- The model returns a **probability**, not a calibrated confidence (Random Forest
  vote frequency). Bands are a product decision, not model truth.
- `signals[]` comes from a regex lexicon in `detector/services/keywords.py`
  (OTP/CVV/gift cards/urgency/threats/…). It **never** changes `risk_score`; it
  exists so notifications can say *why*.
- Raise `FRAUD_DECISION_THRESHOLD` for precision, lower it for recall.

---

## 5. Layout

```
fraud_server/
├── config/                    # settings, urls, wsgi/asgi
├── detector/
│   ├── apps.py                # background model preload
│   ├── auth.py                # optional Bearer / X-Api-Key
│   ├── conf.py                # env config + risk banding
│   ├── models.py              # DetectionLog, TranscriptionLog
│   ├── serializers.py         # accepts nested + flat detect bodies
│   ├── views.py               # /v1/detect, /v1/transcribe, /healthz
│   ├── services/fraud.py      # TF-IDF → cuML inference (locked)
│   ├── services/keywords.py   # signal lexicon + reason text
│   ├── services/stt.py        # Whisper + dual-spelling response
│   └── management/commands/warmup.py
├── requirements.txt
├── run.sh                     # migrate + warmup + serve
├── smoke_test.sh
└── .env.example
```

Inspect traffic later via `sqlite3 db.sqlite3` or `/admin/` (create a superuser
with `manage.py createsuperuser`).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/healthz` stuck on `503 starting` | Models still loading; run `manage.py warmup`. Check the server log. |
| `No module named 'cuml'` | Wrong interpreter. Use `.venv/bin/python`, never system Python. |
| `cuml-cu13==0.0.0a0` | Unpinned cuML from PyPI. Reinstall with the NVIDIA index flags above. |
| `CUDA driver version is insufficient` | Driver must support CUDA 13 (580+). |
| Every `detect` returns `SAFE` with `risk_score: 0.0` | Model failed to load (check `/healthz` → `fraud_model.error`) **or** the app is swallowing a 4xx/5xx (check `DetectionLog.error`). |
| App shows "error" for STT / transcript always empty | Check the response still contains `fullText`/`durationSeconds` camelCase keys. |
| First request times out (30 s) | Run `warmup` before the call, or set `PRELOAD_MODELS=true`. |
| Whisper very slow / requests time out at 30 s | You are on the CPU fallback. Set `WHISPER_DEVICE=cuda`, or lower `WHISPER_BEAM_SIZE` / `WHISPER_CPU_THREADS=2`. |
| `ffmpeg` errors | Install `ffmpeg` on the host. |

### Known limitations

- **ASR quality bounds end-to-end accuracy.** Whisper `tiny` on 15 s phone-call
  audio will be noticeably worse than the 98% figure, which is measured on clean
  text transcripts. The model is a screening signal, not a verdict.
- No speaker diarisation: `speaker` is always `unknown`.
- The app does not retry; a dropped request is a missed analysis window.
- Alert history lives in the app's memory, so the server cannot reconcile
  `VIEWED`/`DISMISSED` state.
- SQLite + one gunicorn worker is the intentional deployment target. For multiple
  hosts, move to Postgres and a separate model service.
