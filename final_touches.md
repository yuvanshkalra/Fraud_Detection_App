# Final touches

What is done, what is left, and in what order. Each item names the exact file so
it can be picked up without re-reading the whole project.

Current state (updated for this Windows laptop): the Django backend runs in
**WSL2 + Ubuntu 24.04** (Linux-only cuML/CUDA stack); the Android app is wired to
the **remote engines** at `http://127.0.0.1:8000/` for USB `adb reverse`; the API
is intentionally open for the first run. Nothing has been deployed. See
[`SETUP_WINDOWS_WSL.md`](SETUP_WINDOWS_WSL.md) for the laptop setup.

---

## 0. Status at a glance

| Area | State |
|---|---|
| Django server (`fraud_server/`) | ✅ implemented, both models load, contract smoke test passes |
| Model artifact (`fraud_detection_package/`) | ✅ loads and predicts with cuML 26.10 |
| Android client | ✅ remote engines bound, both URLs `http://127.0.0.1:8000/` (USB `adb reverse`); debug cleartext manifest added |
| Auth between app and server | ⬜ open API on both sides (chosen for first run) |
| TLS | ⬜ plain HTTP, debug-only |
| Service management (systemd/gunicorn) | ⬜ dev server only |
| Accuracy validation on real calls | ⬜ not started |

---

## 1. Android app — blocking for the first real run

> All paths in this section are relative to `AI_Fraud_Call_Detection_Application/`.

### 1.1 Point both clients at the server

- ✅ `feature/frauddetection/src/main/java/com/fraudcalldetector/feature/frauddetection/di/FraudDetectionModule.kt`
  ```kotlin
  private const val FRAUD_API_BASE_URL = "http://127.0.0.1:8000/"
  ```
- ✅ `feature/stt/src/main/java/com/fraudcalldetector/feature/stt/di/SttModule.kt`
  ```kotlin
  private const val STT_BASE_URL = "http://127.0.0.1:8000/"
  ```
  Trailing slash required. Emulator: `10.0.2.2`. USB: `adb reverse tcp:8000 tcp:8000` + `127.0.0.1`.

### 1.2 Bind the remote engines

- ✅ `FraudDetectionModule.kt`: bound `RemoteFraudDetectionEngine`
- ✅ `SttModule.kt`: bound `RemoteSttEngine`

Without this the app never performs a network call, no matter the URL.

### 1.3 Permit cleartext HTTP in debug builds

- ✅ created `app/src/debug/AndroidManifest.xml`:
  ```xml
  <manifest xmlns:android="http://schemas.android.com/apk/res/android">
      <application android:usesCleartextTraffic="true" />
  </manifest>
  ```
  Release builds keep the default (no cleartext), so production must use HTTPS.

### 1.4 Build and install

```bash
cd AI_Fraud_Call_Detection_Application
./gradlew :app:installDebug
adb logcat | grep -iE "okhttp|fraud|stt|RemoteFraud|RemoteStt"
```

- ⬜ Confirm `HTTP 200` and a parsed `risk_level` in logcat.
- ⬜ Confirm the app asks for and receives `READ_PHONE_STATE`, `READ_CALL_LOG`,
  `RECORD_AUDIO`, `POST_NOTIFICATIONS`.
- ⬜ Make a real inbound call on a physical device and watch a chunk reach the server.

### 1.5 Decide the base-URL strategy

- ✅ Picked and documented in the app README + `SETUP_WINDOWS_WSL.md`:
  - **adb reverse + `127.0.0.1`** — survives IP changes, needs adb attached.
  - **Manual edit + rebuild** — current approach, two files per IP change.
  - **Gradle property → `BuildConfig`** — one untracked property, still a rebuild.

  If the property approach is chosen, `buildFeatures { buildConfig = true }` must be
  added to both feature modules (it is currently enabled only in `:app`).

---

## 2. Android app — non-blocking but worth fixing

- ✅ **Fixed the STT DTO.** `SttResponse` and `SttSegmentDto` now use
  `@SerializedName` for the snake_case keys. The server's dual-spelling payload is
  kept as a compatibility shim.

- ✅ **Documented the `supportsIncrementalAnalysis` discrepancy** in the app README.
  The manager still calls `/v1/detect` per 15 s chunk (the flag remains `false` but
  is not consulted); honouring it would disable all mid-call alerts, so the flag is
  marked misleading rather than changing behaviour.

- ✅ **`call_id` fixed on `/v1/transcribe`.** `SttEngineConfig` carries `callId`,
  `SttManagerImpl` fills it, and `RemoteSttEngine` sends it as the multipart field
  (filename stem retained as fallback).

- ✅ **Verified no mock fallback in release builds.** The mocks are only reachable
  through the DI bindings; a failed remote request returns an empty transcript /
  SAFE analysis, never mock text.

- ✅ **Persist per-chunk segments.** `SttManagerImpl.persistSession()` builds real
  `TranscriptSegment`s; `TranscriptEntity` stores them as JSON (DB version 3).

- ✅ **Persist alert history.** `AlertHistoryRepositoryImpl` serializes the alert
  list to SharedPreferences so records survive process kill.

- ✅ **Fraud flags survive cold restart.** `CallEventEntity` now serializes and
  deserializes `fraudAnalysis.flags` with Gson.

- ✅ **Added retry/back-off** in `RemoteFraudDetectionEngine`: one retry (300 ms)
  on 5xx / network exceptions before surfacing the error.

- ✅ **Show backend unreachability in the UI.** `ActiveCallScreen` renders a
  `BackendErrorBanner` with a Retry action from the previously unused
  `fraudDetectionError` state; the ViewModel clears it on success.

---

## 3. Server — blocking for daily use

### 3.1 Keep the models warm

- ✅ background preload (`detector/apps.py`) + `manage.py warmup`
- ✅ cuML warm-up inference so the first `/v1/detect` is ~13 ms, not ~2.5 s
- ✅ Whisper warm-up and CUDA default with CPU fallback
- ⬜ if the server restarts often, run a one-shot `warmup` before switching traffic
  (or use systemd `ExecStartPre`)

### 3.2 Choose how the phone finds the host

- ⬜ static DHCP lease and/or local DNS for the laptop, **or**
- ✅ `adb reverse tcp:8000 tcp:8000` for USB-only testing (chosen; see `SETUP_WINDOWS_WSL.md`), **or**
- ⬜ a tunnel (Cloudflare/Tailscale) if the phone leaves the LAN

### 3.3 Verify the endpoints from the target device

```bash
curl -s http://<host>:8000/healthz | python3 -m json.tool   # from the phone's browser
./smoke_test.sh                                             # from the laptop
```

- ⬜ `status: ok`, `fraud_model.loaded: true`, `whisper.loaded: true`
- ⬜ `DetectionLog` / `TranscriptionLog` rows appear for real requests

---

## 4. Server — hardening

- ⬜ **Enable auth.** *(Deferred for first run: open API chosen; the app sends no
  header.)* Set `FRAUD_API_KEY` in `fraud_server/.env`, then uncomment the
  header in `core/.../network/AuthInterceptor.kt`:
  ```kotlin
  .header("Authorization", "Bearer $apiKey")
  ```
  The server accepts `Authorization: Bearer …` or `X-Api-Key`. Move the key out of
  source into `local.properties` → `BuildConfig` or EncryptedSharedPreferences.

- ⬜ **TLS.** Put nginx/Caddy in front with a real certificate and switch both
  app URLs to `https://`. Removes the cleartext exception from the release build.

- ⬜ **Run as a service.**
  ```ini
  # /etc/systemd/system/fraud-server.service
  [Service]
  WorkingDirectory=/home/varun/Desktop/Fraud App/fraud_server
  ExecStartPre=/home/varun/Desktop/Fraud App/fraud_server/.venv/bin/python manage.py migrate --noinput
  ExecStart=/home/varun/Desktop/Fraud App/fraud_server/.venv/bin/gunicorn config.wsgi \
      --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 120
  Restart=always
  User=varun
  ```
  One worker only: a second worker duplicates the model in VRAM.

- ✅ **Log retention command added** (`manage.py prune_logs`, `LOG_RETENTION_DAYS=30`).
  Schedule it via cron/timer on long-lived installs; Django's own logs go to stdout.

- ⬜ **Decide on PII.** `LOG_TRANSCRIPTS=false` by default; transcript hashes and
  lengths are still stored. If this ever handles real users, define retention and
  access rules, and restrict `/admin/`.

- ⬜ **Create an admin user** if you want the log UI: `manage.py createsuperuser`.

- ⬜ **Postgres** only if you add a second host or need concurrent writers; SQLite is
  fine for one gunicorn process.

- ⬜ **Monitoring.** `/healthz` is enough for a simple check; add an uptime probe and
  alert on 503. Consider counting 4xx/5xx in the DB — the app hides them.

---

## 5. Model quality

- ⬜ **Tune the bands against real calls.** They are product thresholds, not model
  truth. Collect a set of labeled calls and choose cut-offs that maximise useful
  alerts at an acceptable false-positive rate.
- ⬜ **Recalibrate.** Random Forest `predict_proba` is vote frequency, not a
  calibrated probability. If the UI says "92% risk", consider isotonic/Platt
  calibration on held-out data.
- ⬜ **Measure on ASR output, not clean text.** The 98% accuracy figure is from clean
  transcripts. Re-evaluate using Whisper transcripts of real calls; this is the
  dominant source of error.
- ⬜ **Consider `base`/`small` on GPU** if `tiny` transcripts are too rough —
  measure against the 30 s budget (a few seconds for `tiny` on a 15 s clip).
- ⬜ **Signals are heuristic.** `detector/services/keywords.py` is a regex lexicon.
  It never changes the score, but its wording appears in notifications — review it
  for false-alarm tone.
- ⬜ **Language coverage.** The model is trained on English; `language` is accepted
  but not used for routing. Reject or flag non-English calls explicitly.
- ⬜ **Version provenance.** `model_version` is `rf-cuml-tfidf`; include the artifact
  hash or training date so alerts can be traced to a model build.

---

## 6. End-to-end acceptance test

Run in this order; each step has a concrete pass condition.

- ⬜ `cd fraud_server && ./run.sh` → log shows `Fraud model ready`, `Whisper ready`
- ⬜ `curl /healthz` → `status: ok` from the laptop **and** from the phone's browser
- ⬜ `./smoke_test.sh` → `all checks passed`
- ⬜ `manage.py test detector` → 23 tests OK
- ⬜ app installed with remote bindings → logcat shows a 200 from `/v1/detect`
- ⬜ real inbound call → `TranscriptionLog` rows every ~15 s, transcript accumulates
- ⬜ scam-like script read aloud → notification fires, `risk_level` ≥ `MEDIUM`,
  banner appears; `DetectionLog.is_fraud = 1`
- ⬜ benign call → no notification, `SAFE`
- ⬜ kill the server mid-call → app does not crash; call ends as `SAFE` (expected,
  but confirm it degrades quietly rather than hanging)
- ⬜ restart the app → history shows the persisted call and its risk level

---

## 7. Known gaps (accepted for now)

| Gap | Impact |
|---|---|
| App treats every backend failure as `SAFE` | Silent false negatives |
| No retry on the app side | A dropped chunk is a lost analysis window |
| `speaker` is always `unknown` | No diarisation; Whisper has none built in |
| Alert history is in-memory | Lost on process kill |
| Fraud flag details lost on cold DB restart | Only headline risk level survives |
| Whisper `tiny` on phone audio | Transcript quality limits accuracy more than the model |
| Debug-only cleartext | Release build needs TLS before it can talk to the server |
| Free-text IP in two Kotlin files | Rebuild after every network change |
