# Running the backend on this Windows laptop (WSL2)

The Django backend uses a **cuML GPU model (CUDA 13)**. cuML wheels do not exist
for native Windows, so the backend runs inside **WSL2 + Ubuntu 24.04**. Your
RTX 3050 Laptop GPU (driver 610.91) is visible inside WSL and supports CUDA 13.
The Android app stays on Windows/Android Studio and reaches the server through
`adb reverse`.

> The `fraud_server/.venv` checked into the Windows folder is a **Linux venv**
> built on another machine (`/home/varun/...`) and cannot be used here. A fresh
> venv is created in the WSL copy of the project.

---

## 1. What was installed on this laptop

- Windows features: `Microsoft-Windows-Subsystem-Linux`, `VirtualMachinePlatform`
  (a reboot was required after enabling them).
- WSL distro: **Ubuntu-24.04** (WSL version 2), default user `yuvan`.
- Ubuntu packages: `ffmpeg`, `curl`, `ca-certificates`, `git`, `sudo`, `unzip`.
- `uv` (Python package manager) at `/usr/local/bin/uv`.
- Project copied to `/home/yuvan/fraud-app` (the Windows original is left intact).
- A Python 3.13 venv at `/home/yuvan/fraud-app/fraud_server/.venv` with all
  pinned dependencies from the root `requirements.txt` (including cuML-cu13).

---

## 2. Start the server

**Verified on this laptop:** migrations applied, both models warmed on CUDA,
`/healthz` returns `status: ok` with `fraud_model.loaded: true` and
`whisper.device: cuda`, `smoke_test.sh` → `all checks passed`, 23 Django tests OK,
and `manage.py prune_logs` runs. The server is currently running on port 8000.

Open **PowerShell** (no admin needed):

```powershell
cd "C:\Users\yuvan\Fraud App\Fraud App"
.\start_server.ps1
# or directly:
wsl -d Ubuntu-24.04 --cd /home/yuvan/fraud-app/fraud_server -- ./run.sh
```

`run.sh` applies migrations, warms both models (cuML + Whisper `tiny`) and then
serves on `0.0.0.0:8000`. Warmup matters: the Android client times out after 30 s.
To skip warmup: `WARMUP=0 ./run.sh`; to use gunicorn instead:
`MODE=gunicorn ./run.sh` (keep **one** worker).

Wait for `Fraud model ready`, `Whisper ready`, then `Starting development server`.

In a second PowerShell window, verify:

```powershell
wsl -d Ubuntu-24.04 --cd /home/yuvan/fraud-app/fraud_server -- bash -lc 'curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool'
wsl -d Ubuntu-24.04 --cd /home/yuvan/fraud-app/fraud_server -- ./smoke_test.sh
wsl -d Ubuntu-24.04 --cd /home/yuvan/fraud-app/fraud_server -- .venv/bin/python manage.py test detector
```

Expected: `status: ok`, both models `loaded: true`, `all checks passed`, 23 tests OK.

You can also open <http://127.0.0.1:8000/healthz> directly in a Windows browser
(WSL forwards localhost to Windows automatically).

---

## 3. Connect the Android app (USB)

1. On the phone: **Developer options → USB debugging** on; connect by USB.
2. Confirm the device is visible from Windows: `adb devices` (authorize the prompt).
3. Set up the reverse tunnel (run once per reboot/reconnect):

   ```powershell
   adb reverse tcp:8000 tcp:8000
   ```

   The phone's `127.0.0.1:8000` now maps to the laptop's WSL server and survives
   Wi-Fi IP changes.

4. Build/install the debug app (Android Studio + JDK 17 + SDK 34 required):

   ```powershell
   cd "C:\Users\yuvan\Fraud App\Fraud App\AI_Fraud_Call_Detection_Application"
   .\gradlew.bat :app:installDebug
   adb logcat | Select-String -Pattern "okhttp|fraud|stt|RemoteFraud|RemoteStt"
   ```

   Debug package id: `com.fraudcalldetector.debug`.

5. Make a real inbound call. Logcat should show `HTTP 200` from `/v1/detect` and
   `/v1/transcribe`, and the server log should show `detector.views` lines with
   `call_id`, `chunk`, and `latency_ms`.

The app edits in this folder already use `http://127.0.0.1:8000/` in both
`FraudDetectionModule.kt` and `SttModule.kt`, bind the remote engines, and allow
cleartext traffic in debug builds only.

---

## 4. After editing files on the Windows side

The server runs from the **WSL copy**, so sync your Windows edits before
restarting it:

```powershell
wsl -d Ubuntu-24.04 -u root -- bash -lc 'cd "/mnt/c/Users/yuvan/Fraud App/Fraud App" && tar --exclude=fraud_server/.venv --exclude=__pycache__ --exclude=.git -cf - . | (cd /home/yuvan/fraud-app && tar -xf -) && chown -R yuvan:yuvan /home/yuvan/fraud-app'
```

Then stop the server (Ctrl+C in its window) and start it again. The venv and
`db.sqlite3` live in the WSL copy and are not overwritten by the sync.

---

## 5. Configuration decisions (first-run friendly)

- `fraud_server/.env` (Windows original and WSL copy) sets
  `FRAUD_API_KEY=` → **open API**, matching the app, which sends no auth header.
- `DJANGO_ALLOWED_HOSTS=*`, debug on, bind `0.0.0.0` — fine for a local laptop.
- `WHISPER_DEVICE=cuda` with automatic CPU fallback. GPU is needed to stay under
  the app's 30 s timeout (GPU is well under the budget; a 15 s chunk on CPU takes
  tens of seconds).
- `LOG_TRANSCRIPTS=false` (PII). `LOG_RETENTION_DAYS=30`; prune with
  `.venv/bin/python manage.py prune_logs` or a cron/timer.
- No TLS yet. Release builds will need HTTPS + a reverse proxy before the
  cleartext exception can be removed.

To enable API auth later, set `FRAUD_API_KEY` in `.env` on both sides, then
uncomment the header in
`AI_Fraud_Call_Detection_Application/core/.../network/AuthInterceptor.kt`.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `wsl --list --verbose` shows no distro after install | A reboot was pending from enabling Windows features — restart the laptop. |
| `nvidia-smi` missing in WSL | It is normally provided by the Windows NVIDIA driver; update to a CUDA-13-capable driver (580+). |
| `/healthz` 503 | Models still loading: run `./run.sh` fully or `.venv/bin/python manage.py warmup`; read the server log. |
| `No module named 'cuml'` | Use the WSL venv Python, never system Python or the copied Windows `.venv`. |
| Whisper very slow / app timeouts | CPU fallback; check `nvidia-smi` in WSL and `WHISPER_DEVICE=cuda`. |
| Phone shows nothing / every call SAFE | Check `adb reverse`, cleartext debug manifest, and that logcat shows a 200. The app hides 4xx/5xx (but now shows a "server unreachable" banner). |
| `adb reverse` fails after reconnect | Re-run `adb reverse tcp:8000 tcp:8000`; it does not survive a device reboot. |
| Windows edits not reflected in server | Re-sync (§4) and restart; the server only reads the WSL copy. |
