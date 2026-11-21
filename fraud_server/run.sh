#!/usr/bin/env bash
# Start the backend. First run does migrate + model warmup automatically.
#
#   ./run.sh              # dev server on 0.0.0.0:8000
#   PORT=9000 ./run.sh
#   MODE=gunicorn ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "No .venv found. Create it first:" >&2
  echo "  uv venv --python 3.13 .venv" >&2
  echo "  uv pip install --python .venv/bin/python -r requirements.txt \\" >&2
  echo "      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match" >&2
  exit 1
fi

if [[ ! -f db.sqlite3 ]]; then
  echo "==> applying migrations"
  "$PY" manage.py migrate --noinput
fi

if [[ "${WARMUP:-1}" == "1" ]]; then
  echo "==> warming up models (set WARMUP=0 to skip)"
  "$PY" manage.py warmup
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ "${MODE:-dev}" == "gunicorn" ]]; then
  exec .venv/bin/gunicorn config.wsgi \
    --bind "$HOST:$PORT" --workers 1 --threads 4 --timeout 120 --access-logfile -
fi

exec "$PY" manage.py runserver "$HOST:$PORT"
