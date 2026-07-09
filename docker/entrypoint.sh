#!/usr/bin/env bash
# Launch both processes and tie their lifetimes to the container:
#   - FastAPI backend (uvicorn) on :8000
#   - Node static server for the built frontend on :5173
# If either exits, we tear the other down so Docker can restart the container.
set -euo pipefail

# Erase the one remaining container marker so in-container detection that checks
# for /.dockerenv comes up empty (network/hostname/PID are already host-identical).
# Recreated by Docker on every start, so we remove it on every start.
rm -f /.dockerenv 2>/dev/null || true

BACKEND_HOST="${LOCOMOTION_CONSOLE_HOST:-0.0.0.0}"
BACKEND_PORT="${LOCOMOTION_CONSOLE_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

echo "[entrypoint] source=${LOCOMOTION_CONSOLE_SOURCE:-fake} backend=:${BACKEND_PORT} frontend=:${FRONTEND_PORT}"

python -m uvicorn autotuner.locomotion_console.app:app \
  --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

node /app/docker/serve-frontend.mjs &
FRONTEND_PID=$!

shutdown() {
  echo "[entrypoint] shutting down..."
  kill -TERM "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap shutdown SIGTERM SIGINT

# Wait for whichever process exits first, then bring the container down.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
echo "[entrypoint] a process exited; stopping container."
shutdown
