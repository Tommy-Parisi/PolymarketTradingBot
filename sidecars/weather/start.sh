#!/bin/bash
# Start the weather sidecar. Reads WEATHER_SIDECAR_HOST/PORT from environment.
# Logs go to stdout; redirect as needed.
set -euo pipefail

SIDECAR_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SIDECAR_DIR}/../.." && pwd)"
cd "${SIDECAR_DIR}"

PORT="${WEATHER_SIDECAR_PORT:-8765}"
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "weather sidecar already running on port ${PORT}, skipping start"
    exit 0
fi

# Anchor prediction logs and disk cache to the repo-level var/ regardless of launch directory.
export GEFS_PREDICTION_LOG_DIR="${REPO_ROOT}/var/logs/gefs_predictions"
export GEFS_CACHE_PATH="${REPO_ROOT}/var/cache/gefs_cache.json"

exec .venv/bin/python sidecar.py
