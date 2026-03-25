#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_repo_env
ensure_logs_dir

export BOT_EXCHANGE_BACKEND="${BOT_EXCHANGE_BACKEND:-kalshi}"
export BOT_EXECUTION_MODE="${BOT_EXECUTION_MODE:-live}"
export BOT_RUN_SMOKE_TEST="${BOT_RUN_SMOKE_TEST:-true}"
export BOT_RUN_ONCE="${BOT_RUN_ONCE:-false}"
export BOT_CYCLE_SECONDS="${BOT_CYCLE_SECONDS:-600}"
export BOT_POLICY_MODE="${BOT_POLICY_MODE:-shadow}"
export BOT_MODEL_FORECAST_PATH="${BOT_MODEL_FORECAST_PATH:-var/models/forecast/latest.json}"
export BOT_MODEL_EXECUTION_PATH="${BOT_MODEL_EXECUTION_PATH:-var/models/execution/latest.json}"
export BOT_FORECAST_SHADOW_ENABLED="${BOT_FORECAST_SHADOW_ENABLED:-true}"
export BOT_EXECUTION_SHADOW_ENABLED="${BOT_EXECUTION_SHADOW_ENABLED:-true}"
export BOT_POLICY_SHADOW_ENABLED="${BOT_POLICY_SHADOW_ENABLED:-true}"

run_cargo_bot
