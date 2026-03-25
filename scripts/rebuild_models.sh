#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_repo_env

echo "[1/4] Backfilling outcomes"
BOT_RUN_OUTCOME_BACKFILL=true run_cargo_bot

echo "[2/4] Building datasets"
BOT_RUN_DATASET_BUILD=true run_cargo_bot

echo "[3/4] Training forecast model"
BOT_RUN_FORECAST_TRAIN=true run_cargo_bot

echo "[4/4] Training execution model"
BOT_RUN_EXECUTION_TRAIN=true run_cargo_bot

echo "Model rebuild complete."
