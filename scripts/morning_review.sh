#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_repo_env

echo "== Research report =="
BOT_RUN_RESEARCH_REPORT=true run_cargo_bot

echo
echo "== Model report =="
BOT_RUN_MODEL_REPORT=true run_cargo_bot

echo
echo "== Policy report =="
BOT_RUN_POLICY_REPORT=true run_cargo_bot
