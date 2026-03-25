#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_repo_env
ensure_logs_dir

export BOT_RUN_RESEARCH_CAPTURE_ONLY="${BOT_RUN_RESEARCH_CAPTURE_ONLY:-true}"
export BOT_RUN_ONCE="${BOT_RUN_ONCE:-false}"
export BOT_CYCLE_SECONDS="${BOT_CYCLE_SECONDS:-600}"

run_cargo_bot
