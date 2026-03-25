#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_repo_env() {
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
}

run_cargo_bot() {
  local profile="${BOT_CARGO_PROFILE:-dev}"
  local args=("--quiet")

  if [[ "${profile}" == "release" ]]; then
    args=("--release" "--quiet")
  fi

  (cd "${REPO_ROOT}" && cargo run "${args[@]}")
}

ensure_logs_dir() {
  mkdir -p "${REPO_ROOT}/var/logs"
}
