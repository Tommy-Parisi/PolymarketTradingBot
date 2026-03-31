#!/usr/bin/env bash

# 24/7 Server Data Collection Script
# Purpose: Generate high-quality "smart" execution data by using Claude frequently 
# while staying within a ~$3/day budget. Runs in paper mode to capture organic labels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_repo_env
ensure_logs_dir

# --- Output to TheBank external drive ---
THEBANK_LOG_DIR="/mnt/TheBank/polymarketbot/logs"
if [[ ! -d "${THEBANK_LOG_DIR}" ]]; then
  echo "ERROR: TheBank not mounted at /mnt/TheBank. Run: sudo mount /dev/sdb1 /mnt/TheBank" >&2
  exit 1
fi
LOG_FILE="${THEBANK_LOG_DIR}/24h_collection_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

# --- LOG ROTATION ---
# Clean up logs and research artifacts older than 7 days to save disk space
echo "Performing 7-day log cleanup..."
find var/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
find var/research -name "*.jsonl" -mtime +7 -delete 2>/dev/null || true
find var/cycles -name "*.json" -mtime +7 -delete 2>/dev/null || true
# Also clean up old collection logs on the external drive
find "${THEBANK_LOG_DIR}" -name "24h_collection_*.log" -mtime +7 -delete 2>/dev/null || true

# --- IMMUNITY: Override .env settings that block collection ---
export BOT_CLAUDE_VERTICAL_BLOCKLIST="" # ALLOW Claude to value all verticals
export BOT_CLAUDE_TRIGGER_MODE="on_heuristic_candidates"
export BOT_VALUATION_CACHE_TTL_SECS="60"

# --- PERIODIC BACKGROUND TASKS ---
# Enable periodic outcome backfill and dataset building in the main loop
export BOT_RUN_OUTCOME_BACKFILL="true"
export BOT_RUN_DATASET_BUILD="true"
export BOT_OUTCOME_LOOKBACK_DAYS="14"

# --- Execution & Capture Configuration ---
export BOT_EXECUTION_MODE="paper"
export BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY="true"
export BOT_CARGO_PROFILE="release"
export BOT_CYCLE_SECONDS="600" 

# --- Claude Cost Management (~$3/day target) ---
# Cost per 1k tokens: ~$0.015 (mixed input/output).
# At 144 cycles/day, we can afford ~1.4k tokens per cycle to hit ~$3/day.
export BOT_VALUATION_MARKETS="30"      # Keep batches small to manage token count
export BOT_CLAUDE_EVERY_N_CYCLES="1"   # Check every cycle, but only pay if heuristic likes it
export BOT_VALUATION_TIMEOUT_MS="20000" # 20s per attempt — haiku is fast but allow for cold-start latency

# --- Maximising Data Variety (Lowering the Bar) ---
export BOT_MISPRICING_THRESHOLD="0.01" 
export BOT_MIN_EDGE_PCT="0.01"
export BOT_FALLBACK_MISPRICING_THRESHOLD="0.005"
export BOT_MIN_CANDIDATES="10"

# --- Scanning Strategy ---
export BOT_SCAN_MIN_VOLUME="100"
export BOT_SCAN_MAX_SPREAD_CENTS="15.0"
export BOT_SCAN_SERIES_ALLOWLIST="KXSILVERD,KXGOLDMON,KXBTCD,KXETHD,KXSOLD,KXXRPD,KXNASDAQ100MINY"
export BOT_SCAN_TIER2_CATEGORIES="Sports,Crypto,Financials,Climate and Weather"
export BOT_SCAN_MAX_TIER2_SERIES="40"
export BOT_ENRICHMENT_MARKETS="60"

echo "Starting 24/7 Server Collection Mode (Override Mode)..."
echo "Override: Claude is now ENABLED for Crypto."
echo "Targeting Claude valuations on heuristic candidates every 10m."
echo "Budgeted for ~\$3/day Claude usage."

run_cargo_bot
