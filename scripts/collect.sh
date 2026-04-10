#!/usr/bin/env bash

# 24/7 Server Data Collection Script
# Purpose: Accumulate organic paper execution data across weather, crypto, and
# commodities markets. Claude is disabled — heuristic + forecast model only.
# All BOT_* configuration lives here. .env provides secrets and infra constants only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Single-instance guard ─────────────────────────────────────────────────────
LOCKFILE="/tmp/motorcade_collect.lock"
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
  echo "ERROR: collect.sh is already running (lock held by $(cat ${LOCKFILE} 2>/dev/null)). Exiting." >&2
  exit 1
fi
echo $$ >&9

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
FIXED_LOG="${SCRIPT_DIR}/../var/logs/bot.log"
exec > >(tee -a "${LOG_FILE}" | tee -a "${FIXED_LOG}") 2>&1
echo "Logging to ${LOG_FILE}"

# --- LOG ROTATION ---
echo "Performing 7-day log cleanup..."
find var/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
find var/research -name "*.jsonl" -mtime +7 -delete 2>/dev/null || true
find var/cycles -name "*.json" -mtime +7 -delete 2>/dev/null || true
find "${THEBANK_LOG_DIR}" -name "24h_collection_*.log" -mtime +7 -delete 2>/dev/null || true

# ── Runtime ───────────────────────────────────────────────────────────────────
export BOT_CARGO_PROFILE="release"
export BOT_EXCHANGE_BACKEND="kalshi"
export BOT_EXECUTION_MODE="paper"
export BOT_RUN_ONCE="false"
export BOT_RUN_SMOKE_TEST="false"
export BOT_RUN_REPLAY="false"
export BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY="true"
export BOT_CYCLE_SECONDS="600"
export BOT_STATE_PATH="var/state/runtime_state.json"
export BOT_JOURNAL_PATH="var/logs/trade_journal.jsonl"
export BOT_MARKET_RESOLUTION="best_effort"

# ── Claude (disabled — heuristic + forecast model only) ───────────────────────
export BOT_CLAUDE_TRIGGER_MODE="never"
export BOT_CLAUDE_EVERY_N_CYCLES="1"
export BOT_CLAUDE_VERTICAL_BLOCKLIST=""
export BOT_VALUATION_CACHE_TTL_SECS="60"
export BOT_VALUATION_MARKETS="30"
export BOT_VALUATION_TIMEOUT_MS="20000"

# ── Scanning ──────────────────────────────────────────────────────────────────
export BOT_SCAN_MIN_VOLUME="100"
export BOT_SCAN_MAX_SPREAD_CENTS="15.0"
export BOT_SCAN_MAX_MARKETS="2000"
export BOT_SCAN_SERIES_MAX_PER_FETCH="200"
# Tier 1: explicitly fetched first (avoids KXMVE alphabetical flood)
# KXHIGHT covers all new-format weather cities (BOS,DAL,HOU,SEA,PHX,SATX,LV,ATL,MIN,NOLA,DC,SFO,OKC)
export BOT_SCAN_SERIES_ALLOWLIST="KXHIGHT,KXSILVERD,KXGOLDMON,KXBTCD,KXETHD,KXSOLD,KXXRPD,KXNATGASMON,KXSUGARMON,KXLCATTLEMON,KXCOPPERMON,KXNASDAQ100MINY"
export BOT_SCAN_SERIES_BLOCKLIST="KXQUICKSETTLE,KXNFLDRAFTWR,KXNFLDRAFTTE,KXGAPRIMARY,KXSCOTUSMENTION,KXSPOTSTREAMGLOBAL"
# Tier 2: category-based discovery — sports excluded (negative EV)
export BOT_SCAN_TIER2_CATEGORIES="Crypto,Financials,Climate and Weather"
export BOT_SCAN_MAX_TIER2_SERIES="60"

# ── Enrichment ────────────────────────────────────────────────────────────────
export BOT_ENRICHMENT_MARKETS="80"

# ── Valuation / candidate selection ──────────────────────────────────────────
export BOT_MISPRICING_THRESHOLD="0.01"
export BOT_FALLBACK_MISPRICING_THRESHOLD="0.005"
export BOT_MIN_EDGE_PCT="0.01"
export BOT_MIN_CANDIDATES="10"

# ── Allocation + risk ─────────────────────────────────────────────────────────
export BOT_BANKROLL="10000"
export BOT_MAX_OPEN_EXPOSURE="100000000"  # No kill switch in paper mode
export BOT_MAX_DAILY_LOSS="500"
export BOT_MAX_TRADES_PER_CYCLE="20"
export BOT_MAX_ORDERS_PER_MIN="20"
export BOT_MAX_FRACTION_PER_TRADE="0.06"
export BOT_MAX_TOTAL_FRACTION_PER_CYCLE="0.20"
export BOT_MIN_FRACTION_PER_TRADE="0.005"

# ── Execution allowlist ───────────────────────────────────────────────────────
export BOT_EXECUTION_SERIES_ALLOWLIST="KXHIGHT,KXBTCD,KXETHD,KXXRPD,KXSOLD,KXGOLDD,KXGOLDMON,KXSILVERD,KXSILVERMON,KXSILVERW,KXNATGASMON,KXSUGARMON,KXLCATTLEMON,KXCOPPERMON,KXNASDAQ100MINY"

# ── Models ────────────────────────────────────────────────────────────────────
export BOT_MODEL_FORECAST_PATH="var/models/forecast/latest.json"
export BOT_MODEL_EXECUTION_PATH="var/models/execution/latest.json"
export BOT_FORECAST_SERIES_EXCLUSIONS="KXBTCD,KXBTCW,KXBTC,KXETH,KXETHD,KXXRP,KXXRPD,KXSOL,KXSOLE,KXSOLD,KXDOGE,KXDOGED,KXNASDAQ100U,KXEURUSD,KXBRENTD,KXCOPPERD,KXSILVERD,KXGOLDD,KXPGAR1TOP10,KXPGAR1TOP5,KXPGAMAKECUT,KXPGAR3LEAD,KXATPCHALLENGERMATCH,KXF1POLE,KXF1FASTLAP,KXUFCMOV,KXMVESPORTSMULTIGAMEEXTENDED,KXMVECROSSCATEGORY"
export BOT_EXECUTION_TRAIN_SOURCES="organic_paper,live_real,retroactive_synthetic"
export BOT_EXECUTION_SUPPLEMENTAL_PATHS="var/features/execution/execution_training_retroactive.jsonl"

# ── Policy ────────────────────────────────────────────────────────────────────
export BOT_POLICY_MODE="shadow"
export BOT_POLICY_MIN_EXPECTED_REALIZED_PNL="0.0"
export BOT_POLICY_MAX_ACTIONS_PER_CANDIDATE="4"
export BOT_POLICY_DEFAULT_LEGACY_FALLBACK="true"

# ── Periodic background tasks ─────────────────────────────────────────────────
# Outcome backfill runs automatically every 18 cycles inside the loop.
# Do NOT set BOT_RUN_OUTCOME_BACKFILL here — it's a one-shot mode that exits immediately.
export BOT_OUTCOME_LOOKBACK_DAYS="14"

# ── Replay (disabled) ─────────────────────────────────────────────────────────
export BOT_REPLAY_DAYS="3"
export BOT_REPLAY_CYCLES_PER_DAY="144"
export BOT_REPLAY_BANKROLL="10000"

echo "Starting 24/7 Server Collection Mode..."
echo "Claude: disabled. Forecast model + heuristic only."
echo "Weather: GEFS sidecar covering all KXHIGHT cities."

run_cargo_bot
