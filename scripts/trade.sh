#!/usr/bin/env bash

# Production Trading Script — Live execution, shadow policy, Claude enabled.
#
# Intended use: real money trading once the system is ready to go live.
# Policy mode stays "shadow" until promoted to "active" — see CLAUDE.md for
# the promotion criteria (50+ shadow decisions, mean ERPNL above threshold).
#
# All BOT_* configuration lives here. .env provides secrets and infra constants only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Single-instance guard ─────────────────────────────────────────────────────
LOCKFILE="/tmp/motorcade_trade.lock"
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
  echo "ERROR: trade.sh is already running (lock held by $(cat ${LOCKFILE} 2>/dev/null)). Exiting." >&2
  exit 1
fi
echo $$ >&9

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_repo_env
ensure_logs_dir

# ── Runtime ───────────────────────────────────────────────────────────────────
export BOT_CARGO_PROFILE="release"
export BOT_EXCHANGE_BACKEND="kalshi"
export BOT_EXECUTION_MODE="live"
export BOT_RUN_ONCE="false"
export BOT_RUN_SMOKE_TEST="true"
export BOT_RUN_REPLAY="false"
export BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY="false"
export BOT_CYCLE_SECONDS="60"
export BOT_STATE_PATH="var/state/runtime_state.json"
export BOT_JOURNAL_PATH="var/logs/trade_journal.jsonl"
export BOT_MARKET_RESOLUTION="best_effort"

# ── Claude ────────────────────────────────────────────────────────────────────
# Fires only when the heuristic finds a genuine candidate — avoids wasted calls
# on cycles with nothing worth pricing.
export BOT_CLAUDE_TRIGGER_MODE="on_heuristic_candidates"
export BOT_CLAUDE_EVERY_N_CYCLES="1"
export BOT_CLAUDE_VERTICAL_BLOCKLIST=""
export BOT_VALUATION_CACHE_TTL_SECS="300"
export BOT_VALUATION_MARKETS="250"
export BOT_VALUATION_TIMEOUT_MS="20000"

# ── Scanning ──────────────────────────────────────────────────────────────────
export BOT_SCAN_MIN_VOLUME="100"
export BOT_SCAN_MAX_SPREAD_CENTS="20.0"
export BOT_SCAN_MAX_MARKETS="2000"
export BOT_SCAN_SERIES_MAX_PER_FETCH="200"
# Tier 1: high-priority series fetched first
# KXHIGHT covers all new-format weather cities backed by the GEFS sidecar
export BOT_SCAN_SERIES_ALLOWLIST="KXHIGHT,KXBTCD,KXBTCW,KXBTC,KXETH,KXETHD,KXXRPD,KXSOLD,KXGOLDD,KXGOLDMON,KXSILVERD,KXSILVERMON,KXSILVERW,KXNATGASMON,KXSUGARMON,KXLCATTLEMON,KXCOPPERMON,KXNASDAQ100MINY"
export BOT_SCAN_SERIES_BLOCKLIST="KXQUICKSETTLE,KXNFLDRAFTWR,KXNFLDRAFTTE,KXGAPRIMARY,KXSCOTUSMENTION,KXSPOTSTREAMGLOBAL"
# Tier 2: crypto, financials, weather only — sports removed (negative EV on paper data)
export BOT_SCAN_TIER2_CATEGORIES="Crypto,Financials,Climate and Weather"
export BOT_SCAN_MAX_TIER2_SERIES="60"

# ── Enrichment ────────────────────────────────────────────────────────────────
export BOT_ENRICHMENT_MARKETS="100"

# ── Valuation / candidate selection ──────────────────────────────────────────
# Higher bar than collection mode — only trade genuine edges.
export BOT_MISPRICING_THRESHOLD="0.05"
export BOT_FALLBACK_MISPRICING_THRESHOLD="0.01"
export BOT_MIN_EDGE_PCT="0.04"
export BOT_MIN_CANDIDATES="3"

# ── Allocation + risk ─────────────────────────────────────────────────────────
# Micro-sizing for live execution data collection phase.
# Goal: accumulate real fill/no-fill data cheaply. Size up once execution model is validated.
export BOT_BANKROLL="10000"
export BOT_MAX_OPEN_EXPOSURE="500"       # hard cap on total live exposure
export BOT_MAX_DAILY_LOSS="25"           # stop for the day if we lose $25
export BOT_MAX_TRADES_PER_CYCLE="5"
export BOT_MAX_ORDERS_PER_MIN="10"
export BOT_MAX_FRACTION_PER_TRADE="0.001"   # ~$10 max per trade
export BOT_MAX_TOTAL_FRACTION_PER_CYCLE="0.005"
export BOT_MIN_FRACTION_PER_TRADE="0.0005"  # ~$5 min per trade

export BOT_MAX_NOTIONAL_PER_TICKER="50"  # ~5-10 fills per market before cap

# ── Execution allowlist ───────────────────────────────────────────────────────
# Sports excluded — negative EV on paper data. Weather retained (GEFS sidecar active).
export BOT_EXECUTION_SERIES_ALLOWLIST="KXHIGHT,KXBTCD,KXBTCW,KXBTC,KXETH,KXETHD,KXXRPD,KXSOLD,KXGOLDD,KXGOLDMON,KXSILVERD,KXSILVERMON,KXSILVERW,KXNATGASMON,KXSUGARMON,KXLCATTLEMON,KXCOPPERMON,KXNASDAQ100MINY"

# ── Models ────────────────────────────────────────────────────────────────────
export BOT_MODEL_FORECAST_PATH="var/models/forecast/latest.json"
export BOT_MODEL_EXECUTION_PATH="var/models/execution/latest.json"
export BOT_FORECAST_SHADOW_ENABLED="true"
export BOT_EXECUTION_SHADOW_ENABLED="true"
export BOT_POLICY_SHADOW_ENABLED="true"
export BOT_FORECAST_SERIES_EXCLUSIONS="KXBTCD,KXBTCW,KXBTC,KXETH,KXETHD,KXXRP,KXXRPD,KXSOL,KXSOLE,KXSOLD,KXDOGE,KXDOGED,KXNASDAQ100U,KXEURUSD,KXBRENTD,KXCOPPERD,KXSILVERD,KXGOLDD,KXPGAR1TOP10,KXPGAR1TOP5,KXPGAMAKECUT,KXPGAR3LEAD,KXATPCHALLENGERMATCH,KXF1POLE,KXF1FASTLAP,KXUFCMOV,KXMVESPORTSMULTIGAMEEXTENDED,KXMVECROSSCATEGORY"
export BOT_EXECUTION_TRAIN_SOURCES="organic_paper,live_real,retroactive_synthetic"
export BOT_EXECUTION_SUPPLEMENTAL_PATHS="var/features/execution/execution_training_retroactive.jsonl"

# ── Policy ────────────────────────────────────────────────────────────────────
# Shadow until promoted. See CLAUDE.md for promotion criteria.
export BOT_POLICY_MODE="shadow"
export BOT_POLICY_MIN_EXPECTED_REALIZED_PNL="0.0"
export BOT_POLICY_MAX_ACTIONS_PER_CANDIDATE="4"
export BOT_POLICY_DEFAULT_LEGACY_FALLBACK="true"

# ── Periodic background tasks ─────────────────────────────────────────────────
export BOT_OUTCOME_LOOKBACK_DAYS="14"

# ── Replay (disabled) ─────────────────────────────────────────────────────────
export BOT_REPLAY_DAYS="3"
export BOT_REPLAY_CYCLES_PER_DAY="144"
export BOT_REPLAY_BANKROLL="10000"

echo "Starting Production Mode (live execution, shadow policy)..."
echo "Claude: on_heuristic_candidates. Weather: GEFS sidecar. Policy: shadow."

run_cargo_bot
