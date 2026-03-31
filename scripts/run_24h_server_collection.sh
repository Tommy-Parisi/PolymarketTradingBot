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

# --- IMMUNITY: Override .env settings that block collection ---
export BOT_CLAUDE_VERTICAL_BLOCKLIST="" # ALLOW Claude to value all verticals
export BOT_CLAUDE_TRIGGER_MODE="on_heuristic_candidates"
export BOT_VALUATION_CACHE_TTL_SECS="60"

# --- Execution & Capture Configuration ---
export BOT_EXECUTION_MODE="paper"
export BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY="true"
export BOT_CARGO_PROFILE="release"
export BOT_CYCLE_SECONDS="600" 

# --- Claude Cost Management (~$3/day target) ---
export BOT_VALUATION_MARKETS="30"      
export BOT_CLAUDE_EVERY_N_CYCLES="1"   

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
