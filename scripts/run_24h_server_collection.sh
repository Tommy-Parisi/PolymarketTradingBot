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

# --- Execution & Capture Configuration ---
export BOT_EXECUTION_MODE="paper"
export BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY="true"
export BOT_CARGO_PROFILE="release"
export BOT_CYCLE_SECONDS="600" # 10-minute cycles (144 per day)

# --- Claude Cost Management (~$3/day target) ---
# Cost per 1k tokens: ~$0.015 (mixed input/output). 
# At 144 cycles/day, we can afford ~1.4k tokens per cycle to hit ~$3/day.
export BOT_CLAUDE_TRIGGER_MODE="on_heuristic_candidates"
export BOT_VALUATION_MARKETS="30"      # Keep batches small to manage token count
export BOT_CLAUDE_EVERY_N_CYCLES="1"   # Check every cycle, but only pay if heuristic likes it

# --- Lowering the Bar for Data Density ---
# Trigger Claude even on small perceived edges to grow the "smart" dataset.
export BOT_MISPRICING_THRESHOLD="0.03" 
export BOT_MIN_EDGE_PCT="0.03"
export BOT_FALLBACK_MISPRICING_THRESHOLD="0.01"
export BOT_MIN_CANDIDATES="5"

# --- Scanning Strategy ---
# Focus on liquid 24/7 verticals to ensure steady data flow.
export BOT_SCAN_SERIES_ALLOWLIST="KXSILVERD,KXGOLDMON,KXBTCD,KXETHD,KXSOLD,KXXRPD,KXNASDAQ100MINY"
export BOT_SCAN_TIER2_CATEGORIES="Sports,Crypto,Financials,Climate and Weather"
export BOT_SCAN_MAX_TIER2_SERIES="30"
export BOT_ENRICHMENT_MARKETS="40"

echo "Starting 24/7 Server Collection Mode..."
echo "Targeting Claude valuations on heuristic candidates every 10m."
echo "Budgeted for ~\$3/day Claude usage."

run_cargo_bot
