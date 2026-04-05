# CLAUDE.md

## Project Overview

Rust trading bot for Kalshi event-contract markets. Two parallel pipelines:
1. **Legacy trading loop** — scan, enrich, value, allocate, execute (production-ready)
2. **Research & modeling pipeline** — capture market state, train forecast/execution/policy models (maturing)

The system is transitioning from pure heuristic trading toward data-informed modeling. Shadow-first is the culture: new logic runs in parallel before being trusted.

## Build & Run

```bash
# Build
cargo build --release

# Run (uses .env for all config)
./scripts/run_live_shadow.sh          # live trading + shadow models (recommended)
./scripts/run_research_paper_capture.sh   # paper trading + research capture
./scripts/run_research_capture.sh         # data collection only (no trading)

# Retrain models after new data
./scripts/rebuild_models.sh

# Morning routine (backfill outcomes, rebuild, check fills, evaluate)
./scripts/morning_review.sh

# Pre-release safety check
./scripts/release_check.sh
```


## Architecture

```
src/
├── main.rs            # Runtime orchestration, cycle loop, mode selection
├── data/              # Scanner, websocket delta merge, enrichment (weather/sports/crypto)
├── execution/         # Kalshi API client, paper simulator, execution engine
├── model/             # Legacy valuation (Claude + heuristic fallback), allocator
├── models/            # Forecast model, execution model, reporting
├── research/          # Market-state recording, order-lifecycle logging, outcome backfill
├── datasets/          # Training dataset builder (joins market state + outcomes)
├── features/          # Feature builders for forecast and execution models
├── policy/            # Policy layer (scores action grid, expected PnL)
├── outcomes/          # Resolved outcome backfill from Kalshi API
├── markets/           # Market mapper, market-type helpers
└── replay/            # Multi-day replay/backtesting
```

### Specialist Sidecar Architecture

Forecast is handled by a **motorcade of out-of-process Python/FastAPI sidecars** — one per vertical, each purpose-built for its data sources and semantics. The Rust bot falls back to the bucket model for any vertical not yet covered by a sidecar.

**Pattern:** `src/data/market_enrichment.rs` detects vertical from ticker, calls the appropriate sidecar, and populates `ForecastFeatureRow.specialist_prob_yes`. `src/models/forecast.rs` uses this as a **hard override** of the bucket model (tagged `_specialist` in model version). Sidecar down = silent fallback to bucket, trading continues.

**Active sidecars:**
- `../kalshi_stack/WeatherPredictor/` — XGBoost (AUC 0.9959) for `KXHIGHPHI-*`. Fetches NOAA data. Env: `WEATHER_SPECIALIST_URL`, `NOAA_API_TOKEN`.

**Next to build:** CryptoPredictor (`KXBTCD-*`, `KXETHD-*`) — price distance, rolling vol, momentum from exchange APIs.

**Operating modes** (set via `BOT_POLICY_MODE` in `.env`):
- `legacy` — only legacy path (current trusted mode)
- `shadow` — legacy executes, models run in parallel for comparison
- `active` — policy decisions influence trading (requires validated execution data)

## Key Conventions

### Data Provenance — Never Merge These
Training data must preserve source labels:
- `bootstrap_synthetic` — retroactive artificial bootstrap
- `organic_paper` — paper trading
- `live_real` — real exchange fills

Models default to `organic_paper + live_real` only. Do not silently merge.

### Shadow-First Rollout
Any new policy or model-driven logic must go through `shadow` before `active`. The shadow→active promotion is now partly code-enforced: `BOT_POLICY_MODE=active` will fail at startup if fewer than `BOT_POLICY_ACTIVE_MIN_SHADOW_DECISIONS` (default 50) shadow policy records exist in the last `BOT_POLICY_ACTIVE_SHADOW_LOOKBACK_DAYS` (default 7) days, or if their mean `expected_realized_pnl` is below `BOT_POLICY_ACTIVE_MIN_SHADOW_MEAN_ERPNL` (default -200 bps).

### var/ is Generated — Do Not Commit
The entire `var/` tree (`cycles/`, `logs/`, `research/`, `features/`, `models/`, `state/`) is runtime output. Never commit it. Use `git rm --cached` if needed.

### No Secrets in Commits
`.env`, private keys, API tokens, account-specific credentials must never be committed.

## Known Issues (Priority Order)

See `docs/execution_aware_prediction_plan.md` for the full roadmap.

1. **Execution model is almost entirely synthetic data.** 1,215 organic paper rows vs ~1.16M retroactive synthetic. Accumulate more organic fills before trusting execution model predictions. Active mode remains disabled.
2. **Bucket model underperforms market mid in shadow** (-15.3% Brier lift, 04-05 audit). This is expected — enrichment signals are null for non-weather verticals. Fix: build CryptoPredictor sidecar next, which will cover the highest-volume shadow rows.
3. **Execution GBT not started.** Execution model is still a bucket lookup table. Do not prioritize until organic execution data is sufficient (target: ~5K organic rows).

## Important Files

| File | Purpose |
|------|---------|
| `src/main.rs` | Cycle orchestration, mode logic, entry point |
| `src/model/allocator.rs` | Capital allocation (Kelly-like sizing) |
| `src/models/forecast.rs` | Forecast model inference |
| `src/models/execution.rs` | Execution model inference |
| `src/policy/` | Policy layer — scores action grid |
| `src/datasets/builder.rs` | Training dataset builder |
| `src/outcomes/resolver.rs` | Outcome backfill |
| `docs/execution_aware_prediction_plan.md` | Full modeling roadmap |
| `scripts/evaluate_shadow.py` | Forecast calibration + policy hit-rate analysis |
| `scripts/check_fills.py` | Paper fill win/loss rate vs resolved outcomes |
| `../kalshi_stack/WeatherPredictor/sidecar.py` | Weather specialist sidecar — FastAPI service exposing XGBoost via HTTP |
| `../kalshi_stack/WeatherPredictor/src/modeling/train_weather_model.py` | Offline training for weather specialist model |
| `src/data/market_enrichment.rs` | Calls weather sidecar; populates `specialist_prob_yes` |

## Analysis Scripts (Python)

```bash
python scripts/evaluate_shadow.py        # calibration + policy hit rate
python scripts/check_fills.py           # fill win/loss vs outcomes
python scripts/retroactive_execution_labels.py   # backfill execution labels
python scripts/validate_fair_value_calibration.py
```

## Branch & Release Workflow

See `docs/release_process.md`. In brief:
- Feature branches for meaningful work; don't stack experiments on `main`
- Run `scripts/release_check.sh` before pushing anything affecting runtime safety, reporting, or rollout logic
- Commit categories: runtime changes / model+reporting changes / docs+process changes
