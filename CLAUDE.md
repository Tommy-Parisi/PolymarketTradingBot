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
./scripts/trade.sh     # live trading, Claude enabled, shadow policy (production)
./scripts/collect.sh   # paper trading, no Claude, data collection (runs 24/7 on server)

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
- `sidecars/weather/` — GEFS 31-member ensemble for all live weather cities (`KXHIGHT{BOS,DAL,HOU,SEA,PHX,SATX,LV,ATL,MIN,NOLA,DC,SFO,OKC}`). Cache keyed by `(city, date)`. Env: `WEATHER_SPECIALIST_URL` (default `http://127.0.0.1:8765`). Start: `sidecars/weather/start.sh`.

**Shadow sidecars (do not override bucket model yet):**
- `sidecars/crypto/` — GBM threshold-crossing probability for `KXBTCD-*`, `KXETHD-*`, `KXSOLD-*`, `KXXRPD-*`. Coinbase/Binance price feed, 30s refresh. Env: `CRYPTO_SPECIALIST_URL` (default unset). Shadow only: Rust logs predictions to stderr but does not set `specialist_prob_yes`. Promote after 1-week shadow validates calibration vs bucket. Start: `sidecars/crypto/start.sh`.

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
Any new policy or model-driven logic must go through `shadow` before `active`.

**Going live** (`trade.sh`, `BOT_EXECUTION_MODE=live`) is a manual decision. Before enabling, run:
```bash
python3 scripts/simulate_pnl.py   # must show positive simulated ROI with guards applied
```
Win rate alone is misleading — buying 85-cent contracts at 70% win rate still loses money. Use `simulate_pnl.py` which applies the notional cap and computes actual dollar PnL. The legacy path (heuristic + Claude) is what executes in live+shadow mode.

**Shadow → Active** (`BOT_POLICY_MODE=active`) is code-enforced. `active` will fail at startup unless:
- Execution model has ≥ `BOT_POLICY_ACTIVE_MIN_EXECUTION_LIVE_REAL_ROWS` (default 50) live-real rows
- Mean `markout_5m_bps` across those live-real rows ≥ `BOT_POLICY_ACTIVE_MIN_LIVE_MEAN_MARKOUT_BPS` (default 0)

The gate is based on actual observed trade outcomes from live execution, not estimated ERPNL from the execution model (which has no fill variance on paper data).

### var/ is Generated — Do Not Commit
The entire `var/` tree (`cycles/`, `logs/`, `research/`, `features/`, `models/`, `state/`) is runtime output. Never commit it. Use `git rm --cached` if needed.

### No Secrets in Commits
`.env`, private keys, API tokens, account-specific credentials must never be committed.

## Known Issues (Priority Order)

See `docs/execution_aware_prediction_plan.md` for the full roadmap.

1. **Execution model fill-probability labels are meaningless on paper data.** The paper execution path always fills (`limit >= ask` is true for all candidates), so `fill_30s`/`fill_5m` labels have zero variance. Organic paper rows DO help `markout_5m_bps` and `fill_price` calibration — but fill probability requires live-real data. Active mode is gated on live-real markout, not paper row count.
2. **Bucket model underperforms market mid in shadow** (-18.3% Brier lift, 04-06 audit). Expected — enrichment signals are null for non-weather verticals. Fix: build CryptoPredictor sidecar next, which covers the highest-volume shadow rows (crypto ~96% of resolved markets).
3. **Execution GBT not started.** Execution model is still a bucket lookup table. Do not prioritize until there are 50+ live-real rows with observed fill variance.
4. **Current apparent edge is time-decay capture, not deep modeling.** Crypto and metals performance (~96-100% win rate) comes from buying near-resolution daily markets that Kalshi prices stale. This is a real but fragile and thin edge — it doesn't scale and compresses as liquidity improves. Durable alpha requires sidecar-driven probabilistic forecasts.

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
| `scripts/simulate_pnl.py` | Realistic PnL simulation with live guards applied (use before going live) |
| `sidecars/weather/sidecar.py` | Weather specialist sidecar — GEFS ensemble, multi-city |
| `sidecars/weather/gefs_fetcher.py` | NOMADS GRIB2 fetch + CityConfig dataclass |
| `sidecars/weather/ensemble_predictor.py` | Vote-fraction P(high > threshold) with bias correction table |
| `src/data/market_enrichment.rs` | Calls weather sidecar; populates `specialist_prob_yes` |

## Analysis Scripts (Python)

```bash
python3 scripts/evaluate_verticals.py     # weather + crypto sidecar W/L by vertical (--since YYYY-MM-DD, --detail)
python3 scripts/evaluate_shadow.py        # calibration + policy hit rate
python3 scripts/check_fills.py           # fill win/loss vs outcomes
python3 scripts/simulate_pnl.py          # realistic PnL with live guards (use before going live)
python scripts/retroactive_execution_labels.py   # backfill execution labels
python scripts/validate_fair_value_calibration.py
```

## Branch & Release Workflow

See `docs/release_process.md`. In brief:
- Feature branches for meaningful work; don't stack experiments on `main`
- Run `scripts/release_check.sh` before pushing anything affecting runtime safety, reporting, or rollout logic
- Commit categories: runtime changes / model+reporting changes / docs+process changes
