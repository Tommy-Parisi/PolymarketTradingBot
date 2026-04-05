# Motorcade Plan: Sidecar-Heavy Architecture Roadmap

## What Changed

The original `execution_aware_prediction_plan.md` assumed a two-brain system: one general forecast GBT covering all verticals, plus one execution model. That approach failed for forecasting — a single cross-vertical model trained on market features has nothing to learn beyond market mid when enrichment signals are null.

The better architecture is a **motorcade**: the Rust bot runs in the center, flanked by a growing convoy of out-of-process Python specialist sidecars, each purpose-built for one vertical's data sources and semantics. The execution model is shared across verticals because microstructure behavior generalizes. Forecast does not.

## Current State (2026-04-05)

| Component | Status |
|-----------|--------|
| Data infrastructure (market state, order lifecycle, outcomes, feature builder) | Done |
| Weather specialist sidecar (`KXHIGHPHI-*`, AUC 0.9959) | **THIS MIGHT BE MISLEADING AND SHOULD MAKE US NERVOUS --> INFLATED, SEE ../WeatherPredictor/next_steps.md before expanding to other cities** |
| General forecast GBT (`xgb_v1.ubj`) | **Deleted** — BSS -2.64, architecture flawed; replaced by sidecar pattern |
| Execution model | Bucket lookup table — no GBT yet |
| Policy layer | Wired in shadow mode |
| Shadow mode infrastructure | Done |

The bot is running in shadow. Legacy path executes. Specialist prob overrides bucket model for weather markets. Everything else falls back to bucket.

---

## Architecture 

```
┌─────────────────────────────────────────────────────┐
│                   Rust Bot (main)                    │
│  scan → enrich → forecast → policy → execute         │
│                    │                                 │
│         specialist_prob_yes (Option<f64>)            │
│         overrides bucket model when present          │
└──────────────┬──────────────────────────────────────┘
               │  HTTP (3s timeout, graceful fallback)
    ┌──────────┴──────────────────────────┐
    │                                     │
┌───▼────────────────┐     ┌─────────────▼──────────────┐
│  WeatherPredictor  │     │   CryptoPredictor (next)    │
│  KXHIGHPHI-*       │     │   KXBTCD-*, KXETHD-*        │
│  NOAA → XGBoost    │     │   Exchange API → XGBoost    │
│  AUC 0.9959  ✓     │     │   (to build)                │
└────────────────────┘     └─────────────────────────────┘
```

**Rules:**
- Each sidecar is a FastAPI service with `/health` and `/predict?ticker=` endpoints
- `src/data/market_enrichment.rs` detects vertical from ticker, calls appropriate sidecar
- `src/models/forecast.rs` uses `specialist_prob_yes` as a hard override when present
- Sidecar down = silent fallback to bucket model, trading continues
- New sidecars require: ticker parser, data fetcher, feature generator, trained XGBoost, `sidecar.py`

**Sidecar response contract:**

All sidecars must return:
```json
{
  "probability": 0.62,
  "data_age_secs": 180,
  "data_source_ok": true,
  "model_version": "v2"
}
```

`fetch_specialist_prob` in `market_enrichment.rs` suppresses the result to `None` if `data_source_ok == false` or `data_age_secs` exceeds a per-vertical threshold. This ensures "sidecar up but returning garbage" (stale model, blocked scraper, bad data pull) collapses to `None` rather than silently overriding the forecast. Each vertical sets its own staleness threshold to match its data cadence (e.g. weather: 2h, crypto: 5m, FRED: 48h).

This is additive — the existing weather sidecar just needs `data_source_ok: true` and `data_age_secs` added to its response.

---

## Tasks to complete incrementally 

**ADD MORE CITIES TO THE WEATHER PREDICTOR**

ASK about this:
```
I built a model that trades the weather on Polymarket and CalSheet and it makes me tens of thousands of dollars every single month. I'm going to show you exactly how it works and how you can use it too. First, we pull forecasts from four of the biggest weather supercomputers in the world. These are the four you see on the screen right now. Each one runs dozens of simulations giving us 169 independent temperature predictions. Second, we correct the bias. If the US model always runs two degrees too hot for New York and winter, we subtract that out. These agencies spend billions of dollars on their forecasts. We don't try to outpredict them. We find patterns, then clean the data and find where the market gets it wrong. Third, we calculate the votes.

Each one of those 169 corrected forecasts is a vote for what the high will be. If 101 out of 169 forecasts land on 29 degrees, that's a 60% probability. If Polymarket has it at a 78% chance, that 18% difference is our edge. And the result? A 72% average win rate. If you want to use all of our free Polymarket and CalSheet models, including sports, crypto, and more, search my username in your internet browser or check out my profile.
```

## Roadmap

### Phase 1 — Crypto Specialist Sidecar

**Goal:** Cover `KXBTCD-*`, `KXETHD-*` and any similar threshold-crossing crypto markets.

**Reference:** [khuangaf/CryptocurrencyPrediction](https://github.com/khuangaf/CryptocurrencyPrediction) — well-starred multi-coin deep learning repo covering LSTM/RNN architectures. Use as a base for the sequence model; adapt to threshold-crossing binary classification rather than price regression.

**What to build** (`../kalshi_stack/CryptoPredictor/`, mirrors WeatherPredictor structure):
- Exchange price fetcher (Binance/Coinbase REST, no auth required)
- Feature generator: current price vs. threshold distance, rolling realized vol (1h/4h/24h), price momentum, funding rate, open interest delta, time to close
- XGBoost binary classifier as the initial model (will price close above threshold at market resolution?); LSTM signal from khuangaf as a secondary feature input once validated
- FastAPI sidecar with `/health` and `/predict?ticker=` endpoints
- Ticker parser: `KXBTCD-26APR15-T95000` → asset=BTC, threshold=95000, date=Apr 15

**Enrichment wiring** (Rust side):
- Add crypto ticker detection in `src/data/market_enrichment.rs`
- Add `CRYPTO_SPECIALIST_URL` env var alongside `WEATHER_SPECIALIST_URL`

**Gate:** Sidecar runs in shadow for 1 week. Brier skill score positive on holdout. Predictions logged alongside bucket model for comparison.

---

### Phase 2 — Execution Model: Fix Training Data

Before training an execution GBT, the training data has three known gaps that must be fixed.

**Gap 1 — `raw_edge_pct` / `confidence` are 0.0 in all retroactive rows**
- Retroactive label generator needs access to forecast output at snapshot time
- Fix: during dataset rebuild, join market state snapshots to the forecast feature rows by ticker + timestamp, populate `raw_edge_pct` and `confidence` from the closest preceding forecast row

**Gap 2 — No external fill data**
Priority order:
1. **Kalshi public trade history**: walk historical markets via `/trades` endpoint, reconstruct fill events with surrounding market state features. Real exchange fills, straightforward API work.
2. **Polymarket data**: binary outcome markets on Polygon, fully public, structurally identical (0–100¢, spreads, IOC fills). Thousands of resolved markets. Build an importer that maps Polymarket fills to our `ExecutionTrainingRow` schema.

**Gap 3 — Book depth missing**
- `yes_bid_size` / `yes_ask_size` not collected in market state snapshots
- Add to `MarketStateEvent` schema and populate from scanner/WS delta
- Re-run retroactive label generation after a few weeks of richer snapshots

**Gate:** All three gaps closed AND dataset has >10K rows with non-zero `raw_edge_pct`, plus real fill rows from at least one external source. Gap 2 (real fills) and Gap 3 (book depth) must both be resolved — hitting the row count while either gap remains open does not pass this gate.

---

### Phase 3 — Execution GBT: Train and Wire In

**Offline training** (`scripts/train_execution_gbt.py` — to create, mirrors `train_forecast_gbt.py`):

Target labels (already in `ExecutionTrainingRow`):
- `label_filled_within_30s` — primary fill target
- `label_filled_within_5m` — secondary
- `label_markout_bps_5m` / `label_markout_bps_30m` — adverse selection

Key features:
- `aggressiveness_bps` (price relative to spread)
- `spread_cents`, `book_pressure`
- `yes_bid_size`, `yes_ask_size` (once collected)
- `raw_edge_pct`, `confidence` (once fixed)
- `time_to_close_secs`, `volume`, `vertical`

Train IOC fill model first. GTC is a separate problem — defer until IOC model validates.

**Rust wiring** (`src/models/execution.rs`):
- Load XGBoost `.ubj` artifact at startup
- Replace `empirical_execution_baseline` bucket lookup with GBT inference
- Output `ExecutionEstimate` with fill probabilities and markout estimates

**Gate:** Execution GBT improves fill probability ranking vs. bucket lookup on holdout. Runs in shadow for 1 week without degrading legacy PnL metrics.

---

### Phase 4 — Policy Layer: End-to-End Connection

Currently the policy layer runs in shadow but does not have real specialist probs or execution estimates feeding into it. This phase wires the full chain.

**What to connect:**
- `specialist_prob_yes` from active sidecars → `ForecastOutput.fair_prob_yes`
- `ExecutionEstimate.fill_prob_30s` / `markout_bps_5m` → `PolicyDecision.expected_realized_pnl`
- Policy scores: `IOC at ask`, `IOC at ask+1`, `GTC at bid+1`, `skip` — chooses highest positive expected realized PnL

**Expected realized PnL formula (per candidate action):**
```
erpnl = (fair_prob - fill_price) * fill_prob - fee - adverse_selection_cost
```

**Gate:** Policy shadow logs show positive mean `expected_realized_pnl` on markets where specialist is active. Decision delta vs. legacy is measurable and not obviously worse.

---

### Phase 5 — Shadow Validation and Active Mode Promotion

The existing shadow→active gate is already code-enforced:
- ≥50 shadow policy records in last 7 days
- Mean `expected_realized_pnl` ≥ -200 bps

Extend the validation checklist before promoting:
- Forecast specialist coverage: what % of traded markets had a specialist active?
- Execution model fill prediction accuracy: predicted fill rate vs. actual fill rate
- Markout calibration: predicted adverse selection vs. observed
- Per-vertical breakdown: no vertical should be materially worse than legacy

**Promotion path:**
1. Shadow with real specialist probs (Phase 4 complete)
2. Active with strict notional cap (`BOT_POLICY_MAX_NOTIONAL_ACTIVE`)
3. Active at normal sizing after 2 weeks of cap-mode data

---

### Phase 6 — Additional Specialists (Later)

**Economic indicators (Fed rate, CPI)**
- Start with Fed rate markets only — thickest, cleanest structure
- Harder to beat efficient market without proprietary data edge — only build if there's a clear signal hypothesis

Two reference implementations identified:

**hawkwatchers** ([jgdenby/hawkwatchers](https://github.com/jgdenby/hawkwatchers)) — NLP classification on Fed press release text (Naive Bayes, Logistic Regression, SVM, Decision Trees, trained on releases back to 1994). Lighter weight; good as a quick signal layer that can be stood up quickly.

**FOMC multi-agent system** ([chirindaopensource/multi_agent_system_architecture_for_federal_funds_target_rate_prediction](https://github.com/chirindaopensource/multi_agent_system_architecture_for_federal_funds_target_rate_prediction)) — more rigorous. Based on a solid paper; repo is scaffolding, not a validated tool. The parts worth adapting: FRED macroeconomic indicator ingestion pipeline, CoD prompt engineering, and meeting packet construction with proper information cutoffs. Approach: fork it, replace `generate_synthetic_datasets()` with real FRED API calls, validate output against a few known FOMC meetings before trusting it for live signals.

Suggested build order: hawkwatchers as a fast first signal, FOMC multi-agent as a heavier second layer once the lighter one is validated in shadow.

**Sports**
- Data sourcing approach identified: **[vladkens/twscrape](https://github.com/vladkens/twscrape)** monitoring a curated beat reporter watchlist (~50 reporters per sport: Shams, Woj, Ian Rapoport, etc.), polling timelines every 60s, Claude classifying each tweet ("is this injury news, and which Kalshi markets does it affect?"), per-ticker probability updated on event fire
- Unblocked in principle; implementation pending
- Primary remaining risk: cycle latency (see transport note below) and per-ticker TTL correctness

**Transport: polling with internal state, but cycle latency matters**
The sports sidecar should maintain a per-ticker probability that it updates internally when a tweet or injury event fires. The Rust bot polls it on the normal cycle via `GET /predict?ticker=` and gets the sidecar's current best estimate — no new code path in `market_enrichment.rs`, no locking complexity from mid-cycle push events.

However: if the current cycle is 30s, a tweet that lands just after a poll means up to 30s before the sidecar is queried again. For a market that reprices in 2-3 minutes (e.g. Shams tweets "Tatum out" at 6:58pm, tip-off 7:00pm), that's a meaningful fraction of the edge window. Verify the actual cycle interval before assuming polling is acceptable. If the edge window is tighter than the cycle, push is the right call — a signal channel into the main loop is the complexity cost.

**Per-ticker state must have explicit TTLs**
`data_age_secs` in the response contract handles "the data source went down or returned stale data." It does not handle "the signal was valid at the time but is now wrong." A sidecar that sets `KXNBA-BOS-*` to 0.3 because Tatum is out tonight must reset that state after the game resolves — otherwise tomorrow's market inherits yesterday's injury signal with a fresh `data_age_secs`.

Each per-ticker probability entry needs a `valid_until` timestamp (e.g. game tip-off + a buffer, or market close time), not just a data freshness check. On `/predict`, if `valid_until` is in the past, return `data_source_ok: false` so the Rust side suppresses to `None`. Do not conflate data freshness with signal validity — they expire on different clocks.

---

## General Forecast GBT — Deleted

`var/models/forecast/xgb_v1.ubj` and `scripts/train_forecast_gbt.py` have been removed (2026-04-05). The architecture was fundamentally flawed: a single cross-vertical GBT has nothing to learn beyond market mid when enrichment signals are null for most rows. BSS was -2.64.

The bucket model is the permanent fallback for non-specialist verticals. As sidecar coverage expands (crypto next), the bucket model's role shrinks naturally — no general GBT needed.

---

## Production Gates Summary

| Gate | Condition |
|------|-----------|
| Crypto sidecar to shadow | BSS > 0 on holdout |
| Execution GBT to shadow | Improves fill ranking vs. bucket on holdout |
| Policy active (capped) | ≥50 shadow records, mean erpnl ≥ -200 bps, specialist coverage ≥ 50% of trades |
| Policy active (normal) | 2 weeks capped-mode data, no vertical materially worse than legacy |

---

## Out of Scope

- Multi-exchange routing
- Automatic retraining (manual batch retrain is sufficient)