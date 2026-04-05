# Option 2 Implementation Plan: Execution-Aware Prediction and Trading System

## Summary

Build a two-brain system on top of the existing Rust trading loop:

1. **Event Forecast Brain**
   - Predicts terminal fair probability for a market outcome.

2. **Execution Brain**
   - Predicts whether the edge is actually tradable after fill probability, realized execution price, queue/adverse selection, and time-in-force effects.

The final trading decision will optimize **expected realized PnL**, not theoretical mispricing versus market mid.

This plan keeps the current scanner, orchestration loop, allocator shell, journaling, and exchange client boundaries, but expands the project into a research/training/inference platform roughly twice the current complexity.

The first implementation step will be **data foundation and label capture**, because every later model depends on it.

## System Goal

Replace the current valuation flow with a production-oriented pipeline that answers:

- What is the true probability of the event?
- If we place an order now, what is the expected fill behavior?
- What is the expected realized PnL net of fees, slippage, and adverse selection?
- Which order policy (`IOC`, `GTC`, `hybrid`), price, and size maximize expected realized value under current risk constraints?

Success means the bot chooses trades based on expected realized value rather than raw probability edge and can be evaluated honestly on live-fill outcomes.

## Scope

### In Scope

- Historical data capture for market state, order lifecycle, and outcomes
- Feature store for forecast and execution features
- Offline training datasets and label generation
- Event forecast model
- Execution model
- Policy layer to combine both
- Backtest and replay framework upgrades
- Online inference integration into the Rust bot
- Monitoring, calibration, and drift checks
- Safe rollout path with shadow mode and gated activation

### Out of Scope

- Multi-exchange routing
- Full market-making / continuous quoting engine
- GPU-heavy deep learning infrastructure
- Human-in-the-loop discretionary UI
- Automatic retraining infrastructure beyond scheduled batch retrains
- Cross-venue arbitrage

## Target Architecture

### Existing Components to Reuse

- Scan/orchestration loop in `src/main.rs`
- Scanner in `src/data/market_scanner.rs`
- WS delta ingestion in `src/data/ws_delta.rs`
- Execution engine in `src/execution/engine.rs`
- Exchange adapter in `src/execution/client.rs`
- Replay harness in `src/replay/mod.rs`

### New Top-Level Components

1. **Market State Recorder**
   - Captures market snapshots and short-horizon microstructure data for each candidate market over time.

2. **Order Lifecycle Recorder**
   - Captures every submitted order and all later order-state transitions needed to build execution labels.

3. **Outcome Resolver**
   - Maps expired markets to final resolution labels for forecast-model training.

4. **Feature Store Builder**
   - Builds typed, reproducible feature rows for:
   - event forecasting
   - execution prediction
   - realized-PnL policy evaluation

5. **Offline Modeling Pipeline**
   - Trains and evaluates:
   - event forecast model
   - fill probability model
   - expected fill price model
   - adverse selection / markout model
   - order-policy chooser

6. **Online Inference Layer**
   - Loads model artifacts, computes features in-process, and emits structured trade decisions.

7. **Policy Engine**
   - Selects:
   - whether to trade
   - which side
   - limit price
   - time in force
   - size adjustment
   - expected realized value

## New Public Interfaces and Contracts

### Replace Current Valuation Contract

Current implicit output:

- `fair_prob_yes`
- `confidence`
- `rationale`

New canonical contract:

```rust
pub struct ForecastOutput {
    pub ticker: String,
    pub fair_prob_yes: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub model_version: String,
    pub feature_ts: DateTime<Utc>,
}
```

```rust
pub struct ExecutionEstimate {
    pub ticker: String,
    pub outcome_id: String,
    pub side: Side,
    pub tif: TimeInForce,
    pub candidate_limit_price: f64,
    pub fill_prob_30s: f64,
    pub fill_prob_5m: f64,
    pub expected_fill_price: f64,
    pub expected_slippage_bps: f64,
    pub expected_markout_bps_5m: f64,
    pub expected_markout_bps_30m: f64,
    pub model_version: String,
}
```

```rust
pub struct PolicyDecision {
    pub ticker: String,
    pub outcome_id: String,
    pub side: Side,
    pub should_trade: bool,
    pub limit_price: f64,
    pub time_in_force: TimeInForce,
    pub size_multiplier: f64,
    pub expected_gross_edge: f64,
    pub expected_realized_pnl: f64,
    pub expected_fill_prob: f64,
    pub rejection_reason: Option<String>,
    pub rationale: String,
}
```

### New Runtime Boundary

Add a new module family:

- `src/research/`
- `src/features/`
- `src/models/`
- `src/policy/`
- `src/outcomes/`

### Module Responsibilities

- `src/research/market_recorder.rs`
  - persist snapshot and microstructure observations

- `src/research/order_recorder.rs`
  - persist order lifecycle and reconciliation events

- `src/outcomes/resolver.rs`
  - fetch/derive final market outcome

- `src/features/forecast.rs`
  - build forecast features

- `src/features/execution.rs`
  - build execution features

- `src/models/forecast.rs`
  - online inference wrapper for forecast model artifact

- `src/models/execution.rs`
  - online inference wrapper for execution model artifact

- `src/policy/decision.rs`
  - combine outputs into a trade/no-trade action

## New Artifact Files

Use stable, explicit file layout:

- `var/research/market_state/YYYY-MM-DD/*.jsonl`
- `var/research/order_lifecycle/YYYY-MM-DD/*.jsonl`
- `var/research/outcomes/*.jsonl`
- `var/features/forecast/*.parquet`
- `var/features/execution/*.parquet`
- `var/models/forecast/<version>/`
- `var/models/execution/<version>/`
- `var/backtests/<run_id>/`
- `var/shadow/<run_id>/`

Parquet is the default for feature tables. JSONL remains acceptable for raw event capture.

## Implementation Breakdown

### Phase 0: Freeze the Current Baseline

#### Objective

Create a trustworthy baseline so improvements are measured against the current system.

#### Tasks

1. Define current baseline metrics:
   - candidate count per cycle
   - allocation count per cycle
   - acceptance rate
   - fill rate
   - cancel rate
   - expected edge PnL net fees
   - realized PnL where measurable
2. Extend replay/summary reporting to save a baseline report artifact.
3. Document current assumptions:
   - valuation uses mid price
   - cost assumptions are static
   - execution policy does not optimize expected realized value

#### Acceptance

- One baseline report can be generated from current logs and replay.
- Baseline metrics are stored for comparison with new models.

### Phase 1: Data Foundation and Label Capture

#### Objective

Collect enough raw truth to train forecast and execution models.

#### Tasks

1. Introduce a market-state recorder:
   - capture snapshot scan fields
   - capture WS deltas per ticker during each listen window
   - persist quote/mid/spread/volume observations with timestamps
2. Introduce an order-lifecycle recorder:
   - record order intent
   - record ack
   - record each reconciliation state change
   - record cancel events
   - record terminal status
3. Normalize current journal events into stable schemas.
4. Add a research event writer separate from the current trade journal.
5. Add market-resolution metadata capture:
   - ticker
   - title
   - subtitle if available
   - close time
   - market type
   - event/series IDs if available
6. Add environment-controlled sampling knobs for data volume management.

#### New Schemas

##### MarketStateEvent

```rust
pub struct MarketStateEvent {
    pub ts: DateTime<Utc>,
    pub ticker: String,
    pub title: String,
    pub yes_bid_cents: Option<f64>,
    pub yes_ask_cents: Option<f64>,
    pub mid_prob_yes: Option<f64>,
    pub spread_cents: Option<f64>,
    pub volume: f64,
    pub traded_count_delta: Option<f64>,
    pub source: String, // snapshot | ws_delta
    pub cycle_id: String,
}
```

##### OrderLifecycleEvent

```rust
pub struct OrderLifecycleEvent {
    pub ts: DateTime<Utc>,
    pub client_order_id: String,
    pub order_id: Option<String>,
    pub ticker: String,
    pub outcome_id: String,
    pub side: Side,
    pub tif: TimeInForce,
    pub limit_price: Option<f64>,
    pub requested_qty: f64,
    pub filled_qty: f64,
    pub avg_fill_price: Option<f64>,
    pub status: OrderStatus,
    pub event_type: String, // intent | ack | reconcile | cancel | terminal
}
```

#### Acceptance

- Bot can run normally while writing new research logs.
- No change yet to trading decisions.
- At least one day of raw market-state and order-lifecycle data can be collected losslessly.

### Phase 2: Outcome Resolution Pipeline

#### Objective

Generate final truth labels for the event forecast model.

#### Tasks

1. Add a historical market resolver that polls or backfills final market outcomes.
2. Persist one outcome record per settled ticker.
3. Handle unresolved/void/canceled markets explicitly.
4. Create a joinable mapping:
   - ticker -> final outcome
   - settlement timestamp
   - resolution source
5. Add a backfill command for old research data.

#### Outcome Schema

```rust
pub struct MarketOutcomeRecord {
    pub ticker: String,
    pub resolved_at: DateTime<Utc>,
    pub outcome_yes: Option<bool>,
    pub resolution_status: String, // resolved | canceled | unresolved
    pub source: String,
}
```

#### Acceptance

- Forecast features can be labeled with final market outcomes.
- Canceled/void markets are excluded cleanly from supervised training.

### Phase 3: Feature Specification

#### Objective

Define all model inputs before building models.

#### Forecast Feature Families

1. Market state
   - bid, ask, spread, mid
   - recent spread stability
   - volume
   - recent trade activity
2. Time features
   - time to close
   - hour/day/seasonality
3. Parsed market semantics
   - vertical
   - city/team/asset
   - threshold/line extracted from ticker/title
4. External context
   - NOAA weather forecasts and forecast revisions
   - sports injuries/rest/travel
   - crypto sentiment / price regime
5. Market behavior
   - price drift over recent windows
   - spread compression/expansion
   - quote refresh frequency

#### Execution Feature Families

1. Order parameters
   - side
   - tif
   - candidate price
   - relative aggressiveness to best bid/ask
2. Book context
   - spread
   - recent quote movement
   - recent trades
   - local volatility
3. Market context
   - volume/liquidity bucket
   - time to close
   - vertical
4. Signal context
   - raw forecast edge
   - calibrated edge
   - confidence
5. Lifecycle context
   - whether we already have open orders
   - recent fills in same ticker
   - same-event exposure

#### Acceptance

- A written feature catalog exists with field names, types, null handling, and derivation formulas.
- No model training starts before feature spec is frozen.

### Phase 4: Offline Dataset Builder

#### Objective

Turn raw logs into reproducible training tables.

#### Tasks

1. Add feature extraction jobs:
   - raw -> forecast feature rows
   - raw -> execution feature rows
2. Define labels:

##### Forecast labels

- final outcome yes/no

##### Execution labels

- filled within 30s
- filled within 5m
- realized fill price
- markout after 5m
- markout after 30m
- canceled vs terminal filled
- realized net PnL

3. Add train/validation/test split strategy:
   - strictly time-based
   - no random shuffle
4. Add leakage prevention checks.
5. Write datasets to parquet.

#### Acceptance

- One command can build forecast and execution datasets from research logs.
- Splits are reproducible and leakage-checked.

## Vertical Specialist Model Strategy

Rather than a single general forecast GBT (Issue 1), the preferred architecture is **per-vertical out-of-process specialist sidecars** — each an independently trained XGBoost model served via FastAPI, called by the enrichment layer and overriding the bucket model via `specialist_prob_yes`. The weather sidecar proves the pattern works (AUC 0.9959).

### Roadmap

**Priority 1 — Crypto (next to build)**
- Targets: `KXBTCD-*`, `KXETHD-*` and similar threshold-crossing markets
- Data: Binance/Coinbase REST API (free, no auth required)
- Key features: current price vs. threshold distance, rolling realized vol (1h/4h/24h), momentum, funding rate, open interest delta, time to close
- Architecture: identical to WeatherPredictor; swap NOAA downloader for exchange price fetcher

**Priority 2 — Economic indicators**
- Start with Fed rate markets only (cleanest, thickest)
- Data: CME FedWatch (implied rates), FRED API (prior prints)
- Signal: consensus drift from last print + futures curve vs. market mid
- Harder to beat an efficient market here without a proprietary data edge

**Priority 3 — Sports**
- NFL/NBA game outcome markets
- Blocked until a reliable, timely injury feed is identified — without it the model just replicates market mid
- Do not prioritize until data sourcing is solved

### General Forecast GBT — Deleted (2026-04-05)
The general XGBoost (`var/models/forecast/xgb_v1.ubj`) and its training script have been removed. BSS -2.64; architecture flawed — a cross-vertical GBT cannot learn signal when enrichment is null for most rows. The bucket model is the permanent fallback for non-specialist verticals.

---

### Completed: Weather Vertical Specialist (2026-04-04)

The Philadelphia high-temperature vertical (`KXHIGHPHI-*`) is fully served by the **weather specialist sidecar** (`../kalshi_stack/WeatherPredictor/`) — an out-of-process XGBoost model (AUC 0.9959) that bypasses the bucket model. This is the first realized instance of the per-vertical specialist model pattern. Sports, crypto, and global fallback verticals still use the bucket model.

---

### Phase 5: Event Forecast Model

#### Objective

Predict terminal event probability better than current heuristic/mid-based logic.

#### Model Choice

Default:

- gradient boosted trees for tabular features
- separate model per vertical when enough data exists
- fallback global model otherwise

#### Tasks

1. Train baseline models:
   - weather
   - sports
   - crypto
   - global fallback
2. Add calibration:
   - isotonic or Platt scaling
3. Produce uncertainty proxy:
   - ensemble variance or calibration-derived confidence
4. Save artifacts with metadata:
   - feature schema hash
   - train window
   - validation metrics
   - model version

#### Metrics

- log loss
- Brier score
- calibration curve
- lift over market-mid baseline

#### Acceptance

- Forecast model beats market-mid baseline on at least one vertical and is not materially worse on the others.
- Model artifact can be loaded in Rust inference wrapper.

### Execution Model — Training Data Design

#### How synthetic labels work

`scripts/retroactive_execution_labels.py` takes collected market state snapshots and simulates IOC orders at 8 price offsets around the ask (`[-10, -5, -2, -1, 0, +2, +5, +10]` cents). The fill label is **mechanically deterministic**: `limit >= ask` → filled at ask. This is exact IOC semantics on Kalshi — no approximation. Filled rows get markout computed from subsequent snapshots (mid at T+5m, T+30m).

This makes the execution model fundamentally more tractable than the forecast GBT: microstructure behavior (fill probability, markout) is largely vertical-agnostic, so cross-vertical training is correct and synthetic labels are structurally sound.

#### Known gaps in current synthetic data

- `raw_edge_pct` and `confidence` are hardcoded to `0.0` in all retroactive rows — the model can't learn "is this edge worth taking," only "will it fill." Fix: repopulate from forecast signal at snapshot time during dataset rebuild.
- No GTC representation — every synthetic row is IOC YES-buy. Resting order fill probability requires queue depth and order refresh features.
- `yes_bid_size` / `yes_ask_size` not populated in retroactive rows — model can't learn book depth effects on fill.
- Markout via mid is noisy in thin markets; underestimates adverse selection where bid/ask gaps.

#### External data sources (priority order)

1. **Kalshi public trade history** — `/trades` endpoint returns real public fills per market. Walk historical markets, pair with collected market state snapshots to get real fill events with surrounding features. Straightforward API work.
2. **Polymarket historical data** — binary outcome markets on Polygon blockchain, fully public. Similar 0–100¢ pricing and spread dynamics. Thousands of resolved markets with real fill data. Microstructure features transfer almost directly. Highest volume of real execution data available.
3. **Own paper fills** — highest quality (real `raw_edge_pct`, real `confidence`), currently ~1.4K organic rows vs 246K synthetic. Grows with time; accelerate by running paper mode on more markets.

#### Priority path

1. Fix `raw_edge_pct`/`confidence` in retroactive generation
2. Add Kalshi public trade history scraping
3. Build Polymarket data importer
4. GTC modeling — only after IOC model is validated (more complex queue mechanics)

---

### Phase 6: Execution Model

#### Objective

Predict whether an apparent edge is monetizable.

#### Submodels

1. Fill probability model
   - probability of fill within 30s
   - probability of fill within 5m
2. Fill price model
   - expected fill price conditional on fill
3. Adverse selection / markout model
   - expected price move after fill over 5m and 30m
4. Cancel/no-fill model
   - chance resting order never fills before policy timeout

#### Model Choice

Default:

- gradient boosted trees for all submodels
- one shared execution feature builder
- model family can differ by target type

#### Labels

- binary fill targets
- regression target for fill price
- regression target for post-fill markout

#### Metrics

- AUC / PR-AUC for fill
- MAE for fill price
- MAE / signed error for markout
- realized-value ranking correlation

#### Acceptance

- Model can rank order opportunities better than raw spread or raw edge alone.
- Execution artifact loads in Rust and scores within latency budget.

### Phase 7: Policy Layer

#### Objective

Turn forecast + execution outputs into the actual trade decision.

#### Policy Formula

For each candidate action:

- estimate fair value
- estimate expected fill probability
- estimate expected fill price
- estimate adverse-selection cost
- subtract fees and slippage
- compute expected realized PnL
- choose action with highest positive expected realized PnL

#### Candidate action grid

For each trade candidate, score:

- `IOC` at aggressive price
- `GTC` at current quote
- `GTC` one tick less aggressive
- `hybrid` with configurable IOC fraction
- `skip`

#### Required Outputs

- `should_trade`
- `tif`
- `limit_price`
- `size_multiplier`
- `expected_realized_pnl`
- `expected_fill_prob`
- `rationale`

#### Acceptance

- Policy output is deterministic given models and features.
- `skip` is chosen for low-conviction or execution-toxic setups.

### Phase 8: Integrate Online Inference into the Bot

#### Objective

Swap current valuation-only decisioning for forecast + execution + policy.

#### Tasks

1. Add online feature builders in Rust.
2. Add model artifact loaders.
3. Add inference wrappers:
   - forecast inference
   - execution inference
4. Replace `generate_candidates()` path with:
   - forecast outputs
   - candidate action generation
   - execution scoring
   - policy decision
5. Keep existing risk guards after policy layer.

#### Runtime Flow After Integration

1. scan
2. enrich
3. build forecast features
4. score forecast model
5. generate trade candidates
6. build execution features per candidate action
7. score execution models
8. choose best action
9. apply allocator/risk limits
10. execute

#### Acceptance

- Bot can run in shadow mode without placing changed trades.
- Existing journal/state formats remain backwards compatible.

### Phase 9: Upgrade Allocator for Realized-Value Sizing

#### Objective

Size on expected realized value rather than raw edge.

#### Changes

1. Replace current `edge * confidence` ranking with:
   - expected realized PnL
   - risk-adjusted expected return
2. Replace Kelly inputs:
   - use calibrated fair price and expected execution price
3. Add a fill-adjusted size floor:
   - low fill probability reduces capital allocation
4. Add policy-aware exposure logic:
   - more conservative sizing on long-dated or illiquid markets

#### Acceptance

- Allocation respects existing bankroll caps and risk limits.
- Sizing decisions become sensitive to fillability, not just theoretical edge.

### Phase 10: Backtesting and Replay Overhaul

#### Objective

Evaluate the full system honestly.

#### Tasks

1. Extend replay to consume historical market-state data instead of synthetic candidates only.
2. Replay the actual policy grid against recorded market states.
3. Measure:
   - candidate count
   - order count
   - fill rate
   - realized PnL
   - markout
   - drawdown
   - exposure utilization
4. Add ablation runs:
   - forecast only
   - forecast + static execution assumptions
   - full execution-aware policy
5. Add baseline comparisons:
   - current heuristic system
   - market-mid only
   - naive always-IOC
   - naive always-GTC

#### Acceptance

- One run can compare old system versus Option 2 system apples-to-apples.

### Phase 11: Shadow Mode Rollout

#### Objective

Deploy safely before turning on live execution from the new policy.

#### Tasks

1. Add `BOT_POLICY_MODE`:
   - `legacy`
   - `shadow`
   - `active`
2. In `shadow`:
   - legacy system executes
   - new policy scores and logs hypothetical actions only
3. Log decision deltas:
   - legacy action
   - new action
   - expected realized value difference
4. Review drift and quality daily.

#### Acceptance

- New policy can run for multiple days with zero execution impact.
- Hypothetical performance is measurable before cutover.

### Phase 12: Active Rollout and Monitoring

#### Objective

Gradually promote the new policy to production behavior.

#### Rollout Stages

1. Demo shadow
2. Demo active with capped notional
3. Live shadow
4. Live active with strict caps
5. Live active with normal caps

#### Monitoring

- fill rate by tif and policy
- realized vs predicted fill probability
- realized vs predicted markout
- realized vs predicted expected PnL
- calibration drift
- per-vertical model health
- feature null-rate drift

#### Acceptance

- Promotion only after observed calibration and realized-value ranking are acceptable.

## Detailed Subproblems

### Subproblem A: Structured Market Parsing

Needed early because forecast and execution features depend on event semantics.

#### Deliverables

- parse threshold from ticker/title
- parse direction (`>`, `<`, `above`, `below`)
- parse asset/city/team/entity
- parse event date if available

#### Default

- rule-based parser first
- no LLM parsing in critical path

### Subproblem B: Research Log Durability

Needed before any modeling.

#### Deliverables

- append-only writers
- daily file rotation
- schema version field in each event
- recovery from partial lines

#### Default

- JSONL raw capture
- parquet built later by offline jobs

### Subproblem C: Label Correctness

Needed before trusting any offline metrics.

#### Deliverables

- unambiguous fill labels
- terminal order status normalization
- post-fill markout computation windows
- canceled/void handling
- same-order dedupe across reconciliations

#### Default

- markout windows: 5m and 30m
- fill targets: 30s and 5m

### Subproblem D: Feature Leakage Prevention

Critical for real performance.

#### Deliverables

- forbid using post-decision information in features
- strict timestamp cutoff enforcement
- time-based train/validation/test splits
- leakage tests

#### Default

- no future observations beyond action timestamp
- no using eventual order outcome in features

### Subproblem E: Low-Latency Inference

Needed for online use.

#### Deliverables

- serialized model artifacts with schema version
- Rust inference wrappers
- bounded feature computation time
- fallback behavior when model artifact missing

#### Default

- p95 online inference under 100 ms per cycle batch
- fail closed in active mode if forecast model missing
- fail over to `legacy` policy if execution model missing

### Subproblem F: Honest Evaluation

Needed to avoid demo-only alpha illusions.

#### Deliverables

- realized-value backtests
- shadow logs
- predicted vs realized calibration dashboards
- baseline comparisons

#### Default

- do not measure success by mark-to-mid alone
- primary metric is realized PnL net fees and adverse selection

## Environment and Configuration Additions

Add these env vars:

- `BOT_POLICY_MODE=legacy|shadow|active`
- `BOT_RESEARCH_CAPTURE_ENABLED=true|false`
- `BOT_RESEARCH_DIR=var/research`
- `BOT_MODEL_FORECAST_PATH=...`
- `BOT_MODEL_EXECUTION_PATH=...`
- `BOT_MARKOUT_WINDOWS_SECS=300,1800`
- `BOT_FILL_TARGET_WINDOWS_SECS=30,300`
- `BOT_POLICY_MIN_EXPECTED_REALIZED_PNL=...`
- `BOT_POLICY_MAX_ACTIONS_PER_CANDIDATE=...`
- `BOT_POLICY_DEFAULT_LEGACY_FALLBACK=true|false`

Defaults:

- start with `legacy`
- research capture enabled in demo and live
- active mode disabled until shadow validation passes

## Testing Plan

### Unit Tests

- market parsing
- feature builders
- label builders
- leakage guards
- policy scoring math
- model artifact loading
- fallback behavior

### Integration Tests

- scan -> feature build -> forecast inference
- forecast -> execution feature build -> execution inference
- policy -> allocator -> execution engine handoff
- shadow mode logging without changed execution
- active mode with missing model fails safely

### Replay Tests

- compare legacy vs new policy on same historical windows
- verify deterministic outputs for same feature rows
- verify no trade when expected realized PnL negative

### Acceptance Scenarios

1. High raw edge but low fill probability -> skip or reduce size
2. Moderate edge with strong fillability and low adverse selection -> trade
3. Wide spread and unstable quotes -> policy chooses skip
4. Resting order likely to be picked off -> avoid GTC
5. IOC likely to fill profitably -> choose IOC
6. Model artifacts unavailable -> remain on legacy safely

## Rollout Gates

### Gate 1: Data Readiness

- at least several days of raw market-state data
- enough order-lifecycle data to build fill labels
- outcome resolution coverage acceptable

### Gate 2: Forecast Readiness

- forecast model beats market-mid baseline on holdout data

### Gate 3: Execution Readiness

- execution model improves ranking of realized-value opportunities versus static heuristics

### Gate 4: Shadow Readiness

- shadow policy predictions stable
- predicted fill and markout are not wildly miscalibrated

### Gate 5: Active Readiness

- small-notional live demo performance consistent with shadow expectations

## Risks and Failure Modes

### Main Risks

- insufficient order-lifecycle data early on
- leakage in execution labels
- demo market dynamics not representative of live
- overfitting per-vertical with too little data
- markout estimation too noisy in thin markets

### Mitigations

- shadow mode before activation
- time-split evaluation only
- minimum-data threshold for per-vertical models
- fallback to global model
- explicit confidence gating
- continue using current risk engine even after policy upgrade

## First Step to Begin With

### Step 1 Title

Data Foundation and Research Capture

### Why This Is First

Every later subproblem depends on truthful historical market-state and order-lifecycle data. Without it, the forecast and execution models will be guesswork.

### Step 1 Exact Deliverables

1. Add a new research event schema for market-state capture.
2. Add a new research event schema for order-lifecycle capture.
3. Write append-only daily JSONL files under `var/research/`.
4. Capture scanner snapshot outputs and WS deltas with timestamps.
5. Capture full order lifecycle from intent through terminal state.
6. Add a non-invasive config flag so capture can run immediately without changing trade behavior.

### Step 1 Acceptance Criteria

- Existing bot behavior is unchanged.
- Running one live/demo cycle writes research logs.
- Research logs contain enough information to reconstruct:
  - market state at decision time
  - order parameters
  - order status evolution
- Research logging failure does not crash trading by default; it logs loudly and continues.

## Assumptions and Defaults

- The current Rust runtime remains the online orchestrator.
- Model training can happen out-of-band and artifacts can later be loaded into Rust.
- Tree-based tabular models are the default first model family.
- JSONL raw logs and parquet feature tables are the default storage pattern.
- The current execution engine remains the final safety gate.
- The policy objective is expected realized PnL, not theoretical edge.
- `legacy` remains the default policy mode until shadow validation passes.
- No new external serving service is introduced in the first iteration; inference stays local.
