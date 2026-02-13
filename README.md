# EventTradingBot

Autonomous trading agent for event markets, built in Rust for low-latency execution.

## Strategy Loop (every 10 minutes)

1. Scan 500-1000 active markets.
2. Build fair-value estimates using Claude.
3. Detect mispricing opportunities greater than 8%.
4. Compute position size using Kelly Criterion with a hard cap of 6% bankroll per trade.
5. Execute orders.
6. Route realized profits to cover API inference costs.

## Market Verticals

- Weather markets: parse NOAA data before exchange prices fully react.
- Sports markets: scrape injury reports and price lag.
- Crypto markets: combine on-chain metrics and sentiment signals.

## Planned Components

- `src/main.rs`: scheduler + orchestration entrypoint
- `src/data/`: market ingestion + source adapters (NOAA, sports, on-chain)
- `src/model/`: Claude valuation prompts, caching, and candidate generation
- `src/risk/`: Kelly sizing + bankroll constraints
- `src/execution/`: order routing + fill handling
- `src/accounting/`: PnL + API bill coverage logic

## Execution Engine Design

Order execution is implemented in `src/execution/` with three layers:

- `types.rs`: strict contracts for `TradeSignal`, `OrderRequest`, `ExecutionReport`, and failure modes.
- `client.rs`: `ExchangeClient` trait and `KalshiClient` adapter boundary for API integration.
- `engine.rs`: `ExecutionEngine` that enforces:
  - minimum edge threshold (default 8%)
  - stale signal rejection
  - Kelly-based sizing with hard cap (default max 6% bankroll/trade)
  - max notional per market guardrail
  - retry logic for retryable exchange failures

Execution flow:

1. Validate signal freshness and edge.
2. Compute position size from bankroll and Kelly fraction.
3. Build IOC limit order with idempotent client order ID.
4. Submit order and fetch order state/report.

## Kalshi Client Setup

The live Kalshi client reads auth and routing from environment variables:

- `KALSHI_API_BASE_URL` (default `https://demo-api.kalshi.co`)
- `KALSHI_WS_URL` (default `wss://demo-api.kalshi.co/trade-api/ws/v2`)
- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PEM` or `KALSHI_PRIVATE_KEY_PATH`
- `BOT_EXECUTION_MODE` (`paper` default, set `live` to place real orders)
- `KALSHI_MARKET_ALIASES` (optional): comma-separated alias map, e.g. `btc120k=KXBTC-26DEC31-B120000,nyc90f=KXWEATHER-NYC-90F`
- `BOT_MARKET_RESOLUTION`: `best_effort` (default) or `strict`
- `BOT_MAX_DAILY_LOSS` (default `500`)
- `BOT_MAX_OPEN_EXPOSURE` (default `15000`)
- `BOT_MAX_ORDERS_PER_MIN` (default `20`)
- `BOT_STATE_PATH` (default `var/state/runtime_state.json`)
- `BOT_JOURNAL_PATH` (default `var/logs/trade_journal.jsonl`)
- `BOT_MIN_EDGE_PCT` (default `0.08`; execution-time minimum edge gate)
- `BOT_RUN_SMOKE_TEST` (`true/false`, default `false`)
- `NOAA_POINT` (default `39.7456,-97.0892`)
- `SPORTS_INJURY_API_URL` (optional)
- `CRYPTO_SENTIMENT_API_URL` (optional; defaults to Alternative.me FNG)
- `BOT_BANKROLL` (default `10000`)
- `BOT_VALUATION_MARKETS` (default `250`)
- `BOT_ENRICHMENT_MARKETS` (default `25`)
- `BOT_SCAN_MAX_MARKETS` (default `1000`)
- `BOT_SCAN_MIN_VOLUME` (default `1000`)
- `BOT_SCAN_MAX_SPREAD_CENTS` (default `8`)
- `BOT_SCAN_WS_DELTA_WINDOW_SECS` (default `2`)
- `BOT_MISPRICING_THRESHOLD` (default `0.08`)
- `BOT_MIN_CANDIDATES` (default `0`; when >0, backfills from real valuations if strict threshold returns too few)
- `BOT_FALLBACK_MISPRICING_THRESHOLD` (default `0.02`; used only for the backfill path above)
- `BOT_ADAPTIVE_THRESHOLD_ENABLED` (`true/false`, default `false`; uses liquidity/spread/confidence to set per-market effective thresholds)
- `BOT_ADAPTIVE_THRESHOLD_FLOOR` (default `0.01`; minimum allowed adaptive threshold)
- `BOT_ADAPTIVE_LIQUIDITY_VOLUME_REF` (default `50000`; higher volume lowers threshold toward floor)
- `BOT_ADAPTIVE_SPREAD_REF_CENTS` (default `12`; tighter spreads lower threshold toward floor)
- `BOT_ADAPTIVE_CONFIDENCE_WEIGHT` (default `0.20`)
- `BOT_ADAPTIVE_LIQUIDITY_WEIGHT` (default `0.55`)
- `BOT_ADAPTIVE_SPREAD_WEIGHT` (default `0.25`)
- `BOT_FEE_BPS` (default `15`)
- `BOT_SLIPPAGE_BPS` (default `20`)
- `BOT_VALUATION_BATCH_SIZE` (default `32`)
- `BOT_VALUATION_TIMEOUT_MS` (default `8000`)
- `CLAUDE_MODEL` (default `claude-3-5-sonnet-latest`)
- `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`)
- `ANTHROPIC_API_KEY` (optional; without it, heuristic valuation fallback is used)
- `BOT_ALLOW_HEURISTIC_IN_LIVE` (`true/false`, default `false`; live mode fails closed if heuristic fallback is used unless this is enabled)
- `BOT_MAX_TRADES_PER_CYCLE` (default `5`)
- `BOT_MAX_FRACTION_PER_TRADE` (default `0.06`)
- `BOT_MAX_TOTAL_FRACTION_PER_CYCLE` (default `0.20`)
- `BOT_MIN_FRACTION_PER_TRADE` (default `0.005`)
- `BOT_ENFORCE_EVENT_MUTEX` (default `true`; prevents allocating both sides of the same underlying event root in one cycle)
- `BOT_CYCLE_SECONDS` (default `600`, i.e. 10 minutes)
- `BOT_CLAUDE_EVERY_N_CYCLES` (default `1`; run Claude valuation every N cycles, heuristic on other cycles for cost control)
- `BOT_CLAUDE_TRIGGER_MODE` (`cadence` default, or `on_viable_markets`, or `on_heuristic_candidates`)
  - `on_viable_markets`: Claude runs only on cycles with selected markets (skips no-market cycles).
  - `on_heuristic_candidates`: heuristic runs first; Claude runs only if heuristic finds potential candidates.
- `BOT_RUN_ONCE` (`true/false`, default `false`)
- `BOT_FORCE_TEST_CANDIDATE` (`true/false`, default `false`; injects one deterministic candidate when none pass threshold)
- `BOT_CYCLE_ARTIFACTS_ENABLED` (`true/false`, default `true`)
- `BOT_CYCLE_ARTIFACTS_DIR` (default `var/cycles`)
- `BOT_EXCHANGE_BACKEND` (`kalshi` default, `paper_sim` optional)
- `BOT_RUN_REPLAY` (`true/false`, default `false`)
- `BOT_REPLAY_DAYS` (default `3`)
- `BOT_REPLAY_CYCLES_PER_DAY` (default `144`)
- `BOT_REPLAY_BANKROLL` (default `10000`)
- `BOT_RUN_SUMMARY_ONLY` (`true/false`, default `false`; prints aggregated PnL/trade summary from journal + state and exits)

The client signs each request using Kalshi's `timestamp + METHOD + path` convention with RSA-PSS and sends:
- `KALSHI-ACCESS-KEY`
- `KALSHI-ACCESS-TIMESTAMP`
- `KALSHI-ACCESS-SIGNATURE`

## Market Symbol Mapper

`src/markets/kalshi_mapper.rs` resolves strategy market inputs to Kalshi tickers by:

1. Checking alias overrides from `KALSHI_MARKET_ALIASES`
2. Accepting already ticker-shaped inputs directly
3. Querying `GET /trade-api/v2/markets` and matching against ticker/title/subtitle/event/series fields

In `best_effort`, unresolved inputs log a warning and continue with the original value.
In `strict`, unresolved or ambiguous inputs fail fast before order placement.

## Market Scanning (Fastest Access Path)

Implemented scanner: `src/data/market_scanner.rs`

Current flow:

1. Pull open markets from `GET /trade-api/v2/markets` with cursor pagination and up to 1000 markets/page.
2. Merge short-window WebSocket deltas (`ticker_v2`, `trade`) via `src/data/ws_delta.rs`.
3. Filter to liquid + tight markets (volume + spread thresholds).
4. Pass selected tickers to valuation.

Reasoning for speed:

1. Single large REST snapshots are fastest way to bootstrap 500-1000 markets.
2. WebSocket deltas remove stale quotes between scan windows.
3. Feature enrichment now uses provider-native APIs with in-process caching.

## Enrichment Layer

Implemented: `src/data/market_enrichment.rs`

1. Detects market vertical (weather/sports/crypto).
2. Fetches vertical signals:
- Weather: NOAA (`api.weather.gov`)
- Sports: configurable injury feed (`SPORTS_INJURY_API_URL`)
- Crypto: sentiment feed (`CRYPTO_SENTIMENT_API_URL`, defaults to Alternative.me Fear & Greed)
3. Caches enrichment by ticker with TTL (default 300s).

## Valuation + Candidate Generation

Implemented: `src/model/valuation.rs`

1. Batch valuation (default 32 markets/request) with timeout/retry limits.
2. Claude inference path via Anthropic Messages API.
3. Heuristic fallback when API is unavailable.
4. Prompt-size cap and per-batch token cap.
5. Cache of unchanged market/enrichment inputs.
6. Mispricing candidate generation with fee/slippage-adjusted edge threshold.

## Portfolio Allocation

Implemented: `src/model/allocator.rs`

1. Ranks candidates by edge * confidence.
2. Applies Kelly-style sizing with per-trade and per-cycle bankroll caps.
3. Filters tiny allocations and returns executable trade basket for the cycle.

## Orchestrator Loop

`src/main.rs` now runs the full pipeline on a continuous loop:

1. snapshot + websocket delta scan
2. enrichment
3. valuation
4. candidate generation
5. allocation
6. execution

Default cadence is every 600 seconds (10 minutes). Use `BOT_RUN_ONCE=true` for single-cycle dry runs.

## Deterministic Paper Simulation + Replay

Implemented:

- Paper exchange backend: `src/execution/paper_sim.rs`
- Multi-day replay harness: `src/replay/mod.rs`

Simulation model includes:

1. deterministic per-order latency
2. deterministic slippage
3. deterministic partial-fill / cancel outcomes
4. fee model

Replay mode (`BOT_RUN_REPLAY=true`) runs synthetic multi-day cycles and prints:

1. total orders
2. fill/partial/cancel counts
3. fees paid
4. edge-PnL net fees

For long paper/live runs, generate an end-of-run summary without executing trades:

`set -a; source .env; set +a; BOT_RUN_SUMMARY_ONLY=true cargo run --quiet`

The summary includes:

1. total and per-day order counts
2. filled/partial/canceled/rejected counts
3. traded notional and fees paid
4. expected edge PnL net fees (derived from logged signal edge on order intents)
5. latest runtime state exposure and daily realized PnL snapshot

During normal cycles, the bot also prints a live mark-to-market snapshot of open positions based on current scanned market mids:

1. `position marks: open_positions=... marked_positions=... total_unrealized=...`
2. per-position rows with `entry`, latest `mark`, and `unrealized`

Execution logs now include Kalshi lookup hints and market titles for easier manual inspection.

## Trading Safety Controls

The execution engine now includes production-oriented controls:

1. Pre-trade checks:
- edge, staleness, size bounds, bankroll sufficiency
2. Kill switches:
- daily loss cap, open exposure cap, order-rate cap
3. Post-trade controls:
- poll/reconcile partial orders, auto-cancel lingering orders, startup open-order reconciliation
4. Idempotency and dedupe:
- persistent `client_order_id` memory across restarts
5. Persistent audit trail:
- JSON runtime state and JSONL journal under `BOT_STATE_PATH` / `BOT_JOURNAL_PATH`

Recommended rollout:

1. Run in `paper` mode until journal and state behavior look correct.
2. Switch to Kalshi demo credentials in `live` with `BOT_RUN_SMOKE_TEST=true` first.
3. Promote to production credentials only after a successful demo burn-in period.

## Local Control Page

Use `/Users/tommy/Desktop/Desktop_Thomas_MacBook_Pro/PolymarketTradingBot/control_panel.html` to generate run commands quickly with presets and persistent fields.

Open it locally, choose a preset, then copy the generated command and run it in repo root after:

`set -a; source .env; set +a`

## Notes

This repository is initialized and connected to:

`origin` remote on GitHub.
