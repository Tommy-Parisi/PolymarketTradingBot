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
- `src/model/`: Claude valuation prompts and response parsing
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
- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PEM` or `KALSHI_PRIVATE_KEY_PATH`
- `BOT_EXECUTION_MODE` (`paper` default, set `live` to place real orders)
- `KALSHI_MARKET_ALIASES` (optional): comma-separated alias map, e.g. `btc120k=KXBTC-26DEC31-B120000,nyc90f=KXWEATHER-NYC-90F`
- `BOT_MARKET_RESOLUTION`: `best_effort` (default) or `strict`

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

## Notes

This repository is initialized and connected to:

`origin` remote on GitHub.
