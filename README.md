# PolymarketTradingBot

Autonomous trading agent for [Polymarket](https://polymarket.com), built in Rust for low-latency execution.

## Strategy Loop (every 10 minutes)

1. Scan 500-1000 active markets.
2. Build fair-value estimates using Claude.
3. Detect mispricing opportunities greater than 8%.
4. Compute position size using Kelly Criterion with a hard cap of 6% bankroll per trade.
5. Execute orders.
6. Route realized profits to cover API inference costs.

## Market Verticals

- Weather markets: parse NOAA data before Polymarket updates.
- Sports markets: scrape injury reports and price lag.
- Crypto markets: combine on-chain metrics and sentiment signals.

## Planned Components

- `src/main.rs`: scheduler + orchestration entrypoint
- `src/data/`: market ingestion + source adapters (NOAA, sports, on-chain)
- `src/model/`: Claude valuation prompts and response parsing
- `src/risk/`: Kelly sizing + bankroll constraints
- `src/execution/`: order routing + fill handling
- `src/accounting/`: PnL + API bill coverage logic

## Notes

This repository is initialized and connected to:

`https://github.com/Tommy-Parisi/PolymarketTradingBot.git`
