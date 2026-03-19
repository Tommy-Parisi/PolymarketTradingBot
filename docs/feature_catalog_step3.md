# Step 3 Feature Catalog

This file freezes the initial forecast and execution feature schemas before dataset-building starts.

## Forecast Features

Source struct: `src/features/forecast.rs::ForecastFeatureRow`

- `schema_version`
- `feature_ts`
- `ticker`
- `title`
- `subtitle`
- `market_type`
- `event_ticker`
- `series_ticker`
- `close_time`
- `time_to_close_secs`
- `yes_bid_cents`
- `yes_ask_cents`
- `mid_prob_yes`
- `spread_cents`
- `volume`
- `vertical`
- `weather_signal`
- `sports_injury_signal`
- `crypto_sentiment_signal`
- `entity_primary`
- `entity_secondary`
- `threshold_value`
- `threshold_direction`
- `event_date_hint`
- `source`
- `cycle_id`
- `recent_trade_count_delta`

### Null Handling

- Quote-dependent fields are nullable when market data is incomplete.
- Enrichment fields are nullable when no vertical signal is available.
- Parsed fields are nullable when the rule-based parser cannot infer them.
- `time_to_close_secs` is nullable when close time is missing.

## Execution Features

Source struct: `src/features/execution.rs::ExecutionFeatureRow`

- `schema_version`
- `feature_ts`
- `ticker`
- `outcome_id`
- `side`
- `tif`
- `title`
- `vertical`
- `candidate_limit_price`
- `candidate_observed_price`
- `candidate_fair_price`
- `raw_edge_pct`
- `confidence`
- `yes_bid_cents`
- `yes_ask_cents`
- `spread_cents`
- `mid_prob_yes`
- `volume`
- `time_to_close_secs`
- `price_vs_best_bid_cents`
- `price_vs_best_ask_cents`
- `aggressiveness_bps`
- `open_order_count_same_ticker`
- `recent_fill_count_same_ticker`
- `recent_cancel_count_same_ticker`
- `same_event_exposure_notional`

### Interpretation

- `price_vs_best_bid_cents`: positive means the candidate price is above the current best bid.
- `price_vs_best_ask_cents`: negative means the candidate price is below the current best ask.
- `aggressiveness_bps`: normalized quote aggressiveness against current spread.

## Rule-Based Parsing Defaults

Source helper: `src/features/forecast.rs::parse_market_metadata`

- Direction keywords:
  - `above`, `over`, `greater than`, `>`
  - `below`, `under`, `less than`, `<`
- Threshold extraction:
  - first numeric token in title, subtitle, or ticker
- Entity extraction:
  - split on `vs` / `vs.`
- Vertical inference:
  - weather: `HIGH`, `TEMP`, `WEATHER`
  - sports: sports tickers or `vs`/`winner`
  - crypto: `BTC`, `ETH`, `Bitcoin`

## Status

This is the initial feature contract for Step 3. Dataset builders in Step 4 must target these schemas exactly unless the catalog is updated in-repo first.
