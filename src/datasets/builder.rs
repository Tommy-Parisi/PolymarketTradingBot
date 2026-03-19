use std::collections::{BTreeMap, HashMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::data::market_scanner::ScannedMarket;
use crate::execution::types::{ExecutionError, OrderStatus, TradeSignal};
use crate::features::execution::{
    ExecutionContext, ExecutionFeatureRow, build_execution_feature_row_from_order_event,
};
use crate::features::forecast::{ForecastFeatureRow, build_forecast_feature_row_from_event};
use crate::outcomes::resolver::MarketOutcomeRecord;
use crate::research::events::{MarketStateEvent, OrderLifecycleEvent};

pub const DATASET_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone)]
pub struct DatasetBuildConfig {
    pub enabled: bool,
    pub research_dir: PathBuf,
    pub output_dir: PathBuf,
}

impl DatasetBuildConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_DATASET_BUILD")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            research_dir: PathBuf::from(
                std::env::var("BOT_RESEARCH_DIR").unwrap_or_else(|_| "var/research".to_string()),
            ),
            output_dir: PathBuf::from(
                std::env::var("BOT_FEATURES_DIR").unwrap_or_else(|_| "var/features".to_string()),
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastTrainingRow {
    pub schema_version: String,
    pub split: String,
    pub label_outcome_yes: Option<bool>,
    pub label_resolution_status: String,
    pub feature: ForecastFeatureRow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrainingRow {
    pub schema_version: String,
    pub split: String,
    pub client_order_id: String,
    pub terminal_status: Option<String>,
    pub label_filled_within_30s: bool,
    pub label_filled_within_5m: bool,
    pub label_terminal_filled_qty: f64,
    pub label_terminal_avg_fill_price: Option<f64>,
    pub label_canceled: bool,
    pub label_rejected: bool,
    pub label_markout_bps_5m: Option<f64>,
    pub label_markout_bps_30m: Option<f64>,
    pub label_realized_net_pnl: Option<f64>,
    pub feature: ExecutionFeatureRow,
}

pub async fn run_dataset_build(cfg: &DatasetBuildConfig) -> Result<(), ExecutionError> {
    let market_events = load_market_state_events(&cfg.research_dir)?;
    let order_events = load_order_lifecycle_events(&cfg.research_dir)?;
    let outcomes = load_outcomes(&cfg.research_dir)?;

    let forecast_rows = build_forecast_dataset(&market_events, &outcomes);
    let execution_rows = build_execution_dataset(&market_events, &order_events);

    write_jsonl_dataset(
        &cfg.output_dir.join("forecast").join("forecast_training.jsonl"),
        &forecast_rows,
    )?;
    write_jsonl_dataset(
        &cfg.output_dir.join("execution").join("execution_training.jsonl"),
        &execution_rows,
    )?;

    println!(
        "dataset build complete: forecast_rows={} execution_rows={}",
        forecast_rows.len(),
        execution_rows.len()
    );
    Ok(())
}

fn build_forecast_dataset(
    market_events: &[MarketStateEvent],
    outcomes: &HashMap<String, MarketOutcomeRecord>,
) -> Vec<ForecastTrainingRow> {
    let mut rows = Vec::new();
    for event in market_events {
        let Some(outcome) = outcomes.get(&event.ticker) else {
            continue;
        };
        rows.push(ForecastTrainingRow {
            schema_version: DATASET_SCHEMA_VERSION.to_string(),
            split: String::new(),
            label_outcome_yes: outcome.outcome_yes,
            label_resolution_status: outcome.resolution_status.clone(),
            feature: build_forecast_feature_row_from_event(event),
        });
    }
    assign_splits_forecast(rows)
}

fn build_execution_dataset(
    market_events: &[MarketStateEvent],
    order_events: &[OrderLifecycleEvent],
) -> Vec<ExecutionTrainingRow> {
    let market_index = build_market_index(market_events);
    let mut by_client_order_id: BTreeMap<String, Vec<OrderLifecycleEvent>> = BTreeMap::new();
    for event in order_events {
        by_client_order_id
            .entry(event.client_order_id.clone())
            .or_default()
            .push(event.clone());
    }

    let mut rows = Vec::new();
    for (client_order_id, mut events) in by_client_order_id {
        events.sort_by_key(|e| e.ts);
        let Some(intent) = events.iter().find(|e| e.event_type == "intent") else {
            continue;
        };
        let market = nearest_market_before(&market_index, &intent.ticker, intent.ts);
        let context = derive_execution_context(order_events, intent);
        let signal = synthetic_signal_from_intent(intent);
        let feature = build_execution_feature_row_from_order_event(
            intent,
            market.as_ref(),
            signal.as_ref(),
            &context,
        );
        let terminal = events.iter().rev().find(|e| e.status.is_some());
        let filled_within_30s = events.iter().any(|e| {
            e.filled_qty > 0.0 && (e.ts - intent.ts).num_seconds() >= 0 && (e.ts - intent.ts).num_seconds() <= 30
        });
        let filled_within_5m = events.iter().any(|e| {
            e.filled_qty > 0.0 && (e.ts - intent.ts).num_seconds() >= 0 && (e.ts - intent.ts).num_seconds() <= 300
        });
        let terminal_status = terminal.and_then(|e| e.status).map(|s| format!("{:?}", s));
        let terminal_filled_qty = terminal.map(|e| e.filled_qty).unwrap_or(0.0);
        let terminal_avg_fill_price = terminal.and_then(|e| e.avg_fill_price);
        let label_canceled = terminal
            .and_then(|e| e.status)
            .map(|s| s == OrderStatus::Canceled)
            .unwrap_or(false);
        let label_rejected = terminal
            .and_then(|e| e.status)
            .map(|s| s == OrderStatus::Rejected)
            .unwrap_or(false);
        let markout_bps_5m = compute_markout_bps(&market_index, intent, terminal_avg_fill_price, 300);
        let markout_bps_30m = compute_markout_bps(&market_index, intent, terminal_avg_fill_price, 1800);
        let realized_net_pnl = compute_realized_net_pnl(intent, terminal);

        rows.push(ExecutionTrainingRow {
            schema_version: DATASET_SCHEMA_VERSION.to_string(),
            split: String::new(),
            client_order_id,
            terminal_status,
            label_filled_within_30s: filled_within_30s,
            label_filled_within_5m: filled_within_5m,
            label_terminal_filled_qty: terminal_filled_qty,
            label_terminal_avg_fill_price: terminal_avg_fill_price,
            label_canceled,
            label_rejected,
            label_markout_bps_5m: markout_bps_5m,
            label_markout_bps_30m: markout_bps_30m,
            label_realized_net_pnl: realized_net_pnl,
            feature,
        });
    }
    assign_splits_execution(rows)
}

fn derive_execution_context(
    all_events: &[OrderLifecycleEvent],
    intent: &OrderLifecycleEvent,
) -> ExecutionContext {
    let mut open_order_count_same_ticker = 0u32;
    let mut recent_fill_count_same_ticker = 0u32;
    let mut recent_cancel_count_same_ticker = 0u32;
    let mut same_event_exposure_notional = 0.0f64;

    let event_root = event_root_from_ticker(&intent.ticker);
    for event in all_events.iter().filter(|e| e.ts < intent.ts) {
        if event.ticker == intent.ticker
            && matches!(event.status, Some(OrderStatus::New | OrderStatus::PartiallyFilled))
        {
            open_order_count_same_ticker += 1;
        }
        if event.ticker == intent.ticker
            && matches!(event.status, Some(OrderStatus::Filled | OrderStatus::PartiallyFilled))
            && (intent.ts - event.ts).num_seconds() <= 86_400
        {
            recent_fill_count_same_ticker += 1;
        }
        if event.ticker == intent.ticker
            && matches!(event.status, Some(OrderStatus::Canceled))
            && (intent.ts - event.ts).num_seconds() <= 86_400
        {
            recent_cancel_count_same_ticker += 1;
        }
        if event_root_from_ticker(&event.ticker) == event_root
            && matches!(event.status, Some(OrderStatus::Filled | OrderStatus::PartiallyFilled))
        {
            let price = event
                .avg_fill_price
                .or(event.limit_price)
                .unwrap_or(0.0)
                .max(0.0);
            same_event_exposure_notional += event.filled_qty.max(0.0) * price;
        }
    }

    ExecutionContext {
        open_order_count_same_ticker,
        recent_fill_count_same_ticker,
        recent_cancel_count_same_ticker,
        same_event_exposure_notional,
    }
}

fn build_market_index(
    market_events: &[MarketStateEvent],
) -> HashMap<String, Vec<MarketStateEvent>> {
    let mut out: HashMap<String, Vec<MarketStateEvent>> = HashMap::new();
    for event in market_events {
        out.entry(event.ticker.clone()).or_default().push(event.clone());
    }
    for events in out.values_mut() {
        events.sort_by_key(|e| e.ts);
    }
    out
}

fn nearest_market_before(
    index: &HashMap<String, Vec<MarketStateEvent>>,
    ticker: &str,
    ts: DateTime<Utc>,
) -> Option<ScannedMarket> {
    let events = index.get(ticker)?;
    let event = events.iter().rev().find(|e| e.ts <= ts)?;
    Some(ScannedMarket {
        ticker: event.ticker.clone(),
        title: event.title.clone(),
        subtitle: event.subtitle.clone(),
        market_type: event.market_type.clone(),
        event_ticker: event.event_ticker.clone(),
        series_ticker: event.series_ticker.clone(),
        yes_bid_cents: event.yes_bid_cents,
        yes_ask_cents: event.yes_ask_cents,
        volume: event.volume,
        close_time: event.close_time,
    })
}

fn synthetic_signal_from_intent(intent: &OrderLifecycleEvent) -> Option<TradeSignal> {
    let observed_price = intent
        .signal_observed_price
        .or(intent.limit_price)?;
    Some(TradeSignal {
        market_id: intent.ticker.clone(),
        outcome_id: intent.outcome_id.clone(),
        side: intent.side,
        fair_price: intent.signal_fair_price.unwrap_or(observed_price),
        observed_price,
        edge_pct: intent.signal_edge_pct.unwrap_or(0.0),
        confidence: intent.signal_confidence.unwrap_or(0.0),
        signal_timestamp: intent.ts,
    })
}

fn compute_markout_bps(
    market_index: &HashMap<String, Vec<MarketStateEvent>>,
    intent: &OrderLifecycleEvent,
    fill_price: Option<f64>,
    horizon_secs: i64,
) -> Option<f64> {
    let fill_price = fill_price?;
    let events = market_index.get(&intent.ticker)?;
    let target_ts = intent.ts + chrono::Duration::seconds(horizon_secs);
    let post = events.iter().find(|e| e.ts >= target_ts)?;
    let mid = post.mid_prob_yes?;
    let signed = if intent.outcome_id.eq_ignore_ascii_case("yes") {
        mid - fill_price
    } else {
        (1.0 - mid) - fill_price
    };
    Some((signed / fill_price.max(0.0001)) * 10_000.0)
}

fn compute_realized_net_pnl(
    intent: &OrderLifecycleEvent,
    terminal: Option<&OrderLifecycleEvent>,
) -> Option<f64> {
    let terminal = terminal?;
    let fair = intent.signal_fair_price?;
    let fill = terminal.avg_fill_price?;
    let qty = terminal.filled_qty.max(0.0);
    let fee = terminal.fee_paid.unwrap_or(0.0).max(0.0);
    let signed_unit = if intent.outcome_id.eq_ignore_ascii_case("yes") {
        fair - fill
    } else {
        fair - fill
    };
    Some((signed_unit * qty) - fee)
}

fn event_root_from_ticker(ticker: &str) -> String {
    match ticker.rfind('-') {
        Some(i) if i > 0 => ticker[..i].to_string(),
        _ => ticker.to_string(),
    }
}

fn load_market_state_events(root: &Path) -> Result<Vec<MarketStateEvent>, ExecutionError> {
    let mut out = Vec::new();
    let dir = root.join("market_state");
    if !dir.exists() {
        return Ok(out);
    }
    for day_dir in fs::read_dir(dir).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let day_dir = day_dir.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let day_path = day_dir.path();
        if !day_path.is_dir() {
            continue;
        }
        for file in fs::read_dir(day_path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let file = file.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            let file_path = file.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let text =
                fs::read_to_string(&file_path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<MarketStateEvent>(line) {
                    out.push(event);
                }
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    Ok(out)
}

fn load_order_lifecycle_events(root: &Path) -> Result<Vec<OrderLifecycleEvent>, ExecutionError> {
    let mut out = Vec::new();
    let dir = root.join("order_lifecycle");
    if !dir.exists() {
        return Ok(out);
    }
    for day_dir in fs::read_dir(dir).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let day_dir = day_dir.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let day_path = day_dir.path();
        if !day_path.is_dir() {
            continue;
        }
        for file in fs::read_dir(day_path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let file = file.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            let file_path = file.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let text =
                fs::read_to_string(&file_path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<OrderLifecycleEvent>(line) {
                    out.push(event);
                }
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    Ok(out)
}

fn load_outcomes(root: &Path) -> Result<HashMap<String, MarketOutcomeRecord>, ExecutionError> {
    let mut out = HashMap::new();
    let path = root.join("outcomes").join("outcomes.jsonl");
    if !path.exists() {
        return Ok(out);
    }
    let text = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_str::<MarketOutcomeRecord>(line) {
            out.insert(record.ticker.clone(), record);
        }
    }
    Ok(out)
}

fn assign_splits_forecast(mut rows: Vec<ForecastTrainingRow>) -> Vec<ForecastTrainingRow> {
    rows.sort_by_key(|r| r.feature.feature_ts);
    let total = rows.len().max(1);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.split = split_name(idx, total).to_string();
    }
    rows
}

fn assign_splits_execution(mut rows: Vec<ExecutionTrainingRow>) -> Vec<ExecutionTrainingRow> {
    rows.sort_by_key(|r| r.feature.feature_ts);
    let total = rows.len().max(1);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.split = split_name(idx, total).to_string();
    }
    rows
}

fn split_name(index: usize, total: usize) -> &'static str {
    let ratio = index as f64 / total as f64;
    if ratio < 0.70 {
        "train"
    } else if ratio < 0.85 {
        "validation"
    } else {
        "test"
    }
}

fn write_jsonl_dataset<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), ExecutionError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for row in rows {
        let line = serde_json::to_string(row).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn split_assignment_is_time_based() {
        let mut rows = Vec::new();
        for i in 0..10 {
            rows.push(ForecastTrainingRow {
                schema_version: DATASET_SCHEMA_VERSION.to_string(),
                split: String::new(),
                label_outcome_yes: Some(i % 2 == 0),
                label_resolution_status: "resolved".to_string(),
                feature: ForecastFeatureRow {
                    schema_version: "v1".to_string(),
                    feature_ts: Utc.timestamp_opt(i, 0).single().unwrap(),
                    ticker: format!("T{i}"),
                    title: "x".to_string(),
                    subtitle: None,
                    market_type: None,
                    event_ticker: None,
                    series_ticker: None,
                    close_time: None,
                    time_to_close_secs: None,
                    yes_bid_cents: None,
                    yes_ask_cents: None,
                    mid_prob_yes: None,
                    spread_cents: None,
                    volume: 0.0,
                    vertical: "other".to_string(),
                    weather_signal: None,
                    sports_injury_signal: None,
                    crypto_sentiment_signal: None,
                    entity_primary: None,
                    entity_secondary: None,
                    threshold_value: None,
                    threshold_direction: None,
                    event_date_hint: None,
                    source: "x".to_string(),
                    cycle_id: None,
                    recent_trade_count_delta: None,
                },
            });
        }
        let rows = assign_splits_forecast(rows);
        assert_eq!(rows[0].split, "train");
        assert_eq!(rows[8].split, "validation");
        assert_eq!(rows[9].split, "test");
    }
}
