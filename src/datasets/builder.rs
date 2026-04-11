use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
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
    /// Only load market-state and order-lifecycle events from the last N days.
    /// Prevents in-process dataset builds from OOMing as research data grows.
    /// Set via BOT_DATASET_MAX_DAYS (default 30). 0 = no limit (use with caution).
    pub max_days: u64,
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
            max_days: std::env::var("BOT_DATASET_MAX_DAYS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30),
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
    pub execution_source_class: String,
    pub is_bootstrap_synthetic: bool,
    pub is_organic_paper: bool,
    pub is_live_real: bool,
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
    // Load order events first (small — ~1MB) so we know which tickers need market snapshots.
    let order_events = load_order_lifecycle_events(&cfg.research_dir, cfg.max_days)?;
    let outcomes = load_outcomes(&cfg.research_dir)?;

    // Tickers we have orders for — used to limit what we keep in the market index.
    let order_tickers: HashSet<String> = order_events.iter().map(|e| e.ticker.clone()).collect();

    // Single streaming pass over market_state JSONL: builds forecast rows + market index
    // without ever materialising a Vec<MarketStateEvent> of the full dataset.
    let (forecast_rows_unsplit, mut market_index) =
        scan_market_state_events(&cfg.research_dir, cfg.max_days, &outcomes, &order_tickers)?;

    for events in market_index.values_mut() {
        events.sort_by_key(|e| e.ts);
    }

    let forecast_rows = assign_splits_forecast(forecast_rows_unsplit);
    let execution_rows = build_execution_dataset(&market_index, &order_events, &outcomes);
    let bootstrap_rows: Vec<_> = execution_rows
        .iter()
        .filter(|row| row.is_bootstrap_synthetic)
        .cloned()
        .collect();
    let organic_paper_rows: Vec<_> = execution_rows
        .iter()
        .filter(|row| row.is_organic_paper)
        .cloned()
        .collect();
    let live_real_rows: Vec<_> = execution_rows
        .iter()
        .filter(|row| row.is_live_real)
        .cloned()
        .collect();

    write_jsonl_dataset(
        &cfg.output_dir.join("forecast").join("forecast_training.jsonl"),
        &forecast_rows,
    )?;
    write_jsonl_dataset(
        &cfg.output_dir.join("execution").join("execution_training.jsonl"),
        &execution_rows,
    )?;
    write_jsonl_dataset(
        &cfg.output_dir
            .join("execution")
            .join("execution_training_bootstrap.jsonl"),
        &bootstrap_rows,
    )?;
    write_jsonl_dataset(
        &cfg.output_dir
            .join("execution")
            .join("execution_training_organic_paper.jsonl"),
        &organic_paper_rows,
    )?;
    write_jsonl_dataset(
        &cfg.output_dir
            .join("execution")
            .join("execution_training_live_real.jsonl"),
        &live_real_rows,
    )?;

    println!(
        "dataset build complete: forecast_rows={} execution_rows={} bootstrap_rows={} organic_paper_rows={} live_real_rows={}",
        forecast_rows.len(),
        execution_rows.len(),
        bootstrap_rows.len(),
        organic_paper_rows.len(),
        live_real_rows.len()
    );
    Ok(())
}

/// Streams market_state JSONL files in one pass, producing:
/// - forecast rows (for any ticker with a resolved outcome)
/// - market index (only for tickers in `order_tickers`, used by execution model)
///
/// Never materialises a Vec<MarketStateEvent> of the full dataset, keeping peak
/// memory proportional to (forecast rows + traded-ticker snapshots) rather than
/// all observed market events.
fn scan_market_state_events(
    root: &Path,
    max_days: u64,
    outcomes: &HashMap<String, MarketOutcomeRecord>,
    order_tickers: &HashSet<String>,
) -> Result<(Vec<ForecastTrainingRow>, HashMap<String, Vec<MarketStateEvent>>), ExecutionError> {
    let mut forecast_rows: Vec<ForecastTrainingRow> = Vec::new();
    let mut market_index: HashMap<String, Vec<MarketStateEvent>> = HashMap::new();

    let dir = root.join("market_state");
    if !dir.exists() {
        return Ok((forecast_rows, market_index));
    }
    for day_dir in fs::read_dir(dir).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let day_dir = day_dir.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let day_path = day_dir.path();
        if !day_path.is_dir() {
            continue;
        }
        if !day_dir_is_within_window(&day_path, max_days) {
            continue;
        }
        for file in fs::read_dir(day_path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let file = file.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            let file_path = file.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let reader = BufReader::new(
                File::open(&file_path).map_err(|e| ExecutionError::Exchange(e.to_string()))?,
            );
            for line in reader.lines() {
                let line = line.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<MarketStateEvent>(&line) else {
                    continue;
                };
                if let Some(outcome) = outcomes.get(&event.ticker) {
                    forecast_rows.push(ForecastTrainingRow {
                        schema_version: DATASET_SCHEMA_VERSION.to_string(),
                        split: String::new(),
                        label_outcome_yes: outcome.outcome_yes,
                        label_resolution_status: outcome.resolution_status.clone(),
                        feature: build_forecast_feature_row_from_event(&event),
                    });
                }
                if order_tickers.contains(&event.ticker) {
                    market_index.entry(event.ticker.clone()).or_default().push(event);
                }
            }
        }
    }
    Ok((forecast_rows, market_index))
}

fn parse_window_secs_env(var: &str, default: &str) -> Vec<i64> {
    std::env::var(var)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect()
}

fn build_execution_dataset(
    market_index: &HashMap<String, Vec<MarketStateEvent>>,
    order_events: &[OrderLifecycleEvent],
    outcomes: &HashMap<String, MarketOutcomeRecord>,
) -> Vec<ExecutionTrainingRow> {
    // Fill-target windows come from BOT_FILL_TARGET_WINDOWS_SECS (comma-separated secs).
    // Defaults: 30s, 300s (5 min).
    let fill_windows = parse_window_secs_env("BOT_FILL_TARGET_WINDOWS_SECS", "30,300");
    let fill_window_0 = fill_windows.first().copied().unwrap_or(30);
    let fill_window_1 = fill_windows.get(1).copied().unwrap_or(300);

    // Markout windows come from BOT_MARKOUT_WINDOWS_SECS (comma-separated secs).
    // Defaults: 300s (5 min), 1800s (30 min).
    let markout_windows = parse_window_secs_env("BOT_MARKOUT_WINDOWS_SECS", "300,1800");
    let markout_window_0 = markout_windows.first().copied().unwrap_or(300);
    let markout_window_1 = markout_windows.get(1).copied().unwrap_or(1800);

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
            e.filled_qty > 0.0
                && (e.ts - intent.ts).num_seconds() >= 0
                && (e.ts - intent.ts).num_seconds() <= fill_window_0
        });
        let filled_within_5m = events.iter().any(|e| {
            e.filled_qty > 0.0
                && (e.ts - intent.ts).num_seconds() >= 0
                && (e.ts - intent.ts).num_seconds() <= fill_window_1
        });
        let terminal_status = terminal.and_then(|e| e.status).map(|s| format!("{:?}", s));
        let terminal_filled_qty = terminal.map(|e| e.filled_qty).unwrap_or(0.0);
        let terminal_avg_fill_price = terminal.and_then(|e| e.avg_fill_price);
        let execution_source_class = classify_execution_source(intent, &events, terminal);
        let label_canceled = terminal
            .and_then(|e| e.status)
            .map(|s| s == OrderStatus::Canceled)
            .unwrap_or(false);
        let label_rejected = terminal
            .and_then(|e| e.status)
            .map(|s| s == OrderStatus::Rejected)
            .unwrap_or(false);
        let markout_bps_5m =
            compute_markout_bps(&market_index, intent, terminal_avg_fill_price, markout_window_0);
        let markout_bps_30m =
            compute_markout_bps(&market_index, intent, terminal_avg_fill_price, markout_window_1);
        let realized_net_pnl = compute_realized_net_pnl(intent, terminal, outcomes);

        rows.push(ExecutionTrainingRow {
            schema_version: DATASET_SCHEMA_VERSION.to_string(),
            split: String::new(),
            client_order_id,
            is_bootstrap_synthetic: execution_source_class == "bootstrap_synthetic",
            is_organic_paper: execution_source_class == "organic_paper",
            is_live_real: execution_source_class == "live_real",
            execution_source_class,
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
        yes_bid_size: event.yes_bid_size,
        yes_ask_size: event.yes_ask_size,
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
        signal_origin: intent.signal_origin.clone(),
    })
}

fn classify_execution_source(
    intent: &OrderLifecycleEvent,
    events: &[OrderLifecycleEvent],
    terminal: Option<&OrderLifecycleEvent>,
) -> String {
    if matches!(intent.signal_origin.as_deref(), Some("bootstrap_synthetic")) {
        return "bootstrap_synthetic".to_string();
    }

    let is_paper = intent.execution_mode.as_deref() == Some("paper")
        || terminal
            .and_then(|event| event.execution_mode.as_deref())
            .map(|mode| mode == "paper")
            .unwrap_or(false)
        || events.iter().any(|event| {
            event
                .order_id
                .as_deref()
                .map(|id| id.starts_with("paper-") || id.starts_with("sim-"))
                .unwrap_or(false)
        });

    if is_paper && intent.is_synthetic {
        return "bootstrap_synthetic".to_string();
    }
    if is_paper {
        return "organic_paper".to_string();
    }
    "live_real".to_string()
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
    outcomes: &HashMap<String, MarketOutcomeRecord>,
) -> Option<f64> {
    let terminal = terminal?;
    let fill = terminal.avg_fill_price?;
    let qty = terminal.filled_qty.max(0.0);
    let fee = terminal.fee_paid.unwrap_or(0.0).max(0.0);
    let outcome = outcomes.get(&intent.ticker)?;
    if outcome.resolution_status == "canceled" {
        return None;
    }
    let outcome_yes = outcome.outcome_yes?;
    let is_yes_side = intent.outcome_id.eq_ignore_ascii_case("yes");
    let winner = if is_yes_side { outcome_yes } else { !outcome_yes };
    let pnl = if winner {
        (1.0 - fill) * qty - fee
    } else {
        -fill * qty - fee
    };
    Some(pnl)
}

fn event_root_from_ticker(ticker: &str) -> String {
    match ticker.rfind('-') {
        Some(i) if i > 0 => ticker[..i].to_string(),
        _ => ticker.to_string(),
    }
}

fn day_dir_is_within_window(day_path: &Path, max_days: u64) -> bool {
    if max_days == 0 {
        return true;
    }
    let name = match day_path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return true, // can't parse, include to be safe
    };
    let day = match chrono::NaiveDate::parse_from_str(name, "%Y-%m-%d") {
        Ok(d) => d,
        Err(_) => return true, // non-date directory, include
    };
    let cutoff = (Utc::now() - chrono::Duration::days(max_days as i64)).date_naive();
    day >= cutoff
}

fn load_order_lifecycle_events(root: &Path, max_days: u64) -> Result<Vec<OrderLifecycleEvent>, ExecutionError> {
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
        if !day_dir_is_within_window(&day_path, max_days) {
            continue;
        }
        for file in fs::read_dir(day_path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let file = file.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            let file_path = file.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let reader = BufReader::new(
                File::open(&file_path).map_err(|e| ExecutionError::Exchange(e.to_string()))?,
            );
            for line in reader.lines() {
                let line = line.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<OrderLifecycleEvent>(&line) {
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
    let reader = BufReader::new(
        File::open(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?,
    );
    for line in reader.lines() {
        let line = line.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_str::<MarketOutcomeRecord>(&line) {
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
    fn split_name_boundary_values() {
        // 10 rows: indices 0..9, split at 0.70 and 0.85
        // index 0..6 (ratio < 0.70) → train
        // index 7 (0.70 ≤ ratio < 0.85) → validation
        // index 8..9 (ratio ≥ 0.85) → test
        assert_eq!(split_name(0, 10), "train");
        assert_eq!(split_name(6, 10), "train");  // 6/10 = 0.6 < 0.70
        assert_eq!(split_name(7, 10), "validation"); // 7/10 = 0.70
        assert_eq!(split_name(8, 10), "validation"); // 8/10 = 0.80
        assert_eq!(split_name(9, 10), "test");   // 9/10 = 0.90 ≥ 0.85
    }

    #[test]
    fn classify_execution_source_detects_bootstrap_synthetic() {
        let intent = OrderLifecycleEvent {
            schema_version: "v1".to_string(),
            ts: Utc::now(),
            client_order_id: "c1".to_string(),
            order_id: None,
            ticker: "KXTEST".to_string(),
            outcome_id: "yes".to_string(),
            side: crate::execution::types::Side::Buy,
            tif: crate::execution::types::TimeInForce::Gtc,
            limit_price: Some(0.5),
            requested_qty: 10.0,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            signal_origin: Some("bootstrap_synthetic".to_string()),
            execution_mode: Some("paper".to_string()),
            status: None,
            is_synthetic: false,
            event_type: "intent".to_string(),
            error: None,
        };
        let result = classify_execution_source(&intent, &[intent.clone()], None);
        assert_eq!(result, "bootstrap_synthetic");
    }

    #[test]
    fn classify_execution_source_detects_organic_paper() {
        let intent = OrderLifecycleEvent {
            schema_version: "v1".to_string(),
            ts: Utc::now(),
            client_order_id: "c2".to_string(),
            order_id: None,
            ticker: "KXTEST".to_string(),
            outcome_id: "yes".to_string(),
            side: crate::execution::types::Side::Buy,
            tif: crate::execution::types::TimeInForce::Gtc,
            limit_price: Some(0.5),
            requested_qty: 10.0,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            signal_origin: Some("model_candidate".to_string()),
            execution_mode: Some("paper".to_string()),
            status: None,
            is_synthetic: false,
            event_type: "intent".to_string(),
            error: None,
        };
        let result = classify_execution_source(&intent, &[intent.clone()], None);
        assert_eq!(result, "organic_paper");
    }

    #[test]
    fn classify_execution_source_detects_paper_from_order_id_prefix() {
        let intent = OrderLifecycleEvent {
            schema_version: "v1".to_string(),
            ts: Utc::now(),
            client_order_id: "c3".to_string(),
            order_id: None,
            ticker: "KXTEST".to_string(),
            outcome_id: "yes".to_string(),
            side: crate::execution::types::Side::Buy,
            tif: crate::execution::types::TimeInForce::Gtc,
            limit_price: Some(0.5),
            requested_qty: 10.0,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            signal_origin: None,
            execution_mode: None,      // no execution_mode set
            status: None,
            is_synthetic: false,
            event_type: "intent".to_string(),
            error: None,
        };
        // Sibling event has order_id with paper- prefix
        let ack = OrderLifecycleEvent {
            order_id: Some("paper-abc123".to_string()),
            event_type: "ack".to_string(),
            ..intent.clone()
        };
        let result = classify_execution_source(&intent, &[intent.clone(), ack], None);
        assert_eq!(result, "organic_paper");
    }

    #[test]
    fn classify_execution_source_detects_live_real() {
        let intent = OrderLifecycleEvent {
            schema_version: "v1".to_string(),
            ts: Utc::now(),
            client_order_id: "c4".to_string(),
            order_id: None,
            ticker: "KXTEST".to_string(),
            outcome_id: "yes".to_string(),
            side: crate::execution::types::Side::Buy,
            tif: crate::execution::types::TimeInForce::Gtc,
            limit_price: Some(0.5),
            requested_qty: 10.0,
            filled_qty: 10.0,
            avg_fill_price: Some(0.50),
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            signal_origin: Some("model_candidate".to_string()),
            execution_mode: Some("live".to_string()),
            status: Some(OrderStatus::Filled),
            is_synthetic: false,
            event_type: "intent".to_string(),
            error: None,
        };
        let result = classify_execution_source(&intent, &[intent.clone()], None);
        assert_eq!(result, "live_real");
    }

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
                    feature_ts: match Utc.timestamp_opt(i, 0).single() {
                        Some(ts) => ts,
                        None => {
                            eprintln!("warning: skipping row with unparseable timestamp {i}");
                            continue;
                        }
                    },
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
                    yes_bid_size: None,
                    yes_ask_size: None,
                    mid_prob_yes: None,
                    spread_cents: None,
                    book_pressure: None,
                    volume: 0.0,
                    vertical: "other".to_string(),
                    weather_signal: None,
                    sports_injury_signal: None,
                    crypto_sentiment_signal: None,
                    finance_price_signal: None,
                    specialist_prob_yes: None,
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

    fn make_intent(ticker: &str, outcome_id: &str) -> OrderLifecycleEvent {
        OrderLifecycleEvent {
            schema_version: "v1".to_string(),
            ts: Utc::now(),
            client_order_id: "c-test".to_string(),
            order_id: None,
            ticker: ticker.to_string(),
            outcome_id: outcome_id.to_string(),
            side: crate::execution::types::Side::Buy,
            tif: crate::execution::types::TimeInForce::Gtc,
            limit_price: None,
            requested_qty: 10.0,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            signal_origin: None,
            execution_mode: None,
            status: None,
            is_synthetic: false,
            event_type: "intent".to_string(),
            error: None,
        }
    }

    fn make_terminal(fill: f64, qty: f64, fee: f64) -> OrderLifecycleEvent {
        OrderLifecycleEvent {
            filled_qty: qty,
            avg_fill_price: Some(fill),
            fee_paid: Some(fee),
            event_type: "fill".to_string(),
            status: Some(crate::execution::types::OrderStatus::Filled),
            ..make_intent("KXTEST-YES", "yes")
        }
    }

    fn make_outcome(ticker: &str, outcome_yes: Option<bool>, status: &str) -> MarketOutcomeRecord {
        MarketOutcomeRecord {
            schema_version: "v1".to_string(),
            ticker: ticker.to_string(),
            resolved_at: Utc::now(),
            outcome_yes,
            resolution_status: status.to_string(),
            source: "test".to_string(),
            close_time: None,
        }
    }

    #[test]
    fn realized_pnl_yes_win() {
        // YES buy at 0.40, YES wins → pnl = (1.0 - 0.40) * 10.0 - 0.05 = 5.95
        let intent = make_intent("KXTEST-YES", "yes");
        let terminal = make_terminal(0.40, 10.0, 0.05);
        let mut outcomes = HashMap::new();
        outcomes.insert("KXTEST-YES".to_string(), make_outcome("KXTEST-YES", Some(true), "resolved"));
        let pnl = compute_realized_net_pnl(&intent, Some(&terminal), &outcomes);
        assert!((pnl.unwrap() - 5.95).abs() < 1e-9);
    }

    #[test]
    fn realized_pnl_yes_loss() {
        // YES buy at 0.40, NO wins → pnl = -0.40 * 10.0 - 0.05 = -4.05
        let intent = make_intent("KXTEST-YES", "yes");
        let terminal = make_terminal(0.40, 10.0, 0.05);
        let mut outcomes = HashMap::new();
        outcomes.insert("KXTEST-YES".to_string(), make_outcome("KXTEST-YES", Some(false), "resolved"));
        let pnl = compute_realized_net_pnl(&intent, Some(&terminal), &outcomes);
        assert!((pnl.unwrap() - (-4.05)).abs() < 1e-9);
    }

    #[test]
    fn realized_pnl_no_win() {
        // NO buy at 0.60, YES loses → NO wins → pnl = (1.0 - 0.60) * 10.0 - 0.05 = 3.95
        let intent = make_intent("KXTEST-NO", "no");
        let terminal = make_terminal(0.60, 10.0, 0.05);
        let mut outcomes = HashMap::new();
        outcomes.insert("KXTEST-NO".to_string(), make_outcome("KXTEST-NO", Some(false), "resolved"));
        let pnl = compute_realized_net_pnl(&intent, Some(&terminal), &outcomes);
        assert!((pnl.unwrap() - 3.95).abs() < 1e-9);
    }

    #[test]
    fn realized_pnl_no_loss() {
        // NO buy at 0.60, YES wins → NO loses → pnl = -0.60 * 10.0 - 0.05 = -6.05
        let intent = make_intent("KXTEST-NO", "no");
        let terminal = make_terminal(0.60, 10.0, 0.05);
        let mut outcomes = HashMap::new();
        outcomes.insert("KXTEST-NO".to_string(), make_outcome("KXTEST-NO", Some(true), "resolved"));
        let pnl = compute_realized_net_pnl(&intent, Some(&terminal), &outcomes);
        assert!((pnl.unwrap() - (-6.05)).abs() < 1e-9);
    }

    #[test]
    fn realized_pnl_canceled_returns_none() {
        // canceled outcome → None
        let intent = make_intent("KXTEST-YES", "yes");
        let terminal = make_terminal(0.40, 10.0, 0.05);
        let mut outcomes = HashMap::new();
        outcomes.insert("KXTEST-YES".to_string(), make_outcome("KXTEST-YES", None, "canceled"));
        let pnl = compute_realized_net_pnl(&intent, Some(&terminal), &outcomes);
        assert!(pnl.is_none());
    }
}
