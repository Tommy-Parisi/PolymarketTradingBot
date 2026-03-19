use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;

use crate::data::market_scanner::{ScanTrace, ScannedMarket};
use crate::execution::types::ExecutionError;
use crate::research::events::{MarketStateEvent, RESEARCH_SCHEMA_VERSION};

#[derive(Debug, Clone)]
pub struct ResearchCaptureConfig {
    pub enabled: bool,
    pub root_dir: PathBuf,
}

impl ResearchCaptureConfig {
    pub fn from_env() -> Self {
        let enabled = matches!(
            std::env::var("BOT_RESEARCH_CAPTURE_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes"
        );
        let root_dir = PathBuf::from(
            std::env::var("BOT_RESEARCH_DIR").unwrap_or_else(|_| "var/research".to_string()),
        );
        Self { enabled, root_dir }
    }
}

pub fn record_scan_trace(
    cfg: &ResearchCaptureConfig,
    cycle_id: &str,
    snapshot_markets: &[ScannedMarket],
    trace: &ScanTrace,
) -> Result<(), ExecutionError> {
    if !cfg.enabled {
        return Ok(());
    }

    let mut events = Vec::with_capacity(snapshot_markets.len() + trace.deltas.len());
    for market in snapshot_markets {
        events.push(market_to_event(market, cycle_id, "snapshot", None));
    }
    for delta in &trace.deltas {
        if let Some(market) = trace.final_markets.iter().find(|m| m.ticker == delta.ticker) {
            events.push(MarketStateEvent {
                schema_version: RESEARCH_SCHEMA_VERSION.to_string(),
                ts: delta.observed_at,
                ticker: market.ticker.clone(),
                title: market.title.clone(),
                subtitle: market.subtitle.clone(),
                market_type: market.market_type.clone(),
                event_ticker: market.event_ticker.clone(),
                series_ticker: market.series_ticker.clone(),
                close_time: market.close_time,
                yes_bid_cents: delta.yes_bid_cents.or(market.yes_bid_cents),
                yes_ask_cents: delta.yes_ask_cents.or(market.yes_ask_cents),
                mid_prob_yes: implied_mid_prob_yes(
                    delta.yes_bid_cents.or(market.yes_bid_cents),
                    delta.yes_ask_cents.or(market.yes_ask_cents),
                ),
                spread_cents: implied_spread_cents(
                    delta.yes_bid_cents.or(market.yes_bid_cents),
                    delta.yes_ask_cents.or(market.yes_ask_cents),
                ),
                volume: market.volume,
                traded_count_delta: delta.traded_count_delta,
                source: "ws_delta".to_string(),
                cycle_id: cycle_id.to_string(),
            });
        }
    }

    append_events(cfg, "market_state", "market_state.jsonl", &events)
}

fn market_to_event(
    market: &ScannedMarket,
    cycle_id: &str,
    source: &str,
    traded_count_delta: Option<f64>,
) -> MarketStateEvent {
    MarketStateEvent {
        schema_version: RESEARCH_SCHEMA_VERSION.to_string(),
        ts: Utc::now(),
        ticker: market.ticker.clone(),
        title: market.title.clone(),
        subtitle: market.subtitle.clone(),
        market_type: market.market_type.clone(),
        event_ticker: market.event_ticker.clone(),
        series_ticker: market.series_ticker.clone(),
        close_time: market.close_time,
        yes_bid_cents: market.yes_bid_cents,
        yes_ask_cents: market.yes_ask_cents,
        mid_prob_yes: implied_mid_prob_yes(market.yes_bid_cents, market.yes_ask_cents),
        spread_cents: implied_spread_cents(market.yes_bid_cents, market.yes_ask_cents),
        volume: market.volume,
        traded_count_delta,
        source: source.to_string(),
        cycle_id: cycle_id.to_string(),
    }
}

fn implied_mid_prob_yes(bid: Option<f64>, ask: Option<f64>) -> Option<f64> {
    Some(((bid? + ask?) / 2.0 / 100.0).clamp(0.0, 1.0))
}

fn implied_spread_cents(bid: Option<f64>, ask: Option<f64>) -> Option<f64> {
    let (bid, ask) = (bid?, ask?);
    (ask >= bid).then_some(ask - bid)
}

fn append_events(
    cfg: &ResearchCaptureConfig,
    category: &str,
    filename: &str,
    events: &[MarketStateEvent],
) -> Result<(), ExecutionError> {
    if events.is_empty() {
        return Ok(());
    }
    let day = Utc::now().format("%Y-%m-%d").to_string();
    let path = cfg.root_dir.join(category).join(day).join(filename);
    append_json_lines(&path, events)
}

fn append_json_lines<T: serde::Serialize>(path: &Path, rows: &[T]) -> Result<(), ExecutionError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
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
