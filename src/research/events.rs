use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::execution::types::{OrderStatus, Side, TimeInForce};

pub const RESEARCH_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStateEvent {
    pub schema_version: String,
    pub ts: DateTime<Utc>,
    pub ticker: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub market_type: Option<String>,
    pub event_ticker: Option<String>,
    pub series_ticker: Option<String>,
    pub close_time: Option<DateTime<Utc>>,
    pub yes_bid_cents: Option<f64>,
    pub yes_ask_cents: Option<f64>,
    pub mid_prob_yes: Option<f64>,
    pub spread_cents: Option<f64>,
    pub volume: f64,
    pub traded_count_delta: Option<f64>,
    pub source: String,
    pub cycle_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLifecycleEvent {
    pub schema_version: String,
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
    pub fee_paid: Option<f64>,
    pub signal_fair_price: Option<f64>,
    pub signal_observed_price: Option<f64>,
    pub signal_edge_pct: Option<f64>,
    pub signal_confidence: Option<f64>,
    #[serde(default)]
    pub signal_origin: Option<String>,
    #[serde(default)]
    pub execution_mode: Option<String>,
    pub status: Option<OrderStatus>,
    pub event_type: String,
    pub error: Option<String>,
}
