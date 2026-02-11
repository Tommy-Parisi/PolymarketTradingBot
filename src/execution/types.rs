use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimeInForce {
    Gtc,
    Ioc,
    Fok,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub market_id: String,
    pub outcome_id: String,
    pub side: Side,
    pub fair_price: f64,
    pub observed_price: f64,
    pub edge_pct: f64,
    pub confidence: f64,
    pub signal_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub client_order_id: String,
    pub market_id: String,
    pub outcome_id: String,
    pub side: Side,
    pub order_type: OrderType,
    pub limit_price: Option<f64>,
    pub quantity: f64,
    pub time_in_force: TimeInForce,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAck {
    pub order_id: String,
    pub client_order_id: String,
    pub accepted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub order_id: String,
    pub client_order_id: String,
    pub status: OrderStatus,
    pub filled_qty: f64,
    pub avg_fill_price: Option<f64>,
    pub fee_paid: f64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub min_edge_pct: f64,
    pub max_bankroll_fraction_per_trade: f64,
    pub max_market_notional: f64,
    pub stale_signal_after_secs: i64,
    pub max_retries: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            min_edge_pct: 0.08,
            max_bankroll_fraction_per_trade: 0.06,
            max_market_notional: 1_000.0,
            stale_signal_after_secs: 60,
            max_retries: 2,
        }
    }
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("signal edge {edge:.4} below minimum {min_edge:.4}")]
    EdgeTooSmall { edge: f64, min_edge: f64 },

    #[error("signal is stale: age_secs={age_secs}, max_age_secs={max_age_secs}")]
    StaleSignal { age_secs: i64, max_age_secs: i64 },

    #[error("invalid price: {price}")]
    InvalidPrice { price: f64 },

    #[error("invalid quantity: {quantity}")]
    InvalidQuantity { quantity: f64 },

    #[error("notional {notional:.4} exceeds market cap {cap:.4}")]
    NotionalCapExceeded { notional: f64, cap: f64 },

    #[error("exchange error: {0}")]
    Exchange(String),

    #[error("retryable exchange error: {0}")]
    RetryableExchange(String),
}

pub fn new_client_order_id(market_id: &str) -> String {
    format!("{}-{}", market_id, Uuid::new_v4())
}
