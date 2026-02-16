use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::sleep;

use crate::execution::client::ExchangeClient;
use crate::execution::types::{
    new_client_order_id, EngineConfig, ExecutionError, ExecutionReport, OrderAck, OrderRequest, OrderStatus,
    OrderType, Side, TimeInForce, TradeSignal,
};

pub struct ExecutionEngine {
    client: Arc<dyn ExchangeClient>,
    config: EngineConfig,
    mode: ExecutionMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DailyPnlSummary {
    pub orders_reported: u64,
    pub filled_orders: u64,
    pub traded_notional: f64,
    pub fees_paid: f64,
    pub expected_edge_pnl_net_fees: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PnlSummary {
    pub journal_path: String,
    pub state_path: String,
    pub orders_reported: u64,
    pub filled_orders: u64,
    pub partial_orders: u64,
    pub canceled_orders: u64,
    pub rejected_orders: u64,
    pub traded_notional: f64,
    pub fees_paid: f64,
    pub expected_edge_pnl_net_fees: f64,
    pub state_day: Option<String>,
    pub state_daily_realized_pnl: Option<f64>,
    pub state_open_exposure_notional: Option<f64>,
    pub by_day: BTreeMap<String, DailyPnlSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Paper,
    Live,
}

impl ExecutionEngine {
    pub fn new(client: Arc<dyn ExchangeClient>, config: EngineConfig, mode: ExecutionMode) -> Self {
        Self { client, config, mode }
    }

    pub async fn execute_signal(
        &self,
        signal: &TradeSignal,
        bankroll: f64,
    ) -> Result<ExecutionReport, ExecutionError> {
        self.validate_signal(signal)?;
        let total_quantity = self.compute_position_size(signal, bankroll)?;
        self.validate_quantity_bounds(total_quantity)?;

        let policy = self.execution_policy();
        match policy {
            ExecutionPolicy::Ioc => {
                self.execute_single_order(signal, bankroll, total_quantity, TimeInForce::Ioc)
                    .await
            }
            ExecutionPolicy::Gtc => {
                self.execute_single_order(signal, bankroll, total_quantity, TimeInForce::Gtc)
                    .await
            }
            ExecutionPolicy::Hybrid => {
                let ioc_qty = (total_quantity * self.config.hybrid_ioc_fraction)
                    .max(self.config.min_order_quantity)
                    .min(total_quantity);
                let first_report = self
                    .execute_single_order(signal, bankroll, ioc_qty, TimeInForce::Ioc)
                    .await?;

                let remaining = (total_quantity - first_report.filled_qty).max(0.0);
                if remaining < self.config.min_order_quantity {
                    return Ok(first_report);
                }
                if matches!(first_report.status, OrderStatus::Rejected) {
                    return Ok(first_report);
                }

                match self
                    .execute_single_order(signal, bankroll, remaining, TimeInForce::Gtc)
                    .await
                {
                    Ok(report) => Ok(report),
                    Err(err) => {
                        self.append_journal(
                            "execution_policy_fallback_failed",
                            json!({
                                "reason": err.to_string(),
                                "fallback_time_in_force": "gtc",
                                "remaining_qty": remaining,
                                "market_id": signal.market_id
                            }),
                        )?;
                        Ok(first_report)
                    }
                }
            }
        }
    }

    async fn execute_single_order(
        &self,
        signal: &TradeSignal,
        bankroll: f64,
        quantity: f64,
        time_in_force: TimeInForce,
    ) -> Result<ExecutionReport, ExecutionError> {
        self.validate_quantity_bounds(quantity)?;
        let order = self.build_order(signal, quantity, time_in_force)?;
        self.pre_trade_checks(&order, bankroll)?;

        let mut state = self.load_state()?;
        self.check_kill_switches(&state, &order)?;
        if state.seen_client_order_ids.contains(&order.client_order_id) {
            return Err(ExecutionError::DuplicateClientOrderId {
                client_order_id: order.client_order_id,
            });
        }

        self.record_pre_submit_state(&mut state, &order);
        self.append_journal(
            "order_intent",
            json!({
                "order": &order,
                "mode": format!("{:?}", self.mode),
                "signal_edge_pct": signal.edge_pct,
                "signal_fair_price": signal.fair_price,
                "signal_observed_price": signal.observed_price
            }),
        )?;
        self.save_state(&state)?;

        let mut report = if self.mode == ExecutionMode::Paper {
            ExecutionReport {
                order_id: format!("paper-{}", order.client_order_id),
                client_order_id: order.client_order_id.clone(),
                status: OrderStatus::Filled,
                submitted_time_in_force: Some(order.time_in_force),
                filled_qty: order.quantity,
                avg_fill_price: order.limit_price,
                fee_paid: 0.0,
                updated_at: Utc::now(),
            }
        } else {
            let (ack, initial) = match self.submit_with_retries(&order).await {
                Ok(v) => v,
                Err(err) => {
                    let _ = self.append_journal(
                        "order_error",
                        json!({
                            "market_id": order.market_id,
                            "outcome_id": order.outcome_id,
                            "side": format!("{:?}", order.side),
                            "time_in_force": format!("{:?}", order.time_in_force),
                            "error": err.to_string()
                        }),
                    );
                    return Err(err);
                }
            };
            self.append_journal("order_ack", json!({"ack": ack}))?;
            self.reconcile_post_submit(ack, initial, order.time_in_force).await?
        };

        if report.submitted_time_in_force.is_none() {
            report.submitted_time_in_force = Some(order.time_in_force);
        }
        self.finalize_state_from_report(&mut state, &order, &report);
        self.append_journal("order_report", json!({"report": &report}))?;
        self.save_state(&state)?;
        Ok(report)
    }

    pub async fn reconcile_open_orders(&self) -> Result<(), ExecutionError> {
        let mut state = self.load_state()?;
        let open_orders = state.open_orders.clone();
        if open_orders.is_empty() {
            return Ok(());
        }

        let mut survivors = Vec::new();
        for tracked in open_orders {
            let mut report = self.client.get_order(&tracked.order_id).await?;
            if matches!(report.status, OrderStatus::New | OrderStatus::PartiallyFilled)
                && self.execution_policy() == ExecutionPolicy::Ioc
            {
                let _ = self.client.cancel_order(&tracked.order_id).await;
                report = self.client.get_order(&tracked.order_id).await.unwrap_or(report);
            }
            self.append_journal(
                "reconcile_order",
                json!({"order_id": tracked.order_id, "status": format!("{:?}", report.status)}),
            )?;
            if matches!(report.status, OrderStatus::New | OrderStatus::PartiallyFilled) {
                survivors.push(tracked);
            }
        }
        state.open_orders = survivors;
        self.save_state(&state)?;
        Ok(())
    }

    pub async fn run_smoke_test(&self) -> Result<(), ExecutionError> {
        self.client.smoke_test().await
    }

    fn validate_signal(&self, signal: &TradeSignal) -> Result<(), ExecutionError> {
        if signal.edge_pct < self.config.min_edge_pct {
            return Err(ExecutionError::EdgeTooSmall {
                edge: signal.edge_pct,
                min_edge: self.config.min_edge_pct,
            });
        }
        if !(0.0..=1.0).contains(&signal.observed_price) {
            return Err(ExecutionError::InvalidPrice {
                price: signal.observed_price,
            });
        }
        let age_secs = (Utc::now() - signal.signal_timestamp).num_seconds();
        if age_secs > self.config.stale_signal_after_secs {
            return Err(ExecutionError::StaleSignal {
                age_secs,
                max_age_secs: self.config.stale_signal_after_secs,
            });
        }
        Ok(())
    }

    fn compute_position_size(&self, signal: &TradeSignal, bankroll: f64) -> Result<f64, ExecutionError> {
        let kelly_fraction = approximate_kelly_fraction(signal);
        let capped_fraction = kelly_fraction.min(self.config.max_bankroll_fraction_per_trade);
        let capped_notional = bankroll * capped_fraction;
        let notional = capped_notional.min(self.config.max_market_notional);
        if notional <= 0.0 {
            return Err(ExecutionError::InvalidQuantity { quantity: 0.0 });
        }
        let quantity = notional / signal.observed_price.max(0.01);
        if quantity <= 0.0 {
            return Err(ExecutionError::InvalidQuantity { quantity });
        }
        Ok(quantity)
    }

    fn validate_quantity_bounds(&self, quantity: f64) -> Result<(), ExecutionError> {
        if quantity < self.config.min_order_quantity || quantity > self.config.max_order_quantity {
            return Err(ExecutionError::QuantityOutOfBounds {
                quantity,
                min: self.config.min_order_quantity,
                max: self.config.max_order_quantity,
            });
        }
        Ok(())
    }

    fn build_order(
        &self,
        signal: &TradeSignal,
        quantity: f64,
        time_in_force: TimeInForce,
    ) -> Result<OrderRequest, ExecutionError> {
        let limit_price = compute_limit_price(signal.side, signal.observed_price);
        let notional = quantity * limit_price;
        if notional > self.config.max_market_notional {
            return Err(ExecutionError::NotionalCapExceeded {
                notional,
                cap: self.config.max_market_notional,
            });
        }
        Ok(OrderRequest {
            client_order_id: new_client_order_id(&signal.market_id),
            market_id: signal.market_id.clone(),
            outcome_id: signal.outcome_id.clone(),
            side: signal.side,
            order_type: OrderType::Limit,
            limit_price: Some(limit_price),
            quantity,
            time_in_force,
            created_at: Utc::now(),
        })
    }

    fn pre_trade_checks(&self, order: &OrderRequest, bankroll: f64) -> Result<(), ExecutionError> {
        if order.market_id.trim().is_empty() {
            return Err(ExecutionError::Exchange("market_id cannot be empty".to_string()));
        }
        let limit_price = order.limit_price.ok_or_else(|| ExecutionError::InvalidPrice { price: -1.0 })?;
        let required_notional = limit_price * order.quantity;
        if required_notional > bankroll {
            return Err(ExecutionError::InsufficientCapital {
                required_notional,
                bankroll,
            });
        }
        Ok(())
    }

    fn check_kill_switches(&self, state: &RuntimeState, order: &OrderRequest) -> Result<(), ExecutionError> {
        if state.daily_realized_pnl <= -self.config.max_daily_loss {
            return Err(ExecutionError::KillSwitch {
                reason: format!(
                    "daily loss {:.2} breached max {:.2}",
                    -state.daily_realized_pnl,
                    self.config.max_daily_loss
                ),
            });
        }

        let order_notional = order.quantity * order.limit_price.unwrap_or(0.0);
        if state.open_exposure_notional + order_notional > self.config.max_open_exposure_notional {
            return Err(ExecutionError::KillSwitch {
                reason: format!(
                    "open exposure {:.2} + {:.2} exceeds limit {:.2}",
                    state.open_exposure_notional,
                    order_notional,
                    self.config.max_open_exposure_notional
                ),
            });
        }

        let now = Utc::now().timestamp();
        let orders_last_min = state
            .recent_order_unix_secs
            .iter()
            .filter(|ts| now - **ts <= 60)
            .count();
        if orders_last_min >= self.config.max_orders_per_minute {
            return Err(ExecutionError::KillSwitch {
                reason: format!(
                    "order rate {} in last minute exceeded limit {}",
                    orders_last_min,
                    self.config.max_orders_per_minute
                ),
            });
        }
        Ok(())
    }

    fn record_pre_submit_state(&self, state: &mut RuntimeState, order: &OrderRequest) {
        state.roll_day_if_needed();
        state.recent_order_unix_secs.push(Utc::now().timestamp());
        state.recent_order_unix_secs.retain(|ts| Utc::now().timestamp() - *ts <= 60);
        state.seen_client_order_ids.push(order.client_order_id.clone());
    }

    fn finalize_state_from_report(&self, state: &mut RuntimeState, order: &OrderRequest, report: &ExecutionReport) {
        let fill_price = report.avg_fill_price.unwrap_or(order.limit_price.unwrap_or(0.0));
        let filled_notional = report.filled_qty * fill_price;

        // Conservative: treat fees as realized loss.
        state.daily_realized_pnl -= report.fee_paid.max(0.0);

        match report.status {
            OrderStatus::New | OrderStatus::PartiallyFilled => {
                if !state.open_orders.iter().any(|o| o.order_id == report.order_id) {
                    state.open_orders.push(OpenOrderState {
                        order_id: report.order_id.clone(),
                        client_order_id: report.client_order_id.clone(),
                        market_id: order.market_id.clone(),
                        notional: filled_notional.max(order.quantity * order.limit_price.unwrap_or(0.0)),
                        created_at: Utc::now(),
                    });
                }
                state.open_exposure_notional += filled_notional;
            }
            OrderStatus::Filled => {
                state.open_exposure_notional += filled_notional;
                state.open_orders.retain(|o| o.order_id != report.order_id);
            }
            OrderStatus::Canceled | OrderStatus::Rejected => {
                state.open_orders.retain(|o| o.order_id != report.order_id);
            }
        }
    }

    async fn submit_with_retries(
        &self,
        order: &OrderRequest,
    ) -> Result<(OrderAck, ExecutionReport), ExecutionError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.client.place_order(order).await {
                Ok(ack) => {
                    let report = self.client.get_order(&ack.order_id).await?;
                    return Ok((ack, report));
                }
                Err(ExecutionError::RetryableExchange(_)) if attempt <= self.config.max_retries => {
                    sleep(Duration::from_millis(250 * attempt as u64)).await;
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn reconcile_post_submit(
        &self,
        ack: OrderAck,
        mut report: ExecutionReport,
        time_in_force: TimeInForce,
    ) -> Result<ExecutionReport, ExecutionError> {
        if !matches!(report.status, OrderStatus::New | OrderStatus::PartiallyFilled) {
            return Ok(report);
        }

        for _ in 0..self.config.reconcile_poll_attempts {
            sleep(Duration::from_millis(self.config.reconcile_poll_interval_ms)).await;
            report = self.client.get_order(&ack.order_id).await?;
            if matches!(
                report.status,
                OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Rejected
            ) {
                return Ok(report);
            }
        }

        if time_in_force == TimeInForce::Ioc {
            let _ = self.client.cancel_order(&ack.order_id).await;
            let final_report = self.client.get_order(&ack.order_id).await.unwrap_or(report);
            return Ok(final_report);
        }

        Ok(report)
    }

    fn load_state(&self) -> Result<RuntimeState, ExecutionError> {
        let path = Path::new(&self.config.state_path);
        if !path.exists() {
            return Ok(RuntimeState::default_for_today());
        }
        let text = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let mut state: RuntimeState =
            serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        state.roll_day_if_needed();
        Ok(state)
    }

    fn save_state(&self, state: &RuntimeState) -> Result<(), ExecutionError> {
        let path = Path::new(&self.config.state_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        }
        let data = serde_json::to_string_pretty(state).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        fs::write(path, data).map_err(|e| ExecutionError::Exchange(e.to_string()))
    }

    fn append_journal(&self, event: &str, payload: serde_json::Value) -> Result<(), ExecutionError> {
        let path = Path::new(&self.config.journal_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        }
        let entry = JournalEntry {
            ts: Utc::now(),
            event: event.to_string(),
            payload,
        };
        let line = serde_json::to_string(&entry).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        f.write_all(line.as_bytes())
            .and_then(|_| f.write_all(b"\n"))
            .map_err(|e| ExecutionError::Exchange(e.to_string()))
    }

    fn execution_policy(&self) -> ExecutionPolicy {
        match self.config.execution_policy.to_ascii_lowercase().as_str() {
            "gtc" => ExecutionPolicy::Gtc,
            "hybrid" => ExecutionPolicy::Hybrid,
            _ => ExecutionPolicy::Ioc,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionPolicy {
    Ioc,
    Gtc,
    Hybrid,
}

fn compute_limit_price(side: Side, observed_price: f64) -> f64 {
    match side {
        Side::Buy => (observed_price + 0.01).clamp(0.01, 0.99),
        Side::Sell => (observed_price - 0.01).clamp(0.01, 0.99),
    }
}

fn approximate_kelly_fraction(signal: &TradeSignal) -> f64 {
    let p = signal.fair_price.clamp(0.01, 0.99);
    let q = 1.0 - p;
    let b = (1.0 - signal.observed_price).max(0.01) / signal.observed_price.max(0.01);
    let raw = ((b * p) - q) / b;
    raw.max(0.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeState {
    day: String,
    daily_realized_pnl: f64,
    open_exposure_notional: f64,
    recent_order_unix_secs: Vec<i64>,
    seen_client_order_ids: Vec<String>,
    open_orders: Vec<OpenOrderState>,
}

impl RuntimeState {
    fn default_for_today() -> Self {
        Self {
            day: Utc::now().format("%Y-%m-%d").to_string(),
            daily_realized_pnl: 0.0,
            open_exposure_notional: 0.0,
            recent_order_unix_secs: Vec::new(),
            seen_client_order_ids: Vec::new(),
            open_orders: Vec::new(),
        }
    }

    fn roll_day_if_needed(&mut self) {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        if self.day != today {
            self.day = today;
            self.daily_realized_pnl = 0.0;
            self.recent_order_unix_secs.clear();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenOrderState {
    order_id: String,
    client_order_id: String,
    market_id: String,
    notional: f64,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JournalEntry {
    ts: DateTime<Utc>,
    event: String,
    payload: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct SummaryIntentOrder {
    client_order_id: String,
}

#[derive(Debug, Deserialize)]
struct SummaryIntentPayload {
    order: SummaryIntentOrder,
    #[serde(default)]
    signal_edge_pct: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SummaryReportPayload {
    report: ExecutionReport,
}

pub fn summarize_performance_paths(state_path: &str, journal_path: &str) -> Result<PnlSummary, ExecutionError> {
    let mut summary = PnlSummary {
        journal_path: journal_path.to_string(),
        state_path: state_path.to_string(),
        ..PnlSummary::default()
    };
    let mut edge_by_client_order_id: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    let journal_p = Path::new(journal_path);
    if journal_p.exists() {
        let text = fs::read_to_string(journal_p).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: JournalEntry =
                serde_json::from_str(line).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            let day = entry.ts.format("%Y-%m-%d").to_string();
            let day_summary = summary.by_day.entry(day).or_default();

            if entry.event == "order_intent" {
                if let Ok(intent) = serde_json::from_value::<SummaryIntentPayload>(entry.payload) {
                    edge_by_client_order_id.insert(
                        intent.order.client_order_id,
                        intent.signal_edge_pct.unwrap_or(0.0).max(0.0),
                    );
                }
                continue;
            }

            if entry.event != "order_report" {
                continue;
            }
            let payload = match serde_json::from_value::<SummaryReportPayload>(entry.payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let report = payload.report;
            summary.orders_reported += 1;
            day_summary.orders_reported += 1;
            match report.status {
                OrderStatus::Filled => {
                    summary.filled_orders += 1;
                    day_summary.filled_orders += 1;
                }
                OrderStatus::PartiallyFilled => summary.partial_orders += 1,
                OrderStatus::Canceled => summary.canceled_orders += 1,
                OrderStatus::Rejected => summary.rejected_orders += 1,
                OrderStatus::New => {}
            }
            let fill_price = report.avg_fill_price.unwrap_or(0.0).max(0.0);
            let filled_qty = report.filled_qty.max(0.0);
            let notional = fill_price * filled_qty;
            let fee = report.fee_paid.max(0.0);
            let edge = edge_by_client_order_id
                .get(&report.client_order_id)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let edge_pnl_net_fees = (notional * edge) - fee;

            summary.traded_notional += notional;
            summary.fees_paid += fee;
            summary.expected_edge_pnl_net_fees += edge_pnl_net_fees;

            day_summary.traded_notional += notional;
            day_summary.fees_paid += fee;
            day_summary.expected_edge_pnl_net_fees += edge_pnl_net_fees;
        }
    }

    let state_p = Path::new(state_path);
    if state_p.exists() {
        let text = fs::read_to_string(state_p).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let state: RuntimeState = serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        summary.state_day = Some(state.day);
        summary.state_daily_realized_pnl = Some(state.daily_realized_pnl);
        summary.state_open_exposure_notional = Some(state.open_exposure_notional);
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::execution::client::ExchangeClient;
    use crate::execution::types::Side;

    fn base_signal() -> TradeSignal {
        TradeSignal {
            market_id: "m1".to_string(),
            outcome_id: "yes".to_string(),
            side: Side::Buy,
            fair_price: 0.62,
            observed_price: 0.53,
            edge_pct: 0.09,
            confidence: 0.8,
            signal_timestamp: Utc::now(),
        }
    }

    fn test_engine() -> ExecutionEngine {
        let mut cfg = EngineConfig::default();
        cfg.state_path = "/tmp/event_trading_bot_state_test.json".to_string();
        cfg.journal_path = "/tmp/event_trading_bot_journal_test.jsonl".to_string();
        ExecutionEngine {
            client: Arc::new(NoopClient),
            config: cfg,
            mode: ExecutionMode::Paper,
        }
    }

    #[test]
    fn rejects_low_edge_signal() {
        let engine = test_engine();
        let mut signal = base_signal();
        signal.edge_pct = 0.04;
        let result = engine.validate_signal(&signal);
        assert!(matches!(result, Err(ExecutionError::EdgeTooSmall { .. })));
    }

    #[test]
    fn rejects_stale_signal() {
        let engine = test_engine();
        let mut signal = base_signal();
        signal.signal_timestamp = Utc::now() - chrono::Duration::seconds(120);
        let result = engine.validate_signal(&signal);
        assert!(matches!(result, Err(ExecutionError::StaleSignal { .. })));
    }

    #[test]
    fn position_size_is_capped_by_bankroll_and_market_limit() {
        let engine = test_engine();
        let signal = base_signal();
        let size = engine
            .compute_position_size(&signal, 1_000_000.0)
            .expect("expected valid size");
        assert!(size * signal.observed_price <= 1_000.0 + 1e-6);
    }

    #[test]
    fn quantity_bounds_are_enforced() {
        let mut cfg = EngineConfig::default();
        cfg.min_order_quantity = 5.0;
        cfg.max_order_quantity = 10.0;
        cfg.state_path = "/tmp/event_trading_bot_state_test.json".to_string();
        cfg.journal_path = "/tmp/event_trading_bot_journal_test.jsonl".to_string();
        let engine = ExecutionEngine {
            client: Arc::new(NoopClient),
            config: cfg,
            mode: ExecutionMode::Paper,
        };
        assert!(matches!(
            engine.validate_quantity_bounds(3.0),
            Err(ExecutionError::QuantityOutOfBounds { .. })
        ));
    }

    struct NoopClient;

    #[async_trait]
    impl ExchangeClient for NoopClient {
        async fn place_order(&self, _request: &OrderRequest) -> Result<OrderAck, ExecutionError> {
            Err(ExecutionError::Exchange("noop".to_string()))
        }

        async fn get_order(&self, _order_id: &str) -> Result<ExecutionReport, ExecutionError> {
            Err(ExecutionError::Exchange("noop".to_string()))
        }

        async fn cancel_order(&self, _order_id: &str) -> Result<(), ExecutionError> {
            Ok(())
        }

        async fn smoke_test(&self) -> Result<(), ExecutionError> {
            Ok(())
        }
    }
}
