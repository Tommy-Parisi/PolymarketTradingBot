use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::sleep;

use crate::data::market_scanner::ScannedMarket;
use crate::execution::client::ExchangeClient;
use crate::execution::types::{
    new_client_order_id, EngineConfig, ExecutionError, ExecutionReport, OrderAck, OrderRequest, OrderStatus,
    OrderType, Side, TimeInForce, TradeSignal,
};
use crate::research::market_recorder::ResearchCaptureConfig;
use crate::research::order_recorder::{
    record_order_ack, record_order_error, record_order_intent, record_order_report,
};

pub struct ExecutionEngine {
    client: Arc<dyn ExchangeClient>,
    config: EngineConfig,
    mode: ExecutionMode,
}

const RUNTIME_STATE_SCHEMA_VERSION: &str = "v2";
const JOURNAL_SCHEMA_VERSION: &str = "v1";

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
        market: Option<&ScannedMarket>,
    ) -> Result<ExecutionReport, ExecutionError> {
        self.validate_signal(signal)?;
        let total_quantity = self.compute_position_size(signal, bankroll)?;
        self.validate_quantity_bounds(total_quantity)?;

        let policy = self.execution_policy();
        match policy {
            ExecutionPolicy::Ioc => {
                self.execute_single_order(signal, bankroll, total_quantity, TimeInForce::Ioc, market)
                    .await
            }
            ExecutionPolicy::Gtc => {
                self.execute_single_order(signal, bankroll, total_quantity, TimeInForce::Gtc, market)
                    .await
            }
            ExecutionPolicy::Hybrid => {
                let ioc_qty = (total_quantity * self.config.hybrid_ioc_fraction)
                    .max(self.config.min_order_quantity)
                    .min(total_quantity);
                let first_report = self
                    .execute_single_order(signal, bankroll, ioc_qty, TimeInForce::Ioc, market)
                    .await?;

                let remaining = (total_quantity - first_report.filled_qty).max(0.0);
                if remaining < self.config.min_order_quantity {
                    return Ok(first_report);
                }
                if matches!(first_report.status, OrderStatus::Rejected) {
                    return Ok(first_report);
                }

                match self
                    .execute_single_order(signal, bankroll, remaining, TimeInForce::Gtc, market)
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
        market: Option<&ScannedMarket>,
    ) -> Result<ExecutionReport, ExecutionError> {
        self.validate_quantity_bounds(quantity)?;
        let order = self.build_order(signal, quantity, time_in_force, market)?;
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
        if let Err(err) = record_order_intent(&ResearchCaptureConfig::from_env(), &order, signal) {
            eprintln!("research capture warning (order_intent): {err}");
        }
        self.save_state(&state)?;

        let mut report = if self.mode == ExecutionMode::Paper {
            // Simulate fill based on real market quote.
            // observed_price is the ask for Buy orders (set by the valuation pipeline).
            // IOC fills iff limit >= ask; fill price = ask, not limit.
            let ask = signal.observed_price;
            let limit = order.limit_price.unwrap_or(0.0);
            let (status, filled_qty, avg_fill_price) = if limit >= ask {
                (OrderStatus::Filled, order.quantity, Some(ask))
            } else {
                (OrderStatus::Canceled, 0.0, None)
            };
            ExecutionReport {
                order_id: format!("paper-{}", order.client_order_id),
                client_order_id: order.client_order_id.clone(),
                status,
                submitted_time_in_force: Some(order.time_in_force),
                filled_qty,
                avg_fill_price,
                fee_paid: 0.0,
                updated_at: Utc::now(),
            }
        } else {
            let (ack, initial) = match self.submit_with_retries(&order).await {
                Ok(v) => v,
                Err(err) => {
                    if let Err(rec_err) = record_order_error(
                        &ResearchCaptureConfig::from_env(),
                        &order,
                        "submit_error",
                        &err.to_string(),
                    ) {
                        eprintln!("research capture warning (submit_error): {rec_err}");
                    }
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
            if let Err(err) = record_order_ack(&ResearchCaptureConfig::from_env(), &order, &ack) {
                eprintln!("research capture warning (order_ack): {err}");
            }
            self.reconcile_post_submit(&order, ack, initial, order.time_in_force).await?
        };

        if report.submitted_time_in_force.is_none() {
            report.submitted_time_in_force = Some(order.time_in_force);
        }
        self.finalize_state_from_report(&mut state, &order, &report);
        self.append_journal("order_report", json!({"report": &report}))?;
        if let Err(err) = record_order_report(
            &ResearchCaptureConfig::from_env(),
            &order,
            &report,
            "terminal",
        ) {
            eprintln!("research capture warning (order_terminal): {err}");
        }
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
            let mut report = match self.client.get_order(&tracked.order_id).await {
                Ok(report) => report,
                Err(ExecutionError::Exchange(msg)) if exchange_not_found(&msg) => {
                    let _ = self.append_journal(
                        "reconcile_order_pruned",
                        json!({
                            "order_id": tracked.order_id,
                            "client_order_id": tracked.client_order_id,
                            "market_id": tracked.market_id,
                            "reason": "exchange_not_found"
                        }),
                    );
                    continue;
                }
                Err(err) => return Err(err),
            };
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
        market: Option<&ScannedMarket>,
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
            market_yes_bid_size: market.and_then(|m| m.yes_bid_size),
            market_yes_ask_size: market.and_then(|m| m.yes_ask_size),
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
        order: &OrderRequest,
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
            if let Err(err) = record_order_report(
                &ResearchCaptureConfig::from_env(),
                order,
                &report,
                "reconcile",
            ) {
                eprintln!("research capture warning (reconcile): {err}");
            }
            if matches!(
                report.status,
                OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Rejected
            ) {
                return Ok(report);
            }
        }

        if time_in_force == TimeInForce::Ioc {
            if let Err(err) = record_order_error(
                &ResearchCaptureConfig::from_env(),
                order,
                "cancel_requested",
                "ioc reconcile requested cancel",
            ) {
                eprintln!("research capture warning (cancel_request): {err}");
            }
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
        if text.trim().is_empty() {
            return Ok(RuntimeState::default_for_today());
        }
        let mut state: RuntimeState = match serde_json::from_str(&text) {
            Ok(state) => state,
            Err(err) => {
                quarantine_corrupt_file(path, "runtime_state");
                eprintln!(
                    "state recovery warning: failed to parse {}; moved corrupt file aside and reset state: {}",
                    path.display(),
                    err
                );
                return Ok(RuntimeState::default_for_today());
            }
        };
        state.schema_version = RUNTIME_STATE_SCHEMA_VERSION.to_string();
        state.roll_day_if_needed();
        Ok(state)
    }

    fn save_state(&self, state: &RuntimeState) -> Result<(), ExecutionError> {
        let path = Path::new(&self.config.state_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        }
        let data = serde_json::to_string_pretty(state).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let tmp_path = temp_path_for(path, "tmp");
        fs::write(&tmp_path, data).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        fs::rename(&tmp_path, path).map_err(|e| ExecutionError::Exchange(e.to_string()))
    }

    fn append_journal(&self, event: &str, payload: serde_json::Value) -> Result<(), ExecutionError> {
        let path = Path::new(&self.config.journal_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        }
        let entry = JournalEntry {
            schema_version: JOURNAL_SCHEMA_VERSION.to_string(),
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

fn exchange_not_found(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("404 not found")
        || lower.contains("\"code\":\"not_found\"")
        || lower.contains("code\":\"not_found")
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

fn temp_path_for(path: &Path, suffix: &str) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(format!(".{}", suffix));
    PathBuf::from(os)
}

fn quarantine_corrupt_file(path: &Path, label: &str) {
    let corrupt_path = temp_path_for(path, &format!("{}.corrupt", label));
    let _ = fs::rename(path, corrupt_path);
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
    #[serde(default = "default_runtime_state_schema_version")]
    schema_version: String,
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
            schema_version: RUNTIME_STATE_SCHEMA_VERSION.to_string(),
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

fn default_runtime_state_schema_version() -> String {
    RUNTIME_STATE_SCHEMA_VERSION.to_string()
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
    #[serde(default = "default_journal_schema_version")]
    schema_version: String,
    ts: DateTime<Utc>,
    event: String,
    payload: serde_json::Value,
}

fn default_journal_schema_version() -> String {
    JOURNAL_SCHEMA_VERSION.to_string()
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
            let entry: JournalEntry = match serde_json::from_str(line) {
                Ok(entry) => entry,
                Err(_) => continue,
            };
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
        if let Ok(state) = serde_json::from_str::<RuntimeState>(&text) {
            summary.state_day = Some(state.day);
            summary.state_daily_realized_pnl = Some(state.daily_realized_pnl);
            summary.state_open_exposure_notional = Some(state.open_exposure_notional);
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::execution::client::ExchangeClient;
    use crate::execution::types::{OrderType, Side};
    use tempfile::tempdir;

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
            signal_origin: Some("test".to_string()),
        }
    }

    #[test]
    fn corrupt_runtime_state_is_quarantined_and_reset() {
        let dir = tempdir().unwrap();
        let state_path = dir.path().join("runtime_state.json");
        fs::write(&state_path, "{not json").unwrap();

        let client: Arc<dyn ExchangeClient> = Arc::new(NoopClient);
        let mut cfg = EngineConfig::default();
        cfg.state_path = state_path.display().to_string();
        cfg.journal_path = dir.path().join("journal.jsonl").display().to_string();
        let engine = ExecutionEngine::new(client, cfg, ExecutionMode::Paper);

        let state = engine.load_state().unwrap();
        assert_eq!(state.schema_version, RUNTIME_STATE_SCHEMA_VERSION);
        assert_eq!(state.daily_realized_pnl, 0.0);
        assert!(!state_path.exists());
        assert!(dir.path().join("runtime_state.json.runtime_state.corrupt").exists());
    }

    #[test]
    fn save_state_uses_current_schema_version() {
        let dir = tempdir().unwrap();
        let state_path = dir.path().join("runtime_state.json");

        let client: Arc<dyn ExchangeClient> = Arc::new(NoopClient);
        let mut cfg = EngineConfig::default();
        cfg.state_path = state_path.display().to_string();
        cfg.journal_path = dir.path().join("journal.jsonl").display().to_string();
        let engine = ExecutionEngine::new(client, cfg, ExecutionMode::Paper);

        let state = RuntimeState::default_for_today();
        engine.save_state(&state).unwrap();

        let raw = fs::read_to_string(&state_path).unwrap();
        assert!(raw.contains(&format!("\"schema_version\": \"{}\"", RUNTIME_STATE_SCHEMA_VERSION)));
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

    #[tokio::test]
    async fn reconcile_prunes_exchange_not_found_orders() {
        let dir = tempdir().unwrap();
        let state_path = dir.path().join("runtime_state.json");
        let journal_path = dir.path().join("trade_journal.jsonl");

        let state = RuntimeState {
            schema_version: RUNTIME_STATE_SCHEMA_VERSION.to_string(),
            day: Utc::now().format("%Y-%m-%d").to_string(),
            daily_realized_pnl: 0.0,
            open_exposure_notional: 0.0,
            recent_order_unix_secs: Vec::new(),
            seen_client_order_ids: Vec::new(),
            open_orders: vec![OpenOrderState {
                order_id: "missing-order".to_string(),
                client_order_id: "cid-1".to_string(),
                market_id: "KXTEST".to_string(),
                notional: 10.0,
                created_at: Utc::now(),
            }],
        };
        fs::write(&state_path, serde_json::to_string(&state).unwrap()).unwrap();

        let mut cfg = EngineConfig::default();
        cfg.state_path = state_path.display().to_string();
        cfg.journal_path = journal_path.display().to_string();
        let engine = ExecutionEngine::new(Arc::new(MissingOrderClient), cfg, ExecutionMode::Live);

        engine.reconcile_open_orders().await.unwrap();

        let reloaded: RuntimeState = serde_json::from_str(&fs::read_to_string(&state_path).unwrap()).unwrap();
        assert!(reloaded.open_orders.is_empty());

        let journal = fs::read_to_string(&journal_path).unwrap();
        assert!(journal.contains("\"event\":\"reconcile_order_pruned\""));
    }

    #[test]
    fn summarize_performance_ignores_malformed_journal_lines() {
        let dir = tempdir().unwrap();
        let state_path = dir.path().join("runtime_state.json");
        let journal_path = dir.path().join("trade_journal.jsonl");

        let order = OrderRequest {
            client_order_id: "cid-1".to_string(),
            market_id: "KXTEST".to_string(),
            outcome_id: "yes".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            limit_price: Some(0.50),
            quantity: 10.0,
            time_in_force: TimeInForce::Ioc,
            created_at: Utc::now(),
            market_yes_bid_size: None,
            market_yes_ask_size: None,
        };
        let intent = serde_json::json!({
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "ts": Utc::now(),
            "event": "order_intent",
            "payload": {
                "order": order,
                "mode": "Live",
                "signal_edge_pct": 0.10
            }
        });
        let report = serde_json::json!({
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "ts": Utc::now(),
            "event": "order_report",
            "payload": {
                "report": {
                    "order_id": "oid-1",
                    "client_order_id": "cid-1",
                    "status": "Filled",
                    "submitted_time_in_force": "Ioc",
                    "filled_qty": 10.0,
                    "avg_fill_price": 0.50,
                    "fee_paid": 0.01,
                    "updated_at": Utc::now()
                }
            }
        });
        let content = format!("{{bad json\n{}\n{}\n", intent, report);
        fs::write(&journal_path, content).unwrap();
        fs::write(
            &state_path,
            serde_json::to_string(&RuntimeState::default_for_today()).unwrap(),
        )
        .unwrap();

        let summary = summarize_performance_paths(
            &state_path.display().to_string(),
            &journal_path.display().to_string(),
        )
        .unwrap();
        assert_eq!(summary.orders_reported, 1);
        assert_eq!(summary.filled_orders, 1);
        assert!(summary.traded_notional > 0.0);
    }

    struct NoopClient;

    struct MissingOrderClient;

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

    #[async_trait]
    impl ExchangeClient for MissingOrderClient {
        async fn place_order(&self, _request: &OrderRequest) -> Result<OrderAck, ExecutionError> {
            Err(ExecutionError::Exchange("noop".to_string()))
        }

        async fn get_order(&self, _order_id: &str) -> Result<ExecutionReport, ExecutionError> {
            Err(ExecutionError::Exchange(
                "GET /trade-api/v2/portfolio/orders/x failed (404 Not Found): {\"error\":{\"code\":\"not_found\"}}"
                    .to_string(),
            ))
        }

        async fn cancel_order(&self, _order_id: &str) -> Result<(), ExecutionError> {
            Ok(())
        }

        async fn smoke_test(&self) -> Result<(), ExecutionError> {
            Ok(())
        }
    }
}
