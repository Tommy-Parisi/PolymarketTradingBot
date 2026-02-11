use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;

use crate::execution::client::ExchangeClient;
use crate::execution::types::{ExecutionError, ExecutionReport, OrderAck, OrderRequest, OrderStatus, Side};

#[derive(Debug, Clone)]
pub struct PaperSimConfig {
    pub fee_bps: f64,
    pub max_slippage_bps: f64,
    pub min_latency_ms: u64,
    pub max_latency_ms: u64,
}

impl Default for PaperSimConfig {
    fn default() -> Self {
        Self {
            fee_bps: 12.0,
            max_slippage_bps: 35.0,
            min_latency_ms: 40,
            max_latency_ms: 220,
        }
    }
}

pub struct PaperSimClient {
    cfg: PaperSimConfig,
    inner: Mutex<SimBook>,
}

#[derive(Default)]
struct SimBook {
    seq: u64,
    orders: HashMap<String, SimOrder>,
}

struct SimOrder {
    request: OrderRequest,
    ready_at: Instant,
    fill_ratio: f64,
    slippage_bps: f64,
    polls_after_ready: u32,
    canceled: bool,
}

impl PaperSimClient {
    pub fn new(cfg: PaperSimConfig) -> Self {
        Self {
            cfg,
            inner: Mutex::new(SimBook::default()),
        }
    }
}

#[async_trait]
impl ExchangeClient for PaperSimClient {
    async fn place_order(&self, request: &OrderRequest) -> Result<OrderAck, ExecutionError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| ExecutionError::Exchange("paper sim lock poisoned".to_string()))?;
        inner.seq += 1;
        let order_id = format!("sim-{}", inner.seq);

        let seed = stable_hash(&(request.client_order_id.as_str(), request.market_id.as_str()));
        let fill_ratio_bucket = (seed % 4) as usize;
        let fill_ratio = [0.0, 0.35, 0.65, 1.0][fill_ratio_bucket];

        let slip_unit = (seed % 10_000) as f64 / 10_000.0;
        let slippage_bps = (slip_unit * 2.0 - 1.0) * self.cfg.max_slippage_bps;

        let lat_span = self.cfg.max_latency_ms.saturating_sub(self.cfg.min_latency_ms).max(1);
        let latency_ms = self.cfg.min_latency_ms + (seed % lat_span as u64);
        let accepted = Instant::now();
        let sim_order = SimOrder {
            request: request.clone(),
            ready_at: accepted + Duration::from_millis(latency_ms),
            fill_ratio,
            slippage_bps,
            polls_after_ready: 0,
            canceled: false,
        };
        inner.orders.insert(order_id.clone(), sim_order);

        Ok(OrderAck {
            order_id,
            client_order_id: request.client_order_id.clone(),
            accepted_at: Utc::now(),
        })
    }

    async fn get_order(&self, order_id: &str) -> Result<ExecutionReport, ExecutionError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| ExecutionError::Exchange("paper sim lock poisoned".to_string()))?;
        let now = Instant::now();
        let order = inner
            .orders
            .get_mut(order_id)
            .ok_or_else(|| ExecutionError::Exchange(format!("order not found: {order_id}")))?;

        let req = &order.request;
        if order.canceled {
            return Ok(make_report(order_id, req, OrderStatus::Canceled, 0.0, None, 0.0));
        }
        if now < order.ready_at {
            return Ok(make_report(order_id, req, OrderStatus::New, 0.0, None, 0.0));
        }

        order.polls_after_ready += 1;

        let base_price = req.limit_price.unwrap_or(0.5).clamp(0.01, 0.99);
        let signed_bps = match req.side {
            Side::Buy => order.slippage_bps,
            Side::Sell => -order.slippage_bps,
        };
        let fill_price = (base_price * (1.0 + signed_bps / 10_000.0)).clamp(0.01, 0.99);

        if order.fill_ratio <= f64::EPSILON {
            return Ok(make_report(order_id, req, OrderStatus::Canceled, 0.0, None, 0.0));
        }

        let filled_qty = req.quantity * order.fill_ratio;
        let fee = filled_qty * fill_price * (self.cfg.fee_bps / 10_000.0);

        if order.fill_ratio < 1.0 {
            if order.polls_after_ready == 1 {
                return Ok(make_report(
                    order_id,
                    req,
                    OrderStatus::PartiallyFilled,
                    filled_qty,
                    Some(fill_price),
                    fee,
                ));
            }
            return Ok(make_report(
                order_id,
                req,
                OrderStatus::Canceled,
                filled_qty,
                Some(fill_price),
                fee,
            ));
        }

        Ok(make_report(
            order_id,
            req,
            OrderStatus::Filled,
            filled_qty,
            Some(fill_price),
            fee,
        ))
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| ExecutionError::Exchange("paper sim lock poisoned".to_string()))?;
        let Some(order) = inner.orders.get_mut(order_id) else {
            return Err(ExecutionError::Exchange(format!("order not found: {order_id}")));
        };
        order.canceled = true;
        Ok(())
    }

    async fn smoke_test(&self) -> Result<(), ExecutionError> {
        Ok(())
    }
}

fn make_report(
    order_id: &str,
    req: &OrderRequest,
    status: OrderStatus,
    filled_qty: f64,
    avg_fill_price: Option<f64>,
    fee_paid: f64,
) -> ExecutionReport {
    let _ = req;
    ExecutionReport {
        order_id: order_id.to_string(),
        client_order_id: req.client_order_id.clone(),
        status,
        filled_qty,
        avg_fill_price,
        fee_paid,
        updated_at: Utc::now(),
    }
}

fn stable_hash<T: Hash>(v: &T) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut s);
    s.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::types::{OrderType, TimeInForce};

    fn req(id: &str) -> OrderRequest {
        OrderRequest {
            client_order_id: id.to_string(),
            market_id: "KXTEST".to_string(),
            outcome_id: "yes".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            limit_price: Some(0.5),
            quantity: 100.0,
            time_in_force: TimeInForce::Ioc,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn deterministic_for_same_order_id() {
        let sim = PaperSimClient::new(PaperSimConfig::default());
        let a1 = sim.place_order(&req("X")).await.expect("ack1");
        let r1 = sim.get_order(&a1.order_id).await.expect("r1");
        tokio::time::sleep(Duration::from_millis(300)).await;
        let r1b = sim.get_order(&a1.order_id).await.expect("r1b");

        let a2 = sim.place_order(&req("X")).await.expect("ack2");
        tokio::time::sleep(Duration::from_millis(300)).await;
        let r2 = sim.get_order(&a2.order_id).await.expect("r2");

        assert_eq!(r1.client_order_id, r2.client_order_id);
        assert_eq!(r1b.avg_fill_price.is_some(), r2.avg_fill_price.is_some());
    }
}
