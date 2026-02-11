use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::time::sleep;

use crate::execution::client::ExchangeClient;
use crate::execution::types::{
    new_client_order_id, EngineConfig, ExecutionError, ExecutionReport, OrderRequest, OrderType, Side,
    OrderStatus, TimeInForce, TradeSignal,
};

pub struct ExecutionEngine {
    client: Arc<dyn ExchangeClient>,
    config: EngineConfig,
    mode: ExecutionMode,
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

        let quantity = self.compute_position_size(signal, bankroll)?;
        let order = self.build_order(signal, quantity)?;

        if self.mode == ExecutionMode::Paper {
            return Ok(ExecutionReport {
                order_id: format!("paper-{}", order.client_order_id),
                client_order_id: order.client_order_id,
                status: OrderStatus::Filled,
                filled_qty: order.quantity,
                avg_fill_price: order.limit_price,
                fee_paid: 0.0,
                updated_at: Utc::now(),
            });
        }

        self.submit_with_retries(&order).await
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

    fn build_order(&self, signal: &TradeSignal, quantity: f64) -> Result<OrderRequest, ExecutionError> {
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
            time_in_force: TimeInForce::Ioc,
            created_at: Utc::now(),
        })
    }

    async fn submit_with_retries(
        &self,
        order: &OrderRequest,
    ) -> Result<ExecutionReport, ExecutionError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.client.place_order(order).await {
                Ok(ack) => return self.client.get_order(&ack.order_id).await,
                Err(ExecutionError::RetryableExchange(err)) if attempt <= self.config.max_retries => {
                    let backoff_ms = 250 * attempt as u64;
                    sleep(Duration::from_millis(backoff_ms)).await;
                    let _ = err;
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
    }
}

fn compute_limit_price(side: Side, observed_price: f64) -> f64 {
    match side {
        Side::Buy => (observed_price + 0.01).min(1.0),
        Side::Sell => (observed_price - 0.01).max(0.0),
    }
}

fn approximate_kelly_fraction(signal: &TradeSignal) -> f64 {
    let p = signal.fair_price.clamp(0.01, 0.99);
    let q = 1.0 - p;
    let b = (1.0 - signal.observed_price).max(0.01) / signal.observed_price.max(0.01);
    let raw = ((b * p) - q) / b;
    raw.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::execution::client::ExchangeClient;
    use crate::execution::types::{ExecutionReport, OrderAck, OrderRequest};
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

    #[test]
    fn rejects_low_edge_signal() {
        let cfg = EngineConfig::default();
        let engine = ExecutionEngine {
            client: Arc::new(NoopClient),
            config: cfg,
            mode: ExecutionMode::Paper,
        };
        let mut signal = base_signal();
        signal.edge_pct = 0.04;
        let result = engine.validate_signal(&signal);
        assert!(matches!(result, Err(ExecutionError::EdgeTooSmall { .. })));
    }

    #[test]
    fn rejects_stale_signal() {
        let cfg = EngineConfig::default();
        let engine = ExecutionEngine {
            client: Arc::new(NoopClient),
            config: cfg,
            mode: ExecutionMode::Paper,
        };
        let mut signal = base_signal();
        signal.signal_timestamp = Utc::now() - chrono::Duration::seconds(120);
        let result = engine.validate_signal(&signal);
        assert!(matches!(result, Err(ExecutionError::StaleSignal { .. })));
    }

    #[test]
    fn position_size_is_capped_by_bankroll_and_market_limit() {
        let cfg = EngineConfig::default();
        let engine = ExecutionEngine {
            client: Arc::new(NoopClient),
            config: cfg,
            mode: ExecutionMode::Paper,
        };
        let signal = base_signal();
        let size = engine
            .compute_position_size(&signal, 1_000_000.0)
            .expect("expected valid size");

        // Notional cap is 1000, so quantity should not imply notional larger than that by much.
        assert!(size * signal.observed_price <= 1_000.0 + 1e-6);
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
    }
}
