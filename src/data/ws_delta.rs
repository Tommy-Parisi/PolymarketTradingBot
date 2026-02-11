use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::execution::types::ExecutionError;

#[derive(Debug, Clone)]
pub struct WsDeltaConfig {
    pub ws_url: String,
    pub listen_window: Duration,
}

impl Default for WsDeltaConfig {
    fn default() -> Self {
        Self {
            ws_url: std::env::var("KALSHI_WS_URL")
                .unwrap_or_else(|_| "wss://demo-api.kalshi.co/trade-api/ws/v2".to_string()),
            listen_window: Duration::from_secs(2),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MarketDelta {
    pub ticker: String,
    pub yes_bid_cents: Option<f64>,
    pub yes_ask_cents: Option<f64>,
    pub traded_count_delta: Option<f64>,
}

pub struct KalshiWsDeltaIngestor {
    cfg: WsDeltaConfig,
}

impl KalshiWsDeltaIngestor {
    pub fn new(cfg: WsDeltaConfig) -> Self {
        Self { cfg }
    }

    pub async fn collect_deltas(&self, tickers: &[String]) -> Result<Vec<MarketDelta>, ExecutionError> {
        if self.cfg.listen_window.is_zero() || tickers.is_empty() {
            return Ok(Vec::new());
        }

        let (mut socket, _) = connect_async(&self.cfg.ws_url)
            .await
            .map_err(|e| ExecutionError::RetryableExchange(format!("ws connect failed: {e}")))?;

        let subscribe = json!({
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker_v2", "trade"],
                "market_tickers": tickers,
            }
        });
        socket
            .send(Message::Text(subscribe.to_string().into()))
            .await
            .map_err(|e| ExecutionError::RetryableExchange(format!("ws subscribe failed: {e}")))?;

        let deadline = Instant::now() + self.cfg.listen_window;
        let mut deltas: Vec<MarketDelta> = Vec::new();

        loop {
            if Instant::now() >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let next_msg = tokio::time::timeout(remaining, socket.next()).await;
            let msg = match next_msg {
                Ok(Some(Ok(m))) => m,
                Ok(Some(Err(e))) => {
                    return Err(ExecutionError::RetryableExchange(format!(
                        "ws read failed: {e}"
                    )))
                }
                Ok(None) => break,
                Err(_) => break,
            };

            if let Message::Text(text) = msg {
                if let Some(delta) = parse_ws_delta(&text) {
                    deltas.push(delta);
                }
            }
        }

        let _ = socket.close(None).await;
        Ok(deltas)
    }
}

pub fn parse_ws_delta(raw: &str) -> Option<MarketDelta> {
    let v: Value = serde_json::from_str(raw).ok()?;
    let msg_type = v
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| v.get("channel").and_then(Value::as_str))?;
    let data = v.get("msg").unwrap_or(&v);

    let ticker = data
        .get("market_ticker")
        .or_else(|| data.get("marketTicker"))
        .and_then(Value::as_str)?
        .to_string();

    if msg_type.eq_ignore_ascii_case("ticker_v2") || msg_type.eq_ignore_ascii_case("ticker") {
        return Some(MarketDelta {
            ticker,
            yes_bid_cents: parse_price_cents(
                data.get("yes_bid_dollars")
                    .or_else(|| data.get("yesBidDollars"))
                    .or_else(|| data.get("yes_bid"))
                    .or_else(|| data.get("yesBid")),
            ),
            yes_ask_cents: parse_price_cents(
                data.get("yes_ask_dollars")
                    .or_else(|| data.get("yesAskDollars"))
                    .or_else(|| data.get("yes_ask"))
                    .or_else(|| data.get("yesAsk")),
            ),
            traded_count_delta: None,
        });
    }

    if msg_type.eq_ignore_ascii_case("trade") {
        let traded_count_delta = data
            .get("count")
            .or_else(|| data.get("size"))
            .or_else(|| data.get("qty"))
            .and_then(value_as_f64);
        return Some(MarketDelta {
            ticker,
            yes_bid_cents: None,
            yes_ask_cents: None,
            traded_count_delta,
        });
    }

    None
}

fn parse_price_cents(v: Option<&Value>) -> Option<f64> {
    let v = v?;
    if let Some(s) = v.as_str() {
        if let Ok(dollars) = s.parse::<f64>() {
            return Some(dollars * 100.0);
        }
    }
    value_as_f64(v)
}

fn value_as_f64(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_f64() {
        return Some(n);
    }
    v.as_str()?.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ticker_v2_payload() {
        let raw = r#"{"type":"ticker_v2","msg":{"market_ticker":"KXBTC-TEST","yes_bid_dollars":"0.47","yes_ask_dollars":"0.53"}}"#;
        let d = parse_ws_delta(raw).expect("delta expected");
        assert_eq!(d.ticker, "KXBTC-TEST");
        assert_eq!(d.yes_bid_cents, Some(47.0));
        assert_eq!(d.yes_ask_cents, Some(53.0));
    }

    #[test]
    fn parses_trade_payload() {
        let raw = r#"{"type":"trade","msg":{"market_ticker":"KXBTC-TEST","count":"22"}}"#;
        let d = parse_ws_delta(raw).expect("delta expected");
        assert_eq!(d.ticker, "KXBTC-TEST");
        assert_eq!(d.traded_count_delta, Some(22.0));
    }
}
