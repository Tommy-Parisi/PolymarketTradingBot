use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;

use crate::data::ws_delta::{KalshiWsDeltaIngestor, MarketDelta, WsDeltaConfig};
use crate::execution::types::ExecutionError;

#[derive(Debug, Clone)]
pub struct ScannerConfig {
    pub api_base_url: String,
    pub max_markets: usize,
    pub min_volume: f64,
    pub max_spread_cents: f64,
    pub ws_delta_window_secs: u64,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            api_base_url: std::env::var("KALSHI_API_BASE_URL")
                .unwrap_or_else(|_| "https://demo-api.kalshi.co".to_string()),
            max_markets: std::env::var("BOT_SCAN_MAX_MARKETS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1000)
                .max(10),
            min_volume: std::env::var("BOT_SCAN_MIN_VOLUME")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(1_000.0)
                .max(0.0),
            max_spread_cents: std::env::var("BOT_SCAN_MAX_SPREAD_CENTS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(8.0)
                .max(0.5),
            ws_delta_window_secs: std::env::var("BOT_SCAN_WS_DELTA_WINDOW_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(2)
                .max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScannedMarket {
    pub ticker: String,
    pub title: String,
    pub yes_bid_cents: Option<f64>,
    pub yes_ask_cents: Option<f64>,
    pub volume: f64,
    pub close_time: Option<DateTime<Utc>>,
}

impl ScannedMarket {
    pub fn spread_cents(&self) -> Option<f64> {
        match (self.yes_bid_cents, self.yes_ask_cents) {
            (Some(bid), Some(ask)) if ask >= bid => Some(ask - bid),
            _ => None,
        }
    }
}

pub struct KalshiMarketScanner {
    cfg: ScannerConfig,
    http: Client,
}

impl KalshiMarketScanner {
    pub fn new(cfg: ScannerConfig) -> Self {
        Self {
            cfg,
            http: Client::new(),
        }
    }

    pub async fn scan_snapshot(&self) -> Result<Vec<ScannedMarket>, ExecutionError> {
        let mut out = Vec::with_capacity(self.cfg.max_markets);
        let mut cursor: Option<String> = None;

        while out.len() < self.cfg.max_markets {
            let page_size = (self.cfg.max_markets - out.len()).min(1000).to_string();
            let mut req = self
                .http
                .get(format!("{}/trade-api/v2/markets", self.cfg.api_base_url))
                .query(&[("status", "open"), ("limit", page_size.as_str())]);
            if let Some(c) = &cursor {
                req = req.query(&[("cursor", c.as_str())]);
            }

            let resp = req
                .send()
                .await
                .map_err(|e| ExecutionError::RetryableExchange(e.to_string()))?;
            let status = resp.status();
            let body = resp
                .text()
                .await
                .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            if !status.is_success() {
                return Err(ExecutionError::Exchange(format!(
                    "GET /trade-api/v2/markets failed ({status}): {body}"
                )));
            }

            let page: MarketsResponse =
                serde_json::from_str(&body).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            out.extend(page.markets.into_iter().map(to_scanned_market));
            match page.cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => break,
            }
        }

        Ok(out)
    }

    pub async fn scan_snapshot_with_deltas(&self) -> Result<Vec<ScannedMarket>, ExecutionError> {
        let snapshot = self.scan_snapshot().await?;
        let mut index = to_index_map(snapshot);
        let tickers: Vec<String> = index.keys().cloned().collect();

        let delta_cfg = WsDeltaConfig {
            ws_url: std::env::var("KALSHI_WS_URL")
                .unwrap_or_else(|_| "wss://demo-api.kalshi.co/trade-api/ws/v2".to_string()),
            listen_window: Duration::from_secs(self.cfg.ws_delta_window_secs),
        };
        let ingestor = KalshiWsDeltaIngestor::new(delta_cfg);
        if let Ok(deltas) = ingestor.collect_deltas(&tickers).await {
            apply_deltas(&mut index, deltas);
        }

        Ok(index.into_values().collect())
    }

    pub fn select_for_valuation(&self, markets: Vec<ScannedMarket>) -> Vec<ScannedMarket> {
        let mut filtered: Vec<ScannedMarket> = markets
            .into_iter()
            .filter(|m| m.volume >= self.cfg.min_volume)
            .filter(|m| m.spread_cents().map(|s| s <= self.cfg.max_spread_cents).unwrap_or(false))
            .collect();

        filtered.sort_by(|a, b| b.volume.total_cmp(&a.volume));
        filtered.truncate(self.cfg.max_markets);
        filtered
    }
}

fn to_index_map(markets: Vec<ScannedMarket>) -> HashMap<String, ScannedMarket> {
    markets.into_iter().map(|m| (m.ticker.clone(), m)).collect()
}

fn apply_deltas(index: &mut HashMap<String, ScannedMarket>, deltas: Vec<MarketDelta>) {
    for delta in deltas {
        if let Some(m) = index.get_mut(&delta.ticker) {
            if delta.yes_bid_cents.is_some() {
                m.yes_bid_cents = delta.yes_bid_cents;
            }
            if delta.yes_ask_cents.is_some() {
                m.yes_ask_cents = delta.yes_ask_cents;
            }
            if let Some(traded) = delta.traded_count_delta {
                m.volume += traded.max(0.0);
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct MarketsResponse {
    #[serde(default)]
    markets: Vec<KalshiMarketWire>,
    cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KalshiMarketWire {
    ticker: String,
    title: Option<String>,
    #[serde(alias = "yes_bid", alias = "yesBid")]
    yes_bid: Option<f64>,
    #[serde(alias = "yes_ask", alias = "yesAsk")]
    yes_ask: Option<f64>,
    #[serde(alias = "yes_bid_dollars", alias = "yesBidDollars")]
    yes_bid_dollars: Option<String>,
    #[serde(alias = "yes_ask_dollars", alias = "yesAskDollars")]
    yes_ask_dollars: Option<String>,
    volume: Option<f64>,
    #[serde(alias = "close_time", alias = "closeTime")]
    close_time: Option<DateTime<Utc>>,
}

fn to_scanned_market(m: KalshiMarketWire) -> ScannedMarket {
    let yes_bid_cents = m
        .yes_bid_dollars
        .as_deref()
        .and_then(parse_price_dollars_to_cents)
        .or(m.yes_bid);
    let yes_ask_cents = m
        .yes_ask_dollars
        .as_deref()
        .and_then(parse_price_dollars_to_cents)
        .or(m.yes_ask);

    ScannedMarket {
        ticker: m.ticker,
        title: m.title.unwrap_or_default(),
        yes_bid_cents,
        yes_ask_cents,
        volume: m.volume.unwrap_or(0.0),
        close_time: m.close_time,
    }
}

fn parse_price_dollars_to_cents(raw: &str) -> Option<f64> {
    raw.parse::<f64>().ok().map(|d| d * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_computes_correctly() {
        let m = ScannedMarket {
            ticker: "X".to_string(),
            title: "".to_string(),
            yes_bid_cents: Some(43.0),
            yes_ask_cents: Some(49.0),
            volume: 10_000.0,
            close_time: None,
        };
        assert_eq!(m.spread_cents(), Some(6.0));
    }

    #[test]
    fn filter_selects_liquid_tight_markets() {
        let scanner = KalshiMarketScanner::new(ScannerConfig {
            max_markets: 10,
            min_volume: 100.0,
            max_spread_cents: 8.0,
            ..ScannerConfig::default()
        });
        let markets = vec![
            ScannedMarket {
                ticker: "A".to_string(),
                title: "".to_string(),
                yes_bid_cents: Some(40.0),
                yes_ask_cents: Some(45.0),
                volume: 1_000.0,
                close_time: None,
            },
            ScannedMarket {
                ticker: "B".to_string(),
                title: "".to_string(),
                yes_bid_cents: Some(20.0),
                yes_ask_cents: Some(40.0),
                volume: 5_000.0,
                close_time: None,
            },
        ];
        let selected = scanner.select_for_valuation(markets);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].ticker, "A");
    }

    #[test]
    fn parses_dollar_price_fields_to_cents() {
        let wire = KalshiMarketWire {
            ticker: "KXTEST".to_string(),
            title: Some("Test".to_string()),
            yes_bid: None,
            yes_ask: None,
            yes_bid_dollars: Some("0.43".to_string()),
            yes_ask_dollars: Some("0.49".to_string()),
            volume: Some(1234.0),
            close_time: None,
        };
        let parsed = to_scanned_market(wire);
        assert_eq!(parsed.yes_bid_cents, Some(43.0));
        assert_eq!(parsed.yes_ask_cents, Some(49.0));
    }

    #[test]
    fn apply_delta_updates_quote_and_volume() {
        let mut map = to_index_map(vec![ScannedMarket {
            ticker: "KXBTC".to_string(),
            title: "x".to_string(),
            yes_bid_cents: Some(40.0),
            yes_ask_cents: Some(45.0),
            volume: 100.0,
            close_time: None,
        }]);
        apply_deltas(
            &mut map,
            vec![MarketDelta {
                ticker: "KXBTC".to_string(),
                yes_bid_cents: Some(41.0),
                yes_ask_cents: Some(46.0),
                traded_count_delta: Some(12.0),
            }],
        );
        let m = map.get("KXBTC").expect("market should exist");
        assert_eq!(m.yes_bid_cents, Some(41.0));
        assert_eq!(m.yes_ask_cents, Some(46.0));
        assert_eq!(m.volume, 112.0);
    }
}
