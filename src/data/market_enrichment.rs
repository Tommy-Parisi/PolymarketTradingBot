use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;

use crate::data::market_scanner::ScannedMarket;
use crate::execution::types::ExecutionError;

#[derive(Debug, Clone)]
pub struct EnrichmentConfig {
    pub ttl_secs: u64,
    pub noaa_point: String,
    pub sports_injury_api_url: Option<String>,
    pub crypto_sentiment_api_url: Option<String>,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            ttl_secs: 300,
            noaa_point: std::env::var("NOAA_POINT").unwrap_or_else(|_| "39.7456,-97.0892".to_string()),
            sports_injury_api_url: std::env::var("SPORTS_INJURY_API_URL").ok(),
            crypto_sentiment_api_url: std::env::var("CRYPTO_SENTIMENT_API_URL")
                .ok()
                .or_else(|| Some("https://api.alternative.me/fng/?limit=1".to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketEnrichment {
    pub ticker: String,
    pub vertical: MarketVertical,
    pub weather_signal: Option<f64>,
    pub sports_injury_signal: Option<f64>,
    pub crypto_sentiment_signal: Option<f64>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketVertical {
    Weather,
    Sports,
    Crypto,
    Other,
}

pub struct MarketEnricher {
    cfg: EnrichmentConfig,
    http: Client,
    cache: Mutex<HashMap<String, CacheEntry>>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    expires_at: Instant,
    data: MarketEnrichment,
}

impl MarketEnricher {
    pub fn new(cfg: EnrichmentConfig) -> Self {
        Self {
            cfg,
            http: Client::new(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub async fn enrich_batch(
        &self,
        markets: &[ScannedMarket],
    ) -> Result<Vec<MarketEnrichment>, ExecutionError> {
        let mut out = Vec::with_capacity(markets.len());
        for m in markets {
            out.push(self.enrich_market(m).await?);
        }
        Ok(out)
    }

    pub async fn enrich_market(&self, market: &ScannedMarket) -> Result<MarketEnrichment, ExecutionError> {
        if let Some(cached) = self.get_cached(&market.ticker) {
            return Ok(cached);
        }

        let vertical = detect_vertical(market);
        let (weather_signal, sports_injury_signal, crypto_sentiment_signal) = match vertical {
            MarketVertical::Weather => (self.fetch_noaa_weather_signal().await?, None, None),
            MarketVertical::Sports => (None, self.fetch_sports_injury_signal().await?, None),
            MarketVertical::Crypto => (None, None, self.fetch_crypto_sentiment_signal().await?),
            MarketVertical::Other => (None, None, None),
        };

        let enriched = MarketEnrichment {
            ticker: market.ticker.clone(),
            vertical,
            weather_signal,
            sports_injury_signal,
            crypto_sentiment_signal,
            generated_at: Utc::now(),
        };
        self.put_cached(&market.ticker, enriched.clone());
        Ok(enriched)
    }

    fn get_cached(&self, ticker: &str) -> Option<MarketEnrichment> {
        let guard = self.cache.lock().ok()?;
        let e = guard.get(ticker)?;
        if Instant::now() < e.expires_at {
            Some(e.data.clone())
        } else {
            None
        }
    }

    fn put_cached(&self, ticker: &str, data: MarketEnrichment) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.insert(
                ticker.to_string(),
                CacheEntry {
                    expires_at: Instant::now() + Duration::from_secs(self.cfg.ttl_secs),
                    data,
                },
            );
        }
    }

    async fn fetch_noaa_weather_signal(&self) -> Result<Option<f64>, ExecutionError> {
        let points_url = format!("https://api.weather.gov/points/{}", self.cfg.noaa_point);
        let points = self
            .http
            .get(points_url)
            .header("User-Agent", "event-trading-bot (support@example.com)")
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(e.to_string()))?;
        let points_status = points.status();
        let points_text = points
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !points_status.is_success() {
            return Err(ExecutionError::Exchange(format!(
                "NOAA points failed ({points_status}): {points_text}"
            )));
        }
        let parsed_points: NoaaPointsResponse =
            serde_json::from_str(&points_text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let forecast_url = match parsed_points.properties.and_then(|p| p.forecast_hourly) {
            Some(url) => url,
            None => return Ok(None),
        };

        let forecast = self
            .http
            .get(forecast_url)
            .header("User-Agent", "event-trading-bot (support@example.com)")
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(e.to_string()))?;
        let forecast_status = forecast.status();
        let forecast_text = forecast
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !forecast_status.is_success() {
            return Err(ExecutionError::Exchange(format!(
                "NOAA forecast failed ({forecast_status}): {forecast_text}"
            )));
        }
        let parsed_forecast: NoaaForecastResponse =
            serde_json::from_str(&forecast_text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;

        let temp = parsed_forecast
            .properties
            .and_then(|p| p.periods)
            .and_then(|mut v| v.drain(..).next())
            .and_then(|p| p.temperature)
            .map(|t| (t / 100.0).clamp(-1.0, 1.0));
        Ok(temp)
    }

    async fn fetch_sports_injury_signal(&self) -> Result<Option<f64>, ExecutionError> {
        let Some(url) = &self.cfg.sports_injury_api_url else {
            return Ok(None);
        };
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !status.is_success() {
            return Err(ExecutionError::Exchange(format!(
                "sports injury feed failed ({status}): {text}"
            )));
        }
        let parsed: GenericSignalResponse =
            serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        Ok(parsed.signal)
    }

    async fn fetch_crypto_sentiment_signal(&self) -> Result<Option<f64>, ExecutionError> {
        let Some(url) = &self.cfg.crypto_sentiment_api_url else {
            return Ok(None);
        };
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !status.is_success() {
            return Err(ExecutionError::Exchange(format!(
                "crypto sentiment feed failed ({status}): {text}"
            )));
        }

        if let Ok(parsed) = serde_json::from_str::<FearGreedResponse>(&text) {
            let score = parsed
                .data
                .and_then(|mut d| d.drain(..).next())
                .and_then(|r| r.value)
                .map(|v| ((v / 100.0) * 2.0) - 1.0);
            return Ok(score);
        }

        let parsed: GenericSignalResponse =
            serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        Ok(parsed.signal)
    }
}

fn detect_vertical(market: &ScannedMarket) -> MarketVertical {
    let text = format!("{} {}", market.ticker, market.title).to_ascii_lowercase();
    if text.contains("temp") || text.contains("weather") || text.contains("rain") || text.contains("snow") {
        return MarketVertical::Weather;
    }
    if text.contains("nba")
        || text.contains("nfl")
        || text.contains("mlb")
        || text.contains("nhl")
        || text.contains("soccer")
        || text.contains("injury")
    {
        return MarketVertical::Sports;
    }
    if text.contains("btc")
        || text.contains("bitcoin")
        || text.contains("eth")
        || text.contains("crypto")
        || text.contains("sol")
    {
        return MarketVertical::Crypto;
    }
    MarketVertical::Other
}

#[derive(Debug, Deserialize)]
struct NoaaPointsResponse {
    properties: Option<NoaaPointsProperties>,
}

#[derive(Debug, Deserialize)]
struct NoaaPointsProperties {
    #[serde(alias = "forecastHourly", alias = "forecast_hourly")]
    forecast_hourly: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NoaaForecastResponse {
    properties: Option<NoaaForecastProperties>,
}

#[derive(Debug, Deserialize)]
struct NoaaForecastProperties {
    periods: Option<Vec<NoaaForecastPeriod>>,
}

#[derive(Debug, Deserialize)]
struct NoaaForecastPeriod {
    temperature: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct GenericSignalResponse {
    signal: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct FearGreedResponse {
    data: Option<Vec<FearGreedRecord>>,
}

#[derive(Debug, Deserialize)]
struct FearGreedRecord {
    value: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_market_verticals() {
        let weather = ScannedMarket {
            ticker: "KXWEATHER-NYC".to_string(),
            title: "NYC high temp above 90F".to_string(),
            yes_bid_cents: None,
            yes_ask_cents: None,
            volume: 0.0,
            close_time: None,
        };
        let crypto = ScannedMarket {
            ticker: "KXBTC-TEST".to_string(),
            title: "Bitcoin above 120k".to_string(),
            yes_bid_cents: None,
            yes_ask_cents: None,
            volume: 0.0,
            close_time: None,
        };
        assert_eq!(detect_vertical(&weather), MarketVertical::Weather);
        assert_eq!(detect_vertical(&crypto), MarketVertical::Crypto);
    }
}
