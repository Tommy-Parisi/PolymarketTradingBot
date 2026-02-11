use std::collections::HashMap;

use reqwest::Client;
use serde::Deserialize;

use crate::execution::types::ExecutionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionMode {
    BestEffort,
    Strict,
}

pub struct KalshiMarketMapper {
    api_base_url: String,
    aliases: HashMap<String, String>,
    http: Client,
}

impl KalshiMarketMapper {
    pub fn from_env() -> Self {
        let api_base_url =
            std::env::var("KALSHI_API_BASE_URL").unwrap_or_else(|_| "https://demo-api.kalshi.co".to_string());
        let aliases = parse_aliases(std::env::var("KALSHI_MARKET_ALIASES").unwrap_or_default().as_str());
        Self {
            api_base_url,
            aliases,
            http: Client::new(),
        }
    }

    pub async fn resolve_market_ticker(&self, market_input: &str) -> Result<String, ExecutionError> {
        let input = market_input.trim();
        if input.is_empty() {
            return Err(ExecutionError::Exchange("empty market input".to_string()));
        }

        let alias_key = normalize(input);
        if let Some(ticker) = self.aliases.get(&alias_key) {
            return Ok(ticker.clone());
        }

        if looks_like_ticker(input) {
            return Ok(input.to_ascii_uppercase());
        }

        let candidates = self.fetch_open_markets().await?;
        choose_market_ticker(input, &candidates)
    }

    async fn fetch_open_markets(&self) -> Result<Vec<KalshiMarket>, ExecutionError> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..5 {
            let mut req = self
                .http
                .get(format!("{}/trade-api/v2/markets", self.api_base_url))
                .query(&[("status", "open"), ("limit", "200")]);
            if let Some(c) = &cursor {
                req = req.query(&[("cursor", c.as_str())]);
            }

            let resp = req
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
                    "GET /trade-api/v2/markets failed ({status}): {text}"
                )));
            }
            let page: MarketsResponse =
                serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            all.extend(page.markets);
            match page.cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => break,
            }
        }
        Ok(all)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct KalshiMarket {
    ticker: String,
    title: Option<String>,
    subtitle: Option<String>,
    #[serde(alias = "event_ticker", alias = "eventTicker")]
    event_ticker: Option<String>,
    #[serde(alias = "series_ticker", alias = "seriesTicker")]
    series_ticker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MarketsResponse {
    #[serde(default)]
    markets: Vec<KalshiMarket>,
    cursor: Option<String>,
}

fn choose_market_ticker(input: &str, markets: &[KalshiMarket]) -> Result<String, ExecutionError> {
    let key = normalize(input);
    let mut exact = Vec::new();
    let mut fuzzy = Vec::new();

    for market in markets {
        let fields = market_fields(market);
        if fields.iter().any(|f| normalize(f) == key) {
            exact.push(market.ticker.clone());
        } else if fields.iter().any(|f| normalize(f).contains(&key)) {
            fuzzy.push(market.ticker.clone());
        }
    }

    if exact.len() == 1 {
        return Ok(exact[0].clone());
    }
    if exact.len() > 1 {
        return Err(ExecutionError::Exchange(format!(
            "ambiguous market input '{input}', exact matches: {}",
            exact.into_iter().take(5).collect::<Vec<_>>().join(", ")
        )));
    }
    if fuzzy.len() == 1 {
        return Ok(fuzzy[0].clone());
    }
    if fuzzy.len() > 1 {
        return Err(ExecutionError::Exchange(format!(
            "ambiguous market input '{input}', fuzzy matches: {}",
            fuzzy.into_iter().take(5).collect::<Vec<_>>().join(", ")
        )));
    }

    Err(ExecutionError::Exchange(format!(
        "unable to resolve market input '{input}' to a Kalshi ticker"
    )))
}

fn market_fields(m: &KalshiMarket) -> Vec<&str> {
    let mut out = Vec::new();
    out.push(m.ticker.as_str());
    if let Some(v) = &m.title {
        out.push(v.as_str());
    }
    if let Some(v) = &m.subtitle {
        out.push(v.as_str());
    }
    if let Some(v) = &m.event_ticker {
        out.push(v.as_str());
    }
    if let Some(v) = &m.series_ticker {
        out.push(v.as_str());
    }
    out
}

fn parse_aliases(raw: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in raw.split(',') {
        let trimmed = pair.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((k, v)) = trimmed.split_once('=') {
            let key = normalize(k);
            let value = v.trim().to_ascii_uppercase();
            if !key.is_empty() && !value.is_empty() {
                map.insert(key, value);
            }
        }
    }
    map
}

fn looks_like_ticker(input: &str) -> bool {
    input.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') && input.len() >= 6
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect::<String>()
}

pub fn resolution_mode_from_env() -> ResolutionMode {
    match std::env::var("BOT_MARKET_RESOLUTION")
        .unwrap_or_else(|_| "best_effort".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "strict" => ResolutionMode::Strict,
        _ => ResolutionMode::BestEffort,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_alias_map() {
        let m = parse_aliases("btc_120k=KXBTC-2026-B120K, weather_nyc = KXWTHR-NYC-90F");
        assert_eq!(m.get("btc120k").map(String::as_str), Some("KXBTC-2026-B120K"));
        assert_eq!(m.get("weathernyc").map(String::as_str), Some("KXWTHR-NYC-90F"));
    }

    #[test]
    fn resolves_exact_ticker_field_match() {
        let markets = vec![KalshiMarket {
            ticker: "KXBTC-26DEC31-B120000".to_string(),
            title: Some("Bitcoin above 120k by year end".to_string()),
            subtitle: None,
            event_ticker: Some("KXBTC-26DEC31".to_string()),
            series_ticker: None,
        }];
        let resolved = choose_market_ticker("KXBTC-26DEC31-B120000", &markets).expect("should resolve");
        assert_eq!(resolved, "KXBTC-26DEC31-B120000");
    }

    #[test]
    fn resolves_title_fuzzy_single_match() {
        let markets = vec![KalshiMarket {
            ticker: "KXWEATHER-NYC-90F".to_string(),
            title: Some("NYC high temperature above 90F".to_string()),
            subtitle: None,
            event_ticker: None,
            series_ticker: None,
        }];
        let resolved = choose_market_ticker("NYC high temperature above 90", &markets).expect("should resolve");
        assert_eq!(resolved, "KXWEATHER-NYC-90F");
    }
}
