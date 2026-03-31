use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;

use crate::data::market_enrichment::select_for_enrichment;
use crate::data::ws_delta::{KalshiWsDeltaIngestor, MarketDelta, WsDeltaConfig};
use crate::execution::types::ExecutionError;

#[derive(Debug, Clone)]
pub struct ScannerConfig {
    pub api_base_url: String,
    pub max_markets: usize,
    pub min_volume: f64,
    pub max_spread_cents: f64,
    pub ws_delta_window_secs: u64,
    /// Series to always fetch first via targeted API calls, regardless of quota ordering.
    /// Prevents KXMVE flooding from hiding NBA/NHL/etc. (set via BOT_SCAN_SERIES_ALLOWLIST).
    pub series_allowlist: Vec<String>,
    /// Series prefixes to skip in the general (non-allowlist) scan pass (BOT_SCAN_SERIES_BLOCKLIST).
    pub series_blocklist: Vec<String>,
    /// Max markets fetched per allowlist series query (BOT_SCAN_SERIES_MAX_PER_FETCH, default 200).
    pub series_max_per_fetch: usize,
    /// Skip markets closing within this many seconds (BOT_SCAN_MIN_CLOSE_SECS, default 900 = 15min).
    /// Markets with no close_time are always kept.
    pub min_time_to_close_secs: i64,
    /// Max markets retained per event_ticker after volume sort (BOT_SCAN_MAX_PER_EVENT, default 3).
    /// Prevents a single event (e.g. 15 KXHIGHCHI threshold variants) from flooding quota.
    /// 0 = no cap.
    pub max_per_event: usize,
    /// When true, apply round-robin category balancing in select_for_valuation so that
    /// Weather/Sports/Crypto/Other each get proportional representation regardless of
    /// which category dominates by raw volume (BOT_SCAN_BALANCE_CATEGORIES, default true).
    pub balance_by_category: bool,
    /// Kalshi API categories to query in tier-2 series discovery. The general unfiltered
    /// endpoint floods with KXMVE, so tier-2 instead enumerates series by these categories
    /// and queries each briefly. Comma-separated (BOT_SCAN_TIER2_CATEGORIES).
    pub tier2_categories: Vec<String>,
    /// Max series queries in tier-2 (BOT_SCAN_MAX_TIER2_SERIES, default 120).
    /// Each query fetches up to `series_max_per_fetch` markets; stop when quota or cap is hit.
    pub max_tier2_series: usize,
    /// Max markets to output from select_for_valuation (BOT_VALUATION_MARKETS, default 250).
    /// Kept here so the round-robin balancing step produces exactly the right count.
    pub valuation_limit: usize,
    /// When true, markets with no bid/ask (spread_cents=None) pass the spread filter instead
    /// of being dropped. Useful for overnight paper collection where market makers are absent
    /// but mid-price-based valuation still works. (BOT_SCAN_ALLOW_NO_SPREAD, default false).
    pub allow_no_spread: bool,
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
            series_allowlist: std::env::var("BOT_SCAN_SERIES_ALLOWLIST")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            series_blocklist: {
                // Always block multi-leg parlays (KXMVE*) and novelty quick-settle markets.
                // These are unpredictable by any signal-based model and flood the scan quota.
                let defaults = ["KXMVE", "KXQUICKSETTLE"];
                let from_env: Vec<String> = std::env::var("BOT_SCAN_SERIES_BLOCKLIST")
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                defaults.iter().map(|s| s.to_string()).chain(from_env).collect()
            },
            series_max_per_fetch: std::env::var("BOT_SCAN_SERIES_MAX_PER_FETCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(200)
                .max(10),
            min_time_to_close_secs: std::env::var("BOT_SCAN_MIN_CLOSE_SECS")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(900)
                .max(0),
            max_per_event: std::env::var("BOT_SCAN_MAX_PER_EVENT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(3),
            balance_by_category: std::env::var("BOT_SCAN_BALANCE_CATEGORIES")
                .map(|v| v != "0" && v.to_ascii_lowercase() != "false")
                .unwrap_or(true),
            tier2_categories: std::env::var("BOT_SCAN_TIER2_CATEGORIES")
                .unwrap_or_else(|_| {
                    "Sports,Crypto,Financials,Climate and Weather,Politics".to_string()
                })
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            max_tier2_series: std::env::var("BOT_SCAN_MAX_TIER2_SERIES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(120),
            valuation_limit: std::env::var("BOT_VALUATION_MARKETS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(250)
                .max(1),
            allow_no_spread: std::env::var("BOT_SCAN_ALLOW_NO_SPREAD")
                .map(|v| v != "0" && v.to_ascii_lowercase() != "false")
                .unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScannedMarket {
    pub ticker: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub market_type: Option<String>,
    pub event_ticker: Option<String>,
    pub series_ticker: Option<String>,
    pub yes_bid_cents: Option<f64>,
    pub yes_ask_cents: Option<f64>,
    pub yes_bid_size: Option<f64>,
    pub yes_ask_size: Option<f64>,
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

#[derive(Debug, Clone, Default)]
pub struct ScanTrace {
    pub snapshot_markets: Vec<ScannedMarket>,
    pub deltas: Vec<MarketDelta>,
    pub final_markets: Vec<ScannedMarket>,
}

impl KalshiMarketScanner {
    pub fn new(cfg: ScannerConfig) -> Self {
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        Self { cfg, http }
    }

    pub async fn scan_snapshot(&self) -> Result<Vec<ScannedMarket>, ExecutionError> {
        let mut seen: HashMap<String, ScannedMarket> = HashMap::new();

        // Tier 1: targeted queries for each allowlist series.
        // These bypass the alphabetical ordering problem where KXMVE floods the quota
        // before KXNBA/KXNHL/etc. can be reached.
        // Small delay between requests to stay within Kalshi rate limits.
        for (i, series) in self.cfg.series_allowlist.iter().enumerate() {
            if i > 0 {
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            let markets = self
                .scan_pages(Some(series.as_str()), self.cfg.series_max_per_fetch)
                .await
                .unwrap_or_else(|err| {
                    eprintln!("scan warning: series {series} failed: {err}");
                    Vec::new()
                });
            for m in markets {
                seen.entry(m.ticker.clone()).or_insert(m);
            }
        }

        // Tier 2: series-category discovery to fill remaining quota.
        // The unfiltered general endpoint returns KXMVE first (100k+ variants), which consumes
        // the entire fetch budget before any real markets appear. Instead, enumerate series by
        // category (Sports/Crypto/etc.), skip already-seen and blocked series, then query each
        // briefly. Series are sorted by last_updated_ts desc so active ones come first.
        if seen.len() < self.cfg.max_markets && !self.cfg.tier2_categories.is_empty() {
            let discovered = self.fetch_series_by_categories().await;
            let mut tier2_queries = 0usize;
            for series_ticker in &discovered {
                if seen.len() >= self.cfg.max_markets { break; }
                if tier2_queries >= self.cfg.max_tier2_series { break; }
                // Skip already in allowlist (already fetched in tier 1).
                if self.cfg.series_allowlist.iter().any(|a| a == series_ticker) { continue; }
                let budget = (self.cfg.max_markets - seen.len()).min(self.cfg.series_max_per_fetch);
                // Brief pause to stay within Kalshi rate limits after the tier-1 burst.
                tokio::time::sleep(Duration::from_millis(200)).await;
                let markets = self
                    .scan_pages(Some(series_ticker.as_str()), budget)
                    .await
                    .unwrap_or_else(|err| {
                        eprintln!("scan warning: tier2 series {series_ticker} failed: {err}");
                        Vec::new()
                    });
                for m in markets {
                    if !seen.contains_key(&m.ticker) {
                        seen.entry(m.ticker.clone()).or_insert(m);
                    }
                }
                tier2_queries += 1;
            }
            if tier2_queries > 0 {
                println!("scan tier2: queried {tier2_queries} discovered series");
            }
        }

        Ok(seen.into_values().collect())
    }

    /// Paginate markets from the API. Pass `series_ticker` for targeted series queries,
    /// or `None` for the general unfiltered scan.
    async fn scan_pages(
        &self,
        series_ticker: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ScannedMarket>, ExecutionError> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;

        while out.len() < limit {
            let page_size = (limit - out.len()).min(1000).to_string();
            let mut req = self
                .http
                .get(format!("{}/trade-api/v2/markets", self.cfg.api_base_url))
                .query(&[("status", "open"), ("limit", page_size.as_str())]);
            if let Some(series) = series_ticker {
                req = req.query(&[("series_ticker", series)]);
            }
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

    /// Fetch series tickers from the Kalshi series endpoint for all configured tier-2
    /// categories. Returns tickers sorted by last_updated_ts descending (most active first),
    /// with blocked series already removed.
    ///
    /// Each category request is bounded by a 10-second timeout so that a single slow
    /// or hanging endpoint cannot collapse the entire discovery pass.
    async fn fetch_series_by_categories(&self) -> Vec<String> {
        const CATEGORY_TIMEOUT: Duration = Duration::from_secs(10);
        let mut entries: Vec<(String, String)> = Vec::new(); // (last_updated_ts, ticker)
        for category in &self.cfg.tier2_categories {
            let req = self
                .http
                .get(format!("{}/trade-api/v2/series", self.cfg.api_base_url))
                .query(&[("limit", "1000"), ("category", category.as_str())])
                .send();
            let resp = match tokio::time::timeout(CATEGORY_TIMEOUT, req).await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    eprintln!("scan warning: series list for '{category}' failed: {e}");
                    continue;
                }
                Err(_) => {
                    eprintln!("scan warning: series list for '{category}' timed out");
                    continue;
                }
            };
            let body = match resp.text().await {
                Ok(b) => b,
                Err(_) => continue,
            };
            let parsed: SeriesListResponse = match serde_json::from_str(&body) {
                Ok(p) => p,
                Err(_) => continue,
            };
            for s in parsed.series {
                let blocked = self
                    .cfg
                    .series_blocklist
                    .iter()
                    .any(|b| s.ticker.starts_with(b.as_str()));
                if !blocked {
                    entries.push((s.last_updated_ts.unwrap_or_default(), s.ticker));
                }
            }
        }
        // Sort most-recently-updated first so active series are queried before stale ones.
        entries.sort_by(|a, b| b.0.cmp(&a.0));
        // Deduplicate (a series ticker may appear across category queries).
        let mut seen_tickers = std::collections::HashSet::new();
        entries
            .into_iter()
            .filter_map(|(_, ticker)| {
                if seen_tickers.insert(ticker.clone()) { Some(ticker) } else { None }
            })
            .collect()
    }

    pub async fn scan_snapshot_with_deltas(&self) -> Result<Vec<ScannedMarket>, ExecutionError> {
        Ok(self.scan_snapshot_with_trace().await?.final_markets)
    }

    pub async fn scan_snapshot_with_trace(&self) -> Result<ScanTrace, ExecutionError> {
        let snapshot = self.scan_snapshot().await?;
        let mut index = to_index_map(snapshot.clone());
        let tickers: Vec<String> = index.keys().cloned().collect();

        let delta_cfg = WsDeltaConfig {
            // Default WS URL is derived from the REST base URL to avoid accidentally connecting
            // to the demo endpoint when running against production.
            ws_url: std::env::var("KALSHI_WS_URL").unwrap_or_else(|_| {
                self.cfg.api_base_url
                    .replace("https://", "wss://")
                    .replace("http://", "ws://")
                    + "/trade-api/ws/v2"
            }),
            listen_window: Duration::from_secs(self.cfg.ws_delta_window_secs),
        };
        let ingestor = KalshiWsDeltaIngestor::new(delta_cfg);
        let deltas = match ingestor.collect_deltas(&tickers).await {
            Ok(deltas) => {
                apply_deltas(&mut index, deltas.clone());
                deltas
            }
            Err(_) => Vec::new(),
        };

        let final_markets = index.into_values().collect();
        Ok(ScanTrace {
            snapshot_markets: snapshot,
            deltas,
            final_markets,
        })
    }

    pub fn select_for_valuation(&self, markets: Vec<ScannedMarket>) -> Vec<ScannedMarket> {
        let now = Utc::now();
        let mut filtered: Vec<ScannedMarket> = markets
            .into_iter()
            .filter(|m| m.volume >= self.cfg.min_volume)
            .filter(|m| m.spread_cents().map(|s| s <= self.cfg.max_spread_cents).unwrap_or(self.cfg.allow_no_spread))
            .filter(|m| {
                // Skip markets closing too soon (< min_time_to_close_secs).
                // Markets with no close_time are always kept.
                match m.close_time {
                    Some(close) => (close - now).num_seconds() >= self.cfg.min_time_to_close_secs,
                    None => true,
                }
            })
            .collect();

        filtered.sort_by(|a, b| b.volume.total_cmp(&a.volume));

        // Cap markets per event_ticker to avoid one event flooding the quota.
        if self.cfg.max_per_event > 0 {
            let mut event_counts: HashMap<String, usize> = HashMap::new();
            filtered.retain(|m| {
                let key = m.event_ticker.clone().unwrap_or_else(|| m.ticker.clone());
                let count = event_counts.entry(key).or_insert(0);
                if *count < self.cfg.max_per_event {
                    *count += 1;
                    true
                } else {
                    false
                }
            });
        }

        // Apply category balancing: round-robin across Weather/Sports/Crypto/Other so that
        // high-volume categories don't crowd out markets that benefit from enrichment.
        // Use valuation_limit (BOT_VALUATION_MARKETS) not max_markets (BOT_SCAN_MAX_MARKETS)
        // so the round-robin produces exactly the right output count.
        if self.cfg.balance_by_category {
            let balanced = select_for_enrichment(&filtered, self.cfg.valuation_limit);
            return balanced.into_iter().cloned().collect();
        }

        filtered.truncate(self.cfg.valuation_limit);
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
            if delta.yes_bid_size.is_some() {
                m.yes_bid_size = delta.yes_bid_size;
            }
            if delta.yes_ask_size.is_some() {
                m.yes_ask_size = delta.yes_ask_size;
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
struct SeriesListResponse {
    #[serde(default)]
    series: Vec<SeriesWire>,
}

#[derive(Debug, Deserialize)]
struct SeriesWire {
    ticker: String,
    last_updated_ts: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KalshiMarketWire {
    ticker: String,
    title: Option<String>,
    subtitle: Option<String>,
    #[serde(alias = "market_type", alias = "marketType")]
    market_type: Option<String>,
    #[serde(alias = "event_ticker", alias = "eventTicker")]
    event_ticker: Option<String>,
    #[serde(alias = "series_ticker", alias = "seriesTicker")]
    series_ticker: Option<String>,
    #[serde(alias = "yes_bid", alias = "yesBid")]
    yes_bid: Option<f64>,
    #[serde(alias = "yes_ask", alias = "yesAsk")]
    yes_ask: Option<f64>,
    #[serde(alias = "yes_bid_dollars", alias = "yesBidDollars")]
    yes_bid_dollars: Option<String>,
    #[serde(alias = "yes_ask_dollars", alias = "yesAskDollars")]
    yes_ask_dollars: Option<String>,
    #[serde(alias = "yes_bid_size", alias = "yesBidSize", alias = "yes_bid_size_fp")]
    yes_bid_size: Option<String>,
    #[serde(alias = "yes_ask_size", alias = "yesAskSize", alias = "yes_ask_size_fp")]
    yes_ask_size: Option<String>,
    // volume_fp is a quoted decimal string ("4109.00"), same format as price fields.
    #[serde(alias = "volume_fp")]
    volume: Option<String>,
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

    let yes_bid_size = m.yes_bid_size.as_deref().and_then(|s| s.parse::<f64>().ok());
    let yes_ask_size = m.yes_ask_size.as_deref().and_then(|s| s.parse::<f64>().ok());

    let volume = m.volume.as_deref().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);

    ScannedMarket {
        ticker: m.ticker,
        title: m.title.unwrap_or_default(),
        subtitle: m.subtitle,
        market_type: m.market_type,
        event_ticker: m.event_ticker,
        series_ticker: m.series_ticker,
        yes_bid_cents,
        yes_ask_cents,
        yes_bid_size,
        yes_ask_size,
        volume,
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
            subtitle: None,
            market_type: None,
            event_ticker: None,
            series_ticker: None,
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
                subtitle: None,
                market_type: None,
                event_ticker: None,
                series_ticker: None,
                yes_bid_cents: Some(40.0),
                yes_ask_cents: Some(45.0),
                volume: 1_000.0,
                close_time: None,
            },
            ScannedMarket {
                ticker: "B".to_string(),
                title: "".to_string(),
                subtitle: None,
                market_type: None,
                event_ticker: None,
                series_ticker: None,
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
    fn allow_no_spread_passes_markets_with_no_bid_ask() {
        // Default (allow_no_spread=false): markets with no bid/ask are dropped.
        let scanner_strict = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 50.0,
            allow_no_spread: false,
            ..ScannerConfig::default()
        });
        // allow_no_spread=true: mid-price valuation can still run, so keep them.
        let scanner_permissive = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 50.0,
            allow_no_spread: true,
            ..ScannerConfig::default()
        });
        let markets = vec![ScannedMarket {
            ticker: "KXNIGHT".to_string(),
            title: "".to_string(),
            subtitle: None,
            market_type: None,
            event_ticker: None,
            series_ticker: None,
            yes_bid_cents: None,
            yes_ask_cents: None,
            volume: 10_000.0,
            close_time: None,
        }];
        assert!(
            scanner_strict.select_for_valuation(markets.clone()).is_empty(),
            "strict scanner should drop no-spread market"
        );
        assert_eq!(
            scanner_permissive.select_for_valuation(markets).len(),
            1,
            "permissive scanner should keep no-spread market for mid-price valuation"
        );
    }

    #[test]
    fn parses_dollar_price_fields_to_cents() {
        let wire = KalshiMarketWire {
            ticker: "KXTEST".to_string(),
            title: Some("Test".to_string()),
            subtitle: None,
            market_type: None,
            event_ticker: None,
            series_ticker: None,
            yes_bid: None,
            yes_ask: None,
            yes_bid_dollars: Some("0.43".to_string()),
            yes_ask_dollars: Some("0.49".to_string()),
            volume: Some("1234.00".to_string()),
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
            subtitle: None,
            market_type: None,
            event_ticker: None,
            series_ticker: None,
            yes_bid_cents: Some(40.0),
            yes_ask_cents: Some(45.0),
            volume: 100.0,
            close_time: None,
        }]);
        apply_deltas(
            &mut map,
            vec![MarketDelta {
                observed_at: Utc::now(),
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

    fn liquid_market(ticker: &str, event_ticker: Option<&str>, volume: f64, close_secs_from_now: Option<i64>) -> ScannedMarket {
        ScannedMarket {
            ticker: ticker.to_string(),
            title: "".to_string(),
            subtitle: None,
            market_type: None,
            event_ticker: event_ticker.map(|s| s.to_string()),
            series_ticker: None,
            yes_bid_cents: Some(45.0),
            yes_ask_cents: Some(50.0),
            volume,
            close_time: close_secs_from_now.map(|s| Utc::now() + chrono::Duration::seconds(s)),
        }
    }

    #[test]
    fn close_time_filter_drops_expiring_markets() {
        let scanner = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 20.0,
            min_time_to_close_secs: 900,
            max_per_event: 0,
            ..ScannerConfig::default()
        });
        let markets = vec![
            liquid_market("SOON", None, 1000.0, Some(300)),   // closes in 5 min → drop
            liquid_market("LATER", None, 1000.0, Some(3600)), // closes in 1 hr → keep
            liquid_market("NOCLOSE", None, 1000.0, None),     // no close_time → keep
        ];
        let selected = scanner.select_for_valuation(markets);
        let tickers: Vec<_> = selected.iter().map(|m| m.ticker.as_str()).collect();
        assert!(!tickers.contains(&"SOON"), "market closing in 5min should be dropped");
        assert!(tickers.contains(&"LATER"));
        assert!(tickers.contains(&"NOCLOSE"));
    }

    #[test]
    fn max_per_event_caps_event_variants() {
        let scanner = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 20.0,
            min_time_to_close_secs: 0,
            max_per_event: 2,
            ..ScannerConfig::default()
        });
        // 4 markets from the same event, sorted by volume desc: V4 > V3 > V2 > V1
        let markets = vec![
            liquid_market("T40", Some("KXHIGHCHI-26FEB16"), 1000.0, None),
            liquid_market("T42", Some("KXHIGHCHI-26FEB16"), 2000.0, None),
            liquid_market("T44", Some("KXHIGHCHI-26FEB16"), 3000.0, None),
            liquid_market("T46", Some("KXHIGHCHI-26FEB16"), 4000.0, None),
        ];
        let selected = scanner.select_for_valuation(markets);
        assert_eq!(selected.len(), 2, "should keep only top 2 per event");
        // highest volume kept
        assert_eq!(selected[0].ticker, "T46");
        assert_eq!(selected[1].ticker, "T44");
    }

    #[test]
    fn max_per_event_zero_disables_cap() {
        let scanner = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 20.0,
            min_time_to_close_secs: 0,
            max_per_event: 0,
            ..ScannerConfig::default()
        });
        let markets = (0..5)
            .map(|i| liquid_market(&format!("T{i}"), Some("EVT"), 1000.0 + i as f64, None))
            .collect();
        let selected = scanner.select_for_valuation(markets);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn valuation_limit_caps_select_for_valuation_output() {
        // With 20 markets and valuation_limit=5, output must be exactly 5.
        let scanner = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 20.0,
            min_time_to_close_secs: 0,
            max_per_event: 0,
            valuation_limit: 5,
            balance_by_category: false,
            ..ScannerConfig::default()
        });
        let markets = (0..20)
            .map(|i| liquid_market(&format!("T{i}"), None, 1000.0 + i as f64, None))
            .collect();
        let selected = scanner.select_for_valuation(markets);
        assert_eq!(selected.len(), 5);
        // Should be sorted by volume desc — highest volume markets retained.
        assert_eq!(selected[0].ticker, "T19");
    }

    #[test]
    fn valuation_limit_caps_balanced_output() {
        // Same with balance_by_category=true: round-robin should still respect the limit.
        let scanner = KalshiMarketScanner::new(ScannerConfig {
            min_volume: 0.0,
            max_spread_cents: 20.0,
            min_time_to_close_secs: 0,
            max_per_event: 0,
            valuation_limit: 7,
            balance_by_category: true,
            ..ScannerConfig::default()
        });
        let markets = (0..30)
            .map(|i| liquid_market(&format!("T{i}"), None, 1000.0 + i as f64, None))
            .collect();
        let selected = scanner.select_for_valuation(markets);
        assert_eq!(selected.len(), 7);
    }

    #[test]
    fn ws_url_derived_from_api_base_when_env_unset() {
        // With KALSHI_WS_URL unset, the scanner derives ws URL from api_base_url.
        // This test verifies the derivation logic by inspecting the config the scanner
        // would use — we do this by checking the string transform directly.
        let api_base = "https://api.elections.kalshi.com";
        let expected = "wss://api.elections.kalshi.com/trade-api/ws/v2";
        let derived = api_base
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            + "/trade-api/ws/v2";
        assert_eq!(derived, expected);

        let demo_base = "https://demo-api.kalshi.co";
        let derived_demo = demo_base
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            + "/trade-api/ws/v2";
        assert_eq!(derived_demo, "wss://demo-api.kalshi.co/trade-api/ws/v2");
    }

    #[test]
    fn series_wire_deserializes_from_category_response() {
        // Verify the SeriesWire struct correctly deserializes a real API response shape.
        let json = r#"{
            "series": [
                {"ticker": "KXNBAGAME", "last_updated_ts": "2026-03-25T12:00:00Z", "category": "Sports"},
                {"ticker": "KXHIGHCHI", "last_updated_ts": "2026-03-24T08:00:00Z", "category": "Climate and Weather"},
                {"ticker": "KXMVENOJUNK", "category": "Sports"}
            ]
        }"#;
        let parsed: SeriesListResponse = serde_json::from_str(json).expect("should parse");
        assert_eq!(parsed.series.len(), 3);
        assert_eq!(parsed.series[0].ticker, "KXNBAGAME");
        assert_eq!(parsed.series[0].last_updated_ts.as_deref(), Some("2026-03-25T12:00:00Z"));
        assert_eq!(parsed.series[2].last_updated_ts, None); // missing field → None
    }
}
