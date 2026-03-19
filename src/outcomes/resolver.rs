use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, Duration, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::execution::types::ExecutionError;
use crate::research::events::MarketStateEvent;

pub const OUTCOME_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone)]
pub struct OutcomeResolverConfig {
    pub enabled: bool,
    pub api_base_url: String,
    pub research_dir: PathBuf,
    pub lookback_days: i64,
}

impl OutcomeResolverConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_OUTCOME_BACKFILL")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            api_base_url: std::env::var("KALSHI_API_BASE_URL")
                .unwrap_or_else(|_| "https://demo-api.kalshi.co".to_string()),
            research_dir: PathBuf::from(
                std::env::var("BOT_RESEARCH_DIR").unwrap_or_else(|_| "var/research".to_string()),
            ),
            lookback_days: std::env::var("BOT_OUTCOME_LOOKBACK_DAYS")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(14)
                .max(1),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketOutcomeRecord {
    pub schema_version: String,
    pub ticker: String,
    pub resolved_at: DateTime<Utc>,
    pub outcome_yes: Option<bool>,
    pub resolution_status: String,
    pub source: String,
    pub close_time: Option<DateTime<Utc>>,
}

pub async fn run_outcome_backfill(cfg: &OutcomeResolverConfig) -> Result<(), ExecutionError> {
    let candidates = collect_candidate_tickers(cfg)?;
    if candidates.is_empty() {
        println!("outcome backfill: no candidate tickers found");
        return Ok(());
    }

    let resolved = load_existing_outcomes(cfg)?;
    let client = Client::new();
    let mut new_records = Vec::new();

    for (ticker, close_time) in candidates {
        if resolved.contains(&ticker) {
            continue;
        }
        match fetch_outcome_record(&client, &cfg.api_base_url, &ticker, close_time).await {
            Ok(Some(record)) => {
                println!(
                    "outcome resolved: ticker={} status={} outcome_yes={:?}",
                    record.ticker, record.resolution_status, record.outcome_yes
                );
                new_records.push(record);
            }
            Ok(None) => {}
            Err(err) => eprintln!("outcome backfill warning for {}: {}", ticker, err),
        }
    }

    if new_records.is_empty() {
        println!("outcome backfill: no new resolved outcomes");
        return Ok(());
    }
    append_outcomes(cfg, &new_records)?;
    println!("outcome backfill complete: wrote {} new records", new_records.len());
    Ok(())
}

fn collect_candidate_tickers(
    cfg: &OutcomeResolverConfig,
) -> Result<BTreeMap<String, Option<DateTime<Utc>>>, ExecutionError> {
    let mut out = BTreeMap::new();
    let cutoff_day = Utc::now().date_naive() - Duration::days(cfg.lookback_days);
    let root = cfg.research_dir.join("market_state");
    if !root.exists() {
        return Ok(out);
    }

    for entry in fs::read_dir(root).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let entry = entry.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(day_name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(day) = chrono::NaiveDate::parse_from_str(day_name, "%Y-%m-%d") else {
            continue;
        };
        if day < cutoff_day {
            continue;
        }
        for file in fs::read_dir(path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let file = file.map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            let file_path = file.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let text =
                fs::read_to_string(&file_path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let event: MarketStateEvent = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let should_consider = event
                    .close_time
                    .map(|close| close <= Utc::now())
                    .unwrap_or(false);
                if should_consider {
                    out.entry(event.ticker).or_insert(event.close_time);
                }
            }
        }
    }
    Ok(out)
}

fn load_existing_outcomes(cfg: &OutcomeResolverConfig) -> Result<BTreeSet<String>, ExecutionError> {
    let mut out = BTreeSet::new();
    let path = outcomes_path(cfg);
    if !path.exists() {
        return Ok(out);
    }
    let text = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_str::<MarketOutcomeRecord>(line) {
            out.insert(record.ticker);
        }
    }
    Ok(out)
}

async fn fetch_outcome_record(
    client: &Client,
    api_base_url: &str,
    ticker: &str,
    close_time: Option<DateTime<Utc>>,
) -> Result<Option<MarketOutcomeRecord>, ExecutionError> {
    let path = format!("/trade-api/v2/markets/{ticker}");
    let url = format!("{}{}", api_base_url, path);
    let resp = client
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
            "GET {} failed ({}): {}",
            path, status, text
        )));
    }

    let root: Value =
        serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let market = root.get("market").unwrap_or(&root);
    let resolution_status = parse_resolution_status(market);
    let outcome_yes = parse_outcome_yes(market);

    let is_resolved = matches!(resolution_status.as_str(), "resolved" | "canceled");
    if !is_resolved {
        return Ok(None);
    }

    Ok(Some(MarketOutcomeRecord {
        schema_version: OUTCOME_SCHEMA_VERSION.to_string(),
        ticker: ticker.to_string(),
        resolved_at: Utc::now(),
        outcome_yes,
        resolution_status,
        source: "kalshi_market_get".to_string(),
        close_time,
    }))
}

fn parse_resolution_status(market: &Value) -> String {
    let status = market
        .get("status")
        .or_else(|| market.get("market_status"))
        .or_else(|| market.get("marketStatus"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    if matches!(
        status.as_str(),
        "settled" | "resolved" | "finalized" | "closed" | "expired"
    ) {
        return "resolved".to_string();
    }
    if matches!(status.as_str(), "canceled" | "void") {
        return "canceled".to_string();
    }
    if market.get("result").is_some()
        || market.get("settlement_value").is_some()
        || market.get("yes_result").is_some()
    {
        return "resolved".to_string();
    }
    "unresolved".to_string()
}

fn parse_outcome_yes(market: &Value) -> Option<bool> {
    if let Some(v) = market
        .get("yes_result")
        .or_else(|| market.get("yesResult"))
        .and_then(Value::as_bool)
    {
        return Some(v);
    }
    if let Some(v) = market
        .get("settlement_value")
        .or_else(|| market.get("settlementValue"))
        .and_then(value_as_f64)
    {
        if (v - 1.0).abs() < f64::EPSILON {
            return Some(true);
        }
        if v.abs() < f64::EPSILON {
            return Some(false);
        }
    }
    if let Some(v) = market
        .get("result")
        .or_else(|| market.get("outcome"))
        .or_else(|| market.get("winner"))
        .and_then(Value::as_str)
    {
        let normalized = v.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "yes" | "true" | "1") {
            return Some(true);
        }
        if matches!(normalized.as_str(), "no" | "false" | "0") {
            return Some(false);
        }
    }
    None
}

fn value_as_f64(value: &Value) -> Option<f64> {
    if let Some(v) = value.as_f64() {
        return Some(v);
    }
    value.as_str()?.parse::<f64>().ok()
}

fn append_outcomes(
    cfg: &OutcomeResolverConfig,
    records: &[MarketOutcomeRecord],
) -> Result<(), ExecutionError> {
    let path = outcomes_path(cfg);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for record in records {
        let line =
            serde_json::to_string(record).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    Ok(())
}

fn outcomes_path(cfg: &OutcomeResolverConfig) -> PathBuf {
    cfg.research_dir.join("outcomes").join("outcomes.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bool_yes_result() {
        let v: Value = serde_json::json!({
            "status": "settled",
            "yes_result": true
        });
        assert_eq!(parse_resolution_status(&v), "resolved");
        assert_eq!(parse_outcome_yes(&v), Some(true));
    }

    #[test]
    fn parses_numeric_settlement_value() {
        let v: Value = serde_json::json!({
            "market_status": "resolved",
            "settlement_value": "0"
        });
        assert_eq!(parse_resolution_status(&v), "resolved");
        assert_eq!(parse_outcome_yes(&v), Some(false));
    }

    #[test]
    fn parses_canceled_market() {
        let v: Value = serde_json::json!({
            "status": "canceled"
        });
        assert_eq!(parse_resolution_status(&v), "canceled");
        assert_eq!(parse_outcome_yes(&v), None);
    }
}
