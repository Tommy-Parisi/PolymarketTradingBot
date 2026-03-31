use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::data::market_enrichment::{MarketEnrichment, MarketVertical};
use crate::data::market_scanner::ScannedMarket;
use crate::research::events::MarketStateEvent;

pub const FORECAST_FEATURE_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedMarketMetadata {
    pub vertical: String,
    pub entity_primary: Option<String>,
    pub entity_secondary: Option<String>,
    pub threshold_value: Option<f64>,
    pub threshold_direction: Option<String>,
    pub event_date_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastFeatureRow {
    pub schema_version: String,
    pub feature_ts: DateTime<Utc>,
    pub ticker: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub market_type: Option<String>,
    pub event_ticker: Option<String>,
    pub series_ticker: Option<String>,
    pub close_time: Option<DateTime<Utc>>,
    pub time_to_close_secs: Option<i64>,
    pub yes_bid_cents: Option<f64>,
    pub yes_ask_cents: Option<f64>,
    pub yes_bid_size: Option<f64>,
    pub yes_ask_size: Option<f64>,
    pub mid_prob_yes: Option<f64>,
    pub spread_cents: Option<f64>,
    pub book_pressure: Option<f64>,
    pub volume: f64,
    pub vertical: String,
    pub weather_signal: Option<f64>,
    pub sports_injury_signal: Option<f64>,
    pub crypto_sentiment_signal: Option<f64>,
    pub finance_price_signal: Option<f64>,
    pub entity_primary: Option<String>,
    pub entity_secondary: Option<String>,
    pub threshold_value: Option<f64>,
    pub threshold_direction: Option<String>,
    pub event_date_hint: Option<String>,
    pub source: String,
    pub cycle_id: Option<String>,
    pub recent_trade_count_delta: Option<f64>,
}

pub fn build_forecast_feature_row(
    market: &ScannedMarket,
    enrichment: Option<&MarketEnrichment>,
    feature_ts: DateTime<Utc>,
) -> ForecastFeatureRow {
    let parsed = parse_market_metadata(
        &market.ticker,
        &market.title,
        market.subtitle.as_deref(),
        enrichment.map(|v| v.vertical),
    );
    ForecastFeatureRow {
        schema_version: FORECAST_FEATURE_SCHEMA_VERSION.to_string(),
        feature_ts,
        ticker: market.ticker.clone(),
        title: market.title.clone(),
        subtitle: market.subtitle.clone(),
        market_type: market.market_type.clone(),
        event_ticker: market.event_ticker.clone(),
        series_ticker: market.series_ticker.clone(),
        close_time: market.close_time,
        time_to_close_secs: market.close_time.map(|close| (close - feature_ts).num_seconds()),
        yes_bid_cents: market.yes_bid_cents,
        yes_ask_cents: market.yes_ask_cents,
        yes_bid_size: market.yes_bid_size,
        yes_ask_size: market.yes_ask_size,
        mid_prob_yes: market_mid_prob_yes(market.yes_bid_cents, market.yes_ask_cents),
        spread_cents: market.spread_cents(),
        book_pressure: calculate_book_pressure(market.yes_bid_size, market.yes_ask_size),
        volume: market.volume,
        vertical: parsed.vertical,
        weather_signal: enrichment.and_then(|e| e.weather_signal),
        sports_injury_signal: enrichment.and_then(|e| e.sports_injury_signal),
        crypto_sentiment_signal: enrichment.and_then(|e| e.crypto_sentiment_signal),
        finance_price_signal: enrichment.and_then(|e| e.finance_price_signal),
        entity_primary: parsed.entity_primary,
        entity_secondary: parsed.entity_secondary,
        threshold_value: parsed.threshold_value,
        threshold_direction: parsed.threshold_direction,
        event_date_hint: parsed.event_date_hint,
        source: "online_market_snapshot".to_string(),
        cycle_id: None,
        recent_trade_count_delta: None,
    }
}

pub fn build_forecast_feature_row_from_event(event: &MarketStateEvent) -> ForecastFeatureRow {
    let parsed = parse_market_metadata(
        &event.ticker,
        &event.title,
        event.subtitle.as_deref(),
        infer_vertical_from_event(event),
    );
    ForecastFeatureRow {
        schema_version: FORECAST_FEATURE_SCHEMA_VERSION.to_string(),
        feature_ts: event.ts,
        ticker: event.ticker.clone(),
        title: event.title.clone(),
        subtitle: event.subtitle.clone(),
        market_type: event.market_type.clone(),
        event_ticker: event.event_ticker.clone(),
        series_ticker: event.series_ticker.clone(),
        close_time: event.close_time,
        time_to_close_secs: event.close_time.map(|close| (close - event.ts).num_seconds()),
        yes_bid_cents: event.yes_bid_cents,
        yes_ask_cents: event.yes_ask_cents,
        yes_bid_size: event.yes_bid_size,
        yes_ask_size: event.yes_ask_size,
        mid_prob_yes: event.mid_prob_yes,
        spread_cents: event.spread_cents,
        book_pressure: calculate_book_pressure(event.yes_bid_size, event.yes_ask_size),
        volume: event.volume,
        vertical: parsed.vertical,
        weather_signal: None,
        sports_injury_signal: None,
        crypto_sentiment_signal: None,
        finance_price_signal: event.finance_price_signal,
        entity_primary: parsed.entity_primary,
        entity_secondary: parsed.entity_secondary,
        threshold_value: parsed.threshold_value,
        threshold_direction: parsed.threshold_direction,
        event_date_hint: parsed.event_date_hint,
        source: event.source.clone(),
        cycle_id: Some(event.cycle_id.clone()),
        recent_trade_count_delta: event.traded_count_delta,
    }
}

fn market_mid_prob_yes(bid: Option<f64>, ask: Option<f64>) -> Option<f64> {
    Some(((bid? + ask?) / 2.0 / 100.0).clamp(0.0, 1.0))
}

fn calculate_book_pressure(bid_size: Option<f64>, ask_size: Option<f64>) -> Option<f64> {
    let b = bid_size?;
    let a = ask_size?;
    if b + a == 0.0 {
        return Some(0.5);
    }
    Some(b / (b + a))
}

fn infer_vertical_from_event(event: &MarketStateEvent) -> Option<MarketVertical> {
    let ticker = event.ticker.to_ascii_uppercase();
    let title = event.title.to_ascii_uppercase();
    if ticker.contains("HIGH") || title.contains("TEMP") || title.contains("WEATHER") {
        return Some(MarketVertical::Weather);
    }
    if ticker.contains("NBA") || ticker.contains("NFL") || ticker.contains("TABLETENNIS") {
        return Some(MarketVertical::Sports);
    }
    if ticker.contains("BTC") || ticker.contains("ETH") || title.contains("BITCOIN") {
        return Some(MarketVertical::Crypto);
    }
    None
}

pub fn parse_market_metadata(
    ticker: &str,
    title: &str,
    subtitle: Option<&str>,
    vertical_hint: Option<MarketVertical>,
) -> ParsedMarketMetadata {
    let merged = match subtitle {
        Some(sub) => format!("{title} {sub}"),
        None => title.to_string(),
    };
    let normalized = merged.replace('*', " ");
    let lowered = normalized.to_ascii_lowercase();
    let threshold_direction = if lowered.contains(" >")
        || lowered.contains(" above ")
        || lowered.contains(" over ")
        || lowered.contains(" greater than ")
    {
        Some("above".to_string())
    } else if lowered.contains(" <")
        || lowered.contains(" below ")
        || lowered.contains(" under ")
        || lowered.contains(" less than ")
    {
        Some("below".to_string())
    } else {
        None
    };

    let threshold_value = extract_threshold_value(&normalized).or_else(|| extract_threshold_from_ticker(ticker));
    let event_date_hint = extract_date_hint(ticker).or_else(|| extract_date_hint(&normalized));
    let (entity_primary, entity_secondary) = extract_entities(&normalized);

    ParsedMarketMetadata {
        vertical: vertical_name(vertical_hint.unwrap_or_else(|| infer_vertical_from_title_or_ticker(ticker, title))),
        entity_primary,
        entity_secondary,
        threshold_value,
        threshold_direction,
        event_date_hint,
    }
}

fn vertical_name(vertical: MarketVertical) -> String {
    match vertical {
        MarketVertical::Weather  => "weather",
        MarketVertical::Sports   => "sports",
        MarketVertical::Esports  => "esports",
        MarketVertical::Finance  => "finance",
        MarketVertical::Politics => "politics",
        MarketVertical::Crypto   => "crypto",
        MarketVertical::Other    => "other",
    }
    .to_string()
}

fn infer_vertical_from_title_or_ticker(ticker: &str, title: &str) -> MarketVertical {
    // Delegate to detect_vertical via a lightweight ScannedMarket proxy.
    // This avoids duplicating the classification logic.
    use crate::data::market_scanner::ScannedMarket;
    let proxy = ScannedMarket {
        ticker: ticker.to_string(),
        title: title.to_string(),
        subtitle: None,
        market_type: None,
        event_ticker: None,
        series_ticker: None,
        yes_bid_cents: None,
        yes_ask_cents: None,
        yes_bid_size: None,
        yes_ask_size: None,
        volume: 0.0,
        close_time: None,
    };
    crate::data::market_enrichment::detect_vertical(&proxy)
}

fn extract_threshold_value(raw: &str) -> Option<f64> {
    let mut buf = String::new();
    let mut started = false;
    for c in raw.chars() {
        if c.is_ascii_digit() || (started && c == '.') {
            buf.push(c);
            started = true;
        } else if started {
            break;
        }
    }
    if buf.is_empty() {
        None
    } else {
        buf.parse::<f64>().ok()
    }
}

/// Extract a threshold value specifically from a Kalshi ticker segment.
/// Looks for `-T{number}` or `-B{number}` patterns (e.g. `KXHIGHINFLATION-26DEC-T3.8` → 3.8,
/// `KXBNB-26MAR2417-B537` → 537). This avoids the title-based extractor grabbing date digits
/// from the ticker (e.g. `26` from `26DEC`).
/// Extract a threshold value specifically from a Kalshi ticker segment.
/// Looks for `-T{number}` or `-B{number}` patterns (e.g. `KXHIGHINFLATION-26DEC-T3.8` → 3.8,
/// `KXBNB-26MAR2417-B537` → 537). This avoids the title-based extractor grabbing date digits
/// from the ticker (e.g. `26` from `26DEC`).
pub fn extract_threshold_from_ticker(ticker: &str) -> Option<f64> {
    let upper = ticker.to_ascii_uppercase();
    for segment in upper.split('-') {
        // Segment must start with T or B followed immediately by a digit
        let rest = match segment.strip_prefix('T').or_else(|| segment.strip_prefix('B')) {
            Some(r) if r.starts_with(|c: char| c.is_ascii_digit()) => r,
            _ => continue,
        };
        // Take the longest leading numeric prefix (handles "305.0" and "537")
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(rest.len());
        if let Ok(v) = rest[..end].parse::<f64>() {
            return Some(v);
        }
    }
    None
}

fn extract_date_hint(raw: &str) -> Option<String> {
    let upper = raw.to_ascii_uppercase();
    for token in upper.split(|c: char| !c.is_ascii_alphanumeric()) {
        if token.len() >= 7 && token.chars().any(|c| c.is_ascii_digit()) && token.chars().any(|c| c.is_ascii_alphabetic()) {
            return Some(token.to_string());
        }
    }
    None
}

fn extract_entities(raw: &str) -> (Option<String>, Option<String>) {
    if let Some((left, right)) = raw.split_once(" vs ") {
        return (Some(left.trim().to_string()), Some(right.trim().to_string()));
    }
    if let Some((left, right)) = raw.split_once(" vs. ") {
        return (Some(left.trim().to_string()), Some(right.trim().to_string()));
    }
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_weather_threshold_and_direction() {
        let parsed = parse_market_metadata(
            "KXHIGHCHI-26FEB16-T45",
            "Will the high temp in Chicago be >45° on Feb 16, 2026?",
            None,
            Some(MarketVertical::Weather),
        );
        assert_eq!(parsed.vertical, "weather");
        assert_eq!(parsed.threshold_direction.as_deref(), Some("above"));
        assert_eq!(parsed.threshold_value, Some(45.0));
    }

    #[test]
    fn parses_sports_entities() {
        let parsed = parse_market_metadata(
            "KXNCAAHOCKEYGAME-26FEB131800RMUNIA-TIE",
            "Robert Morris vs Niagara Winner?",
            None,
            Some(MarketVertical::Sports),
        );
        assert_eq!(parsed.entity_primary.as_deref(), Some("Robert Morris"));
        assert_eq!(parsed.entity_secondary.as_deref(), Some("Niagara Winner?"));
    }

    #[test]
    fn infers_weather_vertical_from_ticker() {
        let parsed = parse_market_metadata(
            "KXHIGH-NYC-26MAR25",
            "Will the high temp exceed 60F?",
            None,
            None,
        );
        assert_eq!(parsed.vertical, "weather");
    }

    #[test]
    fn infers_sports_vertical_from_ticker_nba() {
        let parsed = parse_market_metadata("KXNBAGAME-123", "Game outcome", None, None);
        assert_eq!(parsed.vertical, "sports");
    }

    #[test]
    fn infers_crypto_vertical_from_ticker() {
        let parsed = parse_market_metadata("KXBTC-26DEC-B120000", "Bitcoin above 120k", None, None);
        assert_eq!(parsed.vertical, "crypto");
    }

    #[test]
    fn infers_other_vertical_as_fallback() {
        let parsed = parse_market_metadata("KXPOLITICS-001", "Will X happen?", None, None);
        assert_eq!(parsed.vertical, "other");
    }

    #[test]
    fn parses_threshold_direction_below() {
        let parsed = parse_market_metadata(
            "KXWEATHER",
            "Will the temperature be below 32F?",
            None,
            Some(MarketVertical::Weather),
        );
        assert_eq!(parsed.threshold_direction.as_deref(), Some("below"));
    }

    #[test]
    fn parses_threshold_direction_above_via_greater_than() {
        let parsed = parse_market_metadata(
            "KXTEST",
            "Will X be greater than 100?",
            None,
            None,
        );
        assert_eq!(parsed.threshold_direction.as_deref(), Some("above"));
    }

    #[test]
    fn no_threshold_when_no_number_in_title() {
        let parsed = parse_market_metadata("KXTEST", "Will it rain tomorrow?", None, None);
        assert!(parsed.threshold_value.is_none());
    }

    #[test]
    fn subtitle_merged_for_entity_extraction() {
        let parsed = parse_market_metadata(
            "KXSPORTS",
            "Team A vs Team B",
            Some("in the final"),
            Some(MarketVertical::Sports),
        );
        assert_eq!(parsed.entity_primary.as_deref(), Some("Team A"));
        assert_eq!(parsed.entity_secondary.as_deref(), Some("Team B in the final"));
    }

    #[test]
    fn market_mid_prob_yes_calculation() {
        // bid=45, ask=55 → mid = (45+55)/2/100 = 0.50
        use super::market_mid_prob_yes;
        assert_eq!(market_mid_prob_yes(Some(45.0), Some(55.0)), Some(0.50));
        assert_eq!(market_mid_prob_yes(None, Some(55.0)), None);
        assert_eq!(market_mid_prob_yes(Some(45.0), None), None);
    }

    #[test]
    fn threshold_extracted_from_ticker_t_prefix() {
        // KXHIGHINFLATION-26DEC-T3.8 — old code grabbed 26 (date); new code gets 3.8
        use super::extract_threshold_from_ticker;
        assert_eq!(extract_threshold_from_ticker("KXHIGHINFLATION-26DEC-T3.8"), Some(3.8));
        assert_eq!(extract_threshold_from_ticker("KXAIRFARECPI-26APR10-T310.0"), Some(310.0));
        assert_eq!(extract_threshold_from_ticker("KXGOLDD-26MAR2517-T4400"), Some(4400.0));
    }

    #[test]
    fn threshold_extracted_from_ticker_b_prefix() {
        use super::extract_threshold_from_ticker;
        assert_eq!(extract_threshold_from_ticker("KXBNB-26MAR2417-B537"), Some(537.0));
        assert_eq!(extract_threshold_from_ticker("KXHYPE-26MAR2501-B61"), Some(61.0));
    }

    #[test]
    fn threshold_from_ticker_not_confused_by_date_segment() {
        // 26MAR24 should not be extracted as a threshold; the T/B segment should win
        use super::extract_threshold_from_ticker;
        let v = extract_threshold_from_ticker("KXNASDAQ100U-26MAR25H1200-T24539.99");
        assert_eq!(v, Some(24539.99));
    }

    #[test]
    fn threshold_from_ticker_none_when_no_tb_segment() {
        use super::extract_threshold_from_ticker;
        assert_eq!(extract_threshold_from_ticker("KXNBAGAME-26MAR24BOSNYC-BOS"), None);
        assert_eq!(extract_threshold_from_ticker("KXTEST"), None);
    }

    #[test]
    fn parse_market_metadata_uses_ticker_threshold_when_title_has_none() {
        // Title has no number, so falls through to ticker extraction
        let parsed = parse_market_metadata(
            "KXHIGHINFLATION-26DEC-T3.8",
            "Will inflation exceed target?",
            None,
            None,
        );
        assert_eq!(parsed.threshold_value, Some(3.8));
    }

    #[test]
    fn parse_market_metadata_prefers_title_threshold_over_ticker() {
        // Title clearly has the threshold; should not defer to ticker
        let parsed = parse_market_metadata(
            "KXHIGHINFLATION-26DEC-T3.8",
            "Will inflation exceed 4.0%?",
            None,
            None,
        );
        assert_eq!(parsed.threshold_value, Some(4.0));
    }
}
