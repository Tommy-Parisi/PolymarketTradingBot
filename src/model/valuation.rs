use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

use crate::data::market_enrichment::MarketEnrichment;
use crate::data::market_scanner::ScannedMarket;
use crate::execution::types::{ExecutionError, Side, TradeSignal};

#[derive(Debug, Clone)]
pub struct ValuationConfig {
    pub model: String,
    pub anthropic_base_url: String,
    pub anthropic_api_key: Option<String>,
    pub allow_heuristic_in_live: bool,
    pub batch_size: usize,
    pub max_retries: u32,
    pub timeout_ms: u64,
    pub max_tokens_per_batch: u32,
    pub max_prompt_chars: usize,
    pub cache_ttl_secs: u64,
    pub mispricing_threshold: f64,
    pub min_candidates: usize,
    pub fallback_mispricing_threshold: f64,
    pub adaptive_threshold_enabled: bool,
    pub adaptive_threshold_floor: f64,
    pub adaptive_liquidity_volume_ref: f64,
    pub adaptive_spread_ref_cents: f64,
    pub adaptive_confidence_weight: f64,
    pub adaptive_liquidity_weight: f64,
    pub adaptive_spread_weight: f64,
    pub fee_bps: f64,
    pub slippage_bps: f64,
}

impl Default for ValuationConfig {
    fn default() -> Self {
        Self {
            model: std::env::var("CLAUDE_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".to_string()),
            anthropic_base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            allow_heuristic_in_live: matches!(
                std::env::var("BOT_ALLOW_HEURISTIC_IN_LIVE")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            batch_size: 32,
            max_retries: 2,
            timeout_ms: 8_000,
            max_tokens_per_batch: 2_000,
            max_prompt_chars: 32_000,
            cache_ttl_secs: 600,
            mispricing_threshold: 0.08,
            min_candidates: 0,
            fallback_mispricing_threshold: 0.02,
            adaptive_threshold_enabled: false,
            adaptive_threshold_floor: 0.01,
            adaptive_liquidity_volume_ref: 50_000.0,
            adaptive_spread_ref_cents: 12.0,
            adaptive_confidence_weight: 0.20,
            adaptive_liquidity_weight: 0.55,
            adaptive_spread_weight: 0.25,
            fee_bps: 15.0,
            slippage_bps: 20.0,
        }
    }
}

impl ValuationConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("BOT_MISPRICING_THRESHOLD") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.mispricing_threshold = parsed.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_MIN_CANDIDATES") {
            if let Ok(parsed) = v.parse::<usize>() {
                cfg.min_candidates = parsed;
            }
        }
        if let Ok(v) = std::env::var("BOT_FALLBACK_MISPRICING_THRESHOLD") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.fallback_mispricing_threshold = parsed.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_THRESHOLD_ENABLED") {
            cfg.adaptive_threshold_enabled = matches!(
                v.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            );
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_THRESHOLD_FLOOR") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.adaptive_threshold_floor = parsed.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_LIQUIDITY_VOLUME_REF") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.adaptive_liquidity_volume_ref = parsed.max(1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_SPREAD_REF_CENTS") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.adaptive_spread_ref_cents = parsed.max(0.1);
            }
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_CONFIDENCE_WEIGHT") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.adaptive_confidence_weight = parsed.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_LIQUIDITY_WEIGHT") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.adaptive_liquidity_weight = parsed.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_ADAPTIVE_SPREAD_WEIGHT") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.adaptive_spread_weight = parsed.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_FEE_BPS") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.fee_bps = parsed.max(0.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_SLIPPAGE_BPS") {
            if let Ok(parsed) = v.parse::<f64>() {
                cfg.slippage_bps = parsed.max(0.0);
            }
        }
        if let Ok(v) = std::env::var("BOT_VALUATION_BATCH_SIZE") {
            if let Ok(parsed) = v.parse::<usize>() {
                cfg.batch_size = parsed.max(1);
            }
        }
        if let Ok(v) = std::env::var("BOT_VALUATION_TIMEOUT_MS") {
            if let Ok(parsed) = v.parse::<u64>() {
                cfg.timeout_ms = parsed.max(100);
            }
        }
        if let Ok(v) = std::env::var("BOT_ALLOW_HEURISTIC_IN_LIVE") {
            cfg.allow_heuristic_in_live = matches!(
                v.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            );
        }
        cfg
    }
}

#[derive(Debug, Clone)]
pub struct ValuationInput {
    pub market: ScannedMarket,
    pub enrichment: Option<MarketEnrichment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketValuation {
    pub ticker: String,
    pub fair_prob_yes: f64,
    pub market_mid_prob_yes: f64,
    pub market_volume: f64,
    pub spread_cents: Option<f64>,
    pub confidence: f64,
    pub rationale: String,
    pub stale_after: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CandidateTrade {
    pub ticker: String,
    pub side: Side,
    pub outcome_id: String,
    pub fair_price: f64,
    pub observed_price: f64,
    pub edge_pct: f64,
    pub confidence: f64,
    pub rationale: String,
}

#[derive(Debug, Clone, Default)]
pub struct ValuationRunSummary {
    pub used_claude: bool,
    pub used_heuristic: bool,
    pub fallback_reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CandidateEdgeSnapshot {
    pub ticker: String,
    pub raw_edge: f64,
    pub adjusted_edge: f64,
    pub effective_threshold: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct CandidateDiagnostics {
    pub strict_threshold: f64,
    pub fallback_threshold: f64,
    pub min_candidates: usize,
    pub adaptive_threshold_enabled: bool,
    pub total_cost_prob: f64,
    pub strict_count: usize,
    pub relaxed_count: usize,
    pub top_edges: Vec<CandidateEdgeSnapshot>,
}

pub struct ClaudeValuationEngine {
    cfg: ValuationConfig,
    http: Client,
    cache: Mutex<HashMap<String, CacheEntry>>,
    last_run_summary: Mutex<ValuationRunSummary>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    expires_at: Instant,
    value: MarketValuation,
}

impl ClaudeValuationEngine {
    pub fn new(cfg: ValuationConfig) -> Self {
        Self {
            cfg,
            http: Client::new(),
            cache: Mutex::new(HashMap::new()),
            last_run_summary: Mutex::new(ValuationRunSummary::default()),
        }
    }

    pub async fn value_markets(&self, inputs: &[ValuationInput]) -> Result<Vec<MarketValuation>, ExecutionError> {
        self.value_markets_with_claude_enabled(inputs, true).await
    }

    pub async fn value_markets_with_claude_enabled(
        &self,
        inputs: &[ValuationInput],
        claude_enabled: bool,
    ) -> Result<Vec<MarketValuation>, ExecutionError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        self.reset_last_run_summary();

        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.cfg.batch_size) {
            let mut unresolved = Vec::new();
            for i in chunk {
                if let Some(cached) = self.get_cached(i) {
                    out.push(cached);
                } else {
                    unresolved.push(i.clone());
                }
            }

            if unresolved.is_empty() {
                continue;
            }

            let (inferred, mode) = self.infer_batch(&unresolved, claude_enabled).await?;
            self.record_batch_mode(mode);
            let input_by_ticker: HashMap<String, ValuationInput> =
                unresolved.iter().map(|i| (i.market.ticker.clone(), i.clone())).collect();
            for v in inferred {
                if let Some(input) = input_by_ticker.get(&v.ticker) {
                    self.put_cached(input, &v);
                }
                out.push(v);
            }
        }
        Ok(out)
    }

    pub fn generate_candidates(&self, valuations: &[MarketValuation]) -> Vec<CandidateTrade> {
        let mut out = self.generate_candidates_with_threshold(
            valuations,
            self.cfg.mispricing_threshold,
            false,
        );

        if self.cfg.min_candidates > out.len() {
            let mut relaxed = self.generate_candidates_with_threshold(
                valuations,
                self.cfg.fallback_mispricing_threshold,
                true,
            );
            relaxed.retain(|r| {
                !out.iter()
                    .any(|o| o.ticker == r.ticker && o.outcome_id == r.outcome_id)
            });
            out.extend(relaxed);
            out.sort_by(|a, b| b.edge_pct.total_cmp(&a.edge_pct));
            out.truncate(self.cfg.min_candidates);
        }

        out
    }

    fn generate_candidates_with_threshold(
        &self,
        valuations: &[MarketValuation],
        threshold: f64,
        relaxed_mode: bool,
    ) -> Vec<CandidateTrade> {
        let mut out = Vec::new();
        let total_cost_prob = (self.cfg.fee_bps + self.cfg.slippage_bps) / 10_000.0;
        for v in valuations {
            let raw_edge = v.fair_prob_yes - v.market_mid_prob_yes;
            let adjusted = raw_edge.abs() - total_cost_prob;
            let effective_threshold = self.effective_threshold_for_valuation(v, threshold);
            if adjusted < effective_threshold {
                continue;
            }

            let (outcome_id, fair_price, observed_price, side) = if raw_edge >= 0.0 {
                (
                    "yes".to_string(),
                    v.fair_prob_yes.clamp(0.01, 0.99),
                    v.market_mid_prob_yes.clamp(0.01, 0.99),
                    Side::Buy,
                )
            } else {
                (
                    "no".to_string(),
                    (1.0 - v.fair_prob_yes).clamp(0.01, 0.99),
                    (1.0 - v.market_mid_prob_yes).clamp(0.01, 0.99),
                    Side::Buy,
                )
            };

            let mut rationale = v.rationale.clone();
            if relaxed_mode {
                rationale = format!("{rationale} [relaxed-threshold-candidate]");
            }

            out.push(CandidateTrade {
                ticker: v.ticker.clone(),
                side,
                outcome_id,
                fair_price,
                observed_price,
                edge_pct: adjusted,
                confidence: v.confidence,
                rationale,
            });
        }

        out.sort_by(|a, b| b.edge_pct.total_cmp(&a.edge_pct));
        out
    }

    pub fn candidate_to_signal(&self, c: &CandidateTrade) -> TradeSignal {
        TradeSignal {
            market_id: c.ticker.clone(),
            outcome_id: c.outcome_id.clone(),
            side: c.side,
            fair_price: c.fair_price,
            observed_price: c.observed_price,
            edge_pct: c.edge_pct,
            confidence: c.confidence,
            signal_timestamp: Utc::now(),
        }
    }

    pub fn candidate_diagnostics(&self, valuations: &[MarketValuation], top_n: usize) -> CandidateDiagnostics {
        let strict = self.generate_candidates_with_threshold(
            valuations,
            self.cfg.mispricing_threshold,
            false,
        );
        let relaxed = self.generate_candidates_with_threshold(
            valuations,
            self.cfg.fallback_mispricing_threshold,
            true,
        );
        let total_cost_prob = (self.cfg.fee_bps + self.cfg.slippage_bps) / 10_000.0;

        let mut edges: Vec<CandidateEdgeSnapshot> = valuations
            .iter()
            .map(|v| {
                let raw_edge = v.fair_prob_yes - v.market_mid_prob_yes;
                CandidateEdgeSnapshot {
                    ticker: v.ticker.clone(),
                    raw_edge,
                    adjusted_edge: raw_edge.abs() - total_cost_prob,
                    effective_threshold: self.effective_threshold_for_valuation(v, self.cfg.mispricing_threshold),
                    confidence: v.confidence,
                }
            })
            .collect();
        edges.sort_by(|a, b| b.adjusted_edge.total_cmp(&a.adjusted_edge));
        edges.truncate(top_n.max(1));

        CandidateDiagnostics {
            strict_threshold: self.cfg.mispricing_threshold,
            fallback_threshold: self.cfg.fallback_mispricing_threshold,
            min_candidates: self.cfg.min_candidates,
            adaptive_threshold_enabled: self.cfg.adaptive_threshold_enabled,
            total_cost_prob,
            strict_count: strict.len(),
            relaxed_count: relaxed.len(),
            top_edges: edges,
        }
    }

    pub fn last_run_summary(&self) -> ValuationRunSummary {
        self.last_run_summary
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    pub fn allow_heuristic_in_live(&self) -> bool {
        self.cfg.allow_heuristic_in_live
    }

    async fn infer_batch(
        &self,
        inputs: &[ValuationInput],
        claude_enabled: bool,
    ) -> Result<(Vec<MarketValuation>, BatchMode), ExecutionError> {
        if !claude_enabled {
            return Ok((
                self.heuristic_batch(inputs),
                BatchMode::Heuristic("claude cadence disabled for this cycle".to_string()),
            ));
        }
        if self.cfg.anthropic_api_key.is_none() {
            return Ok((
                self.heuristic_batch(inputs),
                BatchMode::Heuristic("missing ANTHROPIC_API_KEY".to_string()),
            ));
        }

        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.try_infer_claude(inputs).await {
                Ok(v) => return Ok((v, BatchMode::Claude)),
                Err(err @ ExecutionError::RetryableExchange(_)) if attempt <= self.cfg.max_retries => {
                    let _ = err;
                    tokio::time::sleep(Duration::from_millis(250 * attempt as u64)).await;
                }
                Err(err) => {
                    return Ok((
                        self.heuristic_batch(inputs),
                        BatchMode::Heuristic(format!("claude fallback: {err}")),
                    ))
                }
            }
        }
    }

    async fn try_infer_claude(&self, inputs: &[ValuationInput]) -> Result<Vec<MarketValuation>, ExecutionError> {
        let api_key = self
            .cfg
            .anthropic_api_key
            .as_ref()
            .ok_or_else(|| ExecutionError::Exchange("missing ANTHROPIC_API_KEY".to_string()))?;

        let prompt = build_prompt(inputs, self.cfg.max_prompt_chars);
        let body = ClaudeRequest {
            model: self.cfg.model.clone(),
            max_tokens: self.cfg.max_tokens_per_batch,
            temperature: 0.0,
            system: "You are a pricing engine. Output strict JSON only.".to_string(),
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: prompt,
            }],
        };

        let req = self
            .http
            .post(format!("{}/v1/messages", self.cfg.anthropic_base_url))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body);

        let resp = timeout(Duration::from_millis(self.cfg.timeout_ms), req.send())
            .await
            .map_err(|_| ExecutionError::RetryableExchange("claude timeout".to_string()))?
            .map_err(|e| ExecutionError::RetryableExchange(e.to_string()))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !status.is_success() {
            if status.is_server_error() || status.as_u16() == 429 {
                return Err(ExecutionError::RetryableExchange(format!(
                    "claude status {status}: {text}"
                )));
            }
            return Err(ExecutionError::Exchange(format!(
                "claude status {status}: {text}"
            )));
        }

        let response: ClaudeResponse =
            serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let content_text = response
            .content
            .iter()
            .find(|c| c.kind == "text")
            .map(|c| c.text.clone())
            .ok_or_else(|| ExecutionError::Exchange("claude missing text content".to_string()))?;

        let parsed = parse_claude_valuation_items(&content_text)?;

        let mut by_ticker: HashMap<String, &ValuationInput> = HashMap::new();
        for i in inputs {
            by_ticker.insert(i.market.ticker.clone(), i);
        }

        let mut out = Vec::new();
        for p in parsed {
            if let Some(input) = by_ticker.get(&p.ticker) {
                out.push(MarketValuation {
                    ticker: p.ticker,
                    fair_prob_yes: p.fair_prob_yes.clamp(0.01, 0.99),
                    market_mid_prob_yes: market_mid_prob_yes(&input.market).unwrap_or(0.5),
                    market_volume: input.market.volume,
                    spread_cents: input.market.spread_cents(),
                    confidence: p.confidence.clamp(0.0, 1.0),
                    rationale: p.rationale,
                    stale_after: Utc::now() + chrono::Duration::minutes(10),
                });
            }
        }

        if out.is_empty() {
            return Err(ExecutionError::Exchange("claude returned no valid valuations".to_string()));
        }
        Ok(out)
    }

    fn heuristic_batch(&self, inputs: &[ValuationInput]) -> Vec<MarketValuation> {
        inputs
            .iter()
            .map(|i| {
                let mid = market_mid_prob_yes(&i.market).unwrap_or(0.5);
                let enrich_bias = i
                    .enrichment
                    .as_ref()
                    .map(enrichment_bias)
                    .unwrap_or(0.0)
                    .clamp(-0.15, 0.15);
                let fair = (mid + enrich_bias).clamp(0.01, 0.99);
                MarketValuation {
                    ticker: i.market.ticker.clone(),
                    fair_prob_yes: fair,
                    market_mid_prob_yes: mid,
                    market_volume: i.market.volume,
                    spread_cents: i.market.spread_cents(),
                    confidence: i.enrichment.as_ref().map(|_| 0.62).unwrap_or(0.45),
                    rationale: "heuristic valuation fallback".to_string(),
                    stale_after: Utc::now() + chrono::Duration::minutes(10),
                }
            })
            .collect()
    }

    fn cache_key(&self, i: &ValuationInput) -> String {
        let mid = market_mid_prob_yes(&i.market).unwrap_or(0.5);
        let enrich = i
            .enrichment
            .as_ref()
            .map(enrichment_bias)
            .unwrap_or(0.0);
        format!(
            "{}|{:.4}|{:.1}|{:.4}",
            i.market.ticker, mid, i.market.volume, enrich
        )
    }

    fn get_cached(&self, i: &ValuationInput) -> Option<MarketValuation> {
        let key = self.cache_key(i);
        let guard = self.cache.lock().ok()?;
        let e = guard.get(&key)?;
        if Instant::now() <= e.expires_at {
            Some(e.value.clone())
        } else {
            None
        }
    }

    fn put_cached(&self, i: &ValuationInput, v: &MarketValuation) {
        let key = self.cache_key(i);
        if let Ok(mut guard) = self.cache.lock() {
            guard.insert(
                key,
                CacheEntry {
                    expires_at: Instant::now() + Duration::from_secs(self.cfg.cache_ttl_secs),
                    value: v.clone(),
                },
            );
        }
    }

    fn reset_last_run_summary(&self) {
        if let Ok(mut g) = self.last_run_summary.lock() {
            *g = ValuationRunSummary::default();
        }
    }

    fn record_batch_mode(&self, mode: BatchMode) {
        if let Ok(mut g) = self.last_run_summary.lock() {
            match mode {
                BatchMode::Claude => g.used_claude = true,
                BatchMode::Heuristic(reason) => {
                    g.used_heuristic = true;
                    if !g.fallback_reasons.iter().any(|r| r == &reason) {
                        g.fallback_reasons.push(reason);
                    }
                }
            }
        }
    }

    fn effective_threshold_for_valuation(&self, valuation: &MarketValuation, base_threshold: f64) -> f64 {
        let base = base_threshold.clamp(0.0, 1.0);
        if !self.cfg.adaptive_threshold_enabled {
            return base;
        }

        let floor = self.cfg.adaptive_threshold_floor.min(base);
        let liquidity_score = (valuation.market_volume / self.cfg.adaptive_liquidity_volume_ref).clamp(0.0, 1.0);
        let spread_score = valuation
            .spread_cents
            .map(|s| (1.0 - (s / self.cfg.adaptive_spread_ref_cents)).clamp(0.0, 1.0))
            .unwrap_or(0.5);
        let confidence_score = valuation.confidence.clamp(0.0, 1.0);
        let w_conf = self.cfg.adaptive_confidence_weight;
        let w_liq = self.cfg.adaptive_liquidity_weight;
        let w_spread = self.cfg.adaptive_spread_weight;
        let weight_sum = (w_conf + w_liq + w_spread).max(1e-9);
        let quality = ((w_conf * confidence_score) + (w_liq * liquidity_score) + (w_spread * spread_score))
            / weight_sum;

        floor + ((base - floor) * (1.0 - quality.clamp(0.0, 1.0)))
    }
}

#[derive(Debug, Clone)]
enum BatchMode {
    Claude,
    Heuristic(String),
}

fn parse_claude_valuation_items(content_text: &str) -> Result<Vec<ClaudeValuationItem>, ExecutionError> {
    if let Ok(parsed) = serde_json::from_str::<Vec<ClaudeValuationItem>>(content_text) {
        return Ok(parsed);
    }

    if let Some(inner) = strip_markdown_code_fence(content_text) {
        if let Ok(parsed) = serde_json::from_str::<Vec<ClaudeValuationItem>>(inner) {
            return Ok(parsed);
        }
    }

    if let Some(json_array) = extract_outer_json_array(content_text) {
        if let Ok(parsed) = serde_json::from_str::<Vec<ClaudeValuationItem>>(&json_array) {
            return Ok(parsed);
        }
    }

    Err(ExecutionError::Exchange(
        "invalid claude valuation JSON: unable to parse model response as JSON array".to_string(),
    ))
}

fn strip_markdown_code_fence(s: &str) -> Option<&str> {
    let trimmed = s.trim();
    if !trimmed.starts_with("```") {
        return None;
    }
    let after_open = trimmed.strip_prefix("```")?;
    let newline_idx = after_open.find('\n')?;
    let after_header = &after_open[(newline_idx + 1)..];
    let close_idx = after_header.rfind("```")?;
    Some(after_header[..close_idx].trim())
}

fn extract_outer_json_array(s: &str) -> Option<String> {
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    if end <= start {
        return None;
    }
    Some(s[start..=end].trim().to_string())
}

fn build_prompt(inputs: &[ValuationInput], max_chars: usize) -> String {
    let mut prompt = String::from(
        "For each market, estimate fair_prob_yes in [0,1], confidence in [0,1], and short rationale.\nReturn ONLY JSON array of objects: {ticker, fair_prob_yes, confidence, rationale}\nMarkets:\n",
    );
    for i in inputs {
        let mid = market_mid_prob_yes(&i.market).unwrap_or(0.5);
        let enrich = i
            .enrichment
            .as_ref()
            .map(enrichment_bias)
            .unwrap_or(0.0);
        let line = format!(
            "- ticker={} title={} mid_prob_yes={:.4} volume={:.1} enrich_bias={:.4}\n",
            i.market.ticker, i.market.title, mid, i.market.volume, enrich
        );
        if prompt.len() + line.len() > max_chars {
            break;
        }
        prompt.push_str(&line);
    }
    prompt
}

fn market_mid_prob_yes(market: &ScannedMarket) -> Option<f64> {
    let bid = market.yes_bid_cents?;
    let ask = market.yes_ask_cents?;
    Some(((bid + ask) / 2.0 / 100.0).clamp(0.01, 0.99))
}

fn enrichment_bias(e: &MarketEnrichment) -> f64 {
    e.weather_signal
        .or(e.sports_injury_signal)
        .or(e.crypto_sentiment_signal)
        .unwrap_or(0.0)
        * 0.08
}

#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: u32,
    temperature: f64,
    system: String,
    messages: Vec<ClaudeMessage>,
}

#[derive(Debug, Serialize)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    content: Vec<ClaudeContent>,
}

#[derive(Debug, Deserialize)]
struct ClaudeContent {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeValuationItem {
    ticker: String,
    fair_prob_yes: f64,
    confidence: f64,
    rationale: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::market_scanner::ScannedMarket;

    fn input(mid_bid: f64, mid_ask: f64) -> ValuationInput {
        ValuationInput {
            market: ScannedMarket {
                ticker: "KXTEST".to_string(),
                title: "Bitcoin above 120k".to_string(),
                yes_bid_cents: Some(mid_bid),
                yes_ask_cents: Some(mid_ask),
                volume: 10_000.0,
                close_time: None,
            },
            enrichment: None,
        }
    }

    #[test]
    fn candidate_generation_respects_threshold() {
        let engine = ClaudeValuationEngine::new(ValuationConfig::default());
        let vals = vec![MarketValuation {
            ticker: "KX".to_string(),
            fair_prob_yes: 0.70,
            market_mid_prob_yes: 0.50,
            market_volume: 10_000.0,
            spread_cents: Some(4.0),
            confidence: 0.8,
            rationale: "x".to_string(),
            stale_after: Utc::now(),
        }];
        let c = engine.generate_candidates(&vals);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].outcome_id, "yes");
    }

    #[test]
    fn candidate_generation_can_backfill_with_relaxed_threshold() {
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            min_candidates: 1,
            fallback_mispricing_threshold: 0.01,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![MarketValuation {
            ticker: "KX".to_string(),
            fair_prob_yes: 0.53,
            market_mid_prob_yes: 0.50,
            market_volume: 10_000.0,
            spread_cents: Some(4.0),
            confidence: 0.8,
            rationale: "x".to_string(),
            stale_after: Utc::now(),
        }];
        let c = engine.generate_candidates(&vals);
        assert_eq!(c.len(), 1);
        assert!(c[0].rationale.contains("relaxed-threshold-candidate"));
    }

    #[test]
    fn adaptive_threshold_lowers_bar_for_liquid_tight_markets() {
        let cfg = ValuationConfig {
            adaptive_threshold_enabled: true,
            mispricing_threshold: 0.08,
            adaptive_threshold_floor: 0.01,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![MarketValuation {
            ticker: "KX".to_string(),
            fair_prob_yes: 0.56,
            market_mid_prob_yes: 0.52,
            market_volume: 120_000.0,
            spread_cents: Some(2.0),
            confidence: 0.85,
            rationale: "x".to_string(),
            stale_after: Utc::now(),
        }];
        let c = engine.generate_candidates(&vals);
        assert_eq!(c.len(), 1);
    }

    #[tokio::test]
    async fn heuristic_valuation_works_without_api_key() {
        let cfg = ValuationConfig {
            anthropic_api_key: None,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let out = engine
            .value_markets(&[input(45.0, 55.0)])
            .await
            .expect("should work");
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn cadence_disabled_forces_heuristic_mode() {
        let cfg = ValuationConfig {
            anthropic_api_key: Some("test-key".to_string()),
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let out = engine
            .value_markets_with_claude_enabled(&[input(45.0, 55.0)], false)
            .await
            .expect("should use heuristic");
        assert_eq!(out.len(), 1);
        let summary = engine.last_run_summary();
        assert!(summary.used_heuristic);
        assert!(summary
            .fallback_reasons
            .iter()
            .any(|r| r.contains("cadence disabled")));
    }

    #[test]
    fn parses_plain_json_array_response() {
        let raw =
            r#"[{"ticker":"KX1","fair_prob_yes":0.61,"confidence":0.72,"rationale":"alpha"}]"#;
        let parsed = parse_claude_valuation_items(raw).expect("json should parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].ticker, "KX1");
    }

    #[test]
    fn parses_fenced_json_array_response() {
        let raw = "```json\n[{\"ticker\":\"KX2\",\"fair_prob_yes\":0.42,\"confidence\":0.55,\"rationale\":\"beta\"}]\n```";
        let parsed = parse_claude_valuation_items(raw).expect("fenced json should parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].ticker, "KX2");
    }

    #[test]
    fn parses_embedded_json_array_response() {
        let raw = "Here is the output:\n[{\"ticker\":\"KX3\",\"fair_prob_yes\":0.35,\"confidence\":0.51,\"rationale\":\"gamma\"}]";
        let parsed = parse_claude_valuation_items(raw).expect("embedded json should parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].ticker, "KX3");
    }
}
