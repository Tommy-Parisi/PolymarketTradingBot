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
use crate::features::forecast::extract_threshold_from_ticker;

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
            timeout_ms: 20_000,
            max_tokens_per_batch: 8_000,
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
            fee_bps: 5.0,
            slippage_bps: 5.0,
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
    pub yes_ask_cents: Option<f64>,
    pub yes_bid_cents: Option<f64>,
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

    #[allow(dead_code)]
    pub async fn value_markets(&self, inputs: &[ValuationInput]) -> Result<Vec<MarketValuation>, ExecutionError> {
        self.value_markets_with_claude_enabled(inputs, true).await
    }

    pub async fn value_markets_with_claude_enabled(
        &self,
        inputs: &[ValuationInput],
        claude_enabled: bool,
    ) -> Result<Vec<MarketValuation>, ExecutionError> {
        self.value_markets_impl(inputs, claude_enabled, true).await
    }

    /// Like `value_markets_with_claude_enabled` but does not write results to the cache.
    /// Used for heuristic pre-passes (e.g. OnHeuristicCandidates) where the results are
    /// only used for candidate screening and will be superseded by a Claude pass.
    pub async fn value_markets_screening(
        &self,
        inputs: &[ValuationInput],
    ) -> Result<Vec<MarketValuation>, ExecutionError> {
        self.value_markets_impl(inputs, false, false).await
    }

    async fn value_markets_impl(
        &self,
        inputs: &[ValuationInput],
        claude_enabled: bool,
        write_cache: bool,
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
            for mut v in inferred {
                if let Some(input) = input_by_ticker.get(&v.ticker) {
                    // Attach actual bid/ask from market data — Claude doesn't return these.
                    v.yes_ask_cents = input.market.yes_ask_cents;
                    v.yes_bid_cents = input.market.yes_bid_cents;
                    if write_cache {
                        self.put_cached(input, &v);
                    }
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
            // Skip markets where the price has converged to near-certainty. These are markets
            // that have already effectively resolved (live scores, in-game events) — the price
            // is correct and any apparent edge is an artefact of the model lagging behind.
            // YES bid ≥ 90¢ → market thinks YES is near-certain; no real NO edge available.
            // YES ask ≤ 10¢ → market thinks NO is near-certain; no real YES edge available.
            // The mid_prob_yes guards catch the same condition when bid/ask are absent (None),
            // preventing near-zero observed prices that the clamp would otherwise mask.
            if v.yes_bid_cents.map(|b| b >= 90.0).unwrap_or(false)
                || v.yes_ask_cents.map(|a| a <= 10.0).unwrap_or(false)
                || v.market_mid_prob_yes >= 0.90
                || v.market_mid_prob_yes <= 0.10
            {
                continue;
            }

            // Use actual ask/bid for edge — paying mid is not achievable with IOC.
            // YES buy: edge = fair - ask. NO buy: edge = (1-fair) - (1-bid) = bid - fair... wait,
            // NO ask = (100 - yes_bid) / 100, so NO edge = (1-fair) - no_ask.
            let yes_ask = v.yes_ask_cents.map(|c| c / 100.0).unwrap_or(v.market_mid_prob_yes);
            let no_ask = v.yes_bid_cents.map(|c| (100.0 - c) / 100.0).unwrap_or(1.0 - v.market_mid_prob_yes);

            let yes_edge = v.fair_prob_yes - yes_ask;
            let no_edge = (1.0 - v.fair_prob_yes) - no_ask;

            let (raw_edge, outcome_id, fair_price, observed_price) = if yes_edge >= no_edge {
                (yes_edge, "yes".to_string(), v.fair_prob_yes.clamp(0.01, 0.99), yes_ask.clamp(0.01, 0.99))
            } else {
                (no_edge, "no".to_string(), (1.0 - v.fair_prob_yes).clamp(0.01, 0.99), no_ask.clamp(0.01, 0.99))
            };
            let side = Side::Buy;

            let adjusted = raw_edge - total_cost_prob;
            let effective_threshold = self.effective_threshold_for_valuation(v, threshold);
            if adjusted < effective_threshold {
                continue;
            }

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
            signal_origin: Some(if c.rationale.contains("forced deterministic test candidate") {
                "bootstrap_synthetic".to_string()
            } else {
                "model_candidate".to_string()
            }),
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
                let yes_ask = v.yes_ask_cents.map(|c| c / 100.0).unwrap_or(v.market_mid_prob_yes);
                let no_ask = v.yes_bid_cents.map(|c| (100.0 - c) / 100.0).unwrap_or(1.0 - v.market_mid_prob_yes);
                let raw_edge = (v.fair_prob_yes - yes_ask).max((1.0 - v.fair_prob_yes) - no_ask);
                CandidateEdgeSnapshot {
                    ticker: v.ticker.clone(),
                    raw_edge,
                    adjusted_edge: raw_edge - total_cost_prob,
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
            messages: vec![
                ClaudeMessage {
                    role: "user".to_string(),
                    content: prompt,
                },
                // Assistant prefill: forces the response to begin with '[', guaranteeing
                // we get a JSON array without any preamble text.
                ClaudeMessage {
                    role: "assistant".to_string(),
                    content: "[".to_string(),
                },
            ],
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

        // The assistant prefill consumed the opening '[', so we restore it before parsing.
        let content_with_prefix = format!("[{content_text}");
        let parsed = parse_claude_valuation_items(&content_with_prefix).map_err(|e| {
            let preview: String = content_with_prefix.chars().take(300).collect();
            ExecutionError::Exchange(format!("{e} | raw: {preview}"))
        })?;

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
                    yes_ask_cents: input.market.yes_ask_cents,
                    yes_bid_cents: input.market.yes_bid_cents,
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
                    .map(|e| enrichment_bias(e, &i.market.ticker))
                    .unwrap_or(0.0)
                    .clamp(-0.15, 0.15);
                let fair = (mid + enrich_bias).clamp(0.01, 0.99);
                MarketValuation {
                    ticker: i.market.ticker.clone(),
                    fair_prob_yes: fair,
                    market_mid_prob_yes: mid,
                    yes_ask_cents: i.market.yes_ask_cents,
                    yes_bid_cents: i.market.yes_bid_cents,
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
            .map(|e| enrichment_bias(e, &i.market.ticker))
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
    let now = chrono::Utc::now();
    let mut prompt = String::from(
        "For each market, estimate fair_prob_yes in [0,1], confidence in [0,1], and short rationale.\nReturn ONLY JSON array of objects: {ticker, fair_prob_yes, confidence, rationale}\nMarkets:\n",
    );
    for i in inputs {
        let mid = market_mid_prob_yes(&i.market).unwrap_or(0.5);
        let enrich = i
            .enrichment
            .as_ref()
            .map(|e| enrichment_bias(e, &i.market.ticker))
            .unwrap_or(0.0);
        let spread = i.market.spread_cents().map(|s| format!("{:.1}c", s)).unwrap_or_else(|| "?".to_string());
        let ttc = i.market.close_time.map(|c| {
            let hours = (c - now).num_minutes() as f64 / 60.0;
            if hours < 1.0 {
                format!("{:.0}m", hours * 60.0)
            } else if hours < 48.0 {
                format!("{:.1}h", hours)
            } else {
                format!("{:.1}d", hours / 24.0)
            }
        }).unwrap_or_else(|| "?".to_string());
        let line = format!(
            "- ticker={} title={} mid={:.4} spread={} ttc={} vol={:.0} enrich={:.4}\n",
            i.market.ticker, i.market.title, mid, spread, ttc, i.market.volume, enrich
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

/// Converts an enrichment signal into an additive probability bias in roughly [-0.08, +0.08].
///
/// For weather: `weather_signal` is raw °F from NOAA. We normalize it relative to the
/// market-specific threshold extracted from the ticker (e.g. "will high exceed 55°F?").
///   signal = (forecast_temp - threshold) / 15.0  clamped to [-1, 1]
/// A forecast 15°F above the threshold → +1.0 → +0.08 probability bias toward YES.
/// If no threshold can be extracted the weather signal contributes nothing (ambiguous direction).
///
/// Sports/crypto signals are already in [-1, 1] from their respective feeds.
fn enrichment_bias(e: &MarketEnrichment, ticker: &str) -> f64 {
    let raw_signal = if let Some(temp) = e.weather_signal {
        match extract_threshold_from_ticker(ticker) {
            Some(threshold) => ((temp - threshold) / 15.0).clamp(-1.0, 1.0),
            None => 0.0, // without a threshold we can't tell if temp is good or bad for YES
        }
    } else {
        // For crypto: prefer threshold-relative price signal (more specific than sentiment).
        // For esports: use win-probability signal when available.
        // Fall back through the chain to 0.0 when nothing is configured.
        e.sports_injury_signal
            .or(e.esports_signal)
            .or(e.finance_price_signal)
            .or(e.crypto_price_signal)
            .or(e.crypto_sentiment_signal)
            .unwrap_or(0.0)
    };
    raw_signal * 0.05
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
                subtitle: None,
                market_type: None,
                event_ticker: None,
                series_ticker: None,
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
            yes_ask_cents: Some(52.0),
            yes_bid_cents: Some(48.0),
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
            yes_ask_cents: Some(52.0),
            yes_bid_cents: Some(48.0),
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
            yes_ask_cents: Some(53.0),
            yes_bid_cents: Some(51.0),
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

    fn make_valuation(fair: f64, ask_cents: Option<f64>, bid_cents: Option<f64>) -> MarketValuation {
        let mid = ask_cents.and_then(|a| bid_cents.map(|b| (a + b) / 2.0 / 100.0)).unwrap_or(0.5);
        MarketValuation {
            ticker: "KX".to_string(),
            fair_prob_yes: fair,
            market_mid_prob_yes: mid,
            yes_ask_cents: ask_cents,
            yes_bid_cents: bid_cents,
            market_volume: 5_000.0,
            spread_cents: ask_cents.and_then(|a| bid_cents.map(|b| a - b)),
            confidence: 0.8,
            rationale: "test".to_string(),
            stale_after: Utc::now(),
        }
    }

    #[test]
    fn ask_based_edge_yes_buy_route() {
        // fair=0.70, ask=0.55 → yes_edge=0.15, no_ask=(100-55)/100=0.45, no_edge=(1-0.70)-0.45=-0.15
        // YES edge wins → outcome_id=yes, observed_price=0.55
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![make_valuation(0.70, Some(55.0), Some(45.0))];
        let candidates = engine.generate_candidates(&vals);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].outcome_id, "yes");
        assert!((candidates[0].observed_price - 0.55).abs() < 1e-9);
        assert!((candidates[0].fair_price - 0.70).abs() < 1e-9);
    }

    #[test]
    fn ask_based_edge_no_buy_route() {
        // fair=0.25, ask=0.55, bid=0.45
        // yes_edge = 0.25 - 0.55 = -0.30
        // no_ask = (100-45)/100 = 0.55
        // no_edge = (1-0.25) - 0.55 = 0.75 - 0.55 = 0.20
        // NO edge wins → outcome_id=no, observed_price=0.55, fair_price=1-0.25=0.75
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![make_valuation(0.25, Some(55.0), Some(45.0))];
        let candidates = engine.generate_candidates(&vals);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].outcome_id, "no");
        assert!((candidates[0].observed_price - 0.55).abs() < 1e-9);
        assert!((candidates[0].fair_price - 0.75).abs() < 1e-9);
    }

    #[test]
    fn edge_below_threshold_produces_no_candidate() {
        // fair=0.52, ask=0.52 → raw_edge=0.0, which is below threshold 0.08
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![make_valuation(0.52, Some(52.0), Some(48.0))];
        let candidates = engine.generate_candidates(&vals);
        assert!(candidates.is_empty(), "expected no candidates for edge=0");
    }

    #[test]
    fn fee_and_slippage_reduce_effective_edge() {
        // fair=0.65, ask=0.55 → raw_edge=0.10
        // fee_bps=15, slippage_bps=20 → total_cost = 35/10000 = 0.0035
        // adjusted = 0.10 - 0.0035 = 0.0965, threshold=0.08 → should pass
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 15.0,
            slippage_bps: 20.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![make_valuation(0.65, Some(55.0), Some(45.0))];
        let candidates = engine.generate_candidates(&vals);
        assert_eq!(candidates.len(), 1);
        // edge_pct ≈ 0.0965
        assert!((candidates[0].edge_pct - 0.0965).abs() < 0.001);
    }

    #[test]
    fn edge_strictly_above_threshold_is_included_and_at_or_below_is_not() {
        // Filter uses adjusted < threshold (strict), so edge must be > threshold to pass.
        // Due to float arithmetic, test with clear margin above/below rather than equality.
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);

        // fair=0.65, ask=0.55 → yes_edge=0.10 > 0.08 → included
        let above = engine.generate_candidates(&[make_valuation(0.65, Some(55.0), Some(45.0))]);
        assert_eq!(above.len(), 1);

        // fair=0.60, ask=0.55 → yes_edge=0.05 < 0.08 → excluded
        let below = engine.generate_candidates(&[make_valuation(0.60, Some(55.0), Some(45.0))]);
        assert!(below.is_empty());
    }

    #[test]
    fn mid_price_used_as_fallback_when_no_ask() {
        // No ask_cents → falls back to market_mid_prob_yes = 0.50
        // fair=0.62 → yes_edge = 0.62 - 0.50 = 0.12, threshold=0.08 → pass
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        // Construct manually since make_valuation mid depends on bid/ask
        let vals = vec![MarketValuation {
            ticker: "KX".to_string(),
            fair_prob_yes: 0.62,
            market_mid_prob_yes: 0.50,
            yes_ask_cents: None,
            yes_bid_cents: None,
            market_volume: 5_000.0,
            spread_cents: None,
            confidence: 0.8,
            rationale: "test".to_string(),
            stale_after: Utc::now(),
        }];
        let candidates = engine.generate_candidates(&vals);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].outcome_id, "yes");
        assert!((candidates[0].observed_price - 0.50).abs() < 1e-9);
    }

    #[test]
    fn converged_yes_bid_suppresses_candidate() {
        // YES bid=99 → market near-certain YES; any apparent NO edge is spurious.
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        // fair=0.58, NO ask=(100-99)/100=0.01 → raw NO edge = 0.42 − huge apparent edge
        let vals = vec![make_valuation(0.58, Some(100.0), Some(99.0))];
        let candidates = engine.generate_candidates(&vals);
        assert!(candidates.is_empty(), "converged market (bid=99) should produce no candidates");
    }

    #[test]
    fn converged_yes_ask_suppresses_candidate() {
        // YES ask=1 → market near-certain NO; any apparent YES edge is spurious.
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        let vals = vec![make_valuation(0.58, Some(1.0), Some(0.0))];
        let candidates = engine.generate_candidates(&vals);
        assert!(candidates.is_empty(), "converged market (ask=1) should produce no candidates");
    }

    #[test]
    fn non_converged_market_still_produces_candidate() {
        // YES bid=68, ask=72 → normal open market; should still generate candidates when edge > threshold.
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);
        // fair=0.82, ask=0.72 → YES edge=0.10 > 0.08
        let vals = vec![make_valuation(0.82, Some(72.0), Some(68.0))];
        let candidates = engine.generate_candidates(&vals);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].outcome_id, "yes");
    }

    #[test]
    fn converged_mid_without_bid_ask_suppresses_candidate() {
        // When bid/ask are None the clamp on no_ask hides the near-zero price, so we gate on
        // market_mid_prob_yes directly. mid=0.97 → no real NO edge; mid=0.03 → no real YES edge.
        let cfg = ValuationConfig {
            mispricing_threshold: 0.08,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            ..ValuationConfig::default()
        };
        let engine = ClaudeValuationEngine::new(cfg);

        // High mid (near-certain YES) with no bid/ask: apparent NO edge is huge but spurious.
        let val_high = MarketValuation {
            ticker: "KXHIGH".to_string(),
            fair_prob_yes: 0.38, // heuristic disagrees, but market is near-certain
            market_mid_prob_yes: 0.97,
            yes_ask_cents: None,
            yes_bid_cents: None,
            market_volume: 5_000.0,
            spread_cents: None,
            confidence: 0.8,
            rationale: "test".to_string(),
            stale_after: Utc::now(),
        };
        let candidates = engine.generate_candidates(&[val_high]);
        assert!(
            candidates.is_empty(),
            "converged market (mid=0.97, no bid/ask) should produce no candidates"
        );

        // Low mid (near-certain NO) with no bid/ask: apparent YES edge is huge but spurious.
        let val_low = MarketValuation {
            ticker: "KXLOW".to_string(),
            fair_prob_yes: 0.62,
            market_mid_prob_yes: 0.03,
            yes_ask_cents: None,
            yes_bid_cents: None,
            market_volume: 5_000.0,
            spread_cents: None,
            confidence: 0.8,
            rationale: "test".to_string(),
            stale_after: Utc::now(),
        };
        let candidates = engine.generate_candidates(&[val_low]);
        assert!(
            candidates.is_empty(),
            "converged market (mid=0.03, no bid/ask) should produce no candidates"
        );
    }

    fn weather_enrichment(temp_f: f64) -> MarketEnrichment {
        crate::data::market_enrichment::MarketEnrichment {
            ticker: "KXHIGH".to_string(),
            vertical: crate::data::market_enrichment::MarketVertical::Weather,
            weather_signal: Some(temp_f),
            sports_injury_signal: None,
            crypto_sentiment_signal: None,
            crypto_price_signal: None,
            esports_signal: None,
            finance_price_signal: None,
            generated_at: Utc::now(),
        }
    }

    #[test]
    fn enrichment_bias_weather_forecast_above_threshold() {
        // ticker threshold = 55°F, forecast = 70°F → (70-55)/15 = 1.0 clamped → bias = +0.08
        let e = weather_enrichment(70.0);
        let bias = enrichment_bias(&e, "KXHIGHCHI-26MAR25-T55");
        assert!((bias - 0.08).abs() < 1e-9, "bias={bias}");
    }

    #[test]
    fn enrichment_bias_weather_forecast_below_threshold() {
        // forecast = 40°F, threshold = 55°F → (40-55)/15 = -1.0 clamped → bias = -0.08
        let e = weather_enrichment(40.0);
        let bias = enrichment_bias(&e, "KXHIGHCHI-26MAR25-T55");
        assert!((bias - (-0.08)).abs() < 1e-9, "bias={bias}");
    }

    #[test]
    fn enrichment_bias_weather_partial_signal() {
        // forecast = 62°F, threshold = 55°F → (62-55)/15 = 0.4667 → bias ≈ 0.0373
        let e = weather_enrichment(62.0);
        let bias = enrichment_bias(&e, "KXHIGHCHI-26MAR25-T55");
        let expected = (7.0_f64 / 15.0) * 0.08;
        assert!((bias - expected).abs() < 1e-9, "bias={bias} expected={expected}");
    }

    #[test]
    fn enrichment_bias_weather_no_threshold_in_ticker_returns_zero() {
        // No -T segment → can't determine direction → no bias
        let e = weather_enrichment(85.0);
        let bias = enrichment_bias(&e, "KXWEATHER-NYC");
        assert_eq!(bias, 0.0, "bias should be 0 when no threshold can be extracted");
    }

    #[test]
    fn enrichment_bias_sports_injury_unchanged() {
        // Sports signal is already in [-1, 1]; just scaled by 0.08
        let e = crate::data::market_enrichment::MarketEnrichment {
            ticker: "KXNBA".to_string(),
            vertical: crate::data::market_enrichment::MarketVertical::Sports,
            weather_signal: None,
            sports_injury_signal: Some(0.5),
            crypto_sentiment_signal: None,
            crypto_price_signal: None,
            esports_signal: None,
            finance_price_signal: None,
            generated_at: Utc::now(),
        };
        let bias = enrichment_bias(&e, "KXNBAGAME-26MAR24BOSNYC-BOS");
        assert!((bias - 0.04).abs() < 1e-9, "bias={bias}");
    }
}
