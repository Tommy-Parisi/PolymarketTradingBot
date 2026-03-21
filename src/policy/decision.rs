use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::execution::types::{ExecutionError, Side, TimeInForce};
use crate::features::execution::{ExecutionContext, build_execution_feature_row_from_forecast};
use crate::features::forecast::ForecastFeatureRow;
use crate::model::valuation::CandidateTrade;
use crate::models::execution::{ExecutionEstimate, ExecutionModel};
use crate::models::forecast::ForecastOutput;
use crate::{data::market_scanner::ScannedMarket};

#[derive(Debug, Clone)]
pub struct PolicyConfig {
    pub mode: PolicyMode,
    pub shadow_enabled: bool,
    pub shadow_root: PathBuf,
    pub min_expected_realized_pnl: f64,
    pub max_actions_per_candidate: usize,
    pub default_legacy_fallback: bool,
}

impl PolicyConfig {
    pub fn from_env() -> Self {
        Self {
            mode: PolicyMode::from_env(),
            shadow_enabled: matches!(
                std::env::var("BOT_POLICY_SHADOW_ENABLED")
                    .unwrap_or_else(|_| "true".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            shadow_root: PathBuf::from(
                std::env::var("BOT_SHADOW_DIR").unwrap_or_else(|_| "var/shadow".to_string()),
            ),
            min_expected_realized_pnl: std::env::var("BOT_POLICY_MIN_EXPECTED_REALIZED_PNL")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0),
            max_actions_per_candidate: std::env::var("BOT_POLICY_MAX_ACTIONS_PER_CANDIDATE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4)
                .max(1),
            default_legacy_fallback: matches!(
                std::env::var("BOT_POLICY_DEFAULT_LEGACY_FALLBACK")
                    .unwrap_or_else(|_| "true".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PolicyMode {
    Legacy,
    Shadow,
    Active,
}

impl PolicyMode {
    pub fn from_env() -> Self {
        match std::env::var("BOT_POLICY_MODE")
            .unwrap_or_else(|_| "legacy".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "shadow" => Self::Shadow,
            "active" => Self::Active,
            _ => Self::Legacy,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyDecision {
    pub ticker: String,
    pub outcome_id: String,
    pub side: Side,
    pub should_trade: bool,
    pub limit_price: f64,
    pub time_in_force: TimeInForce,
    pub size_multiplier: f64,
    pub expected_gross_edge: f64,
    pub expected_realized_pnl: f64,
    pub expected_fill_prob: f64,
    pub rejection_reason: Option<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyShadowRecord {
    pub cycle_id: String,
    pub recorded_at: DateTime<Utc>,
    pub ticker: String,
    pub outcome_id: String,
    pub legacy_edge_pct: f64,
    pub legacy_confidence: f64,
    pub chosen_tif: TimeInForce,
    pub chosen_limit_price: f64,
    pub should_trade: bool,
    pub size_multiplier: f64,
    pub expected_fill_prob: f64,
    pub expected_gross_edge: f64,
    pub expected_realized_pnl: f64,
    pub rejection_reason: Option<String>,
    pub rationale: String,
}

struct ActionCandidate {
    tif: TimeInForce,
    limit_price: f64,
}

pub fn decide_shadow_policy(
    cfg: &PolicyConfig,
    execution_model: &ExecutionModel,
    market: &ScannedMarket,
    forecast: &ForecastFeatureRow,
    forecast_output: &ForecastOutput,
    candidate: &CandidateTrade,
) -> PolicyDecision {
    let actions = build_action_grid(market, candidate, cfg.max_actions_per_candidate);
    let mut best: Option<PolicyDecision> = None;
    for action in actions {
        let feature = build_execution_feature_row_from_forecast(
            forecast,
            market,
            candidate,
            action.tif,
            action.limit_price,
            &ExecutionContext {
                open_order_count_same_ticker: 0,
                recent_fill_count_same_ticker: 0,
                recent_cancel_count_same_ticker: 0,
                same_event_exposure_notional: 0.0,
            },
        );
        let execution = execution_model.predict(&feature);
        let decision = score_action(cfg, candidate, forecast_output, execution);
        if best
            .as_ref()
            .map(|current| decision.expected_realized_pnl > current.expected_realized_pnl)
            .unwrap_or(true)
        {
            best = Some(decision);
        }
    }

    best.unwrap_or_else(|| PolicyDecision {
        ticker: candidate.ticker.clone(),
        outcome_id: candidate.outcome_id.clone(),
        side: candidate.side,
        should_trade: cfg.default_legacy_fallback,
        limit_price: candidate.observed_price,
        time_in_force: TimeInForce::Gtc,
        size_multiplier: if cfg.default_legacy_fallback { candidate.confidence.clamp(0.1, 1.0) } else { 0.0 },
        expected_gross_edge: candidate.fair_price - candidate.observed_price,
        expected_realized_pnl: if cfg.default_legacy_fallback { candidate.edge_pct * 100.0 } else { 0.0 },
        expected_fill_prob: if cfg.default_legacy_fallback { 1.0 } else { 0.0 },
        rejection_reason: (!cfg.default_legacy_fallback).then_some("no_action_candidates".to_string()),
        rationale: if cfg.default_legacy_fallback {
            "policy fallback to legacy candidate".to_string()
        } else {
            "policy fallback".to_string()
        },
    })
}

pub fn record_shadow_decisions(
    cfg: &PolicyConfig,
    cycle_id: &str,
    candidates: &[(&CandidateTrade, &PolicyDecision)],
) -> Result<(), ExecutionError> {
    if !cfg.shadow_enabled || candidates.is_empty() {
        return Ok(());
    }
    let day = Utc::now().format("%Y-%m-%d").to_string();
    let path = cfg
        .shadow_root
        .join("policy")
        .join(day)
        .join("policy_shadow.jsonl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for (candidate, decision) in candidates {
        let record = PolicyShadowRecord {
            cycle_id: cycle_id.to_string(),
            recorded_at: Utc::now(),
            ticker: candidate.ticker.clone(),
            outcome_id: candidate.outcome_id.clone(),
            legacy_edge_pct: candidate.edge_pct,
            legacy_confidence: candidate.confidence,
            chosen_tif: decision.time_in_force,
            chosen_limit_price: decision.limit_price,
            should_trade: decision.should_trade,
            size_multiplier: decision.size_multiplier,
            expected_fill_prob: decision.expected_fill_prob,
            expected_gross_edge: decision.expected_gross_edge,
            expected_realized_pnl: decision.expected_realized_pnl,
            rejection_reason: decision.rejection_reason.clone(),
            rationale: decision.rationale.clone(),
        };
        let line = serde_json::to_string(&record)
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    Ok(())
}

fn build_action_grid(
    market: &ScannedMarket,
    candidate: &CandidateTrade,
    max_actions: usize,
) -> Vec<ActionCandidate> {
    let mut prices = vec![
        (TimeInForce::Gtc, candidate.observed_price),
        (TimeInForce::Ioc, market.yes_ask_cents.map(|v| v / 100.0).unwrap_or(candidate.observed_price)),
    ];
    if let Some(spread) = market.spread_cents() {
        let one_tick_less_aggressive = (candidate.observed_price - (spread / 200.0)).clamp(0.01, 0.99);
        let hybrid_price = (candidate.observed_price + (spread / 400.0)).clamp(0.01, 0.99);
        prices.push((TimeInForce::Gtc, one_tick_less_aggressive));
        prices.push((TimeInForce::Ioc, hybrid_price));
    } else {
        prices.push((TimeInForce::Gtc, (candidate.observed_price - 0.01).clamp(0.01, 0.99)));
        prices.push((TimeInForce::Ioc, (candidate.observed_price + 0.01).clamp(0.01, 0.99)));
    }
    prices.truncate(max_actions);
    prices
        .into_iter()
        .map(|(tif, limit_price)| ActionCandidate {
            tif,
            limit_price,
        })
        .collect()
}

fn score_action(
    cfg: &PolicyConfig,
    candidate: &CandidateTrade,
    forecast_output: &ForecastOutput,
    execution: ExecutionEstimate,
) -> PolicyDecision {
    let chosen_fair_price = if candidate.outcome_id.eq_ignore_ascii_case("yes") {
        forecast_output.fair_prob_yes
    } else {
        1.0 - forecast_output.fair_prob_yes
    }
    .clamp(0.01, 0.99);
    let expected_gross_edge = chosen_fair_price - execution.expected_fill_price;
    let markout_value = (execution.expected_markout_bps_5m / 10_000.0) * execution.expected_fill_price;
    let expected_realized_pnl = execution.fill_prob_5m
        * ((expected_gross_edge + markout_value) / execution.expected_fill_price.max(0.01))
        * 100.0;
    let should_trade = expected_realized_pnl >= cfg.min_expected_realized_pnl;
    let size_multiplier = if should_trade {
        (execution.fill_prob_5m * forecast_output.confidence).clamp(0.10, 1.0)
    } else {
        0.0
    };
    let rejection_reason = (!should_trade).then_some(format!(
        "expected_realized_pnl_below_threshold({:.4} < {:.4})",
        expected_realized_pnl, cfg.min_expected_realized_pnl
    ));
    PolicyDecision {
        ticker: candidate.ticker.clone(),
        outcome_id: candidate.outcome_id.clone(),
        side: candidate.side,
        should_trade,
        limit_price: execution.candidate_limit_price,
        time_in_force: execution.tif,
        size_multiplier,
        expected_gross_edge,
        expected_realized_pnl,
        expected_fill_prob: execution.fill_prob_5m,
        rejection_reason,
        rationale: format!(
            "policy shadow: fair={:.4} fill5m={:.4} fill_price={:.4} markout5m_bps={:.2}",
            chosen_fair_price,
            execution.fill_prob_5m,
            execution.expected_fill_price,
            execution.expected_markout_bps_5m
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::forecast::ForecastOutput;

    #[test]
    fn policy_rejects_negative_expected_value() {
        let cfg = PolicyConfig {
            mode: PolicyMode::Shadow,
            shadow_enabled: true,
            shadow_root: PathBuf::from("var/shadow"),
            min_expected_realized_pnl: 0.0,
            max_actions_per_candidate: 4,
            default_legacy_fallback: true,
        };
        let candidate = CandidateTrade {
            ticker: "KXTEST".to_string(),
            side: Side::Buy,
            outcome_id: "yes".to_string(),
            fair_price: 0.55,
            observed_price: 0.50,
            edge_pct: 0.05,
            confidence: 0.7,
            rationale: "x".to_string(),
        };
        let decision = score_action(
            &cfg,
            &candidate,
            &ForecastOutput {
                ticker: "KXTEST".to_string(),
                fair_prob_yes: 0.45,
                uncertainty: 0.2,
                confidence: 0.7,
                model_version: "m".to_string(),
                feature_ts: Utc::now(),
            },
            ExecutionEstimate {
                ticker: "KXTEST".to_string(),
                outcome_id: "yes".to_string(),
                side: Side::Buy,
                tif: TimeInForce::Gtc,
                candidate_limit_price: 0.50,
                fill_prob_30s: 0.5,
                fill_prob_5m: 0.5,
                expected_fill_price: 0.60,
                expected_slippage_bps: 0.0,
                expected_markout_bps_5m: -10.0,
                expected_markout_bps_30m: 0.0,
                model_version: "e".to_string(),
            },
        );
        assert!(!decision.should_trade);
    }
}
