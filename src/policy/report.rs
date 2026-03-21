use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::execution::types::{ExecutionError, TimeInForce};

#[derive(Debug, Clone)]
pub struct PolicyReportConfig {
    pub enabled: bool,
    pub cycle_dir: PathBuf,
}

impl PolicyReportConfig {
    pub fn from_env(default_cycle_dir: &str) -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_POLICY_REPORT")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            cycle_dir: PathBuf::from(
                std::env::var("BOT_POLICY_REPORT_CYCLE_DIR")
                    .unwrap_or_else(|_| default_cycle_dir.to_string()),
            ),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PolicyReportSummary {
    pub cycle_dir: String,
    pub cycle_files_seen: usize,
    pub completed_cycles: usize,
    pub cycles_with_policy: usize,
    pub policy_modes: std::collections::BTreeMap<String, usize>,
    pub candidates_total: usize,
    pub policy_decisions_total: usize,
    pub decisions_should_trade: usize,
    pub decisions_rejected: usize,
    pub avg_expected_realized_pnl: Option<f64>,
    pub avg_size_multiplier: Option<f64>,
    pub avg_expected_fill_prob: Option<f64>,
    pub avg_rank_delta: Option<f64>,
    pub improved_rank_count: usize,
    pub worsened_rank_count: usize,
    pub unchanged_rank_count: usize,
    pub chosen_tif_counts: std::collections::BTreeMap<String, usize>,
    pub top_promotions: Vec<PolicyPromotion>,
}

#[derive(Debug, Serialize)]
pub struct PolicyPromotion {
    pub ticker: String,
    pub outcome_id: String,
    pub legacy_rank: usize,
    pub policy_rank: usize,
    pub rank_delta: i64,
    pub expected_realized_pnl: f64,
    pub should_trade: bool,
    pub chosen_time_in_force: TimeInForce,
}

#[derive(Debug, Deserialize)]
struct CycleArtifactLite {
    #[allow(dead_code)]
    started_at: DateTime<Utc>,
    #[allow(dead_code)]
    finished_at: DateTime<Utc>,
    status: String,
    #[serde(default = "default_policy_mode")]
    policy_mode: String,
    #[serde(default)]
    candidates: Vec<ArtifactCandidateLite>,
    #[serde(default)]
    policy_decisions: Vec<ArtifactPolicyDecisionLite>,
}

fn default_policy_mode() -> String {
    "legacy".to_string()
}

#[derive(Debug, Deserialize)]
struct ArtifactCandidateLite {
    ticker: String,
    outcome_id: String,
    legacy_rank: Option<usize>,
    policy_rank: Option<usize>,
    rank_delta: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ArtifactPolicyDecisionLite {
    ticker: String,
    outcome_id: String,
    should_trade: bool,
    chosen_time_in_force: TimeInForce,
    size_multiplier: f64,
    expected_fill_prob: f64,
    expected_realized_pnl: f64,
}

pub fn run_policy_report(cfg: &PolicyReportConfig) -> Result<PolicyReportSummary, ExecutionError> {
    let cycle_files = list_cycle_files(&cfg.cycle_dir)?;
    let mut completed_cycles = 0usize;
    let mut cycles_with_policy = 0usize;
    let mut candidates_total = 0usize;
    let mut policy_decisions_total = 0usize;
    let mut decisions_should_trade = 0usize;
    let mut decisions_rejected = 0usize;
    let mut erpnl_total = 0.0;
    let mut erpnl_count = 0.0;
    let mut size_total = 0.0;
    let mut size_count = 0.0;
    let mut fill_total = 0.0;
    let mut fill_count = 0.0;
    let mut rank_total = 0.0;
    let mut rank_count = 0.0;
    let mut improved_rank_count = 0usize;
    let mut worsened_rank_count = 0usize;
    let mut unchanged_rank_count = 0usize;
    let mut chosen_tif_counts = std::collections::BTreeMap::new();
    let mut policy_modes = std::collections::BTreeMap::new();
    let mut promotions = Vec::new();

    for path in &cycle_files {
        let raw = match fs::read_to_string(path) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let artifact = match serde_json::from_str::<CycleArtifactLite>(&raw) {
            Ok(v) => v,
            Err(_) => continue,
        };
        *policy_modes.entry(artifact.policy_mode.clone()).or_insert(0) += 1;
        if artifact.status == "completed" {
            completed_cycles += 1;
        }
        if !artifact.policy_decisions.is_empty() {
            cycles_with_policy += 1;
        }
        candidates_total += artifact.candidates.len();
        policy_decisions_total += artifact.policy_decisions.len();

        for candidate in &artifact.candidates {
            if let Some(delta) = candidate.rank_delta {
                rank_total += delta as f64;
                rank_count += 1.0;
                if delta > 0 {
                    improved_rank_count += 1;
                } else if delta < 0 {
                    worsened_rank_count += 1;
                } else {
                    unchanged_rank_count += 1;
                }
            }
        }

        for decision in &artifact.policy_decisions {
            if decision.should_trade {
                decisions_should_trade += 1;
            } else {
                decisions_rejected += 1;
            }
            erpnl_total += decision.expected_realized_pnl;
            erpnl_count += 1.0;
            size_total += decision.size_multiplier;
            size_count += 1.0;
            fill_total += decision.expected_fill_prob;
            fill_count += 1.0;
            *chosen_tif_counts
                .entry(format!("{:?}", decision.chosen_time_in_force))
                .or_insert(0) += 1;
        }

        for candidate in &artifact.candidates {
            let Some(policy_rank) = candidate.policy_rank else {
                continue;
            };
            let Some(legacy_rank) = candidate.legacy_rank else {
                continue;
            };
            let Some(rank_delta) = candidate.rank_delta else {
                continue;
            };
            let Some(decision) = artifact
                .policy_decisions
                .iter()
                .find(|d| d.ticker == candidate.ticker && d.outcome_id == candidate.outcome_id)
            else {
                continue;
            };
            promotions.push(PolicyPromotion {
                ticker: candidate.ticker.clone(),
                outcome_id: candidate.outcome_id.clone(),
                legacy_rank,
                policy_rank,
                rank_delta,
                expected_realized_pnl: decision.expected_realized_pnl,
                should_trade: decision.should_trade,
                chosen_time_in_force: decision.chosen_time_in_force,
            });
        }
    }

    promotions.sort_by(|a, b| {
        b.rank_delta
            .cmp(&a.rank_delta)
            .then_with(|| b.expected_realized_pnl.total_cmp(&a.expected_realized_pnl))
    });
    promotions.truncate(10);

    Ok(PolicyReportSummary {
        cycle_dir: cfg.cycle_dir.display().to_string(),
        cycle_files_seen: cycle_files.len(),
        completed_cycles,
        cycles_with_policy,
        policy_modes,
        candidates_total,
        policy_decisions_total,
        decisions_should_trade,
        decisions_rejected,
        avg_expected_realized_pnl: avg(erpnl_total, erpnl_count),
        avg_size_multiplier: avg(size_total, size_count),
        avg_expected_fill_prob: avg(fill_total, fill_count),
        avg_rank_delta: avg(rank_total, rank_count),
        improved_rank_count,
        worsened_rank_count,
        unchanged_rank_count,
        chosen_tif_counts,
        top_promotions: promotions,
    })
}

fn list_cycle_files(dir: &Path) -> Result<Vec<PathBuf>, ExecutionError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let path = entry
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?
            .path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn avg(total: f64, count: f64) -> Option<f64> {
    (count > 0.0).then_some(total / count)
}
