use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::datasets::builder::ExecutionTrainingRow;
use crate::execution::types::{ExecutionError, Side, TimeInForce};
use crate::features::execution::ExecutionFeatureRow;

pub const EXECUTION_MODEL_SCHEMA_VERSION: &str = "v1";
pub const EXECUTION_MODEL_KIND: &str = "empirical_execution_baseline";

#[derive(Debug, Clone)]
pub struct ExecutionTrainingConfig {
    pub enabled: bool,
    pub dataset_path: PathBuf,
    pub output_root: PathBuf,
    pub min_bucket_samples: usize,
    pub include_source_classes: Vec<String>,
}

impl ExecutionTrainingConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_EXECUTION_TRAIN")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            dataset_path: PathBuf::from(std::env::var("BOT_EXECUTION_DATASET_PATH").unwrap_or_else(
                |_| "var/features/execution/execution_training.jsonl".to_string(),
            )),
            output_root: PathBuf::from(std::env::var("BOT_MODEL_EXECUTION_DIR").unwrap_or_else(
                |_| "var/models/execution".to_string(),
            )),
            min_bucket_samples: std::env::var("BOT_MODEL_EXECUTION_MIN_BUCKET_SAMPLES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1),
            include_source_classes: std::env::var("BOT_EXECUTION_TRAIN_SOURCES")
                .unwrap_or_else(|_| "organic_paper,live_real".to_string())
                .split(',')
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionRuntimeConfig {
    pub model_path: Option<PathBuf>,
    pub shadow_enabled: bool,
    pub shadow_root: PathBuf,
    pub min_bucket_samples: usize,
}

impl ExecutionRuntimeConfig {
    pub fn from_env() -> Self {
        Self {
            model_path: std::env::var("BOT_MODEL_EXECUTION_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|path| !path.as_os_str().is_empty()),
            shadow_enabled: matches!(
                std::env::var("BOT_EXECUTION_SHADOW_ENABLED")
                    .unwrap_or_else(|_| "true".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            shadow_root: PathBuf::from(
                std::env::var("BOT_SHADOW_DIR").unwrap_or_else(|_| "var/shadow".to_string()),
            ),
            min_bucket_samples: std::env::var("BOT_MODEL_EXECUTION_MIN_BUCKET_SAMPLES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEstimate {
    pub ticker: String,
    pub outcome_id: String,
    pub side: Side,
    pub tif: TimeInForce,
    pub candidate_limit_price: f64,
    pub fill_prob_30s: f64,
    pub fill_prob_5m: f64,
    pub expected_fill_price: f64,
    pub expected_slippage_bps: f64,
    pub expected_markout_bps_5m: f64,
    pub expected_markout_bps_30m: f64,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionShadowRecord {
    pub cycle_id: String,
    pub recorded_at: DateTime<Utc>,
    pub ticker: String,
    pub outcome_id: String,
    pub side: Side,
    pub tif: TimeInForce,
    pub candidate_limit_price: f64,
    pub raw_edge_pct: f64,
    pub confidence: f64,
    pub fill_prob_30s: f64,
    pub fill_prob_5m: f64,
    pub expected_fill_price: f64,
    pub expected_slippage_bps: f64,
    pub expected_markout_bps_5m: f64,
    pub expected_markout_bps_30m: f64,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionModelArtifact {
    pub schema_version: String,
    pub model_kind: String,
    pub model_version: String,
    pub trained_at: DateTime<Utc>,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
    pub feature_schema_version: String,
    pub metrics: ExecutionModelMetrics,
    pub global: ExecutionBucket,
    pub by_vertical: HashMap<String, ExecutionBucket>,
    pub by_vertical_tif: HashMap<String, ExecutionBucket>,
    pub by_vertical_liquidity: HashMap<String, ExecutionBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionModelMetrics {
    pub validation_brier_fill_30s: Option<f64>,
    pub validation_brier_fill_5m: Option<f64>,
    pub validation_mae_fill_price: Option<f64>,
    pub validation_mae_markout_5m: Option<f64>,
    pub test_brier_fill_30s: Option<f64>,
    pub test_brier_fill_5m: Option<f64>,
    pub test_mae_fill_price: Option<f64>,
    pub test_mae_markout_5m: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RunningMean {
    pub total: f64,
    pub count: f64,
}

impl RunningMean {
    fn update(&mut self, value: f64) {
        self.total += value;
        self.count += 1.0;
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0.0).then_some(self.total / self.count)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ProbabilityBucket {
    pub positives: f64,
    pub total: f64,
}

impl ProbabilityBucket {
    fn update(&mut self, outcome: bool) {
        self.total += 1.0;
        if outcome {
            self.positives += 1.0;
        }
    }

    fn posterior_mean(&self, prior_mean: f64, prior_strength: f64) -> f64 {
        ((self.positives + (prior_mean * prior_strength)) / (self.total + prior_strength))
            .clamp(0.001, 0.999)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ExecutionBucket {
    pub total: f64,
    pub fill_30s: ProbabilityBucket,
    pub fill_5m: ProbabilityBucket,
    pub fill_price: RunningMean,
    pub markout_5m_bps: RunningMean,
    pub markout_30m_bps: RunningMean,
}

impl ExecutionBucket {
    fn update(&mut self, row: &ExecutionTrainingRow) {
        self.total += 1.0;
        self.fill_30s.update(row.label_filled_within_30s);
        self.fill_5m.update(row.label_filled_within_5m);
        if let Some(fill_price) = row.label_terminal_avg_fill_price {
            self.fill_price.update(fill_price);
        }
        if let Some(markout) = row.label_markout_bps_5m {
            self.markout_5m_bps.update(markout);
        }
        if let Some(markout) = row.label_markout_bps_30m {
            self.markout_30m_bps.update(markout);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionModel {
    artifact: ExecutionModelArtifact,
    min_bucket_samples: usize,
}

impl ExecutionModel {
    pub fn from_artifact(artifact: ExecutionModelArtifact, min_bucket_samples: usize) -> Self {
        Self {
            artifact,
            min_bucket_samples: min_bucket_samples.max(1),
        }
    }

    pub fn load_from_path(path: &Path, min_bucket_samples: usize) -> Result<Self, ExecutionError> {
        let raw = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let artifact = serde_json::from_str::<ExecutionModelArtifact>(&raw)
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        Ok(Self::from_artifact(artifact, min_bucket_samples))
    }

    pub fn predict(&self, feature: &ExecutionFeatureRow) -> ExecutionEstimate {
        let global_fill_30s = self.artifact.global.fill_30s.posterior_mean(0.25, 4.0);
        let global_fill_5m = self.artifact.global.fill_5m.posterior_mean(0.35, 4.0);
        let global_fill_price = self
            .artifact
            .global
            .fill_price
            .mean()
            .unwrap_or(feature.candidate_limit_price);
        let global_markout_5m = self.artifact.global.markout_5m_bps.mean().unwrap_or(0.0);
        let global_markout_30m = self.artifact.global.markout_30m_bps.mean().unwrap_or(0.0);

        let mut fill_30s = global_fill_30s;
        let mut fill_5m = global_fill_5m;
        let mut expected_fill_price = global_fill_price;
        let mut expected_markout_5m = global_markout_5m;
        let mut expected_markout_30m = global_markout_30m;

        if let Some(bucket) = self.lookup(&self.artifact.by_vertical, &feature.vertical) {
            blend_bucket(
                bucket,
                &mut fill_30s,
                &mut fill_5m,
                &mut expected_fill_price,
                &mut expected_markout_5m,
                &mut expected_markout_30m,
                global_fill_30s,
                global_fill_5m,
                global_fill_price,
                global_markout_5m,
                global_markout_30m,
                0.40,
            );
        }
        let tif_key = format!("{}|{:?}", feature.vertical, feature.tif);
        if let Some(bucket) = self.lookup(&self.artifact.by_vertical_tif, &tif_key) {
            blend_bucket(
                bucket,
                &mut fill_30s,
                &mut fill_5m,
                &mut expected_fill_price,
                &mut expected_markout_5m,
                &mut expected_markout_30m,
                global_fill_30s,
                global_fill_5m,
                global_fill_price,
                global_markout_5m,
                global_markout_30m,
                0.35,
            );
        }
        let liq_key = format!("{}|{}", feature.vertical, liquidity_bucket(feature.volume));
        if let Some(bucket) = self.lookup(&self.artifact.by_vertical_liquidity, &liq_key) {
            blend_bucket(
                bucket,
                &mut fill_30s,
                &mut fill_5m,
                &mut expected_fill_price,
                &mut expected_markout_5m,
                &mut expected_markout_30m,
                global_fill_30s,
                global_fill_5m,
                global_fill_price,
                global_markout_5m,
                global_markout_30m,
                0.25,
            );
        }

        let expected_slippage_bps = (((expected_fill_price - feature.candidate_limit_price)
            / feature.candidate_limit_price.max(0.0001))
            * 10_000.0)
            .max(-10_000.0)
            .min(10_000.0);

        ExecutionEstimate {
            ticker: feature.ticker.clone(),
            outcome_id: feature.outcome_id.clone(),
            side: feature.side,
            tif: feature.tif,
            candidate_limit_price: feature.candidate_limit_price,
            fill_prob_30s: fill_30s.clamp(0.001, 0.999),
            fill_prob_5m: fill_5m.clamp(0.001, 0.999),
            expected_fill_price: expected_fill_price.clamp(0.001, 0.999),
            expected_slippage_bps,
            expected_markout_bps_5m: expected_markout_5m,
            expected_markout_bps_30m: expected_markout_30m,
            model_version: self.artifact.model_version.clone(),
        }
    }

    fn lookup<'a>(
        &self,
        buckets: &'a HashMap<String, ExecutionBucket>,
        key: &str,
    ) -> Option<&'a ExecutionBucket> {
        let bucket = buckets.get(key)?;
        (bucket.total >= self.min_bucket_samples as f64).then_some(bucket)
    }
}

pub async fn run_execution_training(cfg: &ExecutionTrainingConfig) -> Result<(), ExecutionError> {
    let rows = load_execution_training_rows(&cfg.dataset_path)?
        .into_iter()
        .filter(|row| {
            cfg.include_source_classes
                .iter()
                .any(|source| source == &row.execution_source_class)
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(ExecutionError::Exchange(format!(
            "no execution training rows found at {}",
            cfg.dataset_path.display()
        )));
    }
    let train_rows: Vec<_> = rows.iter().filter(|row| row.split == "train").cloned().collect();
    let validation_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.split == "validation")
        .cloned()
        .collect();
    let test_rows: Vec<_> = rows.iter().filter(|row| row.split == "test").cloned().collect();

    let mut artifact = train_artifact(&train_rows, validation_rows.len(), test_rows.len());
    let model = ExecutionModel::from_artifact(artifact.clone(), cfg.min_bucket_samples);
    artifact.metrics = ExecutionModelMetrics {
        validation_brier_fill_30s: eval_brier(&model, &validation_rows, |e| e.fill_prob_30s, |r| {
            if r.label_filled_within_30s { 1.0 } else { 0.0 }
        }),
        validation_brier_fill_5m: eval_brier(&model, &validation_rows, |e| e.fill_prob_5m, |r| {
            if r.label_filled_within_5m { 1.0 } else { 0.0 }
        }),
        validation_mae_fill_price: eval_mae_opt(&model, &validation_rows, |e| e.expected_fill_price, |r| {
            r.label_terminal_avg_fill_price
        }),
        validation_mae_markout_5m: eval_mae_opt(&model, &validation_rows, |e| e.expected_markout_bps_5m, |r| {
            r.label_markout_bps_5m
        }),
        test_brier_fill_30s: eval_brier(&model, &test_rows, |e| e.fill_prob_30s, |r| {
            if r.label_filled_within_30s { 1.0 } else { 0.0 }
        }),
        test_brier_fill_5m: eval_brier(&model, &test_rows, |e| e.fill_prob_5m, |r| {
            if r.label_filled_within_5m { 1.0 } else { 0.0 }
        }),
        test_mae_fill_price: eval_mae_opt(&model, &test_rows, |e| e.expected_fill_price, |r| {
            r.label_terminal_avg_fill_price
        }),
        test_mae_markout_5m: eval_mae_opt(&model, &test_rows, |e| e.expected_markout_bps_5m, |r| {
            r.label_markout_bps_5m
        }),
    };
    write_artifact(&cfg.output_root, &artifact)?;
    println!(
        "execution training complete: version={} train_rows={} validation_rows={} test_rows={} sources={}",
        artifact.model_version,
        artifact.train_rows,
        artifact.validation_rows,
        artifact.test_rows,
        cfg.include_source_classes.join(",")
    );
    println!(
        "execution metrics: val_brier_fill30={:?} val_brier_fill5m={:?} val_mae_fill={:?} val_mae_markout5m={:?}",
        artifact.metrics.validation_brier_fill_30s,
        artifact.metrics.validation_brier_fill_5m,
        artifact.metrics.validation_mae_fill_price,
        artifact.metrics.validation_mae_markout_5m
    );
    Ok(())
}

pub fn load_runtime_model(cfg: &ExecutionRuntimeConfig) -> Result<Option<ExecutionModel>, ExecutionError> {
    let Some(path) = cfg.model_path.as_ref() else {
        return Ok(None);
    };
    if !path.exists() {
        return Err(ExecutionError::Exchange(format!(
            "execution model path does not exist: {}",
            path.display()
        )));
    }
    Ok(Some(ExecutionModel::load_from_path(path, cfg.min_bucket_samples)?))
}

pub fn record_shadow_outputs(
    cfg: &ExecutionRuntimeConfig,
    cycle_id: &str,
    rows: &[(&ExecutionFeatureRow, ExecutionEstimate)],
) -> Result<(), ExecutionError> {
    if !cfg.shadow_enabled || rows.is_empty() {
        return Ok(());
    }
    let day = Utc::now().format("%Y-%m-%d").to_string();
    let path = cfg
        .shadow_root
        .join("execution")
        .join(day)
        .join("execution_shadow.jsonl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for (feature, output) in rows {
        let record = ExecutionShadowRecord {
            cycle_id: cycle_id.to_string(),
            recorded_at: Utc::now(),
            ticker: feature.ticker.clone(),
            outcome_id: feature.outcome_id.clone(),
            side: feature.side,
            tif: feature.tif,
            candidate_limit_price: feature.candidate_limit_price,
            raw_edge_pct: feature.raw_edge_pct,
            confidence: feature.confidence,
            fill_prob_30s: output.fill_prob_30s,
            fill_prob_5m: output.fill_prob_5m,
            expected_fill_price: output.expected_fill_price,
            expected_slippage_bps: output.expected_slippage_bps,
            expected_markout_bps_5m: output.expected_markout_bps_5m,
            expected_markout_bps_30m: output.expected_markout_bps_30m,
            model_version: output.model_version.clone(),
        };
        let line = serde_json::to_string(&record)
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    Ok(())
}

fn train_artifact(
    train_rows: &[ExecutionTrainingRow],
    validation_rows: usize,
    test_rows: usize,
) -> ExecutionModelArtifact {
    let mut global = ExecutionBucket::default();
    let mut by_vertical = HashMap::new();
    let mut by_vertical_tif = HashMap::new();
    let mut by_vertical_liquidity = HashMap::new();
    for row in train_rows {
        global.update(row);
        by_vertical
            .entry(row.feature.vertical.clone())
            .or_insert_with(ExecutionBucket::default)
            .update(row);
        by_vertical_tif
            .entry(format!("{}|{:?}", row.feature.vertical, row.feature.tif))
            .or_insert_with(ExecutionBucket::default)
            .update(row);
        by_vertical_liquidity
            .entry(format!("{}|{}", row.feature.vertical, liquidity_bucket(row.feature.volume)))
            .or_insert_with(ExecutionBucket::default)
            .update(row);
    }
    ExecutionModelArtifact {
        schema_version: EXECUTION_MODEL_SCHEMA_VERSION.to_string(),
        model_kind: EXECUTION_MODEL_KIND.to_string(),
        model_version: format!("execution-{}", Utc::now().format("%Y%m%dT%H%M%SZ")),
        trained_at: Utc::now(),
        train_rows: train_rows.len(),
        validation_rows,
        test_rows,
        feature_schema_version: train_rows
            .first()
            .map(|row| row.feature.schema_version.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        metrics: ExecutionModelMetrics::default(),
        global,
        by_vertical,
        by_vertical_tif,
        by_vertical_liquidity,
    }
}

fn load_execution_training_rows(path: &Path) -> Result<Vec<ExecutionTrainingRow>, ExecutionError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let mut rows = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str::<ExecutionTrainingRow>(line)
                .map_err(|e| ExecutionError::Exchange(e.to_string()))?,
        );
    }
    Ok(rows)
}

fn write_artifact(root: &Path, artifact: &ExecutionModelArtifact) -> Result<(), ExecutionError> {
    let version_dir = root.join(&artifact.model_version);
    fs::create_dir_all(&version_dir).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let json = serde_json::to_string_pretty(artifact)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    fs::write(version_dir.join("artifact.json"), json.as_bytes())
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    fs::write(root.join("latest.json"), json.as_bytes())
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let mut manifest = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("manifest.jsonl"))
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    manifest
        .write_all(format!("{json}\n").as_bytes())
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    Ok(())
}

fn blend_bucket(
    bucket: &ExecutionBucket,
    fill_30s: &mut f64,
    fill_5m: &mut f64,
    expected_fill_price: &mut f64,
    expected_markout_5m: &mut f64,
    expected_markout_30m: &mut f64,
    global_fill_30s: f64,
    global_fill_5m: f64,
    global_fill_price: f64,
    global_markout_5m: f64,
    global_markout_30m: f64,
    weight: f64,
) {
    *fill_30s = ((*fill_30s * (1.0 - weight))
        + (bucket.fill_30s.posterior_mean(global_fill_30s, 8.0) * weight))
        .clamp(0.001, 0.999);
    *fill_5m = ((*fill_5m * (1.0 - weight))
        + (bucket.fill_5m.posterior_mean(global_fill_5m, 8.0) * weight))
        .clamp(0.001, 0.999);
    *expected_fill_price = (*expected_fill_price * (1.0 - weight))
        + (bucket.fill_price.mean().unwrap_or(global_fill_price) * weight);
    *expected_markout_5m = (*expected_markout_5m * (1.0 - weight))
        + (bucket.markout_5m_bps.mean().unwrap_or(global_markout_5m) * weight);
    *expected_markout_30m = (*expected_markout_30m * (1.0 - weight))
        + (bucket.markout_30m_bps.mean().unwrap_or(global_markout_30m) * weight);
}

fn liquidity_bucket(volume: f64) -> &'static str {
    if volume < 1_000.0 {
        "low"
    } else if volume < 10_000.0 {
        "medium"
    } else {
        "high"
    }
}

fn eval_brier<FP, FL>(
    model: &ExecutionModel,
    rows: &[ExecutionTrainingRow],
    pred: FP,
    label: FL,
) -> Option<f64>
where
    FP: Fn(&ExecutionEstimate) -> f64,
    FL: Fn(&ExecutionTrainingRow) -> f64,
{
    let mut total = 0.0;
    let mut count = 0.0;
    for row in rows {
        let p = pred(&model.predict(&row.feature));
        let y = label(row);
        total += (p - y).powi(2);
        count += 1.0;
    }
    (count > 0.0).then_some(total / count)
}

fn eval_mae_opt<FP, FL>(
    model: &ExecutionModel,
    rows: &[ExecutionTrainingRow],
    pred: FP,
    label: FL,
) -> Option<f64>
where
    FP: Fn(&ExecutionEstimate) -> f64,
    FL: Fn(&ExecutionTrainingRow) -> Option<f64>,
{
    let mut total = 0.0;
    let mut count = 0.0;
    for row in rows {
        let Some(y) = label(row) else { continue; };
        total += (pred(&model.predict(&row.feature)) - y).abs();
        count += 1.0;
    }
    (count > 0.0).then_some(total / count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(split: &str, vertical: &str, tif: TimeInForce, fill_5m: bool) -> ExecutionTrainingRow {
        ExecutionTrainingRow {
            schema_version: "v1".to_string(),
            split: split.to_string(),
            client_order_id: "1".to_string(),
            execution_source_class: "organic_paper".to_string(),
            is_bootstrap_synthetic: false,
            is_organic_paper: true,
            is_live_real: false,
            terminal_status: Some(if fill_5m { "Filled" } else { "Canceled" }.to_string()),
            label_filled_within_30s: fill_5m,
            label_filled_within_5m: fill_5m,
            label_terminal_filled_qty: if fill_5m { 10.0 } else { 0.0 },
            label_terminal_avg_fill_price: if fill_5m { Some(0.52) } else { None },
            label_canceled: !fill_5m,
            label_rejected: false,
            label_markout_bps_5m: if fill_5m { Some(42.0) } else { None },
            label_markout_bps_30m: if fill_5m { Some(30.0) } else { None },
            label_realized_net_pnl: None,
            feature: ExecutionFeatureRow {
                schema_version: "v1".to_string(),
                feature_ts: Utc::now(),
                ticker: "KXTEST".to_string(),
                outcome_id: "yes".to_string(),
                side: Side::Buy,
                tif,
                title: "x".to_string(),
                vertical: vertical.to_string(),
                candidate_limit_price: 0.50,
                candidate_observed_price: 0.50,
                candidate_fair_price: 0.56,
                raw_edge_pct: 0.06,
                confidence: 0.8,
                yes_bid_cents: Some(48.0),
                yes_ask_cents: Some(52.0),
                spread_cents: Some(4.0),
                mid_prob_yes: Some(0.50),
                volume: 5000.0,
                time_to_close_secs: Some(3600),
                price_vs_best_bid_cents: Some(2.0),
                price_vs_best_ask_cents: Some(-2.0),
                aggressiveness_bps: Some(-5000.0),
                open_order_count_same_ticker: 0,
                recent_fill_count_same_ticker: 0,
                recent_cancel_count_same_ticker: 0,
                same_event_exposure_notional: 0.0,
            },
        }
    }

    #[test]
    fn execution_model_produces_bounded_fill_probs() {
        let train_rows = vec![
            sample_row("train", "weather", TimeInForce::Gtc, true),
            sample_row("train", "weather", TimeInForce::Gtc, true),
            sample_row("train", "weather", TimeInForce::Gtc, false),
        ];
        let artifact = train_artifact(&train_rows, 0, 0);
        let model = ExecutionModel::from_artifact(artifact, 1);
        let out = model.predict(&sample_row("test", "weather", TimeInForce::Gtc, true).feature);
        assert!(out.fill_prob_30s > 0.0 && out.fill_prob_30s < 1.0);
        assert!(out.fill_prob_5m > 0.0 && out.fill_prob_5m < 1.0);
    }
}
