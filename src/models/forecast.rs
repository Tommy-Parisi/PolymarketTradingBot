use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::datasets::builder::ForecastTrainingRow;
use crate::execution::types::ExecutionError;
use crate::features::forecast::ForecastFeatureRow;

pub const FORECAST_MODEL_SCHEMA_VERSION: &str = "v1";
pub const FORECAST_MODEL_KIND: &str = "empirical_shrinkage_baseline";

#[derive(Debug, Clone)]
pub struct ForecastTrainingConfig {
    pub enabled: bool,
    pub dataset_path: PathBuf,
    pub output_root: PathBuf,
    pub min_bucket_samples: usize,
}

impl ForecastTrainingConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_FORECAST_TRAIN")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            dataset_path: PathBuf::from(std::env::var("BOT_FORECAST_DATASET_PATH").unwrap_or_else(
                |_| "var/features/forecast/forecast_training.jsonl".to_string(),
            )),
            output_root: PathBuf::from(std::env::var("BOT_MODEL_FORECAST_DIR").unwrap_or_else(
                |_| "var/models/forecast".to_string(),
            )),
            min_bucket_samples: std::env::var("BOT_MODEL_FORECAST_MIN_BUCKET_SAMPLES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForecastRuntimeConfig {
    pub model_path: Option<PathBuf>,
    pub shadow_enabled: bool,
    pub shadow_root: PathBuf,
    pub min_bucket_samples: usize,
}

impl ForecastRuntimeConfig {
    pub fn from_env() -> Self {
        let model_path = std::env::var("BOT_MODEL_FORECAST_PATH")
            .ok()
            .map(PathBuf::from)
            .filter(|path| !path.as_os_str().is_empty());
        let shadow_enabled = matches!(
            std::env::var("BOT_FORECAST_SHADOW_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes"
        );
        let shadow_root = PathBuf::from(
            std::env::var("BOT_SHADOW_DIR").unwrap_or_else(|_| "var/shadow".to_string()),
        );
        let min_bucket_samples = std::env::var("BOT_MODEL_FORECAST_MIN_BUCKET_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5)
            .max(1);
        Self {
            model_path,
            shadow_enabled,
            shadow_root,
            min_bucket_samples,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastOutput {
    pub ticker: String,
    pub fair_prob_yes: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub model_version: String,
    pub feature_ts: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForecastShadowRecord {
    pub cycle_id: String,
    pub recorded_at: DateTime<Utc>,
    pub ticker: String,
    pub title: String,
    pub vertical: String,
    pub market_mid_prob_yes: Option<f64>,
    pub fair_prob_yes: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastModelArtifact {
    pub schema_version: String,
    pub model_kind: String,
    pub model_version: String,
    pub trained_at: DateTime<Utc>,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
    pub feature_schema_version: String,
    pub metrics: ForecastModelMetrics,
    pub global: RateBucket,
    pub vertical: HashMap<String, RateBucket>,
    pub vertical_direction: HashMap<String, RateBucket>,
    pub vertical_entity: HashMap<String, RateBucket>,
    pub vertical_threshold: HashMap<String, RateBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ForecastModelMetrics {
    pub validation_log_loss: Option<f64>,
    pub validation_brier: Option<f64>,
    pub validation_market_mid_log_loss: Option<f64>,
    pub validation_market_mid_brier: Option<f64>,
    pub test_log_loss: Option<f64>,
    pub test_brier: Option<f64>,
    pub test_market_mid_log_loss: Option<f64>,
    pub test_market_mid_brier: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RateBucket {
    pub positives: f64,
    pub total: f64,
}

impl RateBucket {
    fn update(&mut self, outcome_yes: bool) {
        self.total += 1.0;
        if outcome_yes {
            self.positives += 1.0;
        }
    }

    fn posterior_mean(&self, prior_mean: f64, prior_strength: f64) -> f64 {
        ((self.positives + (prior_mean * prior_strength)) / (self.total + prior_strength))
            .clamp(0.001, 0.999)
    }

    fn confidence(&self, prior_strength: f64) -> f64 {
        (self.total / (self.total + prior_strength)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone)]
pub struct ForecastModel {
    artifact: ForecastModelArtifact,
    min_bucket_samples: usize,
}

impl ForecastModel {
    pub fn from_artifact(artifact: ForecastModelArtifact, min_bucket_samples: usize) -> Self {
        Self {
            artifact,
            min_bucket_samples: min_bucket_samples.max(1),
        }
    }

    pub fn load_from_path(path: &Path, min_bucket_samples: usize) -> Result<Self, ExecutionError> {
        let raw = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let artifact = serde_json::from_str::<ForecastModelArtifact>(&raw)
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        Ok(Self::from_artifact(artifact, min_bucket_samples))
    }

    pub fn artifact(&self) -> &ForecastModelArtifact {
        &self.artifact
    }

    pub fn predict(&self, feature: &ForecastFeatureRow) -> ForecastOutput {
        let global_mean = self.artifact.global.posterior_mean(0.5, 2.0);
        let global_conf = self.artifact.global.confidence(2.0);

        let mut weighted_prob = global_mean * global_conf.max(0.10);
        let mut weight_sum = global_conf.max(0.10);

        if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical, &feature.vertical) {
            let mean = bucket.posterior_mean(global_mean, 8.0);
            let weight = bucket.confidence(8.0) * 1.2;
            weighted_prob += mean * weight;
            weight_sum += weight;
        }
        if let Some(direction) = feature.threshold_direction.as_deref() {
            let key = format!("{}|{}", feature.vertical, direction);
            if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical_direction, &key) {
                let mean = bucket.posterior_mean(global_mean, 10.0);
                let weight = bucket.confidence(10.0) * 1.0;
                weighted_prob += mean * weight;
                weight_sum += weight;
            }
        }
        if let Some(entity) = feature.entity_primary.as_deref() {
            let key = format!("{}|{}", feature.vertical, normalize_key(entity));
            if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical_entity, &key) {
                let mean = bucket.posterior_mean(global_mean, 12.0);
                let weight = bucket.confidence(12.0) * 0.9;
                weighted_prob += mean * weight;
                weight_sum += weight;
            }
        }
        if let Some(key) = threshold_bucket_key(feature) {
            if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical_threshold, &key) {
                let mean = bucket.posterior_mean(global_mean, 10.0);
                let weight = bucket.confidence(10.0) * 0.8;
                weighted_prob += mean * weight;
                weight_sum += weight;
            }
        }

        let fair_prob_yes = (weighted_prob / weight_sum.max(0.0001)).clamp(0.001, 0.999);
        let effective_n = self.effective_support(feature);
        let uncertainty = (1.0 / (effective_n + 1.0).sqrt()).clamp(0.01, 0.50);
        let confidence = (1.0 - uncertainty).clamp(0.0, 1.0);

        ForecastOutput {
            ticker: feature.ticker.clone(),
            fair_prob_yes,
            uncertainty,
            confidence,
            model_version: self.artifact.model_version.clone(),
            feature_ts: feature.feature_ts,
        }
    }

    fn lookup_bucket<'a>(
        &self,
        buckets: &'a HashMap<String, RateBucket>,
        key: &str,
    ) -> Option<&'a RateBucket> {
        let bucket = buckets.get(key)?;
        if bucket.total >= self.min_bucket_samples as f64 {
            Some(bucket)
        } else {
            None
        }
    }

    fn effective_support(&self, feature: &ForecastFeatureRow) -> f64 {
        let mut total = self.artifact.global.total;
        if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical, &feature.vertical) {
            total += bucket.total * 0.5;
        }
        if let Some(direction) = feature.threshold_direction.as_deref() {
            let key = format!("{}|{}", feature.vertical, direction);
            if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical_direction, &key) {
                total += bucket.total * 0.35;
            }
        }
        if let Some(entity) = feature.entity_primary.as_deref() {
            let key = format!("{}|{}", feature.vertical, normalize_key(entity));
            if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical_entity, &key) {
                total += bucket.total * 0.25;
            }
        }
        if let Some(key) = threshold_bucket_key(feature) {
            if let Some(bucket) = self.lookup_bucket(&self.artifact.vertical_threshold, &key) {
                total += bucket.total * 0.25;
            }
        }
        total
    }
}

pub async fn run_forecast_training(cfg: &ForecastTrainingConfig) -> Result<(), ExecutionError> {
    let rows = load_forecast_training_rows(&cfg.dataset_path)?;
    if rows.is_empty() {
        return Err(ExecutionError::Exchange(format!(
            "no forecast training rows found at {}",
            cfg.dataset_path.display()
        )));
    }

    let train_rows: Vec<_> = rows.iter().filter(|row| row.split == "train").cloned().collect();
    if train_rows.is_empty() {
        return Err(ExecutionError::Exchange(
            "forecast dataset does not contain any train rows".to_string(),
        ));
    }
    let validation_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.split == "validation")
        .cloned()
        .collect();
    let test_rows: Vec<_> = rows.iter().filter(|row| row.split == "test").cloned().collect();

    let artifact = train_artifact(
        &train_rows,
        validation_rows.len(),
        test_rows.len(),
    );
    let model = ForecastModel::from_artifact(artifact, cfg.min_bucket_samples);
    let metrics = ForecastModelMetrics {
        validation_log_loss: evaluate_log_loss(&model, &validation_rows),
        validation_brier: evaluate_brier(&model, &validation_rows),
        validation_market_mid_log_loss: evaluate_market_mid_log_loss(&validation_rows),
        validation_market_mid_brier: evaluate_market_mid_brier(&validation_rows),
        test_log_loss: evaluate_log_loss(&model, &test_rows),
        test_brier: evaluate_brier(&model, &test_rows),
        test_market_mid_log_loss: evaluate_market_mid_log_loss(&test_rows),
        test_market_mid_brier: evaluate_market_mid_brier(&test_rows),
    };

    let mut finalized = model.artifact().clone();
    finalized.metrics = metrics;

    write_artifact(&cfg.output_root, &finalized)?;

    println!(
        "forecast training complete: version={} train_rows={} validation_rows={} test_rows={}",
        finalized.model_version, finalized.train_rows, finalized.validation_rows, finalized.test_rows
    );
    println!(
        "forecast metrics: val_log_loss={:?} val_brier={:?} val_mid_log_loss={:?} val_mid_brier={:?}",
        finalized.metrics.validation_log_loss,
        finalized.metrics.validation_brier,
        finalized.metrics.validation_market_mid_log_loss,
        finalized.metrics.validation_market_mid_brier
    );
    println!(
        "forecast metrics: test_log_loss={:?} test_brier={:?} test_mid_log_loss={:?} test_mid_brier={:?}",
        finalized.metrics.test_log_loss,
        finalized.metrics.test_brier,
        finalized.metrics.test_market_mid_log_loss,
        finalized.metrics.test_market_mid_brier
    );

    Ok(())
}

pub fn load_runtime_model(cfg: &ForecastRuntimeConfig) -> Result<Option<ForecastModel>, ExecutionError> {
    let Some(path) = cfg.model_path.as_ref() else {
        return Ok(None);
    };
    if !path.exists() {
        return Err(ExecutionError::Exchange(format!(
            "forecast model path does not exist: {}",
            path.display()
        )));
    }
    let model = ForecastModel::load_from_path(path, cfg.min_bucket_samples)?;
    Ok(Some(model))
}

pub fn record_shadow_outputs(
    cfg: &ForecastRuntimeConfig,
    cycle_id: &str,
    rows: &[(&ForecastFeatureRow, ForecastOutput)],
) -> Result<(), ExecutionError> {
    if !cfg.shadow_enabled || rows.is_empty() {
        return Ok(());
    }
    let day = Utc::now().format("%Y-%m-%d").to_string();
    let path = cfg
        .shadow_root
        .join("forecast")
        .join(day)
        .join("forecast_shadow.jsonl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    for (feature, output) in rows {
        let record = ForecastShadowRecord {
            cycle_id: cycle_id.to_string(),
            recorded_at: Utc::now(),
            ticker: feature.ticker.clone(),
            title: feature.title.clone(),
            vertical: feature.vertical.clone(),
            market_mid_prob_yes: feature.mid_prob_yes,
            fair_prob_yes: output.fair_prob_yes,
            uncertainty: output.uncertainty,
            confidence: output.confidence,
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
    train_rows: &[ForecastTrainingRow],
    validation_rows: usize,
    test_rows: usize,
) -> ForecastModelArtifact {
    let model_version = format!("forecast-{}", Utc::now().format("%Y%m%dT%H%M%SZ"));
    let mut global = RateBucket::default();
    let mut vertical = HashMap::new();
    let mut vertical_direction = HashMap::new();
    let mut vertical_entity = HashMap::new();
    let mut vertical_threshold = HashMap::new();

    for row in train_rows {
        let Some(outcome_yes) = row.label_outcome_yes else {
            continue;
        };
        global.update(outcome_yes);
        vertical
            .entry(row.feature.vertical.clone())
            .or_insert_with(RateBucket::default)
            .update(outcome_yes);
        if let Some(direction) = row.feature.threshold_direction.as_deref() {
            vertical_direction
                .entry(format!("{}|{}", row.feature.vertical, direction))
                .or_insert_with(RateBucket::default)
                .update(outcome_yes);
        }
        if let Some(entity) = row.feature.entity_primary.as_deref() {
            vertical_entity
                .entry(format!("{}|{}", row.feature.vertical, normalize_key(entity)))
                .or_insert_with(RateBucket::default)
                .update(outcome_yes);
        }
        if let Some(key) = threshold_bucket_key(&row.feature) {
            vertical_threshold
                .entry(key)
                .or_insert_with(RateBucket::default)
                .update(outcome_yes);
        }
    }

    ForecastModelArtifact {
        schema_version: FORECAST_MODEL_SCHEMA_VERSION.to_string(),
        model_kind: FORECAST_MODEL_KIND.to_string(),
        model_version,
        trained_at: Utc::now(),
        train_rows: train_rows.len(),
        validation_rows,
        test_rows,
        feature_schema_version: train_rows
            .first()
            .map(|row| row.feature.schema_version.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        metrics: ForecastModelMetrics::default(),
        global,
        vertical,
        vertical_direction,
        vertical_entity,
        vertical_threshold,
    }
}

fn load_forecast_training_rows(path: &Path) -> Result<Vec<ForecastTrainingRow>, ExecutionError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let mut rows = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str::<ForecastTrainingRow>(line)
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if row.label_resolution_status != "resolved" || row.label_outcome_yes.is_none() {
            continue;
        }
        rows.push(row);
    }
    Ok(rows)
}

fn write_artifact(root: &Path, artifact: &ForecastModelArtifact) -> Result<(), ExecutionError> {
    let version_dir = root.join(&artifact.model_version);
    fs::create_dir_all(&version_dir).map_err(|e| ExecutionError::Exchange(e.to_string()))?;

    let artifact_json =
        serde_json::to_string_pretty(artifact).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let version_path = version_dir.join("artifact.json");
    fs::write(&version_path, artifact_json.as_bytes())
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;

    fs::create_dir_all(root).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    fs::write(root.join("latest.json"), artifact_json.as_bytes())
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;

    let mut manifest = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("manifest.jsonl"))
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    manifest
        .write_all(
            format!(
                "{}\n",
                serde_json::to_string(artifact)
                    .map_err(|e| ExecutionError::Exchange(e.to_string()))?
            )
            .as_bytes(),
        )
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    Ok(())
}

fn evaluate_log_loss(model: &ForecastModel, rows: &[ForecastTrainingRow]) -> Option<f64> {
    average_metric(rows, |row| {
        let y = label_as_prob(row)?;
        let p = model.predict(&row.feature).fair_prob_yes.clamp(0.001, 0.999);
        Some(-(y * p.ln() + (1.0 - y) * (1.0 - p).ln()))
    })
}

fn evaluate_brier(model: &ForecastModel, rows: &[ForecastTrainingRow]) -> Option<f64> {
    average_metric(rows, |row| {
        let y = label_as_prob(row)?;
        let p = model.predict(&row.feature).fair_prob_yes;
        Some((p - y).powi(2))
    })
}

fn evaluate_market_mid_log_loss(rows: &[ForecastTrainingRow]) -> Option<f64> {
    average_metric(rows, |row| {
        let y = label_as_prob(row)?;
        let p = row.feature.mid_prob_yes?.clamp(0.001, 0.999);
        Some(-(y * p.ln() + (1.0 - y) * (1.0 - p).ln()))
    })
}

fn evaluate_market_mid_brier(rows: &[ForecastTrainingRow]) -> Option<f64> {
    average_metric(rows, |row| {
        let y = label_as_prob(row)?;
        let p = row.feature.mid_prob_yes?;
        Some((p - y).powi(2))
    })
}

fn average_metric<F>(rows: &[ForecastTrainingRow], mut f: F) -> Option<f64>
where
    F: FnMut(&ForecastTrainingRow) -> Option<f64>,
{
    let mut total = 0.0;
    let mut count = 0.0;
    for row in rows {
        if let Some(value) = f(row) {
            total += value;
            count += 1.0;
        }
    }
    if count > 0.0 {
        Some(total / count)
    } else {
        None
    }
}

fn label_as_prob(row: &ForecastTrainingRow) -> Option<f64> {
    row.label_outcome_yes.map(|v| if v { 1.0 } else { 0.0 })
}

fn threshold_bucket_key(feature: &ForecastFeatureRow) -> Option<String> {
    let threshold = feature.threshold_value?;
    let direction = feature.threshold_direction.as_deref().unwrap_or("unknown");
    let bucket = ((threshold / 5.0).round() * 5.0) as i64;
    Some(format!("{}|{}|{}", feature.vertical, direction, bucket))
}

fn normalize_key(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sample_row(split: &str, ticker: &str, vertical: &str, outcome_yes: bool) -> ForecastTrainingRow {
        ForecastTrainingRow {
            schema_version: "v1".to_string(),
            split: split.to_string(),
            label_outcome_yes: Some(outcome_yes),
            label_resolution_status: "resolved".to_string(),
            feature: ForecastFeatureRow {
                schema_version: "v1".to_string(),
                feature_ts: Utc.timestamp_opt(1, 0).single().unwrap(),
                ticker: ticker.to_string(),
                title: "Will the high temp in Houston be >70?".to_string(),
                subtitle: None,
                market_type: None,
                event_ticker: None,
                series_ticker: None,
                close_time: None,
                time_to_close_secs: Some(3600),
                yes_bid_cents: Some(0.40),
                yes_ask_cents: Some(0.60),
                mid_prob_yes: Some(0.50),
                spread_cents: Some(0.20),
                volume: 100.0,
                vertical: vertical.to_string(),
                weather_signal: None,
                sports_injury_signal: None,
                crypto_sentiment_signal: None,
                entity_primary: Some("Houston".to_string()),
                entity_secondary: None,
                threshold_value: Some(70.0),
                threshold_direction: Some("above".to_string()),
                event_date_hint: None,
                source: "test".to_string(),
                cycle_id: None,
                recent_trade_count_delta: None,
            },
        }
    }

    #[test]
    fn trained_model_produces_bounded_probability() {
        let train_rows = vec![
            sample_row("train", "A", "weather", true),
            sample_row("train", "B", "weather", true),
            sample_row("train", "C", "weather", false),
            sample_row("train", "D", "sports", false),
        ];
        let artifact = train_artifact(&train_rows, 0, 0);
        let model = ForecastModel::from_artifact(artifact, 1);
        let out = model.predict(&sample_row("test", "E", "weather", true).feature);
        assert!(out.fair_prob_yes > 0.0 && out.fair_prob_yes < 1.0);
        assert!(out.confidence >= 0.0 && out.confidence <= 1.0);
    }
}
