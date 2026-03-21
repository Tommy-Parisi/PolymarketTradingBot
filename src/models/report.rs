use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::datasets::builder::{ExecutionTrainingRow, ForecastTrainingRow};
use crate::execution::types::ExecutionError;
use crate::models::execution::ExecutionModel;
use crate::models::forecast::ForecastModel;

#[derive(Debug, Clone)]
pub struct ModelReportConfig {
    pub enabled: bool,
    pub forecast_dataset_path: PathBuf,
    pub execution_dataset_path: PathBuf,
    pub forecast_model_path: PathBuf,
    pub execution_model_path: PathBuf,
    pub forecast_min_bucket_samples: usize,
    pub execution_min_bucket_samples: usize,
}

impl ModelReportConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_MODEL_REPORT")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            forecast_dataset_path: PathBuf::from(
                std::env::var("BOT_FORECAST_DATASET_PATH")
                    .unwrap_or_else(|_| "var/features/forecast/forecast_training.jsonl".to_string()),
            ),
            execution_dataset_path: PathBuf::from(
                std::env::var("BOT_EXECUTION_DATASET_PATH")
                    .unwrap_or_else(|_| "var/features/execution/execution_training.jsonl".to_string()),
            ),
            forecast_model_path: PathBuf::from(
                std::env::var("BOT_MODEL_FORECAST_REPORT_PATH")
                    .or_else(|_| std::env::var("BOT_MODEL_FORECAST_PATH"))
                    .unwrap_or_else(|_| "var/models/forecast/latest.json".to_string()),
            ),
            execution_model_path: PathBuf::from(
                std::env::var("BOT_MODEL_EXECUTION_REPORT_PATH")
                    .or_else(|_| std::env::var("BOT_MODEL_EXECUTION_PATH"))
                    .unwrap_or_else(|_| "var/models/execution/latest.json".to_string()),
            ),
            forecast_min_bucket_samples: std::env::var("BOT_MODEL_FORECAST_MIN_BUCKET_SAMPLES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1),
            execution_min_bucket_samples: std::env::var("BOT_MODEL_EXECUTION_MIN_BUCKET_SAMPLES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelReportSummary {
    pub forecast: ForecastReportSummary,
    pub execution: ExecutionReportSummary,
}

#[derive(Debug, Serialize)]
pub struct ForecastReportSummary {
    pub model_path: String,
    pub dataset_path: String,
    pub model_version: String,
    pub total_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
    pub validation_log_loss: Option<f64>,
    pub validation_brier: Option<f64>,
    pub validation_market_mid_log_loss: Option<f64>,
    pub validation_market_mid_brier: Option<f64>,
    pub test_log_loss: Option<f64>,
    pub test_brier: Option<f64>,
    pub test_market_mid_log_loss: Option<f64>,
    pub test_market_mid_brier: Option<f64>,
    pub validation_log_loss_lift_vs_mid: Option<f64>,
    pub validation_brier_lift_vs_mid: Option<f64>,
    pub test_log_loss_lift_vs_mid: Option<f64>,
    pub test_brier_lift_vs_mid: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ExecutionReportSummary {
    pub model_path: String,
    pub dataset_path: String,
    pub model_version: String,
    pub total_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
    pub validation_brier_fill_30s: Option<f64>,
    pub validation_brier_fill_5m: Option<f64>,
    pub validation_mae_fill_price: Option<f64>,
    pub validation_mae_markout_5m: Option<f64>,
    pub test_brier_fill_30s: Option<f64>,
    pub test_brier_fill_5m: Option<f64>,
    pub test_mae_fill_price: Option<f64>,
    pub test_mae_markout_5m: Option<f64>,
}

pub fn run_model_report(cfg: &ModelReportConfig) -> Result<ModelReportSummary, ExecutionError> {
    let forecast_model =
        ForecastModel::load_from_path(&cfg.forecast_model_path, cfg.forecast_min_bucket_samples)?;
    let execution_model =
        ExecutionModel::load_from_path(&cfg.execution_model_path, cfg.execution_min_bucket_samples)?;
    let forecast_rows = load_jsonl::<ForecastTrainingRow>(&cfg.forecast_dataset_path)?;
    let execution_rows = load_jsonl::<ExecutionTrainingRow>(&cfg.execution_dataset_path)?;

    let forecast_validation: Vec<_> = forecast_rows
        .iter()
        .filter(|row| row.split == "validation" && row.label_outcome_yes.is_some())
        .collect();
    let forecast_test: Vec<_> = forecast_rows
        .iter()
        .filter(|row| row.split == "test" && row.label_outcome_yes.is_some())
        .collect();
    let execution_validation: Vec<_> = execution_rows
        .iter()
        .filter(|row| row.split == "validation")
        .collect();
    let execution_test: Vec<_> = execution_rows.iter().filter(|row| row.split == "test").collect();

    let forecast = ForecastReportSummary {
        model_path: cfg.forecast_model_path.display().to_string(),
        dataset_path: cfg.forecast_dataset_path.display().to_string(),
        model_version: forecast_model.artifact().model_version.clone(),
        total_rows: forecast_rows.len(),
        validation_rows: forecast_validation.len(),
        test_rows: forecast_test.len(),
        validation_log_loss: forecast_log_loss(&forecast_model, &forecast_validation),
        validation_brier: forecast_brier(&forecast_model, &forecast_validation),
        validation_market_mid_log_loss: forecast_mid_log_loss(&forecast_validation),
        validation_market_mid_brier: forecast_mid_brier(&forecast_validation),
        test_log_loss: forecast_log_loss(&forecast_model, &forecast_test),
        test_brier: forecast_brier(&forecast_model, &forecast_test),
        test_market_mid_log_loss: forecast_mid_log_loss(&forecast_test),
        test_market_mid_brier: forecast_mid_brier(&forecast_test),
        validation_log_loss_lift_vs_mid: lift(
            forecast_mid_log_loss(&forecast_validation),
            forecast_log_loss(&forecast_model, &forecast_validation),
        ),
        validation_brier_lift_vs_mid: lift(
            forecast_mid_brier(&forecast_validation),
            forecast_brier(&forecast_model, &forecast_validation),
        ),
        test_log_loss_lift_vs_mid: lift(
            forecast_mid_log_loss(&forecast_test),
            forecast_log_loss(&forecast_model, &forecast_test),
        ),
        test_brier_lift_vs_mid: lift(
            forecast_mid_brier(&forecast_test),
            forecast_brier(&forecast_model, &forecast_test),
        ),
    };

    let execution = ExecutionReportSummary {
        model_path: cfg.execution_model_path.display().to_string(),
        dataset_path: cfg.execution_dataset_path.display().to_string(),
        model_version: execution_model
            .predict(&execution_rows.first().map(|r| r.feature.clone()).unwrap_or_else(empty_execution_feature))
            .model_version,
        total_rows: execution_rows.len(),
        validation_rows: execution_validation.len(),
        test_rows: execution_test.len(),
        validation_brier_fill_30s: execution_brier(&execution_model, &execution_validation, true),
        validation_brier_fill_5m: execution_brier(&execution_model, &execution_validation, false),
        validation_mae_fill_price: execution_mae_fill_price(&execution_model, &execution_validation),
        validation_mae_markout_5m: execution_mae_markout_5m(&execution_model, &execution_validation),
        test_brier_fill_30s: execution_brier(&execution_model, &execution_test, true),
        test_brier_fill_5m: execution_brier(&execution_model, &execution_test, false),
        test_mae_fill_price: execution_mae_fill_price(&execution_model, &execution_test),
        test_mae_markout_5m: execution_mae_markout_5m(&execution_model, &execution_test),
    };

    Ok(ModelReportSummary { forecast, execution })
}

fn load_jsonl<T>(path: &PathBuf) -> Result<Vec<T>, ExecutionError>
where
    T: for<'de> Deserialize<'de>,
{
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
            serde_json::from_str::<T>(line).map_err(|e| ExecutionError::Exchange(e.to_string()))?,
        );
    }
    Ok(rows)
}

fn forecast_log_loss(model: &ForecastModel, rows: &[&ForecastTrainingRow]) -> Option<f64> {
    avg_iter(rows.iter().filter_map(|row| {
        let y = if row.label_outcome_yes? { 1.0 } else { 0.0 };
        let p = model.predict(&row.feature).fair_prob_yes.clamp(0.001, 0.999);
        Some(-(y * p.ln() + (1.0 - y) * (1.0 - p).ln()))
    }))
}

fn forecast_brier(model: &ForecastModel, rows: &[&ForecastTrainingRow]) -> Option<f64> {
    avg_iter(rows.iter().filter_map(|row| {
        let y = if row.label_outcome_yes? { 1.0 } else { 0.0 };
        let p = model.predict(&row.feature).fair_prob_yes;
        Some((p - y).powi(2))
    }))
}

fn forecast_mid_log_loss(rows: &[&ForecastTrainingRow]) -> Option<f64> {
    avg_iter(rows.iter().filter_map(|row| {
        let y = if row.label_outcome_yes? { 1.0 } else { 0.0 };
        let p = row.feature.mid_prob_yes?.clamp(0.001, 0.999);
        Some(-(y * p.ln() + (1.0 - y) * (1.0 - p).ln()))
    }))
}

fn forecast_mid_brier(rows: &[&ForecastTrainingRow]) -> Option<f64> {
    avg_iter(rows.iter().filter_map(|row| {
        let y = if row.label_outcome_yes? { 1.0 } else { 0.0 };
        let p = row.feature.mid_prob_yes?;
        Some((p - y).powi(2))
    }))
}

fn execution_brier(model: &ExecutionModel, rows: &[&ExecutionTrainingRow], fill_30s: bool) -> Option<f64> {
    avg_iter(rows.iter().map(|row| {
        let estimate = model.predict(&row.feature);
        let p = if fill_30s {
            estimate.fill_prob_30s
        } else {
            estimate.fill_prob_5m
        };
        let y = if if fill_30s {
            row.label_filled_within_30s
        } else {
            row.label_filled_within_5m
        } {
            1.0
        } else {
            0.0
        };
        (p - y).powi(2)
    }))
}

fn execution_mae_fill_price(model: &ExecutionModel, rows: &[&ExecutionTrainingRow]) -> Option<f64> {
    avg_iter(rows.iter().filter_map(|row| {
        let y = row.label_terminal_avg_fill_price?;
        Some((model.predict(&row.feature).expected_fill_price - y).abs())
    }))
}

fn execution_mae_markout_5m(model: &ExecutionModel, rows: &[&ExecutionTrainingRow]) -> Option<f64> {
    avg_iter(rows.iter().filter_map(|row| {
        let y = row.label_markout_bps_5m?;
        Some((model.predict(&row.feature).expected_markout_bps_5m - y).abs())
    }))
}

fn avg_iter<I>(iter: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut total = 0.0;
    let mut count = 0.0;
    for value in iter {
        total += value;
        count += 1.0;
    }
    (count > 0.0).then_some(total / count)
}

fn lift(baseline: Option<f64>, model: Option<f64>) -> Option<f64> {
    match (baseline, model) {
        (Some(b), Some(m)) if b.abs() > f64::EPSILON => Some((b - m) / b),
        _ => None,
    }
}

fn empty_execution_feature() -> crate::features::execution::ExecutionFeatureRow {
    crate::features::execution::ExecutionFeatureRow {
        schema_version: "v1".to_string(),
        feature_ts: chrono::Utc::now(),
        ticker: String::new(),
        outcome_id: "yes".to_string(),
        side: crate::execution::types::Side::Buy,
        tif: crate::execution::types::TimeInForce::Gtc,
        title: String::new(),
        vertical: "unknown".to_string(),
        candidate_limit_price: 0.5,
        candidate_observed_price: 0.5,
        candidate_fair_price: 0.5,
        raw_edge_pct: 0.0,
        confidence: 0.0,
        yes_bid_cents: None,
        yes_ask_cents: None,
        spread_cents: None,
        mid_prob_yes: None,
        volume: 0.0,
        time_to_close_secs: None,
        price_vs_best_bid_cents: None,
        price_vs_best_ask_cents: None,
        aggressiveness_bps: None,
        open_order_count_same_ticker: 0,
        recent_fill_count_same_ticker: 0,
        recent_cancel_count_same_ticker: 0,
        same_event_exposure_notional: 0.0,
    }
}
