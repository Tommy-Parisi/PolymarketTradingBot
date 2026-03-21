use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::execution::types::ExecutionError;
use crate::research::events::{MarketStateEvent, OrderLifecycleEvent};

#[derive(Debug, Clone)]
pub struct ResearchReportConfig {
    pub enabled: bool,
    pub research_dir: PathBuf,
}

impl ResearchReportConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("BOT_RUN_RESEARCH_REPORT")
                    .unwrap_or_else(|_| "false".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes"
            ),
            research_dir: PathBuf::from(
                std::env::var("BOT_RESEARCH_DIR").unwrap_or_else(|_| "var/research".to_string()),
            ),
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ResearchReportSummary {
    pub research_dir: String,
    pub exists: bool,
    pub market_state_files: usize,
    pub order_lifecycle_files: usize,
    pub outcomes_file_exists: bool,
    pub market_state_rows: usize,
    pub order_lifecycle_rows: usize,
    pub outcome_rows: usize,
    pub unique_market_tickers: usize,
    pub unique_order_tickers: usize,
    pub unique_client_order_ids: usize,
    pub market_rows_by_day: BTreeMap<String, usize>,
    pub order_rows_by_day: BTreeMap<String, usize>,
}

pub fn run_research_report(cfg: &ResearchReportConfig) -> Result<ResearchReportSummary, ExecutionError> {
    if !cfg.research_dir.exists() {
        return Ok(ResearchReportSummary {
            research_dir: cfg.research_dir.display().to_string(),
            exists: false,
            market_state_files: 0,
            order_lifecycle_files: 0,
            outcomes_file_exists: false,
            market_state_rows: 0,
            order_lifecycle_rows: 0,
            outcome_rows: 0,
            unique_market_tickers: 0,
            unique_order_tickers: 0,
            unique_client_order_ids: 0,
            market_rows_by_day: BTreeMap::new(),
            order_rows_by_day: BTreeMap::new(),
        });
    }

    let (market_state_files, market_rows_by_day, market_tickers) =
        scan_market_state(&cfg.research_dir.join("market_state"))?;
    let (order_lifecycle_files, order_rows_by_day, order_tickers, client_order_ids) =
        scan_order_lifecycle(&cfg.research_dir.join("order_lifecycle"))?;
    let outcome_path = cfg.research_dir.join("outcomes").join("outcomes.jsonl");
    let outcome_rows = count_lines_if_exists(&outcome_path)?;

    Ok(ResearchReportSummary {
        research_dir: cfg.research_dir.display().to_string(),
        exists: true,
        market_state_files,
        order_lifecycle_files,
        outcomes_file_exists: outcome_path.exists(),
        market_state_rows: market_rows_by_day.values().sum(),
        order_lifecycle_rows: order_rows_by_day.values().sum(),
        outcome_rows,
        unique_market_tickers: market_tickers.len(),
        unique_order_tickers: order_tickers.len(),
        unique_client_order_ids: client_order_ids.len(),
        market_rows_by_day,
        order_rows_by_day,
    })
}

fn scan_market_state(
    dir: &Path,
) -> Result<(usize, BTreeMap<String, usize>, std::collections::BTreeSet<String>), ExecutionError> {
    let mut file_count = 0usize;
    let mut rows_by_day = BTreeMap::new();
    let mut tickers = std::collections::BTreeSet::new();
    if !dir.exists() {
        return Ok((file_count, rows_by_day, tickers));
    }
    for day_entry in fs::read_dir(dir).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let day_path = day_entry
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?
            .path();
        if !day_path.is_dir() {
            continue;
        }
        let day = day_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let mut row_count = 0usize;
        for file in fs::read_dir(&day_path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let path = file
                .map_err(|e| ExecutionError::Exchange(e.to_string()))?
                .path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            file_count += 1;
            let raw = fs::read_to_string(&path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            for line in raw.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                row_count += 1;
                if let Ok(event) = serde_json::from_str::<MarketStateEvent>(line) {
                    tickers.insert(event.ticker);
                }
            }
        }
        rows_by_day.insert(day, row_count);
    }
    Ok((file_count, rows_by_day, tickers))
}

fn scan_order_lifecycle(
    dir: &Path,
) -> Result<
    (
        usize,
        BTreeMap<String, usize>,
        std::collections::BTreeSet<String>,
        std::collections::BTreeSet<String>,
    ),
    ExecutionError,
> {
    let mut file_count = 0usize;
    let mut rows_by_day = BTreeMap::new();
    let mut tickers = std::collections::BTreeSet::new();
    let mut client_order_ids = std::collections::BTreeSet::new();
    if !dir.exists() {
        return Ok((file_count, rows_by_day, tickers, client_order_ids));
    }
    for day_entry in fs::read_dir(dir).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
        let day_path = day_entry
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?
            .path();
        if !day_path.is_dir() {
            continue;
        }
        let day = day_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let mut row_count = 0usize;
        for file in fs::read_dir(&day_path).map_err(|e| ExecutionError::Exchange(e.to_string()))? {
            let path = file
                .map_err(|e| ExecutionError::Exchange(e.to_string()))?
                .path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            file_count += 1;
            let raw = fs::read_to_string(&path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
            for line in raw.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                row_count += 1;
                if let Ok(event) = serde_json::from_str::<OrderLifecycleEvent>(line) {
                    tickers.insert(event.ticker);
                    client_order_ids.insert(event.client_order_id);
                }
            }
        }
        rows_by_day.insert(day, row_count);
    }
    Ok((file_count, rows_by_day, tickers, client_order_ids))
}

fn count_lines_if_exists(path: &Path) -> Result<usize, ExecutionError> {
    if !path.exists() {
        return Ok(0);
    }
    let raw = fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    Ok(raw.lines().filter(|line| !line.trim().is_empty()).count())
}
