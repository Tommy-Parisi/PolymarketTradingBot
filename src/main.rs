use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::time::MissedTickBehavior;

mod data;
mod datasets;
mod execution;
mod features;
mod markets;
mod model;
mod models;
mod outcomes;
mod policy;
mod replay;
mod research;

use data::market_enrichment::{EnrichmentConfig, MarketEnricher, select_for_enrichment};
use data::market_scanner::{KalshiMarketScanner, ScannedMarket, ScannerConfig};
use datasets::builder::{DatasetBuildConfig, run_dataset_build};
use execution::client::{ExchangeClient, KalshiClient};
use execution::engine::{ExecutionEngine, ExecutionMode, summarize_performance_paths};
use execution::paper_sim::{PaperSimClient, PaperSimConfig};
use execution::types::{EngineConfig, OrderStatus, TimeInForce};
use features::execution::{ExecutionContext, build_execution_feature_row_from_forecast};
use features::forecast::build_forecast_feature_row;
use markets::kalshi_mapper::{resolution_mode_from_env, KalshiMarketMapper, ResolutionMode};
use model::allocator::{AllocatedTrade, AllocationConfig, PortfolioAllocator};
use model::valuation::{CandidateTrade, ClaudeValuationEngine, MarketValuation, ValuationConfig, ValuationInput};
use models::execution::{
    ExecutionModel, ExecutionModelArtifact, ExecutionRuntimeConfig, ExecutionTrainingConfig,
    load_runtime_model as load_execution_runtime_model,
    record_shadow_outputs as record_execution_shadow_outputs, run_execution_training,
};
use models::forecast::{
    ForecastModel, ForecastModelArtifact, ForecastRuntimeConfig, ForecastTrainingConfig, load_runtime_model,
    record_shadow_outputs, run_forecast_training,
};
use models::report::{ModelReportConfig, run_model_report};
use outcomes::resolver::{OutcomeResolverConfig, run_outcome_backfill};
use policy::decision::{PolicyConfig, PolicyDecision, PolicyMode, decide_shadow_policy, record_shadow_decisions};
use policy::report::{PolicyReportConfig, run_policy_report};
use research::market_recorder::{ResearchCaptureConfig, record_scan_trace};
use research::report::{ResearchReportConfig, run_research_report};

struct BotRuntime {
    mode: ExecutionMode,
    engine: ExecutionEngine,
    scanner: KalshiMarketScanner,
    enricher: MarketEnricher,
    valuator: ClaudeValuationEngine,
    allocator: PortfolioAllocator,
    mapper: KalshiMarketMapper,
    resolution_mode: ResolutionMode,
    bankroll: f64,
    valuation_limit: usize,
    enrichment_limit: usize,
    claude_every_n_cycles: u64,
    claude_trigger_mode: ClaudeTriggerMode,
    cycle_counter: AtomicU64,
    research_capture: ResearchCaptureConfig,
    forecast_model: Option<ForecastModel>,
    forecast_runtime: ForecastRuntimeConfig,
    execution_model: Option<ExecutionModel>,
    execution_runtime: ExecutionRuntimeConfig,
    policy_config: PolicyConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeTriggerMode {
    Never,
    Cadence,
    OnViableMarkets,
    OnHeuristicCandidates,
}

#[tokio::main]
async fn main() {
    if run_summary_only_mode() {
        let cfg = engine_config_from_env();
        match summarize_performance_paths(&cfg.state_path, &cfg.journal_path) {
            Ok(summary) => match serde_json::to_string_pretty(&summary) {
                Ok(s) => println!("{s}"),
                Err(err) => eprintln!("failed to render pnl summary: {err}"),
            },
            Err(err) => eprintln!("failed to build pnl summary: {err}"),
        }
        return;
    }

    let policy_report_cfg = PolicyReportConfig::from_env(&cycle_artifacts_dir());
    if policy_report_cfg.enabled {
        match run_policy_report(&policy_report_cfg) {
            Ok(summary) => match serde_json::to_string_pretty(&summary) {
                Ok(s) => println!("{s}"),
                Err(err) => eprintln!("failed to render policy report: {err}"),
            },
            Err(err) => eprintln!("policy report failed: {err}"),
        }
        return;
    }

    let research_report_cfg = ResearchReportConfig::from_env();
    if research_report_cfg.enabled {
        match run_research_report(&research_report_cfg) {
            Ok(summary) => match serde_json::to_string_pretty(&summary) {
                Ok(s) => println!("{s}"),
                Err(err) => eprintln!("failed to render research report: {err}"),
            },
            Err(err) => eprintln!("research report failed: {err}"),
        }
        return;
    }

    let model_report_cfg = ModelReportConfig::from_env();
    if model_report_cfg.enabled {
        match run_model_report(&model_report_cfg) {
            Ok(summary) => match serde_json::to_string_pretty(&summary) {
                Ok(s) => println!("{s}"),
                Err(err) => eprintln!("failed to render model report: {err}"),
            },
            Err(err) => eprintln!("model report failed: {err}"),
        }
        return;
    }

    let outcome_cfg = OutcomeResolverConfig::from_env();
    if outcome_cfg.enabled {
        if let Err(err) = run_outcome_backfill(&outcome_cfg).await {
            eprintln!("outcome backfill failed: {err}");
        }
        return;
    }

    let dataset_cfg = DatasetBuildConfig::from_env();
    if dataset_cfg.enabled {
        if let Err(err) = run_dataset_build(&dataset_cfg).await {
            eprintln!("dataset build failed: {err}");
        }
        return;
    }

    let forecast_training_cfg = ForecastTrainingConfig::from_env();
    if forecast_training_cfg.enabled {
        if let Err(err) = run_forecast_training(&forecast_training_cfg).await {
            eprintln!("forecast training failed: {err}");
        }
        return;
    }

    let execution_training_cfg = ExecutionTrainingConfig::from_env();
    if execution_training_cfg.enabled {
        if let Err(err) = run_execution_training(&execution_training_cfg).await {
            eprintln!("execution training failed: {err}");
        }
        return;
    }

    if research_capture_only_mode_enabled() {
        if let Err(err) = run_research_capture_mode().await {
            eprintln!("research capture mode failed: {err}");
        }
        return;
    }

    if research_paper_capture_mode_enabled() {
        run_research_paper_capture_mode().await;
        return;
    }

    if replay_mode_enabled() {
        replay::run_multi_day_replay().await;
        return;
    }

    let client: Arc<dyn ExchangeClient> = match exchange_backend_from_env().as_str() {
        "paper_sim" => Arc::new(PaperSimClient::new(PaperSimConfig::default())),
        _ => match KalshiClient::from_env() {
            Ok(client) => Arc::new(client),
            Err(err) => {
                eprintln!("missing/invalid Kalshi env config: {err}");
                return;
            }
        },
    };

    let mode = execution_mode_from_env();
    let engine_cfg = engine_config_from_env();
    let forecast_runtime = ForecastRuntimeConfig::from_env();
    let forecast_model = match load_runtime_model(&forecast_runtime) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("forecast model load warning: {err}");
            None
        }
    };
    let execution_runtime = ExecutionRuntimeConfig::from_env();
    let execution_model = match load_execution_runtime_model(&execution_runtime) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("execution model load warning: {err}");
            None
        }
    };
    if let Err(err) = validate_startup_paths(&engine_cfg, &forecast_runtime, &execution_runtime) {
        eprintln!("startup validation failed: {err}");
        return;
    }
    let policy_config = PolicyConfig::from_env();
    if let Err(err) = validate_active_policy_requirements(
        &policy_config,
        forecast_model.as_ref(),
        execution_model.as_ref(),
    ) {
        eprintln!("policy active mode validation failed: {err}");
        return;
    }
    let engine = ExecutionEngine::new(client, engine_cfg, mode);

    let runtime = BotRuntime {
        mode,
        engine,
        scanner: KalshiMarketScanner::new(ScannerConfig::default()),
        enricher: MarketEnricher::new(EnrichmentConfig::default()),
        valuator: ClaudeValuationEngine::new(ValuationConfig::from_env()),
        allocator: PortfolioAllocator::new(allocation_config_from_env()),
        mapper: KalshiMarketMapper::from_env(),
        resolution_mode: resolution_mode_from_env(),
        bankroll: bankroll_from_env(),
        valuation_limit: valuation_market_limit_from_env(),
        enrichment_limit: enrichment_market_limit_from_env(),
        claude_every_n_cycles: claude_every_n_cycles_from_env(),
        claude_trigger_mode: claude_trigger_mode_from_env(),
        cycle_counter: AtomicU64::new(0),
        research_capture: ResearchCaptureConfig::from_env(),
        forecast_model,
        forecast_runtime,
        execution_model,
        execution_runtime,
        policy_config,
    };

    if runtime.mode == ExecutionMode::Live && smoke_test_enabled() {
        if let Err(err) = runtime.engine.run_smoke_test().await {
            eprintln!("kalshi smoke test failed: {err}");
            return;
        }
    }
    if runtime.mode == ExecutionMode::Live {
        if let Err(err) = runtime.engine.reconcile_open_orders().await {
            eprintln!("startup reconcile warning: {err}");
        }
    }

    if run_once_mode() {
        run_cycle(&runtime).await;
        return;
    }

    run_cycle(&runtime).await;
    let cycle_secs = cycle_seconds_from_env();
    let mut interval = tokio::time::interval(Duration::from_secs(cycle_secs));
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        run_cycle(&runtime).await;
    }
}

fn validate_startup_paths(
    engine_cfg: &EngineConfig,
    forecast_runtime: &ForecastRuntimeConfig,
    execution_runtime: &ExecutionRuntimeConfig,
) -> Result<(), String> {
    ensure_parent_dir(&engine_cfg.state_path)?;
    ensure_parent_dir(&engine_cfg.journal_path)?;
    ensure_optional_path_parent(forecast_runtime.model_path.as_ref())?;
    ensure_optional_path_parent(execution_runtime.model_path.as_ref())?;
    Ok(())
}

fn ensure_parent_dir(path_str: &str) -> Result<(), String> {
    let path = Path::new(path_str);
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent)
        .map_err(|err| format!("failed to prepare {}: {}", parent.display(), err))
}

fn ensure_optional_path_parent(path: Option<&PathBuf>) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent)
        .map_err(|err| format!("failed to prepare {}: {}", parent.display(), err))
}

fn validate_active_policy_requirements(
    policy_config: &PolicyConfig,
    forecast_model: Option<&ForecastModel>,
    execution_model: Option<&ExecutionModel>,
) -> Result<(), String> {
    if policy_config.mode != PolicyMode::Active {
        return Ok(());
    }

    let forecast_model =
        forecast_model.ok_or_else(|| "active mode requires a loaded forecast model".to_string())?;
    let execution_model =
        execution_model.ok_or_else(|| "active mode requires a loaded execution model".to_string())?;

    validate_forecast_artifact_for_active(policy_config, forecast_model.artifact())?;
    validate_execution_artifact_for_active(policy_config, execution_model.artifact())?;
    validate_shadow_calibration(policy_config)?;
    Ok(())
}

fn validate_forecast_artifact_for_active(
    policy_config: &PolicyConfig,
    artifact: &ForecastModelArtifact,
) -> Result<(), String> {
    let age_hours = (Utc::now() - artifact.trained_at).num_hours();
    if age_hours > policy_config.active_max_model_age_hours {
        return Err(format!(
            "forecast model {} is too old for active mode (age={}h > limit={}h)",
            artifact.model_version, age_hours, policy_config.active_max_model_age_hours
        ));
    }
    if artifact.train_rows < policy_config.active_min_forecast_train_rows {
        return Err(format!(
            "forecast model {} has too little training data for active mode (train_rows={} < min={})",
            artifact.model_version, artifact.train_rows, policy_config.active_min_forecast_train_rows
        ));
    }
    Ok(())
}

fn validate_execution_artifact_for_active(
    policy_config: &PolicyConfig,
    artifact: &ExecutionModelArtifact,
) -> Result<(), String> {
    let age_hours = (Utc::now() - artifact.trained_at).num_hours();
    if age_hours > policy_config.active_max_model_age_hours {
        return Err(format!(
            "execution model {} is too old for active mode (age={}h > limit={}h)",
            artifact.model_version, age_hours, policy_config.active_max_model_age_hours
        ));
    }
    if artifact.train_rows < policy_config.active_min_execution_train_rows {
        return Err(format!(
            "execution model {} has too little training data for active mode (train_rows={} < min={})",
            artifact.model_version, artifact.train_rows, policy_config.active_min_execution_train_rows
        ));
    }
    if policy_config.active_require_live_real
        && artifact.live_real_rows < policy_config.active_min_execution_live_real_rows
    {
        return Err(format!(
            "execution model {} lacks enough live-real rows for active mode (live_real_rows={} < min={})",
            artifact.model_version,
            artifact.live_real_rows,
            policy_config.active_min_execution_live_real_rows
        ));
    }
    Ok(())
}

fn validate_shadow_calibration(policy_config: &PolicyConfig) -> Result<(), String> {
    let shadow_policy_dir = policy_config.shadow_root.join("policy");

    let cutoff = (Utc::now() - chrono::Duration::days(policy_config.active_shadow_lookback_days))
        .date_naive();

    let mut total_decisions = 0usize;
    let mut should_trade_count = 0usize;
    let mut erpnl_sum = 0.0f64;

    let entries = match std::fs::read_dir(&shadow_policy_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "shadow policy directory not found: {}; run in shadow mode first",
                shadow_policy_dir.display()
            ));
        }
        Err(e) => return Err(format!("cannot read shadow dir: {e}")),
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => return Err(format!("shadow dir entry error: {e}")),
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(day_str) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(day) = chrono::NaiveDate::parse_from_str(day_str, "%Y-%m-%d") else {
            continue;
        };
        if day < cutoff {
            continue;
        }
        let file_path = path.join("policy_shadow.jsonl");
        let text = match std::fs::read_to_string(&file_path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot read shadow file: {e}")),
        };
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            total_decisions += 1;
            let should_trade = v.get("should_trade").and_then(|x| x.as_bool()).unwrap_or(false);
            if should_trade {
                let erpnl = v
                    .get("expected_realized_pnl")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0);
                should_trade_count += 1;
                erpnl_sum += erpnl;
            }
        }
    }

    if total_decisions < policy_config.active_min_shadow_decisions {
        return Err(format!(
            "active mode requires at least {} shadow decisions in the last {} days (found {}); \
             run in shadow mode first",
            policy_config.active_min_shadow_decisions,
            policy_config.active_shadow_lookback_days,
            total_decisions,
        ));
    }

    if should_trade_count > 0 {
        let mean_erpnl = erpnl_sum / should_trade_count as f64;
        if mean_erpnl < policy_config.active_min_shadow_mean_erpnl {
            return Err(format!(
                "shadow mean expected_realized_pnl ({:.2}) is below threshold ({:.2}); \
                 recalibrate before enabling active mode",
                mean_erpnl, policy_config.active_min_shadow_mean_erpnl,
            ));
        }
    }

    Ok(())
}

async fn run_research_capture_mode() -> Result<(), execution::types::ExecutionError> {
    let scanner = KalshiMarketScanner::new(ScannerConfig::default());
    let enricher = MarketEnricher::new(EnrichmentConfig::default());
    let research_capture = ResearchCaptureConfig::from_env();
    let enrichment_limit = enrichment_market_limit_from_env();
    let run_once = run_once_mode();
    let every = Duration::from_secs(cycle_seconds_from_env());

    if run_once {
        run_research_capture_cycle(&scanner, &enricher, &research_capture, enrichment_limit, 1).await?;
        return Ok(());
    }

    let mut interval = tokio::time::interval(every);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut cycle_number = 0u64;
    loop {
        interval.tick().await;
        cycle_number += 1;
        if let Err(err) =
            run_research_capture_cycle(&scanner, &enricher, &research_capture, enrichment_limit, cycle_number).await
        {
            eprintln!("research capture cycle failed: {err}");
        }
    }
}

async fn run_research_capture_cycle(
    scanner: &KalshiMarketScanner,
    enricher: &MarketEnricher,
    research_capture: &ResearchCaptureConfig,
    enrichment_limit: usize,
    cycle_number: u64,
) -> Result<(), execution::types::ExecutionError> {
    let started_at = Utc::now();
    let cycle_id = format!("capture-{}-{}", started_at.format("%Y%m%dT%H%M%S"), cycle_number);
    println!("research capture cycle #{} starting", cycle_number);

    let scan_trace = scanner.scan_snapshot_with_trace().await?;

    let selected = scanner.select_for_valuation(scan_trace.final_markets.clone());
    let to_enrich = select_for_enrichment(&selected, enrichment_limit);
    let enriched = enricher.enrich_batch(&to_enrich).await?;

    let mut enrichment_by_ticker = std::collections::HashMap::new();
    for e in &enriched {
        enrichment_by_ticker.insert(e.ticker.clone(), e.clone());
    }

    // Record after enrichment so finance_price_signal is included
    record_scan_trace(
        research_capture,
        &cycle_id,
        &scan_trace.snapshot_markets,
        &scan_trace,
        Some(&enrichment_by_ticker),
    )?;

    println!(
        "research capture cycle complete: selected_markets={} enriched_markets={} research_dir={}",
        selected.len(),
        enriched.len(),
        research_capture.root_dir.display()
    );
    Ok(())
}

async fn run_research_paper_capture_mode() {
    let client: Arc<dyn ExchangeClient> = Arc::new(PaperSimClient::new(PaperSimConfig::default()));
    let runtime = BotRuntime {
        mode: ExecutionMode::Paper,
        engine: ExecutionEngine::new(client, engine_config_from_env(), ExecutionMode::Paper),
        scanner: KalshiMarketScanner::new(ScannerConfig::default()),
        enricher: MarketEnricher::new(EnrichmentConfig::default()),
        valuator: ClaudeValuationEngine::new(ValuationConfig::from_env()),
        allocator: PortfolioAllocator::new(allocation_config_from_env()),
        mapper: KalshiMarketMapper::from_env(),
        resolution_mode: resolution_mode_from_env(),
        bankroll: bankroll_from_env(),
        valuation_limit: valuation_market_limit_from_env(),
        enrichment_limit: enrichment_market_limit_from_env(),
        claude_every_n_cycles: claude_every_n_cycles_from_env(),
        claude_trigger_mode: claude_trigger_mode_from_env(),
        cycle_counter: AtomicU64::new(0),
        research_capture: ResearchCaptureConfig::from_env(),
        forecast_model: None,
        forecast_runtime: ForecastRuntimeConfig::from_env(),
        execution_model: None,
        execution_runtime: ExecutionRuntimeConfig::from_env(),
        policy_config: PolicyConfig::from_env(),
    };

    if run_once_mode() {
        run_cycle(&runtime).await;
        return;
    }

    let every = Duration::from_secs(cycle_seconds_from_env());
    let mut interval = tokio::time::interval(every);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        run_cycle(&runtime).await;
    }
}

async fn run_cycle(runtime: &BotRuntime) {
    let cycle_number = runtime.cycle_counter.fetch_add(1, Ordering::Relaxed) + 1;
    let cadence_use_claude = should_use_claude_for_cycle(cycle_number, runtime.claude_every_n_cycles);
    let started_at = Utc::now();
    let cycle_id = format!("cycle-{}-{}", started_at.format("%Y%m%dT%H%M%S"), cycle_number);
    println!(
        "starting cycle #{} (claude_trigger_mode={:?} cadence_claude_enabled={} cadence={})",
        cycle_number, runtime.claude_trigger_mode, cadence_use_claude, runtime.claude_every_n_cycles
    );

    // Periodically run outcome backfill and dataset build (every 3 hours / 18 cycles)
    if cycle_number > 1 && cycle_number % 18 == 0 {
        println!("periodic background tasks: starting outcome backfill and dataset build");
        let outcome_cfg = OutcomeResolverConfig::from_env();
        if let Err(err) = run_outcome_backfill(&outcome_cfg).await {
            eprintln!("periodic outcome backfill failed: {err}");
        }
        let dataset_cfg = DatasetBuildConfig::from_env();
        if let Err(err) = run_dataset_build(&dataset_cfg).await {
            eprintln!("periodic dataset build failed: {err}");
        }
    }

    let scan_trace = match runtime.scanner.scan_snapshot_with_trace().await {
        Ok(trace) => trace,
        Err(err) => {
            eprintln!("market scan failed: {err}");
            persist_cycle_artifact(CycleArtifact {
                started_at,
                finished_at: Utc::now(),
                status: "scan_failed".to_string(),
                message: Some(err.to_string()),
                selected_markets: Vec::new(),
                valuations: Vec::new(),
                candidates: Vec::new(),
                policy_mode: format!("{:?}", runtime.policy_config.mode),
                policy_decisions: Vec::new(),
                allocations: Vec::new(),
                executions: Vec::new(),
            });
            return;
        }
    };
    let scanned = scan_trace.final_markets.clone();
    log_position_marks_from_journal(&scanned);
    let selected = runtime.scanner.select_for_valuation(scanned);
    if selected.is_empty() {
        eprintln!("no markets selected for valuation");
        // Still record scan trace (no enrichment available)
        if let Err(err) = record_scan_trace(
            &runtime.research_capture,
            &cycle_id,
            &scan_trace.snapshot_markets,
            &scan_trace,
            None,
        ) {
            eprintln!("research capture warning (market_state): {err}");
        }
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "no_markets".to_string(),
            message: Some("no markets selected for valuation".to_string()),
            selected_markets: Vec::new(),
            valuations: Vec::new(),
            candidates: Vec::new(),
            policy_mode: format!("{:?}", runtime.policy_config.mode),
            policy_decisions: Vec::new(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("selected {} markets for valuation", selected.len());

    let to_enrich = select_for_enrichment(&selected, runtime.enrichment_limit);
    let enrichments = match runtime.enricher.enrich_batch(&to_enrich).await {
        Ok(v) => v,
        Err(err) => {
            eprintln!("enrichment failed: {err}");
            // Record scan trace without enrichment before bailing out
            if let Err(e) = record_scan_trace(
                &runtime.research_capture,
                &cycle_id,
                &scan_trace.snapshot_markets,
                &scan_trace,
                None,
            ) {
                eprintln!("research capture warning (market_state): {e}");
            }
            persist_cycle_artifact(CycleArtifact {
                started_at,
                finished_at: Utc::now(),
                status: "enrichment_failed".to_string(),
                message: Some(err.to_string()),
                selected_markets: selected
                    .iter()
                    .map(|m| artifact_market(m))
                    .take(100)
                    .collect(),
                valuations: Vec::new(),
                candidates: Vec::new(),
                policy_mode: format!("{:?}", runtime.policy_config.mode),
                policy_decisions: Vec::new(),
                allocations: Vec::new(),
                executions: Vec::new(),
            });
            return;
        }
    };
    println!("enriched {} markets", enrichments.len());

    let mut enrichment_by_ticker = std::collections::HashMap::new();
    for e in enrichments {
        enrichment_by_ticker.insert(e.ticker.clone(), e);
    }

    // Record market state after enrichment so finance_price_signal is captured
    if let Err(err) = record_scan_trace(
        &runtime.research_capture,
        &cycle_id,
        &scan_trace.snapshot_markets,
        &scan_trace,
        Some(&enrichment_by_ticker),
    ) {
        eprintln!("research capture warning (market_state): {err}");
    }

    let valuation_limit = runtime.valuation_limit.min(selected.len());
    let valuation_inputs: Vec<ValuationInput> = selected
        .iter()
        .take(valuation_limit)
        .map(|m| ValuationInput {
            market: m.clone(),
            enrichment: enrichment_by_ticker.get(&m.ticker).cloned(),
        })
        .collect();

    // Collect forecast model fair prices to override heuristic valuations below.
    let mut forecast_fair_prices: std::collections::HashMap<String, (f64, f64)> =
        std::collections::HashMap::new();

    if let Some(model) = &runtime.forecast_model {
        let shadow_rows: Vec<_> = valuation_inputs
            .iter()
            .map(|input| {
                let feature = build_forecast_feature_row(
                    &input.market,
                    input.enrichment.as_ref(),
                    started_at,
                );
                let output = model.predict(&feature);
                (feature, output)
            })
            .collect();
        if let Some(top) = shadow_rows
            .iter()
            .filter_map(|(feature, output)| {
                let market_mid = feature.mid_prob_yes?;
                Some((feature, output, (output.fair_prob_yes - market_mid).abs()))
            })
            .max_by(|a, b| a.2.total_cmp(&b.2))
        {
            println!(
                "forecast shadow: model_version={} rows={} top_delta_ticker={} market_mid={:.4} fair={:.4} delta={:.4}",
                top.1.model_version,
                shadow_rows.len(),
                top.0.ticker,
                top.0.mid_prob_yes.unwrap_or(0.0),
                top.1.fair_prob_yes,
                top.2
            );
        } else {
            println!(
                "forecast shadow: model_version={} rows={} (no comparable market mids)",
                shadow_rows
                    .first()
                    .map(|(_, output)| output.model_version.as_str())
                    .unwrap_or("unknown"),
                shadow_rows.len()
            );
        }
        // Series prefixes where the forecast model override is suppressed because
        // real-time external data (e.g. live price feed) makes the market correct by
        // construction. Controlled by BOT_FORECAST_SERIES_EXCLUSIONS (comma-separated).
        let forecast_exclusions: Vec<String> = std::env::var("BOT_FORECAST_SERIES_EXCLUSIONS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        for (feature, output) in &shadow_rows {
            let excluded = forecast_exclusions.iter().any(|prefix| {
                feature.ticker.starts_with(prefix.as_str())
                    || feature.series_ticker.as_deref().unwrap_or("").starts_with(prefix.as_str())
            });
            if !excluded {
                forecast_fair_prices.insert(feature.ticker.clone(), (output.fair_prob_yes, output.confidence));
            }
        }
        let borrowed_rows: Vec<_> = shadow_rows.iter().map(|(feature, output)| (feature, output.clone())).collect();
        if let Err(err) = record_shadow_outputs(&runtime.forecast_runtime, &cycle_id, &borrowed_rows) {
            eprintln!("forecast shadow warning: {err}");
        }
    }

    let (valuations, claude_attempted) = match runtime.claude_trigger_mode {
        ClaudeTriggerMode::Never => {
            let valuations = match runtime
                .valuator
                .value_markets_with_claude_enabled(&valuation_inputs, false)
                .await
            {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("valuation failed: {err}");
                    persist_cycle_artifact(CycleArtifact {
                        started_at,
                        finished_at: Utc::now(),
                        status: "valuation_failed".to_string(),
                        message: Some(err.to_string()),
                        selected_markets: selected
                            .iter()
                            .map(|m| artifact_market(m))
                            .take(100)
                            .collect(),
                        valuations: Vec::new(),
                        candidates: Vec::new(),
                        policy_mode: format!("{:?}", runtime.policy_config.mode),
                        policy_decisions: Vec::new(),
                        allocations: Vec::new(),
                        executions: Vec::new(),
                    });
                    return;
                }
            };
            (valuations, false)
        }
        ClaudeTriggerMode::Cadence => {
            let valuations = match runtime
                .valuator
                .value_markets_with_claude_enabled(&valuation_inputs, cadence_use_claude)
                .await
            {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("valuation failed: {err}");
                    persist_cycle_artifact(CycleArtifact {
                        started_at,
                        finished_at: Utc::now(),
                        status: "valuation_failed".to_string(),
                        message: Some(err.to_string()),
                        selected_markets: selected
                            .iter()
                            .map(|m| artifact_market(m))
                            .take(100)
                            .collect(),
                        valuations: Vec::new(),
                        candidates: Vec::new(),
                        policy_mode: format!("{:?}", runtime.policy_config.mode),
                        policy_decisions: Vec::new(),
                        allocations: Vec::new(),
                        executions: Vec::new(),
                    });
                    return;
                }
            };
            (valuations, cadence_use_claude)
        }
        ClaudeTriggerMode::OnViableMarkets => {
            let use_claude = !valuation_inputs.is_empty();
            let valuations = match runtime
                .valuator
                .value_markets_with_claude_enabled(&valuation_inputs, use_claude)
                .await
            {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("valuation failed: {err}");
                    persist_cycle_artifact(CycleArtifact {
                        started_at,
                        finished_at: Utc::now(),
                        status: "valuation_failed".to_string(),
                        message: Some(err.to_string()),
                        selected_markets: selected
                            .iter()
                            .map(|m| artifact_market(m))
                            .take(100)
                            .collect(),
                        valuations: Vec::new(),
                        candidates: Vec::new(),
                        policy_mode: format!("{:?}", runtime.policy_config.mode),
                        policy_decisions: Vec::new(),
                        allocations: Vec::new(),
                        executions: Vec::new(),
                    });
                    return;
                }
            };
            (valuations, use_claude)
        }
        ClaudeTriggerMode::OnHeuristicCandidates => {
            let heuristic_vals = match runtime
                .valuator
                .value_markets_screening(&valuation_inputs)
                .await
            {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("valuation failed: {err}");
                    persist_cycle_artifact(CycleArtifact {
                        started_at,
                        finished_at: Utc::now(),
                        status: "valuation_failed".to_string(),
                        message: Some(err.to_string()),
                        selected_markets: selected
                            .iter()
                            .map(|m| artifact_market(m))
                            .take(100)
                            .collect(),
                        valuations: Vec::new(),
                        candidates: Vec::new(),
                        policy_mode: format!("{:?}", runtime.policy_config.mode),
                        policy_decisions: Vec::new(),
                        allocations: Vec::new(),
                        executions: Vec::new(),
                    });
                    return;
                }
            };
            let heuristic_candidates = runtime.valuator.generate_candidates(&heuristic_vals);
            if heuristic_candidates.is_empty() {
                println!(
                    "claude trigger: skipped (no heuristic candidates for this cycle)"
                );
                (heuristic_vals, false)
            } else {
                println!(
                    "claude trigger: heuristic found {} candidates, running claude valuation",
                    heuristic_candidates.len()
                );
                let claude_vals = match runtime
                    .valuator
                    .value_markets_with_claude_enabled(&valuation_inputs, true)
                    .await
                {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("valuation failed: {err}");
                        persist_cycle_artifact(CycleArtifact {
                            started_at,
                            finished_at: Utc::now(),
                            status: "valuation_failed".to_string(),
                            message: Some(err.to_string()),
                            selected_markets: selected
                                .iter()
                                .map(|m| artifact_market(m))
                                .take(100)
                                .collect(),
                            valuations: Vec::new(),
                            candidates: Vec::new(),
                            policy_mode: format!("{:?}", runtime.policy_config.mode),
                            policy_decisions: Vec::new(),
                            allocations: Vec::new(),
                            executions: Vec::new(),
                        });
                        return;
                    }
                };
                (claude_vals, true)
            }
        }
    };
    println!("valued {} markets", valuations.len());

    // Override heuristic fair prices with forecast model output when available.
    let valuations = if forecast_fair_prices.is_empty() {
        valuations
    } else {
        let overridden = valuations
            .into_iter()
            .map(|mut v| {
                if let Some(&(fair, conf)) = forecast_fair_prices.get(&v.ticker) {
                    v.fair_prob_yes = fair;
                    v.confidence = conf;
                    v.rationale = format!("forecast-model fair={:.4}", fair);
                }
                v
            })
            .collect();
        overridden
    };

    let valuation_summary = runtime.valuator.last_run_summary();
    let fallback_reason_text = if valuation_summary.fallback_reasons.is_empty() {
        "none".to_string()
    } else {
        valuation_summary.fallback_reasons.join(" | ")
    };
    println!(
        "valuation mode: used_claude={} used_heuristic={} fallback_reasons={}",
        valuation_summary.used_claude, valuation_summary.used_heuristic, fallback_reason_text
    );
    let mut candidates = runtime.valuator.generate_candidates(&valuations);
    let bootstrap_paper_capture_enabled =
        runtime.mode == ExecutionMode::Paper && research_paper_capture_mode_enabled() && force_test_candidate_enabled();
    let mut used_forced_bootstrap_candidate = false;
    if bootstrap_paper_capture_enabled && candidates.is_empty() {
        if let Some(injected) = build_forced_test_candidate(&selected, &valuations) {
            println!(
                "injecting fallback bootstrap paper candidate: ticker={} edge={:.4}",
                injected.ticker, injected.edge_pct
            );
            candidates.insert(0, injected);
            used_forced_bootstrap_candidate = true;
        }
    } else if candidates.is_empty() && force_test_candidate_enabled() {
        if let Some(injected) = build_forced_test_candidate(&selected, &valuations) {
            println!(
                "injecting deterministic test candidate: ticker={} edge={:.4}",
                injected.ticker, injected.edge_pct
            );
            candidates.push(injected);
        }
    }
    if candidates.is_empty() {
        let diagnostics = runtime.valuator.candidate_diagnostics(&valuations, 5);
        eprintln!("no mispricing candidates above threshold");
        eprintln!(
            "candidate diagnostics: strict_threshold={:.4} fallback_threshold={:.4} min_candidates={} adaptive_enabled={} total_cost_prob={:.4} strict_count={} relaxed_count={}",
            diagnostics.strict_threshold,
            diagnostics.fallback_threshold,
            diagnostics.min_candidates,
            diagnostics.adaptive_threshold_enabled,
            diagnostics.total_cost_prob,
            diagnostics.strict_count,
            diagnostics.relaxed_count
        );
        for edge in diagnostics.top_edges {
            eprintln!(
                "top edge: ticker={} raw_edge={:.4} adjusted_edge={:.4} effective_threshold={:.4} confidence={:.2}",
                edge.ticker, edge.raw_edge, edge.adjusted_edge, edge.effective_threshold, edge.confidence
            );
        }
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "no_candidates".to_string(),
            message: Some("no mispricing candidates above threshold".to_string()),
            selected_markets: selected
                .iter()
                .map(|m| artifact_market(m))
                .take(100)
                .collect(),
            valuations: valuations.iter().map(artifact_valuation).collect(),
            candidates: Vec::new(),
            policy_mode: format!("{:?}", runtime.policy_config.mode),
            policy_decisions: Vec::new(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("generated {} candidates", candidates.len());
    println!("top candidate rationale: {}", candidates[0].rationale);

    let mut policy_decision_artifacts = Vec::new();
    let mut policy_decisions_by_key: std::collections::HashMap<(String, String), PolicyDecision> =
        std::collections::HashMap::new();

    if let Some(model) = &runtime.execution_model {
        let market_by_ticker: std::collections::HashMap<_, _> = selected
            .iter()
            .map(|market| (market.ticker.clone(), market))
            .collect();
        let forecast_by_ticker: std::collections::HashMap<_, _> = valuation_inputs
            .iter()
            .map(|input| {
                let feature = build_forecast_feature_row(
                    &input.market,
                    input.enrichment.as_ref(),
                    started_at,
                );
                (input.market.ticker.clone(), feature)
            })
            .collect();
        let shadow_rows: Vec<_> = candidates
            .iter()
            .filter_map(|candidate| {
                let market = market_by_ticker.get(&candidate.ticker)?;
                let forecast = forecast_by_ticker.get(&candidate.ticker)?;
                let feature = build_execution_feature_row_from_forecast(
                    forecast,
                    market,
                    candidate,
                    TimeInForce::Gtc,
                    candidate.observed_price,
                    &ExecutionContext {
                        open_order_count_same_ticker: 0,
                        recent_fill_count_same_ticker: 0,
                        recent_cancel_count_same_ticker: 0,
                        same_event_exposure_notional: 0.0,
                    },
                );
                let estimate = model.predict(&feature);
                Some((feature, estimate))
            })
            .collect();
        if let Some(top) = shadow_rows
            .iter()
            .max_by(|a, b| a.1.fill_prob_5m.total_cmp(&b.1.fill_prob_5m))
        {
            println!(
                "execution shadow: model_version={} rows={} top_fill5m_ticker={} fill5m={:.4} markout5m_bps={:.2}",
                top.1.model_version,
                shadow_rows.len(),
                top.0.ticker,
                top.1.fill_prob_5m,
                top.1.expected_markout_bps_5m
            );
        }
        let borrowed_rows: Vec<_> = shadow_rows.iter().map(|(feature, estimate)| (feature, estimate.clone())).collect();
        if let Err(err) = record_execution_shadow_outputs(&runtime.execution_runtime, &cycle_id, &borrowed_rows) {
            eprintln!("execution shadow warning: {err}");
        }
    }

    if let (Some(forecast_model), Some(execution_model)) = (&runtime.forecast_model, &runtime.execution_model) {
        let market_by_ticker: std::collections::HashMap<_, _> =
            selected.iter().map(|market| (market.ticker.clone(), market)).collect();
        let mut shadow_decisions = Vec::new();
        for input in &valuation_inputs {
            let forecast_feature =
                build_forecast_feature_row(&input.market, input.enrichment.as_ref(), started_at);
            let forecast_output = forecast_model.predict(&forecast_feature);
            if let Some(candidate) = candidates.iter().find(|c| c.ticker == input.market.ticker) {
                let decision = decide_shadow_policy(
                    &runtime.policy_config,
                    execution_model,
                    market_by_ticker
                        .get(&candidate.ticker)
                        .copied()
                        .unwrap_or(&input.market),
                    &forecast_feature,
                    &forecast_output,
                    candidate,
                );
                policy_decision_artifacts.push(artifact_policy_decision(candidate, &decision));
                policy_decisions_by_key.insert(
                    (candidate.ticker.clone(), candidate.outcome_id.clone()),
                    decision.clone(),
                );
                shadow_decisions.push((candidate, decision));
            }
        }
        if let Some((candidate, decision)) = shadow_decisions.iter().max_by(|a, b| {
            a.1.expected_realized_pnl.total_cmp(&b.1.expected_realized_pnl)
        }) {
            println!(
                "policy shadow: mode={:?} rows={} top_ticker={} should_trade={} tif={:?} expected_realized_pnl={:.4}",
                runtime.policy_config.mode,
                shadow_decisions.len(),
                candidate.ticker,
                decision.should_trade,
                decision.time_in_force,
                decision.expected_realized_pnl
            );
        }
        let borrowed: Vec<_> = shadow_decisions.iter().map(|(candidate, decision)| (*candidate, decision)).collect();
        if let Err(err) = record_shadow_decisions(&runtime.policy_config, &cycle_id, &borrowed) {
            eprintln!("policy shadow warning: {err}");
        }
    } else if runtime.policy_config.mode == PolicyMode::Active {
        eprintln!("policy active mode requested but forecast/execution models are not both loaded");
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "policy_unavailable_active".to_string(),
            message: Some("policy active mode requires both forecast and execution models".to_string()),
            selected_markets: selected
                .iter()
                .map(|m| artifact_market(m))
                .take(100)
                .collect(),
            valuations: valuations.iter().map(artifact_valuation).collect(),
            candidates: candidate_artifacts_from(&candidates),
            policy_mode: format!("{:?}", runtime.policy_config.mode),
            policy_decisions: Vec::new(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }

    if runtime.mode == ExecutionMode::Live
        && valuation_summary.used_heuristic
        && !runtime.valuator.allow_heuristic_in_live()
        && (claude_attempted || !matches!(runtime.claude_trigger_mode, ClaudeTriggerMode::OnHeuristicCandidates))
    {
        eprintln!(
            "live cycle aborted: heuristic valuation fallback was used and BOT_ALLOW_HEURISTIC_IN_LIVE is false"
        );
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "valuation_fail_closed_live".to_string(),
            message: Some("heuristic valuation fallback used in live mode".to_string()),
            selected_markets: selected
                .iter()
                .map(|m| artifact_market(m))
                .take(100)
                .collect(),
            valuations: valuations.iter().map(artifact_valuation).collect(),
            candidates: candidate_artifacts_from(&candidates),
            policy_mode: format!("{:?}", runtime.policy_config.mode),
            policy_decisions: policy_decision_artifacts.clone(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }

    let allocation_candidates = if runtime.policy_config.mode == PolicyMode::Active {
        let mut ranked = Vec::new();
        let mut keys: Vec<_> = policy_decisions_by_key
            .iter()
            .filter(|(_, decision)| decision.should_trade)
            .collect();
        keys.sort_by(|a, b| b.1.expected_realized_pnl.total_cmp(&a.1.expected_realized_pnl));
        for ((ticker, outcome_id), decision) in keys {
            let Some(candidate) = candidates
                .iter()
                .find(|c| &c.ticker == ticker && &c.outcome_id == outcome_id)
            else {
                continue;
            };
            let mut c = candidate.clone();
            c.observed_price = decision.limit_price;
            c.edge_pct = decision.expected_realized_pnl.max(0.0) / 100.0;
            c.confidence = decision.expected_fill_prob.clamp(0.0, 1.0);
            c.fair_price = decision.chosen_fair_price;
            c.rationale = format!(
                "{} [policy_rank tif={:?} erpnl={:.4}]",
                c.rationale, decision.time_in_force, decision.expected_realized_pnl
            );
            ranked.push(c);
        }
        ranked
    } else {
        candidates.clone()
    };
    let candidate_artifacts: Vec<ArtifactCandidate> = candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            artifact_candidate(
                c,
                Some(idx),
                policy_decisions_by_key
                    .get(&(c.ticker.clone(), c.outcome_id.clone()))
                    .and_then(|_| {
                        allocation_candidates
                            .iter()
                            .position(|pc| pc.ticker == c.ticker && pc.outcome_id == c.outcome_id)
                    }),
            )
        })
        .collect();
    let allocations = if used_forced_bootstrap_candidate {
        allocation_candidates
            .first()
            .cloned()
            .map(|candidate| {
                let fraction = runtime.allocator.min_fraction_per_trade().max(0.01);
                AllocatedTrade {
                    candidate,
                    bankroll_fraction: fraction,
                    notional: runtime.bankroll * fraction,
                }
            })
            .into_iter()
            .collect()
    } else {
        runtime.allocator.allocate(runtime.bankroll, allocation_candidates)
    };
    if allocations.is_empty() {
        eprintln!("allocator produced no trades");
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "no_allocations".to_string(),
            message: Some("allocator produced no trades".to_string()),
            selected_markets: selected
                .iter()
                .map(|m| artifact_market(m))
                .take(100)
                .collect(),
            valuations: valuations.iter().map(artifact_valuation).collect(),
            candidates: candidate_artifacts,
            policy_mode: format!("{:?}", runtime.policy_config.mode),
            policy_decisions: policy_decision_artifacts.clone(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    let mut guarded_allocations = if used_forced_bootstrap_candidate {
        println!("bootstrap paper capture: bypassing allocator/risk guards for one fallback research execution");
        allocations
    } else {
        let ticker_risk = load_ticker_risk_state_from_journal();
        let max_notional_per_ticker = max_notional_per_ticker_from_env();
        let reentry_cooldown_secs = reentry_cooldown_secs_from_env() as i64;
        let invalid_param_cooldown_secs = invalid_param_cooldown_secs_from_env();
        let now = Utc::now();
        let mut guarded_allocations = Vec::new();
        for a in allocations {
            if let Some(r) = ticker_risk.get(&a.candidate.ticker) {
                if r.open_notional_estimate + a.notional > max_notional_per_ticker {
                    eprintln!(
                        "risk guard: skipping {} due to per-ticker notional cap (current={:.2} + proposed={:.2} > cap={:.2})",
                        a.candidate.ticker, r.open_notional_estimate, a.notional, max_notional_per_ticker
                    );
                    continue;
                }
                if let Some(last_fill) = r.last_fill_ts {
                    let age_secs = (now - last_fill).num_seconds();
                    if age_secs >= 0 && age_secs < reentry_cooldown_secs {
                        eprintln!(
                            "risk guard: skipping {} due to re-entry cooldown (age={}s < cooldown={}s)",
                            a.candidate.ticker, age_secs, reentry_cooldown_secs
                        );
                        continue;
                    }
                }
                if let Some(last_invalid) = r.last_invalid_parameters_ts {
                    let age_secs = (now - last_invalid).num_seconds();
                    if age_secs >= 0 && age_secs < invalid_param_cooldown_secs {
                        eprintln!(
                            "risk guard: skipping {} due to invalid-parameters cooldown (age={}s < cooldown={}s)",
                            a.candidate.ticker, age_secs, invalid_param_cooldown_secs
                        );
                        continue;
                    }
                }
                let key = order_key(&a.candidate.outcome_id, a.candidate.side);
                if r.open_order_keys.contains(&key) {
                    eprintln!(
                        "risk guard: skipping {} due to existing open order on same outcome/side ({})",
                        a.candidate.ticker, key
                    );
                    continue;
                }
            }
            guarded_allocations.push(a);
        }
        guarded_allocations
    };
    if runtime.policy_config.mode == PolicyMode::Active {
        let mut policy_guarded = Vec::new();
        for mut allocated in guarded_allocations {
            let key = (
                allocated.candidate.ticker.clone(),
                allocated.candidate.outcome_id.clone(),
            );
            let Some(decision) = policy_decisions_by_key.get(&key) else {
                continue;
            };
            if !decision.should_trade {
                continue;
            }
            allocated.notional *= decision.size_multiplier.max(0.0);
            allocated.bankroll_fraction *= decision.size_multiplier.max(0.0);
            if allocated.notional > 0.0
                && allocated.bankroll_fraction >= runtime.allocator.min_fraction_per_trade()
            {
                policy_guarded.push(allocated);
            }
        }
        guarded_allocations = policy_guarded;
    }
    if guarded_allocations.is_empty() {
        eprintln!("allocator produced no trades after risk guard");
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "no_allocations".to_string(),
            message: Some("allocator produced no trades after risk guard".to_string()),
            selected_markets: selected
                .iter()
                .map(|m| artifact_market(m))
                .take(100)
                .collect(),
            valuations: valuations.iter().map(artifact_valuation).collect(),
            candidates: candidate_artifacts,
            policy_mode: format!("{:?}", runtime.policy_config.mode),
            policy_decisions: policy_decision_artifacts.clone(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("allocator selected {} trades", guarded_allocations.len());
    let title_by_ticker: std::collections::HashMap<String, String> = selected
        .iter()
        .map(|m| (m.ticker.clone(), m.title.clone()))
        .collect();
    let market_by_ticker: std::collections::HashMap<String, &ScannedMarket> = selected
        .iter()
        .map(|m| (m.ticker.clone(), m))
        .collect();

    let mut execution_artifacts = Vec::new();
    for allocated in guarded_allocations {
        let mut signal = runtime.valuator.candidate_to_signal(&allocated.candidate);
        if runtime.policy_config.mode == PolicyMode::Active {
            if let Some(decision) = policy_decisions_by_key.get(&(
                allocated.candidate.ticker.clone(),
                allocated.candidate.outcome_id.clone(),
            )) {
                signal.observed_price = decision.limit_price;
            }
        }
        match runtime.mapper.resolve_market_ticker(&signal.market_id).await {
            Ok(ticker) => signal.market_id = ticker,
            Err(err) if runtime.resolution_mode == ResolutionMode::BestEffort => {
                if runtime.mode == ExecutionMode::Live {
                    eprintln!("market resolution failed in live mode (skipping trade): {err}");
                    execution_artifacts.push(ArtifactExecution {
                        source_ticker: allocated.candidate.ticker.clone(),
                        resolved_market_id: signal.market_id.clone(),
                        bankroll_fraction: allocated.bankroll_fraction,
                        notional: allocated.notional,
                        result: "resolution_failed_live_skip".to_string(),
                        error: Some(err.to_string()),
                        report: None,
                    });
                    continue;
                }
                eprintln!("market resolution warning (best-effort): {err}");
            }
            Err(err) => {
                eprintln!("market resolution failed (strict mode): {err}");
                execution_artifacts.push(ArtifactExecution {
                    source_ticker: allocated.candidate.ticker.clone(),
                    resolved_market_id: signal.market_id.clone(),
                    bankroll_fraction: allocated.bankroll_fraction,
                    notional: allocated.notional,
                    result: "resolution_failed".to_string(),
                    error: Some(err.to_string()),
                    report: None,
                });
                continue;
            }
        }

        let market_ptr = market_by_ticker.get(&allocated.candidate.ticker).cloned();
        match runtime.engine.execute_signal(&signal, allocated.notional, market_ptr).await {
            Ok(report) => {
                let lookup_hint = kalshi_lookup_hint(&signal.market_id);
                let market_title = title_by_ticker
                    .get(&signal.market_id)
                    .or_else(|| title_by_ticker.get(&allocated.candidate.ticker))
                    .cloned()
                    .unwrap_or_else(|| "<title unavailable>".to_string());
                let status_label = report_status_label(&report.status);
                if matches!(report.status, OrderStatus::Filled | OrderStatus::PartiallyFilled)
                    && report.filled_qty > 0.0
                {
                    println!(
                        "order {}: ticker={} title=\"{}\" outcome={} side={:?} fraction={:.4} target_notional={:.2} filled_qty={:.4} fill_price={:?} tif={:?} final_status={:?} kalshi_hint={}",
                        status_label,
                        allocated.candidate.ticker,
                        market_title,
                        signal.outcome_id,
                        signal.side,
                        allocated.bankroll_fraction,
                        allocated.notional,
                        report.filled_qty,
                        report.avg_fill_price,
                        report.submitted_time_in_force,
                        report.status,
                        lookup_hint
                    );
                } else if matches!(report.status, OrderStatus::Filled | OrderStatus::PartiallyFilled) {
                    eprintln!(
                        "order {} (zero_qty anomaly): ticker={} title=\"{}\" outcome={} side={:?} fraction={:.4} target_notional={:.2} filled_qty={:.4} fill_price={:?} tif={:?} final_status={:?} kalshi_hint={}",
                        status_label,
                        allocated.candidate.ticker,
                        market_title,
                        signal.outcome_id,
                        signal.side,
                        allocated.bankroll_fraction,
                        allocated.notional,
                        report.filled_qty,
                        report.avg_fill_price,
                        report.submitted_time_in_force,
                        report.status,
                        lookup_hint
                    );
                } else {
                    eprintln!(
                        "order {} (no fill): ticker={} title=\"{}\" outcome={} side={:?} fraction={:.4} target_notional={:.2} filled_qty={:.4} fill_price={:?} tif={:?} final_status={:?} kalshi_hint={}",
                        status_label,
                        allocated.candidate.ticker,
                        market_title,
                        signal.outcome_id,
                        signal.side,
                        allocated.bankroll_fraction,
                        allocated.notional,
                        report.filled_qty,
                        report.avg_fill_price,
                        report.submitted_time_in_force,
                        report.status,
                        lookup_hint
                    );
                }
                execution_artifacts.push(ArtifactExecution {
                    source_ticker: allocated.candidate.ticker.clone(),
                    resolved_market_id: signal.market_id.clone(),
                    bankroll_fraction: allocated.bankroll_fraction,
                    notional: allocated.notional,
                    result: status_label.to_string(),
                    error: None,
                    report: Some(report),
                });
            }
            Err(err) => {
                eprintln!("execution failed for {}: {err}", allocated.candidate.ticker);
                execution_artifacts.push(ArtifactExecution {
                    source_ticker: allocated.candidate.ticker.clone(),
                    resolved_market_id: signal.market_id.clone(),
                    bankroll_fraction: allocated.bankroll_fraction,
                    notional: allocated.notional,
                    result: "execution_failed".to_string(),
                    error: Some(err.to_string()),
                    report: None,
                });
            }
        }
    }

    persist_cycle_artifact(CycleArtifact {
        started_at,
        finished_at: Utc::now(),
        status: "completed".to_string(),
        message: None,
        selected_markets: selected
            .iter()
            .map(|m| artifact_market(m))
            .take(100)
            .collect(),
        valuations: valuations.iter().map(artifact_valuation).collect(),
        candidates: candidate_artifacts,
        policy_mode: format!("{:?}", runtime.policy_config.mode),
        policy_decisions: policy_decision_artifacts,
        allocations: execution_artifacts
            .iter()
            .map(|e| ArtifactAllocation {
                ticker: e.source_ticker.clone(),
                bankroll_fraction: e.bankroll_fraction,
                notional: e.notional,
            })
            .collect(),
        executions: execution_artifacts,
    });
}

fn execution_mode_from_env() -> ExecutionMode {
    match std::env::var("BOT_EXECUTION_MODE")
        .unwrap_or_else(|_| "paper".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "live" => ExecutionMode::Live,
        _ => ExecutionMode::Paper,
    }
}

fn run_once_mode() -> bool {
    matches!(
        std::env::var("BOT_RUN_ONCE")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn replay_mode_enabled() -> bool {
    matches!(
        std::env::var("BOT_RUN_REPLAY")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn research_capture_only_mode_enabled() -> bool {
    matches!(
        std::env::var("BOT_RUN_RESEARCH_CAPTURE_ONLY")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn research_paper_capture_mode_enabled() -> bool {
    matches!(
        std::env::var("BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn run_summary_only_mode() -> bool {
    matches!(
        std::env::var("BOT_RUN_SUMMARY_ONLY")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn exchange_backend_from_env() -> String {
    std::env::var("BOT_EXCHANGE_BACKEND")
        .unwrap_or_else(|_| "kalshi".to_string())
        .to_ascii_lowercase()
}

fn cycle_seconds_from_env() -> u64 {
    std::env::var("BOT_CYCLE_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(600)
        .max(5)
}

fn claude_every_n_cycles_from_env() -> u64 {
    std::env::var("BOT_CLAUDE_EVERY_N_CYCLES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1)
        .max(1)
}

fn should_use_claude_for_cycle(cycle_number: u64, every_n_cycles: u64) -> bool {
    every_n_cycles <= 1 || ((cycle_number - 1) % every_n_cycles == 0)
}

fn claude_trigger_mode_from_env() -> ClaudeTriggerMode {
    match std::env::var("BOT_CLAUDE_TRIGGER_MODE")
        .unwrap_or_else(|_| "cadence".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "never" | "off" | "disabled" => ClaudeTriggerMode::Never,
        "on_viable_markets" | "on_selected_markets" => ClaudeTriggerMode::OnViableMarkets,
        "on_heuristic_candidates" => ClaudeTriggerMode::OnHeuristicCandidates,
        _ => ClaudeTriggerMode::Cadence,
    }
}

fn engine_config_from_env() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    if let Ok(v) = std::env::var("BOT_MIN_EDGE_PCT") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.min_edge_pct = parsed.clamp(0.0, 1.0);
        }
    }
    if let Ok(v) = std::env::var("BOT_MAX_DAILY_LOSS") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.max_daily_loss = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_MAX_OPEN_EXPOSURE") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.max_open_exposure_notional = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_MAX_ORDERS_PER_MIN") {
        if let Ok(parsed) = v.parse::<usize>() {
            cfg.max_orders_per_minute = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_STATE_PATH") {
        cfg.state_path = v;
    }
    if let Ok(v) = std::env::var("BOT_JOURNAL_PATH") {
        cfg.journal_path = v;
    }
    if let Ok(v) = std::env::var("BOT_EXEC_POLICY") {
        cfg.execution_policy = v.to_ascii_lowercase();
    }
    if let Ok(v) = std::env::var("BOT_HYBRID_IOC_FRACTION") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.hybrid_ioc_fraction = parsed.clamp(0.05, 1.0);
        }
    }
    cfg
}

fn smoke_test_enabled() -> bool {
    matches!(
        std::env::var("BOT_RUN_SMOKE_TEST")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn valuation_market_limit_from_env() -> usize {
    std::env::var("BOT_VALUATION_MARKETS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(250)
}

fn enrichment_market_limit_from_env() -> usize {
    std::env::var("BOT_ENRICHMENT_MARKETS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100)
}

fn bankroll_from_env() -> f64 {
    std::env::var("BOT_BANKROLL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(10_000.0)
}

fn allocation_config_from_env() -> AllocationConfig {
    let mut cfg = AllocationConfig::default();
    if let Ok(v) = std::env::var("BOT_MAX_TRADES_PER_CYCLE") {
        if let Ok(parsed) = v.parse::<usize>() {
            cfg.max_trades_per_cycle = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_MAX_FRACTION_PER_TRADE") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.max_fraction_per_trade = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_MAX_TOTAL_FRACTION_PER_CYCLE") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.max_total_fraction_per_cycle = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_MIN_FRACTION_PER_TRADE") {
        if let Ok(parsed) = v.parse::<f64>() {
            cfg.min_fraction_per_trade = parsed;
        }
    }
    if let Ok(v) = std::env::var("BOT_ENFORCE_EVENT_MUTEX") {
        cfg.enforce_event_mutex = matches!(
            v.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        );
    }
    cfg
}

fn force_test_candidate_enabled() -> bool {
    matches!(
        std::env::var("BOT_FORCE_TEST_CANDIDATE")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn cycle_artifacts_enabled() -> bool {
    !matches!(
        std::env::var("BOT_CYCLE_ARTIFACTS_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no"
    )
}

fn cycle_artifacts_dir() -> String {
    std::env::var("BOT_CYCLE_ARTIFACTS_DIR").unwrap_or_else(|_| "var/cycles".to_string())
}

fn journal_path_from_env() -> String {
    std::env::var("BOT_JOURNAL_PATH").unwrap_or_else(|_| "var/logs/trade_journal.jsonl".to_string())
}

fn max_notional_per_ticker_from_env() -> f64 {
    std::env::var("BOT_MAX_NOTIONAL_PER_TICKER")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(500.0)
        .max(1.0)
}

fn reentry_cooldown_secs_from_env() -> u64 {
    std::env::var("BOT_REENTRY_COOLDOWN_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3600)
        .max(0)
}

fn invalid_param_cooldown_secs_from_env() -> i64 {
    std::env::var("BOT_INVALID_PARAM_COOLDOWN_SECS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(21_600)
        .max(0)
}

fn kalshi_lookup_hint(ticker: &str) -> String {
    format!("https://kalshi.com/search?query={}", ticker)
}

fn report_status_label(status: &OrderStatus) -> &'static str {
    match status {
        OrderStatus::New => "accepted",
        OrderStatus::PartiallyFilled => "partially_filled",
        OrderStatus::Filled => "filled",
        OrderStatus::Canceled => "canceled",
        OrderStatus::Rejected => "rejected",
    }
}

fn log_position_marks_from_journal(scanned: &[data::market_scanner::ScannedMarket]) {
    let path = journal_path_from_env();
    let journal_path = Path::new(&path);
    if !journal_path.exists() {
        return;
    }
    let text = match fs::read_to_string(journal_path) {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut intents: std::collections::HashMap<String, IntentRecordLite> = std::collections::HashMap::new();
    let mut positions: std::collections::HashMap<(String, String), PositionMarkLite> = std::collections::HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let entry = match serde_json::from_str::<JournalLineLite>(line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.event == "order_intent" {
            if let Ok(payload) = serde_json::from_value::<IntentPayloadLite>(entry.payload) {
                intents.insert(
                    payload.order.client_order_id.clone(),
                    IntentRecordLite {
                        order: payload.order,
                        mode: payload.mode,
                    },
                );
            }
            continue;
        }
        if entry.event != "order_report" {
            continue;
        }
        let payload = match serde_json::from_value::<ReportPayloadLite>(entry.payload) {
            Ok(p) => p,
            Err(_) => continue,
        };
        if !matches!(payload.report.status, OrderStatus::Filled | OrderStatus::PartiallyFilled) {
            continue;
        }
        let Some(intent) = intents.get(&payload.report.client_order_id) else {
            continue;
        };
        if !intent.is_live() {
            continue;
        }
        let qty = payload.report.filled_qty.max(0.0);
        let price = payload.report.avg_fill_price.unwrap_or(0.0).max(0.0);
        if qty <= 0.0 || price <= 0.0 {
            continue;
        }
        let key = (intent.order.market_id.clone(), intent.order.outcome_id.clone());
        let pos = positions.entry(key).or_default();
        match intent.order.side {
            execution::types::Side::Buy => {
                pos.net_qty += qty;
                pos.cost_basis += qty * price;
            }
            execution::types::Side::Sell => {
                pos.net_qty -= qty;
                pos.cost_basis -= qty * price;
            }
        }
    }

    if positions.is_empty() {
        return;
    }
    let quote_by_ticker: std::collections::HashMap<&str, &data::market_scanner::ScannedMarket> =
        scanned.iter().map(|m| (m.ticker.as_str(), m)).collect();
    let mut rows = Vec::new();
    let mut total_unrealized = 0.0;
    let mut marked_count = 0usize;
    for ((ticker, outcome), pos) in positions {
        if pos.net_qty <= 0.0 {
            continue;
        }
        let avg_entry = pos.cost_basis / pos.net_qty;
        let mark = quote_by_ticker
            .get(ticker.as_str())
            .and_then(|m| midpoint_prob_from_market(m))
            .map(|yes_mid| if outcome.eq_ignore_ascii_case("no") { 1.0 - yes_mid } else { yes_mid });
        let unrealized = mark.map(|m| (m - avg_entry) * pos.net_qty);
        if let Some(u) = unrealized {
            total_unrealized += u;
            marked_count += 1;
        }
        rows.push((ticker, outcome, pos.net_qty, avg_entry, mark, unrealized));
    }
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| {
        let ua = a.5.unwrap_or(f64::NEG_INFINITY);
        let ub = b.5.unwrap_or(f64::NEG_INFINITY);
        ub.total_cmp(&ua)
    });
    println!(
        "position marks: open_positions={} marked_positions={} total_unrealized={:.4}",
        rows.len(),
        marked_count,
        total_unrealized
    );
    for (ticker, outcome, qty, entry, mark, pnl) in rows.into_iter().take(5) {
        println!(
            "position: ticker={} outcome={} qty={:.4} entry={:.4} mark={:?} unrealized={:?} kalshi_hint={}",
            ticker,
            outcome,
            qty,
            entry,
            mark.map(|v| format!("{v:.4}")),
            pnl.map(|v| format!("{v:.4}")),
            kalshi_lookup_hint(&ticker)
        );
    }
}

fn load_ticker_risk_state_from_journal() -> std::collections::HashMap<String, TickerRiskState> {
    let path = journal_path_from_env();
    let journal_path = Path::new(&path);
    if !journal_path.exists() {
        return std::collections::HashMap::new();
    }
    let text = match fs::read_to_string(journal_path) {
        Ok(t) => t,
        Err(_) => return std::collections::HashMap::new(),
    };
    let mut intents: std::collections::HashMap<String, IntentRecordLite> = std::collections::HashMap::new();
    let mut state: std::collections::HashMap<String, TickerRiskState> = std::collections::HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let entry = match serde_json::from_str::<JournalLineLite>(line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.event == "order_intent" {
            if let Ok(payload) = serde_json::from_value::<IntentPayloadLite>(entry.payload) {
                intents.insert(
                    payload.order.client_order_id.clone(),
                    IntentRecordLite {
                        order: payload.order,
                        mode: payload.mode,
                    },
                );
            }
            continue;
        }
        if entry.event == "order_error" {
            if let Ok(payload) = serde_json::from_value::<OrderErrorPayloadLite>(entry.payload) {
                if payload.error.to_ascii_lowercase().contains("invalid_parameters") {
                    let row = state.entry(payload.market_id).or_default();
                    row.last_invalid_parameters_ts = Some(entry.ts);
                }
            }
            continue;
        }
        if entry.event != "order_report" {
            continue;
        }
        let payload = match serde_json::from_value::<ReportPayloadLite>(entry.payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(intent) = intents.get(&payload.report.client_order_id) else {
            continue;
        };
        if !intent.is_live() {
            continue;
        }
        let row = state.entry(intent.order.market_id.clone()).or_default();
        let key = order_key(&intent.order.outcome_id, intent.order.side);
        match payload.report.status {
            OrderStatus::New | OrderStatus::PartiallyFilled => {
                row.open_order_keys.insert(key);
            }
            OrderStatus::Filled => {
                row.open_order_keys.remove(&key);
                let notional = payload.report.filled_qty.max(0.0)
                    * payload.report.avg_fill_price.unwrap_or(0.0).max(0.0);
                if notional > 0.0 {
                    row.open_notional_estimate += notional;
                    if row
                        .last_fill_ts
                        .map(|t| payload.report.updated_at > t)
                        .unwrap_or(true)
                    {
                        row.last_fill_ts = Some(payload.report.updated_at);
                    }
                }
            }
            OrderStatus::Canceled | OrderStatus::Rejected => {
                row.open_order_keys.remove(&key);
            }
        }
    }
    state
}

#[derive(Debug, Deserialize)]
struct JournalLineLite {
    ts: DateTime<Utc>,
    event: String,
    payload: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone)]
struct IntentOrderLite {
    client_order_id: String,
    market_id: String,
    outcome_id: String,
    side: execution::types::Side,
}

#[derive(Debug, Clone)]
struct IntentRecordLite {
    order: IntentOrderLite,
    mode: Option<String>,
}

impl IntentRecordLite {
    fn is_live(&self) -> bool {
        matches!(
            self.mode
                .as_deref()
                .map(|value| value.eq_ignore_ascii_case("Live"))
                .unwrap_or(false),
            true
        )
    }
}

#[derive(Debug, Deserialize)]
struct IntentPayloadLite {
    order: IntentOrderLite,
    mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReportPayloadLite {
    report: execution::types::ExecutionReport,
}

#[derive(Debug, Deserialize)]
struct OrderErrorPayloadLite {
    market_id: String,
    error: String,
}

#[derive(Debug, Default)]
struct PositionMarkLite {
    net_qty: f64,
    cost_basis: f64,
}

#[derive(Debug, Default)]
struct TickerRiskState {
    open_notional_estimate: f64,
    last_fill_ts: Option<DateTime<Utc>>,
    last_invalid_parameters_ts: Option<DateTime<Utc>>,
    open_order_keys: std::collections::HashSet<String>,
}

fn order_key(outcome_id: &str, side: execution::types::Side) -> String {
    format!("{}:{:?}", outcome_id.to_ascii_lowercase(), side)
}

/// Flat edge bump used when the model is unavailable and a heuristic candidate
/// must be constructed. Exposed as a named constant so the value is easy to find
/// and adjust without grepping for magic numbers.
const HEURISTIC_EDGE_FALLBACK: f64 = 0.03;

fn build_forced_test_candidate(
    selected: &[data::market_scanner::ScannedMarket],
    valuations: &[MarketValuation],
) -> Option<CandidateTrade> {
    if let Some(v) = valuations.first() {
        let observed_yes = v.market_mid_prob_yes.clamp(0.01, 0.99);
        if observed_yes <= 0.95 {
            let fair = (observed_yes + HEURISTIC_EDGE_FALLBACK).clamp(0.01, 0.99);
            return Some(CandidateTrade {
                ticker: v.ticker.clone(),
                side: execution::types::Side::Buy,
                outcome_id: "yes".to_string(),
                fair_price: fair,
                observed_price: observed_yes,
                edge_pct: (fair - observed_yes).abs().max(0.01),
                confidence: 0.70,
                rationale: "forced deterministic test candidate".to_string(),
            });
        }
        let observed_no = (1.0 - observed_yes).clamp(0.01, 0.99);
        let fair_no = (observed_no + HEURISTIC_EDGE_FALLBACK).clamp(0.01, 0.99);
        return Some(CandidateTrade {
            ticker: v.ticker.clone(),
            side: execution::types::Side::Buy,
            outcome_id: "no".to_string(),
            fair_price: fair_no,
            observed_price: observed_no,
            edge_pct: (fair_no - observed_no).abs().max(0.01),
            confidence: 0.70,
            rationale: "forced deterministic test candidate".to_string(),
        });
    }

    let first = selected.first()?;
    let observed_yes = midpoint_prob_from_market(first)?;
    let fair_yes = (observed_yes + HEURISTIC_EDGE_FALLBACK).clamp(0.01, 0.99);
    Some(CandidateTrade {
        ticker: first.ticker.clone(),
        side: execution::types::Side::Buy,
        outcome_id: "yes".to_string(),
        fair_price: fair_yes,
        observed_price: observed_yes,
        edge_pct: (fair_yes - observed_yes).abs().max(0.01),
        confidence: 0.70,
        rationale: "forced deterministic test candidate".to_string(),
    })
}

fn midpoint_prob_from_market(market: &data::market_scanner::ScannedMarket) -> Option<f64> {
    let bid = market.yes_bid_cents?;
    let ask = market.yes_ask_cents?;
    if ask < bid {
        return None;
    }
    Some((((bid + ask) / 2.0) / 100.0).clamp(0.01, 0.99))
}

fn persist_cycle_artifact(artifact: CycleArtifact) {
    if !cycle_artifacts_enabled() {
        return;
    }
    let dir = cycle_artifacts_dir();
    if let Err(err) = fs::create_dir_all(&dir) {
        eprintln!("cycle artifact warning: failed to create dir {dir}: {err}");
        return;
    }
    let filename = format!(
        "{}_{}.json",
        artifact.started_at.format("%Y%m%dT%H%M%S"),
        artifact.finished_at.timestamp_subsec_millis()
    );
    let path = Path::new(&dir).join(filename);
    let json = match serde_json::to_vec_pretty(&artifact) {
        Ok(j) => j,
        Err(err) => {
            eprintln!("cycle artifact warning: failed to serialize artifact: {err}");
            return;
        }
    };
    if let Err(err) = fs::write(&path, json) {
        eprintln!(
            "cycle artifact warning: failed to write {}: {}",
            path.display(),
            err
        );
    }
}

fn artifact_market(m: &data::market_scanner::ScannedMarket) -> ArtifactMarket {
    ArtifactMarket {
        ticker: m.ticker.clone(),
        title: m.title.clone(),
        yes_bid_cents: m.yes_bid_cents,
        yes_ask_cents: m.yes_ask_cents,
        volume: m.volume,
    }
}

fn artifact_valuation(v: &MarketValuation) -> ArtifactValuation {
    ArtifactValuation {
        ticker: v.ticker.clone(),
        fair_prob_yes: v.fair_prob_yes,
        market_mid_prob_yes: v.market_mid_prob_yes,
        market_volume: v.market_volume,
        spread_cents: v.spread_cents,
        confidence: v.confidence,
        rationale: v.rationale.clone(),
        stale_after: v.stale_after,
    }
}

fn artifact_candidate(
    c: &CandidateTrade,
    legacy_rank: Option<usize>,
    policy_rank: Option<usize>,
) -> ArtifactCandidate {
    ArtifactCandidate {
        ticker: c.ticker.clone(),
        outcome_id: c.outcome_id.clone(),
        side: c.side,
        fair_price: c.fair_price,
        observed_price: c.observed_price,
        edge_pct: c.edge_pct,
        confidence: c.confidence,
        legacy_rank,
        policy_rank,
        rank_delta: match (legacy_rank, policy_rank) {
            (Some(l), Some(p)) => Some(l as i64 - p as i64),
            _ => None,
        },
        rationale: c.rationale.clone(),
    }
}

fn candidate_artifacts_from(candidates: &[CandidateTrade]) -> Vec<ArtifactCandidate> {
    candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| artifact_candidate(c, Some(idx), None))
        .collect()
}

fn artifact_policy_decision(candidate: &CandidateTrade, decision: &PolicyDecision) -> ArtifactPolicyDecision {
    ArtifactPolicyDecision {
        ticker: candidate.ticker.clone(),
        outcome_id: candidate.outcome_id.clone(),
        legacy_edge_pct: candidate.edge_pct,
        legacy_confidence: candidate.confidence,
        should_trade: decision.should_trade,
        chosen_limit_price: decision.limit_price,
        chosen_time_in_force: decision.time_in_force,
        size_multiplier: decision.size_multiplier,
        expected_fill_prob: decision.expected_fill_prob,
        expected_gross_edge: decision.expected_gross_edge,
        expected_realized_pnl: decision.expected_realized_pnl,
        rejection_reason: decision.rejection_reason.clone(),
        rationale: decision.rationale.clone(),
    }
}

#[derive(Debug, Serialize)]
struct CycleArtifact {
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
    status: String,
    message: Option<String>,
    selected_markets: Vec<ArtifactMarket>,
    valuations: Vec<ArtifactValuation>,
    candidates: Vec<ArtifactCandidate>,
    policy_mode: String,
    policy_decisions: Vec<ArtifactPolicyDecision>,
    allocations: Vec<ArtifactAllocation>,
    executions: Vec<ArtifactExecution>,
}

#[derive(Debug, Serialize)]
struct ArtifactMarket {
    ticker: String,
    title: String,
    yes_bid_cents: Option<f64>,
    yes_ask_cents: Option<f64>,
    volume: f64,
}

#[derive(Debug, Serialize)]
struct ArtifactValuation {
    ticker: String,
    fair_prob_yes: f64,
    market_mid_prob_yes: f64,
    market_volume: f64,
    spread_cents: Option<f64>,
    confidence: f64,
    rationale: String,
    stale_after: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct ArtifactCandidate {
    ticker: String,
    outcome_id: String,
    side: execution::types::Side,
    fair_price: f64,
    observed_price: f64,
    edge_pct: f64,
    confidence: f64,
    legacy_rank: Option<usize>,
    policy_rank: Option<usize>,
    rank_delta: Option<i64>,
    rationale: String,
}

#[derive(Debug, Serialize, Clone)]
struct ArtifactPolicyDecision {
    ticker: String,
    outcome_id: String,
    legacy_edge_pct: f64,
    legacy_confidence: f64,
    should_trade: bool,
    chosen_limit_price: f64,
    chosen_time_in_force: execution::types::TimeInForce,
    size_multiplier: f64,
    expected_fill_prob: f64,
    expected_gross_edge: f64,
    expected_realized_pnl: f64,
    rejection_reason: Option<String>,
    rationale: String,
}

#[derive(Debug, Serialize)]
struct ArtifactAllocation {
    ticker: String,
    bankroll_fraction: f64,
    notional: f64,
}

#[derive(Debug, Serialize)]
struct ArtifactExecution {
    source_ticker: String,
    resolved_market_id: String,
    bankroll_fraction: f64,
    notional: f64,
    result: String,
    error: Option<String>,
    report: Option<execution::types::ExecutionReport>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::models::execution::{ExecutionBucket, ExecutionModelMetrics};
    use crate::models::forecast::{ForecastModelMetrics, RateBucket};

    #[test]
    fn startup_validation_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let state_path = dir.path().join("nested/state/runtime_state.json");
        let journal_path = dir.path().join("nested/logs/trade_journal.jsonl");
        let forecast_model_path = dir.path().join("nested/models/forecast/latest.json");
        let execution_model_path = dir.path().join("nested/models/execution/latest.json");

        let engine_cfg = EngineConfig {
            state_path: state_path.display().to_string(),
            journal_path: journal_path.display().to_string(),
            ..EngineConfig::default()
        };
        let forecast_runtime = ForecastRuntimeConfig {
            model_path: Some(forecast_model_path.clone()),
            shadow_enabled: true,
            shadow_root: dir.path().join("shadow"),
            min_bucket_samples: 5,
        };
        let execution_runtime = ExecutionRuntimeConfig {
            model_path: Some(execution_model_path.clone()),
            shadow_enabled: true,
            shadow_root: dir.path().join("shadow"),
            min_bucket_samples: 5,
        };

        validate_startup_paths(&engine_cfg, &forecast_runtime, &execution_runtime).unwrap();

        assert!(state_path.parent().unwrap().exists());
        assert!(journal_path.parent().unwrap().exists());
        assert!(forecast_model_path.parent().unwrap().exists());
        assert!(execution_model_path.parent().unwrap().exists());
    }

    fn active_policy_config() -> PolicyConfig {
        PolicyConfig {
            mode: PolicyMode::Active,
            shadow_enabled: true,
            shadow_root: PathBuf::from("/tmp"),
            min_expected_realized_pnl: 0.0,
            markout_veto_threshold_bps: -300.0,
            max_actions_per_candidate: 4,
            default_legacy_fallback: true,
            active_max_model_age_hours: 24 * 14,
            active_min_forecast_train_rows: 1_000,
            active_min_execution_train_rows: 100,
            active_min_execution_live_real_rows: 25,
            active_require_live_real: true,
            active_min_shadow_decisions: 50,
            active_shadow_lookback_days: 7,
            active_min_shadow_mean_erpnl: -200.0,
        }
    }

    fn make_policy_config_for_shadow_test(shadow_root: &std::path::Path) -> PolicyConfig {
        PolicyConfig {
            shadow_root: shadow_root.to_path_buf(),
            active_require_live_real: false,
            active_min_shadow_decisions: 2,
            active_shadow_lookback_days: 30,
            ..active_policy_config()
        }
    }

    fn write_shadow_records(dir: &std::path::Path, records: &[(&str, bool, f64)]) {
        use std::io::Write;
        let day_dir = dir.join("policy").join("2026-03-30");
        fs::create_dir_all(&day_dir).unwrap();
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(day_dir.join("policy_shadow.jsonl"))
            .unwrap();
        for (cycle_id, should_trade, erpnl) in records {
            let line = format!(
                r#"{{"cycle_id":"{cycle_id}","recorded_at":"2026-03-30T00:00:00Z","ticker":"KXNBA-X","outcome_id":"yes","legacy_edge_pct":0.1,"legacy_confidence":0.9,"chosen_tif":"Gtc","chosen_limit_price":0.5,"should_trade":{should_trade},"size_multiplier":1.0,"expected_fill_prob":0.5,"expected_gross_edge":0.1,"expected_realized_pnl":{erpnl},"rejection_reason":null,"rationale":"test"}}"#
            );
            writeln!(f, "{line}").unwrap();
        }
    }

    #[test]
    fn shadow_gate_passes_with_sufficient_data() {
        let dir = tempdir().unwrap();
        write_shadow_records(dir.path(), &[("c1", true, 5.0), ("c2", true, 3.0), ("c3", false, 0.0)]);
        let cfg = make_policy_config_for_shadow_test(dir.path());
        assert!(validate_shadow_calibration(&cfg).is_ok());
    }

    #[test]
    fn shadow_gate_fails_too_few_decisions() {
        let dir = tempdir().unwrap();
        write_shadow_records(dir.path(), &[("c1", true, 5.0)]);
        let mut cfg = make_policy_config_for_shadow_test(dir.path());
        cfg.active_min_shadow_decisions = 5;
        let err = validate_shadow_calibration(&cfg).unwrap_err();
        assert!(err.contains("shadow decisions"), "{err}");
    }

    #[test]
    fn shadow_gate_fails_low_mean_erpnl() {
        let dir = tempdir().unwrap();
        write_shadow_records(dir.path(), &[("c1", true, -500.0), ("c2", true, -600.0), ("c3", true, -400.0)]);
        let cfg = make_policy_config_for_shadow_test(dir.path());
        let err = validate_shadow_calibration(&cfg).unwrap_err();
        assert!(err.contains("below threshold"), "{err}");
    }

    fn good_forecast_model() -> ForecastModel {
        ForecastModel::from_artifact(
            ForecastModelArtifact {
                schema_version: "v1".to_string(),
                model_kind: "test".to_string(),
                model_version: "forecast-test".to_string(),
                trained_at: Utc::now(),
                train_rows: 5_000,
                validation_rows: 100,
                test_rows: 100,
                feature_schema_version: "v1".to_string(),
                metrics: ForecastModelMetrics::default(),
                global: RateBucket { positives: 50.0, total: 100.0 },
                vertical: std::collections::HashMap::new(),
                vertical_direction: std::collections::HashMap::new(),
                vertical_entity: std::collections::HashMap::new(),
                vertical_threshold: std::collections::HashMap::new(),
            },
            5,
        )
    }

    fn execution_model_with_live_rows(live_real_rows: usize) -> ExecutionModel {
        ExecutionModel::from_artifact(
            ExecutionModelArtifact {
                schema_version: "v1".to_string(),
                model_kind: "test".to_string(),
                model_version: "execution-test".to_string(),
                trained_at: Utc::now(),
                train_rows: 500,
                validation_rows: 50,
                test_rows: 50,
                included_source_classes: vec!["organic_paper".to_string(), "live_real".to_string()],
                bootstrap_rows: 0,
                organic_paper_rows: 200,
                retroactive_synthetic_rows: 0,
                live_real_rows,
                feature_schema_version: "v1".to_string(),
                metrics: ExecutionModelMetrics::default(),
                global: ExecutionBucket::default(),
                by_vertical: std::collections::HashMap::new(),
                by_vertical_tif: std::collections::HashMap::new(),
                by_vertical_liquidity: std::collections::HashMap::new(),
                by_aggressiveness: std::collections::HashMap::new(),
            },
            5,
        )
    }

    fn active_policy_config_with_shadow(shadow_root: &std::path::Path) -> PolicyConfig {
        PolicyConfig {
            shadow_root: shadow_root.to_path_buf(),
            active_require_live_real: true,
            active_min_shadow_decisions: 0,
            ..active_policy_config()
        }
    }

    #[test]
    fn active_policy_validation_rejects_insufficient_live_real_rows() {
        let dir = tempdir().unwrap();
        // Create empty policy dir so shadow gate passes (min decisions = 0)
        fs::create_dir_all(dir.path().join("policy")).unwrap();
        let cfg = active_policy_config_with_shadow(dir.path());
        let forecast = good_forecast_model();
        let execution = execution_model_with_live_rows(0);

        let err = validate_active_policy_requirements(&cfg, Some(&forecast), Some(&execution))
            .expect_err("expected active policy validation to fail");
        assert!(err.contains("live-real rows"));
    }

    #[test]
    fn active_policy_validation_accepts_sufficient_models() {
        let dir = tempdir().unwrap();
        write_shadow_records(dir.path(), &[("c1", true, 5.0), ("c2", true, 3.0)]);
        let cfg = active_policy_config_with_shadow(dir.path());
        let forecast = good_forecast_model();
        let execution = execution_model_with_live_rows(50);

        validate_active_policy_requirements(&cfg, Some(&forecast), Some(&execution)).unwrap();
    }
}
