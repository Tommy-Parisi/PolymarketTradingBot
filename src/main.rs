use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::time::MissedTickBehavior;

mod data;
mod execution;
mod markets;
mod model;
mod replay;

use data::market_enrichment::{EnrichmentConfig, MarketEnricher};
use data::market_scanner::{KalshiMarketScanner, ScannerConfig};
use execution::client::{ExchangeClient, KalshiClient};
use execution::engine::{ExecutionEngine, ExecutionMode};
use execution::paper_sim::{PaperSimClient, PaperSimConfig};
use execution::types::EngineConfig;
use markets::kalshi_mapper::{resolution_mode_from_env, KalshiMarketMapper, ResolutionMode};
use model::allocator::{AllocationConfig, PortfolioAllocator};
use model::valuation::{CandidateTrade, ClaudeValuationEngine, MarketValuation, ValuationConfig, ValuationInput};

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
}

#[tokio::main]
async fn main() {
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
    let engine = ExecutionEngine::new(client, engine_config_from_env(), mode);
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

async fn run_cycle(runtime: &BotRuntime) {
    let started_at = Utc::now();
    println!("starting cycle");

    let scanned = match runtime.scanner.scan_snapshot_with_deltas().await {
        Ok(markets) => markets,
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
                allocations: Vec::new(),
                executions: Vec::new(),
            });
            return;
        }
    };
    let selected = runtime.scanner.select_for_valuation(scanned);
    if selected.is_empty() {
        eprintln!("no markets selected for valuation");
        persist_cycle_artifact(CycleArtifact {
            started_at,
            finished_at: Utc::now(),
            status: "no_markets".to_string(),
            message: Some("no markets selected for valuation".to_string()),
            selected_markets: Vec::new(),
            valuations: Vec::new(),
            candidates: Vec::new(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("selected {} markets for valuation", selected.len());

    let enrich_count = runtime.enrichment_limit.min(selected.len());
    let enrichments = match runtime.enricher.enrich_batch(&selected[..enrich_count]).await {
        Ok(v) => v,
        Err(err) => {
            eprintln!("enrichment failed: {err}");
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

    let valuation_limit = runtime.valuation_limit.min(selected.len());
    let valuation_inputs: Vec<ValuationInput> = selected
        .iter()
        .take(valuation_limit)
        .map(|m| ValuationInput {
            market: m.clone(),
            enrichment: enrichment_by_ticker.get(&m.ticker).cloned(),
        })
        .collect();

    let valuations = match runtime.valuator.value_markets(&valuation_inputs).await {
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
                allocations: Vec::new(),
                executions: Vec::new(),
            });
            return;
        }
    };
    println!("valued {} markets", valuations.len());
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
    if runtime.mode == ExecutionMode::Live
        && valuation_summary.used_heuristic
        && !runtime.valuator.allow_heuristic_in_live()
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
            candidates: Vec::new(),
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }

    let mut candidates = runtime.valuator.generate_candidates(&valuations);
    if candidates.is_empty() && force_test_candidate_enabled() {
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
            "candidate diagnostics: strict_threshold={:.4} fallback_threshold={:.4} min_candidates={} total_cost_prob={:.4} strict_count={} relaxed_count={}",
            diagnostics.strict_threshold,
            diagnostics.fallback_threshold,
            diagnostics.min_candidates,
            diagnostics.total_cost_prob,
            diagnostics.strict_count,
            diagnostics.relaxed_count
        );
        for edge in diagnostics.top_edges {
            eprintln!(
                "top edge: ticker={} raw_edge={:.4} adjusted_edge={:.4} confidence={:.2}",
                edge.ticker, edge.raw_edge, edge.adjusted_edge, edge.confidence
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
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("generated {} candidates", candidates.len());
    println!("top candidate rationale: {}", candidates[0].rationale);

    let candidate_artifacts: Vec<ArtifactCandidate> = candidates.iter().map(artifact_candidate).collect();
    let allocations = runtime.allocator.allocate(runtime.bankroll, candidates);
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
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("allocator selected {} trades", allocations.len());

    let mut execution_artifacts = Vec::new();
    for allocated in allocations {
        let mut signal = runtime.valuator.candidate_to_signal(&allocated.candidate);
        match runtime.mapper.resolve_market_ticker(&signal.market_id).await {
            Ok(ticker) => signal.market_id = ticker,
            Err(err) if runtime.resolution_mode == ResolutionMode::BestEffort => {
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

        match runtime.engine.execute_signal(&signal, allocated.notional).await {
            Ok(report) => {
                println!(
                    "order executed: ticker={} fraction={:.4} notional={:.2} report={:?}",
                    allocated.candidate.ticker, allocated.bankroll_fraction, allocated.notional, report
                );
                execution_artifacts.push(ArtifactExecution {
                    source_ticker: allocated.candidate.ticker.clone(),
                    resolved_market_id: signal.market_id.clone(),
                    bankroll_fraction: allocated.bankroll_fraction,
                    notional: allocated.notional,
                    result: "executed".to_string(),
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
        .unwrap_or(25)
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

fn build_forced_test_candidate(
    selected: &[data::market_scanner::ScannedMarket],
    valuations: &[MarketValuation],
) -> Option<CandidateTrade> {
    if let Some(v) = valuations.first() {
        let observed_yes = v.market_mid_prob_yes.clamp(0.01, 0.99);
        if observed_yes <= 0.95 {
            let fair = (observed_yes + 0.03).clamp(0.01, 0.99);
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
        let fair_no = (observed_no + 0.03).clamp(0.01, 0.99);
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
    let fair_yes = (observed_yes + 0.03).clamp(0.01, 0.99);
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
        confidence: v.confidence,
        rationale: v.rationale.clone(),
        stale_after: v.stale_after,
    }
}

fn artifact_candidate(c: &CandidateTrade) -> ArtifactCandidate {
    ArtifactCandidate {
        ticker: c.ticker.clone(),
        outcome_id: c.outcome_id.clone(),
        side: c.side,
        fair_price: c.fair_price,
        observed_price: c.observed_price,
        edge_pct: c.edge_pct,
        confidence: c.confidence,
        rationale: c.rationale.clone(),
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
