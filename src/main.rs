use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::time::MissedTickBehavior;

mod data;
mod execution;
mod markets;
mod model;
mod replay;

use data::market_enrichment::{EnrichmentConfig, MarketEnricher};
use data::market_scanner::{KalshiMarketScanner, ScannerConfig};
use execution::client::{ExchangeClient, KalshiClient};
use execution::engine::{ExecutionEngine, ExecutionMode, summarize_performance_paths};
use execution::paper_sim::{PaperSimClient, PaperSimConfig};
use execution::types::{EngineConfig, OrderStatus};
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
    claude_every_n_cycles: u64,
    claude_trigger_mode: ClaudeTriggerMode,
    cycle_counter: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeTriggerMode {
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
        claude_every_n_cycles: claude_every_n_cycles_from_env(),
        claude_trigger_mode: claude_trigger_mode_from_env(),
        cycle_counter: AtomicU64::new(0),
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
    let cycle_number = runtime.cycle_counter.fetch_add(1, Ordering::Relaxed) + 1;
    let cadence_use_claude = should_use_claude_for_cycle(cycle_number, runtime.claude_every_n_cycles);
    let started_at = Utc::now();
    println!(
        "starting cycle #{} (claude_trigger_mode={:?} cadence_claude_enabled={} cadence={})",
        cycle_number, runtime.claude_trigger_mode, cadence_use_claude, runtime.claude_every_n_cycles
    );

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
    log_position_marks_from_journal(&scanned);
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

    let (valuations, claude_attempted) = match runtime.claude_trigger_mode {
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
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }
    println!("generated {} candidates", candidates.len());
    println!("top candidate rationale: {}", candidates[0].rationale);

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
            allocations: Vec::new(),
            executions: Vec::new(),
        });
        return;
    }

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

    let mut execution_artifacts = Vec::new();
    for allocated in guarded_allocations {
        let mut signal = runtime.valuator.candidate_to_signal(&allocated.candidate);
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

        match runtime.engine.execute_signal(&signal, allocated.notional).await {
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
    let mut intents: std::collections::HashMap<String, IntentOrderLite> = std::collections::HashMap::new();
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
                intents.insert(payload.order.client_order_id.clone(), payload.order);
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
        let qty = payload.report.filled_qty.max(0.0);
        let price = payload.report.avg_fill_price.unwrap_or(0.0).max(0.0);
        if qty <= 0.0 || price <= 0.0 {
            continue;
        }
        let key = (intent.market_id.clone(), intent.outcome_id.clone());
        let pos = positions.entry(key).or_default();
        match intent.side {
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
    let mut intents: std::collections::HashMap<String, IntentOrderLite> = std::collections::HashMap::new();
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
                intents.insert(payload.order.client_order_id.clone(), payload.order);
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
        let row = state.entry(intent.market_id.clone()).or_default();
        let key = order_key(&intent.outcome_id, intent.side);
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

#[derive(Debug, Deserialize)]
struct IntentPayloadLite {
    order: IntentOrderLite,
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
        market_volume: v.market_volume,
        spread_cents: v.spread_cents,
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

fn candidate_artifacts_from(candidates: &[CandidateTrade]) -> Vec<ArtifactCandidate> {
    candidates.iter().map(artifact_candidate).collect()
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
