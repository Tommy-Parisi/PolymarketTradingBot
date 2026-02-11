use std::sync::Arc;

mod data;
mod execution;
mod markets;
mod model;

use data::market_enrichment::{EnrichmentConfig, MarketEnricher};
use data::market_scanner::{KalshiMarketScanner, ScannerConfig};
use execution::client::KalshiClient;
use execution::engine::{ExecutionEngine, ExecutionMode};
use execution::types::EngineConfig;
use markets::kalshi_mapper::{resolution_mode_from_env, KalshiMarketMapper, ResolutionMode};
use model::allocator::{AllocationConfig, PortfolioAllocator};
use model::valuation::{ClaudeValuationEngine, ValuationConfig, ValuationInput};

#[tokio::main]
async fn main() {
    let client = match KalshiClient::from_env() {
        Ok(client) => Arc::new(client),
        Err(err) => {
            eprintln!("missing/invalid Kalshi env config: {err}");
            return;
        }
    };

    let mode = execution_mode_from_env();
    let engine = ExecutionEngine::new(client, engine_config_from_env(), mode);

    let scanner = KalshiMarketScanner::new(ScannerConfig::default());
    let scanned = match scanner.scan_snapshot_with_deltas().await {
        Ok(markets) => markets,
        Err(err) => {
            eprintln!("market scan failed: {err}");
            return;
        }
    };
    let selected = scanner.select_for_valuation(scanned);
    if selected.is_empty() {
        eprintln!("no markets selected for valuation");
        return;
    }
    println!("selected {} markets for valuation", selected.len());

    let enricher = MarketEnricher::new(EnrichmentConfig::default());
    let sample_size = selected.len().min(25);
    let enrichments = match enricher.enrich_batch(&selected[..sample_size]).await {
        Ok(v) => v,
        Err(err) => {
            eprintln!("enrichment failed: {err}");
            return;
        }
    };
    println!("enriched {} markets", enrichments.len());
    if mode == ExecutionMode::Live && smoke_test_enabled() {
        if let Err(err) = engine.run_smoke_test().await {
            eprintln!("kalshi smoke test failed: {err}");
            return;
        }
    }
    if mode == ExecutionMode::Live {
        if let Err(err) = engine.reconcile_open_orders().await {
            eprintln!("startup reconcile warning: {err}");
        }
    }

    let mut enrichment_by_ticker = std::collections::HashMap::new();
    for e in enrichments {
        enrichment_by_ticker.insert(e.ticker.clone(), e);
    }

    let valuation_limit = valuation_market_limit_from_env().min(selected.len());
    let valuation_inputs: Vec<ValuationInput> = selected
        .iter()
        .take(valuation_limit)
        .map(|m| ValuationInput {
            market: m.clone(),
            enrichment: enrichment_by_ticker.get(&m.ticker).cloned(),
        })
        .collect();

    let valuator = ClaudeValuationEngine::new(ValuationConfig::default());
    let valuations = match valuator.value_markets(&valuation_inputs).await {
        Ok(v) => v,
        Err(err) => {
            eprintln!("valuation failed: {err}");
            return;
        }
    };
    println!("valued {} markets", valuations.len());

    let candidates = valuator.generate_candidates(&valuations);
    if candidates.is_empty() {
        eprintln!("no mispricing candidates above threshold");
        return;
    }
    println!("generated {} candidates", candidates.len());
    println!("top candidate rationale: {}", candidates[0].rationale);
    let mapper = KalshiMarketMapper::from_env();
    let resolution_mode = resolution_mode_from_env();

    let bankroll = bankroll_from_env();
    let allocator = PortfolioAllocator::new(allocation_config_from_env());
    let allocations = allocator.allocate(bankroll, candidates);
    if allocations.is_empty() {
        eprintln!("allocator produced no trades");
        return;
    }
    println!("allocator selected {} trades", allocations.len());

    for allocated in allocations {
        let mut signal = valuator.candidate_to_signal(&allocated.candidate);
        match mapper.resolve_market_ticker(&signal.market_id).await {
            Ok(ticker) => signal.market_id = ticker,
            Err(err) if resolution_mode == ResolutionMode::BestEffort => {
                eprintln!("market resolution warning (best-effort): {err}");
            }
            Err(err) => {
                eprintln!("market resolution failed (strict mode): {err}");
                continue;
            }
        }

        match engine.execute_signal(&signal, allocated.notional).await {
            Ok(report) => println!(
                "order executed: ticker={} notional={:.2} report={:?}",
                allocated.candidate.ticker, allocated.notional, report
            ),
            Err(err) => eprintln!("execution failed for {}: {err}", allocated.candidate.ticker),
        }
    }
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

fn engine_config_from_env() -> EngineConfig {
    let mut cfg = EngineConfig::default();
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
