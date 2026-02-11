use std::sync::Arc;

use chrono::Utc;

mod execution;
mod markets;

use execution::client::KalshiClient;
use execution::engine::{ExecutionEngine, ExecutionMode};
use execution::types::{EngineConfig, Side, TradeSignal};
use markets::kalshi_mapper::{resolution_mode_from_env, KalshiMarketMapper, ResolutionMode};

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

    let mut signal = TradeSignal {
        market_id: "market-123".to_string(),
        outcome_id: "yes".to_string(),
        side: Side::Buy,
        fair_price: 0.62,
        observed_price: 0.53,
        edge_pct: 0.09,
        confidence: 0.78,
        signal_timestamp: Utc::now(),
    };
    let mapper = KalshiMarketMapper::from_env();
    let resolution_mode = resolution_mode_from_env();
    match mapper.resolve_market_ticker(&signal.market_id).await {
        Ok(ticker) => signal.market_id = ticker,
        Err(err) if resolution_mode == ResolutionMode::BestEffort => {
            eprintln!("market resolution warning (best-effort): {err}");
        }
        Err(err) => {
            eprintln!("market resolution failed (strict mode): {err}");
            return;
        }
    }

    let bankroll = 10_000.0;
    match engine.execute_signal(&signal, bankroll).await {
        Ok(report) => println!("order executed: {:?}", report),
        Err(err) => eprintln!("execution failed: {err}"),
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
