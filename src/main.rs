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
    let engine = ExecutionEngine::new(client, EngineConfig::default(), mode);

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
