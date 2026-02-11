use std::sync::Arc;

use chrono::{Duration, Utc};

use crate::execution::engine::{ExecutionEngine, ExecutionMode};
use crate::execution::paper_sim::{PaperSimClient, PaperSimConfig};
use crate::execution::types::{EngineConfig, OrderStatus, Side, TradeSignal};
use crate::model::allocator::{AllocationConfig, PortfolioAllocator};
use crate::model::valuation::CandidateTrade;

pub async fn run_multi_day_replay() {
    let days = std::env::var("BOT_REPLAY_DAYS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(3)
        .max(1);
    let cycles_per_day = std::env::var("BOT_REPLAY_CYCLES_PER_DAY")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(144)
        .max(1);
    let bankroll = std::env::var("BOT_REPLAY_BANKROLL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(10_000.0)
        .max(100.0);

    let mut cfg = EngineConfig::default();
    cfg.state_path = "/tmp/event_trading_bot_replay_state.json".to_string();
    cfg.journal_path = "/tmp/event_trading_bot_replay_journal.jsonl".to_string();
    cfg.max_orders_per_minute = 1_000_000;
    cfg.max_open_exposure_notional = 1_000_000_000.0;
    cfg.max_daily_loss = 1_000_000_000.0;
    cfg.max_order_quantity = 1_000_000.0;
    cfg.reconcile_poll_attempts = 0;
    cfg.reconcile_poll_interval_ms = 0;
    let engine = ExecutionEngine::new(
        Arc::new(PaperSimClient::new(PaperSimConfig {
            fee_bps: 12.0,
            max_slippage_bps: 35.0,
            min_latency_ms: 0,
            max_latency_ms: 1,
        })),
        cfg,
        ExecutionMode::Live,
    );
    let allocator = PortfolioAllocator::new(AllocationConfig::default());

    let mut total_orders = 0usize;
    let mut filled = 0usize;
    let mut partial = 0usize;
    let mut canceled = 0usize;
    let mut fees = 0.0;
    let mut realized_edge_pnl = 0.0;

    for day in 0..days {
        for cycle in 0..cycles_per_day {
            let candidates = synthetic_candidates(day, cycle);
            let allocations = allocator.allocate(bankroll, candidates);
            for a in allocations {
                total_orders += 1;
                let signal = TradeSignal {
                    market_id: a.candidate.ticker.clone(),
                    outcome_id: a.candidate.outcome_id.clone(),
                    side: a.candidate.side,
                    fair_price: a.candidate.fair_price,
                    observed_price: a.candidate.observed_price,
                    edge_pct: a.candidate.edge_pct,
                    confidence: a.candidate.confidence,
                    signal_timestamp: Utc::now() - Duration::seconds(1),
                };
                if let Ok(report) = engine.execute_signal(&signal, a.notional).await {
                    match report.status {
                        OrderStatus::Filled => filled += 1,
                        OrderStatus::PartiallyFilled => partial += 1,
                        OrderStatus::Canceled => canceled += 1,
                        _ => {}
                    }
                    fees += report.fee_paid;
                    let fill_price = report.avg_fill_price.unwrap_or(signal.observed_price);
                    let unit_edge = match signal.side {
                        Side::Buy => signal.fair_price - fill_price,
                        Side::Sell => fill_price - signal.fair_price,
                    };
                    realized_edge_pnl += unit_edge * report.filled_qty - report.fee_paid;
                }
            }
        }
    }

    println!("replay complete");
    println!("days={} cycles_per_day={}", days, cycles_per_day);
    println!("orders_total={total_orders}");
    println!("filled={filled} partial={partial} canceled={canceled}");
    println!("fees_paid={:.4}", fees);
    println!("edge_pnl_net_fees={:.4}", realized_edge_pnl);
}

fn synthetic_candidates(day: u32, cycle: u32) -> Vec<CandidateTrade> {
    let mut out = Vec::new();
    for i in 0..12_u32 {
        let base = ((day * 131 + cycle * 17 + i * 7) % 100) as f64 / 100.0;
        let mid = (0.35 + 0.3 * base).clamp(0.05, 0.95);
        let fair = (mid + (0.03 + 0.12 * (base - 0.5))).clamp(0.01, 0.99);
        let edge = (fair - mid).abs();
        let confidence = (0.45 + 0.5 * base).clamp(0.3, 0.95);
        let (outcome_id, side, fair_price, observed_price) = if fair >= mid {
            ("yes".to_string(), Side::Buy, fair, mid)
        } else {
            ("no".to_string(), Side::Buy, 1.0 - fair, 1.0 - mid)
        };
        out.push(CandidateTrade {
            ticker: format!("SIM-MKT-{i}"),
            side,
            outcome_id,
            fair_price,
            observed_price,
            edge_pct: edge,
            confidence,
            rationale: "synthetic replay candidate".to_string(),
        });
    }
    out
}
