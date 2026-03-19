use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use chrono::Utc;

use crate::execution::types::{
    ExecutionError, ExecutionReport, OrderAck, OrderRequest, OrderStatus, TradeSignal,
};
use crate::research::events::{OrderLifecycleEvent, RESEARCH_SCHEMA_VERSION};
use crate::research::market_recorder::ResearchCaptureConfig;

pub fn record_order_intent(
    cfg: &ResearchCaptureConfig,
    order: &OrderRequest,
    signal: &TradeSignal,
) -> Result<(), ExecutionError> {
    append_event(
        cfg,
        OrderLifecycleEvent {
            schema_version: RESEARCH_SCHEMA_VERSION.to_string(),
            ts: Utc::now(),
            client_order_id: order.client_order_id.clone(),
            order_id: None,
            ticker: order.market_id.clone(),
            outcome_id: order.outcome_id.clone(),
            side: order.side,
            tif: order.time_in_force,
            limit_price: order.limit_price,
            requested_qty: order.quantity,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: Some(signal.fair_price),
            signal_observed_price: Some(signal.observed_price),
            signal_edge_pct: Some(signal.edge_pct),
            signal_confidence: Some(signal.confidence),
            status: None,
            event_type: "intent".to_string(),
            error: None,
        },
    )
}

pub fn record_order_ack(
    cfg: &ResearchCaptureConfig,
    order: &OrderRequest,
    ack: &OrderAck,
) -> Result<(), ExecutionError> {
    append_event(
        cfg,
        OrderLifecycleEvent {
            schema_version: RESEARCH_SCHEMA_VERSION.to_string(),
            ts: ack.accepted_at,
            client_order_id: ack.client_order_id.clone(),
            order_id: Some(ack.order_id.clone()),
            ticker: order.market_id.clone(),
            outcome_id: order.outcome_id.clone(),
            side: order.side,
            tif: order.time_in_force,
            limit_price: order.limit_price,
            requested_qty: order.quantity,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            status: Some(OrderStatus::New),
            event_type: "ack".to_string(),
            error: None,
        },
    )
}

pub fn record_order_report(
    cfg: &ResearchCaptureConfig,
    order: &OrderRequest,
    report: &ExecutionReport,
    event_type: &str,
) -> Result<(), ExecutionError> {
    append_event(
        cfg,
        OrderLifecycleEvent {
            schema_version: RESEARCH_SCHEMA_VERSION.to_string(),
            ts: report.updated_at,
            client_order_id: report.client_order_id.clone(),
            order_id: Some(report.order_id.clone()),
            ticker: order.market_id.clone(),
            outcome_id: order.outcome_id.clone(),
            side: order.side,
            tif: report
                .submitted_time_in_force
                .unwrap_or(order.time_in_force),
            limit_price: order.limit_price,
            requested_qty: order.quantity,
            filled_qty: report.filled_qty,
            avg_fill_price: report.avg_fill_price,
            fee_paid: Some(report.fee_paid),
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            status: Some(report.status),
            event_type: event_type.to_string(),
            error: None,
        },
    )
}

pub fn record_order_error(
    cfg: &ResearchCaptureConfig,
    order: &OrderRequest,
    event_type: &str,
    error: &str,
) -> Result<(), ExecutionError> {
    append_event(
        cfg,
        OrderLifecycleEvent {
            schema_version: RESEARCH_SCHEMA_VERSION.to_string(),
            ts: Utc::now(),
            client_order_id: order.client_order_id.clone(),
            order_id: None,
            ticker: order.market_id.clone(),
            outcome_id: order.outcome_id.clone(),
            side: order.side,
            tif: order.time_in_force,
            limit_price: order.limit_price,
            requested_qty: order.quantity,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: None,
            signal_fair_price: None,
            signal_observed_price: None,
            signal_edge_pct: None,
            signal_confidence: None,
            status: None,
            event_type: event_type.to_string(),
            error: Some(error.to_string()),
        },
    )
}

fn append_event(cfg: &ResearchCaptureConfig, event: OrderLifecycleEvent) -> Result<(), ExecutionError> {
    if !cfg.enabled {
        return Ok(());
    }
    let day = Utc::now().format("%Y-%m-%d").to_string();
    let path = cfg
        .root_dir
        .join("order_lifecycle")
        .join(day)
        .join("order_lifecycle.jsonl");
    append_json_line(&path, &event)
}

fn append_json_line<T: serde::Serialize>(path: &Path, row: &T) -> Result<(), ExecutionError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    let line = serde_json::to_string(row).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|e| ExecutionError::Exchange(e.to_string()))
}
