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
            signal_origin: signal.signal_origin.clone(),
            execution_mode: Some(detect_execution_mode().to_string()),
            is_synthetic: false,
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
            signal_origin: None,
            execution_mode: Some(detect_execution_mode().to_string()),
            is_synthetic: false,
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
            signal_origin: None,
            execution_mode: Some(infer_execution_mode_from_report(report).to_string()),
            is_synthetic: false,
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
            signal_origin: None,
            execution_mode: Some(detect_execution_mode().to_string()),
            is_synthetic: false,
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

fn detect_execution_mode() -> &'static str {
    if matches!(
        std::env::var("BOT_RUN_RESEARCH_PAPER_CAPTURE_ONLY")
            .unwrap_or_else(|_| "false".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    ) {
        "paper"
    } else {
        match std::env::var("BOT_EXECUTION_MODE")
            .unwrap_or_else(|_| "live".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "paper" => "paper",
            _ => "live",
        }
    }
}

fn infer_execution_mode_from_report(report: &ExecutionReport) -> &'static str {
    if report.order_id.starts_with("paper-") || report.order_id.starts_with("sim-") {
        "paper"
    } else {
        detect_execution_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::types::{OrderType, Side, TimeInForce};
    use tempfile::TempDir;

    fn temp_cfg(tmp: &TempDir) -> ResearchCaptureConfig {
        ResearchCaptureConfig {
            enabled: true,
            root_dir: tmp.path().to_path_buf(),
        }
    }

    fn make_order() -> OrderRequest {
        OrderRequest {
            client_order_id: "coid-001".to_string(),
            market_id: "KXTEST-001".to_string(),
            outcome_id: "yes".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            limit_price: Some(0.55),
            quantity: 10.0,
            time_in_force: TimeInForce::Gtc,
            created_at: Utc::now(),
            market_yes_bid_size: None,
            market_yes_ask_size: None,
        }
    }

    fn make_signal() -> TradeSignal {
        TradeSignal {
            market_id: "KXTEST-001".to_string(),
            outcome_id: "yes".to_string(),
            side: Side::Buy,
            fair_price: 0.65,
            observed_price: 0.55,
            edge_pct: 0.10,
            confidence: 0.8,
            signal_timestamp: Utc::now(),
            signal_origin: Some("model_candidate".to_string()),
        }
    }

    #[test]
    fn record_order_intent_writes_valid_json() {
        let tmp = TempDir::new().unwrap();
        let cfg = temp_cfg(&tmp);
        let order = make_order();
        let signal = make_signal();

        record_order_intent(&cfg, &order, &signal).expect("should write without error");

        // Find the written file
        let day = Utc::now().format("%Y-%m-%d").to_string();
        let path = tmp
            .path()
            .join("order_lifecycle")
            .join(&day)
            .join("order_lifecycle.jsonl");
        assert!(path.exists(), "file should be created");

        let content = std::fs::read_to_string(&path).unwrap();
        let line = content.lines().next().expect("should have one line");
        let parsed: serde_json::Value = serde_json::from_str(line).expect("should be valid JSON");

        assert_eq!(parsed["event_type"], "intent");
        assert_eq!(parsed["client_order_id"], "coid-001");
        assert_eq!(parsed["ticker"], "KXTEST-001");
        assert_eq!(parsed["signal_fair_price"], 0.65);
        assert_eq!(parsed["signal_edge_pct"], 0.10);
    }

    #[test]
    fn record_order_error_writes_error_field() {
        let tmp = TempDir::new().unwrap();
        let cfg = temp_cfg(&tmp);
        let order = make_order();

        record_order_error(&cfg, &order, "submit_error", "exchange rejected")
            .expect("should write");

        let day = Utc::now().format("%Y-%m-%d").to_string();
        let path = tmp
            .path()
            .join("order_lifecycle")
            .join(&day)
            .join("order_lifecycle.jsonl");
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(content.lines().next().unwrap()).unwrap();

        assert_eq!(parsed["event_type"], "submit_error");
        assert_eq!(parsed["error"], "exchange rejected");
    }

    #[test]
    fn disabled_cfg_skips_write() {
        let tmp = TempDir::new().unwrap();
        let cfg = ResearchCaptureConfig {
            enabled: false,
            root_dir: tmp.path().to_path_buf(),
        };
        let order = make_order();
        let signal = make_signal();

        record_order_intent(&cfg, &order, &signal).expect("should be Ok even when disabled");

        let day = Utc::now().format("%Y-%m-%d").to_string();
        let path = tmp
            .path()
            .join("order_lifecycle")
            .join(&day)
            .join("order_lifecycle.jsonl");
        assert!(!path.exists(), "no file should be created when disabled");
    }

    #[test]
    fn infer_execution_mode_from_report_detects_paper_prefix() {
        let report = ExecutionReport {
            order_id: "paper-12345".to_string(),
            client_order_id: "c1".to_string(),
            status: OrderStatus::Filled,
            submitted_time_in_force: None,
            filled_qty: 10.0,
            avg_fill_price: Some(0.55),
            fee_paid: 0.0,
            updated_at: Utc::now(),
        };
        assert_eq!(infer_execution_mode_from_report(&report), "paper");
    }

    #[test]
    fn infer_execution_mode_from_report_detects_sim_prefix() {
        let report = ExecutionReport {
            order_id: "sim-99".to_string(),
            client_order_id: "c2".to_string(),
            status: OrderStatus::Canceled,
            submitted_time_in_force: None,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: 0.0,
            updated_at: Utc::now(),
        };
        assert_eq!(infer_execution_mode_from_report(&report), "paper");
    }
}
