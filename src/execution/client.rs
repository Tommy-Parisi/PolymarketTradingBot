use async_trait::async_trait;
use base64::Engine as _;
use chrono::Utc;
use reqwest::{Client, Method, StatusCode};
use std::error::Error as _;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::SigningKey;
use rsa::rand_core::OsRng;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use crate::execution::types::{ExecutionError, ExecutionReport, OrderAck, OrderRequest, OrderStatus, Side};

const KALSHI_ACCESS_KEY: &str = "KALSHI-ACCESS-KEY";
const KALSHI_ACCESS_TIMESTAMP: &str = "KALSHI-ACCESS-TIMESTAMP";
const KALSHI_ACCESS_SIGNATURE: &str = "KALSHI-ACCESS-SIGNATURE";

#[async_trait]
pub trait ExchangeClient: Send + Sync {
    async fn place_order(&self, request: &OrderRequest) -> Result<OrderAck, ExecutionError>;
    async fn get_order(&self, order_id: &str) -> Result<ExecutionReport, ExecutionError>;
    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError>;
    async fn smoke_test(&self) -> Result<(), ExecutionError>;
}

#[derive(Debug, Clone)]
pub struct KalshiClientConfig {
    pub api_base_url: String,
    pub access_key_id: String,
    pub private_key_pem: String,
}

pub struct KalshiClient {
    config: KalshiClientConfig,
    private_key: RsaPrivateKey,
    http: Client,
}

impl KalshiClient {
    pub fn new(config: KalshiClientConfig) -> Result<Self, ExecutionError> {
        let private_key = parse_rsa_key(&config.private_key_pem)?;
        Ok(Self {
            config,
            private_key,
            http: Client::new(),
        })
    }

    pub fn from_env() -> Result<Self, ExecutionError> {
        let api_base_url =
            std::env::var("KALSHI_API_BASE_URL").unwrap_or_else(|_| "https://demo-api.kalshi.co".to_string());
        let access_key_id =
            std::env::var("KALSHI_API_KEY_ID").map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let private_key_pem = read_private_key_from_env()?;
        Self::new(KalshiClientConfig {
            api_base_url,
            access_key_id,
            private_key_pem,
        })
    }

    fn auth_headers(&self, method: Method, path: &str) -> Result<[(String, String); 3], ExecutionError> {
        let timestamp_ms = Utc::now().timestamp_millis().to_string();
        let msg = format!("{}{}{}", timestamp_ms, method, path);
        let signature = sign_pss_base64(&self.private_key, msg.as_bytes())?;

        Ok([
            (KALSHI_ACCESS_KEY.to_string(), self.config.access_key_id.clone()),
            (KALSHI_ACCESS_TIMESTAMP.to_string(), timestamp_ms),
            (KALSHI_ACCESS_SIGNATURE.to_string(), signature),
        ])
    }
}

#[async_trait]
impl ExchangeClient for KalshiClient {
    async fn place_order(&self, request: &OrderRequest) -> Result<OrderAck, ExecutionError> {
        let path = "/trade-api/v2/portfolio/orders";
        let side = kalshi_side_from_outcome(&request.outcome_id)?;
        let action = match request.side {
            Side::Buy => "buy",
            Side::Sell => "sell",
        };
        let price_dollars = request.limit_price.map(prob_to_dollars_string);
        let (yes_price_dollars, no_price_dollars) = if side == "yes" {
            (price_dollars, None)
        } else {
            (None, price_dollars)
        };
        let body = CreateOrderRequest {
            ticker: request.market_id.clone(),
            client_order_id: request.client_order_id.clone(),
            side,
            action: action.to_string(),
            count: request.quantity.max(0.0).round() as u64,
            count_fp: Some(to_fp_string(request.quantity)),
            yes_price_dollars,
            no_price_dollars,
            order_type: "limit".to_string(),
        };
        let body_text = serde_json::to_string(&body).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let url = format!("{}{}", self.config.api_base_url, path);
        let headers = self.auth_headers(Method::POST, path)?;

        let mut req = self
            .http
            .post(url)
            .header("Content-Type", "application/json")
            .body(body_text);
        for (k, v) in headers {
            req = req.header(k, v);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(format_reqwest_error("place_order", &e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !status.is_success() {
            return Err(build_http_error("POST", path, status, &text));
        }

        let parsed: CreateOrderEnvelope =
            serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let order = parsed
            .order
            .ok_or_else(|| ExecutionError::Exchange(format!("missing order payload: {text}")))?;
        let order_id = order
            .order_id
            .or(order.id)
            .ok_or_else(|| ExecutionError::Exchange(format!("missing order id: {text}")))?;
        Ok(OrderAck {
            order_id,
            client_order_id: request.client_order_id.clone(),
            accepted_at: Utc::now(),
        })
    }

    async fn get_order(&self, order_id: &str) -> Result<ExecutionReport, ExecutionError> {
        let path = format!("/trade-api/v2/portfolio/orders/{order_id}");
        let url = format!("{}{}", self.config.api_base_url, path);
        let headers = self.auth_headers(Method::GET, &path)?;

        let mut req = self.http.get(url);
        for (k, v) in headers {
            req = req.header(k, v);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(format_reqwest_error("get_order", &e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !status.is_success() {
            return Err(build_http_error("GET", &path, status, &text));
        }

        let parsed: GetOrderEnvelope =
            serde_json::from_str(&text).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        let order = parsed
            .order
            .ok_or_else(|| ExecutionError::Exchange(format!("missing order payload: {text}")))?;
        Ok(ExecutionReport {
            order_id: order
                .order_id
                .or(order.id)
                .unwrap_or_else(|| order_id.to_string()),
            client_order_id: order.client_order_id.unwrap_or_default(),
            status: map_order_status(order.status.as_deref()),
            filled_qty: order
                .filled_count_fp
                .as_deref()
                .and_then(parse_fp_string)
                .or(order.filled_count)
                .unwrap_or(0.0),
            avg_fill_price: order
                .yes_price_dollars
                .as_deref()
                .and_then(parse_prob_dollars_string)
                .or_else(|| order.yes_price_cents.map(cents_to_prob)),
            fee_paid: order.fee.unwrap_or(0.0),
            updated_at: Utc::now(),
        })
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError> {
        let path = format!("/trade-api/v2/portfolio/orders/{order_id}");
        let url = format!("{}{}", self.config.api_base_url, path);
        let headers = self.auth_headers(Method::DELETE, &path)?;

        let mut req = self.http.delete(url);
        for (k, v) in headers {
            req = req.header(k, v);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(format_reqwest_error("cancel_order", &e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if status.is_success() {
            return Ok(());
        }
        Err(build_http_error("DELETE", &path, status, &text))
    }

    async fn smoke_test(&self) -> Result<(), ExecutionError> {
        let path = "/trade-api/v2/portfolio/orders?limit=1";
        let auth_path = "/trade-api/v2/portfolio/orders";
        let url = format!("{}{}", self.config.api_base_url, path);
        let headers = self.auth_headers(Method::GET, auth_path)?;
        let mut req = self.http.get(url);
        for (k, v) in headers {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ExecutionError::RetryableExchange(format_reqwest_error("smoke_test", &e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        if !status.is_success() {
            return Err(build_http_error("GET", path, status, &text));
        }
        Ok(())
    }
}

fn parse_rsa_key(pem: &str) -> Result<RsaPrivateKey, ExecutionError> {
    RsaPrivateKey::from_pkcs8_pem(pem)
        .or_else(|_| RsaPrivateKey::from_pkcs1_pem(pem))
        .map_err(|e| ExecutionError::Exchange(format!("invalid Kalshi private key PEM: {e}")))
}

fn sign_pss_base64(private_key: &RsaPrivateKey, payload: &[u8]) -> Result<String, ExecutionError> {
    let signing_key = SigningKey::<Sha256>::new(private_key.clone());
    let mut rng = OsRng;
    let signature = signing_key.sign_with_rng(&mut rng, payload);
    Ok(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()))
}

fn read_private_key_from_env() -> Result<String, ExecutionError> {
    if let Ok(pem) = std::env::var("KALSHI_PRIVATE_KEY_PEM") {
        return Ok(pem);
    }
    if let Ok(path) = std::env::var("KALSHI_PRIVATE_KEY_PATH") {
        let pem = std::fs::read_to_string(path).map_err(|e| ExecutionError::Exchange(e.to_string()))?;
        return Ok(pem);
    }
    Err(ExecutionError::Exchange(
        "missing KALSHI_PRIVATE_KEY_PEM or KALSHI_PRIVATE_KEY_PATH".to_string(),
    ))
}

fn kalshi_side_from_outcome(outcome_id: &str) -> Result<String, ExecutionError> {
    match outcome_id.to_ascii_lowercase().as_str() {
        "yes" => Ok("yes".to_string()),
        "no" => Ok("no".to_string()),
        other => Err(ExecutionError::Exchange(format!(
            "unsupported outcome_id '{other}', expected 'yes' or 'no'"
        ))),
    }
}

fn prob_to_dollars_string(prob: f64) -> String {
    format!("{:.4}", prob.clamp(0.0, 1.0))
}

fn cents_to_prob(cents: u32) -> f64 {
    (cents as f64) / 100.0
}

fn parse_prob_dollars_string(raw: &str) -> Option<f64> {
    raw.parse::<f64>().ok()
}

fn to_fp_string(v: f64) -> String {
    // Kalshi requires fixed-point quantity strings with at most 2 decimal places.
    format!("{:.2}", v.max(0.0))
}

fn parse_fp_string(raw: &str) -> Option<f64> {
    raw.parse::<f64>().ok()
}

fn map_order_status(status: Option<&str>) -> OrderStatus {
    match status.unwrap_or_default().to_ascii_lowercase().as_str() {
        "open" | "resting" | "new" => OrderStatus::New,
        "executed" | "filled" => OrderStatus::Filled,
        "partially_filled" | "partial" => OrderStatus::PartiallyFilled,
        "canceled" | "cancelled" => OrderStatus::Canceled,
        "rejected" => OrderStatus::Rejected,
        _ => OrderStatus::New,
    }
}

fn build_http_error(method: &str, path: &str, status: StatusCode, body: &str) -> ExecutionError {
    let msg = format!("{method} {path} failed ({status}): {body}");
    if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        return ExecutionError::RetryableExchange(msg);
    }
    ExecutionError::Exchange(msg)
}

fn format_reqwest_error(context: &str, e: &reqwest::Error) -> String {
    let mut out = format!(
        "{context}: {e} [is_connect={} is_timeout={} is_status={} url={}]",
        e.is_connect(),
        e.is_timeout(),
        e.is_status(),
        e.url()
            .map(|u| u.as_str().to_string())
            .unwrap_or_else(|| "<none>".to_string())
    );
    let mut src = e.source();
    while let Some(s) = src {
        out.push_str(&format!("; source={s}"));
        src = s.source();
    }
    out
}

#[derive(Debug, Serialize)]
struct CreateOrderRequest {
    ticker: String,
    client_order_id: String,
    side: String,
    action: String,
    count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    count_fp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    yes_price_dollars: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    no_price_dollars: Option<String>,
    order_type: String,
}

#[derive(Debug, Deserialize)]
struct CreateOrderEnvelope {
    order: Option<KalshiOrderPayload>,
}

#[derive(Debug, Deserialize)]
struct GetOrderEnvelope {
    order: Option<KalshiOrderPayload>,
}

#[derive(Debug, Deserialize)]
struct KalshiOrderPayload {
    #[serde(alias = "order_id", alias = "orderId")]
    order_id: Option<String>,
    id: Option<String>,
    #[serde(alias = "client_order_id", alias = "clientOrderId")]
    client_order_id: Option<String>,
    status: Option<String>,
    #[serde(alias = "filled_count", alias = "filledCount")]
    filled_count: Option<f64>,
    #[serde(alias = "filled_count_fp", alias = "filledCountFp")]
    filled_count_fp: Option<String>,
    #[serde(alias = "yes_price", alias = "yesPrice")]
    yes_price_cents: Option<u32>,
    #[serde(alias = "yes_price_dollars", alias = "yesPriceDollars")]
    yes_price_dollars: Option<String>,
    fee: Option<f64>,
}
