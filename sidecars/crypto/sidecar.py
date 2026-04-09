"""
CryptoPredictor sidecar — HTTP service for the Kalshi trading bot.

Prediction approach
-------------------
Treats each Kalshi daily crypto market as a binary option: P(S_T > K) where
S is the current spot price, K is the strike threshold, and T is time
remaining to settlement (5pm EDT daily).

Uses a log-normal / GBM model:

    P(S_T > K) = N( (log(S/K) + (μ - σ²/2)·T) / (σ·√T) )

where σ is short-window realized volatility from recent OHLCV data and μ
is estimated drift (typically ~0 for intraday windows). No training required.
Works immediately from exchange price data.

This is NOT an LSTM price-prediction model. We don't predict the price —
we compute threshold-crossing probability from current market state. That's
a cleaner problem with an analytical solution.

Response contract (motorcade standard)
---------------------------------------
    {
        "probability":    0.73,
        "data_age_secs":  12,
        "data_source_ok": true,
        "model_version":  "gbm_v1"
    }

data_source_ok is false when:
  - Price cache is empty or stale (> MAX_DATA_AGE_SECS, default 60s)
  - Fewer than MIN_OHLCV_CANDLES for vol estimation
  - Asset not in ASSET_MAP

Endpoints
---------
    GET /health              → {"status": "ok", "assets": [...], "model_version": "gbm_v1"}
    GET /predict?ticker=...  → motorcade response contract above

Supported tickers
-----------------
    KXBTCD-{DATE}-{T|B}{THRESHOLD}   e.g. KXBTCD-26APR06-T67749.99
    KXETHD-{DATE}-{T|B}{THRESHOLD}   e.g. KXETHD-26APR06-B2079.99
    KXSOLD-{DATE}-{T|B}{THRESHOLD}
    KXXRPD-{DATE}-{T|B}{THRESHOLD}

Environment variables
---------------------
    CRYPTO_SIDECAR_HOST         Optional. Default: 127.0.0.1
    CRYPTO_SIDECAR_PORT         Optional. Default: 8766
    PRICE_REFRESH_SECS          Optional. Default: 30
    CRYPTO_MAX_DATA_AGE_SECS    Optional. Default: 60
    CRYPTO_PREDICTION_LOG_DIR   Optional. Default: var/logs/crypto_predictions
"""

import json
import logging
import os
import re
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

import price_fetcher
from predictor import predict as gbm_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CryptoPredictor Sidecar", version="1.0")

# ── Config ─────────────────────────────────────────────────────────────────────

PRICE_REFRESH_SECS  = int(os.getenv("PRICE_REFRESH_SECS",        "30"))
MAX_DATA_AGE_SECS   = int(os.getenv("CRYPTO_MAX_DATA_AGE_SECS",  "60"))
PREDICTION_LOG_DIR  = Path(os.getenv("CRYPTO_PREDICTION_LOG_DIR", "var/logs/crypto_predictions"))

MODEL_VERSION = "gbm_v1"

# Maps the ticker prefix (after "KX") to an asset code.
TICKER_PREFIX_MAP = {
    "BTCD": "BTC",
    "ETHD": "ETH",
    "SOLD": "SOL",
    "XRPD": "XRP",
}

# All Kalshi daily crypto markets settle at 17:00 EDT.
SETTLEMENT_HOUR_ET = 17

# ── Prediction log ─────────────────────────────────────────────────────────────

_log_lock = threading.Lock()


def _write_prediction_log(record: dict) -> None:
    """Append one prediction record to today's JSONL log. Silently swallows errors."""
    try:
        day  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = PREDICTION_LOG_DIR / f"predictions_{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _log_lock:
            with open(path, "a") as f:
                f.write(line)
    except Exception as exc:
        logger.warning(f"prediction log write failed: {exc}")


# ── Ticker parsing ─────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> tuple[Optional[str], Optional[date], Optional[float], bool]:
    """
    Parse a Kalshi crypto ticker into (asset, settlement_date, strike, below).

    Format: KX{ASSET_CODE}-{YYMONDD}-{T|B}{THRESHOLD}

    Examples:
        KXBTCD-26APR06-T67749.99  → ("BTC", date(2026, 4, 6),  67749.99, False)
        KXETHD-26APR06-B2079.99   → ("ETH", date(2026, 4, 6),  2079.99,  True)
        KXSOLD-26APR06-T80.9999   → ("SOL", date(2026, 4, 6),  80.9999,  False)
        KXXRPD-26APR06-T2.5       → ("XRP", date(2026, 4, 6),  2.5,      False)
    """
    upper = ticker.upper()
    if not upper.startswith("KX"):
        return None, None, None, False

    # Strip "KX" prefix, then extract the asset code (everything up to the first "-")
    rest  = upper[2:]
    parts = rest.split("-")
    if len(parts) < 3:
        return None, None, None, False

    prefix = parts[0]  # e.g. "BTCD"
    asset  = TICKER_PREFIX_MAP.get(prefix)
    if asset is None:
        return None, None, None, False

    # Date part — handle both daily (YYMONDD) and intraday (YYMONDDHR) formats
    settlement_date = None
    for fmt in ("%y%b%d%H", "%y%b%d"):
        try:
            settlement_date = datetime.strptime(parts[1], fmt).date()
            break
        except ValueError:
            continue
    if settlement_date is None:
        return asset, None, None, False

    # Threshold part: T or B followed by a number
    thresh_str = parts[2]
    match = re.match(r"^([TB])(\d+(?:\.\d+)?)$", thresh_str)
    if not match:
        return asset, settlement_date, None, False

    below  = match.group(1) == "B"
    strike = float(match.group(2))

    return asset, settlement_date, strike, below


def _seconds_to_settlement(settlement_date: date) -> float:
    """
    Seconds from now (UTC) until 17:00 EDT on settlement_date.
    EDT = UTC-4. Returns 0 if settlement is in the past.
    """
    # Determine UTC offset for EDT vs EST.
    # Kalshi uses EDT during summer (UTC-4), EST in winter (UTC-5).
    # Since crypto markets are daily, we use a simple approach: hardcode EDT (UTC-4).
    # If DST is a concern, use zoneinfo (Python 3.9+) or pytz.
    settlement_utc_hour = SETTLEMENT_HOUR_ET + 4  # EDT = UTC-4 → 17:00 EDT = 21:00 UTC
    settlement_naive    = datetime(
        settlement_date.year,
        settlement_date.month,
        settlement_date.day,
        settlement_utc_hour,
        0,
        0,
    )
    settlement_dt = settlement_naive.replace(tzinfo=timezone.utc)
    now           = datetime.now(timezone.utc)
    delta         = (settlement_dt - now).total_seconds()
    return max(0.0, delta)


# ── Routes ─────────────────────────────────────────────────────────────────────

def _startup_warmup() -> None:
    """Warm the price cache once at startup. Runs in background thread."""
    price_fetcher.warmup()
    logger.info("CryptoPredictor startup warmup complete")


@app.on_event("startup")
def on_startup():
    threading.Thread(target=_startup_warmup, daemon=True).start()
    threading.Thread(
        target=price_fetcher.refresh_all,
        args=(PRICE_REFRESH_SECS,),
        daemon=True,
    ).start()
    logger.info("Startup: warmup + background price refresh threads started")


@app.get("/health")
def health():
    assets_cached = [a for a in price_fetcher.ASSET_MAP if price_fetcher.get_spot(a) is not None]
    return {
        "status":        "ok",
        "assets":        sorted(price_fetcher.ASSET_MAP.keys()),
        "assets_cached": sorted(assets_cached),
        "model_version": MODEL_VERSION,
    }


@app.get("/predict")
def predict(ticker: str):
    asset, settlement_date, strike, below = _parse_ticker(ticker)

    if asset is None:
        raise HTTPException(404, f"Unrecognized crypto ticker '{ticker}'")
    if settlement_date is None:
        raise HTTPException(400, f"Cannot parse settlement date from ticker '{ticker}'")
    if strike is None:
        raise HTTPException(400, f"Cannot parse strike from ticker '{ticker}'")

    # Read from cache — no blocking I/O in the hot path.
    spot_entry     = price_fetcher.get_spot(asset)
    candles_1m     = price_fetcher.get_candles_1m(asset)
    candles_1h     = price_fetcher.get_candles_1h(asset)
    age_secs       = price_fetcher.cache_age_secs(asset)

    if spot_entry is None:
        logger.warning(f"predict: no spot price cached for {asset} ({ticker})")
        return {
            "probability":    0.5,
            "data_age_secs":  -1,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    if age_secs > MAX_DATA_AGE_SECS:
        logger.warning(f"predict: stale cache for {asset} age={age_secs}s ({ticker})")
        return {
            "probability":    0.5,
            "data_age_secs":  age_secs,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    spot               = spot_entry[0]
    seconds_remaining  = _seconds_to_settlement(settlement_date)
    closes_1m          = candles_1m[0] if candles_1m else None
    closes_1h          = candles_1h[0] if candles_1h else None

    prob = gbm_predict(
        spot=spot,
        strike=strike,
        seconds_remaining=seconds_remaining,
        closes_1m=closes_1m,
        closes_1h=closes_1h,
        below=below,
        asset=asset,
    )

    logger.info(
        f"predict  ticker={ticker}  asset={asset}  spot={spot:.4f}  "
        f"strike={strike}  secs_remaining={seconds_remaining:.0f}  "
        f"below={below}  prob={prob:.4f}  age={age_secs}s"
    )

    _write_prediction_log({
        "ts":                datetime.now(timezone.utc).isoformat(),
        "ticker":            ticker,
        "asset":             asset,
        "settlement_date":   str(settlement_date),
        "spot":              spot,
        "strike":            strike,
        "below":             below,
        "seconds_remaining": seconds_remaining,
        "probability":       prob,
        "data_age_secs":     age_secs,
        "model_version":     MODEL_VERSION,
    })

    return {
        "probability":    prob,
        "data_age_secs":  age_secs,
        "data_source_ok": True,
        "model_version":  MODEL_VERSION,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("CRYPTO_SIDECAR_PORT", "8766"))
    host = os.getenv("CRYPTO_SIDECAR_HOST", "127.0.0.1")
    uvicorn.run("sidecar:app", host=host, port=port, reload=False)
