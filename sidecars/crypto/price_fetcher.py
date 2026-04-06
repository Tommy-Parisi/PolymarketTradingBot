"""
Price fetcher for CryptoPredictor sidecar.

Fetches spot price and recent OHLCV candles from Coinbase Advanced Trade API
(public endpoints, no auth required). Results are cached to avoid hammering
the API on every predict call.

Data sourced:
  - Spot price: GET /api/v3/brokerage/best_bid_ask (or similar public endpoint)
  - OHLCV: GET /api/v3/brokerage/market/candles
    Candle granularity: ONE_MINUTE for short-window vol, ONE_HOUR for long-window

Fallback: Binance REST API (api.binance.com/api/v3/) — same OHLCV structure,
no auth, global availability. Use if Coinbase is unavailable.

Cache structure:
  - _spot_cache: {asset -> (price, fetch_time)}
  - _ohlcv_cache: {(asset, granularity) -> ([candles], fetch_time)}

Both caches refresh every PRICE_REFRESH_SECS (default 30s). The predict
endpoint reads from cache only — no blocking network calls in the hot path.

Assets supported: BTC, ETH, SOL, XRP (maps to {asset}-USD trading pairs)

Example candle row from Coinbase:
    {"start": "1712345678", "low": "66100.0", "high": "66500.0",
     "open": "66200.0", "close": "66300.0", "volume": "123.45"}
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ASSET_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
}

COINBASE_BASE          = "https://api.exchange.coinbase.com"  # public, no auth
COINBASE_SPOT_BASE     = "https://api.coinbase.com"            # v2 spot price fallback

# Coinbase Exchange candle granularities in seconds
GRANULARITY_ONE_MINUTE = 60    # 1-minute candles
GRANULARITY_ONE_HOUR   = 3600  # 1-hour candles

# Number of candles to fetch for vol estimation windows.
# 15m vol → 15 one-minute candles; 1h vol → 60; 4h vol → 4 one-hour candles.
CANDLES_1M_COUNT = 70   # ~70 minutes; covers 15m + 1h windows with room
CANDLES_1H_COUNT = 8    # 8 hours; covers the 4h window

# Stale data threshold — callers check this
MAX_DATA_AGE_SECS = 60

# ── Cache ──────────────────────────────────────────────────────────────────────
# Keyed by asset string (e.g. "BTC"). Thread-safe via a single lock.

_cache_lock   = threading.Lock()
_spot_cache:  dict[str, tuple[float, datetime]] = {}   # asset → (price, fetch_time)
_ohlcv_1m:    dict[str, tuple[list, datetime]]  = {}   # asset → (candles, fetch_time)
_ohlcv_1h:    dict[str, tuple[list, datetime]]  = {}   # asset → (candles, fetch_time)


# ── Coinbase fetch ─────────────────────────────────────────────────────────────

def _coinbase_spot(product_id: str, session: requests.Session) -> Optional[float]:
    """Fetch mid price from Coinbase Exchange ticker (public, no auth)."""
    try:
        url  = f"{COINBASE_BASE}/products/{product_id}/ticker"
        resp = session.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        bid  = float(data.get("bid", 0))
        ask  = float(data.get("ask", 0))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        price = float(data.get("price", 0))
        return price if price > 0 else None
    except Exception as exc:
        logger.debug(f"Coinbase Exchange ticker failed for {product_id}: {exc}")
        return None


def _coinbase_candles(
    product_id: str,
    granularity: int,
    count: int,
    session: requests.Session,
) -> Optional[list]:
    """Fetch recent OHLCV candles from Coinbase Exchange (public, no auth).
    Returns list of close prices (float), ordered oldest→newest.

    Coinbase Exchange candle format: [timestamp, low, high, open, close, volume]
    Results are returned newest-first, so we reverse before returning.
    """
    try:
        end   = int(time.time())
        start = end - granularity * count
        url   = f"{COINBASE_BASE}/products/{product_id}/candles"
        resp  = session.get(
            url,
            params={"granularity": granularity, "start": start, "end": end},
            timeout=5,
        )
        resp.raise_for_status()
        candles = resp.json()
        # Each entry: [timestamp, low, high, open, close, volume]
        closes  = [float(c[4]) for c in reversed(candles)]
        return closes if len(closes) >= 2 else None
    except Exception as exc:
        logger.debug(f"Coinbase Exchange candles failed for {product_id} gran={granularity}: {exc}")
        return None


# ── Refresh logic ──────────────────────────────────────────────────────────────

def _refresh_asset(asset: str, session: requests.Session) -> None:
    """Refresh spot price and OHLCV candles for one asset. Writes to cache."""
    product_id = ASSET_MAP[asset]
    now        = datetime.now(timezone.utc)

    spot      = _coinbase_spot(product_id, session)
    closes_1m = _coinbase_candles(product_id, GRANULARITY_ONE_MINUTE, CANDLES_1M_COUNT, session)
    closes_1h = _coinbase_candles(product_id, GRANULARITY_ONE_HOUR,   CANDLES_1H_COUNT, session)

    if spot is None:
        logger.warning(f"{asset}: spot fetch failed")

    with _cache_lock:
        if spot is not None:
            _spot_cache[asset]  = (spot, now)
        if closes_1m is not None:
            _ohlcv_1m[asset]    = (closes_1m, now)
        if closes_1h is not None:
            _ohlcv_1h[asset]    = (closes_1h, now)

    logger.info(
        f"cache updated: {asset}  spot={spot}  "
        f"1m_candles={len(closes_1m) if closes_1m else 0}  "
        f"1h_candles={len(closes_1h) if closes_1h else 0}"
    )


def refresh_all(refresh_secs: int) -> None:
    """Background thread: refresh all assets every refresh_secs. Runs forever."""
    session = requests.Session()
    while True:
        for asset in ASSET_MAP:
            try:
                _refresh_asset(asset, session)
            except Exception as exc:
                logger.error(f"refresh error for {asset}: {exc}", exc_info=True)
        time.sleep(refresh_secs)


def warmup() -> None:
    """Single-pass warmup for all assets. Called once at startup in a background thread."""
    session = requests.Session()
    for asset in ASSET_MAP:
        try:
            _refresh_asset(asset, session)
        except Exception as exc:
            logger.error(f"warmup error for {asset}: {exc}", exc_info=True)
    logger.info("Price fetcher warmup complete")


# ── Cache read API ─────────────────────────────────────────────────────────────

def get_spot(asset: str) -> Optional[tuple[float, datetime]]:
    """Return (price, fetch_time) or None if not cached."""
    with _cache_lock:
        return _spot_cache.get(asset)


def get_candles_1m(asset: str) -> Optional[tuple[list, datetime]]:
    """Return (closes_list, fetch_time) or None if not cached."""
    with _cache_lock:
        return _ohlcv_1m.get(asset)


def get_candles_1h(asset: str) -> Optional[tuple[list, datetime]]:
    """Return (closes_list, fetch_time) or None if not cached."""
    with _cache_lock:
        return _ohlcv_1h.get(asset)


def cache_age_secs(asset: str) -> int:
    """Return seconds since the spot price was last fetched. -1 if never cached."""
    entry = get_spot(asset)
    if entry is None:
        return -1
    _, fetch_time = entry
    return int((datetime.now(timezone.utc) - fetch_time).total_seconds())
