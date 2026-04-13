"""
WeatherPredictor sidecar — HTTP service for the Kalshi trading bot.

Prediction path
---------------
Uses GEFS 31-member ensemble forecasts from NOMADS. A background thread fetches
and caches ensemble data for every supported city for today and tomorrow,
refreshing every GEFS_REFRESH_SECS.
The /predict endpoint reads from that cache and returns in well under the Rust
bot's 3-second timeout.

Response contract (motorcade standard)
---------------------------------------
    {
        "probability":    0.62,       # P(daily_high > floor_strike_f)
        "data_age_secs":  1800,       # seconds since GEFS data was fetched
        "data_source_ok": true,       # false → Rust bot falls back to bucket model
        "model_version":  "gefs_v1"
    }

data_source_ok is false when:
  - GEFS cache is empty (startup warmup not complete)
  - Cached data is older than MAX_DATA_AGE_SECS (default 2 h)
  - Fewer than MIN_MEMBERS_REQUIRED ensemble members succeeded

Endpoints
---------
    GET /health              → {"status": "ok", "cities": [...], "cache_keys": [...], "model_version": "gefs_v1"}
    GET /predict?ticker=...  → motorcade response contract above

Supported tickers
-----------------
New format:  KXHIGHT{CITY}-{DATE}-{T|B}{THRESHOLD}
  Examples:  KXHIGHTBOS-26APR01-T70, KXHIGHTDAL-26APR01-B84.5

Legacy format (Philadelphia): KXHIGH{PHI|PHIL|PHILLY|PHL}-{DATE}-{T|B}{THRESHOLD}

Environment variables
---------------------
    WEATHER_SIDECAR_HOST         Optional. Default: 127.0.0.1
    WEATHER_SIDECAR_PORT         Optional. Default: 8765
    GEFS_REFRESH_SECS            Optional. Default: 7200 (2 hours)
    GEFS_MAX_DATA_AGE_SECS       Optional. Default: 7200 (2 hours)
    GEFS_PREDICTION_LOG_DIR      Optional. Default: var/logs/gefs_predictions
"""

import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from gefs_fetcher import CityConfig, GEFSResult, fetch_ensemble_daily_highs, MEMBERS
from ensemble_predictor import predict as ensemble_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="WeatherPredictor Sidecar", version="3.0")

# ── Config ─────────────────────────────────────────────────────────────────────

GEFS_REFRESH_SECS  = int(os.getenv("GEFS_REFRESH_SECS",      "7200"))
MAX_DATA_AGE_SECS  = int(os.getenv("GEFS_MAX_DATA_AGE_SECS", "7200"))
PREDICTION_LOG_DIR = Path(os.getenv("GEFS_PREDICTION_LOG_DIR", "var/logs/gefs_predictions"))

# Minimum corrected-ensemble-mean distance from threshold (°F).
# When |corrected_mean - threshold| < this value the ensemble is straddling
# the threshold — forecast is near-coinflip and not actionable.
# Tune via GEFS_MIN_SPREAD_F. Default 5.0°F.
# Raised from 3.0 → 5.0 after Apr-10 analysis: all high-confidence ABOVE losses had
# ensemble overshoot of +0.2–+3.4°F — a 5°F filter blocks all of them.
GEFS_MIN_SPREAD_F = float(os.getenv("GEFS_MIN_SPREAD_F", "5.0"))

# Path for the on-disk cache snapshot. Loaded at startup so predictions are
# available immediately after a restart without waiting for warmup to complete.
# Set GEFS_CACHE_PATH to an absolute path for reliability.
GEFS_CACHE_PATH = Path(os.getenv("GEFS_CACHE_PATH", "var/cache/gefs_cache.json"))

MODEL_VERSION = "gefs_v2"

# ── City map ───────────────────────────────────────────────────────────────────
#
# Keys are the city codes extracted by _parse_ticker (everything alpha after "KXHIGH").
# New Kalshi format: KXHIGHT{CITY} → extracted code = "T" + city letters.
# Legacy Philly format: KXHIGH{PHI|PHIL|PHILLY|PHL} → extracted code = city letters.
#
# Coordinates are for the primary official weather station (usually ASOS airport).
# BBox is a 3° window centred on the station (1.5° each direction, rounded).

def _bbox(lat: float, lon: float, pad: float = 1.5) -> dict:
    return {
        "toplat":    str(round(lat + pad, 1)),
        "bottomlat": str(round(lat - pad, 1)),
        "leftlon":   str(round(lon - pad, 1)),
        "rightlon":  str(round(lon + pad, 1)),
    }

# ── Per-city bias corrections ──────────────────────────────────────────────────
#
# Applied to each ensemble member's raw forecast before voting.
# Format: { city_code: { month_int: bias_f } }
#
# Sign convention (matches ensemble_predictor.py which ADDS city_bias_f):
#   Positive = GEFS runs cold for this city/month → add degrees to warm up.
#   Negative = GEFS runs warm for this city/month → subtract degrees to cool down.
#
# Evidence-based values (derived from GEFS ensemble mean vs resolved outcomes,
# Apr 2026 data). All unlisted cities default to 0.0 until data is available.
#
# Confirmed GEFS warm bias (ensemble mean above threshold, outcome = NO):
#   Dallas  Apr-10: ensemble +3.4°F above T78, actual below. Correction: -3.5°F.
#   Houston Apr-10: ensemble +2.7°F above T77, actual below. Correction: -3.0°F.
#   OKC     Apr-10: ensemble +0.2°F above T78, actual below. Correction: -1.5°F.
#   LV      Apr-10: ensemble +0.3°F above T80, actual below. Correction: -1.0°F.
#   Atlanta Apr-10: ensemble +3.4°F above T76, actual below. Correction: -2.5°F (1 data point).
#
# Cities with insufficient data are set to 0.0 to avoid introducing new bias.
# Refine monthly as more GEFS prediction logs accumulate against resolved outcomes.
CITY_WARM_BIAS_F: dict[str, dict[int, float]] = {
    "TDAL":  {4: -3.5, 3: -3.5},
    "THOU":  {4: -3.0, 3: -3.0},
    "TOKC":  {4: -1.5, 3: -1.5},
    "TLV":   {4: -1.0, 3: -1.0},
    # ATL Apr-10: ensemble +3.4°F above T76, actual below. One data point — conservative.
    "TATL":  {4: -2.5, 3: -2.5},
    # All other cities: 0.0 (no correction) until more data is available.
    # Previous values for TBOS, TSEA, TPHX, TSATX, TMIN, TNOLA, TDC, TSFO
    # were positive (warm-up) but lacked confirming evidence and may have been
    # counterproductive. Re-add only after GEFS-vs-outcome residuals confirm direction.
}

CITY_MAP: dict[str, CityConfig] = {
    # ── New-format cities (KXHIGHT prefix) ──────────────────────────────────
    "TBOS":  CityConfig("Boston",          42.36, -71.01, _bbox(42.36, -71.01)),
    "TDAL":  CityConfig("Dallas",          32.90, -97.04, _bbox(32.90, -97.04)),
    "THOU":  CityConfig("Houston",         29.99, -95.34, _bbox(29.99, -95.34)),
    "TSEA":  CityConfig("Seattle",         47.45, -122.31, _bbox(47.45, -122.31)),
    "TPHX":  CityConfig("Phoenix",         33.44, -112.01, _bbox(33.44, -112.01)),
    "TSATX": CityConfig("San Antonio",     29.53, -98.47, _bbox(29.53, -98.47)),
    "TLV":   CityConfig("Las Vegas",       36.08, -115.15, _bbox(36.08, -115.15)),
    "TATL":  CityConfig("Atlanta",         33.64,  -84.43, _bbox(33.64,  -84.43)),
    "TMIN":  CityConfig("Minneapolis",     44.88,  -93.22, _bbox(44.88,  -93.22)),
    "TNOLA": CityConfig("New Orleans",     29.99,  -90.26, _bbox(29.99,  -90.26)),
    "TDC":   CityConfig("Washington DC",   38.85,  -77.04, _bbox(38.85,  -77.04)),
    "TSFO":  CityConfig("San Francisco",   37.62, -122.38, _bbox(37.62, -122.38)),
    "TOKC":  CityConfig("Oklahoma City",   35.39,  -97.60, _bbox(35.39,  -97.60)),
    # ── Legacy Philadelphia codes (KXHIGH{PHI|PHIL|PHILLY|PHL}) ─────────────
    "PHI":   CityConfig("Philadelphia",    39.87,  -75.24, _bbox(39.87, -75.24)),
    "PHIL":  CityConfig("Philadelphia",    39.87,  -75.24, _bbox(39.87, -75.24)),
    "PHILLY":CityConfig("Philadelphia",    39.87,  -75.24, _bbox(39.87, -75.24)),
    "PHL":   CityConfig("Philadelphia",    39.87,  -75.24, _bbox(39.87, -75.24)),
}

# ── Prediction log ─────────────────────────────────────────────────────────────

_log_lock = threading.Lock()


def _write_prediction_log(record: dict) -> None:
    """Append one prediction record to today's JSONL log. Silently swallows errors."""
    try:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = PREDICTION_LOG_DIR / f"predictions_{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _log_lock:
            with open(path, "a") as f:
                f.write(line)
    except Exception as exc:
        logger.warning(f"prediction log write failed: {exc}")


# ── Cache ──────────────────────────────────────────────────────────────────────
#
# Keyed by (city_code, target_date). Each value is a GEFSResult.
# A single lock guards all reads and writes.

_cache: dict[tuple[str, date], GEFSResult] = {}
_cache_lock = threading.Lock()


def _save_disk_cache() -> None:
    """Persist current cache to GEFS_CACHE_PATH as JSON. Called after every refresh."""
    try:
        GEFS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {}
        with _cache_lock:
            for (city_code, target_date), result in _cache.items():
                key = f"{city_code}|{target_date}"
                snapshot[key] = {
                    "city_code":           city_code,
                    "target_date":         str(target_date),
                    "member_highs_f":      result.member_highs_f,
                    "run_time":            result.run_time.isoformat(),
                    "fetch_time":          result.fetch_time.isoformat(),
                    "n_members":           result.n_members,
                    "city":                result.city,
                    "forecast_hours_used": result.forecast_hours_used,
                }
        tmp = GEFS_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot))
        tmp.replace(GEFS_CACHE_PATH)
    except Exception as exc:
        logger.warning(f"disk cache save failed: {exc}")


def _load_disk_cache() -> int:
    """Load cache from GEFS_CACHE_PATH into _cache. Skips entries older than MAX_DATA_AGE_SECS.
    Returns the number of entries loaded."""
    if not GEFS_CACHE_PATH.exists():
        return 0
    try:
        snapshot = json.loads(GEFS_CACHE_PATH.read_text())
        now = datetime.now(timezone.utc)
        loaded = 0
        with _cache_lock:
            for key, rec in snapshot.items():
                fetch_time = datetime.fromisoformat(rec["fetch_time"])
                age_secs = (now - fetch_time).total_seconds()
                if age_secs > MAX_DATA_AGE_SECS:
                    continue
                target_date = date.fromisoformat(rec["target_date"])
                result = GEFSResult(
                    member_highs_f      = rec["member_highs_f"],
                    run_time            = datetime.fromisoformat(rec["run_time"]),
                    target_date         = target_date,
                    fetch_time          = fetch_time,
                    n_members           = rec["n_members"],
                    city                = rec["city"],
                    forecast_hours_used = rec.get("forecast_hours_used", []),
                )
                city_code = rec["city_code"]
                _cache[(city_code, target_date)] = result
                loaded += 1
        return loaded
    except Exception as exc:
        logger.warning(f"disk cache load failed: {exc}")
        return 0


def _refresh(city_code: str, target_date: date) -> None:
    """Fetch GEFS data for one (city, target_date) and store in cache."""
    city_cfg = CITY_MAP[city_code]
    logger.info(f"GEFS refresh: city={city_cfg.name} date={target_date}")
    result = fetch_ensemble_daily_highs(target_date, city_cfg)
    if result is not None:
        with _cache_lock:
            _cache[(city_code, target_date)] = result
        logger.info(
            f"GEFS cache updated: city={city_cfg.name} date={target_date} "
            f"members={result.n_members}"
        )
        _save_disk_cache()
    else:
        logger.warning(f"GEFS refresh failed: city={city_cfg.name} date={target_date}")


def _background_refresh() -> None:
    """Background thread: refresh all cities × {-1, today, +1, +2} every GEFS_REFRESH_SECS.
    Covers today-1 because markets for yesterday can still be open awaiting resolution.
    Covers today+2 because weather markets open ~14:00 UTC roughly 2 days before resolution.
    Sleeps first so startup warmup and first background fetch don't overlap."""
    time.sleep(GEFS_REFRESH_SECS)
    while True:
        today = datetime.now(timezone.utc).date()
        for city_code in CITY_MAP:
            # Deduplicate: multiple Philly aliases all point to same config — skip dupes.
            if CITY_MAP[city_code].name == "Philadelphia" and city_code != "PHI":
                continue
            for target_date in [today - timedelta(days=1), today, today + timedelta(days=1), today + timedelta(days=2)]:
                try:
                    _refresh(city_code, target_date)
                except Exception as exc:
                    logger.error(
                        f"GEFS refresh error: city={city_code} date={target_date}: {exc}",
                        exc_info=True,
                    )
        time.sleep(GEFS_REFRESH_SECS)


# ── Ticker parsing ─────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> tuple[Optional[str], Optional[date], Optional[float], bool]:
    """
    Parse a Kalshi weather ticker into (city_code, target_date, floor_strike_f, below).

    New format:     KXHIGHT{CITY}-{DATE}-{T|B}{THRESHOLD}
    Legacy format:  KXHIGH{CITY}-{DATE}-{T|B}{THRESHOLD}

    Examples:
        KXHIGHTBOS-26APR01-T70   → ("TBOS",  date(2026, 4, 1),   70.0, False)
        KXHIGHTDAL-26APR01-B84.5 → ("TDAL",  date(2026, 4, 1),   84.5, True)
        KXHIGHPHI-26APR15-T55    → ("PHI",   date(2026, 4, 15),  55.0, False)
    """
    upper = ticker.upper()
    if not upper.startswith("KXHIGH"):
        return None, None, None, False

    rest     = upper[len("KXHIGH"):]
    city_end = next((i for i, c in enumerate(rest) if not c.isalpha()), len(rest))
    city     = rest[:city_end]

    date_match = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", upper)
    target_date = None
    if date_match:
        try:
            target_date = datetime.strptime(date_match.group(1), "%y%b%d").date()
        except ValueError:
            pass

    thresh_match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", upper)
    floor_strike_f = float(thresh_match.group(2)) if thresh_match else None
    below = thresh_match.group(1) == "B" if thresh_match else False

    return city, target_date, floor_strike_f, below


# ── Routes ─────────────────────────────────────────────────────────────────────

def _startup_warmup() -> None:
    """Warm the cache for all cities in the background so uvicorn starts immediately.
    Runs once at startup, then hands off to _background_refresh."""
    today = datetime.now(timezone.utc).date()
    seen_names: set[str] = set()
    for city_code, city_cfg in CITY_MAP.items():
        if city_cfg.name in seen_names:
            continue
        seen_names.add(city_cfg.name)
        for target_date in [today - timedelta(days=1), today, today + timedelta(days=1), today + timedelta(days=2)]:
            try:
                _refresh(city_code, target_date)
            except Exception as exc:
                logger.error(
                    f"Startup GEFS fetch failed: city={city_cfg.name} date={target_date}: {exc}"
                )
    logger.info("Startup warmup complete")


@app.on_event("startup")
def on_startup():
    # Load disk cache first — gives immediate coverage while warmup fetches fresh data.
    n = _load_disk_cache()
    if n:
        logger.info(f"Startup: loaded {n} entries from disk cache ({GEFS_CACHE_PATH})")
    else:
        logger.info(f"Startup: no usable disk cache found at {GEFS_CACHE_PATH}")
    # Warmup runs in background — uvicorn is ready to serve immediately.
    threading.Thread(target=_startup_warmup, daemon=True).start()
    threading.Thread(target=_background_refresh, daemon=True).start()
    logger.info("Startup: warmup + background refresh threads started")


@app.get("/health")
def health():
    with _cache_lock:
        cache_keys = sorted(f"{c}:{d}" for c, d in _cache.keys())
    return {
        "status":        "ok",
        "cities":        sorted(set(cfg.name for cfg in CITY_MAP.values())),
        "cache_keys":    cache_keys,
        "model_version": MODEL_VERSION,
    }


@app.get("/predict")
def predict(ticker: str):
    city_code, target_date, floor_strike_f, below = _parse_ticker(ticker)

    if city_code is None or city_code not in CITY_MAP:
        raise HTTPException(404, f"Unsupported city code '{city_code}' in ticker '{ticker}'")
    if target_date is None:
        raise HTTPException(400, f"Cannot parse target date from ticker '{ticker}'")
    if floor_strike_f is None:
        raise HTTPException(400, f"Cannot parse floor_strike_f from ticker '{ticker}'")

    with _cache_lock:
        result = _cache.get((city_code, target_date))

    if result is None:
        logger.warning(f"predict: cache miss for city={city_code} date={target_date} ({ticker})")
        return {
            "probability":    0.5,
            "data_age_secs":  -1,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    data_age_secs = int((datetime.now(timezone.utc) - result.fetch_time).total_seconds())

    # For past target dates the weather already happened — a refetch would not
    # produce better data, so the age limit does not apply.
    today = datetime.now(timezone.utc).date()
    if target_date >= today and data_age_secs > MAX_DATA_AGE_SECS:
        logger.warning(
            f"predict: stale cache for city={city_code} date={target_date} "
            f"(age={data_age_secs}s > max={MAX_DATA_AGE_SECS}s)"
        )
        return {
            "probability":    0.5,
            "data_age_secs":  data_age_secs,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    month = target_date.month
    city_bias = CITY_WARM_BIAS_F.get(city_code, {}).get(month, 0.0)

    # Spread filter: reject predictions where the corrected ensemble mean is too
    # close to the threshold. At ±3°F the ensemble straddles the line — accuracy
    # drops sharply and the signal is near-coinflip. Return data_source_ok=false
    # so the Rust bot falls back to the bucket model rather than acting on noise.
    corrected_mean = (
        sum(result.member_highs_f) / len(result.member_highs_f)
    ) + city_bias
    ensemble_spread = abs(corrected_mean - floor_strike_f)
    if ensemble_spread < GEFS_MIN_SPREAD_F:
        logger.warning(
            f"predict: spread too small for {ticker} "
            f"(corrected_mean={corrected_mean:.1f}°F threshold={floor_strike_f}°F "
            f"spread={ensemble_spread:.1f}°F < min={GEFS_MIN_SPREAD_F}°F) — suppressing"
        )
        return {
            "probability":    0.5,
            "data_age_secs":  data_age_secs,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    prob = ensemble_predict(
        member_highs_f=result.member_highs_f,
        floor_strike_f=floor_strike_f,
        target_date=target_date,
        members=MEMBERS[:result.n_members],
        city_bias_f=city_bias,
    )
    if below:
        prob = 1.0 - prob

    logger.info(
        f"predict  ticker={ticker}  city={city_code}  threshold={floor_strike_f}°F  "
        f"below={below}  prob={prob:.4f}  members={result.n_members}  "
        f"city_bias={city_bias:+.1f}°F  age={data_age_secs}s"
    )

    _write_prediction_log({
        "ts":             datetime.now(timezone.utc).isoformat(),
        "ticker":         ticker,
        "city":           city_code,
        "target_date":    str(target_date),
        "threshold_f":    floor_strike_f,
        "probability":    prob,
        "n_members":      result.n_members,
        "member_highs_f": result.member_highs_f,
        "run_time":       result.run_time.isoformat(),
        "data_age_secs":  data_age_secs,
        "city_bias_f":    city_bias,
        "model_version":  MODEL_VERSION,
    })

    return {
        "probability":    prob,
        "data_age_secs":  data_age_secs,
        "data_source_ok": True,
        "model_version":  MODEL_VERSION,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEATHER_SIDECAR_PORT", "8765"))
    host = os.getenv("WEATHER_SIDECAR_HOST", "127.0.0.1")
    uvicorn.run("sidecar:app", host=host, port=port, reload=False)
