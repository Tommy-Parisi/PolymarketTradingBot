"""
GBM threshold-crossing probability predictor.

Core formula
------------
Treats the asset price as a geometric Brownian motion over the remaining
settlement window and computes the probability the price exceeds (or falls
below) the strike at settlement time.

    P(S_T > K) = N( d )

where:
    d  = (log(S/K) + (μ - σ²/2) · T) / (σ · √T)
    S  = current spot price
    K  = strike threshold from ticker
    T  = time to settlement in years (seconds_remaining / (365.25 * 86400))
    σ  = annualized realized volatility (from recent OHLCV)
    μ  = drift estimate (default 0.0 for intraday; can be calibrated)
    N  = standard normal CDF

For "below" markets (B prefix), return 1 - P(S_T > K).

Volatility estimation
---------------------
Uses close-to-close realized vol over recent candles. Multiple windows are
computed (15m, 1h, 4h) and the shortest window with sufficient data is used.
Intraday crypto vol is fast-moving so short windows are more predictive near
settlement.

    σ_realized = std(log(close_t / close_{t-1})) * sqrt(annualization_factor)

Drift
-----
For intraday settlement (T < 1 day), drift has minimal impact and defaults
to 0.0. Optionally estimate from a longer OHLCV window if desired.

Settlement time
---------------
All Kalshi daily crypto markets settle at 17:00 EDT (21:00 UTC or 22:00 UTC
depending on DST). Compute T from UTC now to next 17:00 EDT.

Bias correction
---------------
Optional calibration table (asset -> additive correction to d) fitted on
historical Kalshi vs. GBM prediction residuals. Start without it; add after
accumulating shadow predictions.

Example:
    prob = predict(spot=66300.0, strike=67000.0, seconds_remaining=7200,
                   candles_1m=[...], below=False)
    # → 0.31  (BTC needs to rally $700 in 2 hours, ~31% chance)
"""

import math
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SECONDS_PER_YEAR = 365.25 * 86400

# Minimum number of returns (candles - 1) for a vol window to be considered valid.
MIN_RETURNS_15M = 10
MIN_RETURNS_1H  = 30
MIN_RETURNS_4H  = 3

# Annualization factors: candles per year for each granularity.
CANDLES_PER_YEAR_1M = 365.25 * 24 * 60     # one-minute candles
CANDLES_PER_YEAR_1H = 365.25 * 24          # one-hour candles

# Floor for σ to prevent division-by-zero in degenerate cases (0.1% annualized).
MIN_VOL = 0.001

# Probability dampening: shrinks the raw GBM probability toward 0.5 before
# returning. The GBM model is correct in expectation but severely overconfident
# in practice — it assumes log-normal prices with no fat tails or jumps, so
# near-1 and near-0 outputs cause catastrophic Brier loss when the market
# moves against the model. A dampening factor of 0.75 compresses the distance
# from 0.5 by 25%: raw 0.97 → 0.854, raw 0.03 → 0.146.
# Tune via CRYPTO_PROB_DAMPENING env var after accumulating shadow predictions.
PROB_DAMPENING = float(os.getenv("CRYPTO_PROB_DAMPENING", "0.75"))

# Optional bias-correction table: asset → additive adjustment to d.
# Start at zero; tune after shadow accumulation.
BIAS_CORRECTIONS: dict[str, float] = {
    "BTC": 0.0,
    "ETH": 0.0,
    "SOL": 0.0,
    "XRP": 0.0,
}


# ── Vol estimation ─────────────────────────────────────────────────────────────

def _log_returns(closes: list[float]) -> list[float]:
    """Compute log returns from a list of close prices."""
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def _realized_vol(returns: list[float], candles_per_year: float) -> float:
    """Annualized realized vol from a list of log returns."""
    n = len(returns)
    if n < 2:
        return MIN_VOL
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return max(math.sqrt(variance * candles_per_year), MIN_VOL)


def estimate_vol(
    closes_1m: Optional[list[float]],
    closes_1h: Optional[list[float]],
) -> float:
    """
    Pick the best vol estimate from available candles.
    Prefer the shortest window with sufficient data — intraday crypto vol is
    fast-moving and a short recent window tracks regime better near settlement.

    Priority: 15m window (15 one-minute returns) → 1h window (60 1m returns)
              → 4h window (4 one-hour returns) → fallback constant.
    """
    if closes_1m and len(closes_1m) >= MIN_RETURNS_15M + 1:
        tail_15m  = closes_1m[-(MIN_RETURNS_15M + 1):]
        rets_15m  = _log_returns(tail_15m)
        vol_15m   = _realized_vol(rets_15m, CANDLES_PER_YEAR_1M)
        logger.debug(f"vol: 15m window σ={vol_15m:.4f}")
        return vol_15m

    if closes_1m and len(closes_1m) >= MIN_RETURNS_1H + 1:
        tail_1h  = closes_1m[-(MIN_RETURNS_1H + 1):]
        rets_1h  = _log_returns(tail_1h)
        vol_1h   = _realized_vol(rets_1h, CANDLES_PER_YEAR_1M)
        logger.debug(f"vol: 1h window (1m candles) σ={vol_1h:.4f}")
        return vol_1h

    if closes_1h and len(closes_1h) >= MIN_RETURNS_4H + 1:
        tail_4h  = closes_1h[-(MIN_RETURNS_4H + 1):]
        rets_4h  = _log_returns(tail_4h)
        vol_4h   = _realized_vol(rets_4h, CANDLES_PER_YEAR_1H)
        logger.debug(f"vol: 4h window (1h candles) σ={vol_4h:.4f}")
        return vol_4h

    # No sufficient data — return a conservative default (crypto 80% annual ≈ typical BTC)
    logger.warning("vol: insufficient candles, using default 0.80")
    return 0.80


# ── Normal CDF ────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc for accuracy."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ── Core predictor ─────────────────────────────────────────────────────────────

def predict(
    spot: float,
    strike: float,
    seconds_remaining: float,
    closes_1m: Optional[list[float]],
    closes_1h: Optional[list[float]],
    below: bool,
    asset: str = "",
    mu: float = 0.0,
) -> float:
    """
    Compute P(S_T > strike) using the GBM threshold-crossing formula.

    Args:
        spot:              Current spot price S.
        strike:            Kalshi market strike threshold K.
        seconds_remaining: Seconds until settlement at 17:00 EDT.
        closes_1m:         Recent 1-minute close prices for vol estimation.
        closes_1h:         Recent 1-hour close prices for vol estimation.
        below:             If True, return P(S_T < strike) = 1 - P(S_T > strike).
        asset:             Asset code (e.g. "BTC") for bias correction lookup.
        mu:                Drift estimate. Defaults to 0.0 (negligible intraday).

    Returns:
        Probability in (0, 1), clamped to [0.001, 0.999].
    """
    if spot <= 0 or strike <= 0:
        logger.warning(f"predict: invalid spot={spot} or strike={strike}")
        return 0.5

    T = seconds_remaining / SECONDS_PER_YEAR
    if T <= 0:
        # At or past settlement — price is effectively known.
        result = 1.0 if spot > strike else 0.0
        return (1.0 - result) if below else result

    sigma = estimate_vol(closes_1m, closes_1h)
    bias  = BIAS_CORRECTIONS.get(asset, 0.0)

    d = (math.log(spot / strike) + (mu - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d += bias

    prob = _norm_cdf(d)
    if below:
        prob = 1.0 - prob

    prob = max(0.001, min(0.999, prob))

    # Shrink toward 0.5 to correct GBM overconfidence (fat tails, vol jumps).
    prob = 0.5 + (prob - 0.5) * PROB_DAMPENING
    prob = max(0.001, min(0.999, prob))

    logger.debug(
        f"gbm  asset={asset}  spot={spot:.4f}  strike={strike:.4f}  "
        f"T={T*365.25:.4f}d  σ={sigma:.4f}  d={d:.4f}  prob={prob:.4f}  below={below}"
    )
    return prob
