"""
Tests for crypto ticker parsing.

Tickers follow the pattern:
    KXBTCD-{YYMONDD}-{T|B}{THRESHOLD}

Examples:
    KXBTCD-26APR06-T67749.99   → asset=BTC, date=2026-04-06, above=True,  strike=67749.99
    KXETHD-26APR06-B2079.99    → asset=ETH, date=2026-04-06, above=False, strike=2079.99
    KXSOLD-26APR06-T80.9999    → asset=SOL, date=2026-04-06, above=True,  strike=80.9999
    KXXRPD-26APR06-T2.5        → asset=XRP, date=2026-04-06, above=True,  strike=2.5
"""

import sys
import os
from datetime import date

# Allow importing sidecar.py from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidecar import _parse_ticker


# ── Happy-path tests ──────────────────────────────────────────────────────────

def test_btc_above():
    asset, dt, strike, below = _parse_ticker("KXBTCD-26APR06-T67749.99")
    assert asset  == "BTC"
    assert dt     == date(2026, 4, 6)
    assert strike == 67749.99
    assert below  is False


def test_eth_below():
    asset, dt, strike, below = _parse_ticker("KXETHD-26APR06-B2079.99")
    assert asset  == "ETH"
    assert dt     == date(2026, 4, 6)
    assert strike == 2079.99
    assert below  is True


def test_sol_above():
    asset, dt, strike, below = _parse_ticker("KXSOLD-26APR06-T80.9999")
    assert asset  == "SOL"
    assert dt     == date(2026, 4, 6)
    assert strike == 80.9999
    assert below  is False


def test_xrp_above():
    asset, dt, strike, below = _parse_ticker("KXXRPD-26APR06-T2.5")
    assert asset  == "XRP"
    assert dt     == date(2026, 4, 6)
    assert strike == 2.5
    assert below  is False


def test_case_insensitive():
    asset, dt, strike, below = _parse_ticker("kxbtcd-26apr06-t67749.99")
    assert asset  == "BTC"
    assert dt     == date(2026, 4, 6)
    assert strike == 67749.99
    assert below  is False


def test_integer_strike():
    asset, dt, strike, below = _parse_ticker("KXBTCD-26APR06-T70000")
    assert asset  == "BTC"
    assert strike == 70000.0
    assert below  is False


def test_different_date():
    asset, dt, strike, below = _parse_ticker("KXETHD-26DEC31-T3500.0")
    assert asset == "ETH"
    assert dt    == date(2026, 12, 31)


def test_intraday_midnight():
    asset, dt, strike, below = _parse_ticker("KXBTCD-26APR0700-T68699.99")
    assert asset  == "BTC"
    assert dt     == date(2026, 4, 7)
    assert strike == 68699.99
    assert below  is False


def test_intraday_17h():
    asset, dt, strike, below = _parse_ticker("KXBTCD-26APR0717-T68499.99")
    assert asset  == "BTC"
    assert dt     == date(2026, 4, 7)
    assert strike == 68499.99


def test_intraday_eth():
    asset, dt, strike, below = _parse_ticker("KXETHD-26APR1017-T2089.99")
    assert asset  == "ETH"
    assert dt     == date(2026, 4, 10)
    assert strike == 2089.99


# ── Error / unrecognized cases ────────────────────────────────────────────────

def test_unrecognized_prefix_returns_none_asset():
    asset, dt, strike, below = _parse_ticker("KXHIGHTBOS-26APR06-T70")
    assert asset is None


def test_weather_ticker_not_recognized():
    asset, dt, strike, below = _parse_ticker("KXHIGHTDAL-26APR06-B84.5")
    assert asset is None


def test_wrong_prefix():
    asset, dt, strike, below = _parse_ticker("BTCD-26APR06-T67749.99")
    assert asset is None


def test_missing_threshold():
    # Only 2 dash-separated parts → parser can't find threshold, returns all None
    asset, dt, strike, below = _parse_ticker("KXBTCD-26APR06")
    assert asset   is None
    assert strike  is None


def test_bad_date():
    asset, dt, strike, below = _parse_ticker("KXBTCD-BADDATE-T67749.99")
    assert asset == "BTC"
    assert dt    is None


def test_empty_string():
    asset, dt, strike, below = _parse_ticker("")
    assert asset is None
