#!/usr/bin/env python3
"""
Render a standalone HTML dashboard for sidecar vertical performance.

The dashboard is built from the same append-only research/output files used by
the existing CLI evaluator, but it emits a self-contained HTML snapshot with
interactive filters and recent executed-trade tables for weather, crypto, and
FED sidecars.
"""

from __future__ import annotations

import argparse
import bisect
import html
import json
import math
import os
from collections import defaultdict
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESEARCH_DIR = REPO_ROOT / "var" / "research"
DEFAULT_OUTPUT = REPO_ROOT / "vertical_dashboard.html"

CRYPTO_PREFIXES = ("KXBTCD", "KXETHD", "KXSOLD", "KXXRPD")
WEATHER_PREFIXES = ("KXHIGHT",)
FED_PREFIXES = ("KXFED", "KXFOMC")
FEE_RATE = 0.07
TRADE_TABLE_LIMIT = 18
SIDECAR_ACTIVE_MAX_AGE_MINUTES = 20
BOT_ACTIVE_MAX_AGE_MINUTES = 15

SIDECARS = (
    {
        "key": "weather",
        "label": "Weather",
        "accent": "#2ec4b6",
        "description": "GEFS ensemble override for KXHIGHT* markets",
        "prediction_dir": REPO_ROOT / "sidecars" / "weather" / "var" / "logs" / "gefs_predictions",
    },
    {
        "key": "crypto",
        "label": "Crypto",
        "accent": "#ff9f1c",
        "description": "GBM threshold-crossing override for crypto daily markets",
        "prediction_dir": REPO_ROOT / "sidecars" / "crypto" / "var" / "logs" / "crypto_predictions",
    },
    {
        "key": "fed",
        "label": "Fed",
        "accent": "#e71d36",
        "description": "TF-IDF + MLP override for KXFED/KXFOMC markets",
        "prediction_dir": REPO_ROOT / "sidecars" / "hawkwatchers" / "var" / "logs" / "fed_predictions",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-dir",
        default=str(DEFAULT_RESEARCH_DIR),
        help="Research directory to read (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output HTML path (default: %(default)s)",
    )
    parser.add_argument(
        "--since",
        default="0000-00-00",
        help="Only include rows on or after DATE (YYYY-MM-DD). Default: all time.",
    )
    parser.add_argument(
        "--auto-refresh-seconds",
        type=int,
        default=60,
        help="Client-side browser refresh interval in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--bot-log-path",
        default=os.environ.get("VERTICAL_DASHBOARD_BOT_LOG_PATH", ""),
        help="Optional path to a main bot log file to tail into the dashboard.",
    )
    parser.add_argument(
        "--bot-log-lines",
        type=int,
        default=60,
        help="How many trailing log lines to embed when --bot-log-path is set.",
    )
    return parser.parse_args()


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def isoformat(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def vertical_for_ticker(ticker: str) -> str | None:
    if ticker.startswith(CRYPTO_PREFIXES):
        return "crypto"
    if ticker.startswith(WEATHER_PREFIXES):
        return "weather"
    if ticker.startswith(FED_PREFIXES):
        return "fed"
    return None


def effective_prob(ticker: str, probability: float | None) -> float | None:
    if probability is None:
        return None
    # Both sidecars (weather, crypto) already log P(YES) directly — the weather
    # sidecar does `if below: prob = 1 - prob` before writing, and the crypto
    # predictor does the same. No inversion is needed here.
    return probability


def order_won(order: dict[str, Any], outcome_yes: bool) -> bool:
    side = str(order.get("side", "")).lower()
    outcome_id = str(order.get("outcome_id", "yes")).lower()
    if side == "buy":
        return outcome_yes == (outcome_id == "yes")
    return outcome_yes != (outcome_id == "yes")


def calc_pnl(order: dict[str, Any], won: bool) -> float:
    fill = safe_float(order.get("avg_fill_price")) or 0.0
    qty = safe_float(order.get("filled_qty")) or 0.0
    fee = safe_float(order.get("fee_paid"))
    if fee is None:
        fee = fill * qty * FEE_RATE
    cost = fill * qty
    return (qty - cost - fee) if won else -(cost + fee)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_outcomes(path: Path) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for row in load_jsonl(path):
        ticker = row.get("ticker")
        if ticker is None:
            continue
        outcomes[str(ticker)] = bool(row.get("result", False))
    return outcomes


def load_filled_orders(base: Path, since: str) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    if not base.exists():
        return []

    for date_dir in sorted(base.iterdir()):
        if not date_dir.is_dir() or date_dir.name < since:
            continue
        for path in sorted(date_dir.glob("*.jsonl")):
            for row in load_jsonl(path):
                ticker = str(row.get("ticker", ""))
                if vertical_for_ticker(ticker) is None:
                    continue
                filled_qty = safe_float(row.get("filled_qty")) or 0.0
                if filled_qty <= 0:
                    continue
                client_order_id = str(row.get("client_order_id") or "")
                key = client_order_id or f"{ticker}:{row.get('ts')}:{filled_qty}"
                existing = deduped.get(key)
                row_ts = parse_ts(row.get("ts"))
                row["_parsed_ts"] = isoformat(row_ts)
                if existing is None:
                    deduped[key] = row
                    continue
                existing_qty = safe_float(existing.get("filled_qty")) or 0.0
                existing_ts = parse_ts(existing.get("ts"))
                should_replace = filled_qty > existing_qty or (
                    math.isclose(filled_qty, existing_qty)
                    and (row_ts or datetime.min.replace(tzinfo=UTC))
                    > (existing_ts or datetime.min.replace(tzinfo=UTC))
                )
                if should_replace:
                    deduped[key] = row

    rows = list(deduped.values())
    rows.sort(key=lambda row: row.get("_parsed_ts") or "")
    return rows


def load_market_snapshots(base: Path, since: str) -> dict[str, dict[str, list[Any]]]:
    raw: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    if not base.exists():
        return {}

    for date_dir in sorted(base.iterdir()):
        if not date_dir.is_dir() or date_dir.name < since:
            continue
        for path in sorted(date_dir.glob("*.jsonl")):
            for row in load_jsonl(path):
                ticker = str(row.get("ticker", ""))
                vertical = vertical_for_ticker(ticker)
                if vertical is None:
                    continue
                if (
                    row.get("specialist_prob_yes") is None
                    and row.get("crypto_specialist_prob_yes") is None
                    and row.get("fed_specialist_prob_yes") is None
                ):
                    continue
                ts = parse_ts(row.get("ts"))
                if ts is None:
                    continue
                raw[ticker].append((ts, row))

    indexed: dict[str, dict[str, list[Any]]] = {}
    for ticker, rows in raw.items():
        rows.sort(key=lambda item: item[0])
        indexed[ticker] = {
            "ts": [item[0].timestamp() for item in rows],
            "rows": [item[1] for item in rows],
        }
    return indexed


def find_snapshot(indexed_snapshots: dict[str, dict[str, list[Any]]], ticker: str, order_ts: datetime | None) -> dict[str, Any] | None:
    if order_ts is None:
        return None
    ticker_rows = indexed_snapshots.get(ticker)
    if not ticker_rows:
        return None
    timestamps = ticker_rows["ts"]
    rows = ticker_rows["rows"]
    pos = bisect.bisect_right(timestamps, order_ts.timestamp()) - 1
    if pos >= 0:
        return rows[pos]
    return rows[0] if rows else None


def load_sidecar_predictions(pred_dir: Path, vertical: str, since: str, outcomes: dict[str, bool]) -> list[dict[str, Any]]:
    latest_by_ticker: dict[str, dict[str, Any]] = {}
    if not pred_dir.exists():
        return []

    for path in sorted(pred_dir.glob("predictions_*.jsonl")):
        day = path.stem.replace("predictions_", "")
        if day < since:
            continue
        for row in load_jsonl(path):
            ticker = str(row.get("ticker", ""))
            if vertical_for_ticker(ticker) != vertical:
                continue
            ts = parse_ts(row.get("ts"))
            probability = safe_float(row.get("probability"))
            payload = {
                "ticker": ticker,
                "ts": isoformat(ts),
                "vertical": vertical,
                "probability": probability,
                "effective_probability": effective_prob(ticker, probability),
                "resolved": ticker in outcomes,
                "outcome_yes": outcomes.get(ticker),
            }
            existing = latest_by_ticker.get(ticker)
            if existing is None or (payload["ts"] or "") >= (existing["ts"] or ""):
                latest_by_ticker[ticker] = payload

    rows = list(latest_by_ticker.values())
    rows.sort(key=lambda row: row["ts"] or "")
    return rows


def build_trade_rows(
    orders: list[dict[str, Any]],
    outcomes: dict[str, bool],
    market_snapshots: dict[str, dict[str, list[Any]]],
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for order in orders:
        ticker = str(order.get("ticker", ""))
        vertical = vertical_for_ticker(ticker)
        if vertical is None:
            continue
        ts = parse_ts(order.get("ts"))
        snapshot = find_snapshot(market_snapshots, ticker, ts)
        outcome_yes = outcomes.get(ticker)
        resolved = ticker in outcomes
        won = order_won(order, outcome_yes) if resolved else None
        pnl = calc_pnl(order, won) if resolved and won is not None else None
        fill_price = safe_float(order.get("avg_fill_price"))
        filled_qty = safe_float(order.get("filled_qty")) or 0.0
        notional = (fill_price or 0.0) * filled_qty
        specialist_prob = None
        if snapshot is not None:
            if vertical == "weather":
                specialist_prob = safe_float(snapshot.get("specialist_prob_yes"))
            elif vertical == "crypto":
                specialist_prob = safe_float(snapshot.get("crypto_specialist_prob_yes"))
            elif vertical == "fed":
                specialist_prob = safe_float(snapshot.get("fed_specialist_prob_yes"))
        trades.append(
            {
                "client_order_id": order.get("client_order_id"),
                "ticker": ticker,
                "vertical": vertical,
                "ts": isoformat(ts),
                "close_time": snapshot.get("close_time") if snapshot else None,
                "title": snapshot.get("title") if snapshot else None,
                "series_ticker": snapshot.get("series_ticker") if snapshot else ticker.split("-")[0],
                "outcome_id": str(order.get("outcome_id", "")).lower(),
                "side": str(order.get("side", "")).lower(),
                "execution_mode": order.get("execution_mode"),
                "signal_origin": order.get("signal_origin"),
                "filled_qty": filled_qty,
                "avg_fill_price": fill_price,
                "fee_paid": safe_float(order.get("fee_paid")),
                "notional": notional,
                "signal_fair_price": safe_float(order.get("signal_fair_price")),
                "signal_observed_price": safe_float(order.get("signal_observed_price")),
                "signal_edge_pct": safe_float(order.get("signal_edge_pct")),
                "signal_confidence": safe_float(order.get("signal_confidence")),
                "specialist_prob_yes": specialist_prob,
                "effective_specialist_prob_yes": effective_prob(ticker, specialist_prob),
                "resolved": resolved,
                "outcome_yes": outcome_yes,
                "won": won,
                "pnl": pnl,
            }
        )
    trades.sort(key=lambda row: row["ts"] or "", reverse=True)
    return trades


def count_jsonl_files(base: Path, since: str) -> int:
    if not base.exists():
        return 0
    count = 0
    for date_dir in sorted(base.iterdir()):
        if date_dir.is_dir() and date_dir.name >= since:
            count += len(list(date_dir.glob("*.jsonl")))
    return count


def count_prediction_files(base: Path, since: str) -> int:
    if not base.exists():
        return 0
    count = 0
    for path in base.glob("predictions_*.jsonl"):
        day = path.stem.replace("predictions_", "")
        if day >= since:
            count += 1
    return count


def latest_ts_from_rows(rows: list[dict[str, Any]]) -> datetime | None:
    timestamps = [parse_ts(row.get("ts")) for row in rows]
    valid = [ts for ts in timestamps if ts is not None]
    return max(valid) if valid else None


def latest_ts_in_tree(base: Path, since: str) -> datetime | None:
    latest: datetime | None = None
    if not base.exists():
        return None
    for date_dir in sorted(base.iterdir()):
        if not date_dir.is_dir() or date_dir.name < since:
            continue
        for path in sorted(date_dir.glob("*.jsonl")):
            for row in load_jsonl(path):
                ts = parse_ts(row.get("ts"))
                if ts is not None and (latest is None or ts > latest):
                    latest = ts
    return latest


def status_payload(label: str, latest_ts: datetime | None, max_age_minutes: int, now: datetime) -> dict[str, Any]:
    if latest_ts is None:
        return {
            "label": label,
            "state": "inactive",
            "text": "No recent data",
            "latest_ts": None,
            "age_minutes": None,
            "max_age_minutes": max_age_minutes,
        }
    age_minutes = max(0.0, (now - latest_ts).total_seconds() / 60.0)
    state = "active" if age_minutes <= max_age_minutes else "stale"
    text = (
        f"Active, updated {age_minutes:.1f}m ago"
        if state == "active"
        else f"Stale, last update {age_minutes:.1f}m ago"
    )
    return {
        "label": label,
        "state": state,
        "text": text,
        "latest_ts": isoformat(latest_ts),
        "age_minutes": age_minutes,
        "max_age_minutes": max_age_minutes,
    }


def read_log_tail(path: Path | None, line_count: int) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "lines": [],
            "line_count": 0,
            "updated_at": None,
            "error": None,
        }
    try:
        exists = path.exists()
        if not exists:
            return {
                "path": str(path),
                "exists": False,
                "lines": [],
                "line_count": 0,
                "updated_at": None,
                "error": None,
            }
        tail = deque(maxlen=max(1, line_count))
        with path.open(errors="replace") as handle:
            for line in handle:
                tail.append(line.rstrip("\n"))
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return {
            "path": str(path),
            "exists": True,
            "lines": list(tail),
            "line_count": len(tail),
            "updated_at": isoformat(updated_at),
            "error": None,
        }
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "lines": [],
            "line_count": 0,
            "updated_at": None,
            "error": str(exc),
        }


def build_payload(research_dir: Path, since: str, bot_log_path: Path | None = None, bot_log_lines: int = 60) -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    outcomes = load_outcomes(research_dir / "outcomes" / "outcomes.jsonl")
    orders = load_filled_orders(research_dir / "order_lifecycle", since)
    market_snapshots = load_market_snapshots(research_dir / "market_state", since)
    trades = build_trade_rows(orders, outcomes, market_snapshots)

    predictions: list[dict[str, Any]] = []
    prediction_sources: dict[str, Any] = {}
    sidecar_status: dict[str, Any] = {}
    for sidecar in SIDECARS:
        rows = load_sidecar_predictions(sidecar["prediction_dir"], sidecar["key"], since, outcomes)
        predictions.extend(rows)
        latest_prediction_ts = latest_ts_from_rows(rows)
        prediction_sources[sidecar["key"]] = {
            "dir": str(sidecar["prediction_dir"]),
            "file_count": count_prediction_files(sidecar["prediction_dir"], since),
            "prediction_count": len(rows),
            "latest_prediction_ts": isoformat(latest_prediction_ts),
        }
        sidecar_status[sidecar["key"]] = status_payload(
            sidecar["label"],
            latest_prediction_ts,
            SIDECAR_ACTIVE_MAX_AGE_MINUTES,
            now,
        )

    latest_market_state_ts = latest_ts_in_tree(research_dir / "market_state", since)
    latest_order_ts = latest_ts_from_rows(orders)
    bot_latest_ts_candidates = [ts for ts in (latest_market_state_ts, latest_order_ts) if ts is not None]
    bot_latest_ts = max(bot_latest_ts_candidates) if bot_latest_ts_candidates else None
    bot_status = status_payload("Main bot", bot_latest_ts, BOT_ACTIVE_MAX_AGE_MINUTES, now)
    bot_status["signals"] = {
        "latest_market_state_ts": isoformat(latest_market_state_ts),
        "latest_order_ts": isoformat(latest_order_ts),
    }
    bot_log = read_log_tail(bot_log_path, bot_log_lines)

    return {
        "generated_at": isoformat(now),
        "since": since,
        "research_dir": str(research_dir),
        "sidecars": [
            {
                "key": sidecar["key"],
                "label": sidecar["label"],
                "accent": sidecar["accent"],
                "description": sidecar["description"],
                "prediction_dir": str(sidecar["prediction_dir"]),
            }
            for sidecar in SIDECARS
        ],
        "source_counts": {
            "outcome_rows": len(outcomes),
            "filled_orders": len(orders),
            "trade_rows": len(trades),
            "market_state_files": count_jsonl_files(research_dir / "market_state", since),
            "order_lifecycle_files": count_jsonl_files(research_dir / "order_lifecycle", since),
            "prediction_sources": prediction_sources,
        },
        "status": {
            "bot": bot_status,
            "sidecars": sidecar_status,
        },
        "bot_log": bot_log,
        "trades": trades,
        "predictions": predictions,
    }


def render_html(payload: dict[str, Any], auto_refresh_seconds: int) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    generated_at = html.escape(payload.get("generated_at") or "unknown")
    since_label = "all time" if payload.get("since") == "0000-00-00" else html.escape(payload["since"])
    research_dir = html.escape(payload.get("research_dir") or "")
    auto_refresh_meta = (
        f'  <meta http-equiv="refresh" content="{auto_refresh_seconds}">\n'
        if auto_refresh_seconds > 0
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
{auto_refresh_meta}  <meta name="auto-refresh-seconds" content="{auto_refresh_seconds}">
  <title>Motorcade Vertical Dashboard</title>
  <style>
    :root {{
      --bg: #08111c;
      --panel: rgba(10, 19, 32, 0.86);
      --panel-strong: #0f1c2e;
      --muted: #8ea3bd;
      --text: #eff6ff;
      --border: rgba(173, 201, 235, 0.14);
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
      --good: #29d391;
      --bad: #ff6b6b;
      --warn: #ffd166;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(46, 196, 182, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 159, 28, 0.16), transparent 30%),
        linear-gradient(180deg, #09101a 0%, #060c14 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1480px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero {{
      background: linear-gradient(145deg, rgba(12, 25, 42, 0.95), rgba(7, 15, 27, 0.92));
      border: 1px solid var(--border);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(46, 196, 182, 0.12), transparent 35%),
        linear-gradient(120deg, transparent 55%, rgba(255, 159, 28, 0.09), transparent 80%);
      pointer-events: none;
    }}
    .hero-inner {{
      padding: 28px;
      position: relative;
      z-index: 1;
      display: grid;
      gap: 22px;
    }}
    .eyebrow {{
      color: #7ce4d8;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-weight: 700;
    }}
    h1 {{
      margin: 8px 0 10px;
      font-size: clamp(2rem, 3.8vw, 3.4rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
      max-width: 12ch;
    }}
    .lede {{
      color: #c1d0e2;
      max-width: 70ch;
      margin: 0;
      line-height: 1.55;
      font-size: 1rem;
    }}
    .hero-meta, .range-picker, .summary-grid, .section-grid, .mini-grid, .table-wrap {{
      display: grid;
      gap: 14px;
    }}
    .hero-meta {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .meta-chip, .range-picker button {{
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      padding: 10px 14px;
      font-size: 0.93rem;
    }}
    .status-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 0.88rem;
      color: var(--text);
      white-space: nowrap;
    }}
    .status-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--warn);
      box-shadow: 0 0 0 4px rgba(255, 209, 102, 0.12);
      flex: 0 0 auto;
    }}
    .status-pill.active .status-dot {{
      background: var(--good);
      box-shadow: 0 0 0 4px rgba(41, 211, 145, 0.12);
    }}
    .status-pill.stale .status-dot {{
      background: var(--warn);
      box-shadow: 0 0 0 4px rgba(255, 209, 102, 0.12);
    }}
    .status-pill.inactive .status-dot {{
      background: var(--bad);
      box-shadow: 0 0 0 4px rgba(255, 107, 107, 0.12);
    }}
    .range-picker {{
      grid-template-columns: repeat(auto-fit, minmax(120px, max-content));
      align-items: start;
    }}
    .range-picker button {{
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
      font-weight: 600;
      text-align: left;
    }}
    .range-picker button:hover {{
      transform: translateY(-1px);
      border-color: rgba(255, 255, 255, 0.3);
    }}
    .range-picker button.active {{
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.4);
    }}
    .summary-grid {{
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin-top: 24px;
    }}
    .stat-card, .section-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .stat-card {{
      padding: 18px 18px 16px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: clamp(1.5rem, 2vw, 2.2rem);
      line-height: 1;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
    }}
    .stat-sub {{
      color: #c8d7e7;
      font-size: 0.94rem;
    }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .muted {{ color: var(--muted); }}
    .section-grid {{
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      margin-top: 26px;
    }}
    .section-card {{
      padding: 22px;
      overflow: hidden;
      position: relative;
    }}
    .section-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto auto 0;
      height: 4px;
      width: 100%;
      background: linear-gradient(90deg, var(--accent), transparent 85%);
      opacity: 0.95;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    .section-title {{
      margin: 0;
      font-size: 1.5rem;
      letter-spacing: -0.03em;
    }}
    .section-copy {{
      margin: 6px 0 0;
      color: #c1d0e2;
      line-height: 1.5;
    }}
    .mini-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .mini-card {{
      padding: 14px 0;
      border-top: 1px solid var(--border);
    }}
    .mini-label {{
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 7px;
    }}
    .mini-value {{
      font-size: 1.25rem;
      letter-spacing: -0.03em;
      margin-bottom: 3px;
    }}
    .mini-sub {{
      color: #c4d2e3;
      font-size: 0.91rem;
    }}
    .table-wrap {{
      overflow: auto;
      border-top: 1px solid var(--border);
      padding-top: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    th, td {{
      text-align: left;
      padding: 11px 10px;
      border-bottom: 1px solid rgba(173, 201, 235, 0.09);
      font-size: 0.92rem;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.74rem;
      position: sticky;
      top: 0;
      background: rgba(10, 19, 32, 0.98);
    }}
    .ticker {{
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.79rem;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.08);
      white-space: nowrap;
    }}
    .badge.win {{ color: var(--good); }}
    .badge.loss {{ color: var(--bad); }}
    .badge.open {{ color: var(--warn); }}
    .empty {{
      padding: 18px;
      border: 1px dashed rgba(173, 201, 235, 0.2);
      border-radius: 18px;
      color: #c6d4e4;
      background: rgba(255, 255, 255, 0.02);
      line-height: 1.55;
    }}
    .footer {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
    }}
    .log-card {{
      margin-top: 26px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 22px;
      backdrop-filter: blur(10px);
    }}
    .log-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .log-title {{
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
    }}
    .log-copy {{
      margin: 6px 0 0;
      color: #c1d0e2;
      line-height: 1.5;
    }}
    .log-meta {{
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .log-terminal {{
      margin: 0;
      padding: 16px;
      border-radius: 18px;
      background: #050a12;
      border: 1px solid rgba(173, 201, 235, 0.1);
      color: #d7fbe8;
      font: 12.5px/1.55 "SFMono-Regular", "Menlo", "Consolas", monospace;
      overflow: auto;
      max-height: 460px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 760px) {{
      .shell {{
        width: min(100vw - 20px, 1480px);
        padding-top: 18px;
      }}
      .hero-inner, .section-card {{
        padding: 18px;
      }}
      .mini-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-inner">
        <div>
          <div class="eyebrow">Motorcade Specialist Monitor</div>
          <h1>Vertical Sidecar Performance Dashboard</h1>
          <p class="lede">
            Executed fills, resolved outcomes, and sidecar calibration in one place for the three live specialist paths.
            This snapshot is generated from your local research logs and is designed to replace the outdated dashboard file.
          </p>
        </div>
        <div class="hero-meta">
          <div class="meta-chip"><strong>Generated:</strong> {generated_at}</div>
          <div class="meta-chip"><strong>Date filter:</strong> {since_label}</div>
          <div class="meta-chip"><strong>Research dir:</strong> {research_dir}</div>
        </div>
        <div class="status-row" id="global-status-row"></div>
        <div class="range-picker" id="range-picker"></div>
      </div>
    </section>
    <section class="summary-grid" id="summary-grid"></section>
    <section class="section-grid" id="section-grid"></section>
    <section class="log-card" id="bot-log-card"></section>
    <div class="footer">
      Snapshot file: <code>vertical_dashboard.html</code>. Reloading the page refreshes the view of the current snapshot;
      rerun <code>python3 scripts/render_vertical_dashboard.py</code> to bake in the latest research/output rows.
    </div>
  </div>
  <script>
    const PAYLOAD = {payload_json};
    const AUTO_REFRESH_SECONDS = {auto_refresh_seconds};

    const WINDOWS = [
      {{ key: "all", label: "All time", days: null }},
      {{ key: "30d", label: "Last 30 days", days: 30 }},
      {{ key: "7d", label: "Last 7 days", days: 7 }},
      {{ key: "1d", label: "Last 24h", days: 1 }},
    ];

    const currency = new Intl.NumberFormat("en-US", {{
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    }});
    const number1 = new Intl.NumberFormat("en-US", {{
      maximumFractionDigits: 1,
      minimumFractionDigits: 1,
    }});
    const number2 = new Intl.NumberFormat("en-US", {{
      maximumFractionDigits: 2,
      minimumFractionDigits: 2,
    }});

    let activeWindow = "all";

    function parseDate(value) {{
      return value ? new Date(value) : null;
    }}

    function cutoffForWindow(windowKey) {{
      const windowDef = WINDOWS.find((item) => item.key === windowKey);
      if (!windowDef || !windowDef.days) return null;
      const now = parseDate(PAYLOAD.generated_at) || new Date();
      return new Date(now.getTime() - windowDef.days * 24 * 60 * 60 * 1000);
    }}

    function filterByWindow(rows, windowKey) {{
      const cutoff = cutoffForWindow(windowKey);
      if (!cutoff) return rows.slice();
      return rows.filter((row) => {{
        const dt = parseDate(row.ts);
        return dt && dt >= cutoff;
      }});
    }}

    function pct(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      return `${{number1.format(value)}}%`;
    }}

    function signedCurrency(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      const base = currency.format(Math.abs(value));
      if (value > 0) return `+${{base}}`;
      if (value < 0) return `-${{base}}`;
      return base;
    }}

    function signedNumber(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      if (value > 0) return `+${{number2.format(value)}}`;
      if (value < 0) return `-${{number2.format(Math.abs(value))}}`;
      return number2.format(value);
    }}

    function fmtPrice(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      return number2.format(value);
    }}

    function fmtDate(value) {{
      const dt = parseDate(value);
      if (!dt) return "—";
      return dt.toLocaleString("en-US", {{
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      }});
    }}

    function cssForSigned(value) {{
      if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
      return value > 0 ? "good" : "bad";
    }}

    function summarizeTrades(rows) {{
      const resolved = rows.filter((row) => row.resolved);
      const wins = resolved.filter((row) => row.won === true);
      const losses = resolved.filter((row) => row.won === false);
      const open = rows.filter((row) => !row.resolved);
      const pnl = resolved.reduce((sum, row) => sum + (row.pnl || 0), 0);
      const deployed = resolved.reduce((sum, row) => sum + (row.notional || 0), 0);
      const avgEdge = rows.reduce((sum, row) => sum + (row.signal_edge_pct || 0), 0) / (rows.filter((row) => row.signal_edge_pct !== null).length || 1);
      const avgConfidence = rows.reduce((sum, row) => sum + (row.signal_confidence || 0), 0) / (rows.filter((row) => row.signal_confidence !== null).length || 1);
      const avgSpecialist = rows.reduce((sum, row) => sum + (row.effective_specialist_prob_yes || 0), 0) / (rows.filter((row) => row.effective_specialist_prob_yes !== null).length || 1);
      return {{
        tradeCount: rows.length,
        resolvedCount: resolved.length,
        openCount: open.length,
        winCount: wins.length,
        lossCount: losses.length,
        winRate: resolved.length ? (wins.length / resolved.length) * 100 : null,
        pnl,
        deployed,
        roi: deployed ? (pnl / deployed) * 100 : null,
        avgEdgePct: rows.filter((row) => row.signal_edge_pct !== null).length ? avgEdge * 100 : null,
        avgConfidencePct: rows.filter((row) => row.signal_confidence !== null).length ? avgConfidence * 100 : null,
        avgSpecialistPct: rows.filter((row) => row.effective_specialist_prob_yes !== null).length ? avgSpecialist * 100 : null,
      }};
    }}

    function summarizePredictions(rows) {{
      const resolved = rows.filter((row) => row.resolved && row.effective_probability !== null && row.effective_probability !== undefined);
      if (!resolved.length) {{
        return {{
          resolvedCount: 0,
          openCount: rows.filter((row) => !row.resolved).length,
          directionalAccuracy: null,
          brier: null,
          baselineBrier: null,
          brierLiftPct: null,
        }};
      }}
      const hits = resolved.filter((row) => (row.effective_probability >= 0.5) === row.outcome_yes);
      const brier = resolved.reduce((sum, row) => {{
        const target = row.outcome_yes ? 1 : 0;
        return sum + Math.pow(row.effective_probability - target, 2);
      }}, 0) / resolved.length;
      const baselineBrier = resolved.reduce((sum, row) => {{
        const target = row.outcome_yes ? 1 : 0;
        return sum + Math.pow(0.5 - target, 2);
      }}, 0) / resolved.length;
      return {{
        resolvedCount: resolved.length,
        openCount: rows.filter((row) => !row.resolved).length,
        directionalAccuracy: (hits.length / resolved.length) * 100,
        brier,
        baselineBrier,
        brierLiftPct: baselineBrier ? ((baselineBrier - brier) / baselineBrier) * 100 : null,
      }};
    }}

    function renderSummary(windowKey) {{
      const trades = filterByWindow(PAYLOAD.trades, windowKey);
      const predictions = filterByWindow(PAYLOAD.predictions, windowKey);
      const tradeSummary = summarizeTrades(trades);
      const predSummary = summarizePredictions(predictions);
      const sourceCounts = PAYLOAD.source_counts || {{}};

      const cards = [
        {{
          label: "Executed fills",
          value: tradeSummary.tradeCount,
          sub: `${{tradeSummary.resolvedCount}} resolved / ${{tradeSummary.openCount}} open`,
          tone: "",
        }},
        {{
          label: "Resolved win rate",
          value: pct(tradeSummary.winRate),
          sub: `${{tradeSummary.winCount}} wins / ${{tradeSummary.lossCount}} losses`,
          tone: tradeSummary.winRate === null ? "" : (tradeSummary.winRate >= 50 ? "good" : "bad"),
        }},
        {{
          label: "Net PnL",
          value: signedCurrency(tradeSummary.pnl),
          sub: tradeSummary.roi === null ? "No resolved notional yet" : `ROI ${{pct(tradeSummary.roi)}} on ${{currency.format(tradeSummary.deployed || 0)}} deployed`,
          tone: cssForSigned(tradeSummary.pnl),
        }},
        {{
          label: "Sidecar directional accuracy",
          value: pct(predSummary.directionalAccuracy),
          sub: predSummary.resolvedCount ? `${{predSummary.resolvedCount}} resolved prediction tickers` : "No resolved prediction logs yet",
          tone: predSummary.directionalAccuracy === null ? "" : (predSummary.directionalAccuracy >= 50 ? "good" : "bad"),
        }},
        {{
          label: "Brier lift vs 50/50",
          value: pct(predSummary.brierLiftPct),
          sub: predSummary.brier === null ? "Waiting on outcomes" : `Brier ${{number2.format(predSummary.brier)}} vs ${{number2.format(predSummary.baselineBrier)}} baseline`,
          tone: predSummary.brierLiftPct === null ? "" : (predSummary.brierLiftPct >= 0 ? "good" : "bad"),
        }},
        {{
          label: "Source files scanned",
          value: sourceCounts.order_lifecycle_files || 0,
          sub: `${{sourceCounts.market_state_files || 0}} market-state files / ${{sourceCounts.outcome_rows || 0}} outcomes`,
          tone: "",
        }},
      ];

      document.getElementById("summary-grid").innerHTML = cards.map((card) => `
        <article class="stat-card">
          <div class="stat-label">${{card.label}}</div>
          <div class="stat-value ${{card.tone}}">${{card.value}}</div>
          <div class="stat-sub">${{card.sub}}</div>
        </article>
      `).join("");
    }}

    function renderStatusPill(status) {{
      const title = status.latest_ts ? `${{status.text}}` : status.text;
      return `
        <div class="status-pill ${{status.state}}" title="${{title}}">
          <span class="status-dot"></span>
          <strong>${{status.label}}:</strong> ${{status.state.toUpperCase()}}
        </div>
      `;
    }}

    function renderGlobalStatus() {{
      const globalStatusRow = document.getElementById("global-status-row");
      const statuses = [PAYLOAD.status.bot, ...PAYLOAD.sidecars.map((sidecar) => PAYLOAD.status.sidecars[sidecar.key])];
      globalStatusRow.innerHTML = statuses.map(renderStatusPill).join("");
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function tradeRowMarkup(row) {{
      const badgeClass = row.resolved ? (row.won ? "win" : "loss") : "open";
      const badgeLabel = row.resolved ? (row.won ? "WIN" : "LOSS") : "OPEN";
      const pnlClass = row.pnl === null || row.pnl === undefined ? "" : cssForSigned(row.pnl);
      return `
        <tr>
          <td><span class="badge ${{badgeClass}}">${{badgeLabel}}</span></td>
          <td>${{fmtDate(row.ts)}}</td>
          <td>
            <div class="ticker">${{row.ticker}}</div>
            <div class="muted">${{row.title || row.series_ticker || "—"}}</div>
          </td>
          <td>${{(row.side || "—").toUpperCase()}} ${{(row.outcome_id || "").toUpperCase()}}</td>
          <td>${{fmtPrice(row.avg_fill_price)}}</td>
          <td>${{signedNumber(row.filled_qty)}}</td>
          <td>${{currency.format(row.notional || 0)}}</td>
          <td>${{row.signal_edge_pct === null || row.signal_edge_pct === undefined ? "—" : pct(row.signal_edge_pct * 100)}}</td>
          <td>${{row.signal_confidence === null || row.signal_confidence === undefined ? "—" : pct(row.signal_confidence * 100)}}</td>
          <td>${{row.effective_specialist_prob_yes === null || row.effective_specialist_prob_yes === undefined ? "—" : pct(row.effective_specialist_prob_yes * 100)}}</td>
          <td class="${{pnlClass}}">${{signedCurrency(row.pnl)}}</td>
        </tr>
      `;
    }}

    function renderSections(windowKey) {{
      const sectionGrid = document.getElementById("section-grid");
      const markup = PAYLOAD.sidecars.map((sidecar) => {{
        const trades = filterByWindow(PAYLOAD.trades.filter((row) => row.vertical === sidecar.key), windowKey);
        const predictions = filterByWindow(PAYLOAD.predictions.filter((row) => row.vertical === sidecar.key), windowKey);
        const tradeSummary = summarizeTrades(trades);
        const predSummary = summarizePredictions(predictions);
        const recentTrades = trades.slice(0, {TRADE_TABLE_LIMIT});
        const predictionSource = (PAYLOAD.source_counts.prediction_sources || {{}})[sidecar.key] || {{}};
        const emptyState = `
          <div class="empty">
            No executed fills found for this sidecar in the selected window yet. Once the order lifecycle and outcomes logs populate,
            this section will fill in automatically the next time the dashboard snapshot is rendered.
          </div>
        `;
        return `
          <article class="section-card" style="--accent: ${{sidecar.accent}}">
            <div class="section-header">
              <div>
                <h2 class="section-title">${{sidecar.label}}</h2>
                <p class="section-copy">${{sidecar.description}}</p>
              </div>
              <div>
                <div class="badge">${{tradeSummary.tradeCount}} fills</div>
                <div style="height:8px"></div>
                ${{renderStatusPill(PAYLOAD.status.sidecars[sidecar.key])}}
              </div>
            </div>
            <div class="mini-grid">
              <div class="mini-card">
                <div class="mini-label">Resolved / Open</div>
                <div class="mini-value">${{tradeSummary.resolvedCount}} / ${{tradeSummary.openCount}}</div>
                <div class="mini-sub">${{tradeSummary.winCount}} wins, ${{tradeSummary.lossCount}} losses</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">Win Rate</div>
                <div class="mini-value ${{tradeSummary.winRate !== null && tradeSummary.winRate >= 50 ? "good" : "bad"}}">${{pct(tradeSummary.winRate)}}</div>
                <div class="mini-sub">Resolved executed trades only</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">Net PnL / ROI</div>
                <div class="mini-value ${{cssForSigned(tradeSummary.pnl)}}">${{signedCurrency(tradeSummary.pnl)}}</div>
                <div class="mini-sub">${{tradeSummary.roi === null ? "No resolved notional yet" : `ROI ${{pct(tradeSummary.roi)}} on ${{currency.format(tradeSummary.deployed || 0)}}`}}</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">Avg Edge / Confidence</div>
                <div class="mini-value">${{pct(tradeSummary.avgEdgePct)}}</div>
                <div class="mini-sub">Confidence ${{pct(tradeSummary.avgConfidencePct)}}</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">Calibration</div>
                <div class="mini-value ${{predSummary.brierLiftPct !== null && predSummary.brierLiftPct >= 0 ? "good" : "bad"}}">${{pct(predSummary.brierLiftPct)}}</div>
                <div class="mini-sub">${{predSummary.brier === null ? "No resolved prediction logs" : `Brier ${{number2.format(predSummary.brier)}} / accuracy ${{pct(predSummary.directionalAccuracy)}}`}}</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">Prediction Log Coverage</div>
                <div class="mini-value">${{predictionSource.prediction_count || 0}}</div>
                <div class="mini-sub">${{predictionSource.file_count || 0}} prediction files scanned</div>
              </div>
            </div>
            ${{
              recentTrades.length
                ? `<div class="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Result</th>
                          <th>Time</th>
                          <th>Ticker</th>
                          <th>Order</th>
                          <th>Fill</th>
                          <th>Qty</th>
                          <th>Notional</th>
                          <th>Edge</th>
                          <th>Conf.</th>
                          <th>Sidecar Prob</th>
                          <th>PnL</th>
                        </tr>
                      </thead>
                      <tbody>
                        ${{recentTrades.map(tradeRowMarkup).join("")}}
                      </tbody>
                    </table>
                  </div>`
                : emptyState
            }}
          </article>
        `;
      }}).join("");

      sectionGrid.innerHTML = markup;
    }}

    function renderBotLog() {{
      const card = document.getElementById("bot-log-card");
      const botLog = PAYLOAD.bot_log || {{}};
      if (!botLog.path) {{
        card.innerHTML = `
          <div class="log-header">
            <div>
              <h2 class="log-title">Main Bot Log</h2>
              <p class="log-copy">No bot log path configured for this dashboard snapshot yet.</p>
            </div>
          </div>
          <div class="empty">
            Set <code>VERTICAL_DASHBOARD_BOT_LOG_PATH</code> or pass <code>--bot-log-path</code> when rendering/serving
            the dashboard to embed live bot output here.
          </div>
        `;
        return;
      }}
      if (botLog.error) {{
        card.innerHTML = `
          <div class="log-header">
            <div>
              <h2 class="log-title">Main Bot Log</h2>
              <p class="log-copy">Tailing <code>${{escapeHtml(botLog.path)}}</code></p>
            </div>
          </div>
          <div class="empty">Could not read the log file: ${{escapeHtml(botLog.error)}}</div>
        `;
        return;
      }}
      if (!botLog.exists) {{
        card.innerHTML = `
          <div class="log-header">
            <div>
              <h2 class="log-title">Main Bot Log</h2>
              <p class="log-copy">Configured path: <code>${{escapeHtml(botLog.path)}}</code></p>
            </div>
          </div>
          <div class="empty">The configured log file does not exist yet.</div>
        `;
        return;
      }}
      const body = botLog.lines.length
        ? escapeHtml(botLog.lines.join("\\n"))
        : "Log file is present but currently empty.";
      card.innerHTML = `
        <div class="log-header">
          <div>
            <h2 class="log-title">Main Bot Log</h2>
            <p class="log-copy">Live tail from <code>${{escapeHtml(botLog.path)}}</code></p>
          </div>
          <div class="log-meta">
            ${{botLog.line_count}} lines shown<br>
            Updated ${{fmtDate(botLog.updated_at)}}
          </div>
        </div>
        <pre class="log-terminal">${{body}}</pre>
      `;
    }}

    function renderWindowPicker() {{
      const picker = document.getElementById("range-picker");
      picker.innerHTML = WINDOWS.map((item) => `
        <button class="${{item.key === activeWindow ? "active" : ""}}" data-window="${{item.key}}">
          ${{item.label}}
        </button>
      `).join("");
      picker.querySelectorAll("button").forEach((button) => {{
        button.addEventListener("click", () => {{
          activeWindow = button.dataset.window;
          render();
        }});
      }});
    }}

    function render() {{
      renderGlobalStatus();
      renderWindowPicker();
      renderSummary(activeWindow);
      renderSections(activeWindow);
      renderBotLog();
    }}

    render();

    if (AUTO_REFRESH_SECONDS > 0) {{
      window.setTimeout(() => window.location.reload(), AUTO_REFRESH_SECONDS * 1000);
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    research_dir = Path(args.research_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    bot_log_path = Path(args.bot_log_path).expanduser().resolve() if args.bot_log_path else None

    payload = build_payload(research_dir, args.since, bot_log_path, args.bot_log_lines)
    output_path.write_text(render_html(payload, args.auto_refresh_seconds))
    print(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    main()
