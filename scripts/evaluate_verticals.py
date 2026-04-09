#!/usr/bin/env python3
"""
Evaluate weather and crypto sidecar performance.

Loads filled orders from the order lifecycle log and the crypto sidecar
prediction log, cross-references against resolved outcomes, and prints
a simple W/L breakdown by vertical.

Usage:
    python3 scripts/evaluate_verticals.py
    python3 scripts/evaluate_verticals.py --since 2026-04-07
    python3 scripts/evaluate_verticals.py --since 2026-04-04 --detail
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

CRYPTO_PREFIXES  = ("KXBTCD", "KXETHD", "KXSOLD", "KXXRPD")
WEATHER_PREFIXES = ("KXHIGHT",)
FEE_RATE         = 0.07


# ── Data loading ───────────────────────────────────────────────────────────────

def load_outcomes(path: Path) -> dict[str, bool]:
    outcomes = {}
    if not path.exists():
        return outcomes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                outcomes[r["ticker"]] = bool(r.get("result", False))
            except (json.JSONDecodeError, KeyError):
                pass
    return outcomes


def load_filled_orders(lifecycle_base: Path, since: str) -> list[dict]:
    """Load filled orders from dated subdirectories on or after `since`."""
    rows = []
    if not lifecycle_base.exists():
        return rows
    for date_dir in sorted(lifecycle_base.iterdir()):
        if not date_dir.is_dir() or date_dir.name < since:
            continue
        for f in date_dir.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if (r.get("filled_qty") or 0) > 0:
                            rows.append(r)
                    except json.JSONDecodeError:
                        pass
    return rows


def load_sidecar_predictions(pred_dir: Path, since: str) -> dict[str, dict]:
    """Return ticker -> last sidecar prediction record on or after `since`."""
    preds: dict[str, dict] = {}
    if not pred_dir.exists():
        return preds
    for f in sorted(pred_dir.glob("predictions_*.jsonl")):
        day = f.stem.replace("predictions_", "")
        if day < since:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    preds[r["ticker"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return preds


# ── Helpers ────────────────────────────────────────────────────────────────────

def vertical(ticker: str) -> str:
    if any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
        return "crypto"
    if any(ticker.startswith(p) for p in WEATHER_PREFIXES):
        return "weather"
    return "other"


def order_won(order: dict, outcome_yes: bool) -> bool:
    side = order.get("side", "").lower()          # "buy" or "sell"
    outcome_id = order.get("outcome_id", "yes")    # "yes" or "no"
    if side == "buy":
        return outcome_yes == (outcome_id == "yes")
    return outcome_yes != (outcome_id == "yes")


def calc_pnl(order: dict, won: bool) -> float:
    fill = order.get("avg_fill_price") or 0.0
    qty  = order.get("filled_qty") or 0.0
    cost = fill * qty
    fee  = cost * FEE_RATE
    return (qty - cost - fee) if won else -(cost + fee)


# ── Section printers ───────────────────────────────────────────────────────────

def print_fills_section(title: str, orders: list[dict], outcomes: dict[str, bool], detail: bool) -> None:
    resolved = [(o, outcomes[o["ticker"]]) for o in orders if o.get("ticker") in outcomes]
    open_pos  = [o for o in orders if o.get("ticker") not in outcomes]

    wins   = [o for o, oy in resolved if order_won(o, oy)]
    losses = [o for o, oy in resolved if not order_won(o, oy)]
    total_pnl      = sum(calc_pnl(o, True)  for o, oy in resolved if order_won(o, oy))
    total_pnl     += sum(calc_pnl(o, False) for o, oy in resolved if not order_won(o, oy))
    deployed       = sum((o.get("avg_fill_price") or 0) * (o.get("filled_qty") or 0) for o, _ in resolved)

    n = len(resolved)
    win_pct = wins.__len__() / n * 100 if n else 0.0
    roi     = total_pnl / deployed * 100 if deployed else 0.0

    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")
    print(f"  Filled orders  : {len(orders)}  ({n} resolved, {len(open_pos)} open)")
    if n:
        print(f"  W / L          : {len(wins)} / {len(losses)}  →  {win_pct:.1f}% win rate")
        print(f"  Net PnL        : ${total_pnl:+.2f}  (ROI {roi:+.1f}% on ${deployed:.0f} deployed)")

    # Per-series breakdown
    by_series: dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "open": 0, "pnl": 0.0, "dep": 0.0})
    for o, oy in resolved:
        s = o["ticker"].split("-")[0]
        won = order_won(o, oy)
        by_series[s]["w" if won else "l"] += 1
        by_series[s]["pnl"] += calc_pnl(o, won)
        by_series[s]["dep"] += (o.get("avg_fill_price") or 0) * (o.get("filled_qty") or 0)
    for o in open_pos:
        by_series[o["ticker"].split("-")[0]]["open"] += 1

    if len(by_series) > 1:
        print(f"\n  {'Series':<28} {'W':>4} {'L':>4} {'Open':>5} {'Win%':>7} {'PnL':>9}")
        print(f"  {'-'*60}")
        for s, d in sorted(by_series.items(), key=lambda x: -(x[1]["w"] + x[1]["l"] + x[1]["open"])):
            nr = d["w"] + d["l"]
            wp = f"{d['w']/nr*100:.1f}%" if nr else "   —"
            print(f"  {s:<28} {d['w']:>4} {d['l']:>4} {d['open']:>5} {wp:>7} {d['pnl']:>+9.2f}")

    if detail and resolved:
        print(f"\n  {'Result':<5} {'Ticker':<48} {'Side':>4} {'Fill':>6}")
        print(f"  {'-'*65}")
        for o, oy in sorted(resolved, key=lambda x: x[0].get("ts", "")):
            won = order_won(o, oy)
            fill = o.get("avg_fill_price")
            fill_str = f"{fill:.3f}" if fill is not None else "  ?"
            print(f"  {'WIN ' if won else 'LOSS':<5} {o['ticker']:<48} {o.get('outcome_id','?'):>4} {fill_str:>6}")


def print_sidecar_section(preds: dict[str, dict], outcomes: dict[str, bool], detail: bool) -> None:
    resolved = [(t, p, outcomes[t]) for t, p in preds.items() if t in outcomes]
    open_pos  = [(t, p) for t, p in preds.items() if t not in outcomes]

    n = len(resolved)
    if not n and not open_pos:
        print(f"\n{'='*56}")
        print(f"  CRYPTO SIDECAR SHADOW  (no predictions found)")
        print(f"{'='*56}")
        return

    # A prediction "wins" if sidecar side (prob >= 0.5 → predicts YES) matches outcome
    def sidecar_won(prob: float, outcome_yes: bool) -> bool:
        return (prob >= 0.5) == outcome_yes

    wins   = [(t, p, oy) for t, p, oy in resolved if sidecar_won(p["probability"], oy)]
    losses = [(t, p, oy) for t, p, oy in resolved if not sidecar_won(p["probability"], oy)]
    win_pct = len(wins) / n * 100 if n else 0.0

    brier_scores = [(p["probability"] - (1.0 if oy else 0.0)) ** 2 for _, p, oy in resolved]
    avg_brier = sum(brier_scores) / n if n else 0.0

    print(f"\n{'='*56}")
    print(f"  CRYPTO SIDECAR SHADOW")
    print(f"{'='*56}")
    print(f"  Predicted tickers : {len(preds)}  ({n} resolved, {len(open_pos)} open)")
    if n:
        print(f"  W / L             : {len(wins)} / {len(losses)}  →  {win_pct:.1f}% win rate")
        print(f"  Brier score       : {avg_brier:.4f}")

    if detail and resolved:
        print(f"\n  {'Result':<5} {'Ticker':<48} {'Prob':>6} {'Outcome':>8}")
        print(f"  {'-'*70}")
        for t, p, oy in sorted(resolved, key=lambda x: x[0]):
            won = sidecar_won(p["probability"], oy)
            print(f"  {'WIN ' if won else 'LOSS':<5} {t:<48} {p['probability']:>6.3f} {str(oy):>8}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since",  default="0000-00-00", metavar="DATE",
                        help="Filter to orders/predictions on or after DATE (YYYY-MM-DD). Default: all time.")
    parser.add_argument("--detail", action="store_true", help="Print per-ticker rows")
    args = parser.parse_args()

    outcomes = load_outcomes(REPO_ROOT / "var/research/outcomes/outcomes.jsonl")
    all_orders = load_filled_orders(REPO_ROOT / "var/research/order_lifecycle", args.since)

    crypto_orders  = [o for o in all_orders if vertical(o.get("ticker", "")) == "crypto"]
    weather_orders = [o for o in all_orders if vertical(o.get("ticker", "")) == "weather"]

    since_label = args.since if args.since != "0000-00-00" else "all time"
    print(f"\n  Date filter : {since_label}")
    print(f"  Outcomes DB : {len(outcomes):,} resolved tickers")

    print_fills_section("WEATHER  (filled orders)", weather_orders, outcomes, args.detail)
    print_fills_section("CRYPTO   (filled orders)", crypto_orders,  outcomes, args.detail)

    sidecar_pred_dir = REPO_ROOT / "sidecars/crypto/var/logs/crypto_predictions"
    sidecar_preds = load_sidecar_predictions(sidecar_pred_dir, args.since)
    print_sidecar_section(sidecar_preds, outcomes, args.detail)

    print()


if __name__ == "__main__":
    main()
