#!/usr/bin/env python3
"""
Evaluate sidecar performance by vertical: weather, crypto, and FED.

Loads filled orders from the order lifecycle log, cross-references against
resolved outcomes, and prints W/L + PnL by vertical.  For crypto and FED,
also loads sidecar prediction logs and reports calibration (Brier score,
directional accuracy) independently of whether an order was placed.

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
FED_PREFIXES     = ("KXFED", "KXFOMC")
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

def effective_prob(ticker: str, prob: float) -> float:
    """
    Return P(outcome YES) from a sidecar prediction record.

    All sidecars store P(outcome YES) directly — the weather sidecar already
    inverts for BELOW markets before logging (sidecar.py: `if below: prob = 1 - prob`).
    No additional inversion is needed here.
    """
    return prob


def vertical(ticker: str) -> str:
    if any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
        return "crypto"
    if any(ticker.startswith(p) for p in WEATHER_PREFIXES):
        return "weather"
    if any(ticker.startswith(p) for p in FED_PREFIXES):
        return "fed"
    return "other"


def order_won(order: dict, outcome_yes: bool) -> bool:
    side = order.get("side", "").lower()
    outcome_id = order.get("outcome_id", "yes")
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
    total_pnl  = sum(calc_pnl(o, True)  for o, oy in resolved if order_won(o, oy))
    total_pnl += sum(calc_pnl(o, False) for o, oy in resolved if not order_won(o, oy))
    deployed   = sum((o.get("avg_fill_price") or 0) * (o.get("filled_qty") or 0) for o, _ in resolved)

    n = len(resolved)
    win_pct = len(wins) / n * 100 if n else 0.0
    roi     = total_pnl / deployed * 100 if deployed else 0.0

    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")
    print(f"  Filled orders  : {len(orders)}  ({n} resolved, {len(open_pos)} open)")
    if n:
        print(f"  W / L          : {len(wins)} / {len(losses)}  →  {win_pct:.1f}% win rate")
        print(f"  Net PnL        : ${total_pnl:+.2f}  (ROI {roi:+.1f}% on ${deployed:.0f} deployed)")

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


def print_prediction_section(title: str, preds: dict[str, dict], outcomes: dict[str, bool], detail: bool) -> None:
    """Print sidecar calibration stats independent of whether an order was placed."""
    resolved = [(t, p, outcomes[t]) for t, p in preds.items() if t in outcomes]
    open_pos  = [(t, p) for t, p in preds.items() if t not in outcomes]

    n = len(resolved)

    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")

    if not n and not open_pos:
        print(f"  No predictions found.")
        return

    print(f"  Predicted tickers : {len(preds)}  ({n} resolved, {len(open_pos)} open)")

    if n:
        ep = [(t, effective_prob(t, p["probability"]), oy) for t, p, oy in resolved]

        wins   = [(t, prob, oy) for t, prob, oy in ep if (prob >= 0.5) == oy]
        losses = [(t, prob, oy) for t, prob, oy in ep if (prob >= 0.5) != oy]
        win_pct = len(wins) / n * 100

        brier_scores   = [(prob - (1.0 if oy else 0.0)) ** 2 for _, prob, oy in ep]
        avg_brier      = sum(brier_scores) / n
        baseline_brier = sum((0.5 - (1.0 if oy else 0.0)) ** 2 for _, _, oy in ep) / n
        brier_lift_pct = (baseline_brier - avg_brier) / baseline_brier * 100 if baseline_brier else 0.0

        print(f"  W / L             : {len(wins)} / {len(losses)}  →  {win_pct:.1f}% directional accuracy")
        print(f"  Brier score       : {avg_brier:.4f}  (baseline {baseline_brier:.4f}, lift {brier_lift_pct:+.1f}%)")

        if detail:
            print(f"\n  {'Result':<5} {'Ticker':<48} {'EffProb':>8} {'Outcome':>8}")
            print(f"  {'-'*72}")
            for t, prob, oy in sorted(ep, key=lambda x: x[0]):
                won = (prob >= 0.5) == oy
                print(f"  {'WIN ' if won else 'LOSS':<5} {t:<48} {prob:>8.3f} {str(oy):>8}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since",  default="0000-00-00", metavar="DATE",
                        help="Filter to orders/predictions on or after DATE (YYYY-MM-DD). Default: all time.")
    parser.add_argument("--detail", action="store_true", help="Print per-ticker rows")
    args = parser.parse_args()

    outcomes   = load_outcomes(REPO_ROOT / "var/research/outcomes/outcomes.jsonl")
    all_orders = load_filled_orders(REPO_ROOT / "var/research/order_lifecycle", args.since)

    weather_orders = [o for o in all_orders if vertical(o.get("ticker", "")) == "weather"]
    crypto_orders  = [o for o in all_orders if vertical(o.get("ticker", "")) == "crypto"]
    fed_orders     = [o for o in all_orders if vertical(o.get("ticker", "")) == "fed"]
    other_orders   = [o for o in all_orders if vertical(o.get("ticker", "")) == "other"]

    since_label = args.since if args.since != "0000-00-00" else "all time"
    print(f"\n  Date filter : {since_label}")
    print(f"  Outcomes DB : {len(outcomes):,} resolved tickers")
    print(f"  Total fills : {len(all_orders)}")

    # Fills by vertical
    print_fills_section("WEATHER  (filled orders)", weather_orders, outcomes, args.detail)
    print_fills_section("CRYPTO   (filled orders)", crypto_orders,  outcomes, args.detail)
    print_fills_section("FED      (filled orders)", fed_orders,     outcomes, args.detail)
    if other_orders:
        print_fills_section("OTHER    (filled orders)", other_orders, outcomes, args.detail)

    # Sidecar prediction calibration (independent of fills)
    weather_preds = load_sidecar_predictions(
        REPO_ROOT / "sidecars/weather/var/logs/gefs_predictions", args.since
    )
    crypto_preds = load_sidecar_predictions(
        REPO_ROOT / "sidecars/crypto/var/logs/crypto_predictions", args.since
    )
    fed_preds = load_sidecar_predictions(
        REPO_ROOT / "sidecars/hawkwatchers/var/logs/fed_predictions", args.since
    )
    print_prediction_section("WEATHER SIDECAR (calibration)", weather_preds, outcomes, args.detail)
    print_prediction_section("CRYPTO SIDECAR  (calibration)", crypto_preds,  outcomes, args.detail)
    print_prediction_section("FED SIDECAR     (calibration)", fed_preds,     outcomes, args.detail)

    print()


if __name__ == "__main__":
    main()
