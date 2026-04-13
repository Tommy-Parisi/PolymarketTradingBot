#!/usr/bin/env python3
"""
Deep analysis of GEFS weather predictions vs outcomes.

Merges predictions from both log directories (top-level and sidecar-local),
cross-references against resolved outcomes and filled order lifecycle, and
reports calibration, per-city bias, ensemble spread, and trade performance.

Usage:
    python3 scripts/analyze_weather_gefs.py
    python3 scripts/analyze_weather_gefs.py --since 2026-04-07
    python3 scripts/analyze_weather_gefs.py --detail
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PRED_DIRS = [
    REPO_ROOT / "var/logs/gefs_predictions",
    REPO_ROOT / "sidecars/weather/var/logs/gefs_predictions",
]

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_predictions(since: str) -> dict[str, dict]:
    """Merge predictions from all directories. Later files overwrite earlier ones per ticker."""
    preds: dict[str, dict] = {}
    for d in PRED_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("predictions_*.jsonl")):
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
                        ticker = r.get("ticker")
                        if ticker:
                            preds[ticker] = r
                    except json.JSONDecodeError:
                        pass
    return preds


def load_outcomes() -> dict[str, bool]:
    """Load only non-null outcomes. Many records have outcome_yes=null (backfill race)."""
    path = REPO_ROOT / "var/research/outcomes/outcomes.jsonl"
    outcomes: dict[str, bool] = {}
    if not path.exists():
        return outcomes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                oy = r.get("outcome_yes")
                if "ticker" in r and oy is not None:
                    outcomes[r["ticker"]] = bool(oy)
            except (json.JSONDecodeError, KeyError):
                pass
    return outcomes


def load_weather_fills(since: str) -> list[dict]:
    """Load filled weather orders from order lifecycle."""
    lifecycle_base = REPO_ROOT / "var/research/order_lifecycle"
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
                        ticker = r.get("ticker", "")
                        if (
                            ticker.startswith("KXHIGHT") or ticker.startswith("KXHIGH")
                        ) and (r.get("filled_qty") or 0) > 0:
                            rows.append(r)
                    except json.JSONDecodeError:
                        pass
    return rows


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_ticker(ticker: str):
    """Return (city_code, target_date_str, direction, threshold_f) or None."""
    # New format: KXHIGHT{CITY}-{DATE}-{T|B}{THRESHOLD}
    m = re.match(r"KXHIGHT([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-([TB])([\d.]+)", ticker)
    if m:
        return "T" + m.group(1), m.group(2), m.group(3), float(m.group(4))
    # Legacy Philly: KXHIGH{PHI|PHIL|PHILLY|PHL}-{DATE}-{T|B}{THRESHOLD}
    m = re.match(r"KXHIGH([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-([TB])([\d.]+)", ticker)
    if m:
        return m.group(1), m.group(2), m.group(3), float(m.group(4))
    return None


def effective_prob(ticker: str, raw_prob: float) -> float:
    """Return P(outcome YES). Sidecar already inverts for BELOW markets before storing."""
    return raw_prob


def brier(prob: float, outcome: bool) -> float:
    return (prob - (1.0 if outcome else 0.0)) ** 2


def city_from_code(code: str) -> str:
    mapping = {
        "TBOS": "Boston", "TDAL": "Dallas", "THOU": "Houston", "TSEA": "Seattle",
        "TPHX": "Phoenix", "TSATX": "San Antonio", "TLV": "Las Vegas",
        "TATL": "Atlanta", "TMIN": "Minneapolis", "TNOLA": "New Orleans",
        "TDC": "Washington DC", "TSFO": "San Francisco", "TOKC": "Oklahoma City",
        "PHI": "Philadelphia", "PHIL": "Philadelphia", "PHILLY": "Philadelphia",
        "PHL": "Philadelphia",
    }
    return mapping.get(code, code)


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(preds: dict[str, dict], outcomes: dict[str, bool], fills: list[dict], detail: bool):
    resolved = {t: (p, outcomes[t]) for t, p in preds.items() if t in outcomes}
    open_preds = {t: p for t, p in preds.items() if t not in outcomes}

    print(f"\n{'='*64}")
    print(f"  GEFS WEATHER PREDICTION ANALYSIS")
    print(f"{'='*64}")
    print(f"  Total predictions : {len(preds)} ({len(resolved)} resolved, {len(open_preds)} open)")
    print(f"  Outcomes DB       : {sum(1 for t in outcomes if t.startswith('KXHIGH'))} weather tickers resolved")

    if not resolved:
        print("\n  No resolved predictions found.")
        return

    # ── Overall calibration ────────────────────────────────────────────────────
    ep_list = [(t, effective_prob(t, p["probability"]), oy) for t, (p, oy) in resolved.items()]
    n = len(ep_list)
    correct = sum(1 for _, ep, oy in ep_list if (ep >= 0.5) == oy)
    avg_brier = sum(brier(ep, oy) for _, ep, oy in ep_list) / n
    baseline_brier = sum((0.5 - (1.0 if oy else 0.0)) ** 2 for _, _, oy in ep_list) / n
    lift = (baseline_brier - avg_brier) / baseline_brier * 100 if baseline_brier else 0.0

    print(f"\n{'─'*64}")
    print(f"  OVERALL CALIBRATION  (n={n})")
    print(f"{'─'*64}")
    print(f"  Directional accuracy : {correct}/{n}  ({correct/n*100:.1f}%)")
    print(f"  Brier score          : {avg_brier:.4f}  (baseline {baseline_brier:.4f}, lift {lift:+.1f}%)")

    # Confidence bins
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    bin_labels = ["Low (0.0–0.3)", "Mid-low (0.3–0.5)", "Mid-high (0.5–0.7)", "High (0.7–1.0)"]
    print(f"\n  Confidence bins:")
    print(f"  {'Bin':<22} {'N':>4} {'Corr':>5} {'Acc%':>6} {'AvgP':>6} {'Brier':>7}")
    print(f"  {'─'*52}")
    for (lo, hi), label in zip(bins, bin_labels):
        subset = [(ep, oy) for _, ep, oy in ep_list if lo <= ep < hi]
        if not subset:
            continue
        nb = len(subset)
        nc = sum(1 for ep, oy in subset if (ep >= 0.5) == oy)
        avg_p = sum(ep for ep, _ in subset) / nb
        avg_b = sum(brier(ep, oy) for ep, oy in subset) / nb
        print(f"  {label:<22} {nb:>4} {nc:>5} {nc/nb*100:>5.1f}% {avg_p:>6.3f} {avg_b:>7.4f}")

    # ── Per-city breakdown ─────────────────────────────────────────────────────
    by_city: dict[str, list] = defaultdict(list)
    for t, ep, oy in ep_list:
        parsed = parse_ticker(t)
        if parsed:
            city_code = parsed[0]
            by_city[city_code].append((t, ep, oy))

    print(f"\n{'─'*64}")
    print(f"  PER-CITY CALIBRATION")
    print(f"{'─'*64}")
    print(f"  {'City':<18} {'N':>4} {'Acc%':>6} {'Brier':>7} {'Lift%':>7} {'AvgP':>6} {'OutcomeYes%':>12}")
    print(f"  {'─'*58}")
    city_rows = []
    for code, items in by_city.items():
        nc = len(items)
        correct_c = sum(1 for _, ep, oy in items if (ep >= 0.5) == oy)
        avg_b = sum(brier(ep, oy) for _, ep, oy in items) / nc
        base_b = sum((0.5 - (1.0 if oy else 0.0)) ** 2 for _, _, oy in items) / nc
        lift_c = (base_b - avg_b) / base_b * 100 if base_b else 0.0
        avg_p = sum(ep for _, ep, _ in items) / nc
        yes_rate = sum(1 for _, _, oy in items if oy) / nc * 100
        city_rows.append((code, nc, correct_c, avg_b, lift_c, avg_p, yes_rate))
    for code, nc, corr, avg_b, lift_c, avg_p, yes_rate in sorted(city_rows, key=lambda x: -x[1]):
        name = city_from_code(code)
        print(f"  {name:<18} {nc:>4} {corr/nc*100:>5.1f}% {avg_b:>7.4f} {lift_c:>+6.1f}% {avg_p:>6.3f} {yes_rate:>11.1f}%")

    # ── Directional bias per city ──────────────────────────────────────────────
    # For ABOVE (T) markets: compare predicted_high (mean member highs) vs outcome
    # Use raw GEFS mean ensemble high vs threshold to measure bias
    print(f"\n{'─'*64}")
    print(f"  ENSEMBLE MEAN vs THRESHOLD  (ABOVE markets only, resolved)")
    print(f"{'─'*64}")
    print(f"  Measures how far GEFS ensemble mean sits relative to threshold,")
    print(f"  and whether that predicts the outcome correctly.\n")
    print(f"  {'City':<18} {'N':>4} {'MeanSpread':>11} {'Corr%':>7}  (spread = ensemble_mean - threshold)")
    print(f"  {'─'*50}")
    city_spread: dict[str, list] = defaultdict(list)
    for t, (p_rec, oy) in resolved.items():
        parsed = parse_ticker(t)
        if not parsed or parsed[2] != "T":
            continue
        city_code, _, _, threshold = parsed
        highs = p_rec.get("member_highs_f", [])
        if not highs:
            continue
        mean_high = sum(highs) / len(highs)
        spread = mean_high - threshold
        correct_dir = (spread > 0) == oy
        city_spread[city_code].append((spread, correct_dir, oy, t, p_rec["probability"]))

    for code, items in sorted(city_spread.items(), key=lambda x: -len(x[1])):
        n_c = len(items)
        mean_spread = sum(s for s, _, _, _, _ in items) / n_c
        corr = sum(1 for _, c, _, _, _ in items if c)
        print(f"  {city_from_code(code):<18} {n_c:>4} {mean_spread:>+10.2f}°F {corr/n_c*100:>6.1f}%")

    # ── Bias residual: actual GEFS implied temp vs outcome threshold ───────────
    # "GEFS bias" = how much we need to shift the ensemble to match outcomes
    print(f"\n{'─'*64}")
    print(f"  BIAS RESIDUAL ANALYSIS  (ABOVE markets, resolved)")
    print(f"{'─'*64}")
    print(f"  For wins:  ensemble_mean was above threshold (correct direction)")
    print(f"  For losses: ensemble_mean was wrong side of threshold")
    print(f"  Margin = |ensemble_mean - threshold| on wrong-side losses\n")

    city_margins: dict[str, dict] = defaultdict(lambda: {"correct": [], "wrong": []})
    for code, items in city_spread.items():
        for spread, correct, oy, t, prob in items:
            key = "correct" if correct else "wrong"
            city_margins[code][key].append(abs(spread))

    print(f"  {'City':<18} {'Correct':>8} {'Wrong':>8} {'AvgWrongMargin':>15}")
    print(f"  {'─'*52}")
    for code, d in sorted(city_margins.items(), key=lambda x: -len(x[1]["correct"]) - len(x[1]["wrong"])):
        nc = len(d["correct"])
        nw = len(d["wrong"])
        avg_wrong = sum(d["wrong"]) / nw if nw else 0.0
        print(f"  {city_from_code(code):<18} {nc:>8} {nw:>8} {avg_wrong:>14.2f}°F")

    # ── Trade performance ──────────────────────────────────────────────────────
    # Match fills to predictions
    fill_tickers = {o["ticker"] for o in fills}
    matched_fills = [(o, preds.get(o["ticker"]), outcomes.get(o["ticker"])) for o in fills]
    resolved_fills = [(o, p, oy) for o, p, oy in matched_fills if oy is not None]

    print(f"\n{'─'*64}")
    print(f"  WEATHER FILL PERFORMANCE  (filled orders vs outcomes)")
    print(f"{'─'*64}")
    print(f"  Total weather fills : {len(fills)} ({len(resolved_fills)} resolved)")

    FEE_RATE = 0.07
    if resolved_fills:
        wins, losses = [], []
        for o, p, oy in resolved_fills:
            side = o.get("side", "").lower()
            outcome_id = o.get("outcome_id", "yes")
            won = oy == (outcome_id == "yes") if side == "buy" else oy != (outcome_id == "yes")
            fill = o.get("avg_fill_price") or 0.0
            qty = o.get("filled_qty") or 0.0
            cost = fill * qty
            fee = cost * FEE_RATE
            pnl = (qty - cost - fee) if won else -(cost + fee)
            (wins if won else losses).append((o, p, oy, pnl, fill))

        total_pnl = sum(pnl for *_, pnl, _ in wins) + sum(pnl for *_, pnl, _ in losses)
        deployed = sum((o.get("avg_fill_price") or 0) * (o.get("filled_qty") or 0) for o, *_ in resolved_fills)
        roi = total_pnl / deployed * 100 if deployed else 0.0

        print(f"  W / L     : {len(wins)} / {len(losses)}  ({len(wins)/len(resolved_fills)*100:.1f}% win rate)")
        print(f"  Net PnL   : ${total_pnl:+.2f}  (ROI {roi:+.1f}% on ${deployed:.0f} deployed)")

        # Fill performance by city
        city_fill: dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0, "dep": 0.0})
        for o, p, oy, pnl, fill in wins + losses:
            parsed = parse_ticker(o["ticker"])
            code = parsed[0] if parsed else "?"
            won = pnl > 0
            city_fill[code]["w" if won else "l"] += 1
            city_fill[code]["pnl"] += pnl
            city_fill[code]["dep"] += fill * (o.get("filled_qty") or 0.0)

        print(f"\n  {'City':<18} {'W':>4} {'L':>4} {'Win%':>6} {'PnL':>9}")
        print(f"  {'─'*44}")
        for code, d in sorted(city_fill.items(), key=lambda x: -(x[1]["w"] + x[1]["l"])):
            nr = d["w"] + d["l"]
            wp = f"{d['w']/nr*100:.1f}%" if nr else "  —"
            print(f"  {city_from_code(code):<18} {d['w']:>4} {d['l']:>4} {wp:>6} {d['pnl']:>+9.2f}")

        # Fill vs prediction alignment
        with_pred = [(o, p, oy, pnl, fill) for o, p, oy, pnl, fill in wins + losses if p is not None]
        if with_pred:
            pred_correct_fills = [(o, p, oy, pnl, fill) for o, p, oy, pnl, fill in with_pred
                                   if (effective_prob(o["ticker"], p["probability"]) >= 0.5) == oy]
            pred_wrong_fills   = [(o, p, oy, pnl, fill) for o, p, oy, pnl, fill in with_pred
                                   if (effective_prob(o["ticker"], p["probability"]) >= 0.5) != oy]
            print(f"\n  Fills where sidecar predicted correctly : {len(pred_correct_fills)}/{len(with_pred)}")
            print(f"  Fills where sidecar predicted wrong     : {len(pred_wrong_fills)}/{len(with_pred)}")
            if len(with_pred) < len(resolved_fills):
                print(f"  Fills with no matching prediction       : {len(resolved_fills) - len(with_pred)}")

        if detail:
            print(f"\n  {'Result':<5} {'SidecarP':>9} {'Side':>4} {'Fill':>6}  Ticker")
            print(f"  {'─'*72}")
            for o, p, oy, pnl, fill in sorted(wins + losses, key=lambda x: x[0].get("ts", "")):
                won = pnl > 0
                ep_str = f"{effective_prob(o['ticker'], p['probability']):.3f}" if p else "   —"
                print(f"  {'WIN ' if won else 'LOSS':<5} {ep_str:>9} {o.get('outcome_id','?'):>4} {fill:>6.3f}  {o['ticker']}")

    # ── Predictions with no fill ───────────────────────────────────────────────
    # Tickers sidecar saw but bot didn't trade — what was the outcome?
    resolved_unfilled = {
        t: (p, oy)
        for t, (p, oy) in resolved.items()
        if t not in fill_tickers
    }
    if resolved_unfilled:
        ep_unf = [(t, effective_prob(t, p["probability"]), oy) for t, (p, oy) in resolved_unfilled.items()]
        n_unf = len(ep_unf)
        corr_unf = sum(1 for _, ep, oy in ep_unf if (ep >= 0.5) == oy)
        avg_b_unf = sum(brier(ep, oy) for _, ep, oy in ep_unf) / n_unf

        # Of these, how many had strong predictions (>0.65 or <0.35) that we passed on?
        strong = [(t, ep, oy) for t, ep, oy in ep_unf if ep > 0.65 or ep < 0.35]
        strong_corr = sum(1 for _, ep, oy in strong if (ep >= 0.5) == oy)

        print(f"\n{'─'*64}")
        print(f"  PREDICTED BUT NOT TRADED  (n={n_unf})")
        print(f"{'─'*64}")
        print(f"  Directional accuracy : {corr_unf}/{n_unf}  ({corr_unf/n_unf*100:.1f}%)")
        print(f"  Avg Brier            : {avg_b_unf:.4f}")
        if strong:
            print(f"  Strong signals (p>0.65 or p<0.35): {len(strong)}, {strong_corr}/{len(strong)} correct ({strong_corr/len(strong)*100:.1f}%)")
            print(f"  → These were left on the table (edge existed but order wasn't placed)")

    # ── Spread distribution: how often is the ensemble near the threshold? ─────
    print(f"\n{'─'*64}")
    print(f"  ENSEMBLE SPREAD DISTRIBUTION  (ABOVE markets)")
    print(f"{'─'*64}")
    print(f"  How close is the ensemble mean to the threshold when we're wrong?")
    spread_bins = [(-99, -10), (-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, 99)]
    bin_data: dict = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for code, items in city_spread.items():
        for spread, correct, oy, t, prob in items:
            for lo, hi in spread_bins:
                if lo <= spread < hi:
                    bin_data[(lo, hi)]["correct" if correct else "wrong"] += 1
                    break

    print(f"  {'Spread bin':>15} {'Correct':>8} {'Wrong':>7} {'Acc%':>6}")
    print(f"  {'─'*42}")
    for (lo, hi), d in sorted(bin_data.items()):
        nc, nw = d["correct"], d["wrong"]
        tot = nc + nw
        label = f"[{lo:+.0f}, {hi:+.0f})"
        print(f"  {label:>15} {nc:>8} {nw:>7} {nc/tot*100:>5.1f}%  {'← danger zone' if abs(lo) <= 2 or abs(hi) <= 2 else ''}")

    # ── Summary of actionable findings ────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  SUMMARY OF FINDINGS")
    print(f"{'='*64}")

    # Overall
    if lift > 10:
        print(f"  [GOOD]  Brier lift {lift:+.1f}% — sidecar is beating naive 50/50")
    elif lift > 0:
        print(f"  [WEAK]  Brier lift only {lift:+.1f}% — marginal improvement over 50/50")
    else:
        print(f"  [BAD]   Brier lift {lift:+.1f}% — sidecar is WORSE than 50/50")

    # Identify worst-performing cities
    bad_cities = [(code, nc, avg_b, lift_c) for code, nc, _, avg_b, lift_c, _, _ in city_rows if lift_c < 0 and nc >= 3]
    if bad_cities:
        print(f"\n  [WARN]  Cities with NEGATIVE Brier lift (sidecar hurting):")
        for code, nc, avg_b, lift_c in bad_cities:
            print(f"          {city_from_code(code):<18} n={nc}  Brier={avg_b:.4f}  lift={lift_c:+.1f}%")

    # Systematic bias by city (if ensemble mean is consistently wrong-side of threshold)
    biased_cities = []
    for code, items in city_spread.items():
        if len(items) < 3:
            continue
        mean_sp = sum(s for s, _, _, _, _ in items) / len(items)
        n_wrong = sum(1 for _, c, _, _, _ in items if not c)
        if n_wrong / len(items) > 0.4:
            biased_cities.append((code, mean_sp, n_wrong, len(items)))

    if biased_cities:
        print(f"\n  [BIAS]  Cities where ensemble is on wrong side of threshold >40%:")
        for code, mean_sp, nw, total in biased_cities:
            print(f"          {city_from_code(code):<18} mean_spread={mean_sp:+.1f}°F  wrong={nw}/{total}")
        print(f"          → Review CITY_WARM_BIAS_F corrections in sidecar.py")

    # Near-threshold losses (spread within ±2°F)
    near_losses = sum(
        1
        for code, items in city_spread.items()
        for spread, correct, oy, t, prob in items
        if not correct and abs(spread) <= 2.0
    )
    near_total = sum(
        1
        for code, items in city_spread.items()
        for spread, correct, oy, t, prob in items
        if abs(spread) <= 2.0
    )
    if near_total > 0:
        print(f"\n  [INFO]  Near-threshold (±2°F): {near_total} predictions, {near_losses} wrong ({near_losses/near_total*100:.1f}%)")
        print(f"          → Consider a min_spread filter to avoid trading near-coinflip thresholds")

    # Untradable strong signals
    if resolved_unfilled:
        strong_missed = [(t, ep, oy) for t, ep, oy in ep_unf if (ep > 0.70 or ep < 0.30) and (ep >= 0.5) == oy]
        if strong_missed:
            print(f"\n  [MISS]  {len(strong_missed)} strong correct predictions that weren't traded")
            print(f"          → May be missing market access or edge threshold too strict")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="0000-00-00", metavar="DATE")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    preds   = load_predictions(args.since)
    outcomes = load_outcomes()
    fills   = load_weather_fills(args.since)

    print(f"\n  Prediction dirs scanned:")
    for d in PRED_DIRS:
        files = sorted(d.glob("predictions_*.jsonl")) if d.exists() else []
        print(f"    {d}  ({len(files)} files)")

    analyze(preds, outcomes, fills, args.detail)


if __name__ == "__main__":
    main()
