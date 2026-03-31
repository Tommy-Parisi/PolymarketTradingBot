#!/usr/bin/env python3
"""
Train XGBoost forecast model for event outcome prediction.

Replaces the empirical shrinkage baseline with a proper GBT trained on
market state features. Uses the existing train/val/test splits in the
forecast training dataset.

Prerequisites:
    pip install xgboost scikit-learn numpy

Usage:
    python3 scripts/train_forecast_gbt.py [--data PATH] [--out-dir PATH] [--dry-run]

Outputs:
    var/models/forecast/xgb_v1.ubj          XGBoost model (binary JSON)
    var/models/forecast/xgb_v1_features.json Feature list + encoding map
    var/models/forecast/xgb_v1_eval.json    Evaluation metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="var/features/forecast/forecast_training.jsonl",
                   help="Path to forecast_training.jsonl")
    p.add_argument("--out-dir", default="var/models/forecast",
                   help="Output directory for model artifacts")
    p.add_argument("--dry-run", action="store_true",
                   help="Load data and print stats without training")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-rows-per-vertical", type=int, default=100,
                   help="Drop verticals with fewer labeled rows than this")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# Numeric features passed directly (XGBoost handles nulls natively)
NUMERIC_FEATURES = [
    "mid_prob_yes",
    "spread_cents",
    "time_to_close_secs",
    "threshold_value",
    "finance_price_signal",
    "weather_signal",
    "crypto_sentiment_signal",
    "sports_injury_signal",
    "recent_trade_count_delta",
    "book_pressure",
]

# Categorical features — label-encoded to int, nulls → -1
CATEGORICAL_FEATURES = [
    "vertical",
    "threshold_direction",
    "market_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_category_maps(rows):
    """Return {feature: {value: int}} for categorical features."""
    maps = {}
    for feat in CATEGORICAL_FEATURES:
        vals = sorted({r["feature"][feat] for r in rows if r["feature"][feat] is not None})
        maps[feat] = {v: i for i, v in enumerate(vals)}
    return maps


def featurize(row, cat_maps):
    """Convert a training row dict to a flat float list. None → float('nan')."""
    f = row["feature"]
    vec = []
    for feat in NUMERIC_FEATURES:
        v = f.get(feat)
        vec.append(float(v) if v is not None else float("nan"))
    for feat in CATEGORICAL_FEATURES:
        v = f.get(feat)
        if v is None or v not in cat_maps[feat]:
            vec.append(float("nan"))
        else:
            vec.append(float(cat_maps[feat][v]))
    return vec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label_outcome_yes") is None:
                continue
            if r.get("label_resolution_status") == "canceled":
                continue
            rows.append(r)
    return rows


def split_rows(rows):
    train = [r for r in rows if r["split"] == "train"]
    val   = [r for r in rows if r["split"] == "validation"]
    test  = [r for r in rows if r["split"] == "test"]
    return train, val, test


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def brier_score(preds, labels):
    import numpy as np
    p = np.array(preds)
    y = np.array(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(model_brier, baseline_brier):
    """1 = perfect, 0 = same as baseline, negative = worse."""
    return 1.0 - model_brier / baseline_brier


def log_loss(preds, labels):
    import numpy as np
    eps = 1e-7
    p = np.clip(np.array(preds), eps, 1 - eps)
    y = np.array(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def print_metrics(tag, preds, labels, baseline_preds):
    bs = brier_score(preds, labels)
    bs_mid = brier_score(baseline_preds, labels)
    bss = brier_skill_score(bs, bs_mid)
    ll = log_loss(preds, labels)
    print(f"  {tag}:")
    print(f"    Brier score  : {bs:.5f}  (market mid: {bs_mid:.5f},  skill: {bss:+.3f})")
    print(f"    Log loss     : {ll:.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Dependency check
    try:
        import xgboost as xgb
        import numpy as np
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost scikit-learn numpy")
        sys.exit(1)

    import numpy as np

    # Load
    print(f"Loading {args.data} ...")
    all_rows = load_data(args.data)
    print(f"  Labeled rows: {len(all_rows):,}")

    train_rows, val_rows, test_rows = split_rows(all_rows)
    print(f"  Train: {len(train_rows):,}  Val: {len(val_rows):,}  Test: {len(test_rows):,}")

    # Vertical distribution
    from collections import Counter
    vert_counts = Counter(r["feature"]["vertical"] for r in train_rows)
    print("\nVertical distribution (train):")
    for v, n in vert_counts.most_common():
        yes = sum(r["label_outcome_yes"] for r in train_rows if r["feature"]["vertical"] == v)
        print(f"  {v:20s}  n={n:6,}  yes_rate={yes/n:.3f}")

    # Market mid Brier baseline on validation
    val_mids   = [r["feature"]["mid_prob_yes"] for r in val_rows]
    val_labels = [r["label_outcome_yes"] for r in val_rows]
    bs_mid_val = brier_score(val_mids, val_labels)
    print(f"\nMarket mid Brier (validation): {bs_mid_val:.5f}")

    if args.dry_run:
        print("\n[dry-run] Stopping before training.")
        return

    # Build category maps from train set
    cat_maps = build_category_maps(train_rows)
    print("\nCategory encodings:")
    for feat, mapping in cat_maps.items():
        print(f"  {feat}: {mapping}")

    # Featurize
    print("\nBuilding feature matrices ...")
    X_train = np.array([featurize(r, cat_maps) for r in train_rows], dtype=np.float32)
    y_train = np.array([r["label_outcome_yes"] for r in train_rows], dtype=np.float32)
    X_val   = np.array([featurize(r, cat_maps) for r in val_rows],   dtype=np.float32)
    y_val   = np.array([r["label_outcome_yes"] for r in val_rows],   dtype=np.float32)
    X_test  = np.array([featurize(r, cat_maps) for r in test_rows],  dtype=np.float32)
    y_test  = np.array([r["label_outcome_yes"] for r in test_rows],  dtype=np.float32)

    # Class imbalance weight
    pos_rate = float(y_train.mean())
    neg_rate = 1.0 - pos_rate
    scale_pos_weight = neg_rate / pos_rate if pos_rate > 0 else 1.0
    print(f"Yes rate (train): {pos_rate:.3f}  scale_pos_weight: {scale_pos_weight:.2f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ALL_FEATURES)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=ALL_FEATURES)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=ALL_FEATURES)

    # Train
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      ["logloss", "error"],
        "max_depth":        args.max_depth,
        "learning_rate":    args.learning_rate,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "tree_method":      "hist",
        "seed":             42,
        "n_jobs":           -1,
    }

    print(f"\nTraining XGBoost (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, lr={args.learning_rate}) ...")

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=50,
        early_stopping_rounds=30,
    )

    best_round = model.best_iteration
    print(f"\nBest round: {best_round}")

    # Predict
    val_preds  = model.predict(dval)
    test_preds = model.predict(dtest)
    train_preds_sample = model.predict(xgb.DMatrix(X_train[:10000], feature_names=ALL_FEATURES))

    print("\n--- Evaluation ---")
    print_metrics("Validation", val_preds, y_val, val_mids)
    test_mids = [r["feature"]["mid_prob_yes"] for r in test_rows]
    print_metrics("Test      ", test_preds, y_test, test_mids)

    # Feature importance
    importance = model.get_score(importance_type="gain")
    print("\nFeature importance (gain):")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:15]:
        print(f"  {feat:35s} {score:.2f}")

    # Per-vertical breakdown on validation
    print("\nPer-vertical Brier on validation:")
    for vert in sorted(vert_counts.keys()):
        idxs = [i for i, r in enumerate(val_rows) if r["feature"]["vertical"] == vert]
        if len(idxs) < 10:
            continue
        v_preds  = val_preds[idxs]
        v_labels = y_val[idxs]
        v_mids   = np.array(val_mids)[idxs]
        bs_m  = brier_score(v_mids, v_labels)
        bs_xg = brier_score(v_preds, v_labels)
        skill = brier_skill_score(bs_xg, bs_m)
        print(f"  {vert:20s}  n={len(idxs):5,}  mid={bs_m:.5f}  xgb={bs_xg:.5f}  skill={skill:+.3f}")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_v1.ubj"
    model.save_model(str(model_path))
    print(f"\nSaved model → {model_path}")

    features_path = out_dir / "xgb_v1_features.json"
    features_meta = {
        "model_kind": "xgboost_binary_classifier",
        "model_version": "xgb_v1",
        "feature_names": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": cat_maps,
        "best_round": best_round,
        "params": params,
    }
    with open(features_path, "w") as fh:
        json.dump(features_meta, fh, indent=2)
    print(f"Saved features → {features_path}")

    # Eval metrics
    val_bs  = brier_score(val_preds, y_val)
    test_bs = brier_score(test_preds, y_test)
    eval_path = out_dir / "xgb_v1_eval.json"
    eval_meta = {
        "model_version": "xgb_v1",
        "best_round": best_round,
        "val_brier": val_bs,
        "val_brier_mid_baseline": bs_mid_val,
        "val_brier_skill_score": brier_skill_score(val_bs, bs_mid_val),
        "test_brier": test_bs,
        "test_brier_mid_baseline": brier_score(test_mids, y_test),
        "test_brier_skill_score": brier_skill_score(test_bs, brier_score(test_mids, y_test)),
        "val_log_loss": log_loss(val_preds, y_val),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "yes_rate_train": float(pos_rate),
    }
    with open(eval_path, "w") as fh:
        json.dump(eval_meta, fh, indent=2)
    print(f"Saved eval   → {eval_path}")

    # Verdict
    print("\n" + "=" * 60)
    skill = brier_skill_score(val_bs, bs_mid_val)
    if skill > 0.01:
        print(f"PASS  XGBoost beats market mid by {skill*100:.1f}% Brier skill.")
        print("      Wire into Rust inference (see docs/execution_aware_prediction_plan.md).")
    elif skill > 0:
        print(f"MARGINAL  XGBoost beats market mid by only {skill*100:.2f}% skill.")
        print("          Consider more/better features before wiring into production.")
    else:
        print(f"FAIL  XGBoost does NOT beat market mid (skill={skill:.4f}).")
        print("      Do not replace heuristic baseline yet.")
        print("      Most likely cause: enrichment signals (weather/crypto/sports) are all null.")
        print("      Fix: ensure signals are captured during collection before retraining.")
    print("=" * 60)


if __name__ == "__main__":
    main()
