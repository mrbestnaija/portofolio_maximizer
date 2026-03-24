#!/usr/bin/env python3
"""
scripts/evaluate_directional_classifier.py
--------------------------------------------
Phase 9: Walk-forward evaluation of the directional classifier.

Produces four diagnostic outputs:
  1. Walk-forward directional accuracy (DA) per fold
  2. Expected Calibration Error (ECE) decomposition (10-bin)
  3. Win-rate counterfactual: WR with gate vs without gate
  4. Feature importance (average LR coef magnitude across calibration folds)

Reads:
  data/training/directional_dataset.parquet   — labeled dataset
  data/classifiers/directional_v1.meta.json   — model metadata (optional, for context)

Writes:
  logs/directional_eval_latest.json           — full metrics dict
  visualizations/directional_eval.txt         — ASCII report (console-safe)

Usage:
  python scripts/evaluate_directional_classifier.py
  python scripts/evaluate_directional_classifier.py --min-n 80  # override cold-start floor
  python scripts/evaluate_directional_classifier.py --no-report  # metrics only, no file write
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("data/training/directional_dataset.parquet")
_META_PATH = Path("data/classifiers/directional_v1.meta.json")
_EVAL_OUTPUT = Path("logs/directional_eval_latest.json")
_REPORT_OUTPUT = Path("visualizations/directional_eval.txt")
_COLD_START_MIN_N = 60
_ECE_N_BINS = 10


# ---------------------------------------------------------------------------
# Walk-forward DA
# ---------------------------------------------------------------------------

def _walk_forward_da(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    gap: int = 30,
    c_values: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Walk-forward directional accuracy with inner-loop C selection.
    Returns (fold_results, mean_da).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0]

    _n = len(y)
    _gap = min(gap, _n // 10)
    # Ensure feasibility: need at least n_splits * (test_size + gap) < n
    _ns = n_splits
    for _s in range(n_splits, 0, -1):
        _test_size = _n // (_s + 1)
        if _s * (_test_size + _gap) < _n:
            _ns = _s
            break

    if _ns < 2:
        logger.warning("Too few samples for walk-forward CV (n=%d); using 1 fold", _n)
        _ns = max(1, _ns)

    tscv = TimeSeriesSplit(n_splits=_ns, gap=_gap)
    fold_results: List[Dict[str, Any]] = []
    all_das: List[float] = []

    X_df = pd.DataFrame(X, columns=feature_names)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_df)):
        X_tr, X_te = X_df.iloc[train_idx].values, X_df.iloc[test_idx].values
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            logger.debug("Fold %d: skipped (single class or empty test set)", fold_idx)
            continue

        # Inner-loop: select best C on this fold's train set via 2-fold CV
        best_c, best_inner_da = 1.0, -1.0
        X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
        for c in c_values:
            inner_cv = TimeSeriesSplit(n_splits=2, gap=max(1, _gap // 2))
            inner_das = []
            for itr, ite in inner_cv.split(X_tr_df):
                if len(np.unique(y_tr[itr])) < 2 or len(ite) == 0:
                    continue
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="mean")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=c, max_iter=500, solver="lbfgs", class_weight="balanced"
                    )),
                ])
                pipe.fit(X_tr_df.iloc[itr], y_tr[itr])
                preds = pipe.predict(X_tr_df.iloc[ite])
                inner_das.append(float((preds == y_tr[ite]).mean()))
            if inner_das and np.mean(inner_das) > best_inner_da:
                best_inner_da = float(np.mean(inner_das))
                best_c = c

        # Fit calibrated model on this fold's train set
        _calib_cv = 3 if len(y_tr) >= 90 else 2
        base_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=best_c, max_iter=500, solver="lbfgs", class_weight="balanced"
            )),
        ])
        try:
            calibrated = CalibratedClassifierCV(base_pipe, cv=_calib_cv, method="sigmoid")
            calibrated.fit(pd.DataFrame(X_tr, columns=feature_names), y_tr)
            proba = calibrated.predict_proba(pd.DataFrame(X_te, columns=feature_names))
            preds = (proba[:, 1] >= 0.5).astype(int)
        except Exception as exc:
            logger.warning("Fold %d: calibrated fit failed (%s); using plain pipeline", fold_idx, exc)
            base_pipe.fit(pd.DataFrame(X_tr, columns=feature_names), y_tr)
            preds = base_pipe.predict(pd.DataFrame(X_te, columns=feature_names))

        da = float((preds == y_te).mean())
        all_das.append(da)
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "best_c": best_c,
            "directional_accuracy": round(da, 4),
        })
        logger.info("Fold %d: DA=%.3f (n_train=%d, n_test=%d, C=%.3f)",
                    fold_idx, da, len(y_tr), len(y_te), best_c)

    mean_da = float(np.mean(all_das)) if all_das else float("nan")
    return fold_results, mean_da


# ---------------------------------------------------------------------------
# ECE decomposition
# ---------------------------------------------------------------------------

def _ece_decomposition(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = _ECE_N_BINS,
) -> Dict[str, Any]:
    """
    Expected Calibration Error decomposition.
    Returns dict with ece, bins (confidence, accuracy, fraction, count).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_results = []
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_pred >= lo) & (p_pred < hi) if i < n_bins - 1 else (p_pred >= lo) & (p_pred <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bin_results.append({
                "bin_lower": round(float(lo), 2),
                "bin_upper": round(float(hi), 2),
                "confidence": None,
                "accuracy": None,
                "fraction": 0.0,
                "count": 0,
            })
            continue
        conf = float(p_pred[mask].mean())
        acc = float(y_true[mask].mean())
        frac = n_bin / n_total
        ece += frac * abs(conf - acc)
        bin_results.append({
            "bin_lower": round(float(lo), 2),
            "bin_upper": round(float(hi), 2),
            "confidence": round(conf, 4),
            "accuracy": round(acc, 4),
            "fraction": round(frac, 4),
            "count": n_bin,
        })

    return {"ece": round(float(ece), 4), "n_bins": n_bins, "bins": bin_results}


# ---------------------------------------------------------------------------
# Win-rate counterfactual
# ---------------------------------------------------------------------------

def _win_rate_counterfactual(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    p_up_threshold: float = 0.55,
    p_down_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Compare win rate of gated signals vs baseline (all signals).

    Gate logic mirrors production:
      BUY  passes if p_up >= p_up_threshold
      SELL passes if (1-p_up) >= p_down_threshold  i.e. p_up <= 1-p_down_threshold

    Since labels are 1=price_up, 0=price_down:
      - Gated BUY signals are rows where p_pred >= p_up_threshold; win = y_true=1
      - Gated SELL signals are rows where p_pred <= 1-p_down_threshold; win = y_true=0
    """
    # Baseline: treat every sample as a BUY (predict price up)
    baseline_wr = float(y_true.mean()) if len(y_true) > 0 else float("nan")

    # Gated BUY pass
    buy_mask = p_pred >= p_up_threshold
    n_buy = int(buy_mask.sum())
    buy_wr = float(y_true[buy_mask].mean()) if n_buy > 0 else float("nan")

    # Gated SELL pass (price-down signal: p_up <= 1 - p_down_threshold)
    sell_mask = p_pred <= (1.0 - p_down_threshold)
    n_sell = int(sell_mask.sum())
    sell_wr = float((1 - y_true[sell_mask]).mean()) if n_sell > 0 else float("nan")

    # Blocked by gate
    n_blocked = int((~buy_mask & ~sell_mask).sum())

    return {
        "p_up_threshold_buy": p_up_threshold,
        "p_down_threshold_sell": p_down_threshold,
        "n_total": len(y_true),
        "n_gated_buy": n_buy,
        "n_gated_sell": n_sell,
        "n_blocked": n_blocked,
        "baseline_win_rate": round(baseline_wr, 4) if np.isfinite(baseline_wr) else None,
        "gated_buy_win_rate": round(buy_wr, 4) if np.isfinite(buy_wr) else None,
        "gated_sell_win_rate": round(sell_wr, 4) if np.isfinite(sell_wr) else None,
        "gate_lift_buy": (
            round(buy_wr - baseline_wr, 4)
            if np.isfinite(buy_wr) and np.isfinite(baseline_wr)
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _feature_importance_from_meta(meta_path: Path) -> List[Dict[str, Any]]:
    """Load top3_features from training meta sidecar."""
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("top3_features") or []
    except Exception as exc:
        logger.warning("Could not load meta from %s: %s", meta_path, exc)
        return []


# ---------------------------------------------------------------------------
# ASCII report
# ---------------------------------------------------------------------------

def _build_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Directional Classifier Evaluation Report")
    lines.append(f"Generated: {result.get('evaluated_at', 'unknown')}")
    lines.append("=" * 60)

    # Dataset
    lines.append(f"\n[Dataset]")
    lines.append(f"  n_labeled       : {result.get('n_labeled')}")
    lines.append(f"  n_positive      : {result.get('n_positive')}")
    lines.append(f"  n_negative      : {result.get('n_negative')}")
    lines.append(f"  base_win_rate   : {result.get('base_win_rate')}")

    # Walk-forward DA
    wf = result.get("walk_forward", {})
    lines.append(f"\n[Walk-Forward Directional Accuracy]")
    lines.append(f"  mean_da         : {wf.get('mean_da')}")
    lines.append(f"  n_folds         : {wf.get('n_folds')}")
    for fold in wf.get("folds", []):
        lines.append(
            f"  fold {fold['fold']}: DA={fold['directional_accuracy']:.3f} "
            f"(n_train={fold['n_train']}, n_test={fold['n_test']}, C={fold['best_c']})"
        )

    # ECE
    ece = result.get("ece", {})
    lines.append(f"\n[Expected Calibration Error (ECE)]")
    lines.append(f"  ECE             : {ece.get('ece')} ({_ECE_N_BINS}-bin)")
    lines.append(f"  {'Bin':>12}  {'Confidence':>12}  {'Accuracy':>10}  {'Count':>7}")
    for b in ece.get("bins", []):
        if b["count"] == 0:
            continue
        bin_label = f"[{b['bin_lower']:.1f},{b['bin_upper']:.1f})"
        lines.append(
            f"  {bin_label:>12}  {b['confidence']:>12.3f}  {b['accuracy']:>10.3f}  {b['count']:>7}"
        )

    # Counterfactual
    cf = result.get("counterfactual", {})
    lines.append(f"\n[Win-Rate Counterfactual (gate thresholds)]")
    lines.append(f"  p_up_threshold_buy  : {cf.get('p_up_threshold_buy')}")
    lines.append(f"  n_total             : {cf.get('n_total')}")
    lines.append(f"  n_gated_buy         : {cf.get('n_gated_buy')}")
    lines.append(f"  n_gated_sell        : {cf.get('n_gated_sell')}")
    lines.append(f"  n_blocked           : {cf.get('n_blocked')}")
    lines.append(f"  baseline_win_rate   : {cf.get('baseline_win_rate')}")
    lines.append(f"  gated_buy_win_rate  : {cf.get('gated_buy_win_rate')}")
    lines.append(f"  gate_lift_buy       : {cf.get('gate_lift_buy')}")

    # Feature importance
    fi = result.get("top3_features", [])
    if fi:
        lines.append(f"\n[Feature Importance (top 3, from training meta)]")
        for f in fi:
            lines.append(f"  {f['name']:35s}: coef={f['coef']:+.4f}")

    # Overall verdict
    lines.append(f"\n[Verdict]")
    mean_da = wf.get("mean_da")
    ece_val = ece.get("ece")
    gate_lift = cf.get("gate_lift_buy")
    if mean_da is not None:
        da_ok = mean_da > 0.52
        lines.append(f"  DA > 0.52       : {'PASS' if da_ok else 'WARN'} ({mean_da:.3f})")
    if ece_val is not None:
        ece_ok = ece_val < 0.10
        lines.append(f"  ECE < 0.10      : {'PASS' if ece_ok else 'WARN'} ({ece_val:.4f})")
    if gate_lift is not None:
        lift_ok = gate_lift > 0.0
        lines.append(f"  Gate lift > 0   : {'PASS' if lift_ok else 'WARN'} ({gate_lift:+.4f})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _optimal_gate_threshold(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    candidate_thresholds: List[float] | None = None,
    min_gated: int = 10,
) -> Tuple[float, bool]:
    """
    Return (threshold, optimized) where:
      - threshold: the p_up value that maximises gate_lift_buy subject to
        at least min_gated BUY examples passing the gate.
      - optimized: True if at least one candidate met min_gated; False means
        the threshold is a hardcoded fallback (0.55), NOT a data-driven result.

    Callers MUST check the second return value before treating the threshold
    as meaningful — a False here means the gate is effectively uncalibrated.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(t, 2) for t in np.arange(0.50, 0.91, 0.05)]

    baseline_wr = float(y_true.mean()) if len(y_true) > 0 else float("nan")
    best_thresh = 0.55
    best_lift = float("-inf")
    any_candidate_valid = False

    for thresh in candidate_thresholds:
        mask = p_pred >= thresh
        n_gated = int(mask.sum())
        if n_gated < min_gated:
            continue
        any_candidate_valid = True
        gated_wr = float(y_true[mask].mean())
        lift = gated_wr - baseline_wr
        if lift > best_lift:
            best_lift = lift
            best_thresh = thresh

    if not any_candidate_valid:
        logger.warning(
            "Gate threshold optimization failed: no candidate threshold achieved "
            "min_gated=%d examples (n_test=%d). Returning uncalibrated fallback 0.55. "
            "Increase dataset size before activating classifier gate.",
            min_gated,
            len(y_true),
        )

    return best_thresh, any_candidate_valid


def evaluate(
    dataset_path: Path = _DATASET_PATH,
    meta_path: Path = _META_PATH,
    min_n: int = _COLD_START_MIN_N,
    write_report: bool = True,
    p_up_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Run full evaluation suite. Returns result dict.

    Args:
        p_up_threshold: Gate threshold for BUY signals (p_up >= threshold).
            When None (default), the threshold is auto-optimised by scanning
            [0.50, 0.55, ..., 0.90] and picking the value that maximises
            gate_lift_buy subject to at least 10 gated examples.
    """
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return {"error": "dataset_not_found"}

    try:
        from sklearn.calibration import CalibratedClassifierCV  # noqa: F401 — availability check
        from sklearn.impute import SimpleImputer  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as exc:
        logger.error("scikit-learn not available: %s", exc)
        return {"error": "sklearn_unavailable"}

    from forcester_ts.directional_classifier import _FEATURE_NAMES

    try:
        df = pd.read_parquet(dataset_path)
    except Exception as exc:
        logger.error("Dataset unreadable at %s: %s", dataset_path, exc)
        return {"error": "dataset_unreadable"}
    if "y_directional" not in df.columns:
        return {"error": "missing_label_column"}

    df = df.sort_values("entry_ts", na_position="last")
    y = df["y_directional"].astype(int).values
    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos

    if n_total < min_n or n_pos < 10 or n_neg < 10:
        logger.warning(
            "Cold start: n=%d (min=%d), n_pos=%d, n_neg=%d — evaluation skipped",
            n_total, min_n, n_pos, n_neg,
        )
        return {
            "cold_start": True,
            "n_labeled": n_total,
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

    X_df = df.reindex(columns=_FEATURE_NAMES).astype(float)
    X = X_df.values

    # 1. Walk-forward DA
    logger.info("Running walk-forward DA evaluation (n=%d)...", n_total)
    folds, mean_da = _walk_forward_da(X, y, _FEATURE_NAMES)
    wf_result = {
        "mean_da": round(mean_da, 4) if np.isfinite(mean_da) else None,
        "n_folds": len(folds),
        "folds": folds,
    }

    # 2. ECE on held-out last 20% (time-ordered)
    split_idx = max(1, int(n_total * 0.8))
    X_tr_ece, X_te_ece = X[:split_idx], X[split_idx:]
    y_tr_ece, y_te_ece = y[:split_idx], y[split_idx:]
    ece_result: Dict[str, Any] = {"ece": None, "n_bins": _ECE_N_BINS, "bins": []}
    if len(y_te_ece) >= 10 and len(np.unique(y_tr_ece)) >= 2:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        _calib_cv_ece = 3 if split_idx >= 90 else 2
        base_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", class_weight="balanced")),
        ])
        try:
            cal = CalibratedClassifierCV(base_pipe, cv=_calib_cv_ece, method="sigmoid")
            cal.fit(pd.DataFrame(X_tr_ece, columns=_FEATURE_NAMES), y_tr_ece)
            p_pred = cal.predict_proba(pd.DataFrame(X_te_ece, columns=_FEATURE_NAMES))[:, 1]
            ece_result = _ece_decomposition(y_te_ece, p_pred)
            logger.info("ECE = %.4f (n_test=%d)", ece_result["ece"], len(y_te_ece))
        except Exception as exc:
            logger.warning("ECE computation failed: %s", exc)
    else:
        logger.warning("Too few held-out samples for ECE (n_test=%d)", len(y_te_ece))

    # 3. Win-rate counterfactual on entire labeled set with leave-one-out proxy.
    #    We use the last 20% held-out predictions from the ECE step for consistency.
    #    Gate threshold: auto-optimise when not explicitly provided.
    cf_result: Dict[str, Any] = {}
    if len(y_te_ece) >= 10 and ece_result.get("ece") is not None:
        if p_up_threshold is None:
            _gate_thresh, _thresh_optimized = _optimal_gate_threshold(y_te_ece, p_pred)
            logger.info(
                "Gate threshold: %.2f (%s)",
                _gate_thresh,
                "data-driven" if _thresh_optimized else "UNCALIBRATED FALLBACK",
            )
        else:
            _gate_thresh = p_up_threshold
            _thresh_optimized = True  # explicitly supplied by caller
        cf_result = _win_rate_counterfactual(y_te_ece, p_pred, p_up_threshold=_gate_thresh)
        cf_result["threshold_optimized"] = _thresh_optimized
    else:
        cf_result = {
            "note": "insufficient held-out data for counterfactual",
            "n_total": n_total,
            "threshold_optimized": False,
        }

    # 4. Feature importance from training meta
    top3_features = _feature_importance_from_meta(meta_path)

    result = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(dataset_path),
        "n_labeled": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "base_win_rate": round(float(n_pos / n_total), 4),
        "cold_start": False,
        "walk_forward": wf_result,
        "ece": ece_result,
        "counterfactual": cf_result,
        "top3_features": top3_features,
    }

    if write_report:
        _EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = _EVAL_OUTPUT.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        tmp.replace(_EVAL_OUTPUT)
        logger.info("Wrote eval metrics to %s", _EVAL_OUTPUT)

        report = _build_report(result)
        _REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_OUTPUT.write_text(report, encoding="utf-8")
        logger.info("Wrote ASCII report to %s", _REPORT_OUTPUT)
        print(report)

    return result


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-n",
        type=int,
        default=_COLD_START_MIN_N,
        help=f"Minimum labeled examples required (default: {_COLD_START_MIN_N})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing report files (print to console only)",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "p_up threshold for BUY gate (0.0-1.0). "
            "Default: auto-optimise by scanning [0.50..0.90] to maximise gate lift."
        ),
    )
    args = parser.parse_args(argv)
    result = evaluate(
        min_n=args.min_n,
        write_report=not args.no_report,
        p_up_threshold=args.gate_threshold,
    )
    if result.get("error"):
        print(f"[ERROR] {result['error']}")
        return 1
    if result.get("cold_start"):
        print(
            f"[COLD_START] n={result.get('n_labeled')} < {args.min_n} or class imbalance — "
            "evaluation skipped"
        )
        return 2
    wf_da = result.get("walk_forward", {}).get("mean_da")
    ece_val = result.get("ece", {}).get("ece")
    print(f"[OK] walk_forward_da={wf_da}  ece={ece_val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
