#!/usr/bin/env python3
"""
scripts/train_directional_classifier.py
-----------------------------------------
Phase 9: Train binary directional classifier P(price_up_in_N_bars).

Reads data/training/directional_dataset.parquet produced by
build_directional_training_data.py, trains a scikit-learn Pipeline
(SimpleImputer -> StandardScaler -> LogisticRegression) wrapped in
CalibratedClassifierCV (Platt/sigmoid calibration), and saves:

  data/classifiers/directional_v1.pkl         — fitted CalibratedClassifierCV
  data/classifiers/directional_v1.meta.json   — version fingerprint + metrics

Calibration note: class_weight='balanced' during walk-forward CV tunes the
decision boundary for class-imbalanced data. The final model is wrapped in
CalibratedClassifierCV(method='sigmoid') so that predict_proba() returns
genuine probabilities rather than the distorted scores produced by balanced
weighting alone. Platt sigmoid calibration is used (not isotonic) because it
is reliable for n < 1000 and avoids the overfitting risk of isotonic regression
on small datasets.

Exits with code 2 when cold_start is True (insufficient data).

Usage:
  python scripts/train_directional_classifier.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("data/training/directional_dataset.parquet")
_SUMMARY_PATH = Path("logs/directional_training_latest.json")
_MODEL_DIR = Path("data/classifiers")
_MODEL_PATH = _MODEL_DIR / "directional_v1.pkl"
_META_PATH = _MODEL_DIR / "directional_v1.meta.json"
_MIN_TRAIN_N = 60
_MIN_CLASS = 10

from forcester_ts.directional_classifier import _FEATURE_NAMES


def train(
    dataset_path: Path = _DATASET_PATH,
    model_path: Path = _MODEL_PATH,
    meta_path: Path = _META_PATH,
    c_values: Optional[list] = None,
) -> Dict[str, Any]:
    """Train the directional classifier. Returns result dict."""
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return {"error": "dataset_not_found", "cold_start": True}

    try:
        df = pd.read_parquet(dataset_path)
    except Exception as exc:
        logger.error("Cannot read dataset parquet %s: %s", dataset_path, exc)
        return {"error": "dataset_unreadable", "cold_start": True, "detail": str(exc)}
    if "y_directional" not in df.columns:
        return {"error": "missing_label_column", "cold_start": True}

    df = df.sort_values("entry_ts", na_position="last")
    y = df["y_directional"].astype(int).values

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    if len(y) < _MIN_TRAIN_N or n_pos < _MIN_CLASS or n_neg < _MIN_CLASS:
        logger.warning(
            "Cold start: n=%d, n_pos=%d, n_neg=%d — minimum not met (%d / %d each class)",
            len(y), n_pos, n_neg, _MIN_TRAIN_N, _MIN_CLASS,
        )
        return {
            "cold_start": True,
            "n_train": len(y),
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

    # Align feature columns (fill missing features with NaN)
    X_df = df.reindex(columns=_FEATURE_NAMES)

    # Walk-forward CV for hyperparameter selection
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        logger.error("scikit-learn not available: %s", exc)
        return {"error": "sklearn_unavailable"}

    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0]

    # TimeSeriesSplit: each fold needs at least (test_size + gap) samples beyond the training set.
    # test_size = n // (n_splits + 1); gap = 30. Ensure feasibility: n_splits * (test_size + gap) < n.
    _n = len(y)
    _gap = min(30, _n // 10)  # shrink gap on small datasets to stay feasible
    _n_splits = 1
    for _s in range(5, 0, -1):
        _test_size = _n // (_s + 1)
        if _s * (_test_size + _gap) < _n:
            _n_splits = _s
            break
    tscv = TimeSeriesSplit(n_splits=_n_splits, gap=_gap)

    best_c = 1.0
    best_da = -1.0
    fold_results = []

    for c in c_values:
        das = []
        for train_idx, test_idx in tscv.split(X_df):
            X_tr, X_te = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
                continue
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=c, max_iter=500, solver="lbfgs", class_weight="balanced"
                )),
            ])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            da = float((preds == y_te).mean())
            das.append(da)
        mean_da = float(np.mean(das)) if das else 0.5
        fold_results.append({"C": c, "mean_da": mean_da, "n_folds": len(das)})
        if mean_da > best_da:
            best_da = mean_da
            best_c = c

    logger.info("Best C=%.3f (walk-forward DA=%.3f)", best_c, best_da)

    # Final fit on all data with best C, wrapped in Platt calibration.
    #
    # D4 fix: class_weight='balanced' adjusts penalty weights so the LR decision
    # boundary handles imbalanced labels, but it distorts predict_proba() outputs
    # (overconfident positives on majority class, underconfident on minority).
    # CalibratedClassifierCV(method='sigmoid') applies Platt scaling: fits a
    # sigmoid on out-of-fold probability estimates to re-anchor them to true
    # empirical frequencies.  Sigmoid ('Platt') is used (not 'isotonic') because
    # it is robust for n < 1000 and avoids isotonic's overfitting on small sets.
    #
    # cv_count is set adaptively so each calibration fold has ≥ 30 samples:
    #   n < 90  → cv=2  (~30 per fold after gap)
    #   n ≥ 90  → cv=3  (~30 per fold after gap)
    # Minimum 2 folds enforced; maximum capped at 3 (diminishing returns beyond).
    _n = len(y)
    _calib_cv = 3 if _n >= 90 else 2
    logger.info("Calibration CV folds: %d (n=%d)", _calib_cv, _n)

    base_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=best_c, max_iter=500, solver="lbfgs", class_weight="balanced"
        )),
    ])
    final_model = CalibratedClassifierCV(base_pipe, cv=_calib_cv, method="sigmoid")
    final_model.fit(X_df, y)

    # Save model (CalibratedClassifierCV wrapping the pipeline)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    tmp = model_path.with_suffix(".tmp.pkl")
    joblib.dump(final_model, tmp)
    tmp.replace(model_path)

    # Coefficient feature importance: average LR coefs across calibration folds.
    # CalibratedClassifierCV fits one base_pipe per calibration fold internally;
    # we access each fold's estimator to extract coef_ and average across folds.
    try:
        all_coefs = [
            cc.estimator.named_steps["clf"].coef_[0]
            for cc in final_model.calibrated_classifiers_
        ]
        mean_coefs = np.mean(all_coefs, axis=0)
        feature_importance = {name: float(coef) for name, coef in zip(_FEATURE_NAMES, mean_coefs)}
    except Exception as exc:
        logger.warning("Could not extract feature importance from calibrated model: %s", exc)
        feature_importance = {}
    top3 = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    # Build meta sidecar
    import importlib.metadata as _im
    import sys as _sys
    libraries = {}
    for lib in ("joblib", "numpy", "pandas", "scikit-learn", "statsmodels"):
        try:
            libraries[lib] = _im.version(lib)
        except Exception:
            libraries[lib] = "unknown"

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train": len(y),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "best_c": best_c,
        "walk_forward_da": round(best_da, 4),
        "fold_results": fold_results,
        "top3_features": [{"name": k, "coef": round(v, 4)} for k, v in top3],
        "cold_start": False,
        "calibration_method": "sigmoid",
        "calibration_cv_folds": _calib_cv,
        "feature_names": list(_FEATURE_NAMES),  # persisted for inference-time validation
        "python_version": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        "libraries": libraries,
        "schema_version": 2,  # v2: CalibratedClassifierCV (sigmoid) wrapping base pipeline
    }
    tmp_meta = meta_path.with_suffix(".tmp.json")
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_meta.replace(meta_path)

    logger.info(
        "Saved model to %s (n=%d, DA=%.3f, top_feature=%s)",
        model_path, len(y), best_da, top3[0][0] if top3 else "n/a",
    )
    return meta


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    result = train()
    if result.get("error"):
        print(f"[ERROR] {result['error']}")
        return 1
    if result.get("cold_start"):
        n = result.get("n_train", 0)
        print(f"[COLD_START] n={n} < {_MIN_TRAIN_N} or class imbalance — model not saved")
        return 2
    print(f"[OK] n={result.get('n_train')} walk_forward_da={result.get('walk_forward_da')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
