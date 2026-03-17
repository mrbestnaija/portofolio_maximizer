"""
forcester_ts/directional_classifier.py
---------------------------------------
Phase 9: Binary directional classifier P(price_up_in_N_bars).

Loads a serialized scikit-learn Pipeline from data/classifiers/directional_v1.pkl
and scores feature vectors at inference time. Falls back to None (cold-start)
when the model file does not exist or the training set was too small.

The classifier is trained offline by scripts/train_directional_classifier.py.
This module is inference-only.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("data/classifiers/directional_v1.pkl")
_FEATURE_NAMES = [
    "ensemble_pred_return",
    "ci_width_normalized",
    "snr",
    "model_agreement",
    "directional_vote_fraction",
    "garch_conf",
    "samossa_conf",
    "mssa_rl_conf",
    "igarch_fallback_flag",
    "samossa_evr",
    "hurst_exponent",
    "trend_strength",
    "realized_vol_annualized",
    "adf_pvalue",
    "regime_liquid_rangebound",
    "regime_moderate_trending",
    "regime_high_vol_trending",
    "regime_crisis",
    "recent_return_5d",
    "recent_vol_ratio",
]


class DirectionalClassifier:
    """
    Thin wrapper around a scikit-learn Pipeline for directional scoring.

    Usage:
        clf = DirectionalClassifier()
        p_up = clf.score(features)   # float in [0, 1] or None
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._pipeline = None
        self._loaded = False
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._pipeline is not None
        self._loaded = True
        if not self._model_path.exists():
            logger.debug(
                "DirectionalClassifier: model not found at %s (cold start)", self._model_path
            )
            return False
        try:
            import joblib
            meta_path = self._model_path.with_name(self._model_path.stem + ".meta.json")
            if meta_path.exists():
                import json
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                n = meta.get("n_train", 0)
                if n < 30:
                    logger.warning(
                        "DirectionalClassifier: model trained on only %d examples (< 30); "
                        "scoring disabled",
                        n,
                    )
                    return False
            self._pipeline = joblib.load(self._model_path)
            logger.info(
                "DirectionalClassifier: loaded model from %s", self._model_path
            )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "DirectionalClassifier: failed to load model from %s: %s",
                self._model_path, exc,
            )
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, features: Dict[str, Any]) -> Optional[float]:
        """
        Return P(price_up) in [0, 1], or None if classifier is unavailable.

        Args:
            features: dict produced by TimeSeriesSignalGenerator._extract_classifier_features()

        Returns:
            float in [0, 1] or None (cold start / model unavailable)
        """
        if not self._ensure_loaded():
            return None
        try:
            import pandas as pd
            row = {name: features.get(name, float("nan")) for name in _FEATURE_NAMES}
            X = pd.DataFrame([row], columns=_FEATURE_NAMES)
            proba = self._pipeline.predict_proba(X)
            # Column 1 = P(class=1) = P(price_up)
            p_up = float(proba[0, 1])
            if not math.isfinite(p_up):
                return None
            return float(np.clip(p_up, 0.0, 1.0))
        except Exception as exc:
            logger.warning("DirectionalClassifier.score failed: %s", exc)
            return None

    def reload(self) -> None:
        """Force reload the model from disk on next score() call."""
        self._loaded = False
        self._pipeline = None
        self._load_error = None
