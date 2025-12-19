from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


def maybe_apply_ml_backend(data: pd.DataFrame, ml_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Minimal ML backend stub for TimeGAN/diffusion placeholders.

    If ml_cfg["enabled"] is true, apply a light perturbation/noise-based
    augmentation to emulate an ML generator step without heavy deps.
    """
    if not ml_cfg or not ml_cfg.get("enabled"):
        return data
    backend = str(ml_cfg.get("backend", "timegan")).lower()
    rng = np.random.default_rng(int(ml_cfg.get("seed", 0)) or 0)
    augmented = data.copy()
    if backend in {"timegan", "diffusion"}:
        noise_scale = float(ml_cfg.get("noise_scale", 0.002))
        for col in ["Open", "High", "Low", "Close"]:
            if col in augmented.columns:
                augmented[col] = augmented[col].astype(float) * (1 + rng.normal(0, noise_scale, size=len(augmented)))

        # Enforce basic OHLC invariants so downstream validators/forecasters
        # do not see impossible candles after perturbations.
        if all(col in augmented.columns for col in ("Open", "High", "Low", "Close")):
            eps = 1e-6
            for col in ("Open", "High", "Low", "Close"):
                augmented[col] = np.maximum(augmented[col].astype(float), eps)

            oc_max = augmented[["Open", "Close"]].max(axis=1)
            oc_min = augmented[["Open", "Close"]].min(axis=1)
            augmented["High"] = np.maximum(augmented["High"].astype(float), oc_max)
            augmented["Low"] = np.minimum(augmented["Low"].astype(float), oc_min)
            augmented["High"] = np.maximum(augmented["High"], augmented["Low"])
    # Record provenance
    augmented.attrs["ml_backend"] = backend
    augmented.attrs["ml_enabled"] = True
    return augmented
