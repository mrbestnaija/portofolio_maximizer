from __future__ import annotations

import numpy as np


def ensure_psd(matrix: np.ndarray, n_assets: int) -> np.ndarray:
    if matrix.shape != (n_assets, n_assets):
        return np.eye(n_assets)
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        eps = 1e-4
        return matrix + np.eye(n_assets) * eps


def target_correlation(cfg: dict, n_assets: int) -> np.ndarray:
    mode = str(cfg.get("mode", "static"))
    target = cfg.get("target_matrix") or []
    if mode == "factor":
        strength = float(cfg.get("factor_strength", 0.4))
        strength = float(np.clip(strength, -0.95, 0.95))
        mat = np.full((n_assets, n_assets), strength)
        np.fill_diagonal(mat, 1.0)
        return ensure_psd(mat, n_assets)
    if mode == "rolling":
        jitter = float(cfg.get("jitter", 0.01))
        base = np.array(target, dtype=float) if target else np.eye(n_assets)
        mat = base + np.eye(n_assets) * jitter
        return ensure_psd(mat, n_assets)
    if not target:
        return np.eye(n_assets)
    return ensure_psd(np.array(target, dtype=float), n_assets)


def copula_shocks(rng: np.random.Generator, corr: np.ndarray, n: int, df: float) -> np.ndarray:
    """Generate Student-t copula shocks to approximate tail dependence."""
    n_assets = corr.shape[0]
    g = rng.standard_normal(size=(n, n_assets))
    L = np.linalg.cholesky(corr)
    z = g @ L.T
    chi2 = rng.chisquare(df, size=n) / df
    t_samples = z / np.sqrt(chi2)[:, None]
    return t_samples
