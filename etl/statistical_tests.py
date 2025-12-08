"""
etl.statistical_tests
---------------------

Lightweight statistical helpers for model selection and hyper-parameter
evaluation.

These functions are intentionally simple and dependency-light so they can be
used from:
- forcester_ts numeric invariant tests,
- higher-order hyper-parameter search drivers, and
- brutal/CI health checks.

They do *not* mutate any state; callers remain responsible for deciding what
constitutes an acceptable test outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Any

import numpy as np
from scipy import stats


@dataclass
class DieboldMarianoResult:
    statistic: float
    p_value: float
    better_model: str | None


def diebold_mariano(
    e1: Sequence[float],
    e2: Sequence[float],
    *,
    loss: str = "squared",
    alternative: str = "two-sided",
) -> DieboldMarianoResult:
    """
    Simple Diebold–Mariano style test for equal forecast accuracy.

    Parameters
    ----------
    e1, e2:
        Forecast errors for model 1 and model 2 over the same horizon.
    loss:
        Loss function; "squared" (default) or "absolute".
    alternative:
        "two-sided", "less" (model 1 better), or "greater" (model 2 better).

    Notes
    -----
    This uses a plain t-test on the loss differential as a pragmatic,
    dependency-light approximation. For institutional use, callers may
    replace this with a full Newey–West implementation while keeping
    the same interface.
    """
    x = np.asarray(e1, dtype=float)
    y = np.asarray(e2, dtype=float)
    if x.shape != y.shape:
        raise ValueError("e1 and e2 must have the same shape")

    if loss == "squared":
        d = x ** 2 - y ** 2
    elif loss == "absolute":
        d = np.abs(x) - np.abs(y)
    else:
        raise ValueError(f"Unsupported loss: {loss!r}")

    d = d[~np.isnan(d)]
    if d.size < 3:
        return DieboldMarianoResult(statistic=0.0, p_value=1.0, better_model=None)

    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) or 1e-12
    t_stat = mean_d / (std_d / np.sqrt(d.size))

    if alternative == "two-sided":
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=d.size - 1))
    elif alternative == "less":
        p_val = stats.t.cdf(t_stat, df=d.size - 1)
    elif alternative == "greater":
        p_val = 1 - stats.t.cdf(t_stat, df=d.size - 1)
    else:
        raise ValueError(f"Unsupported alternative: {alternative!r}")

    better: str | None
    if p_val < 0.05:
        # Negative mean_d => model 1 lower loss, positive mean_d => model 2 lower loss
        better = "model_1" if mean_d < 0 else "model_2"
    else:
        better = None

    return DieboldMarianoResult(statistic=t_stat, p_value=float(p_val), better_model=better)


def rank_stability_score(
    scores_per_fold: Dict[str, Sequence[float]],
) -> Tuple[Dict[str, float], float]:
    """
    Compute per-model average rank and an overall stability score.

    Parameters
    ----------
    scores_per_fold:
        Mapping model_name -> list of scores per CV fold (higher better).

    Returns
    -------
    (avg_ranks, stability_score)
        avg_ranks: model_name -> average rank (1 = best).
        stability_score: fraction in [0, 1] where 1 means models keep
        the same ordering across all folds.
    """
    if not scores_per_fold:
        return {}, 0.0

    # Convert to arrays shaped (n_models, n_folds)
    models = sorted(scores_per_fold.keys())
    lengths = {len(v) for v in scores_per_fold.values()}
    if len(lengths) != 1:
        raise ValueError("All score lists must have the same length")
    n_folds = lengths.pop()
    if n_folds == 0:
        return {m: 0.0 for m in models}, 0.0

    matrix = np.vstack([np.asarray(scores_per_fold[m], dtype=float) for m in models])

    # Rank within each fold (1 = best).
    ranks = np.zeros_like(matrix)
    for j in range(n_folds):
        col = matrix[:, j]
        # Higher score -> lower rank value
        order = np.argsort(-col)
        rank_values = np.empty_like(order, dtype=float)
        rank_values[order] = np.arange(1, len(models) + 1, dtype=float)
        ranks[:, j] = rank_values

    avg_ranks = {m: float(np.mean(ranks[i])) for i, m in enumerate(models)}

    # Stability: how often pairwise ordering is preserved across folds.
    stable_pairs = 0
    total_pairs = 0
    for i in range(len(models)):
        for k in range(i + 1, len(models)):
            sign_ref = np.sign(matrix[i, 0] - matrix[k, 0])
            for j in range(1, n_folds):
                diff = matrix[i, j] - matrix[k, j]
                if diff == 0 or sign_ref == 0:
                    continue
                total_pairs += 1
                if np.sign(diff) == sign_ref:
                    stable_pairs += 1

    stability = stable_pairs / total_pairs if total_pairs > 0 else 0.0
    return avg_ranks, float(stability)


class StatisticalTestSuite:
    """Convenience wrapper exposing common validation tests."""

    def test_strategy_significance(
        self, pnl_series: Sequence[float], benchmark_series: Sequence[float] | None = None
    ) -> Dict[str, Any]:
        import numpy as np
        if pnl_series is None:
            return {"p_value": 1.0, "t_stat": 0.0, "better_model": None}
        x = np.asarray(pnl_series, dtype=float)
        x = x[~np.isnan(x)]
        if benchmark_series is not None:
            y = np.asarray(benchmark_series, dtype=float)
            y = y[~np.isnan(y)]
            m = min(len(x), len(y))
            if m < 3:
                return {"p_value": 1.0, "t_stat": 0.0, "better_model": None}
            res = diebold_mariano(x[:m], y[:m])
            return {"p_value": res.p_value, "t_stat": res.statistic, "better_model": res.better_model}
        if len(x) < 3:
            return {"p_value": 1.0, "t_stat": 0.0, "better_model": None}
        t_stat, p_val = stats.ttest_1samp(x, 0.0, nan_policy="omit")
        return {"p_value": float(p_val), "t_stat": float(t_stat), "better_model": None}

    def test_autocorrelation(self, returns: Sequence[float]) -> Dict[str, Any]:
        import numpy as np
        if returns is None:
            return {"ljung_box_p": 1.0}
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        if len(r) < 5:
            return {"ljung_box_p": 1.0}
        # Simple 1-lag Ljung-Box analogue
        acf1 = np.corrcoef(r[:-1], r[1:])[0, 1]
        n = len(r)
        q = n * (n + 2) * (acf1**2) / max(n - 1, 1)
        p_val = 1 - stats.chi2.cdf(q, df=1)
        return {"ljung_box_p": float(p_val)}

    def bootstrap_validation(self, returns: Sequence[float], n_bootstrap: int = 200) -> Dict[str, Any]:
        import numpy as np
        if returns is None:
            return {"sharpe_ci": (0.0, 0.0)}
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        if len(r) < 3:
            return {"sharpe_ci": (0.0, 0.0)}
        boot = []
        for _ in range(int(n_bootstrap)):
            sample = np.random.choice(r, size=len(r), replace=True)
            mean = sample.mean()
            std = sample.std() or 1e-12
            boot.append(mean / std)
        boot = sorted(boot)
        lo = boot[int(0.025 * len(boot))]
        hi = boot[int(0.975 * len(boot))]
        return {"sharpe_ci": (float(lo), float(hi))}


__all__ = ["DieboldMarianoResult", "diebold_mariano", "rank_stability_score", "StatisticalTestSuite"]
