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
from typing import Dict, Sequence, Tuple, Any, Optional

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


@dataclass
class BootstrapIntervals:
    """Container for bootstrap confidence intervals."""

    sharpe_ratio: Tuple[float, float]
    max_drawdown: Tuple[float, float]
    samples: int
    confidence_level: float


class StatisticalTestSuite:
    """Convenience wrapper exposing common validation tests."""

    def _clean_series(self, values: Sequence[float] | None) -> np.ndarray:
        if values is None:
            return np.array([], dtype=float)
        arr = np.asarray(values, dtype=float)
        return arr[~np.isnan(arr)]

    def test_strategy_significance(
        self,
        pnl_series: Sequence[float],
        benchmark_series: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Basic significance test with information ratio.

        - With a benchmark: Diebold–Mariano on diff, plus information ratio.
        - Without a benchmark: one-sample t-test vs 0.
        """
        x = self._clean_series(pnl_series)
        if benchmark_series is not None:
            y = self._clean_series(benchmark_series)
            m = min(len(x), len(y))
            if m < 3:
                return {"p_value": 1.0, "t_stat": 0.0, "information_ratio": 0.0, "significant": False}
            diff = x[:m] - y[:m]
            ir = float(diff.mean() / (diff.std() or 1e-12))
            res = diebold_mariano(diff, np.zeros_like(diff))
            return {
                "p_value": res.p_value,
                "t_stat": res.statistic,
                "information_ratio": ir,
                "significant": res.p_value < 0.05,
            }

        if len(x) < 3:
            return {"p_value": 1.0, "t_stat": 0.0, "information_ratio": 0.0, "significant": False}
        t_stat, p_val = stats.ttest_1samp(x, 0.0, nan_policy="omit")
        ir = float(x.mean() / (x.std() or 1e-12))
        return {"p_value": float(p_val), "t_stat": float(t_stat), "information_ratio": ir, "significant": p_val < 0.05}

    def test_autocorrelation(self, returns: Sequence[float], lags: int = 1) -> Dict[str, Any]:
        """
        Return Ljung–Box style statistic/p-value plus Durbin–Watson.
        """
        r = self._clean_series(returns)
        if len(r) < max(lags + 1, 3):
            return {"ljung_box_stat": 0.0, "ljung_box_p_value": 1.0, "durbin_watson": 0.0}

        acf_vals = []
        for lag in range(1, max(2, lags + 1)):
            corr = float(np.corrcoef(r[:-lag], r[lag:])[0, 1])
            if np.isnan(corr):
                corr = 0.0
            acf_vals.append(corr)

        n = len(r)
        lb_stat = n * (n + 2) * sum((acf_vals[k - 1] ** 2) / max(n - k, 1) for k in range(1, len(acf_vals) + 1))
        lb_p = 1 - stats.chi2.cdf(lb_stat, df=len(acf_vals))

        # Durbin–Watson
        diff = np.diff(r)
        dw = float(np.sum(diff * diff) / (np.sum(r * r) or 1e-12))

        return {"ljung_box_stat": float(lb_stat), "ljung_box_p_value": float(lb_p), "durbin_watson": dw}

    def bootstrap_validation(
        self,
        returns: Sequence[float],
        n_bootstrap: int = 200,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> BootstrapIntervals:
        """
        Bootstrap Sharpe ratio and max drawdown intervals.
        """
        r = self._clean_series(returns)
        if len(r) < 3:
            return BootstrapIntervals(
                sharpe_ratio=(0.0, 0.0),
                max_drawdown=(0.0, 0.0),
                samples=int(n_bootstrap),
                confidence_level=float(confidence_level),
            )

        rng = np.random.default_rng(random_state)
        sharpes = []
        drawdowns = []
        for _ in range(int(n_bootstrap)):
            sample = rng.choice(r, size=len(r), replace=True)
            mu = float(sample.mean())
            sigma = float(sample.std() or 1e-12)
            sharpes.append(mu / sigma)
            drawdowns.append(self._max_drawdown(sample))

        alpha = max(min((1.0 - confidence_level) / 2.0, 0.499), 0.0)
        lo_q = alpha
        hi_q = 1 - alpha
        sr_low, sr_high = np.quantile(sharpes, [lo_q, hi_q])
        dd_low, dd_high = np.quantile(drawdowns, [lo_q, hi_q])

        return BootstrapIntervals(
            sharpe_ratio=(float(sr_low), float(sr_high)),
            max_drawdown=(float(dd_low), float(dd_high)),
            samples=int(n_bootstrap),
            confidence_level=float(confidence_level),
        )

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Compute max drawdown on cumulative return path."""
        curve = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(curve)
        drawdowns = curve / peak - 1.0
        return float(drawdowns.min()) if drawdowns.size else 0.0


__all__ = [
    "DieboldMarianoResult",
    "diebold_mariano",
    "rank_stability_score",
    "BootstrapIntervals",
    "StatisticalTestSuite",
]
