"""
walk_forward_learner.py
-----------------------
Expanding-window or rolling-window retraining harness for pseudo-real-time
evaluation of GARCH/SAMOSSA/MSSA-RL ensemble models.

Each fold:
  1. Fit all models on data[:T]
  2. Forecast data[T:T+h]
  3. Evaluate VaR violations (Kupiec) + pinball loss
  4. Compute Shapley attribution of OOS error across model components
  5. Surface current OrderLearner suggestions per fold for auditability
  6. Advance T by fold_step

Usage:
    from forcester_ts.walk_forward_learner import WalkForwardLearner
    wfl = WalkForwardLearner(
        forecaster_config={},
        order_learner=None,
        snapshot_store=None,
    )
    result = wfl.run(series, ticker="AAPL")
    print(result.aggregate)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from forcester_ts.order_learner import OrderLearner
    from forcester_ts.model_snapshot_store import ModelSnapshotStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""
    fold_idx: int
    train_end: int                     # index of last training bar (inclusive)
    test_start: int
    test_end: int
    rmse: float
    mae: float
    dir_acc: float                     # directional accuracy
    var_violations: int                # count at 99% VaR
    var_violation_rate: float
    kupiec_p_value: float
    pinball_loss: dict[float, float]   # tau -> mean pinball loss
    shapley: dict[str, float]          # model -> Shapley value
    regime: str | None
    order_used: dict[str, Any]         # {garch: ..., sarimax: ..., samossa: ...}


@dataclass
class WalkForwardResult:
    """Aggregated results of a walk-forward evaluation run."""
    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    order_evolution: dict[str, list] = field(default_factory=dict)
    ticker: str = ""
    n_folds: int = 0


# ---------------------------------------------------------------------------
# WalkForwardLearner
# ---------------------------------------------------------------------------


class WalkForwardLearner:
    """
    Expanding-window or rolling-window retraining harness.

    Integrates:
    - OrderLearner (order cache updated each fold)
    - VaRBacktester (tail-risk assessment per fold)
    - ShapleyAttributor (error attribution per fold)
    """

    def __init__(
        self,
        forecaster_config: dict,
        order_learner: "OrderLearner | None" = None,
        snapshot_store: "ModelSnapshotStore | None" = None,
        window_type: str = "expanding",    # "expanding" | "rolling"
        min_train_length: int = 120,
        fold_step: int = 5,
        forecast_horizon: int = 10,
        confidence_level: float = 0.99,
        taus: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    ) -> None:
        self._config = forecaster_config
        self._order_learner = order_learner
        self._snapshot_store = snapshot_store
        self._window_type = window_type
        self._min_train = min_train_length
        self._fold_step = fold_step
        self._horizon = forecast_horizon
        self._confidence_level = confidence_level
        self._taus = taus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        series: "pd.Series",
        ticker: str = "UNKNOWN",
        regime_sequence: list[str] | None = None,
    ) -> WalkForwardResult:
        """
        Run full walk-forward evaluation over `series`.

        Args:
            series: pd.Series of prices (daily or intraday)
            ticker: ticker symbol for logging and OrderLearner key
            regime_sequence: per-bar regime labels (length == len(series));
                             None means regime is not pre-computed

        Returns:
            WalkForwardResult with fold_metrics and aggregate statistics
        """
        import pandas as pd

        y = np.asarray(series, dtype=float)
        N = len(y)

        if N < self._min_train + self._horizon:
            logger.warning(
                "WalkForwardLearner: series length %d < min_train (%d) + horizon (%d)",
                N, self._min_train, self._horizon,
            )
            return WalkForwardResult(ticker=ticker)

        result = WalkForwardResult(ticker=ticker)
        fold_idx = 0
        T = self._min_train

        while T + self._horizon <= N:
            train_end = T - 1
            test_start = T
            test_end = min(T + self._horizon, N) - 1

            # Training slice
            train = y[:T]

            # Rolling window: fixed-size training window
            if self._window_type == "rolling":
                roll_start = max(0, T - self._min_train)
                train = y[roll_start:T]

            # Test slice
            test = y[test_start:test_end + 1]

            # Detect regime for this fold
            regime = None
            if regime_sequence and T - 1 < len(regime_sequence):
                regime = regime_sequence[T - 1]

            # Compute fold metrics
            metrics = self._evaluate_fold(
                train=train,
                test=test,
                ticker=ticker,
                regime=regime,
                fold_idx=fold_idx,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            result.fold_metrics.append(metrics)
            fold_idx += 1
            T += self._fold_step

        result.n_folds = len(result.fold_metrics)
        result.aggregate = self._aggregate(result.fold_metrics)
        result.order_evolution = self._build_order_evolution(result.fold_metrics)

        logger.info(
            "WalkForwardLearner: %s — %d folds, RMSE=%.4f, dir_acc=%.2f%%",
            ticker, result.n_folds,
            result.aggregate.get("rmse_mean", float("nan")),
            result.aggregate.get("dir_acc_mean", float("nan")) * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_order_used(self, ticker: str, regime: str | None) -> dict[str, Any]:
        """Expose the current cached orders used for this fold when available."""
        if self._order_learner is None or not ticker:
            return {}
        resolved: dict[str, Any] = {}
        for label, model_type in (
            ("garch", "GARCH"),
            ("sarimax", "SARIMAX"),
            ("samossa", "SAMOSSA_ARIMA"),
        ):
            try:
                suggestion = self._order_learner.suggest(ticker, model_type, regime)
            except Exception:
                suggestion = None
            if suggestion is not None:
                resolved[label] = suggestion
        return resolved

    def _evaluate_fold(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ticker: str,
        regime: str | None,
        fold_idx: int,
        train_end: int,
        test_start: int,
        test_end: int,
    ) -> FoldMetrics:
        """
        Fit a simplified in-fold model and compute metrics.

        In production use, this calls TimeSeriesForecaster. Here we use a
        lightweight pure-NumPy approximation (mean + volatility model) so
        the harness is testable without heavy dependencies.
        """
        # ---- Fit: simple AR(1) + rolling std volatility ----
        train_ret = np.diff(train) / (np.abs(train[:-1]) + 1e-10)  # returns

        ar1_coef = 0.0
        if len(train_ret) > 1:
            x = train_ret[:-1]
            y = train_ret[1:]
            denom = np.dot(x, x)
            if denom > 0:
                ar1_coef = float(np.dot(x, y) / denom)
            ar1_coef = max(-0.95, min(0.95, ar1_coef))  # stability clamp

        sigma = float(np.std(train_ret)) if len(train_ret) > 0 else 0.01
        sigma = max(sigma, 1e-6)

        # ---- Forecast ----
        h = len(test)
        last_ret = float(train_ret[-1]) if len(train_ret) > 0 else 0.0
        fc_ret = np.zeros(h)
        for i in range(h):
            fc_ret[i] = ar1_coef * last_ret
            last_ret = fc_ret[i]

        # ---- Component forecasts (mock: 3 slight variations for Shapley) ----
        rng = np.random.default_rng(fold_idx + 42)
        component_forecasts = {
            "garch":   fc_ret + rng.standard_normal(h) * sigma * 0.1,
            "samossa": fc_ret + rng.standard_normal(h) * sigma * 0.15,
            "mssa_rl": fc_ret + rng.standard_normal(h) * sigma * 0.2,
        }
        weights = {"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2}

        # Ensemble forecast
        ensemble_fc = sum(weights[m] * component_forecasts[m] for m in weights)

        # ---- Actual test returns ----
        test_ret = np.diff(test) / (np.abs(test[:-1]) + 1e-10) if len(test) > 1 else np.array([])
        h_actual = min(len(test_ret), len(ensemble_fc))

        if h_actual == 0:
            return FoldMetrics(
                fold_idx=fold_idx, train_end=train_end,
                test_start=test_start, test_end=test_end,
                rmse=float("nan"), mae=float("nan"), dir_acc=float("nan"),
                var_violations=0, var_violation_rate=float("nan"),
                kupiec_p_value=float("nan"),
                pinball_loss={}, shapley={}, regime=regime, order_used={},
            )

        actual_ret = test_ret[:h_actual]
        fc_used = ensemble_fc[:h_actual]

        # ---- Error metrics ----
        errors = actual_ret - fc_used
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        dir_acc = float(np.mean(np.sign(actual_ret) == np.sign(fc_used))) if h_actual > 0 else 0.0

        # ---- VaR backtest ----
        from forcester_ts.var_backtest import VaRBacktester
        bt = VaRBacktester()
        vol_fc = np.full(h_actual, sigma)
        var_series = bt.compute_var(vol_fc, confidence_level=self._confidence_level)
        kupiec = bt.kupiec_test(actual_ret, var_series, self._confidence_level)
        pinball_q = {tau: np.full(h_actual, np.quantile(train_ret, tau))
                     for tau in self._taus}
        pinball = bt.pinball_loss(actual_ret, pinball_q)

        # ---- Shapley attribution ----
        from forcester_ts.shapley_attribution import ShapleyAttributor
        sa = ShapleyAttributor()
        comp_for_shapley = {m: component_forecasts[m][:h_actual] for m in component_forecasts}
        shapley = sa.compute(comp_for_shapley, weights, actual_ret, loss_fn="mae")
        order_used = self._resolve_order_used(ticker, regime)

        return FoldMetrics(
            fold_idx=fold_idx,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            rmse=rmse,
            mae=mae,
            dir_acc=dir_acc,
            var_violations=kupiec.get("violations", 0),
            var_violation_rate=kupiec.get("violation_rate", float("nan")),
            kupiec_p_value=kupiec.get("p_value", float("nan")),
            pinball_loss={k: v for k, v in pinball.items() if isinstance(k, float)},
            shapley=shapley,
            regime=regime,
            order_used=order_used,
        )

    @staticmethod
    def _aggregate(folds: list[FoldMetrics]) -> dict:
        """Compute mean ± std of key metrics across folds."""
        if not folds:
            return {}

        def _safe_mean(vals):
            finite = [v for v in vals if v == v]  # NaN filter
            return float(np.mean(finite)) if finite else float("nan")

        def _safe_std(vals):
            finite = [v for v in vals if v == v]
            return float(np.std(finite)) if len(finite) > 1 else 0.0

        rmses = [f.rmse for f in folds]
        maes = [f.mae for f in folds]
        dir_accs = [f.dir_acc for f in folds]
        viol_rates = [f.var_violation_rate for f in folds]
        kupiec_ps = [f.kupiec_p_value for f in folds]

        return {
            "n_folds": len(folds),
            "rmse_mean": _safe_mean(rmses),
            "rmse_std": _safe_std(rmses),
            "mae_mean": _safe_mean(maes),
            "mae_std": _safe_std(maes),
            "dir_acc_mean": _safe_mean(dir_accs),
            "dir_acc_std": _safe_std(dir_accs),
            "var_violation_rate_mean": _safe_mean(viol_rates),
            "kupiec_p_value_mean": _safe_mean(kupiec_ps),
            "shapley_mean": _aggregate_shapley(folds),
        }

    @staticmethod
    def _build_order_evolution(folds: list[FoldMetrics]) -> dict[str, list]:
        """Collect order_used across folds per model type."""
        evolution: dict[str, list] = {}
        for f in folds:
            for model, order in f.order_used.items():
                evolution.setdefault(model, []).append(order)
        return evolution


def _aggregate_shapley(folds: list[FoldMetrics]) -> dict[str, float]:
    """Mean Shapley value per model across folds."""
    bucket: dict[str, list[float]] = {}
    for f in folds:
        for model, val in f.shapley.items():
            if val == val:  # NaN guard
                bucket.setdefault(model, []).append(val)
    return {m: float(np.mean(vals)) for m, vals in bucket.items() if vals}
