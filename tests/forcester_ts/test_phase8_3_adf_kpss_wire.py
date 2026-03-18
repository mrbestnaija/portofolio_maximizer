"""
Phase 8.3 — ADF/KPSS conflict-resolution wire tests.

Covers:
  1. SARIMAXForecaster._select_best_order: forced_d=None leaves ADF in charge.
  2. SARIMAXForecaster._select_best_order: forced_d=1 overrides even when ADF says d=0.
  3. SARIMAXForecaster._select_best_order: forced_d=0 keeps d=0 regardless of ADF.
  4. SARIMAXForecaster.fit: forced_d threaded correctly to _select_best_order.
  5. TimeSeriesForecaster.fit: non_stationary verdict → forced_d=1 passed to sarimax.fit.
  6. TimeSeriesForecaster.fit: stationary verdict → forced_d=None (no override).
  7. TimeSeriesForecaster.fit: conflicted verdict → forced_d=1 (conservative).
  8. TimeSeriesForecaster.fit: missing KPSS → forced_d=None (ADF-only path unchanged).
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch, call
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from forcester_ts.sarimax import SARIMAXForecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stationary_series(n: int = 120) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(rng.standard_normal(n), index=idx, name="price")


def _make_nonstationary_series(n: int = 120) -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    data = np.cumsum(rng.standard_normal(n)) + 100.0
    return pd.Series(data, index=idx, name="price")


# ---------------------------------------------------------------------------
# 1-3: _select_best_order forced_d parameter
# ---------------------------------------------------------------------------

class TestSelectBestOrderForcedD:
    """Unit-test the forced_d parameter in isolation (no real SARIMAX fitting)."""

    def _make_forecaster(self) -> SARIMAXForecaster:
        f = SARIMAXForecaster(
            max_p=1, max_q=1, auto_select=True, enforce_stationarity=True
        )
        return f

    def test_forced_d_none_defers_to_adf(self):
        """When forced_d=None, d comes from _test_stationarity (ADF)."""
        f = self._make_forecaster()
        series = _make_stationary_series()

        captured_orders: list = []

        original_aic = object()

        def fake_select(data, exog=None, forced_d=None):
            # Record what forced_d value arrived and return fixed answer
            captured_orders.append(forced_d)
            return (1, 0, 0), (0, 0, 0, 0)

        f._select_best_order = fake_select  # type: ignore[method-assign]

        # Minimal stubs so fit() doesn't blow up past _select_best_order
        with (
            patch.object(f, "_prepare_series", return_value=series),
            patch.object(f, "_scale_series", return_value=(series, 1.0)),
            patch.object(f, "_align_exogenous", return_value=None),
            patch.object(f, "_fit_model_instance", side_effect=StopIteration),
        ):
            try:
                f.fit(series, forced_d=None)
            except (StopIteration, Exception):
                pass

        assert captured_orders, "forced_d should have been forwarded"
        assert captured_orders[0] is None, f"Expected None, got {captured_orders[0]}"

    def test_forced_d_1_overrides_adf(self):
        """When forced_d=1, _select_best_order receives 1 regardless of stationarity."""
        f = self._make_forecaster()
        series = _make_stationary_series()

        captured: list = []

        def fake_select(data, exog=None, forced_d=None):
            captured.append(forced_d)
            return (1, 0, 0), (0, 0, 0, 0)

        f._select_best_order = fake_select  # type: ignore[method-assign]

        with (
            patch.object(f, "_prepare_series", return_value=series),
            patch.object(f, "_scale_series", return_value=(series, 1.0)),
            patch.object(f, "_align_exogenous", return_value=None),
            patch.object(f, "_fit_model_instance", side_effect=StopIteration),
        ):
            try:
                f.fit(series, forced_d=1)
            except (StopIteration, Exception):
                pass

        assert captured, "forced_d should have been forwarded"
        assert captured[0] == 1, f"Expected 1, got {captured[0]}"

    def test_forced_d_zero_keeps_d_zero(self):
        """forced_d=0 pins d to 0 even on a random-walk series."""
        f = self._make_forecaster()
        series = _make_nonstationary_series()

        # Directly call _select_best_order with a fake _test_stationarity that says d=1
        with patch.object(f, "_test_stationarity", return_value=(False, 1)):
            orders_d0, _ = f._select_best_order(series, forced_d=0)

        # All candidate orders should have d=0
        assert orders_d0[1] == 0, f"Expected d=0, got best_order={orders_d0}"

    def test_forced_d_1_overrides_stationary_series(self):
        """forced_d=1 overrides even when ADF says the series is stationary (d=0)."""
        f = self._make_forecaster()
        series = _make_stationary_series()

        with patch.object(f, "_test_stationarity", return_value=(True, 0)):
            order, _ = f._select_best_order(series, forced_d=1)

        assert order[1] == 1, f"Expected d=1 from forced_d override, got {order[1]}"


# ---------------------------------------------------------------------------
# 5-8: forecaster.py wire — _series_diagnostics → forced_d
# ---------------------------------------------------------------------------

class TestForecasterForcedDWire:
    """
    Integration-style tests: patch _capture_series_diagnostics to return specific
    stationarity_verdict values and verify forced_d passed to sarimax.fit().
    """

    def _make_ts_forecaster(self):
        """Build a minimal TimeSeriesForecaster with SARIMAX enabled."""
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
        cfg = TimeSeriesForecasterConfig(
            sarimax_enabled=True,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
            ensemble_enabled=False,
        )
        return TimeSeriesForecaster(config=cfg)

    def _series(self) -> pd.Series:
        return _make_nonstationary_series(200)

    def _run_with_verdict(self, verdict: str, force_difference: bool):
        """
        Run forecaster.fit with a patched stationarity verdict and capture
        the forced_d argument passed to SARIMAXForecaster.fit.
        """
        tsf = self._make_ts_forecaster()
        series = self._series()

        diagnostics = {
            "stationarity_verdict": verdict,
            "force_difference": force_difference,
            "adf_pvalue": 0.01 if not force_difference else 0.5,
            "kpss_pvalue": 0.06 if not force_difference else 0.01,
        }

        captured_forced_d: list = []

        original_sarimax_fit = SARIMAXForecaster.fit

        def fake_sarimax_fit(self_inner, series, exogenous=None, order_learner=None,
                             ticker="", regime=None, forced_d=None):
            captured_forced_d.append(forced_d)
            raise StopIteration("short-circuit")

        with (
            patch.object(tsf, "_capture_series_diagnostics", return_value=diagnostics),
            patch.object(tsf._instrumentation, "record_artifact"),
            patch.object(tsf._instrumentation, "record_series_snapshot"),
            patch.object(tsf._instrumentation, "set_dataset_metadata"),
            patch.object(tsf._instrumentation, "reset"),
            patch("forcester_ts.forecaster.SARIMAXForecaster.fit", fake_sarimax_fit),
        ):
            try:
                tsf.fit(series)
            except (StopIteration, Exception):
                pass

        return captured_forced_d

    def test_non_stationary_verdict_passes_forced_d_1(self):
        """non_stationary → force_difference=True → forced_d=1."""
        captured = self._run_with_verdict("non_stationary", force_difference=True)
        assert captured, "sarimax.fit was never called"
        assert captured[0] == 1, f"Expected forced_d=1, got {captured[0]}"

    def test_stationary_verdict_passes_forced_d_none(self):
        """stationary → force_difference=False → forced_d=None."""
        captured = self._run_with_verdict("stationary", force_difference=False)
        assert captured, "sarimax.fit was never called"
        assert captured[0] is None, f"Expected forced_d=None, got {captured[0]}"

    def test_conflicted_verdict_passes_forced_d_1(self):
        """conflicted → force_difference=True → forced_d=1 (conservative)."""
        captured = self._run_with_verdict("conflicted", force_difference=True)
        assert captured, "sarimax.fit was never called"
        assert captured[0] == 1, f"Expected forced_d=1 for conflicted, got {captured[0]}"

    def test_no_kpss_no_verdict_passes_forced_d_none(self):
        """When KPSS unavailable, force_difference absent → forced_d=None."""
        tsf = self._make_ts_forecaster()
        series = self._series()

        # Diagnostics without stationarity_verdict (ADF-only run)
        diagnostics = {"adf_pvalue": 0.03}

        captured: list = []

        def fake_sarimax_fit(self_inner, series, exogenous=None, order_learner=None,
                             ticker="", regime=None, forced_d=None):
            captured.append(forced_d)
            raise StopIteration("short-circuit")

        with (
            patch.object(tsf, "_capture_series_diagnostics", return_value=diagnostics),
            patch.object(tsf._instrumentation, "record_artifact"),
            patch.object(tsf._instrumentation, "record_series_snapshot"),
            patch.object(tsf._instrumentation, "set_dataset_metadata"),
            patch.object(tsf._instrumentation, "reset"),
            patch("forcester_ts.forecaster.SARIMAXForecaster.fit", fake_sarimax_fit),
        ):
            try:
                tsf.fit(series)
            except (StopIteration, Exception):
                pass

        assert captured, "sarimax.fit was never called"
        assert captured[0] is None, f"Expected forced_d=None when no verdict, got {captured[0]}"
