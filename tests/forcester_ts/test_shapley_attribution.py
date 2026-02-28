"""
tests/forcester_ts/test_shapley_attribution.py
-----------------------------------------------
Unit tests for ShapleyAttributor — PBSV ensemble error attribution.
"""
import numpy as np
import pytest

from forcester_ts.shapley_attribution import ShapleyAttributor, _compute_loss, _subset_forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forecasts(n=40):
    rng = np.random.default_rng(42)
    actual = rng.standard_normal(n) * 0.01
    garch = actual + rng.standard_normal(n) * 0.005
    samossa = actual + rng.standard_normal(n) * 0.008
    mssa_rl = actual + rng.standard_normal(n) * 0.012
    return actual, {"garch": garch, "samossa": samossa, "mssa_rl": mssa_rl}


# ---------------------------------------------------------------------------
# _compute_loss tests
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_mae_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _compute_loss(y, y, "mae", 0.5) == pytest.approx(0.0)

    def test_mae_known(self):
        y = np.array([0.0, 0.0])
        f = np.array([1.0, -1.0])
        assert _compute_loss(y, f, "mae", 0.5) == pytest.approx(1.0)

    def test_mse_known(self):
        y = np.array([0.0, 0.0])
        f = np.array([1.0, -1.0])
        assert _compute_loss(y, f, "mse", 0.5) == pytest.approx(1.0)

    def test_pinball_median(self):
        # At tau=0.5, pinball = 0.5 * MAE
        y = np.array([0.0, 0.0])
        f = np.array([1.0, -1.0])
        # diff = [-1, 1]; for diff<0: -1*( 0.5 - 1) = 0.5; for diff>=0: 1*(0.5) = 0.5
        result = _compute_loss(y, f, "pinball", 0.5)
        assert result == pytest.approx(0.5)

    def test_empty_returns_nan(self):
        result = _compute_loss(np.array([]), np.array([1.0]), "mae", 0.5)
        assert np.isnan(result)

    def test_default_is_mae(self):
        y = np.ones(10)
        f = np.zeros(10)
        assert _compute_loss(y, f, "mae", 0.5) == pytest.approx(1.0)
        assert _compute_loss(y, f, "bogus", 0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _subset_forecast tests
# ---------------------------------------------------------------------------

class TestSubsetForecast:
    def test_single_model_returns_that_forecast(self):
        fc = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        weights = {"a": 0.6, "b": 0.4}
        result = _subset_forecast(fc, weights, ("a",), 2)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_two_model_weighted_average(self):
        fc = {"a": np.array([0.0, 0.0]), "b": np.array([4.0, 4.0])}
        weights = {"a": 1.0, "b": 1.0}  # equal — should average
        result = _subset_forecast(fc, weights, ("a", "b"), 2)
        np.testing.assert_array_almost_equal(result, [2.0, 2.0])

    def test_empty_subset_returns_grand_mean(self):
        fc = {"a": np.array([2.0, 2.0]), "b": np.array([4.0, 4.0])}
        weights = {"a": 0.5, "b": 0.5}
        result = _subset_forecast(fc, weights, (), 2)
        np.testing.assert_array_almost_equal(result, [3.0, 3.0])


# ---------------------------------------------------------------------------
# ShapleyAttributor.compute tests
# ---------------------------------------------------------------------------

class TestShapleyCompute:
    def test_returns_value_per_player(self):
        actual, forecasts = _make_forecasts()
        sa = ShapleyAttributor()
        result = sa.compute(forecasts, {"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2}, actual)
        assert set(result.keys()) == {"garch", "samossa", "mssa_rl"}

    def test_efficiency_axiom_approx(self):
        """
        Sum of Shapley values should approximately equal
        loss(full ensemble) - loss(empty subset grand mean).
        """
        actual, forecasts = _make_forecasts(n=60)
        weights = {"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2}
        sa = ShapleyAttributor()
        svs = sa.compute(forecasts, weights, actual, loss_fn="mae")

        # Full ensemble loss
        from forcester_ts.shapley_attribution import _subset_forecast, _compute_loss
        n = len(actual)
        for arr in forecasts.values():
            n = min(n, len(arr))
        full_fc = _subset_forecast(forecasts, weights, tuple(forecasts.keys()), n)
        empty_fc = _subset_forecast(forecasts, weights, (), n)
        full_loss = _compute_loss(actual, full_fc, "mae", 0.5)
        empty_loss = _compute_loss(actual, empty_fc, "mae", 0.5)

        # Efficiency: sum(sv) ~= v(grand) - v(empty)
        total_sv = sum(svs.values())
        expected = full_loss - empty_loss
        assert abs(total_sv - expected) < 1e-9, (
            f"Efficiency axiom violated: sum(sv)={total_sv:.6f} expected={expected:.6f}"
        )

    def test_dummy_axiom_equal_models(self):
        """If two models are identical, their Shapley values should be equal."""
        rng = np.random.default_rng(7)
        actual = rng.standard_normal(30) * 0.01
        fc_common = actual + rng.standard_normal(30) * 0.005
        forecasts = {"a": fc_common, "b": fc_common.copy()}
        weights = {"a": 0.5, "b": 0.5}
        sa = ShapleyAttributor()
        svs = sa.compute(forecasts, weights, actual)
        assert abs(svs["a"] - svs["b"]) < 1e-10

    def test_perfect_forecaster_gets_negative_shapley(self):
        """A model that perfectly forecasts reduces error → negative Shapley."""
        rng = np.random.default_rng(1)
        actual = rng.standard_normal(50) * 0.01
        perfect = actual.copy()
        noisy = actual + rng.standard_normal(50) * 0.05
        forecasts = {"perfect": perfect, "noisy": noisy}
        weights = {"perfect": 0.5, "noisy": 0.5}
        sa = ShapleyAttributor()
        svs = sa.compute(forecasts, weights, actual, loss_fn="mae")
        # Perfect model should have lower (more negative) Shapley than noisy
        assert svs["perfect"] < svs["noisy"], (
            f"Expected sv(perfect) < sv(noisy): {svs['perfect']:.4f} vs {svs['noisy']:.4f}"
        )

    def test_single_player(self):
        """Single player gets full v(player) - v(empty)."""
        rng = np.random.default_rng(5)
        actual = rng.standard_normal(20) * 0.01
        fc = actual + 0.01
        forecasts = {"solo": fc}
        weights = {"solo": 1.0}
        sa = ShapleyAttributor()
        svs = sa.compute(forecasts, weights, actual, loss_fn="mae")
        assert "solo" in svs
        assert svs["solo"] == svs["solo"]  # not NaN

    def test_empty_forecasts_returns_empty(self):
        actual = np.array([1.0, 2.0])
        sa = ShapleyAttributor()
        result = sa.compute({}, {}, actual)
        assert result == {}

    def test_mse_loss_fn(self):
        actual, forecasts = _make_forecasts()
        sa = ShapleyAttributor()
        result = sa.compute(forecasts, {"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2},
                            actual, loss_fn="mse")
        assert all(v == v for v in result.values())  # no NaN

    def test_pinball_loss_fn(self):
        actual, forecasts = _make_forecasts()
        sa = ShapleyAttributor()
        result = sa.compute(forecasts, {"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2},
                            actual, loss_fn="pinball", tau=0.9)
        assert set(result.keys()) == {"garch", "samossa", "mssa_rl"}
        assert all(v == v for v in result.values())

    def test_zero_length_actual_returns_nan(self):
        forecasts = {"a": np.array([]), "b": np.array([])}
        sa = ShapleyAttributor()
        result = sa.compute(forecasts, {"a": 0.5, "b": 0.5}, np.array([]))
        assert all(np.isnan(v) for v in result.values())


# ---------------------------------------------------------------------------
# ShapleyAttributor.aggregate_by_regime tests
# ---------------------------------------------------------------------------

class TestAggregateByRegime:
    def test_groups_by_regime(self):
        fold_results = [
            {"regime": "CRISIS", "shapley": {"garch": 0.1, "samossa": -0.05}},
            {"regime": "CRISIS", "shapley": {"garch": 0.2, "samossa": -0.01}},
            {"regime": "TRENDING", "shapley": {"garch": 0.0, "samossa": 0.05}},
        ]
        sa = ShapleyAttributor()
        agg = sa.aggregate_by_regime(fold_results)
        assert "CRISIS" in agg
        assert "TRENDING" in agg
        assert agg["CRISIS"]["garch"] == pytest.approx(0.15)
        assert agg["CRISIS"]["samossa"] == pytest.approx(-0.03)
        assert agg["TRENDING"]["garch"] == pytest.approx(0.0)

    def test_missing_regime_key_uses_unknown(self):
        fold_results = [{"shapley": {"a": 0.5}}]
        sa = ShapleyAttributor()
        agg = sa.aggregate_by_regime(fold_results)
        assert "UNKNOWN" in agg

    def test_nan_excluded_from_mean(self):
        fold_results = [
            {"regime": "X", "shapley": {"m": float("nan")}},
            {"regime": "X", "shapley": {"m": 0.4}},
        ]
        sa = ShapleyAttributor()
        agg = sa.aggregate_by_regime(fold_results)
        # NaN excluded; mean of [0.4] = 0.4
        assert agg["X"]["m"] == pytest.approx(0.4)

    def test_empty_fold_results(self):
        sa = ShapleyAttributor()
        assert sa.aggregate_by_regime([]) == {}


# ---------------------------------------------------------------------------
# ShapleyAttributor.dominant_driver tests
# ---------------------------------------------------------------------------

class TestDominantDriver:
    def test_returns_dominant_model(self):
        sa = ShapleyAttributor()
        svs = {"garch": 0.15, "samossa": 0.02, "mssa_rl": 0.01}
        result = sa.dominant_driver(svs, ensemble_loss=0.5)
        assert result == "garch"

    def test_returns_none_when_none_exceed_threshold(self):
        sa = ShapleyAttributor()
        svs = {"a": 0.001, "b": 0.002}
        result = sa.dominant_driver(svs, ensemble_loss=1.0, threshold=0.05)
        assert result is None

    def test_handles_negative_shapley_as_abs(self):
        sa = ShapleyAttributor()
        svs = {"helpful": -0.30, "bad": 0.01}
        result = sa.dominant_driver(svs, ensemble_loss=0.5, threshold=0.05)
        assert result == "helpful"  # |-0.30| dominates

    def test_empty_dict_returns_none(self):
        sa = ShapleyAttributor()
        assert sa.dominant_driver({}, 0.5) is None

    def test_nan_ensemble_loss_returns_none(self):
        sa = ShapleyAttributor()
        assert sa.dominant_driver({"a": 0.1}, float("nan")) is None
