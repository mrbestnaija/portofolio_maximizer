from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forcester_ts.metrics import (
    compute_regression_metrics,
    directional_accuracy,
    rmse,
    smape,
    tracking_error,
)


def _make_series(vals):
    start = datetime(2024, 1, 1)
    idx = [start + timedelta(days=i) for i in range(len(vals))]
    return pd.Series(vals, index=idx)


def test_rmse_matches_manual_computation():
    actual = _make_series([1.0, 2.0, 3.0, 4.0])
    predicted = _make_series([1.5, 1.0, 2.0, 5.0])

    metric = rmse(actual, predicted)
    assert metric is not None

    a = actual.to_numpy()
    p = predicted.to_numpy()
    expected = float(np.sqrt(np.mean((p - a) ** 2)))
    assert metric == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_smape_respects_bounds_and_symmetry():
    actual = _make_series([100.0, 110.0, 90.0])
    predicted = _make_series([102.0, 112.0, 88.0])

    s1 = smape(actual, predicted)
    s2 = smape(predicted, actual)
    assert s1 is not None and s2 is not None

    # Symmetric by construction.
    assert s1 == pytest.approx(s2, rel=1e-9, abs=1e-9)
    # Bounded in [0, 2].
    assert 0.0 <= s1 <= 2.0


def test_tracking_error_as_residual_std():
    actual = _make_series([10.0, 11.0, 12.0, 13.0])
    predicted = _make_series([9.5, 11.5, 11.0, 14.0])

    metric = tracking_error(actual, predicted)
    assert metric is not None

    residual = predicted.to_numpy() - actual.to_numpy()
    expected = float(np.std(residual, ddof=0))
    assert metric == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_directional_accuracy_counts_matching_signs():
    actual = _make_series([100, 102, 101, 105, 104])
    predicted = _make_series([100, 103, 100, 106, 103])

    da = directional_accuracy(actual, predicted)
    assert da is not None

    a = actual.to_numpy()
    p = predicted.to_numpy()
    a_diff = np.diff(a)
    p_diff = np.diff(p)
    expected = float(np.mean(np.sign(a_diff) == np.sign(p_diff)))
    assert da == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_compute_regression_metrics_returns_consistent_bundle():
    actual = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = _make_series([1.1, 1.9, 3.2, 3.8, 4.9])

    metrics = compute_regression_metrics(actual, predicted)
    assert metrics is not None

    # All primary metrics present and non-negative.
    for key in ("rmse", "smape", "tracking_error"):
        assert key in metrics
        assert metrics[key] >= 0.0
    assert "directional_accuracy" in metrics
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0

    # n_observations matches aligned non-NaN length.
    aligned_len = (
        pd.concat([actual, predicted], axis=1, join="inner")
        .dropna()
        .shape[0]
    )
    assert metrics["n_observations"] == aligned_len
