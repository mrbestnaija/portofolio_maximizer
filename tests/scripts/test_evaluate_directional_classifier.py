"""Phase 9 — evaluate_directional_classifier unit tests."""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import pytest


class TestEvaluateDirectionalClassifier:
    def _make_dataset(self, n: int = 120, seed: int = 0) -> pd.DataFrame:
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        np.random.seed(seed)
        X = np.random.randn(n, len(_FEATURE_NAMES))
        y = (X[:, 0] > 0).astype(int)
        df = pd.DataFrame(X, columns=_FEATURE_NAMES)
        df["y_directional"] = y
        df["ts_signal_id"] = [f"ts_AAPL_eval_{i:04d}" for i in range(n)]
        df["ticker"] = "AAPL"
        df["entry_ts"] = pd.date_range("2022-01-01", periods=n, freq="D").astype(str)
        df["action"] = "BUY"
        df["label_source"] = "price_forward"
        return df

    def test_cold_start_when_dataset_missing(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        result = evaluate(dataset_path=tmp_path / "nonexistent.parquet", write_report=False)
        assert result.get("error") == "dataset_not_found"

    def test_error_when_dataset_unreadable(self, tmp_path):
        """Corrupt parquet returns dataset_unreadable error dict, not an exception."""
        from scripts.evaluate_directional_classifier import evaluate
        bad = tmp_path / "directional_dataset.parquet"
        bad.write_bytes(b"not a parquet file")
        result = evaluate(dataset_path=bad, write_report=False)
        assert result.get("error") == "dataset_unreadable"

    def test_cold_start_when_insufficient_data(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        df = self._make_dataset(n=10)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        result = evaluate(dataset_path=path, min_n=60, write_report=False)
        assert result.get("cold_start") is True

    def test_returns_walk_forward_da(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        df = self._make_dataset(n=120)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        result = evaluate(dataset_path=path, min_n=60, write_report=False)
        assert "walk_forward" in result
        wf = result["walk_forward"]
        assert "mean_da" in wf
        assert wf["n_folds"] >= 1
        if wf["mean_da"] is not None:
            assert 0.0 <= wf["mean_da"] <= 1.0

    def test_returns_ece_decomposition(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        df = self._make_dataset(n=120)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        result = evaluate(dataset_path=path, min_n=60, write_report=False)
        ece = result.get("ece", {})
        assert "ece" in ece
        assert "bins" in ece
        # ECE must be in [0, 1] when not None
        if ece["ece"] is not None:
            assert 0.0 <= ece["ece"] <= 1.0

    def test_returns_win_rate_counterfactual(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        df = self._make_dataset(n=120)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        result = evaluate(dataset_path=path, min_n=60, write_report=False)
        cf = result.get("counterfactual", {})
        assert "n_total" in cf or "note" in cf  # either has data or a note

    def test_writes_report_files(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        from unittest.mock import patch
        df = self._make_dataset(n=120)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        eval_out = tmp_path / "directional_eval_latest.json"
        report_out = tmp_path / "directional_eval.txt"
        with patch("scripts.evaluate_directional_classifier._EVAL_OUTPUT", eval_out), \
             patch("scripts.evaluate_directional_classifier._REPORT_OUTPUT", report_out):
            evaluate(dataset_path=path, min_n=60, write_report=True)
        assert eval_out.exists(), "eval JSON must be written"
        assert report_out.exists(), "ASCII report must be written"

    def test_eval_json_is_valid(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        from unittest.mock import patch
        df = self._make_dataset(n=120)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        eval_out = tmp_path / "directional_eval_latest.json"
        report_out = tmp_path / "directional_eval.txt"
        with patch("scripts.evaluate_directional_classifier._EVAL_OUTPUT", eval_out), \
             patch("scripts.evaluate_directional_classifier._REPORT_OUTPUT", report_out):
            evaluate(dataset_path=path, min_n=60, write_report=True)
        data = json.loads(eval_out.read_text(encoding="utf-8"))
        assert "evaluated_at" in data
        assert "n_labeled" in data
        assert "walk_forward" in data

    def test_report_is_ascii_safe(self, tmp_path):
        from scripts.evaluate_directional_classifier import evaluate
        from unittest.mock import patch
        df = self._make_dataset(n=120)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        eval_out = tmp_path / "directional_eval_latest.json"
        report_out = tmp_path / "directional_eval.txt"
        with patch("scripts.evaluate_directional_classifier._EVAL_OUTPUT", eval_out), \
             patch("scripts.evaluate_directional_classifier._REPORT_OUTPUT", report_out):
            evaluate(dataset_path=path, min_n=60, write_report=True)
        text = report_out.read_text(encoding="utf-8")
        # All characters must be ASCII-safe (ord < 128)
        non_ascii = [ch for ch in text if ord(ch) >= 128]
        assert len(non_ascii) == 0, f"Non-ASCII characters in report: {non_ascii[:10]}"


class TestECEDecomposition:
    def test_perfect_calibration(self):
        """A well-calibrated model should have ECE near 0."""
        from scripts.evaluate_directional_classifier import _ece_decomposition
        # p_pred = y_true (perfect calibration)
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        p = y.astype(float)
        result = _ece_decomposition(y, p)
        assert result["ece"] == pytest.approx(0.0, abs=1e-6)

    def test_ece_bounds(self):
        """ECE must be in [0, 1]."""
        from scripts.evaluate_directional_classifier import _ece_decomposition
        np.random.seed(42)
        y = np.random.randint(0, 2, 100)
        p = np.random.uniform(0, 1, 100)
        result = _ece_decomposition(y, p)
        assert 0.0 <= result["ece"] <= 1.0

    def test_bin_count(self):
        from scripts.evaluate_directional_classifier import _ece_decomposition
        y = np.array([1, 0] * 50)
        p = np.linspace(0, 1, 100)
        result = _ece_decomposition(y, p, n_bins=10)
        assert len(result["bins"]) == 10


class TestWinRateCounterfactual:
    def test_baseline_win_rate(self):
        from scripts.evaluate_directional_classifier import _win_rate_counterfactual
        y = np.array([1, 1, 1, 0, 0])  # 60% win rate
        p = np.array([0.6, 0.7, 0.8, 0.3, 0.4])
        result = _win_rate_counterfactual(y, p, p_up_threshold=0.55)
        assert result["baseline_win_rate"] == pytest.approx(0.6, abs=1e-4)

    def test_gate_blocks_low_confidence(self):
        from scripts.evaluate_directional_classifier import _win_rate_counterfactual
        y = np.array([1, 0, 1, 0])
        # p=[0.8,0.7] pass BUY (>=0.65); p=[0.5,0.4] blocked (neither side clears 0.65)
        # sell_mask: p <= 1-0.65=0.35 -> none of these
        p = np.array([0.8, 0.5, 0.7, 0.4])
        result = _win_rate_counterfactual(y, p, p_up_threshold=0.65, p_down_threshold=0.65)
        assert result["n_gated_buy"] == 2
        assert result["n_blocked"] == 2

    def test_all_blocked(self):
        from scripts.evaluate_directional_classifier import _win_rate_counterfactual
        y = np.array([1, 0, 1])
        # All p_pred in (0.45, 0.55) — neither threshold passed
        p = np.array([0.50, 0.50, 0.50])
        result = _win_rate_counterfactual(y, p, p_up_threshold=0.55, p_down_threshold=0.55)
        assert result["n_blocked"] == 3
        assert result["n_gated_buy"] == 0
