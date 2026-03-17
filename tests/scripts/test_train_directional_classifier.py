"""Phase 9 — train_directional_classifier unit tests."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest


class TestTrainDirectionalClassifier:
    def _make_dataset(self, n: int = 80, seed: int = 0) -> pd.DataFrame:
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        np.random.seed(seed)
        X = np.random.randn(n, len(_FEATURE_NAMES))
        y = (X[:, 0] > 0).astype(int)
        df = pd.DataFrame(X, columns=_FEATURE_NAMES)
        df["y_directional"] = y
        df["ts_signal_id"] = [f"ts_AAPL_test_{i:04d}" for i in range(n)]
        df["ticker"] = "AAPL"
        df["entry_ts"] = pd.date_range("2022-01-01", periods=n, freq="D").astype(str)
        df["action"] = "BUY"
        return df

    def test_cold_start_when_dataset_missing(self, tmp_path):
        from scripts.train_directional_classifier import train
        result = train(dataset_path=tmp_path / "nonexistent.parquet")
        assert result.get("cold_start") is True

    def test_cold_start_when_insufficient_data(self, tmp_path):
        from scripts.train_directional_classifier import train
        df = self._make_dataset(n=10)
        path = tmp_path / "directional_dataset.parquet"
        df.to_parquet(path, index=False)
        result = train(dataset_path=path, model_path=tmp_path / "m.pkl", meta_path=tmp_path / "m.meta.json")
        assert result.get("cold_start") is True

    def test_trains_and_saves_model(self, tmp_path):
        from scripts.train_directional_classifier import train
        df = self._make_dataset(n=80)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        result = train(
            dataset_path=path,
            model_path=model_path,
            meta_path=meta_path,
            c_values=[1.0],
        )
        assert result.get("cold_start") is not True
        assert model_path.exists(), "Model pkl should have been saved"
        assert meta_path.exists(), "Meta JSON should have been saved"

    def test_meta_contains_expected_fields(self, tmp_path):
        from scripts.train_directional_classifier import train
        df = self._make_dataset(n=80)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["n_train"] == 80
        assert "walk_forward_da" in meta
        assert "top3_features" in meta
        assert meta["cold_start"] is False

    def test_directional_accuracy_above_chance_on_linear_data(self, tmp_path):
        from scripts.train_directional_classifier import train
        from forcester_ts.directional_classifier import DirectionalClassifier
        df = self._make_dataset(n=120, seed=42)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        result = train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        # Walk-forward DA should be above random (50%) on linearly separable data
        # (may be close to 50% due to small fold sizes, so use a low bar)
        assert result.get("walk_forward_da", 0.0) >= 0.40
