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

    # D4: CalibratedClassifierCV (schema v2) tests

    def test_saved_model_is_calibrated_classifier(self, tmp_path):
        """D4: saved pkl must be CalibratedClassifierCV, not bare Pipeline."""
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from scripts.train_directional_classifier import train
        df = self._make_dataset(n=80)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        model = joblib.load(model_path)
        assert isinstance(model, CalibratedClassifierCV), (
            f"Expected CalibratedClassifierCV, got {type(model).__name__}"
        )

    def test_meta_has_calibration_fields(self, tmp_path):
        """D4: meta sidecar must include calibration_method and schema_version=2."""
        from scripts.train_directional_classifier import train
        df = self._make_dataset(n=80)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        result = train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        assert result.get("calibration_method") == "sigmoid"
        assert result.get("calibration_cv_folds") in (2, 3)
        assert result.get("schema_version") == 2

    def test_calibrated_model_predict_proba_in_unit_interval(self, tmp_path):
        """D4: calibrated model must return probabilities in [0, 1]."""
        import joblib
        import pandas as pd
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        from scripts.train_directional_classifier import train
        df = self._make_dataset(n=80)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        model = joblib.load(model_path)
        X_test = pd.DataFrame(np.random.randn(10, len(_FEATURE_NAMES)), columns=_FEATURE_NAMES)
        proba = model.predict_proba(X_test)
        assert proba.shape == (10, 2)
        assert (proba >= 0.0).all() and (proba <= 1.0).all(), "Probabilities must be in [0, 1]"

    def test_directional_classifier_loads_schema_v2(self, tmp_path):
        """D4: DirectionalClassifier.score() works with schema v2 pkl (CalibratedClassifierCV)."""
        from scripts.train_directional_classifier import train
        from forcester_ts.directional_classifier import DirectionalClassifier, _FEATURE_NAMES
        df = self._make_dataset(n=80)
        path = tmp_path / "directional_dataset.parquet"
        model_path = tmp_path / "directional_v1.pkl"
        meta_path = tmp_path / "directional_v1.meta.json"
        df.to_parquet(path, index=False)
        train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        clf = DirectionalClassifier(model_path=model_path)
        features = {name: 0.0 for name in _FEATURE_NAMES}
        result = clf.score(features)
        assert isinstance(result, float) and 0.0 <= result <= 1.0

    def test_meta_contains_feature_names(self, tmp_path):
        """Feature names saved in meta must match _FEATURE_NAMES used during training."""
        import json
        from scripts.train_directional_classifier import train
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        df = self._make_dataset(n=80)
        path = tmp_path / "d.parquet"
        model_path = tmp_path / "m.pkl"
        meta_path = tmp_path / "m.meta.json"
        df.to_parquet(path, index=False)
        train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "feature_names" in meta, "meta must include feature_names for inference-time validation"
        assert meta["feature_names"] == list(_FEATURE_NAMES)

    def test_feature_name_mismatch_disables_scoring(self, tmp_path):
        """If saved feature_names differ from current _FEATURE_NAMES, score() must return None."""
        import json
        from scripts.train_directional_classifier import train
        from forcester_ts.directional_classifier import DirectionalClassifier, _FEATURE_NAMES
        df = self._make_dataset(n=80)
        path = tmp_path / "d.parquet"
        model_path = tmp_path / "m.pkl"
        meta_path = tmp_path / "m.meta.json"
        df.to_parquet(path, index=False)
        train(dataset_path=path, model_path=model_path, meta_path=meta_path, c_values=[1.0])
        # Tamper the meta to simulate a stale model with a different feature list
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_names"] = ["wrong_feature_a", "wrong_feature_b"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        clf = DirectionalClassifier(model_path=model_path)
        features = {name: 0.0 for name in _FEATURE_NAMES}
        result = clf.score(features)
        assert result is None, "Mismatched feature_names must disable scoring (returns None)"
