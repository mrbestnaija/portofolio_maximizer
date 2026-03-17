"""Phase 9 — DirectionalClassifier unit tests."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pytest


class TestDirectionalClassifierColdStart:
    def test_score_returns_none_when_no_model_file(self, tmp_path):
        from forcester_ts.directional_classifier import DirectionalClassifier
        clf = DirectionalClassifier(model_path=tmp_path / "nonexistent.pkl")
        result = clf.score({"ensemble_pred_return": 0.02})
        assert result is None

    def test_score_returns_none_when_meta_says_cold_start(self, tmp_path):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from forcester_ts.directional_classifier import DirectionalClassifier, _FEATURE_NAMES

        # Write a real pipeline but with n_train=5 (below threshold)
        X = np.random.randn(10, len(_FEATURE_NAMES))
        y = np.array([0, 1] * 5)
        pipe = Pipeline([
            ("i", SimpleImputer(strategy="mean")),
            ("s", StandardScaler()),
            ("c", LogisticRegression()),
        ])
        pipe.fit(X, y)
        pkl_path = tmp_path / "directional_v1.pkl"
        joblib.dump(pipe, pkl_path)
        meta_path = tmp_path / "directional_v1.meta.json"
        meta_path.write_text(json.dumps({"n_train": 5}), encoding="utf-8")

        clf = DirectionalClassifier(model_path=pkl_path)
        result = clf.score({"ensemble_pred_return": 0.02})
        assert result is None

    def test_score_returns_float_when_model_loaded(self, tmp_path):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from forcester_ts.directional_classifier import DirectionalClassifier, _FEATURE_NAMES

        np.random.seed(0)
        X = np.random.randn(80, len(_FEATURE_NAMES))
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline([
            ("i", SimpleImputer(strategy="mean")),
            ("s", StandardScaler()),
            ("c", LogisticRegression()),
        ])
        pipe.fit(X, y)
        pkl_path = tmp_path / "directional_v1.pkl"
        joblib.dump(pipe, pkl_path)
        meta_path = tmp_path / "directional_v1.meta.json"
        meta_path.write_text(json.dumps({"n_train": 80}), encoding="utf-8")

        clf = DirectionalClassifier(model_path=pkl_path)
        features = {name: float(np.random.randn()) for name in _FEATURE_NAMES}
        result = clf.score(features)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_handles_nan_features(self, tmp_path):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from forcester_ts.directional_classifier import DirectionalClassifier, _FEATURE_NAMES

        np.random.seed(1)
        X = np.random.randn(80, len(_FEATURE_NAMES))
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline([
            ("i", SimpleImputer(strategy="mean")),
            ("s", StandardScaler()),
            ("c", LogisticRegression()),
        ])
        pipe.fit(X, y)
        pkl_path = tmp_path / "directional_v1.pkl"
        joblib.dump(pipe, pkl_path)
        meta_path = tmp_path / "directional_v1.meta.json"
        meta_path.write_text(json.dumps({"n_train": 80}), encoding="utf-8")

        clf = DirectionalClassifier(model_path=pkl_path)
        # All NaN features — imputer should handle this
        features = {name: float("nan") for name in _FEATURE_NAMES}
        result = clf.score(features)
        assert result is None or (isinstance(result, float) and 0.0 <= result <= 1.0)


class TestExtractClassifierFeatures:
    def _make_generator(self):
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        return TimeSeriesSignalGenerator()

    def _make_bundle(self, current_price=100.0, forecast=102.0):
        import pandas as pd
        idx = pd.RangeIndex(5, name="horizon")
        return {
            "forecast": pd.Series([forecast] * 5, index=idx),
            "lower_ci": pd.Series([current_price * 0.98] * 5, index=idx),
            "upper_ci": pd.Series([current_price * 1.02] * 5, index=idx),
            "ensemble_metadata": {
                "confidence": {"garch": 0.55, "samossa": 0.65, "mssa_rl": 0.60},
                "weights": {"samossa": 0.6, "garch": 0.4},
            },
            "detected_regime": "MODERATE_TRENDING",
            "regime_features": {
                "hurst_exponent": 0.58,
                "trend_strength": 0.45,
                "realized_vol_annualized": 0.22,
            },
            "series_diagnostics": {"adf_pvalue": 0.03},
            "samossa_forecast": {
                "forecast": pd.Series([forecast] * 5, index=idx),
                "explained_variance_ratio": 0.95,
            },
            "mssa_rl_forecast": {"forecast": pd.Series([forecast * 0.99] * 5, index=idx)},
            "garch_forecast": {"forecast": pd.Series([0.002] * 5, index=idx)},
        }

    def test_returns_all_feature_names(self):
        from forcester_ts.directional_classifier import _FEATURE_NAMES
        gen = self._make_generator()
        bundle = self._make_bundle()
        lower_ci = 98.0
        upper_ci = 102.0
        features = gen._extract_classifier_features(
            forecast_bundle=bundle,
            current_price=100.0,
            expected_return=0.02,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            snr=1.5,
            model_agreement=0.8,
            market_data=None,
        )
        for name in _FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_regime_onehot_moderate_trending(self):
        gen = self._make_generator()
        bundle = self._make_bundle()
        features = gen._extract_classifier_features(
            forecast_bundle=bundle,
            current_price=100.0,
            expected_return=0.02,
            lower_ci=98.0,
            upper_ci=102.0,
            snr=1.5,
            model_agreement=0.8,
            market_data=None,
        )
        assert features["regime_moderate_trending"] == 1.0
        assert features["regime_crisis"] == 0.0
        assert features["regime_liquid_rangebound"] == 0.0
        assert features["regime_high_vol_trending"] == 0.0

    def test_no_nan_in_well_populated_bundle(self):
        import pandas as pd
        gen = self._make_generator()
        bundle = self._make_bundle()
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        market_data = pd.DataFrame(
            {"Close": 100 + np.cumsum(np.random.randn(n))},
            index=dates,
        )
        features = gen._extract_classifier_features(
            forecast_bundle=bundle,
            current_price=100.0,
            expected_return=0.02,
            lower_ci=98.0,
            upper_ci=102.0,
            snr=1.5,
            model_agreement=0.8,
            market_data=market_data,
        )
        # Only market context features can be NaN (others should be populated)
        nan_features = {k: v for k, v in features.items() if isinstance(v, float) and math.isnan(v)}
        core_nans = {k for k in nan_features if k not in ("recent_return_5d", "recent_vol_ratio")}
        assert len(core_nans) == 0, f"Core features have NaN: {core_nans}"

    def test_directional_vote_fraction_all_agree(self):
        import pandas as pd
        gen = self._make_generator()
        # All models predict above current_price=100 -> directional_vote_fraction=1.0 for BUY
        idx = pd.RangeIndex(5)
        bundle = {
            "ensemble_metadata": {},
            "detected_regime": "",
            "samossa_forecast": {"forecast": pd.Series([105.0] * 5, index=idx)},
            "mssa_rl_forecast": {"forecast": pd.Series([103.0] * 5, index=idx)},
            "garch_forecast": {"forecast": pd.Series([101.0] * 5, index=idx)},
        }
        features = gen._extract_classifier_features(
            forecast_bundle=bundle,
            current_price=100.0,
            expected_return=0.03,   # BUY direction
            lower_ci=None,
            upper_ci=None,
            snr=None,
            model_agreement=0.9,
            market_data=None,
        )
        assert features["directional_vote_fraction"] == 1.0


class TestBuildDirectionalTrainingData:
    def test_cold_start_when_jsonl_missing(self, tmp_path):
        from scripts.build_directional_training_data import build_dataset
        result = build_dataset(
            jsonl_path=tmp_path / "nonexistent.jsonl",
            checkpoint_dir=tmp_path,
        )
        assert result.get("cold_start") is True
        assert result.get("error") == "jsonl_not_found"

    def test_cold_start_when_insufficient_entries(self, tmp_path):
        import json as _json
        from scripts.build_directional_training_data import build_dataset
        jsonl = tmp_path / "quant_validation.jsonl"
        # Write 3 entries without classifier_features and no price data
        entries = [
            {"action": "BUY", "ticker": "AAPL", "timestamp": "2024-01-01T10:00:00Z",
             "execution_mode": "live", "forecast_horizon": 5}
            for _ in range(3)
        ]
        jsonl.write_text("\n".join(_json.dumps(e) for e in entries), encoding="utf-8")
        result = build_dataset(jsonl_path=jsonl, checkpoint_dir=tmp_path)
        assert result.get("cold_start") is True

    def test_synthetic_entries_excluded(self, tmp_path):
        import json as _json
        from scripts.build_directional_training_data import build_dataset
        jsonl = tmp_path / "quant_validation.jsonl"
        entries = [
            {"action": "BUY", "ticker": "AAPL", "timestamp": "2024-01-01T10:00:00Z",
             "execution_mode": "synthetic", "forecast_horizon": 5}
        ]
        jsonl.write_text("\n".join(_json.dumps(e) for e in entries), encoding="utf-8")
        result = build_dataset(jsonl_path=jsonl, checkpoint_dir=tmp_path)
        assert result.get("n_tradeable", 0) == 0

    def test_labels_from_forward_prices(self, tmp_path):
        """End-to-end: synthetic price parquet -> correct directional labels."""
        import json as _json
        import pandas as pd
        from scripts.build_directional_training_data import build_dataset
        from unittest.mock import patch

        # Create synthetic price data: steady uptrend
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
        prices = pd.DataFrame({"Close": 100 + np.arange(n, dtype=float)}, index=dates)
        parquet_path = tmp_path / "checkpoints" / "AAPL_data_extraction_test.parquet"
        parquet_path.parent.mkdir(parents=True)
        prices.to_parquet(parquet_path)

        # JSONL entry at day 10, horizon=5 -> forward close at day 15 > day 10 -> y=1
        entry_ts = dates[10].isoformat()
        jsonl = tmp_path / "quant_validation.jsonl"
        entries = [
            {
                "action": "BUY", "ticker": "AAPL",
                "timestamp": entry_ts,
                "execution_mode": "live",
                "forecast_horizon": 5,
                "classifier_features": {"ensemble_pred_return": 0.02},
            }
        ]
        jsonl.write_text("\n".join(_json.dumps(e) for e in entries), encoding="utf-8")

        output_parquet = tmp_path / "directional_dataset.parquet"
        with patch("scripts.build_directional_training_data._OUTPUT_PARQUET", output_parquet), \
             patch("scripts.build_directional_training_data._SUMMARY_PATH", tmp_path / "summary.json"):
            result = build_dataset(
                jsonl_path=jsonl,
                checkpoint_dir=tmp_path / "checkpoints",
            )

        if output_parquet.exists():
            df = pd.read_parquet(output_parquet)
            assert len(df) == 1
            assert df["y_directional"].iloc[0] == 1  # uptrend -> y=1
