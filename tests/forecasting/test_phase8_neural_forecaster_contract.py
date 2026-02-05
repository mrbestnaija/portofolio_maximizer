"""
Phase 8 Neural Forecaster contract tests.

These tests are intentionally lightweight and will skip/xfail when optional
dependencies or implementations are not yet available.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = ROOT / "config" / "pipeline_config.yml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _xfail_missing(module_name: str, exc: Exception) -> None:
    pytest.xfail(f"{module_name} not available: {exc}")


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - optional path
        _xfail_missing(module_name, exc)


@pytest.mark.gpu
def test_gpu_hardware_visible_via_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not available")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    output = result.stdout.strip()
    assert output


@pytest.mark.gpu
def test_gpu_memory_budget_under_limit() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not available")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    mem_values = [int(v.strip()) for v in result.stdout.splitlines() if v.strip()]
    if not mem_values:
        pytest.skip("No GPU memory values reported")
    assert all(v >= 12_000 for v in mem_values)


def test_pipeline_config_neural_section_present() -> None:
    cfg = _load_yaml(PIPELINE_CONFIG)
    forecasting = cfg.get("pipeline", {}).get("forecasting", {}) or cfg.get("forecasting", {})
    if not forecasting:
        pytest.xfail("forecasting config missing in pipeline_config.yml")
    if "neural" not in forecasting:
        pytest.xfail("neural config not yet defined in pipeline_config.yml")


def test_neural_forecaster_initialization_contract() -> None:
    module = _optional_import("forcester_ts.neural_forecaster")
    NeuralForecaster = getattr(module, "NeuralForecaster", None)
    if NeuralForecaster is None:
        pytest.xfail("NeuralForecaster class missing")
    instance = NeuralForecaster(model_type="patchtst", gpu=False)
    assert getattr(instance, "model_type", None) in {"patchtst", "nhits"}
    assert getattr(instance, "device", "cpu") in {"cpu", "cuda"}


def test_neural_forecaster_fit_forecast_contract() -> None:
    module = _optional_import("forcester_ts.neural_forecaster")
    NeuralForecaster = getattr(module, "NeuralForecaster", None)
    if NeuralForecaster is None:
        pytest.xfail("NeuralForecaster class missing")
    forecaster = NeuralForecaster(model_type="patchtst", gpu=False)

    panel = _sample_panel()
    try:
        forecaster.fit(panel, freq="1H")
        preds = forecaster.forecast(horizon=1)
    except NotImplementedError as exc:
        pytest.xfail(f"Neural forecaster not implemented: {exc}")

    assert preds is not None


def test_feature_forecaster_initialization_contract() -> None:
    module = _optional_import("forcester_ts.feature_forecaster")
    FeatureForecaster = getattr(module, "FeatureForecaster", None)
    if FeatureForecaster is None:
        pytest.xfail("FeatureForecaster class missing")
    instance = FeatureForecaster(gpu=False)
    assert instance is not None


def test_feature_forecaster_fit_forecast_contract() -> None:
    module = _optional_import("forcester_ts.feature_forecaster")
    FeatureForecaster = getattr(module, "FeatureForecaster", None)
    if FeatureForecaster is None:
        pytest.xfail("FeatureForecaster class missing")
    forecaster = FeatureForecaster(gpu=False)
    panel = _sample_panel()
    try:
        features = forecaster.create_features(panel, panel["y"])
        forecaster.fit(panel, exog_features=features)
        preds = forecaster.forecast(steps=1, exog_features=features)
    except NotImplementedError as exc:
        pytest.xfail(f"Feature forecaster not implemented: {exc}")
    assert preds is not None


def test_chronos_benchmark_contract() -> None:
    module = _optional_import("forcester_ts.chronos_benchmark")
    ChronosBenchmark = getattr(module, "ChronosBenchmark", None)
    if ChronosBenchmark is None:
        pytest.xfail("ChronosBenchmark class missing")
    try:
        benchmark = ChronosBenchmark()
        preds = benchmark.forecast(context=_sample_context(), horizon=1)
    except NotImplementedError as exc:
        pytest.xfail(f"Chronos benchmark not implemented: {exc}")
    assert preds is not None


def test_garch_config_volatility_only_flag() -> None:
    cfg = _load_yaml(PIPELINE_CONFIG)
    forecasting = cfg.get("pipeline", {}).get("forecasting", {}) or cfg.get("forecasting", {})
    garch = forecasting.get("garch")
    if not garch:
        pytest.xfail("garch config missing in pipeline_config.yml")
    use_for = garch.get("use_for")
    if use_for is None:
        pytest.xfail("garch.use_for not set in pipeline_config.yml")
    assert str(use_for).lower() == "volatility_only"


def _sample_panel() -> Any:
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2025-01-01", periods=24, freq="H")
    frames = []
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        values = 100 + np.cumsum(np.random.normal(0.0, 0.2, size=len(dates)))
        frame = pd.DataFrame({"unique_id": ticker, "ds": dates, "y": values})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _sample_context() -> Any:
    import numpy as np

    return np.random.normal(0.0, 1.0, size=48)
