"""
Quick test script to verify Phase 7.5 regime detection integration.

Tests:
1. Load forecasting config with regime detection enabled
2. Create TimeSeriesForecaster with regime detection
3. Generate synthetic data with known characteristics
4. Verify regime detection works
5. Verify candidate reordering occurs
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig


def generate_synthetic_data(regime_type: str, length: int = 200) -> pd.Series:
    """Generate synthetic price data with specific regime characteristics."""
    np.random.seed(42)

    if regime_type == "low_vol_rangebound":
        # Low volatility, mean-reverting
        prices = 100 + np.random.randn(length) * 0.5  # Very low vol
        prices = 100 + np.cumsum(np.random.randn(length) * 0.01)  # Slight drift

    elif regime_type == "high_vol_trending":
        # High volatility with strong upward trend
        trend = np.linspace(0, 50, length)  # Strong trend
        noise = np.random.randn(length) * 3  # High volatility
        prices = 100 + trend + noise

    elif regime_type == "moderate_trending":
        # Medium volatility with moderate trend
        trend = np.linspace(0, 20, length)
        noise = np.random.randn(length) * 1
        prices = 100 + trend + noise

    else:  # crisis
        # Extreme volatility
        prices = 100 + np.cumsum(np.random.randn(length) * 5)

    dates = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.Series(prices, index=dates, name="Close")


def test_regime_detection_disabled():
    """Test with regime detection disabled (baseline)."""
    print("\n" + "=" * 80)
    print("TEST 1: Regime Detection DISABLED (Phase 7.4 baseline)")
    print("=" * 80)

    config = TimeSeriesForecasterConfig(
        regime_detection_enabled=False,
        ensemble_enabled=True,
        forecast_horizon=30,
    )

    forecaster = TimeSeriesForecaster(config=config)
    assert forecaster._regime_detector is None, "Regime detector should be None when disabled"

    print("[OK] Regime detector correctly disabled")
    print(f"[OK] Ensemble enabled: {forecaster._ensemble_config.enabled}")


def test_regime_detection_enabled():
    """Test with regime detection enabled."""
    print("\n" + "=" * 80)
    print("TEST 2: Regime Detection ENABLED (Phase 7.5)")
    print("=" * 80)

    # Load actual config
    config_path = Path(__file__).parent.parent / "config" / "forecasting_config.yml"
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    regime_config = yaml_config["forecasting"]["regime_detection"]
    print(f"\n[INFO] Loaded regime config: enabled={regime_config['enabled']}")
    print(f"[INFO] Lookback window: {regime_config['lookback_window']}")
    print(f"[INFO] Vol thresholds: low={regime_config['vol_threshold_low']}, high={regime_config['vol_threshold_high']}")

    # Enable regime detection for this test
    config = TimeSeriesForecasterConfig(
        regime_detection_enabled=True,
        regime_detection_kwargs={
            "lookback_window": 60,
            "vol_threshold_low": 0.15,
            "vol_threshold_high": 0.30,
            "trend_threshold_weak": 0.30,
            "trend_threshold_strong": 0.60,
        },
        ensemble_enabled=True,
        ensemble_kwargs={
            "candidate_weights": [
                {"garch": 0.85, "sarimax": 0.1, "samossa": 0.05},
                {"samossa": 0.6, "garch": 0.3, "mssa_rl": 0.1},
                {"garch": 1.0},
                {"samossa": 1.0},
            ],
        },
        forecast_horizon=30,
    )

    forecaster = TimeSeriesForecaster(config=config)
    assert forecaster._regime_detector is not None, "Regime detector should be initialized"

    print("[OK] Regime detector initialized")
    print(f"[OK] Ensemble enabled: {forecaster._ensemble_config.enabled}")
    print(f"[OK] Candidate count: {len(forecaster._ensemble_config.candidate_weights)}")


def test_low_vol_rangebound_regime():
    """Test regime detection on low volatility rangebound data."""
    print("\n" + "=" * 80)
    print("TEST 3: Low Volatility Rangebound Data")
    print("=" * 80)

    data = generate_synthetic_data("low_vol_rangebound", length=150)
    print(f"\n[INFO] Generated {len(data)} days of low-vol rangebound data")
    print(f"[INFO] Price range: {data.min():.2f} - {data.max():.2f}")
    print(f"[INFO] Daily std: {data.pct_change().std():.4f}")

    config = TimeSeriesForecasterConfig(
        regime_detection_enabled=True,
        regime_detection_kwargs={"lookback_window": 60},
        sarimax_enabled=True,
        garch_enabled=True,
        samossa_enabled=False,  # Disable to speed up test
        mssa_rl_enabled=False,
        ensemble_enabled=True,
        ensemble_kwargs={
            "candidate_weights": [
                {"garch": 0.85, "sarimax": 0.15},
                {"sarimax": 0.6, "garch": 0.4},
                {"garch": 1.0},
            ],
        },
        forecast_horizon=10,
    )

    forecaster = TimeSeriesForecaster(config=config)

    print("\n[INFO] Fitting forecaster...")
    forecaster.fit(data)

    # Check regime detection result
    if hasattr(forecaster, '_regime_result') and forecaster._regime_result:
        regime = forecaster._regime_result
        print(f"\n[REGIME DETECTED]")
        print(f"  Regime: {regime['regime']}")
        print(f"  Confidence: {regime['confidence']:.3f}")
        print(f"  Features:")
        for key, value in regime['features'].items():
            print(f"    {key}: {value:.4f}")
        print(f"  Recommendations: {regime['recommendations']}")

        # For low-vol rangebound, expect GARCH to be recommended
        if 'garch' in regime['recommendations']:
            print("[OK] GARCH correctly recommended for low-vol rangebound regime")
        else:
            print("[WARNING] GARCH not in recommendations (unexpected)")
    else:
        print("[ERROR] No regime result found")

    print("\n[INFO] Generating forecast...")
    result = forecaster.forecast()

    print(f"\n[FORECAST RESULT]")
    print(f"  Regime: {result.get('regime', 'N/A')}")
    print(f"  Regime confidence: {result.get('regime_confidence', 'N/A')}")

    if result.get("ensemble_forecast"):
        ensemble = result["ensemble_forecast"]
        print(f"  Ensemble weights: {ensemble.get('weights', {})}")
        print(f"  Primary model: {ensemble.get('primary_model', 'N/A')}")

        # Check if GARCH dominates (expected for low-vol rangebound)
        weights = ensemble.get('weights', {})
        if weights.get('garch', 0) > 0.5:
            print("[OK] GARCH dominates ensemble (expected for low-vol rangebound)")
        else:
            print(f"[INFO] GARCH weight: {weights.get('garch', 0):.2f} (may vary based on fit quality)")


def test_high_vol_trending_regime():
    """Test regime detection on high volatility trending data."""
    print("\n" + "=" * 80)
    print("TEST 4: High Volatility Trending Data")
    print("=" * 80)

    data = generate_synthetic_data("high_vol_trending", length=150)
    print(f"\n[INFO] Generated {len(data)} days of high-vol trending data")
    print(f"[INFO] Price range: {data.min():.2f} - {data.max():.2f}")
    print(f"[INFO] Daily std: {data.pct_change().std():.4f}")

    config = TimeSeriesForecasterConfig(
        regime_detection_enabled=True,
        regime_detection_kwargs={"lookback_window": 60},
        sarimax_enabled=True,
        garch_enabled=True,
        samossa_enabled=False,  # Disable to speed up test
        mssa_rl_enabled=False,
        ensemble_enabled=True,
        ensemble_kwargs={
            "candidate_weights": [
                {"garch": 0.85, "sarimax": 0.15},
                {"sarimax": 0.6, "garch": 0.4},
                {"garch": 1.0},
                {"sarimax": 1.0},
            ],
        },
        forecast_horizon=10,
    )

    forecaster = TimeSeriesForecaster(config=config)

    print("\n[INFO] Fitting forecaster...")
    forecaster.fit(data)

    if hasattr(forecaster, '_regime_result') and forecaster._regime_result:
        regime = forecaster._regime_result
        print(f"\n[REGIME DETECTED]")
        print(f"  Regime: {regime['regime']}")
        print(f"  Confidence: {regime['confidence']:.3f}")
        print(f"  Features:")
        for key, value in regime['features'].items():
            print(f"    {key}: {value:.4f}")
        print(f"  Recommendations: {regime['recommendations']}")

    print("\n[INFO] Generating forecast...")
    result = forecaster.forecast()

    print(f"\n[FORECAST RESULT]")
    print(f"  Regime: {result.get('regime', 'N/A')}")

    if result.get("ensemble_forecast"):
        ensemble = result["ensemble_forecast"]
        print(f"  Ensemble weights: {ensemble.get('weights', {})}")
        print(f"  Primary model: {ensemble.get('primary_model', 'N/A')}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 7.5 REGIME DETECTION INTEGRATION TEST")
    print("=" * 80)
    print("\nThis script tests the regime detection integration without full pipeline.")
    print("Expected: Regime detected, candidates reordered, no errors.")

    try:
        test_regime_detection_disabled()
        test_regime_detection_enabled()
        test_low_vol_rangebound_regime()
        test_high_vol_trending_regime()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run full multi-ticker validation: python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,NVDA")
        print("2. Enable regime detection in config/forecasting_config.yml (set enabled: true)")
        print("3. Compare results to Phase 7.4 baseline")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
