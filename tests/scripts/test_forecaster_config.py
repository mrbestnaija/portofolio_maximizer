#!/usr/bin/env python3
"""
Test forecaster config loading directly.
"""
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

# Load config
with open("config/forecasting_config.yml", 'r') as f:
    config = yaml.safe_load(f)

forecasting_cfg = config['forecasting']

# Import forecaster
from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

# Create test data (synthetic sine wave)
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=300, freq='D')
prices = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, 300)) + np.random.randn(300) * 2
series = pd.Series(prices, index=dates, name='Close')

print("=" * 70)
print("Testing Forecaster Config Loading")
print("=" * 70)

# Build config from YAML (same as pipeline does)
config_obj = TimeSeriesForecasterConfig(
    forecast_horizon=30,
    sarimax_enabled=forecasting_cfg['sarimax'].get('enabled', True),
    samossa_enabled=forecasting_cfg['samossa'].get('enabled', True),
    mssa_rl_enabled=forecasting_cfg['mssa_rl'].get('enabled', True),
    sarimax_kwargs={k: v for k, v in forecasting_cfg['sarimax'].items() if k != 'enabled'},
    samossa_kwargs={k: v for k, v in forecasting_cfg['samossa'].items() if k != 'enabled'},
    mssa_rl_kwargs={k: v for k, v in forecasting_cfg['mssa_rl'].items() if k != 'enabled'},
)

print("\nConfig object kwargs:")
print(f"  sarimax_kwargs: {config_obj.sarimax_kwargs}")
print(f"  samossa_kwargs: {config_obj.samossa_kwargs}")
print(f"  mssa_rl_kwargs: {config_obj.mssa_rl_kwargs}")

print("\n" + "=" * 70)
print("Creating forecaster and fitting models...")
print("=" * 70)

forecaster = TimeSeriesForecaster(config=config_obj)
forecaster.fit(series)

print("\n" + "=" * 70)
print("Fit completed! Check the logs above for config verification.")
print("=" * 70)
