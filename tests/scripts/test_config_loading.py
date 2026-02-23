#!/usr/bin/env python3
"""
Quick test to verify config loading from YAML.
"""
import sys
import yaml
from pathlib import Path

# Load the config file
config_path = Path("config/forecasting_config.yml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

forecasting = config.get('forecasting', {})

print("=== Config Loading Test ===")
print(f"\nLoaded from: {config_path.absolute()}")

# Check MSSA-RL config
mssa_rl = forecasting.get('mssa_rl', {})
print(f"\nMSSA-RL config:")
print(f"  change_point_threshold: {mssa_rl.get('change_point_threshold')}")
print(f"  window_length: {mssa_rl.get('window_length')}")
print(f"  enabled: {mssa_rl.get('enabled')}")

# Check SARIMAX config
sarimax = forecasting.get('sarimax', {})
print(f"\nSARIMAX config:")
print(f"  max_p: {sarimax.get('max_p')}")
print(f"  max_q: {sarimax.get('max_q')}")
print(f"  seasonal_periods: {sarimax.get('seasonal_periods')}")
print(f"  trend: {sarimax.get('trend')}")

# Check SAMoSSA config
samossa = forecasting.get('samossa', {})
print(f"\nSAMoSSA config:")
print(f"  window_length: {samossa.get('window_length')}")
print(f"  n_components: {samossa.get('n_components')}")
print(f"  enabled: {samossa.get('enabled')}")

print("\n=== Verification ===")
if mssa_rl.get('change_point_threshold') == 3.5:
    print("✓ MSSA-RL threshold is 3.5 (correct)")
else:
    print(f"✗ MSSA-RL threshold is {mssa_rl.get('change_point_threshold')} (should be 3.5)")

if sarimax.get('max_p') == 3 and sarimax.get('seasonal_periods') == 12:
    print("✓ SARIMAX hyperparameters updated (max_p=3, seasonal=12)")
else:
    print(f"✗ SARIMAX not updated (max_p={sarimax.get('max_p')}, seasonal={sarimax.get('seasonal_periods')})")

if samossa.get('window_length') == 60:
    print("✓ SAMoSSA window_length is 60 (correct)")
else:
    print(f"✗ SAMoSSA window_length is {samossa.get('window_length')} (should be 60)")
