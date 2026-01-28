# Phase 8: Neural Forecaster Implementation Guide

**Quick Start Guide for Neural Forecasting Integration**
**Timeline**: 7 weeks to production
**Hardware**: NVIDIA RTX 4060 Ti (16GB VRAM), CUDA 12.9

---

## Week 1: Infrastructure Setup

### Day 1: Environment Setup

```bash
# 1. Activate virtual environment
cd /c/Users/Bestman/personal_projects/portfolio_maximizer_v45
source simpleTrader_env/bin/activate  # Linux/Mac
# simpleTrader_env\Scripts\activate  # Windows

# 2. Install neural forecasting dependencies
pip install neuralforecast==1.6.4
pip install pytorch-forecasting==1.0.0
pip install skforecast==0.12.0
pip install xgboost[gpu]==2.0.3

# 3. Verify GPU setup
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Expected output:
# CUDA Available: True
# GPU: NVIDIA GeForce RTX 4060 Ti
```

### Day 2: Test GPU Acceleration

```python
# scripts/test_gpu_setup.py
import torch
import xgboost as xgb
import numpy as np

def test_pytorch_gpu():
    """Test PyTorch GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    # Simple tensor operation on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)

    print(f"✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return True

def test_xgboost_gpu():
    """Test XGBoost GPU"""
    # Generate sample data
    X = np.random.rand(10000, 20)
    y = np.random.rand(10000)

    # Train with GPU
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'predictor': 'gpu_predictor'
    }

    try:
        model = xgb.train(params, dtrain, num_boost_round=100)
        print("✅ XGBoost GPU working")
        return True
    except Exception as e:
        print(f"❌ XGBoost GPU failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing GPU Setup...")
    pytorch_ok = test_pytorch_gpu()
    xgboost_ok = test_xgboost_gpu()

    if pytorch_ok and xgboost_ok:
        print("\n🎉 All GPU tests passed! Ready for Phase 8.")
    else:
        print("\n⚠️  GPU setup incomplete. Fix issues before proceeding.")
```

### Day 3-4: Create Neural Forecaster Skeleton

```python
# forcester_ts/neural_forecaster.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, NHITS

logger = logging.getLogger(__name__)


class NeuralForecaster:
    """
    GPU-accelerated neural forecasting using PatchTST/NHITS.
    Handles 1-hour intraday forecasting for trending markets.
    """

    def __init__(
        self,
        model_type: str = 'patchtst',
        horizon: int = 1,
        context_length: int = 168,  # 1 week of hourly data
        gpu: bool = True
    ):
        self.model_type = model_type.lower()
        self.horizon = horizon
        self.context_length = context_length
        self.device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'

        # Initialize model
        self.model = None
        self.nf = None  # NeuralForecast wrapper

    def fit(
        self,
        panel_data: pd.DataFrame,
        freq: str = '1H'
    ):
        """
        Train neural forecaster on panel data.

        Args:
            panel_data: Must have columns [unique_id, ds, y]
                - unique_id: ticker symbol
                - ds: timestamp
                - y: target value (price or returns)
            freq: Data frequency ('1H' for hourly)
        """
        # Create model based on type
        if self.model_type == 'patchtst':
            models = [
                PatchTST(
                    h=self.horizon,
                    input_size=self.context_length,
                    patch_len=16,
                    stride=8,
                    hidden_size=128,
                    n_heads=8,
                    dropout=0.1,
                    max_steps=1000,
                    early_stop_patience_steps=10,
                )
            ]
        elif self.model_type == 'nhits':
            models = [
                NHITS(
                    h=self.horizon,
                    input_size=self.context_length,
                    stack_types=['identity', 'trend', 'seasonality'],
                    n_blocks=[1, 1, 1],
                    mlp_units=[[512, 512], [512, 512], [512, 512]],
                    dropout_prob_theta=0.1,
                    max_steps=1000,
                    early_stop_patience_steps=10,
                )
            ]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Initialize NeuralForecast
        self.nf = NeuralForecast(
            models=models,
            freq=freq
        )

        # Train
        logger.info(f"Training {self.model_type} on {len(panel_data)} samples...")
        self.nf.fit(df=panel_data)
        logger.info("Training complete")

    def forecast(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts for all tickers.

        Returns:
            DataFrame with columns [unique_id, ds, PatchTST] or [unique_id, ds, NHITS]
        """
        if self.nf is None:
            raise ValueError("Model not fitted. Call fit() first.")

        h = horizon or self.horizon
        forecasts = self.nf.predict()

        return forecasts

    def get_model_summary(self) -> Dict:
        """Return model diagnostics for confidence scoring."""
        if self.nf is None:
            return {}

        # Get training loss from models
        model = self.nf.models[0]

        return {
            'model_type': self.model_type,
            'training_loss': float(model.loss),  # Final training loss
            'horizon': self.horizon,
            'context_length': self.context_length,
            'device': self.device,
        }
```

### Day 5: Create Feature-Based Forecaster

```python
# forcester_ts/feature_forecaster.py
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from skforecast.ForecasterAutoreg import ForecasterAutoreg

logger = logging.getLogger(__name__)


class FeatureForecaster:
    """
    Feature-based forecasting using skforecast + XGBoost GPU.
    Generates lag/rolling/volume/volatility features for directional edge.
    """

    def __init__(
        self,
        lags: int = 24,
        gpu: bool = True
    ):
        self.lags = lags
        self.gpu = gpu

        # Create XGBoost regressor
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        if gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })

        self.regressor = xgb.XGBRegressor(**xgb_params)

        # Create forecaster
        self.forecaster = ForecasterAutoreg(
            regressor=self.regressor,
            lags=lags
        )

    def create_features(
        self,
        price_series: pd.Series,
        volume_series: pd.Series = None
    ) -> pd.DataFrame:
        """
        Engineer features for forecasting.

        Returns:
            DataFrame with lag/rolling/volume/volatility features
        """
        features = pd.DataFrame(index=price_series.index)

        # Price lags (already handled by ForecasterAutoreg)
        # Rolling statistics
        for window in [6, 12, 24]:
            features[f'ma_{window}h'] = price_series.rolling(window).mean()
            features[f'std_{window}h'] = price_series.rolling(window).std()

        # Returns
        returns = price_series.pct_change()
        features['returns_1h'] = returns
        features['returns_24h'] = price_series.pct_change(24)

        # Volatility
        features['realized_vol_24h'] = returns.rolling(24).std()

        # Volume features (if available)
        if volume_series is not None:
            features['volume_ma_24h'] = volume_series.rolling(24).mean()
            features['volume_ratio'] = volume_series / features['volume_ma_24h']

        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(price_series, 14)

        return features.dropna()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def fit(
        self,
        price_series: pd.Series,
        exog_features: pd.DataFrame = None
    ):
        """
        Train forecaster on price series + external features.

        Args:
            price_series: Target price series
            exog_features: External features (optional)
        """
        logger.info(f"Training FeatureForecaster with {len(price_series)} samples...")

        self.forecaster.fit(
            y=price_series,
            exog=exog_features
        )

        logger.info("Training complete")

    def forecast(
        self,
        steps: int = 1,
        exog_features: pd.DataFrame = None
    ) -> np.ndarray:
        """Generate forecast for next N steps."""
        forecast = self.forecaster.predict(
            steps=steps,
            exog=exog_features
        )

        return forecast

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost model."""
        importance = self.forecaster.regressor.feature_importances_

        feature_names = [f'lag_{i+1}' for i in range(self.lags)]
        if hasattr(self.forecaster, 'exog_'):
            feature_names.extend(self.forecaster.exog_.columns)

        return dict(zip(feature_names, importance))
```

---

## Week 2-3: Integration

### Integrate with Ensemble Coordinator

```python
# In forcester_ts/forecaster.py

from .neural_forecaster import NeuralForecaster
from .feature_forecaster import FeatureForecaster

class TimeSeriesForecaster:
    def __init__(self, config: TimeSeriesForecasterConfig):
        # ... existing code ...

        # Phase 8: Add neural forecasters
        self._neural: Optional[NeuralForecaster] = None
        self._feature: Optional[FeatureForecaster] = None

        if config.neural_enabled:
            self._neural = NeuralForecaster(
                model_type=config.neural_kwargs.get('model_type', 'patchtst'),
                horizon=config.forecast_horizon,
                gpu=config.neural_kwargs.get('gpu', True)
            )

        if config.feature_enabled:
            self._feature = FeatureForecaster(
                lags=config.feature_kwargs.get('lags', 24),
                gpu=config.feature_kwargs.get('gpu', True)
            )
```

---

## Week 4-5: Real-Time Retraining

### Implement Trigger-Based Retraining

```python
# forcester_ts/realtime_trainer.py

class RealtimeTrainer:
    """
    Trigger-based real-time retraining system.
    Monitors volatility, regime changes, tracking error.
    """

    def __init__(self, models, cooldown_minutes=15):
        self.models = models
        self.last_retrain = {}
        self.cooldown = pd.Timedelta(minutes=cooldown_minutes)

    def check_triggers(self, current_data):
        """Check if retrain triggers are active."""
        triggers = {
            'vol_spike': self._check_vol_spike(current_data),
            'regime_change': self._check_regime_change(current_data),
            'tracking_error': self._check_tracking_error(current_data),
        }
        return triggers

    def _check_vol_spike(self, data):
        """Volatility > 2x rolling average?"""
        current_vol = data['returns'].tail(4).std()
        avg_vol = data['returns'].tail(24).std()
        return current_vol > 2.0 * avg_vol

    def retrain_async(self, model_name):
        """Async retrain (non-blocking)."""
        import threading

        def _retrain():
            logger.info(f"Retraining {model_name}...")
            # Retrain logic here
            self.last_retrain[model_name] = pd.Timestamp.now()

        thread = threading.Thread(target=_retrain)
        thread.start()
```

---

## Week 6: Testing & Validation

### Run Shadow Canary Test

```bash
# Run side-by-side: Classical vs Neural
python scripts/compare_forecasters.py \
  --tickers AAPL,MSFT,NVDA \
  --models garch,samossa,patchtst,xgboost \
  --start 2024-07-01 \
  --end 2026-01-18

# Expected output:
# Model Performance (1-hour horizon):
#   GARCH:    RMSE 30.64 (best for MSFT)
#   SAMoSSA:  RMSE 45.32
#   PatchTST: RMSE 28.15 (best for NVDA) ✨
#   XGBoost:  RMSE 32.47 (best DA: 58%)
```

---

## Week 7: Production Hardening

### Add Monitoring & Alerts

```python
# dashboard/neural_monitor.py

class NeuralMonitor:
    """Monitor neural forecaster performance."""

    def check_health(self):
        checks = {
            'gpu_memory': self._check_gpu_memory(),
            'inference_latency': self._check_latency(),
            'model_drift': self._check_drift(),
        }

        if any(not v for v in checks.values()):
            self._send_alert(checks)

    def _check_gpu_memory(self):
        """Ensure GPU memory < 12GB (leave 4GB headroom)."""
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            return used < 12.0
        return True
```

---

## Expected Performance (Phase 8 Complete)

| Ticker | Phase 7.3 | Phase 8 | Model Selected | Improvement |
|--------|-----------|---------|----------------|-------------|
| MSFT | 1.037 | **1.02** | GARCH | +1.6% |
| AAPL | 1.470 | **1.08** | PatchTST+GARCH | **+26.5%** |
| NVDA | 1.453 | **1.05** | PatchTST | **+27.7%** |
| **Overall** | **1.386** | **1.05** | **Adaptive** | **🎯 +24.2%** |

**Target Achievement**: 3/3 tickers at <1.1x RMSE ratio! 🎉

---

## Quick Reference Commands

```bash
# Install dependencies
pip install neuralforecast skforecast xgboost[gpu]

# Test GPU
python scripts/test_gpu_setup.py

# Run neural forecaster
python scripts/run_etl_pipeline.py \
  --tickers AAPL \
  --models patchtst \
  --execution-mode live

# Compare models
python scripts/compare_forecasters.py \
  --tickers AAPL,MSFT,NVDA \
  --models all

# Monitor performance
python dashboard/neural_monitor.py --watch
```

---

**Status**: Ready to begin Week 1 after Phase 7.4/7.5 complete
**Prerequisites**: Phase 7.4 calibration + Phase 7.5 weight optimization
**Timeline**: 7 weeks to production neural forecasting
