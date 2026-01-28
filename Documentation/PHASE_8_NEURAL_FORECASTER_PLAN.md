# Phase 8: Neural Forecaster Integration Plan

**Status**: Planning
**Target**: Add GPU-accelerated neural forecasters for 1-hour intraday trading
**Hardware**: NVIDIA RTX 4060 Ti (16GB), CUDA 12.9

---

## System Specifications

### Hardware Resources
```
GPU: NVIDIA GeForce RTX 4060 Ti
VRAM: 16380 MiB (16GB)
CUDA: 12.9
Driver: 577.00 (nvidia-smi 575.64.04)
Current Usage: 1051MiB (~6%), 2% GPU utilization
Available: ~15GB for training/inference
```

### Trading Context
- **Forecast Horizon**: 1-hour intraday (60-minute returns/price/volatility)
- **Training Frequency**: Dual mode
  - Real-time retrain: On significant market moves or regime changes
  - Daily batch: End-of-day retraining with full dataset
- **Instruments**: Stocks + commodities (10-50 tickers), derivatives future rollout
- **Markets**: Emerging markets (higher volatility, less efficiency)
- **Features**: Price + volume + volatility + cross-sectional (relative strength, sector momentum)

---

## Neural Forecaster Stack

### 1. Mean Forecasting: PatchTST (Primary) + NHITS (Backup)

**Why PatchTST?**
- Transformer-based, state-of-the-art for multivariate time series
- Handles panel data (multi-ticker) natively
- Patch-based attention reduces memory footprint (fits in 16GB)
- Strong on 1-hour intraday horizons
- Via Nixtla NeuralForecast (turnkey PyTorch, GPU-ready)

**Why NHITS as backup?**
- MLP-based, faster inference than transformers
- Better for very short horizons when transformers overfit
- Less VRAM intensive (fallback if memory constrained)

**Integration Path**:
```python
# New file: forcester_ts/neural_forecaster.py
class NeuralForecaster:
    """
    Wrapper for PatchTST/NHITS via NeuralForecast library.
    Handles multi-ticker training, real-time updates, and GPU acceleration.
    """
    def __init__(self, model_type='patchtst', gpu=True):
        self.model_type = model_type
        self.device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'

    def fit(self, panel_data: pd.DataFrame, freq='1H'):
        """
        Train on panel data (multi-ticker).
        panel_data: columns=[unique_id, ds, y, exog_vars...]
        """

    def forecast(self, horizon=1):
        """Generate 1-hour ahead forecasts for all tickers."""

    def get_model_summary(self):
        """Return validation metrics, loss, attention weights."""
```

**Config**:
```yaml
# config/pipeline_config.yml
forecasting:
  neural:
    enabled: true
    model_type: "patchtst"  # or "nhits"
    gpu: true
    horizon: 1  # 1-hour ahead
    context_length: 168  # 1 week of hourly data
    hidden_size: 128
    num_layers: 3
    dropout: 0.1
    batch_size: 32
    learning_rate: 1e-4
    epochs: 50
    early_stopping_patience: 10
    validation_split: 0.2

    # Real-time retraining triggers
    realtime_retrain:
      enabled: true
      triggers:
        - {type: "volatility_spike", threshold: 2.0}  # 2x normal vol
        - {type: "regime_change", threshold: 0.7}     # CP detector
        - {type: "tracking_error", threshold: 1.5}    # RMSE > 1.5x baseline
      max_retrain_frequency: "15min"  # Cooldown period

    # Daily batch retraining
    daily_batch:
      enabled: true
      schedule: "03:00"  # UTC, after market close
      full_dataset: true
      save_checkpoint: true
```

---

### 2. Feature-Based Model: skforecast + XGBoost GPU

**Why skforecast + XGBoost?**
- Interpretable, trading-friendly features (lags, rolling stats, cross-sectional)
- Often wins on directional accuracy (DA) even if RMSE higher
- XGBoost gpu_hist mode fully GPU-accelerated
- Native support for volume/volatility/cross-sectional features
- Fast inference (<1ms per ticker)

**Feature Engineering**:
```python
# New file: forcester_ts/feature_forecaster.py
class FeatureForecaster:
    """
    Lag-based forecaster using skforecast + XGBoost GPU.
    Generates rich feature set for directional edge.
    """
    def __init__(self, gpu=True):
        self.forecaster = ForecasterAutoreg(
            regressor=xgb.XGBRegressor(
                tree_method='gpu_hist',
                gpu_id=0,
                predictor='gpu_predictor'
            ),
            lags=24  # 24 hours of lags
        )

    def create_features(self, price_series, volume_series):
        """
        Create engineered features:
        - Price lags: 1h, 2h, 4h, 8h, 24h
        - Rolling stats: MA, EMA, std, min, max (6h, 12h, 24h windows)
        - Volume features: volume_ratio, volume_ma, volume_spike
        - Volatility: realized_vol (1h, 4h, 24h), GARCH forecast
        - Cross-sectional: relative_strength, sector_momentum, ticker_rank
        """

    def fit(self, train_data, exog_features):
        """Train on enriched feature matrix."""

    def forecast(self, steps=1, exog_features=None):
        """Generate 1-step ahead forecast with feature matrix."""
```

**Config**:
```yaml
forecasting:
  feature_based:
    enabled: true
    model: "xgboost_gpu"

    feature_sets:
      price_lags: [1, 2, 4, 8, 24]  # Hours
      rolling_windows: [6, 12, 24]  # Hours
      volume_features: true
      volatility_features: true
      cross_sectional: true

    xgboost:
      tree_method: "gpu_hist"
      gpu_id: 0
      max_depth: 6
      learning_rate: 0.05
      n_estimators: 200
      subsample: 0.8
      colsample_bytree: 0.8

    feature_importance_threshold: 0.01  # Drop low-importance features
```

---

### 3. Zero-Shot Baseline: Chronos-Bolt

**Why Chronos-Bolt?**
- Pre-trained on massive time series corpus
- Zero-shot (no training needed) - instant deployment
- Excellent for sanity checks and benchmarking
- Read-only (doesn't participate in trading, just validation)

**Integration**:
```python
# New file: forcester_ts/chronos_benchmark.py
class ChronosBenchmark:
    """
    Zero-shot benchmark using Chronos-Bolt from HuggingFace.
    Used for validation only, not for live trading.
    """
    def __init__(self):
        from chronos import ChronosPipeline
        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

    def forecast(self, context, horizon=1):
        """Generate zero-shot forecast (no training)."""

    def benchmark(self, test_data, baseline_forecasts):
        """
        Compare Chronos vs our models on held-out data.
        Log RMSE, sMAPE, DA relative to baseline.
        """
```

**Config**:
```yaml
forecasting:
  benchmark:
    chronos:
      enabled: true
      model: "amazon/chronos-bolt-small"
      run_on: "holdout"  # Only on held-out validation data
      compare_against: ["garch", "patchtst", "feature_based"]
      log_metrics: true
      alert_on_regression: true  # Warn if we're worse than zero-shot
```

---

### 4. Volatility Forecasting: Keep GARCH (Enhanced)

**Why keep GARCH?**
- Already best performer (RMSE 30.64)
- Critical for risk management (position sizing, stops, regime detection)
- Fast, interpretable, well-understood by traders
- Use for volatility targeting, not mean forecasting

**Enhancement**:
```yaml
forecasting:
  garch:
    enabled: true
    use_for: "volatility_only"  # Don't use for mean forecasting

    applications:
      - position_sizing: true     # Scale positions by inverse vol
      - stop_loss: true           # Adaptive stops based on vol forecast
      - regime_detection: true    # High vol = defensive, low vol = aggressive
      - portfolio_risk: true      # VaR calculation

    model_variants:
      - standard: {p: 1, q: 1, vol: "GARCH"}
      - asymmetric: {p: 1, q: 1, vol: "EGARCH"}  # Leverage effect
      - threshold: {p: 1, q: 1, vol: "GJR-GARCH"}  # News impact
```

---

## Ensemble Architecture

### Model Selection Logic

```python
# Enhanced ensemble with neural models
ensemble:
  enabled: true
  confidence_scaling: false  # Config-based selection (Phase 7.3)

  candidate_weights:
    # Neural-dominant (liquid, low-vol regimes)
    - {patchtst: 0.7, feature_based: 0.2, garch: 0.1}
    - {patchtst: 0.5, garch: 0.3, feature_based: 0.2}

    # Feature-dominant (high-vol, trending regimes)
    - {feature_based: 0.6, patchtst: 0.3, garch: 0.1}
    - {feature_based: 0.8, garch: 0.2}

    # GARCH-dominant (ultra high-vol, crisis regimes) - Phase 7.3
    - {garch: 0.85, patchtst: 0.1, feature_based: 0.05}
    - {garch: 0.7, feature_based: 0.3}

    # Classical baseline (illiquid, sparse data)
    - {sarimax: 0.6, samossa: 0.4}

    # Pure model fallbacks
    - {patchtst: 1.0}
    - {feature_based: 1.0}
    - {garch: 1.0}

  regime_detection:
    enabled: true
    features:
      - realized_volatility_24h
      - trend_strength  # ADX-like
      - volume_profile
      - cross_sectional_dispersion
    thresholds:
      high_vol: 0.30    # > 30% daily vol
      trending: 0.60    # > 60% ADX
      crisis: 0.50      # > 50% daily vol + vol spike
```

---

## Training Strategy

### Real-Time Retraining

**Triggers** (check every 15min):
1. **Volatility Spike**: `realized_vol_1h > 2.0 * rolling_avg_vol_24h`
2. **Regime Change**: Change-point detector fires (MSSA-RL or CUSUM)
3. **Tracking Error**: `rolling_rmse_4h > 1.5 * baseline_rmse`
4. **Data Drift**: Distribution shift detected (KS test on returns)

**Process**:
```python
# New file: forcester_ts/realtime_trainer.py
class RealtimeTrainer:
    def __init__(self, models, cooldown_minutes=15):
        self.models = models
        self.last_retrain = {}
        self.cooldown = pd.Timedelta(minutes=cooldown_minutes)

    def check_triggers(self, current_data):
        """Check if any retrain triggers are active."""
        triggers = {
            'vol_spike': self._check_vol_spike(current_data),
            'regime_change': self._check_regime_change(current_data),
            'tracking_error': self._check_tracking_error(current_data),
        }
        return triggers

    def retrain_if_needed(self, triggers):
        """
        Retrain models that need updates.
        Use incremental learning for XGBoost (warm_start).
        Full retrain for PatchTST (but cache embeddings).
        """
        if any(triggers.values()):
            logger.warning("Retrain triggered: %s", triggers)
            # Async retrain (don't block trading)
            self._async_retrain_models()
```

**GPU Memory Management**:
- PatchTST training: ~4GB VRAM (batch_size=32, context=168)
- XGBoost training: ~2GB VRAM (10K samples, 50 features)
- Inference: <500MB (all models combined)
- **Headroom**: ~9GB available (56% free)

### Daily Batch Retraining

**Schedule**: 03:00 UTC (after market close, before next session)

**Process**:
1. Load full day's data (all tickers)
2. Retrain PatchTST on full panel (10-50 tickers)
3. Retrain XGBoost per-ticker with full feature set
4. Update GARCH parameters with latest data
5. Run validation on held-out data (last 24h)
6. Save checkpoints to disk
7. Compare vs Chronos-Bolt benchmark
8. Generate performance report (email/dashboard)

**Checkpointing**:
```yaml
checkpointing:
  enabled: true
  path: "models/checkpoints/"
  keep_last_n: 7  # Keep 1 week of checkpoints
  format: "pytorch"  # For PatchTST/NHITS
  versioning: true
  metadata:
    - training_date
    - validation_rmse
    - training_duration
    - gpu_memory_used
```

---

## Feature Engineering Pipeline

### Price Features
```python
# Lags: 1h, 2h, 4h, 8h, 24h
price_lags = [price.shift(i) for i in [1, 2, 4, 8, 24]]

# Rolling statistics (6h, 12h, 24h windows)
rolling_features = {
    'ma_6h': price.rolling(6).mean(),
    'ma_12h': price.rolling(12).mean(),
    'ma_24h': price.rolling(24).mean(),
    'ema_6h': price.ewm(span=6).mean(),
    'ema_12h': price.ewm(span=12).mean(),
    'std_6h': price.rolling(6).std(),
    'std_24h': price.rolling(24).std(),
    'min_24h': price.rolling(24).min(),
    'max_24h': price.rolling(24).max(),
}

# Technical indicators
tech_indicators = {
    'rsi_14': RSI(price, period=14),
    'macd': MACD(price),
    'bbands': BollingerBands(price, period=20),
}
```

### Volume Features
```python
volume_features = {
    'volume_ratio': volume / volume.rolling(24).mean(),
    'volume_ma_6h': volume.rolling(6).mean(),
    'volume_spike': (volume > 2 * volume.rolling(24).mean()).astype(int),
    'volume_trend': volume.rolling(6).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]),
}
```

### Volatility Features
```python
volatility_features = {
    'realized_vol_1h': returns.rolling(1).std(),
    'realized_vol_4h': returns.rolling(4).std(),
    'realized_vol_24h': returns.rolling(24).std(),
    'garch_forecast_1h': garch_model.forecast(horizon=1),
    'parkinson_vol': parkinson_estimator(high, low),  # High-low estimator
}
```

### Cross-Sectional Features
```python
# Compute relative to sector/market
cross_sectional = {
    'relative_strength': (price / sector_index) / (price / sector_index).rolling(24).mean(),
    'sector_momentum': sector_returns.rolling(24).mean(),
    'ticker_rank': price.pct_change(24).rank(pct=True),  # Percentile rank across tickers
    'dispersion': (price.pct_change() - sector_returns).abs(),  # Idiosyncratic risk
}
```

---

## Implementation Phases

### Phase 8.1: Infrastructure Setup (Week 1)
- [ ] Install NeuralForecast, PyTorch, XGBoost GPU
- [ ] Verify GPU acceleration (test PatchTST + XGBoost on sample data)
- [ ] Create `forcester_ts/neural_forecaster.py` skeleton
- [ ] Create `forcester_ts/feature_forecaster.py` skeleton
- [ ] Create `forcester_ts/chronos_benchmark.py` skeleton
- [ ] Add config schema to `pipeline_config.yml`

### Phase 8.2: PatchTST Integration (Week 2)
- [ ] Implement `NeuralForecaster` class with PatchTST
- [ ] Add panel data preprocessing (unique_id, ds, y format)
- [ ] Implement training loop with validation split
- [ ] Add GPU memory monitoring
- [ ] Test on single ticker (AAPL), then multi-ticker (AAPL, MSFT, NVDA)
- [ ] Measure inference latency (<100ms target)

### Phase 8.3: Feature-Based Model (Week 3)
- [ ] Implement `FeatureForecaster` with skforecast + XGBoost GPU
- [ ] Build feature engineering pipeline (lags, rolling, volume, vol, cross-sectional)
- [ ] Add feature importance tracking
- [ ] Test directional accuracy vs RMSE trade-off
- [ ] Integrate with ensemble coordinator

### Phase 8.4: Real-Time Retraining (Week 4)
- [ ] Implement `RealtimeTrainer` with trigger detection
- [ ] Add async retraining (non-blocking)
- [ ] Test cooldown logic (prevent retrain spam)
- [ ] Add GPU memory checks before retrain
- [ ] Test under simulated volatility spike

### Phase 8.5: Daily Batch Training (Week 5)
- [ ] Implement scheduled batch retraining (03:00 UTC)
- [ ] Add checkpointing and versioning
- [ ] Create validation report (metrics, plots)
- [ ] Add Chronos-Bolt benchmark comparison
- [ ] Email/dashboard alerts on performance

### Phase 8.6: Ensemble Integration (Week 6)
- [ ] Add neural models to ensemble candidate_weights
- [ ] Implement regime detection logic
- [ ] Test dynamic model switching (GARCH → PatchTST → FeatureBased)
- [ ] Validate on multi-ticker panel (10 tickers)
- [ ] Measure ensemble RMSE ratio (target: <1.1x)

### Phase 8.7: Production Hardening (Week 7)
- [ ] Add error handling and fallbacks
- [ ] Implement model health checks
- [ ] Add monitoring and alerting
- [ ] Load testing (50 tickers, 1-hour updates)
- [ ] Documentation and runbooks
- [ ] Canary deployment (shadow mode for 1 week)

---

## Success Metrics

### Model Performance
- **RMSE Ratio**: <1.1x best single model (target from Phase 7.3)
- **Directional Accuracy**: >55% (profitable edge)
- **Sharpe Ratio**: >1.5 on validation set
- **Max Drawdown**: <15% on validation set

### Operational Metrics
- **Inference Latency**: <100ms per ticker (real-time requirement)
- **Training Time**: <10min for daily batch (all tickers)
- **GPU Memory**: <12GB peak usage (leave 4GB headroom)
- **Uptime**: >99.9% (critical for intraday trading)

### Business Metrics
- **PnL Improvement**: >10% vs current ensemble
- **Win Rate**: >52% (vs 50% random)
- **Profit Factor**: >1.5 (gross profit / gross loss)

---

## Risk Management

### Model Risk
- **Overfitting**: Use validation split, early stopping, regularization
- **Data Leakage**: Strict train/val/test splits, no look-ahead bias
- **Regime Shift**: Daily retraining, real-time regime detection
- **Model Degradation**: Continuous monitoring, auto-fallback to GARCH

### Operational Risk
- **GPU Failure**: Fallback to CPU mode (slower but functional)
- **Memory Overflow**: Pre-allocate memory, monitor usage, kill if OOM
- **Training Failure**: Auto-retry 3x, email alert, fallback to previous checkpoint
- **Data Quality**: Validation checks before training (missing data, outliers, drift)

### Trading Risk
- **Position Sizing**: GARCH-based volatility targeting (keep current system)
- **Stop Loss**: Adaptive stops using GARCH forecasts
- **Max Exposure**: Ensemble confidence-weighted position sizing
- **Circuit Breakers**: Halt trading if tracking error > 2x or GPU fails

---

## Monitoring Dashboard

### Real-Time Metrics (Update every 15min)
- GPU utilization, memory usage, temperature
- Inference latency per model
- Tracking error (1h rolling RMSE)
- Model confidence scores
- Active triggers (retrain alerts)

### Daily Reports (Email at 04:00 UTC)
- Training completion status
- Validation metrics (RMSE, sMAPE, DA)
- Chronos-Bolt benchmark comparison
- Feature importance changes
- Model version and checkpoint path

### Weekly Performance Review
- PnL attribution by model
- Win rate and profit factor
- Sharpe ratio trend
- Regime distribution
- Model selection frequency

---

## References

### Libraries
- **NeuralForecast**: https://github.com/Nixtla/neuralforecast
- **skforecast**: https://github.com/JoaquinAmatRodrigo/skforecast
- **XGBoost GPU**: https://xgboost.readthedocs.io/en/latest/gpu/index.html
- **Chronos**: https://github.com/amazon-science/chronos-forecasting

### Papers
- **PatchTST**: "A Time Series is Worth 64 Words" (2023)
- **NHITS**: "Neural Hierarchical Interpolation for Time Series" (2022)
- **Chronos**: "Chronos: Learning the Language of Time Series" (2024)

### Internal Docs
- [AGENT_DEV_CHECKLIST.md](AGENT_DEV_CHECKLIST.md) - Development standards
- [RUNTIME_GUARDRAILS.md](RUNTIME_GUARDRAILS.md) - Operational constraints
- [GPU_PARALLEL_RUNNER_CHECKLIST.md](GPU_PARALLEL_RUNNER_CHECKLIST.md) - GPU optimization
- [PHASE_7.3_COMPLETE.md](PHASE_7.3_COMPLETE.md) - GARCH integration (baseline)

---

**Status**: Ready to begin Phase 8.1 after multi-ticker validation (Phase 7.3)
**Next Action**: Validate GARCH ensemble on AAPL, MSFT, NVDA
**Timeline**: 7 weeks to production-ready neural forecasters
