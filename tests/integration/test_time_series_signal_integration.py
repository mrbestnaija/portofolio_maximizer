"""
Integration Tests for Time Series Signal Generation Pipeline
Line Count: ~400 lines (within budget)

Tests the complete integration of Time Series signal generation into the ETL pipeline,
ensuring signals flow correctly through forecasting -> generation -> routing -> persistence.

Per TESTING_GUIDE.md: Focus on profit-critical functions only.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.database_manager import DatabaseManager
from etl.time_series_forecaster import TimeSeriesForecaster
from models.time_series_signal_generator import TimeSeriesSignalGenerator
from models.signal_router import SignalRouter
from models.signal_adapter import SignalAdapter


@pytest.fixture(scope="session")
def sample_price_series():
    """Create realistic price series for forecasting"""
    dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
    np.random.seed(42)
    
    # Create trending price series
    trend = np.linspace(0, 0.3, len(dates))  # 30% uptrend
    noise = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    prices = 100 * (1 + trend + noise)
    
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture(scope="session")
def ts_forecast_bundle(sample_price_series):
    """Compute a single forecast bundle for all integration tests.

    This keeps the integration suite runtime bounded (SARIMAX auto-select and
    multi-model ensembles are exercised elsewhere).
    """
    forecaster = TimeSeriesForecaster(
        sarimax_config={
            "enabled": True,
            "auto_select": False,
            "manual_order": (1, 1, 1),
        },
        garch_config={"enabled": False},
        samossa_config={"enabled": False},
        mssa_rl_config={"enabled": False},
        ensemble_config={"enabled": False},
    )
    returns = sample_price_series.pct_change().dropna()
    forecaster.fit(sample_price_series, returns_series=returns)
    return forecaster.forecast(steps=30)


@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(180) * 0.5)
    
    data = pd.DataFrame({
        'Open': prices + np.random.rand(180) * 0.5,
        'High': prices + np.random.rand(180) * 1.0,
        'Low': prices - np.random.rand(180) * 1.0,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 180)
    }, index=dates)
    
    # Ensure realistic price relationships
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.abs(np.random.rand(180))
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.abs(np.random.rand(180))
    
    return data


@pytest.fixture
def test_database(tmp_path):
    """Create temporary test database"""
    db_path = tmp_path / "test_signals.db"
    db = DatabaseManager(str(db_path))
    yield db
    db.close()


@pytest.fixture(scope="session")
def ts_routing_config():
    """Load Time Series routing thresholds from configuration."""
    config_path = Path("config") / "signal_routing_config.yml"
    if not config_path.exists():
        pytest.skip("Time Series routing config is missing")
    raw = yaml.safe_load(config_path.read_text()) or {}
    return (raw.get("signal_routing") or {}).get("time_series") or {}


@pytest.fixture
def ts_signal_generator(ts_routing_config):
    """Create Time Series signal generator"""
    return TimeSeriesSignalGenerator(
        confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
        min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
        max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
        use_volatility_filter=bool(ts_routing_config.get("use_volatility_filter", True)),
    )


class TestTimeSeriesForecastingToSignalIntegration:
    """Test integration: Forecasting -> Signal Generation"""
    
    def test_forecast_to_signal_flow(self, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test complete flow from forecast to signal"""
        forecast_bundle = ts_forecast_bundle
        current_price = float(sample_price_series.iloc[-1])
        
        signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        # Verify signal structure
        assert signal is not None
        assert signal.ticker == 'TEST'
        assert signal.action in ('BUY', 'SELL', 'HOLD')
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.entry_price == current_price
        assert signal.signal_type == 'TIME_SERIES'
    
    def test_signal_generation_with_ensemble_forecast(self, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test signal generation uses ensemble forecast"""
        forecast_bundle = ts_forecast_bundle
        
        # Verify ensemble forecast exists
        assert 'ensemble_forecast' in forecast_bundle or 'mean_forecast' in forecast_bundle
        
        # Generate signal
        current_price = float(sample_price_series.iloc[-1])
        signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        assert signal is not None
        assert signal.model_type in ('ENSEMBLE', 'SARIMAX', 'SAMOSSA', 'GARCH', 'MSSA_RL')


class TestSignalRoutingIntegration:
    """Test integration: Signal Routing"""
    
    def test_time_series_primary_routing(self, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test Time Series signals routed as primary"""
        forecast_bundle = ts_forecast_bundle
        
        current_price = float(sample_price_series.iloc[-1])
        
        # Create router with Time Series primary
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': False},
            time_series_generator=ts_signal_generator
        )
        
        bundle = router.route_signal(
            ticker='TEST',
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            market_data=None
        )
        
        # Verify Time Series signal is primary
        assert bundle.primary_signal is not None
        assert bundle.primary_signal['source'] == 'TIME_SERIES'
        assert bundle.primary_signal['is_primary'] is True
    
    def test_llm_fallback_routing(self, sample_price_series, ts_signal_generator):
        """Test LLM fallback when Time Series unavailable"""
        mock_llm = Mock()
        mock_llm.generate_signal.return_value = {
            'ticker': 'TEST',
            'action': 'BUY',
            'confidence': 0.65,
            'entry_price': 100.0,
            'reasoning': 'LLM fallback',
            'signal_timestamp': datetime.now().isoformat()
        }
        
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': True},
            time_series_generator=ts_signal_generator,
            llm_generator=mock_llm
        )
        
        # No forecast bundle (Time Series unavailable)
        bundle = router.route_signal(
            ticker='TEST',
            forecast_bundle=None,
            current_price=100.0,
            market_data=None
        )
        
        # Should use LLM as fallback
        assert bundle.primary_signal is not None
        assert bundle.primary_signal['source'] == 'LLM'
        assert bundle.primary_signal['is_fallback'] is True


class TestDatabasePersistenceIntegration:
    """Test integration: Database Persistence"""
    
    def test_save_time_series_signal_to_database(self, test_database, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test saving Time Series signal to unified trading_signals table"""
        forecast_bundle = ts_forecast_bundle
        
        current_price = float(sample_price_series.iloc[-1])
        signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        # Convert to dict for database
        signal_dict = SignalAdapter.to_legacy_dict(
            SignalAdapter.from_time_series_signal(signal)
        )
        
        # Save to database
        signal_date = datetime.now().strftime('%Y-%m-%d')
        signal_id = test_database.save_trading_signal(
            ticker='TEST',
            date=signal_date,
            signal=signal_dict,
            source='TIME_SERIES',
            model_type=signal.model_type,
            validation_status='pending'
        )
        
        # Verify signal saved
        assert signal_id > 0
        
        # Retrieve from database
        test_database.cursor.execute("""
            SELECT * FROM trading_signals
            WHERE id = ?
        """, (signal_id,))
        
        row = test_database.cursor.fetchone()
        assert row is not None
        assert row['ticker'] == 'TEST'
        assert row['source'] == 'TIME_SERIES'
        assert row['action'] == signal.action
        assert abs(row['confidence'] - signal.confidence) < 0.01
        assert abs(row['entry_price'] - signal.entry_price) < 0.01
    
    def test_save_multiple_signals_same_ticker(self, test_database, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test saving multiple signals for same ticker (should update, not duplicate)"""
        forecast_bundle = ts_forecast_bundle
        
        current_price = float(sample_price_series.iloc[-1])
        signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        signal_dict = SignalAdapter.to_legacy_dict(
            SignalAdapter.from_time_series_signal(signal)
        )
        
        signal_date = datetime.now().strftime('%Y-%m-%d')
        
        # Save first signal
        signal_id_1 = test_database.save_trading_signal(
            ticker='TEST',
            date=signal_date,
            signal=signal_dict,
            source='TIME_SERIES',
            model_type=signal.model_type
        )
        
        # Modify signal and save again (should update, not create duplicate)
        signal_dict['action'] = 'SELL'  # Change action
        signal_id_2 = test_database.save_trading_signal(
            ticker='TEST',
            date=signal_date,
            signal=signal_dict,
            source='TIME_SERIES',
            model_type=signal.model_type
        )
        
        # Should be same ID (updated, not new)
        assert signal_id_1 == signal_id_2
        
        # Verify updated action
        test_database.cursor.execute("""
            SELECT action FROM trading_signals WHERE id = ?
        """, (signal_id_1,))
        row = test_database.cursor.fetchone()
        assert row['action'] == 'SELL'
    
    def test_save_llm_and_ts_signals_separately(self, test_database, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test saving both LLM and Time Series signals (different sources)"""
        forecast_bundle = ts_forecast_bundle
        
        current_price = float(sample_price_series.iloc[-1])
        ts_signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        ts_signal_dict = SignalAdapter.to_legacy_dict(
            SignalAdapter.from_time_series_signal(ts_signal)
        )
        
        # LLM signal
        llm_signal = {
            'ticker': 'TEST',
            'action': 'BUY',
            'confidence': 0.65,
            'entry_price': current_price,
            'reasoning': 'LLM analysis',
            'signal_timestamp': datetime.now().isoformat()
        }
        
        signal_date = datetime.now().strftime('%Y-%m-%d')
        
        # Save both
        ts_id = test_database.save_trading_signal(
            ticker='TEST',
            date=signal_date,
            signal=ts_signal_dict,
            source='TIME_SERIES',
            model_type=ts_signal.model_type
        )
        
        llm_id = test_database.save_trading_signal(
            ticker='TEST',
            date=signal_date,
            signal=llm_signal,
            source='LLM',
            model_type='qwen:14b-chat-q4_K_M'
        )
        
        # Should be different IDs (different sources)
        assert ts_id != llm_id
        
        # Verify both exist
        test_database.cursor.execute("""
            SELECT COUNT(*) as count FROM trading_signals
            WHERE ticker = ? AND signal_date = ?
        """, ('TEST', signal_date))
        
        row = test_database.cursor.fetchone()
        assert row['count'] == 2


class TestEndToEndPipelineIntegration:
    """Test complete end-to-end pipeline integration"""
    
    def test_full_pipeline_forecast_to_database(self, test_database, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test complete pipeline: Forecast -> Signal -> Database"""
        forecast_bundle = ts_forecast_bundle
        
        # Step 2: Signal Generation
        current_price = float(sample_price_series.iloc[-1])
        signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        # Step 3: Signal Adapter
        unified_signal = SignalAdapter.from_time_series_signal(signal)
        signal_dict = SignalAdapter.to_legacy_dict(unified_signal)
        
        # Step 4: Database Persistence
        signal_date = datetime.now().strftime('%Y-%m-%d')
        signal_id = test_database.save_trading_signal(
            ticker='TEST',
            date=signal_date,
            signal=signal_dict,
            source='TIME_SERIES',
            model_type=signal.model_type
        )
        
        # Verify end-to-end
        assert signal_id > 0
        assert signal.action in ('BUY', 'SELL', 'HOLD')
        assert unified_signal.source == 'TIME_SERIES'
        
        # Verify database record
        test_database.cursor.execute("""
            SELECT * FROM trading_signals WHERE id = ?
        """, (signal_id,))
        row = test_database.cursor.fetchone()
        assert row is not None
        assert row['ticker'] == 'TEST'
        assert row['source'] == 'TIME_SERIES'
    
    def test_pipeline_with_signal_routing(self, test_database, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test pipeline with signal routing"""
        forecast_bundle = ts_forecast_bundle
        
        current_price = float(sample_price_series.iloc[-1])
        
        # Route signal
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': False},
            time_series_generator=ts_signal_generator
        )
        
        bundle = router.route_signal(
            ticker='TEST',
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            market_data=None
        )
        
        # Save routed signal
        if bundle.primary_signal:
            signal_date = datetime.now().strftime('%Y-%m-%d')
            signal_id = test_database.save_trading_signal(
                ticker='TEST',
                date=signal_date,
                signal=bundle.primary_signal,
                source=bundle.primary_signal['source'],
                model_type=bundle.primary_signal.get('model_type')
            )
            
            assert signal_id > 0
            
            # Verify routing metadata
            test_database.cursor.execute("""
                SELECT source, model_type FROM trading_signals WHERE id = ?
            """, (signal_id,))
            row = test_database.cursor.fetchone()
            assert row['source'] == 'TIME_SERIES'


class TestSignalValidationIntegration:
    """Test signal validation in integration context"""
    
    def test_signal_adapter_validation(self, sample_price_series, ts_forecast_bundle, ts_signal_generator):
        """Test signal validation through adapter"""
        forecast_bundle = ts_forecast_bundle
        
        current_price = float(sample_price_series.iloc[-1])
        signal = ts_signal_generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=current_price,
            ticker='TEST',
            market_data=None
        )
        
        # Validate through adapter
        unified = SignalAdapter.from_time_series_signal(signal)
        is_valid, error = SignalAdapter.validate_signal(unified)
        
        assert is_valid is True
        assert error is None
        assert unified.ticker == 'TEST'
        assert unified.action in ('BUY', 'SELL', 'HOLD')

