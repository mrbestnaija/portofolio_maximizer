"""Tests for data preprocessing with vectorized transformations."""
import pytest
import numpy as np
import pandas as pd
from etl.preprocessor import Preprocessor

@pytest.fixture
def preprocessor():
    return Preprocessor(train_ratio=0.7, val_ratio=0.15)

@pytest.fixture
def sample_prices():
    dates = pd.date_range('2014-01-01', periods=100)
    return pd.Series(
        np.random.lognormal(0, 0.02, 100).cumprod() * 100,
        index=dates
    )

@pytest.fixture
def sample_data():
    dates = pd.date_range('2014-01-01', periods=100)
    return pd.DataFrame({
        'Close': np.random.lognormal(0, 0.02, 100).cumprod() * 100,
        'Volume': np.random.randint(1e6, 1e7, 100)
    }, index=dates)

def test_compute_returns(preprocessor, sample_prices):
    """Test vectorized log returns calculation."""
    returns = preprocessor.compute_returns(sample_prices)

    assert len(returns) == len(sample_prices) - 1
    assert np.all(np.abs(returns) < 1)  # Sanity check
    assert not returns.isna().any()

def test_normalize(preprocessor, sample_data):
    """Test z-score normalization."""
    normalized, stats = preprocessor.normalize(sample_data)

    # Check near-zero mean and unit variance
    assert np.allclose(normalized.mean(), 0, atol=1e-10)
    assert np.allclose(normalized.std(), 1, atol=1e-10)
    assert 'means' in stats and 'stds' in stats

def test_handle_missing(preprocessor):
    """Test forward-fill missing value handling."""
    data = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [10, 20, np.nan, 40, 50]
    })

    filled = preprocessor.handle_missing(data)
    assert not filled.isna().any().any()

def test_chronological_split(preprocessor, sample_data):
    """Test chronological split with no lookahead bias."""
    train, val, test = preprocessor.chronological_split(sample_data)

    # Check split ratios
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Check chronological order (no lookahead)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

def test_process_pipeline(preprocessor, sample_data):
    """Test full preprocessing pipeline."""
    result = preprocessor.process(sample_data)

    assert 'train' in result and 'val' in result and 'test' in result
    assert 'stats' in result
    assert len(result['train']) + len(result['val']) + len(result['test']) == len(sample_data)