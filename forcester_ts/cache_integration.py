"""
Integration helpers for parameter caching in forecasters.

Provides decorators and utilities to automatically cache learned parameters
and use warm-start initialization.
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .parameter_cache import get_parameter_cache

logger = logging.getLogger(__name__)


def with_parameter_caching(model_name: str):
    """
    Decorator to add automatic parameter caching to forecaster.fit() methods.

    Usage:
        @with_parameter_caching('sarimax')
        def fit(self, series: pd.Series, **kwargs):
            # ... fit logic ...
            return self

    The decorator will:
    1. Check cache for warm-start parameters before fitting
    2. Save learned parameters after successful fit
    3. Track performance metrics for Bayesian updates
    """
    def decorator(fit_method: Callable) -> Callable:
        @wraps(fit_method)
        def wrapper(self, series: pd.Series, **kwargs):
            cache = get_parameter_cache()

            # Extract ticker if available
            ticker = kwargs.get('ticker') or getattr(series, 'name', 'UNKNOWN')
            if isinstance(ticker, tuple):
                ticker = ticker[0] if len(ticker) > 0 else 'UNKNOWN'
            ticker = str(ticker).upper()

            # Check if retraining needed
            should_retrain, reason = cache.should_retrain(
                ticker=ticker,
                model=model_name,
                current_series=series,
                max_cache_age_days=7,
                min_new_data_points=20
            )

            if not should_retrain:
                logger.info(
                    f"Skipping {model_name} training for {ticker}: {reason}"
                )
                # Load cached model state if available
                cached = cache.load(ticker, model_name)
                if cached:
                    _restore_cached_state(self, cached['parameters'])
                    return self

            # Get warm-start parameters
            warm_start = cache.get_warm_start_parameters(
                ticker=ticker,
                model=model_name,
                use_bayesian=True
            )

            if warm_start:
                logger.info(
                    f"Using warm-start for {model_name}/{ticker}: "
                    f"source={warm_start.get('source')}, "
                    f"confidence={warm_start.get('confidence', 0):.2f}"
                )
                _apply_warm_start(self, warm_start)

            # Call original fit method
            result = fit_method(self, series, **kwargs)

            # Extract learned parameters
            parameters = _extract_parameters(self, model_name)
            if not parameters:
                return result

            # Extract performance metrics
            performance = _extract_performance(self, model_name)

            # Save to cache
            cache.save(
                ticker=ticker,
                model=model_name,
                parameters=parameters,
                performance=performance,
                series=series
            )

            logger.debug(
                f"Cached parameters for {model_name}/{ticker}: "
                f"{list(parameters.keys())}"
            )

            return result

        return wrapper
    return decorator


def _apply_warm_start(forecaster: Any, warm_start: Dict[str, Any]) -> None:
    """Apply warm-start parameters to forecaster instance."""
    # SARIMAX warm-start
    if hasattr(forecaster, 'warm_start_order'):
        if 'order' in warm_start:
            forecaster.warm_start_order = warm_start['order']
            logger.debug(f"Applied warm-start order: {warm_start['order']}")

    # Apply hyperparameters
    if 'hyperparameters' in warm_start:
        for key, value in warm_start['hyperparameters'].items():
            if hasattr(forecaster, key):
                setattr(forecaster, key, value)


def _restore_cached_state(forecaster: Any, parameters: Dict[str, Any]) -> None:
    """Restore forecaster state from cached parameters."""
    # SARIMAX
    if 'order' in parameters:
        forecaster.best_order = tuple(parameters['order'])
    if 'seasonal_order' in parameters:
        forecaster.best_seasonal_order = tuple(parameters['seasonal_order'])

    # GARCH
    if 'p' in parameters and 'q' in parameters:
        forecaster.p = parameters['p']
        forecaster.q = parameters['q']

    # SAMoSSA
    if 'window_length' in parameters:
        forecaster.config.window_length = parameters['window_length']
    if 'n_components' in parameters:
        forecaster.config.n_components = parameters['n_components']

    # MSSA-RL
    if 'rank' in parameters:
        forecaster.config.rank = parameters['rank']


def _extract_parameters(forecaster: Any, model_name: str) -> Dict[str, Any]:
    """Extract learned parameters from fitted forecaster."""
    params = {}

    if model_name == 'sarimax':
        if hasattr(forecaster, 'best_order'):
            params['order'] = list(forecaster.best_order)
        if hasattr(forecaster, 'best_seasonal_order'):
            params['seasonal_order'] = list(forecaster.best_seasonal_order)
        if hasattr(forecaster, 'trend'):
            params['trend'] = forecaster.trend

    elif model_name == 'samossa':
        if hasattr(forecaster, 'config'):
            params['window_length'] = forecaster.config.window_length
            params['n_components'] = forecaster.config.n_components
        if hasattr(forecaster, '_explained_variance_ratio'):
            params['explained_variance'] = float(forecaster._explained_variance_ratio)

    elif model_name == 'mssa_rl':
        if hasattr(forecaster, 'config'):
            params['rank'] = forecaster.config.rank
            params['window_length'] = forecaster.config.window_length
        if hasattr(forecaster, '_change_points'):
            params['n_change_points'] = len(forecaster._change_points or [])

    elif model_name == 'garch':
        if hasattr(forecaster, 'p'):
            params['p'] = forecaster.p
        if hasattr(forecaster, 'q'):
            params['q'] = forecaster.q

    return params


def _extract_performance(forecaster: Any, model_name: str) -> Dict[str, float]:
    """Extract performance metrics from fitted forecaster."""
    perf = {}

    if model_name == 'sarimax':
        if hasattr(forecaster, 'fitted_model'):
            if hasattr(forecaster.fitted_model, 'aic'):
                perf['aic'] = float(forecaster.fitted_model.aic)
            if hasattr(forecaster.fitted_model, 'bic'):
                perf['bic'] = float(forecaster.fitted_model.bic)

    elif model_name == 'garch':
        if hasattr(forecaster, 'fitted_model'):
            if hasattr(forecaster.fitted_model, 'aic'):
                perf['aic'] = float(forecaster.fitted_model.aic)

    # RMSE would need validation set to compute
    # Can be added in pipeline stage

    return perf


def check_cache_freshness(
    ticker: str,
    model: str,
    series: pd.Series,
    max_age_days: int = 7
) -> bool:
    """
    Check if cached parameters are fresh enough to skip retraining.

    Args:
        ticker: Ticker symbol
        model: Model name
        series: Current data
        max_age_days: Maximum age to consider fresh

    Returns:
        True if cache is fresh and retraining can be skipped
    """
    cache = get_parameter_cache()
    should_retrain, reason = cache.should_retrain(
        ticker=ticker,
        model=model,
        current_series=series,
        max_cache_age_days=max_age_days
    )
    return not should_retrain


def invalidate_cache_for_ticker(ticker: str) -> None:
    """
    Invalidate all cached parameters for a ticker.

    Useful when fundamental data characteristics change (e.g., stock split).
    """
    cache = get_parameter_cache()
    for model in ['sarimax', 'samossa', 'mssa_rl', 'garch']:
        key = f"{ticker}_{model}"
        if key in cache._memory_cache:
            cache._memory_cache[key] = []
            logger.info(f"Invalidated cache for {ticker}/{model}")
