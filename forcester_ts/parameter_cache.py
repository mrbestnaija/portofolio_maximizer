"""
Bayesian parameter caching system for warm-start training.

Maintains a history of learned parameters with performance metrics,
computes Bayesian priors for new training runs, and supports incremental
learning on unseen data.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParameterRecord:
    """Single record of learned parameters with performance metrics."""
    model: str  # 'sarimax', 'samossa', 'mssa_rl', 'garch'
    ticker: str
    date: str  # ISO format
    parameters: Dict[str, Any]
    performance: Dict[str, float]  # {'aic': ..., 'bic': ..., 'rmse': ...}
    series_length: int
    data_hash: str  # Hash of training data to detect duplicates


@dataclass
class BayesianPrior:
    """Bayesian prior distribution for model parameters."""
    order: Optional[Tuple[int, ...]] = None
    order_probabilities: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    hyperparameters: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # 0-1 score for prior strength
    n_observations: int = 0  # Number of historical observations
    last_updated: Optional[str] = None


class ParameterCache:
    """
    Manages persistent cache of learned model parameters with Bayesian updates.

    Features:
    - Stores historical parameters with performance metrics
    - Computes Bayesian priors from historical best performers
    - Detects unseen data via hashing to trigger retraining
    - Supports warm-start initialization
    """

    def __init__(self, cache_dir: str = "data/model_params"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, List[ParameterRecord]] = defaultdict(list)
        self._load_from_disk()

    def _get_cache_path(self, ticker: str, model: str) -> Path:
        """Get path to cache file for ticker/model combination."""
        return self.cache_dir / f"{ticker}_{model}_params.json"

    def _load_from_disk(self) -> None:
        """Load all cached parameters from disk into memory."""
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*_params.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    key = f"{data[0]['ticker']}_{data[0]['model']}"
                    self._memory_cache[key] = [
                        ParameterRecord(**record) for record in data
                    ]
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}")

    def _compute_data_hash(self, series: pd.Series) -> str:
        """
        Compute hash of series to detect unseen data.

        Uses length, date range, and sample statistics to create fingerprint.
        """
        try:
            fingerprint = {
                'length': len(series),
                'start': str(series.index[0]) if len(series) > 0 else '',
                'end': str(series.index[-1]) if len(series) > 0 else '',
                'mean': float(series.mean()),
                'std': float(series.std()),
                # Sample hash: hash of first/last 10 values
                'sample': hash(tuple(series.iloc[:10].values) + tuple(series.iloc[-10:].values))
            }
            return str(hash(frozenset(fingerprint.items())))
        except Exception:
            return ""

    def has_seen_data(
        self,
        ticker: str,
        model: str,
        series: pd.Series,
        max_age_days: int = 30
    ) -> bool:
        """
        Check if we've trained on this data recently.

        Args:
            ticker: Ticker symbol
            model: Model name
            series: Time series data
            max_age_days: Consider cached parameters stale after this many days

        Returns:
            True if we have recent cached parameters for this data
        """
        key = f"{ticker}_{model}"
        if key not in self._memory_cache or not self._memory_cache[key]:
            return False

        data_hash = self._compute_data_hash(series)
        if not data_hash:
            return False

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        for record in self._memory_cache[key]:
            if record.data_hash == data_hash:
                record_date = datetime.fromisoformat(record.date)
                if record_date >= cutoff_date:
                    logger.info(
                        f"Found cached parameters for {ticker}/{model} "
                        f"(age: {(datetime.now() - record_date).days} days)"
                    )
                    return True

        return False

    def save(
        self,
        ticker: str,
        model: str,
        parameters: Dict[str, Any],
        performance: Dict[str, float],
        series: pd.Series
    ) -> None:
        """
        Save learned parameters with performance metrics.

        Args:
            ticker: Ticker symbol
            model: Model name ('sarimax', 'samossa', etc.)
            parameters: Learned parameters (e.g., {'order': (2,1,1), ...})
            performance: Performance metrics (e.g., {'aic': 1000, 'rmse': 2.1})
            series: Training data used
        """
        record = ParameterRecord(
            model=model,
            ticker=ticker,
            date=datetime.now().isoformat(),
            parameters=parameters,
            performance=performance,
            series_length=len(series),
            data_hash=self._compute_data_hash(series)
        )

        key = f"{ticker}_{model}"
        self._memory_cache[key].append(record)

        # Persist to disk
        cache_path = self._get_cache_path(ticker, model)
        try:
            records_list = [asdict(r) for r in self._memory_cache[key]]
            with open(cache_path, 'w') as f:
                json.dump(records_list, f, indent=2)
            logger.debug(f"Saved parameters to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")

    def load(
        self,
        ticker: str,
        model: str,
        max_age_days: int = 90
    ) -> Optional[Dict[str, Any]]:
        """
        Load most recent cached parameters within age limit.

        Returns:
            Dictionary with 'parameters' and 'performance' keys, or None
        """
        key = f"{ticker}_{model}"
        if key not in self._memory_cache or not self._memory_cache[key]:
            return None

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        recent_records = [
            r for r in self._memory_cache[key]
            if datetime.fromisoformat(r.date) >= cutoff_date
        ]

        if not recent_records:
            return None

        # Return most recent
        recent_records.sort(key=lambda r: r.date, reverse=True)
        best = recent_records[0]

        return {
            'parameters': best.parameters,
            'performance': best.performance,
            'date': best.date,
            'series_length': best.series_length
        }

    def compute_bayesian_prior(
        self,
        ticker: str,
        model: str,
        metric: str = 'rmse',
        lookback_days: int = 180,
        min_observations: int = 3
    ) -> Optional[BayesianPrior]:
        """
        Compute Bayesian prior from historical parameters weighted by performance.

        Args:
            ticker: Ticker symbol
            model: Model name
            metric: Performance metric to optimize ('aic', 'rmse', etc.)
            lookback_days: Consider history within this window
            min_observations: Minimum historical observations required

        Returns:
            BayesianPrior object with order probabilities and hyperparameters
        """
        key = f"{ticker}_{model}"
        if key not in self._memory_cache or not self._memory_cache[key]:
            return None

        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_records = [
            r for r in self._memory_cache[key]
            if datetime.fromisoformat(r.date) >= cutoff_date
            and metric in r.performance
        ]

        if len(recent_records) < min_observations:
            logger.debug(
                f"Insufficient history for Bayesian prior: {len(recent_records)} < {min_observations}"
            )
            return None

        # Extract orders and performance
        order_counts = defaultdict(int)
        order_performance = defaultdict(list)

        for record in recent_records:
            # Try to extract order (works for SARIMAX, GARCH)
            order = None
            if 'order' in record.parameters:
                order = tuple(record.parameters['order'])
            elif 'p' in record.parameters and 'q' in record.parameters:
                # GARCH format
                order = (record.parameters['p'], record.parameters['q'])

            if order:
                order_counts[order] += 1
                order_performance[order].append(record.performance[metric])

        if not order_counts:
            return None

        # Compute probability for each order based on:
        # 1. Frequency of appearance
        # 2. Performance (lower is better for AIC/RMSE)
        order_scores = {}
        total_appearances = sum(order_counts.values())

        for order, count in order_counts.items():
            freq_weight = count / total_appearances

            # Performance weight (inverse of metric, normalized)
            perf_values = order_performance[order]
            avg_perf = np.mean(perf_values)
            if metric.lower() in ['aic', 'bic', 'rmse', 'mse']:
                # Lower is better
                perf_weight = 1.0 / (1.0 + avg_perf)
            else:
                # Higher is better
                perf_weight = avg_perf

            # Combined score
            order_scores[order] = freq_weight * perf_weight

        # Normalize to probabilities
        total_score = sum(order_scores.values())
        order_probabilities = {
            order: score / total_score
            for order, score in order_scores.items()
        }

        # Select most likely order
        best_order = max(order_probabilities.items(), key=lambda x: x[1])[0]
        confidence = order_probabilities[best_order]

        # Compute hyperparameter means from recent good performers
        # (top 30% by performance)
        perf_threshold = np.percentile(
            [r.performance[metric] for r in recent_records],
            30 if metric.lower() in ['aic', 'bic', 'rmse'] else 70
        )

        good_performers = [
            r for r in recent_records
            if (metric.lower() in ['aic', 'bic', 'rmse'] and r.performance[metric] <= perf_threshold) or
               (metric.lower() not in ['aic', 'bic', 'rmse'] and r.performance[metric] >= perf_threshold)
        ]

        hyperparameters = {}
        if good_performers:
            # Average hyperparameters from good performers
            param_keys = set()
            for r in good_performers:
                param_keys.update(r.parameters.keys())

            for key in param_keys:
                if key not in ['order', 'seasonal_order']:  # Skip structural params
                    values = [
                        r.parameters[key] for r in good_performers
                        if key in r.parameters and isinstance(r.parameters[key], (int, float))
                    ]
                    if values:
                        hyperparameters[key] = float(np.mean(values))

        prior = BayesianPrior(
            order=best_order,
            order_probabilities=order_probabilities,
            hyperparameters=hyperparameters,
            confidence=confidence,
            n_observations=len(recent_records),
            last_updated=datetime.now().isoformat()
        )

        logger.info(
            f"Computed Bayesian prior for {ticker}/{model}: "
            f"order={best_order}, confidence={confidence:.2f}, "
            f"n_obs={len(recent_records)}"
        )

        return prior

    def get_warm_start_parameters(
        self,
        ticker: str,
        model: str,
        use_bayesian: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get parameters for warm-start initialization.

        Args:
            ticker: Ticker symbol
            model: Model name
            use_bayesian: Use Bayesian prior if True, else most recent

        Returns:
            Dictionary with warm-start parameters or None
        """
        if use_bayesian:
            prior = self.compute_bayesian_prior(ticker, model)
            if prior:
                return {
                    'order': prior.order,
                    'hyperparameters': prior.hyperparameters,
                    'confidence': prior.confidence,
                    'source': 'bayesian_prior'
                }

        # Fallback to most recent
        recent = self.load(ticker, model)
        if recent:
            return {
                **recent['parameters'],
                'source': 'most_recent'
            }

        return None

    def should_retrain(
        self,
        ticker: str,
        model: str,
        current_series: pd.Series,
        max_cache_age_days: int = 7,
        min_new_data_points: int = 20
    ) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on cache and data freshness.

        Args:
            ticker: Ticker symbol
            model: Model name
            current_series: Latest available data
            max_cache_age_days: Retrain if cached parameters older than this
            min_new_data_points: Retrain if this many new points since last train

        Returns:
            (should_retrain: bool, reason: str)
        """
        key = f"{ticker}_{model}"

        # No cache -> must train
        if key not in self._memory_cache or not self._memory_cache[key]:
            return True, "no_cached_parameters"

        # Get most recent record
        recent_records = sorted(
            self._memory_cache[key],
            key=lambda r: r.date,
            reverse=True
        )
        latest_record = recent_records[0]

        # Check age
        record_date = datetime.fromisoformat(latest_record.date)
        age_days = (datetime.now() - record_date).days
        if age_days > max_cache_age_days:
            return True, f"cache_too_old_{age_days}d"

        # Check if new data available
        new_points = len(current_series) - latest_record.series_length
        if new_points >= min_new_data_points:
            return True, f"new_data_{new_points}_points"

        # Check if data characteristics changed significantly
        current_hash = self._compute_data_hash(current_series)
        if current_hash != latest_record.data_hash:
            return True, "data_distribution_changed"

        return False, "cache_valid"

    def clear_old_entries(self, max_age_days: int = 365, max_entries_per_key: int = 50) -> int:
        """
        Remove old cache entries to prevent unbounded growth.

        Args:
            max_age_days: Remove entries older than this
            max_entries_per_key: Keep at most this many entries per ticker/model

        Returns:
            Number of entries removed
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0

        for key in list(self._memory_cache.keys()):
            # Remove old entries
            original_len = len(self._memory_cache[key])
            self._memory_cache[key] = [
                r for r in self._memory_cache[key]
                if datetime.fromisoformat(r.date) >= cutoff_date
            ]

            # Keep most recent entries if still too many
            if len(self._memory_cache[key]) > max_entries_per_key:
                self._memory_cache[key].sort(key=lambda r: r.date, reverse=True)
                self._memory_cache[key] = self._memory_cache[key][:max_entries_per_key]

            removed_count += original_len - len(self._memory_cache[key])

            # Persist to disk
            ticker, model = key.rsplit('_', 1)
            cache_path = self._get_cache_path(ticker, model)
            try:
                records_list = [asdict(r) for r in self._memory_cache[key]]
                with open(cache_path, 'w') as f:
                    json.dump(records_list, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update cache {cache_path}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache entries")

        return removed_count


# Global cache instance
_global_cache: Optional[ParameterCache] = None


def get_parameter_cache() -> ParameterCache:
    """Get or create global parameter cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ParameterCache()
    return _global_cache
