"""Data storage with efficient time series persistence.

Mathematical Foundation:
- Timestamp indexing: t_i  [t_0, t_n] with monotonic ordering
- Data partitioning: Split by (stage, symbol, date_range)
- Compression: Parquet format for columnar efficiency
- Atomic writes: temp -> rename pattern for consistency

Success Criteria:
- <100ms read for single symbol/year
- <50MB storage per symbol/year
- No data corruption on concurrent access
"""
import inspect
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta

from etl.time_series_cv import TimeSeriesCrossValidator

logger = logging.getLogger(__name__)

class SplitResult(dict):
    """Dictionary with backward-compatible aliases for train/test keys."""

    _ALIASES = {'train': 'training', 'test': 'testing'}

    def __getitem__(self, key):
        return super().__getitem__(self._ALIASES.get(key, key))

    def __contains__(self, key):
        return super().__contains__(self._ALIASES.get(key, key))

    def get(self, key, default=None):
        return super().get(self._ALIASES.get(key, key), default)


class DataStorage:
    """Efficient time series data persistence."""

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self):
        """Create storage structure."""
        stages = ['raw', 'processed', 'training', 'validation', 'testing']
        for stage in stages:
            (self.base_path / stage).mkdir(parents=True, exist_ok=True)

    def save(self, data: pd.DataFrame, stage: str, symbol: str,
             metadata: Optional[Dict] = None, run_id: Optional[str] = None) -> Path:
        """Save time series data with timestamp indexing.
        
        Args:
            data: DataFrame with DatetimeIndex
            stage: Data stage (raw, processed, training, etc.)
            symbol: Symbol identifier (ticker, dataset name, etc.)
            metadata: Optional metadata dictionary to persist alongside data
            run_id: Optional run identifier to prevent overwrites (defaults to timestamp)
            
        Returns:
            Path to saved parquet file
        """
        # Validate timestamp index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        # Sort by timestamp (vectorized)
        data = data.sort_index()

        # Generate path: stage/symbol_YYYYMMDD_HHMMSS[_runid].parquet
        # Include timestamp to prevent silent overwrites during multiple runs
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id:
            filename = f"{symbol}_{timestamp_str}_{run_id}.parquet"
        else:
            filename = f"{symbol}_{timestamp_str}.parquet"
        filepath = self.base_path / stage / filename

        # Atomic write: temp -> rename (preserve frequency)
        temp_path = filepath.with_suffix('.tmp')
        # Store frequency info in metadata if it exists
        freq_metadata = {'freq': str(data.index.freq)} if data.index.freq else {}
        combined_metadata = {**(metadata or {}), **freq_metadata}
        data.to_parquet(temp_path, compression='snappy', index=True)
        temp_path.rename(filepath)

        # Persist metadata alongside parquet for cache introspection
        if combined_metadata:
            safe_metadata: Dict[str, Any] = {}
            for key, value in combined_metadata.items():
                if isinstance(value, (datetime, pd.Timestamp)):
                    safe_metadata[key] = pd.Timestamp(value).isoformat()
                else:
                    safe_metadata[key] = value

            safe_metadata.setdefault('saved_at', datetime.now().isoformat())
            safe_metadata.setdefault('rows', len(data))
            safe_metadata.setdefault('run_id', run_id)
            safe_metadata.setdefault('data_source', metadata.get('data_source') if metadata else None)
            safe_metadata.setdefault('execution_mode', metadata.get('execution_mode') if metadata else None)
            # Store config hash if provided for troubleshooting
            if metadata and 'config_hash' in metadata:
                safe_metadata['config_hash'] = metadata['config_hash']

            metadata_path = filepath.with_suffix('.meta.json')
            metadata_tmp = metadata_path.with_suffix('.tmp')
            with open(metadata_tmp, 'w') as f:
                json.dump(safe_metadata, f, indent=2)
            if metadata_path.exists():
                metadata_path.unlink()
            metadata_tmp.rename(metadata_path)

        logger.info(f"Saved {len(data)} rows to {filepath}")
        return filepath

    def load(self, stage: str, symbol: str,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> pd.DataFrame:
        """Load time series data with date filtering."""
        # Find matching files (vectorized glob)
        pattern = f"{symbol}_*.parquet"
        files = sorted((self.base_path / stage).glob(pattern))

        if not files:
            raise FileNotFoundError(f"No data for {symbol} in {stage}")

        # Load and concatenate (vectorized)
        dfs = [pd.read_parquet(f) for f in files]
        data = pd.concat(dfs).sort_index()

        # Preserve frequency if not present (parquet doesn't store it)
        if data.index.freq is None and len(data) > 1:
            data.index.freq = pd.infer_freq(data.index)

        # Filter by date range (vectorized boolean indexing)
        if start_date:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data.index <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {len(data)} rows for {symbol}")
        return data

    def cleanup_old_files(self, stage: str, retention_days: int = 7):
        """Remove files older than retention period (vectorized).

        Args:
            stage: Data stage to clean
            retention_days: Keep files modified within this many days
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        stage_path = self.base_path / stage

        # Vectorized file operations: get all files with timestamps
        files = np.array(list(stage_path.glob("*.parquet")))
        if len(files) == 0:
            return

        # Vectorized timestamp extraction and comparison
        mtimes = np.array([datetime.fromtimestamp(f.stat().st_mtime) for f in files])
        old_mask = mtimes < cutoff
        old_files = files[old_mask]

        # Atomic deletion
        for f in old_files:
            f.unlink()
            logger.info(f"Deleted old file: {f}")

        if len(old_files) > 0:
            logger.info(f"Cleaned {len(old_files)} files from {stage}")

    @staticmethod
    def train_validation_test_split(data: pd.DataFrame,
                                    train_ratio: float = 0.7,
                                    val_ratio: float = 0.15,
                                    use_cv: bool = False,
                                    n_splits: int = 5,
                                    test_size: float = 0.15,
                                    gap: int = 0,
                                    expanding_window: bool = True) -> Dict[str, pd.DataFrame]:
        """Data splitting with backward-compatible k-fold cross-validation.

        BACKWARD COMPATIBLE: Default behavior (use_cv=False) unchanged.
        NEW: Set use_cv=True for production (prevents training/validation disparity).

        Mathematical Foundation:
        - Simple split (use_cv=False, DEFAULT for backward compatibility):
          * Temporal ordering: t_train < t_val < t_test
          * Split ratios: train=70%, val=15%, test=15%
          * LIMITATION: Training/validation disparity across time index

        - Cross-validation split (use_cv=True, RECOMMENDED):
          * k-fold moving window with test set isolation
          * Prevents data memory decay
          * Ensures representative sampling across entire distribution
          * Test set (15%) completely isolated
          * CV folds cover remaining 85% with moving window

        Args:
            data: Time series data with DatetimeIndex
            train_ratio: Training set proportion (used if use_cv=False)
            val_ratio: Validation set proportion (used if use_cv=False)
            use_cv: If True, use k-fold cross-validation. Default=False (backward compatible)
            n_splits: Number of CV folds (default=5, only used if use_cv=True)
            test_size: Proportion reserved for isolated test set when use_cv=True
            gap: Gap between train and validation windows for CV folds
            expanding_window: If True, use expanding window; otherwise rolling window

        Returns:
            Dictionary with split DataFrames.
            - If use_cv=False: {'training', 'validation', 'testing'} (BACKWARD COMPATIBLE)
            - If use_cv=True: {'cv_folds': List[Dict], 'testing': DataFrame,
                               'n_splits': int, 'split_type': str}
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex for chronological split")

        # Sort by timestamp (vectorized)
        data = data.sort_index()
        has_duplicate_index = bool(getattr(data.index, "has_duplicates", False))

        def _unique_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
            idx = frame.index
            if not isinstance(idx, pd.DatetimeIndex):
                return pd.DatetimeIndex([])
            unique = pd.DatetimeIndex(pd.unique(idx))
            return unique.sort_values()

        def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.iloc[0:0].copy()

        if use_cv:
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1 when using cross-validation")
            if gap < 0:
                raise ValueError("gap must be non-negative when using cross-validation")
            if n_splits < 2:
                raise ValueError("n_splits must be at least 2 for cross-validation")

            if has_duplicate_index:
                # Multi-asset frames commonly share dates across tickers. When the index contains
                # duplicates, use date-based splits (not row-count splits) so train/val/test are
                # separated by time boundaries and do not overlap by date.
                dates = _unique_dates(data)
                n_dates = len(dates)
                if n_dates == 0:
                    empty = _empty_like(data)
                    return {
                        "cv_folds": [],
                        "testing": empty,
                        "n_splits": n_splits,
                        "split_type": "cross_validation",
                        "train": empty,
                        "validation": empty,
                        "test": empty,
                        "training": empty,
                        "testing": empty,
                    }

                test_start = int(n_dates * (1 - test_size))
                test_start = max(1, min(test_start, n_dates))
                test_dates = dates[test_start:]
                cv_size = test_start
                fold_size = cv_size // (n_splits + 1)

                fold_splits = []
                for fold_id in range(n_splits):
                    val_start = (fold_id + 1) * fold_size
                    val_end = min(val_start + fold_size, cv_size)
                    if expanding_window:
                        train_start = 0
                        train_end = val_start
                    else:
                        train_start = max(0, val_start - fold_size)
                        train_end = val_start

                    if gap > 0:
                        train_end = max(train_start, train_end - gap)

                    if train_end <= train_start or val_end <= val_start:
                        continue

                    train_dates = dates[train_start:train_end]
                    val_dates = dates[val_start:val_end]
                    train_df = data.loc[data.index.isin(train_dates)].copy()
                    val_df = data.loc[data.index.isin(val_dates)].copy()

                    if train_df.empty or val_df.empty:
                        continue

                    fold_splits.append(
                        {
                            "fold_id": fold_id,
                            "train": train_df,
                            "validation": val_df,
                        }
                    )

                test_df = data.loc[data.index.isin(test_dates)].copy()
                result: Dict[str, Any] = {
                    "cv_folds": fold_splits,
                    "testing": test_df,
                    "n_splits": n_splits,
                    "split_type": "cross_validation",
                }
                if fold_splits:
                    result["train"] = fold_splits[0]["train"]
                    result["validation"] = fold_splits[0]["validation"]
                else:
                    result["train"] = _empty_like(data)
                    result["validation"] = _empty_like(data)
                result["test"] = test_df
                result["training"] = result["train"]
                result["testing"] = result["test"]
                logger.info(
                    "CV Split (date-based): %s folds, unique_dates=%s, test_rows=%s",
                    len(fold_splits),
                    n_dates,
                    len(test_df),
                )
                return result

            # Use k-fold cross-validation (NEW FEATURE)
            cv_splitter = TimeSeriesCrossValidator(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap,
                expanding_window=expanding_window
            )

            cv_folds, test_indices = cv_splitter.split(data)
            test_data = data.iloc[test_indices].copy()

            # Extract fold data
            fold_splits = []
            for fold in cv_folds:
                fold_data = cv_splitter.get_fold_data(data, fold)
                fold_splits.append({
                    'fold_id': fold.fold_id,
                    'train': fold_data['train'],
                    'validation': fold_data['validation']
                })

            logger.info(f"CV Split: {len(fold_splits)} folds, "
                       f"test={len(test_data)}")

            result: Dict[str, Any] = {
                'cv_folds': fold_splits,
                'testing': test_data,
                'n_splits': n_splits,
                'split_type': 'cross_validation'
            }

            if fold_splits:
                first_fold = fold_splits[0]
                result['train'] = first_fold['train']
                result['validation'] = first_fold['validation']
            else:
                result['train'] = pd.DataFrame(index=data.index, columns=data.columns)
                result['validation'] = pd.DataFrame(index=data.index, columns=data.columns)

            result['test'] = test_data
            result['training'] = result['train']
            result['testing'] = result['test']
            return result

        else:
            # Simple chronological split (DEFAULT - BACKWARD COMPATIBLE)
            if has_duplicate_index:
                dates = _unique_dates(data)
                n_dates = len(dates)
                if n_dates == 0:
                    splits = SplitResult(
                        {
                            "training": _empty_like(data),
                            "validation": _empty_like(data),
                            "testing": _empty_like(data),
                        }
                    )
                    logger.info("Split (date-based): empty dataset")
                    return splits

                train_end = int(n_dates * train_ratio)
                val_end = int(n_dates * (train_ratio + val_ratio))
                train_end = max(1, min(train_end, n_dates))
                val_end = max(train_end, min(val_end, n_dates))

                train_cutoff = dates[train_end - 1]
                val_cutoff = dates[val_end - 1] if val_end > 0 else train_cutoff

                training_data = data.loc[data.index <= train_cutoff].copy()
                validation_data = (
                    data.loc[(data.index > train_cutoff) & (data.index <= val_cutoff)].copy()
                    if val_end > train_end
                    else _empty_like(data)
                )
                testing_data = data.loc[data.index > val_cutoff].copy() if val_end < n_dates else _empty_like(data)

                splits = SplitResult(
                    {
                        "training": training_data,
                        "validation": validation_data,
                        "testing": testing_data,
                    }
                )
                logger.info(
                    "Split (date-based): unique_dates=%s train=%s val=%s test=%s",
                    n_dates,
                    len(splits["training"]),
                    len(splits["validation"]),
                    len(splits["testing"]),
                )
                return splits

            n = len(data)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            training_data = data.iloc[:train_end]
            validation_data = data.iloc[train_end:val_end]
            testing_data = data.iloc[val_end:]

            splits = SplitResult({
                'training': training_data,
                'validation': validation_data,
                'testing': testing_data
            })

            logger.info(f"Split: train={len(splits['training'])}, "
                       f"val={len(splits['validation'])}, "
                       f"test={len(splits['testing'])}")

            return splits
