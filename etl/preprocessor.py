"""Data preprocessing with vectorized transformations."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Vectorized data preprocessing for time series."""

    def __init__(self):
        pass

    def handle_missing(self, data: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """Handle missing data with temporal interpolation."""
        if method == 'forward':
            filled = data.ffill()
        elif method == 'backward':
            filled = data.bfill()
        elif method == 'interpolate':
            filled = data.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fill remaining NaN with mean (only numeric columns)
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        filled[numeric_cols] = filled[numeric_cols].fillna(filled[numeric_cols].mean())

        logger.info(f"Filled missing data using {method}")
        return filled

    def normalize(self, data: pd.DataFrame, method: str = 'zscore',
                 columns: Optional[list] = None) -> Tuple[pd.DataFrame, Dict]:
        """Normalize data."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        normalized = data.copy()
        stats = {}

        for col in columns:
            if col not in data.columns:
                continue

            if method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    normalized[col] = (data[col] - mean) / std
                    stats[col] = {'mean': mean, 'std': std}

        logger.info(f"Normalized {len(columns)} columns")
        return normalized, stats
