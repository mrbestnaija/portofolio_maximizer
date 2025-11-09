"""Data access utilities for visualization dashboards."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence

import pandas as pd

from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class DashboardDataLoader:
    """Helper for retrieving structured data for dashboards."""

    db_manager: DatabaseManager

    @classmethod
    def from_path(cls, db_path: str = "data/portfolio_maximizer.db") -> "DashboardDataLoader":
        return cls(db_manager=DatabaseManager(db_path=db_path))

    def get_price_history(
        self,
        ticker: str,
        lookback_days: Optional[int] = 180,
        columns: Sequence[str] = ("close",),
    ) -> Optional[pd.DataFrame]:
        query = """
            SELECT date, open, high, low, close, volume, adj_close
            FROM ohlcv_data
            WHERE ticker = ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, self.db_manager.conn, params=(ticker,))
        if df.empty:
            logger.info("No OHLCV data found for %s", ticker)
            return None

        df["date"] = pd.to_datetime(df["date"])
        if lookback_days:
            cutoff = df["date"].max() - timedelta(days=int(lookback_days))
            df = df[df["date"] >= cutoff]

        df.set_index("date", inplace=True)
        columns = [col for col in columns if col in df.columns]
        if not columns:
            columns = ["close"]
        return df[columns].rename(columns=str.title)

    def get_forecast_bundle(self, ticker: str) -> Dict[str, Dict[str, pd.Series]]:
        query = """
            SELECT
                forecast_date,
                model_type,
                forecast_horizon,
                forecast_value,
                lower_ci,
                upper_ci,
                diagnostics,
                regression_metrics
            FROM time_series_forecasts
            WHERE ticker = ?
              AND forecast_date = (
                  SELECT MAX(forecast_date) FROM time_series_forecasts WHERE ticker = ?
              )
            ORDER BY model_type, forecast_horizon
        """
        df = pd.read_sql_query(query, self.db_manager.conn, params=(ticker, ticker))
        if df.empty:
            return {}

        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        bundles: Dict[str, Dict[str, pd.Series]] = {}
        forecast_date = df["forecast_date"].iloc[0]

        for model in df["model_type"].unique():
            subset = df[df["model_type"] == model].sort_values("forecast_horizon")
            horizons = subset["forecast_horizon"].astype(int).tolist()
            index = [forecast_date + timedelta(days=h) for h in horizons]

            diagnostics_raw = subset["diagnostics"].dropna().iloc[0] if "diagnostics" in subset and not subset["diagnostics"].dropna().empty else {}
            if isinstance(diagnostics_raw, str):
                try:
                    diagnostics = json.loads(diagnostics_raw)
                except json.JSONDecodeError:
                    diagnostics = {}
            elif isinstance(diagnostics_raw, dict):
                diagnostics = diagnostics_raw
            else:
                diagnostics = {}

            regression_raw = subset["regression_metrics"].dropna().iloc[0] if "regression_metrics" in subset and not subset["regression_metrics"].dropna().empty else {}
            if isinstance(regression_raw, str):
                try:
                    regression_metrics = json.loads(regression_raw)
                except json.JSONDecodeError:
                    regression_metrics = {}
            elif isinstance(regression_raw, dict):
                regression_metrics = regression_raw
            else:
                regression_metrics = {}

            bundles[model] = {
                "forecast": pd.Series(subset["forecast_value"].astype(float).values, index=index),
                "lower_ci": pd.Series(subset["lower_ci"].astype(float).values, index=index)
                if subset["lower_ci"].notna().any()
                else None,
                "upper_ci": pd.Series(subset["upper_ci"].astype(float).values, index=index)
                if subset["upper_ci"].notna().any()
                else None,
                "diagnostics": diagnostics,
                "weights": diagnostics.get("weights") if isinstance(diagnostics, dict) else None,
                "regression_metrics": regression_metrics,
            }

        return bundles

    def get_signal_backtests(self, ticker: Optional[str] = None, limit: int = 20) -> pd.DataFrame:
        query = """
            SELECT
                ticker,
                generated_at,
                lookback_days,
                signals_analyzed,
                hit_rate,
                profit_factor,
                sharpe_ratio,
                information_ratio,
                statistically_significant
            FROM llm_signal_backtests
            {where_clause}
            ORDER BY generated_at DESC
            LIMIT ?
        """
        params: list = []
        where_clause = ""
        if ticker:
            where_clause = "WHERE ticker = ?"
            params.append(ticker)
        params.append(limit)

        df = pd.read_sql_query(query.format(where_clause=where_clause), self.db_manager.conn, params=params)
        if df.empty:
            return df

        df["generated_at"] = pd.to_datetime(df["generated_at"])
        df["statistically_significant"] = df["statistically_significant"].astype(bool)
        return df
