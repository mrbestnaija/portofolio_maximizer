"""
cTrader-backed extractor for OHLCV data.

This extractor uses the demo-first cTrader Open API client for price history
and normalises responses into the standard OHLCV schema expected by the ETL
stack. When cTrader is unavailable or returns an unexpected payload, callers
can still rely on the multi-source failover in DataSourceManager to fall back
to other providers (e.g. yfinance).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from etl.base_extractor import BaseExtractor, ExtractorMetadata
from execution.ctrader_client import CTraderClient, CTraderClientConfig, CTraderClientError

logger = logging.getLogger(__name__)


class CTraderExtractor(BaseExtractor):
    """Extractor that pulls OHLCV data from the cTrader Open API."""

    def __init__(self, name: str = "ctrader", storage=None, cache_hours: int = 24, **kwargs):
        super().__init__(name=name, cache_hours=cache_hours, storage=storage, **kwargs)

        # Resolve environment from config/ctrader_config.yml (demo by default)
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "ctrader_config.yml"
        environment = "demo"
        if cfg_path.exists():
            try:
                raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                c_cfg = raw.get("ctrader") or {}
                environment = str(c_cfg.get("environment") or environment)
            except Exception:
                logger.debug("Unable to parse ctrader_config.yml; defaulting environment=%s", environment)

        client_cfg = CTraderClientConfig.from_env(environment=environment)
        self._client = CTraderClient(config=client_cfg)
        logger.info("CTraderExtractor initialised (environment=%s, account_id=%s)", environment, client_cfg.account_id)

    def _bars_to_frame(self, symbol: str, bars: List[Dict[str, Any]]) -> pd.DataFrame:
        """Normalise raw cTrader bars into OHLCV DataFrame."""
        if not bars:
            return pd.DataFrame()

        frame = pd.DataFrame(bars)

        # Try to locate time and OHLCV columns with common naming conventions.
        time_col = None
        for candidate in ("timestamp", "time", "date", "openTime"):
            if candidate in frame.columns:
                time_col = candidate
                break
        if time_col is None:
            raise CTraderClientError(f"No time-like column found in bars payload for {symbol}")

        col_map: Dict[str, str] = {}
        for logical, candidates in {
            "Open": ("open", "o", "bidOpen", "askOpen"),
            "High": ("high", "h", "bidHigh", "askHigh"),
            "Low": ("low", "l", "bidLow", "askLow"),
            "Close": ("close", "c", "bidClose", "askClose"),
            "Volume": ("volume", "v", "tickVolume"),
        }.items():
            for candidate in candidates:
                if candidate in frame.columns:
                    col_map[logical] = candidate
                    break

        missing = [k for k in ("Open", "High", "Low", "Close") if k not in col_map]
        if missing:
            raise CTraderClientError(
                f"Missing OHLC columns {missing!r} in bars payload for {symbol}; columns={list(frame.columns)}"
            )

        frame = frame.rename(columns=col_map)
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
        frame = frame.set_index(time_col).sort_index()

        ohlcv = frame[["Open", "High", "Low", "Close"]].copy()
        if "Volume" in frame.columns:
            ohlcv["Volume"] = frame["Volume"]
        else:
            ohlcv["Volume"] = 0.0

        ohlcv["ticker"] = symbol
        ohlcv["source"] = self.name
        return ohlcv

    def extract_ohlcv(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        frames: List[pd.DataFrame] = []
        for symbol in tickers:
            try:
                bars = self._client.get_bars(symbol=symbol, start=start_dt, end=end_dt, timeframe="D1")
                frame = self._bars_to_frame(symbol, bars)
                if not frame.empty:
                    frames.append(frame)
            except CTraderClientError as exc:
                logger.error("cTrader OHLCV fetch failed for %s: %s", symbol, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unexpected error normalising cTrader bars for %s: %s", symbol, exc)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=0)

    def validate_data(self, data: pd.DataFrame) -> bool:
        if data is None or data.empty:
            return False
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(set(data.columns)):
            return False
        return True

    def get_metadata(self, ticker: str, data: pd.DataFrame, cache_hit: bool) -> ExtractorMetadata:
        return ExtractorMetadata(
            ticker=ticker,
            rows=int(len(data)),
            columns=list(data.columns),
            cache_hit=cache_hit,
            source=self.name,
        )
