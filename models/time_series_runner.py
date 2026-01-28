"""Parallel runner for TS and LLM generators."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    action: str
    confidence: float
    source: str
    details: Dict[str, Any]


class TimeSeriesRunner:
    """Execute SAMOSSA/SARIMAX/GARCH and optional LLM pipelines in parallel."""

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run_all(self, context: Any) -> List[Signal]:
        tasks = [
            asyncio.get_running_loop().run_in_executor(self.executor, self._run_sarimax, context),
            asyncio.get_running_loop().run_in_executor(self.executor, self._run_garch, context),
            asyncio.get_running_loop().run_in_executor(self.executor, self._run_llm, context),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[Signal] = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Model execution failed: %s", res)
                continue
            if res:
                signals.append(self._normalize_signal(res))
        return signals

    def _run_sarimax(self, context: Any) -> Optional[Dict[str, Any]]:
        try:
            from etl.time_series_forecaster import SARIMAXForecaster
        except Exception as exc:
            logger.debug("SARIMAX forecaster unavailable: %s", exc)
            return None
        forecaster = SARIMAXForecaster()
        result = forecaster.fit_and_forecast(context.market_data)
        return {"source": "sarimax", "result": result}

    def _run_garch(self, context: Any) -> Optional[Dict[str, Any]]:
        try:
            from etl.time_series_forecaster import GARCHForecaster
        except Exception as exc:
            logger.debug("GARCH forecaster unavailable: %s", exc)
            return None
        forecaster = GARCHForecaster()
        forecaster.fit(context.returns)
        result = forecaster.forecast_volatility(context.horizon)
        return {"source": "garch", "result": result}

    def _run_llm(self, context: Any) -> Optional[Dict[str, Any]]:
        raw = os.getenv("PM_ENABLE_OLLAMA")
        if raw is None or raw.strip().lower() not in {"1", "true", "yes", "on"}:
            return None
        try:
            from ai_llm.signal_generator import LLMSignalGenerator
            from ai_llm.ollama_client import OllamaClient
        except Exception as exc:
            logger.debug("LLM generator unavailable: %s", exc)
            return None
        client = OllamaClient()
        generator = LLMSignalGenerator(client)
        signal = generator.generate_signal(context.market_data)
        return {"source": "llm", "result": signal}

    def _normalize_signal(self, payload: Dict[str, Any]) -> Signal:
        source = payload.get("source", "unknown")
        result = payload.get("result") or {}
        ticker = getattr(result, "ticker", None) or result.get("ticker", "UNKNOWN")
        action = getattr(result, "action", None) or result.get("action", "HOLD")
        confidence = float(getattr(result, "confidence", 0.0) or result.get("confidence", 0.0) or 0.0)
        details = result if isinstance(result, dict) else {"raw": result}
        return Signal(ticker=ticker, action=action, confidence=confidence, source=source, details=details)
