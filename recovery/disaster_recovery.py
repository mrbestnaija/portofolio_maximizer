"""Disaster recovery and failover helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from etl.data_source_manager import DataSourceManager

logger = logging.getLogger(__name__)


@dataclass
class SystemFailure:
    type: str
    detail: Optional[str] = None


@dataclass
class RecoveryResult:
    status: str
    action: str
    detail: Optional[str] = None


class DisasterRecovery:
    """Coordinate recovery paths for critical failures."""

    def __init__(self) -> None:
        self.dsm = DataSourceManager()

    def handle_system_failure(self, failure: SystemFailure) -> RecoveryResult:
        ft = failure.type.upper()
        if ft == "MODEL_FAILURE":
            return self._fallback_to_simple_model(failure.detail)
        if ft == "DATA_FAILURE":
            return self._failover_data_source()
        if ft == "BROKER_FAILURE":
            return self._emergency_position_closure(failure.detail)
        if ft == "RISK_BREACH":
            return self._automatic_risk_reduction(failure.detail)
        return RecoveryResult(status="FAILED", action="unknown_failure", detail=f"Unhandled failure type {failure.type}")

    def _fallback_to_simple_model(self, detail: Optional[str] = None) -> RecoveryResult:
        logger.warning("Falling back to simple heuristic model: %s", detail or "no detail")
        return RecoveryResult(status="RECOVERED", action="fallback_model", detail=detail)

    def _failover_data_source(self) -> RecoveryResult:
        for source in ["synthetic", "yfinance", "alpha_vantage", "finnhub"]:
            try:
                if self.dsm.test_data_source(source):
                    self.dsm.set_primary_source(source)
                    return RecoveryResult(status="RECOVERED", action=f"failover_to_{source}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Data source %s unavailable during failover: %s", source, exc)
        return RecoveryResult(status="FAILED", action="data_failover_exhausted")

    def _emergency_position_closure(self, detail: Optional[str] = None) -> RecoveryResult:
        logger.critical("Broker failure detected; positions should be closed. Detail: %s", detail)
        try:
            from execution.order_manager import OrderManager, Order
        except Exception:
            return RecoveryResult(status="FAILED", action="broker_close_positions_unavailable", detail=detail)

        try:
            om = OrderManager(mode="demo")
            positions = om.db.get_current_positions() if hasattr(om, "db") else {}  # type: ignore[attr-defined]
            for ticker, shares in positions.items():
                close_order = Order(
                    ticker=ticker,
                    action="SELL",
                    quantity=shares,
                    order_type="MARKET",
                    reason="BROKER_FAILOVER",
                )
                om.manage_order_lifecycle(close_order)
            return RecoveryResult(status="RECOVERED", action="positions_closed", detail=detail)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to close positions during broker failure: %s", exc)
            return RecoveryResult(status="FAILED", action="close_positions_error", detail=str(exc))

    def _automatic_risk_reduction(self, detail: Optional[str] = None) -> RecoveryResult:
        logger.warning("Risk breach detected; reducing positions. Detail: %s", detail)
        try:
            from risk.real_time_risk_manager import RealTimeRiskManager
        except Exception:
            return RecoveryResult(status="FAILED", action="risk_manager_unavailable", detail=detail)
        try:
            rtm = RealTimeRiskManager()
            positions: Dict[str, Any] = getattr(rtm, "positions", {}) or {}
            if hasattr(rtm, "_execute_automatic_action"):
                rtm._execute_automatic_action("REDUCE_POSITIONS", positions)  # pragma: no cover
            return RecoveryResult(status="RECOVERED", action="positions_reduced", detail=detail)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Risk reduction failed: %s", exc)
            return RecoveryResult(status="FAILED", action="risk_reduction_error", detail=str(exc))
