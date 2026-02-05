"""execution.order_manager
============================

End-to-end order lifecycle orchestration for Portfolio Maximizer. The
historical roadmap expected a massive.com/polygon.io layer here; this module replaces that stub
with a production-ready cTrader integration that honours the same risk and
governance constraints.

Responsibilities
----------------

* Translate validated signals into broker-ready orders using confidence-weighted
  sizing capped at two percent of account equity.
* Run pre-trade checks (cash, risk gates, daily trade limits, circuit breakers)
  before the order ever hits the brokerage.
* Persist executions via ``DatabaseManager`` so downstream dashboards and risk
  tooling continue to function unchanged.
* Default to the cTrader *demo* environment for training + smoke tests as
  mandated in ``UNIFIED_ROADMAP.md``. Live mode remains a configuration switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

import logging
import os

import pandas as pd

from execution.ctrader_client import (
    AccountSnapshot,
    CTraderClient,
    CTraderClientConfig,
    CTraderOrder,
    CTraderOrderError,
    OrderPlacement,
)
from etl.database_manager import DatabaseManager
from risk.real_time_risk_manager import RealTimeRiskManager, RiskReport

logger = logging.getLogger(__name__)


def _infer_asset_class_from_ticker(ticker: str) -> str:
    """Best-effort asset class hint based on ticker string."""
    sym = (ticker or "").upper()
    if sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL"}:
        return "crypto"
    return "equity"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreTradeResult:
    """Outcome of risk/capacity checks before routing an order."""

    passed: bool
    checks: Dict[str, bool]
    reasons: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRequest:
    """Normalized signal information for downstream execution."""

    ticker: str
    action: str
    confidence: float
    current_price: float
    signal_id: Optional[int] = None
    mid_price: Optional[float] = None
    data_source: Optional[str] = None
    execution_mode: Optional[str] = None
    synthetic_dataset_id: Optional[str] = None
    synthetic_generator_version: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LifecycleResult:
    """Full lifecycle output used by scripts + monitoring."""

    status: str
    request: Optional[OrderRequest]
    pre_trade: PreTradeResult
    placement: Optional[OrderPlacement] = None
    account_snapshot: Optional[AccountSnapshot] = None
    risk_report: Optional[RiskReport] = None
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Order Manager
# ---------------------------------------------------------------------------


class OrderManager:
    """Broker-agnostic order lifecycle manager (cTrader implementation)."""

    def __init__(
        self,
        *,
        mode: str = "demo",
        client: Optional[CTraderClient] = None,
        database_manager: Optional[DatabaseManager] = None,
        risk_manager: Optional[RealTimeRiskManager] = None,
        max_position_risk: float = 0.02,
        min_confidence: float = 0.50,
        max_trades_per_day: int = 25,
    ) -> None:
        """Initialize the order manager.

        Args:
            mode: ``demo`` (default) or ``live``.
            client: Optional pre-configured ``CTraderClient``.
            database_manager: Optional ``DatabaseManager`` instance.
            risk_manager: Optional ``RealTimeRiskManager`` instance.
            max_position_risk: Maximum fraction of equity allocated per trade.
            min_confidence: Minimum confidence score required to trade.
            max_trades_per_day: Circuit breaker to avoid runaway order flow.
        """

        cfg = client.config if client else CTraderClientConfig.from_env(environment=mode)
        self.client = client or CTraderClient(cfg)
        self.db_manager = database_manager or DatabaseManager()
        self.risk_manager = risk_manager or RealTimeRiskManager()
        self.max_position_risk = max_position_risk
        self.min_confidence = min_confidence
        self.max_trades_per_day = max_trades_per_day
        self._daily_trade_counter: Dict[date, int] = {}

        logger.info(
            "Order manager bound to cTrader (%s environment)",
            "demo" if cfg.is_demo else "live",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_signal(
        self,
        signal: Dict[str, Any],
        market_data: Optional[pd.DataFrame] = None,
    ) -> LifecycleResult:
        """Run the full lifecycle for a trading signal."""

        request = self._build_request(signal, market_data)
        if not request:
            pre_trade = PreTradeResult(
                passed=False,
                checks={"signal_valid": False},
                reasons={"signal_valid": "Signal missing ticker/action/price"},
            )
            return LifecycleResult(
                status="REJECTED",
                request=None,
                pre_trade=pre_trade,
                reason="Invalid signal payload",
            )

        account_snapshot = self.client.get_account_overview()
        positions_payload = self.client.get_positions()
        positions = self._normalize_positions(positions_payload)
        prices = self._extract_position_prices(positions, market_data)

        risk_report = self.risk_manager.monitor_portfolio_risk(
            portfolio_value=account_snapshot.equity,
            positions=positions,
            position_prices=prices,
        )

        pre_trade = self._run_pre_trade_checks(
            request,
            account_snapshot,
            positions,
            risk_report,
        )

        if not pre_trade.passed:
            return LifecycleResult(
                status="REJECTED",
                request=request,
                pre_trade=pre_trade,
                account_snapshot=account_snapshot,
                risk_report=risk_report,
                reason="; ".join(pre_trade.reasons.values()) or "Pre-trade checks failed",
            )

        try:
            placement = self._execute_order(request, account_snapshot)
            self._record_execution(request, placement)
            self._increment_trade_counter()
            status = "EXECUTED" if placement.filled_volume else "SUBMITTED"
            logger.info(
                "Order lifecycle complete for %s (%s)",
                request.ticker,
                status,
            )
            return LifecycleResult(
                status=status,
                request=request,
                pre_trade=pre_trade,
                placement=placement,
                account_snapshot=account_snapshot,
                risk_report=risk_report,
            )
        except CTraderOrderError as exc:
            logger.error("Order placement failed: %s", exc)
            return LifecycleResult(
                status="FAILED",
                request=request,
                pre_trade=pre_trade,
                account_snapshot=account_snapshot,
                risk_report=risk_report,
                reason=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_request(
        self,
        signal: Dict[str, Any],
        market_data: Optional[pd.DataFrame],
    ) -> Optional[OrderRequest]:
        ticker = (signal.get("ticker") or signal.get("symbol") or "").strip().upper()
        action = (signal.get("action") or "HOLD").strip().upper()
        confidence = float(signal.get("confidence_score", signal.get("confidence", 0)))
        price = signal.get("current_price") or signal.get("price")
        data_source = signal.get("data_source")
        execution_mode = signal.get("execution_mode") or os.getenv("EXECUTION_MODE")
        synthetic_dataset_id = signal.get("synthetic_dataset_id") or os.getenv("SYNTHETIC_DATASET_ID")
        synthetic_generator_version = signal.get("synthetic_generator_version") or signal.get("generator_version")
        run_id = signal.get("run_id") or os.getenv("RUN_ID")
        mid_price = signal.get("mid_price")

        if not price and market_data is not None and not market_data.empty:
            try:
                price = float(market_data["Close"].iloc[-1])
            except Exception:  # pragma: no cover - best effort fallback
                price = None
        if mid_price is None and market_data is not None and not market_data.empty:
            try:
                last = market_data.iloc[-1]
                bid = last.get("Bid") if hasattr(last, "get") else None
                ask = last.get("Ask") if hasattr(last, "get") else None
                if bid is not None and ask is not None and pd.notna(bid) and pd.notna(ask) and float(ask) > float(bid) > 0:
                    mid_price = (float(bid) + float(ask)) / 2.0
                else:
                    high = last.get("High") if hasattr(last, "get") else None
                    low = last.get("Low") if hasattr(last, "get") else None
                    if high is not None and low is not None and pd.notna(high) and pd.notna(low):
                        mid_price = (float(high) + float(low)) / 2.0
                    else:
                        close = last.get("Close") if hasattr(last, "get") else None
                        if close is not None and pd.notna(close):
                            mid_price = float(close)
            except Exception:  # pragma: no cover - best effort
                mid_price = None

        if not ticker or action not in {"BUY", "SELL"} or not price:
            return None

        confidence = max(0.0, min(1.0, confidence or 0.0))
        if confidence < self.min_confidence:
            logger.info(
                "Signal confidence %.2f below threshold %.2f; will fail pre-trade",
                confidence,
                self.min_confidence,
            )

        signal_id_raw = signal.get("id") or signal.get("signal_id")
        signal_id: Optional[int] = None
        if signal_id_raw is not None:
            try:
                signal_id = int(signal_id_raw)
            except (TypeError, ValueError):
                signal_id = None

        return OrderRequest(
            ticker=ticker,
            action=action,
            confidence=confidence,
            current_price=float(price),
            signal_id=signal_id,
            mid_price=mid_price,
            data_source=data_source,
            execution_mode=execution_mode,
            synthetic_dataset_id=synthetic_dataset_id,
            synthetic_generator_version=synthetic_generator_version,
            run_id=run_id,
            metadata={k: v for k, v in signal.items() if k not in {"ticker", "symbol", "action"}},
        )

    def _run_pre_trade_checks(
        self,
        request: OrderRequest,
        account: AccountSnapshot,
        positions: Dict[str, int],
        risk_report: RiskReport,
    ) -> PreTradeResult:
        checks: Dict[str, bool] = {}
        reasons: Dict[str, str] = {}

        # Confidence threshold
        checks["confidence"] = request.confidence >= self.min_confidence
        if not checks["confidence"]:
            reasons["confidence"] = (
                f"Confidence {request.confidence:.2f} below {self.min_confidence:.2f}"
            )

        # Cash / free margin check
        notional = self._target_notional(account, request)
        checks["cash_available"] = account.free_margin >= notional
        if not checks["cash_available"]:
            reasons["cash_available"] = (
                f"Notional ${notional:,.2f} exceeds free margin ${account.free_margin:,.2f}"
            )

        # Position limit check (2% rule enforced during sizing too)
        checks["position_limit"] = notional <= account.equity * self.max_position_risk * 1.2
        if not checks["position_limit"]:
            reasons["position_limit"] = "Requested position breaches 2%% equity risk cap"

        # Daily trade limit
        todays_trades = self._daily_trade_counter.get(date.today(), 0)
        checks["daily_trade_limit"] = todays_trades < self.max_trades_per_day
        if not checks["daily_trade_limit"]:
            reasons["daily_trade_limit"] = "Daily trade limit reached"

        # Circuit breaker from risk manager
        checks["risk_status"] = risk_report.status == "HEALTHY"
        if not checks["risk_status"]:
            reasons["risk_status"] = f"Risk status {risk_report.status} blocks trading"

        passed = all(checks.values())
        return PreTradeResult(passed=passed, checks=checks, reasons=reasons)

    def _execute_order(self, request: OrderRequest, account: AccountSnapshot) -> OrderPlacement:
        volume = self._calculate_volume(request, account)
        order = CTraderOrder(
            symbol=request.ticker,
            side=request.action,
            volume=volume,
            order_type="MARKET",
            comment=request.metadata.get("reasoning") or "Auto-trade via Portfolio Maximizer",
            client_order_id=request.metadata.get("client_order_id"),
            label=request.metadata.get("signal_type"),
        )
        return self.client.place_order(order)

    def _calculate_volume(self, request: OrderRequest, account: AccountSnapshot) -> float:
        target_value = self._target_notional(account, request)
        volume = max(1, int(target_value / max(request.current_price, 1e-6)))
        logger.debug(
            "Calculated trade volume %s for %s (target_value=%s, price=%s)",
            volume,
            request.ticker,
            target_value,
            request.current_price,
        )
        return float(volume)

    def _target_notional(self, account: AccountSnapshot, request: OrderRequest) -> float:
        base_limit = account.equity * self.max_position_risk
        multiplier = request.confidence if request.confidence > 0 else self.min_confidence
        return max(0.0, base_limit * multiplier)

    def _normalize_positions(self, payload: Optional[Dict[str, Any]]) -> Dict[str, int]:
        positions = {}
        if payload is None:
            return positions

        if isinstance(payload, list):
            raw_positions = payload
        else:
            raw_positions = (
                payload.get("positions")
                or payload.get("data", {}).get("positions")
                or payload.get("positionDtos")
                or []
            )

        for item in raw_positions:
            name = (item.get("symbol") or item.get("symbolName") or "").upper()
            volume = int(item.get("volume") or 0)
            if name:
                positions[name] = volume
        return positions

    def _extract_position_prices(
        self,
        positions: Dict[str, int],
        market_data: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        if market_data is not None and not market_data.empty and "Close" in market_data.columns:
            latest_price = float(market_data["Close"].iloc[-1])
        else:
            latest_price = 0.0

        for ticker in positions:
            prices[ticker] = request_safe_price(latest_price)
        return prices

    def _record_execution(self, request: OrderRequest, placement: OrderPlacement) -> None:
        if placement.filled_volume <= 0:
            logger.debug("Skipping trade persistence for unfilled order %s", placement.order_id)
            return

        executed_price = placement.avg_price or request.current_price
        total_value = executed_price * placement.filled_volume
        mid_price = placement.mid_price if placement.mid_price is not None else request.mid_price
        mid_slippage_bps = placement.mid_slippage_bps
        try:
            if mid_slippage_bps is None and mid_price and mid_price > 0 and executed_price is not None:
                mid_slippage_bps = ((executed_price - mid_price) / mid_price) * 1e4
        except Exception:
            mid_slippage_bps = None
        # Allow upstream signal metadata to tag asset_class / instrument_type /
        # option fields when options_trading is enabled. For current spot-only
        # flows this remains a no-op and falls back to ticker heuristics.
        meta = request.metadata or {}
        raw_asset_class = meta.get("asset_class")
        asset_class = (raw_asset_class or _infer_asset_class_from_ticker(request.ticker)).lower()
        instrument_type = (meta.get("instrument_type") or "spot").lower()
        underlying_ticker = meta.get("underlying_ticker")
        strike = meta.get("strike")
        expiry = meta.get("expiry")
        multiplier = meta.get("multiplier") or 1.0
        data_source = request.data_source or meta.get("data_source")
        execution_mode = request.execution_mode or meta.get("execution_mode") or os.getenv("EXECUTION_MODE")
        synthetic_dataset_id = request.synthetic_dataset_id or meta.get("synthetic_dataset_id") or os.getenv("SYNTHETIC_DATASET_ID")
        synthetic_generator_version = request.synthetic_generator_version or meta.get("synthetic_generator_version") or meta.get("generator_version")
        run_id = request.run_id or meta.get("run_id") or os.getenv("RUN_ID")

        self.db_manager.save_trade_execution(
            ticker=request.ticker,
            trade_date=datetime.now(timezone.utc),
            action=request.action,
            shares=placement.filled_volume,
            price=executed_price,
            total_value=total_value,
            commission=0.0,
            signal_id=request.signal_id,
            data_source=data_source,
            execution_mode=execution_mode,
            synthetic_dataset_id=synthetic_dataset_id,
            synthetic_generator_version=synthetic_generator_version,
            run_id=run_id,
            mid_price=mid_price,
            mid_slippage_bps=mid_slippage_bps,
            asset_class=asset_class,
            instrument_type=instrument_type,
            underlying_ticker=underlying_ticker,
            strike=strike,
            expiry=expiry,
            multiplier=multiplier,
        )

    def _increment_trade_counter(self) -> None:
        today = date.today()
        # Drop historical counters to keep dict tiny
        self._daily_trade_counter = {today: self._daily_trade_counter.get(today, 0) + 1}


def request_safe_price(price: float) -> float:
    """Guard against zero/None when estimating position price."""

    if price and price > 0:
        return price
    return 1.0  # fallback for risk calculations when price unavailable


__all__ = [
    "LifecycleResult",
    "OrderManager",
    "OrderRequest",
    "PreTradeResult",
]
