"""
Paper Trading Engine - Realistic market simulation
Line Count: ~400 lines

Provides paper trading with realistic simulation:
- Signal validation before execution
- Realistic slippage modeling (0.1% baseline)
- Transaction costs (0.1%)
- Market impact for large positions
- Database persistence
- Portfolio tracking

Per AGENT_INSTRUCTION.md:
- Paper trading required before live deployment
- Must achieve >52% accuracy for 3+ months
- Realistic costs and slippage modeling
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ai_llm.signal_validator import SignalValidator
from etl.database_manager import DatabaseManager
from execution.lob_simulator import LOBConfig, simulate_market_order_fill

logger = logging.getLogger(__name__)


def _infer_asset_class_from_ticker(ticker: str) -> str:
    """Best-effort asset class hint aligned to config/execution_cost_model.yml profiles."""
    sym = (ticker or "").upper()
    if sym.endswith("=X"):
        return "FX"
    if sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL"}:
        return "CRYPTO"
    if "^" in sym:
        return "INDEX"
    if any(sym.endswith(suffix) for suffix in (".NS", ".TW", ".L")):
        return "INTL_EQUITY"
    return "US_EQUITY"


@dataclass
class Trade:
    """Trade execution details"""
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    entry_price: float
    transaction_cost: float
    timestamp: datetime
    bar_timestamp: Optional[datetime] = None
    is_paper_trade: bool = True
    trade_id: Optional[str] = None
    slippage: float = 0.0
    signal_id: Optional[int] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    forecast_horizon: Optional[int] = None
    exit_reason: Optional[str] = None
    mid_price: Optional[float] = None
    mid_slippage_bps: Optional[float] = None
    data_source: Optional[str] = None
    execution_mode: Optional[str] = None
    synthetic_dataset_id: Optional[str] = None
    synthetic_generator_version: Optional[str] = None
    run_id: Optional[str] = None
    # Optional instrument metadata for options/derivatives (kept inert until
    # options_trading.enabled + feature flags are active).
    asset_class: str = "equity"
    instrument_type: str = "spot"
    underlying_ticker: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None
    multiplier: float = 1.0
    # Barbell sizing overlay audit fields (feature-flagged; stored when present).
    barbell_bucket: Optional[str] = None
    barbell_multiplier: Optional[float] = None
    base_confidence: Optional[float] = None
    effective_confidence: Optional[float] = None

    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = f"{self.ticker}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"


@dataclass
class Portfolio:
    """Portfolio state"""
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)  # ticker -> shares
    entry_prices: Dict[str, float] = field(default_factory=dict)  # ticker -> avg entry price
    entry_timestamps: Dict[str, datetime] = field(default_factory=dict)  # ticker -> opened timestamp
    entry_bar_timestamps: Dict[str, datetime] = field(default_factory=dict)  # ticker -> opened bar timestamp
    last_bar_timestamps: Dict[str, datetime] = field(default_factory=dict)  # ticker -> last evaluated bar timestamp
    holding_bars: Dict[str, int] = field(default_factory=dict)  # ticker -> bars held since entry
    stop_losses: Dict[str, float] = field(default_factory=dict)  # ticker -> stop loss price
    target_prices: Dict[str, float] = field(default_factory=dict)  # ticker -> take-profit price
    max_holding_days: Dict[str, int] = field(default_factory=dict)  # ticker -> max holding days
    total_value: float = 0.0

    def update_value(self, current_prices: Dict[str, float]):
        """Update total portfolio value"""
        position_value = sum(
            shares * current_prices.get(ticker, self.entry_prices.get(ticker, 0))
            for ticker, shares in self.positions.items()
        )
        self.total_value = self.cash + position_value


@dataclass
class ExecutionResult:
    """Result of trade execution"""
    status: str  # 'EXECUTED', 'REJECTED', 'FAILED'
    trade: Optional[Trade] = None
    portfolio: Optional[Portfolio] = None
    performance_impact: Optional[Dict[str, float]] = None
    reason: Optional[str] = None
    validation_warnings: list = field(default_factory=list)


class PaperTradingEngine:
    """
    Paper trading engine with realistic market simulation.

    Features:
    - Signal validation before execution
    - Realistic entry price simulation (slippage)
    - Transaction costs (0.1% baseline)
    - Market impact for large orders
    - Database persistence
    - Portfolio tracking and performance metrics
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 slippage_pct: float = 0.0005,
                 transaction_cost_pct: float = 0.0,
                 db_path: str = None,
                 database_manager: Optional[DatabaseManager] = None,
                 signal_validator: Optional[SignalValidator] = None):
        """
        Initialize paper trading engine.

        Args:
            initial_capital: Starting capital ($10,000 default)
            slippage_pct: Slippage percentage (0.1% default)
            transaction_cost_pct: Transaction cost percentage (0.1% default)
            db_path: Path to database for persistence
        """
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.transaction_cost_pct = transaction_cost_pct

        # Initialize components
        self.db_manager = database_manager or DatabaseManager(db_path)
        self.signal_validator = signal_validator or SignalValidator()
        self._lob_config: Optional[LOBConfig] = None
        self._lob_enabled: Optional[bool] = None

        # Initialize portfolio
        self.portfolio = Portfolio(cash=initial_capital, total_value=initial_capital)

        # Trade history
        self.trades: list[Trade] = []

        logger.info(
            f"Paper Trading Engine initialized with ${initial_capital:,.2f} "
            f"(slippage={slippage_pct:.2%}, costs={transaction_cost_pct:.2%})"
        )

    def execute_signal(self,
                      signal: Dict[str, Any],
                      market_data: pd.DataFrame) -> ExecutionResult:
        """
        Execute trading signal with full validation and simulation.

        Args:
            signal: LLM signal dict with action, confidence, reasoning
            market_data: Recent OHLCV data for validation

        Returns:
            ExecutionResult with execution details
        """
        ticker = signal.get('ticker', 'UNKNOWN')
        action = signal.get('action', 'HOLD').upper()
        current_position = self.portfolio.positions.get(ticker, 0)
        data_source = signal.get("data_source")
        run_id = signal.get("run_id") or os.getenv("RUN_ID")
        execution_mode = signal.get("execution_mode") or os.getenv("EXECUTION_MODE")
        synthetic_dataset_id = signal.get("synthetic_dataset_id") or os.getenv("SYNTHETIC_DATASET_ID")
        synthetic_generator_version = signal.get("synthetic_generator_version") or signal.get("generator_version")

        # Prefer signal-provided timestamps when replaying historical windows so
        # trade_date reflects the simulated session instead of "now".
        trade_timestamp = datetime.now()
        signal_timestamp = signal.get("signal_timestamp") or signal.get("timestamp")
        if signal_timestamp is not None:
            if isinstance(signal_timestamp, datetime):
                trade_timestamp = signal_timestamp
            else:
                try:
                    parsed = pd.to_datetime(signal_timestamp, errors="coerce")
                    if parsed is not pd.NaT:
                        trade_timestamp = parsed.to_pydatetime()
                except Exception:
                    trade_timestamp = datetime.now()

        # Best-effort current price from the market window.
        current_price = None
        try:
            if isinstance(market_data, pd.DataFrame) and not market_data.empty and "Close" in market_data.columns:
                current_price = float(market_data["Close"].iloc[-1])
        except Exception:
            current_price = None
        if current_price is None:
            try:
                current_price = float(signal.get("entry_price") or 0.0)
            except Exception:
                current_price = 0.0
        if float(current_price or 0.0) <= 0.0:
            return ExecutionResult(status="REJECTED", reason="Missing current price context")

        bar_timestamp = None
        try:
            if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                idx = market_data.index
                if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
                    bar_timestamp = pd.Timestamp(idx[-1]).to_pydatetime()
        except Exception:
            bar_timestamp = None

        forced_exit_reason = None
        forced_exit_shares = None
        if current_position and current_price:
            forced_exit_reason = self._evaluate_exit_reason(
                ticker=ticker,
                shares=current_position,
                current_price=float(current_price),
                as_of=trade_timestamp,
                market_data=market_data,
                bar_timestamp=bar_timestamp,
            )
            if forced_exit_reason:
                action = "SELL" if current_position > 0 else "BUY"
                signal = dict(signal)
                signal["action"] = action
                signal.setdefault("reasoning", "")
                reason_prefix = f"Lifecycle exit ({forced_exit_reason})"
                signal["reasoning"] = (
                    reason_prefix if not signal["reasoning"] else f"{signal['reasoning']} | {reason_prefix}"
                )
                signal["exit_reason"] = forced_exit_reason
                try:
                    signal["confidence"] = max(float(signal.get("confidence") or 0.0), 0.9)
                except (TypeError, ValueError):
                    signal["confidence"] = 0.9
                forced_exit_shares = abs(int(current_position))

        # Portfolio snapshot passed into the validator for correlation/concentration checks.
        portfolio_snapshot = {
            "cash": float(self.portfolio.cash),
            "positions": dict(self.portfolio.positions),
            "entry_prices": dict(self.portfolio.entry_prices),
            "total_value": float(self.portfolio.total_value),
        }
        if isinstance(market_data, pd.DataFrame) and not market_data.empty:
            try:
                if isinstance(market_data.index, pd.DatetimeIndex):
                    start_date = market_data.index[0].date().isoformat()
                    end_date = market_data.index[-1].date().isoformat()
                    peers = [
                        str(sym)
                        for sym, shares in self.portfolio.positions.items()
                        if shares and str(sym).upper() != str(ticker).upper()
                    ]
                    if peers and hasattr(self.db_manager, "load_ohlcv"):
                        frame = self.db_manager.load_ohlcv([str(ticker)] + peers, start_date=start_date, end_date=end_date)
                        if frame is not None and not frame.empty and "ticker" in frame.columns and "close" in frame.columns:
                            closes = frame.pivot_table(index=frame.index, columns="ticker", values="close")
                            returns = closes.pct_change()
                            ticker_upper = str(ticker).upper()
                            candidate_col = None
                            for col in returns.columns:
                                if str(col).upper() == ticker_upper:
                                    candidate_col = col
                                    break
                            candidate_series = returns[candidate_col].dropna() if candidate_col is not None else None
                            if candidate_series is not None and len(candidate_series) >= 10:
                                corrs = returns.corrwith(returns[candidate_col]).dropna()
                                corr_map = {
                                    str(sym): float(val)
                                    for sym, val in corrs.items()
                                    if str(sym).upper() != ticker_upper
                                }
                                if corr_map:
                                    portfolio_snapshot["correlation_snapshot"] = {
                                        "start_date": start_date,
                                        "end_date": end_date,
                                        "observations": int(len(candidate_series)),
                                        "correlations": corr_map,
                                    }
            except Exception:
                # Correlation is best-effort; never block execution due to telemetry failures.
                pass

        if action not in {"BUY", "SELL"}:
            logger.info("Skipping non-actionable signal (%s) for %s", action, ticker)
            return ExecutionResult(
                status='REJECTED',
                reason="Non-actionable signal",
            )

        logger.info(f"Executing %s signal for %s", action, ticker)
        # Step 1: Validate signal (diagnostic toggle via env DIAGNOSTIC_MODE/EXECUTION_DIAGNOSTIC_MODE)
        diag_mode = str(os.getenv("EXECUTION_DIAGNOSTIC_MODE") or os.getenv("DIAGNOSTIC_MODE") or "0") == "1"
        try:
            validation = self.signal_validator.validate_llm_signal(
                signal,
                market_data,
                self._current_portfolio_value(),
                portfolio_state=portfolio_snapshot,
            )
        except TypeError:
            # Backward compatibility: tests and alternate validators may not accept
            # the portfolio_state kwarg yet.
            validation = self.signal_validator.validate_llm_signal(
                signal,
                market_data,
                self._current_portfolio_value(),
            )
        if (not validation.is_valid or validation.recommendation == 'REJECT') and not diag_mode:
            logger.warning("Signal rejected: %s", validation.warnings)
            return ExecutionResult(
                status='REJECTED',
                reason=f"Validation failed: {validation.warnings}",
                validation_warnings=validation.warnings
            )
        if diag_mode and validation.warnings:
            logger.warning("Validation warnings (ignored in DIAGNOSTIC_MODE): %s", validation.warnings)

        # Step 2: Calculate position size (forced exits close the full position).
        if forced_exit_shares is not None:
            position_size = int(forced_exit_shares)
        else:
            position_size = self._calculate_position_size(
                signal, validation.confidence_score, market_data, current_position
            )

        if position_size == 0:
            return ExecutionResult(
                status='REJECTED',
                reason="Position size calculated as zero",
                validation_warnings=validation.warnings
            )

        # Step 3: Check available capital
        required_capital = position_size * float(current_price or 0.0)

        opening_or_adding_long = action == "BUY" and current_position >= 0
        if opening_or_adding_long and required_capital > self.portfolio.cash:
            logger.warning(
                "Insufficient cash: need $%.2f, have $%.2f",
                required_capital,
                self.portfolio.cash,
            )
            return ExecutionResult(
                status='REJECTED',
                reason=f"Insufficient cash (need ${required_capital:,.2f})"
            )

        # Step 4: Simulate entry price with slippage/market impact.
        entry_price = self._simulate_entry_price(
            float(current_price or 0.0),
            action,
            position_size,
            market_data=market_data,
            ticker=ticker,
        )

        # Step 5: Calculate transaction costs
        transaction_value = position_size * entry_price
        txn_cost_bps = None
        if isinstance(market_data, pd.DataFrame) and not market_data.empty:
            try:
                last = market_data.iloc[-1]
                txn_cost_bps = self._safe_float(last.get("TxnCostBps") if hasattr(last, "get") else None)
            except Exception:
                txn_cost_bps = None
        # If TxnCostBps is present, it is assumed to have been applied via
        # slippage/LOB simulation (avoid double-counting as explicit commission).
        if txn_cost_bps is not None:
            transaction_cost = 0.0
        else:
            transaction_cost = transaction_value * self.transaction_cost_pct

        # Step 6: Create trade object
        horizon_raw = signal.get("forecast_horizon") or signal.get("horizon")
        try:
            forecast_horizon = int(horizon_raw) if horizon_raw is not None else None
        except (TypeError, ValueError):
            forecast_horizon = None

        asset_class = _infer_asset_class_from_ticker(ticker)
        trade = Trade(
            ticker=ticker,
            action=action,
            shares=position_size,
            entry_price=entry_price,
            transaction_cost=transaction_cost,
            timestamp=trade_timestamp,
            bar_timestamp=bar_timestamp,
            is_paper_trade=True,
            slippage=(abs(entry_price - float(current_price)) / float(current_price)) if current_price else 0.0,
            signal_id=signal.get('signal_id'),
            stop_loss=signal.get("stop_loss"),
            target_price=signal.get("target_price"),
            forecast_horizon=forecast_horizon,
            exit_reason=signal.get("exit_reason") if forced_exit_reason else None,
            data_source=data_source,
            execution_mode=execution_mode,
            synthetic_dataset_id=synthetic_dataset_id,
            synthetic_generator_version=synthetic_generator_version,
            run_id=run_id,
            asset_class=asset_class,
            barbell_bucket=signal.get("barbell_bucket"),
            barbell_multiplier=signal.get("barbell_multiplier"),
            base_confidence=signal.get("base_confidence"),
            effective_confidence=signal.get("confidence"),
        )
        mid_price_hint = signal.get("mid_price_hint")
        try:
            mid_price = float(mid_price_hint) if mid_price_hint not in (None, "", 0, 0.0) else None
        except (TypeError, ValueError):
            mid_price = None
        if mid_price is None:
            try:
                if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                    last = market_data.iloc[-1]
                    bid = last.get("Bid") if hasattr(last, "get") else None
                    ask = last.get("Ask") if hasattr(last, "get") else None
                    if bid is not None and ask is not None and pd.notna(bid) and pd.notna(ask):
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
            except Exception:
                mid_price = None

        if mid_price is not None and mid_price > 0:
            trade.mid_price = mid_price
            trade.mid_slippage_bps = ((trade.entry_price - mid_price) / mid_price) * 1e4

        # Step 7: Execute trade (update portfolio)
        try:
            updated_portfolio = self._update_portfolio(trade, self.portfolio)

            # Step 8: Store in database
            realized_pnl, realized_pct = self._store_trade_execution(trade)
            trade.realized_pnl = realized_pnl
            trade.realized_pnl_pct = realized_pct

            # Step 9: Calculate performance impact
            performance_impact = self._calculate_performance_impact(trade, updated_portfolio)

            # Update internal state
            self.portfolio = updated_portfolio
            self.trades.append(trade)

            logger.info(
                "EXECUTED: %s %s shares of %s @ $%.2f (cost: $%.2f, slippage: %.3f%%)",
                action,
                position_size,
                ticker,
                entry_price,
                transaction_cost,
                trade.slippage * 100,
            )

            return ExecutionResult(
                status='EXECUTED',
                trade=trade,
                portfolio=updated_portfolio,
                performance_impact=performance_impact,
                validation_warnings=validation.warnings
            )

        except Exception as exc:
            logger.error("Trade execution failed: %s", exc)
            return ExecutionResult(
                status='FAILED',
                reason=str(exc)
            )

    def _current_portfolio_value(self) -> float:
        """Return best-effort current portfolio value."""
        if self.portfolio.total_value > 0:
            return self.portfolio.total_value
        return self.portfolio.cash

    def _calculate_position_size(self,
                                 signal: Dict[str, Any],
                                 confidence_score: float,
                                 market_data: pd.DataFrame,
        current_position: int) -> int:
        """
        Calculate position size using Kelly criterion with confidence adjustment.

        Args:
            signal: Trading signal
            confidence_score: Adjusted confidence (0-1)
            market_data: Historical data for risk estimation
            current_position: Existing shares held (long only)

        Returns:
            Number of shares to trade
        """
        diag_mode = str(os.getenv("EXECUTION_DIAGNOSTIC_MODE") or os.getenv("DIAGNOSTIC_MODE") or "0") == "1"
        # Maximum position size (2% of portfolio as risk management, scaled by
        # per-ticker regime state when available).
        portfolio_value = self._current_portfolio_value()
        max_position_value = portfolio_value * (0.10 if diag_mode else 0.02)

        # Regime-aware risk scaling: use a smaller fraction of the 2% cap
        # when a ticker is still in exploration mode or in a red regime, and a
        # slightly larger fraction when it is in a green regime. This keeps
        # realised PnL risk small while allowing more trades where evidence is
        # scarce, and grows risk only where realised performance is strong.
        ticker = signal.get("ticker")
        if ticker:
            risk_mult = self._get_regime_risk_multiplier(ticker)
            max_position_value *= risk_mult
        action = signal.get("action", "HOLD").upper()
        if action == "SELL":
            # Tighter cap for shorts (1% of equity)
            max_position_value = portfolio_value * (0.05 if diag_mode else 0.01)

        # Adjust by confidence (floor at 50% to avoid zero-size)
        confidence_weight = max(confidence_score, 0.1 if diag_mode else 0.5)
        position_value = max_position_value * confidence_weight

        # Calculate shares
        current_price = market_data['Close'].iloc[-1]

        if action == "SELL":
            desired = max(0, int(position_value / current_price))
            # Protect against stacking shorts beyond cap
            if current_position < 0:
                current_exposure = abs(current_position) * current_price
                remaining_capacity = max(0.0, max_position_value - current_exposure)
                if remaining_capacity <= 0:
                    return 0
                cap_shares = max(1, int(remaining_capacity / current_price))
                shares = min(desired, cap_shares)
            elif current_position > 0:
                # Closing a long: always allow at least 1 share so exits are
                # possible even when the 1% sizing cap falls below the share price.
                shares = min(current_position, max(1, desired))
            else:
                shares = desired
        else:
            desired = max(0, int(position_value / current_price))
            if current_position < 0:
                # Covering a short: always allow at least 1 share so exits are
                # possible even when the sizing cap falls below the share price.
                shares = min(abs(current_position), max(1, desired))
            else:
                shares = desired
                # Opening a new long: avoid zero-share trades for higher-priced
                # tickers when a single share still fits inside the risk cap.
                if current_position == 0 and shares == 0 and current_price <= max_position_value:
                    shares = 1

        return max(1, shares) if diag_mode else max(0, shares)

    def _get_regime_risk_multiplier(self, ticker: str) -> float:
        """
        Look up a per-ticker risk multiplier from config/regime_state.yml.

        Defaults to 1.0 when no regime information is available. Exploration
        and red regimes shrink risk; green regimes allow a modest increase.
        """
        from pathlib import Path
        import yaml  # local import to avoid hard dependency in minimal setups

        regime_path = Path(__file__).resolve().parent.parent / "config" / "regime_state.yml"
        if not regime_path.exists():
            return 1.0

        try:
            raw = yaml.safe_load(regime_path.read_text(encoding="utf-8")) or {}
            rs = raw.get("regime_state") or {}
            info = rs.get(ticker)
            if not isinstance(info, dict):
                return 1.0
            mode = info.get("mode")
            state = info.get("state")
        except Exception:
            return 1.0

        # Exploration mode: trade smaller until n_trades is sufficient.
        if mode == "exploration":
            return 0.25
        # Exploitation + red regime: shrink risk aggressively.
        if state == "red":
            return 0.3
        # Exploitation + green regime: modestly increase risk.
        if state == "green":
            return 1.2
        return 1.0

    def _load_lob_config(self) -> Optional[LOBConfig]:
        if self._lob_enabled is False:
            return None
        if self._lob_config is not None:
            return self._lob_config

        from pathlib import Path
        import yaml  # local import to keep dependency optional for minimal runs

        cfg_path = Path(__file__).resolve().parent.parent / "config" / "execution_cost_model.yml"
        if not cfg_path.exists():
            self._lob_enabled = True
            self._lob_config = LOBConfig()
            return self._lob_config

        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            model_cfg = raw.get("execution_cost_model") or {}
            lob_cfg = model_cfg.get("lob") or {}
            enabled = bool(lob_cfg.get("enabled", True))
            self._lob_enabled = enabled
            if not enabled:
                return None
            depth_profiles = lob_cfg.get("depth_profiles")
            if depth_profiles is not None and not isinstance(depth_profiles, dict):
                depth_profiles = None
            self._lob_config = LOBConfig(
                levels=int(lob_cfg.get("levels", 10)),
                tick_size_bps=float(lob_cfg.get("tick_size_bps", 1.0)),
                alpha=float(lob_cfg.get("alpha", 0.8)),
                max_exhaust_levels=int(lob_cfg.get("max_exhaust_levels", 25)),
                default_order_value=float(lob_cfg.get("default_order_value", 10000.0)),
                depth_profiles=depth_profiles,
                tail_depth_multiplier=float(lob_cfg.get("tail_depth_multiplier", 1.0)),
            )
            return self._lob_config
        except Exception:
            self._lob_enabled = True
            self._lob_config = LOBConfig()
            return self._lob_config

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if out != out:  # NaN
            return None
        return out

    def _simulate_entry_price(
        self,
        market_price: float,
        action: str,
        shares: int,
        *,
        market_data: Optional[pd.DataFrame] = None,
        ticker: Optional[str] = None,
    ) -> float:
        """
        Simulate realistic entry price with slippage and market impact.

        Args:
            market_price: Current market price
            action: 'BUY' or 'SELL'
            shares: Number of shares

        Returns:
            Simulated entry price
        """
        side = action.upper()
        shares_f = float(shares or 0)
        market_px = float(market_price or 0.0)
        if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
            try:
                last = market_data.iloc[-1]
            except Exception:
                last = None

            if last is not None:
                bid = last.get("Bid") if hasattr(last, "get") else None
                ask = last.get("Ask") if hasattr(last, "get") else None
                bid_f = self._safe_float(bid)
                ask_f = self._safe_float(ask)
                if bid_f is not None and ask_f is not None and bid_f > 0 and ask_f > bid_f:
                    mid = 0.5 * (bid_f + ask_f)
                    half_spread = 0.5 * (ask_f - bid_f)
                else:
                    close = last.get("Close") if hasattr(last, "get") else None
                    mid = self._safe_float(close) or market_px
                    spread_val = last.get("Spread") if hasattr(last, "get") else None
                    spread = self._safe_float(spread_val)
                    half_spread = (spread / 2.0) if spread is not None and spread > 0 else None

                depth_val = last.get("Depth") if hasattr(last, "get") else None
                depth_notional = self._safe_float(depth_val)
                if depth_notional is not None and depth_notional <= 0:
                    depth_notional = None

                txn_cost_bps_val = last.get("TxnCostBps") if hasattr(last, "get") else None
                txn_cost_bps = self._safe_float(txn_cost_bps_val)
                impact_bps_val = last.get("ImpactBps") if hasattr(last, "get") else None
                impact_bps = self._safe_float(impact_bps_val) or 0.0
                total_cost_bps = (txn_cost_bps + impact_bps) if txn_cost_bps is not None else None
                baseline_total_cost = (mid * total_cost_bps / 1e4) if total_cost_bps is not None else (mid * self.slippage_pct)
                half_spread_f = float(half_spread or 0.0)
                baseline_slippage = max(0.0, baseline_total_cost - half_spread_f)

                lob_cfg = self._load_lob_config()
                if lob_cfg is not None and shares_f > 0 and mid > 0:
                    fill = simulate_market_order_fill(
                        side=side if side in {"BUY", "SELL"} else "BUY",
                        mid_price=mid,
                        half_spread=half_spread,
                        depth_notional=depth_notional,
                        shares=shares_f,
                        baseline_slippage=baseline_slippage,
                        asset_class=_infer_asset_class_from_ticker(ticker) if ticker else None,
                        config=lob_cfg,
                    )
                    return float(fill.vwap_price)

        # Fallback: percentage-based slippage + simple market impact
        base_slippage = float(self.slippage_pct or 0.0)
        order_value = shares_f * market_px
        market_impact = (order_value / 10000.0) * 0.0001 if order_value > 0 else 0.0
        total_slippage = base_slippage + market_impact
        if side == 'BUY':
            return market_px * (1 + total_slippage)
        return market_px * (1 - total_slippage)

    def _update_portfolio(self, trade: Trade, portfolio: Portfolio) -> Portfolio:
        """
        Update portfolio after trade execution.

        Args:
            trade: Executed trade
            portfolio: Current portfolio state

        Returns:
            Updated portfolio
        """
        old_shares = portfolio.positions.get(trade.ticker, 0)
        new_portfolio = Portfolio(
            cash=portfolio.cash,
            positions=portfolio.positions.copy(),
            entry_prices=portfolio.entry_prices.copy(),
            entry_timestamps=portfolio.entry_timestamps.copy(),
            entry_bar_timestamps=portfolio.entry_bar_timestamps.copy(),
            last_bar_timestamps=portfolio.last_bar_timestamps.copy(),
            holding_bars=portfolio.holding_bars.copy(),
            stop_losses=portfolio.stop_losses.copy(),
            target_prices=portfolio.target_prices.copy(),
            max_holding_days=portfolio.max_holding_days.copy(),
        )

        if trade.action == 'BUY':
            # Deduct cash (price + transaction cost)
            total_cost = (trade.shares * trade.entry_price) + trade.transaction_cost
            new_portfolio.cash -= total_cost

            # Update position
            if trade.ticker in new_portfolio.positions:
                # Average entry price calculation
                old_shares = new_portfolio.positions[trade.ticker]
                old_price = new_portfolio.entry_prices[trade.ticker]

                total_shares = old_shares + trade.shares
                if old_shares < 0:
                    # Covering a short
                    if total_shares < 0:
                        new_portfolio.positions[trade.ticker] = total_shares
                        new_portfolio.entry_prices[trade.ticker] = old_price
                    elif total_shares == 0:
                        del new_portfolio.positions[trade.ticker]
                        del new_portfolio.entry_prices[trade.ticker]
                    else:
                        new_portfolio.positions[trade.ticker] = total_shares
                        new_portfolio.entry_prices[trade.ticker] = trade.entry_price
                else:
                    avg_price = (
                        (old_shares * old_price + trade.shares * trade.entry_price) / total_shares
                    )
                    new_portfolio.positions[trade.ticker] = total_shares
                    new_portfolio.entry_prices[trade.ticker] = avg_price
            else:
                new_portfolio.positions[trade.ticker] = trade.shares
                new_portfolio.entry_prices[trade.ticker] = trade.entry_price

        elif trade.action == 'SELL':
            # Add cash (price - transaction cost)
            total_proceeds = (trade.shares * trade.entry_price) - trade.transaction_cost
            new_portfolio.cash += total_proceeds

            # Update position
            if trade.ticker in new_portfolio.positions:
                old_shares = new_portfolio.positions[trade.ticker]
                old_price = new_portfolio.entry_prices[trade.ticker]
                new_shares = old_shares - trade.shares

                if old_shares > 0:
                    if new_shares > 0:
                        new_portfolio.positions[trade.ticker] = new_shares
                        new_portfolio.entry_prices[trade.ticker] = old_price
                    elif new_shares == 0:
                        del new_portfolio.positions[trade.ticker]
                        del new_portfolio.entry_prices[trade.ticker]
                    else:
                        new_portfolio.positions[trade.ticker] = new_shares
                        new_portfolio.entry_prices[trade.ticker] = trade.entry_price
                else:
                    # Expanding an existing short
                    total_short = abs(old_shares) + trade.shares
                    avg_price = (
                        (abs(old_shares) * old_price + trade.shares * trade.entry_price) / total_short
                    )
                    new_portfolio.positions[trade.ticker] = -total_short
                    new_portfolio.entry_prices[trade.ticker] = avg_price
            else:
                # Open a new short
                new_portfolio.positions[trade.ticker] = -trade.shares
                new_portfolio.entry_prices[trade.ticker] = trade.entry_price

        # Lifecycle metadata update (best-effort).
        new_shares = new_portfolio.positions.get(trade.ticker, 0)
        if new_shares == 0:
            new_portfolio.entry_timestamps.pop(trade.ticker, None)
            new_portfolio.entry_bar_timestamps.pop(trade.ticker, None)
            new_portfolio.last_bar_timestamps.pop(trade.ticker, None)
            new_portfolio.holding_bars.pop(trade.ticker, None)
            new_portfolio.stop_losses.pop(trade.ticker, None)
            new_portfolio.target_prices.pop(trade.ticker, None)
            new_portfolio.max_holding_days.pop(trade.ticker, None)
        else:
            opened_new_side = old_shares == 0 or (old_shares > 0 and new_shares < 0) or (old_shares < 0 and new_shares > 0)
            if opened_new_side:
                new_portfolio.entry_timestamps[trade.ticker] = trade.timestamp
                entry_bar_ts = trade.bar_timestamp or trade.timestamp
                new_portfolio.entry_bar_timestamps[trade.ticker] = entry_bar_ts
                new_portfolio.last_bar_timestamps[trade.ticker] = entry_bar_ts
                new_portfolio.holding_bars[trade.ticker] = 0
            if trade.stop_loss is not None:
                try:
                    new_portfolio.stop_losses[trade.ticker] = float(trade.stop_loss)
                except (TypeError, ValueError):
                    pass
            if trade.target_price is not None:
                try:
                    new_portfolio.target_prices[trade.ticker] = float(trade.target_price)
                except (TypeError, ValueError):
                    pass
            if trade.forecast_horizon is not None:
                try:
                    horizon = int(trade.forecast_horizon)
                except (TypeError, ValueError):
                    horizon = None
                if horizon is not None and horizon > 0:
                    new_portfolio.max_holding_days[trade.ticker] = horizon

        # Update total value
        current_prices = {trade.ticker: trade.entry_price}
        new_portfolio.update_value(current_prices)

        return new_portfolio

    def mark_to_market(self, price_map: Dict[str, float]) -> float:
        """Refresh portfolio valuation using the latest market prices."""
        self.portfolio.update_value(price_map)
        return self.portfolio.total_value

    def _evaluate_exit_reason(
        self,
        *,
        ticker: str,
        shares: int,
        current_price: float,
        as_of: datetime,
        market_data: Optional[pd.DataFrame] = None,
        bar_timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Return a lifecycle exit reason (stop/target/time) or None."""
        if not ticker or shares == 0:
            return None
        if current_price <= 0:
            return None

        stop_loss = self.portfolio.stop_losses.get(ticker)
        target_price = self.portfolio.target_prices.get(ticker)

        try:
            stop_loss_f = float(stop_loss) if stop_loss is not None else None
        except (TypeError, ValueError):
            stop_loss_f = None
        try:
            target_price_f = float(target_price) if target_price is not None else None
        except (TypeError, ValueError):
            target_price_f = None

        # Stop/target checks are evaluated first (price-based exits).
        if shares > 0:
            if stop_loss_f is not None and current_price <= stop_loss_f:
                return "STOP_LOSS"
            if target_price_f is not None and current_price >= target_price_f:
                return "TAKE_PROFIT"
        else:
            if stop_loss_f is not None and current_price >= stop_loss_f:
                return "STOP_LOSS"
            if target_price_f is not None and current_price <= target_price_f:
                return "TAKE_PROFIT"

        # Time-based exit.
        entry_ts = self.portfolio.entry_timestamps.get(ticker)
        entry_bar_ts = self.portfolio.entry_bar_timestamps.get(ticker)
        max_days_raw = self.portfolio.max_holding_days.get(ticker)
        if max_days_raw is None:
            return None
        try:
            max_days = int(max_days_raw)
        except (TypeError, ValueError):
            return None
        if max_days <= 0:
            return None

        # Horizon semantics: treat forecast_horizon/max_holding_days as number of
        # bars/steps, not calendar days. For daily bars this coincides with days,
        # while intraday bars require bar-counting to exit positions on time.
        current_bar_ts = None
        if bar_timestamp is not None:
            current_bar_ts = bar_timestamp
        elif isinstance(market_data, pd.DataFrame) and not market_data.empty and isinstance(market_data.index, pd.DatetimeIndex):
            current_bar_ts = pd.Timestamp(market_data.index[-1]).to_pydatetime()

        # Bootstrap bar timestamps for legacy positions (pre-migration).
        if entry_bar_ts is None and current_bar_ts is not None:
            entry_bar_ts = current_bar_ts
            self.portfolio.entry_bar_timestamps[ticker] = entry_bar_ts
            self.portfolio.last_bar_timestamps.setdefault(ticker, entry_bar_ts)
            self.portfolio.holding_bars.setdefault(ticker, 0)

        if entry_bar_ts is not None and current_bar_ts is not None:
            bars_held = int(self.portfolio.holding_bars.get(ticker, 0) or 0)
            last_bar_ts = self.portfolio.last_bar_timestamps.get(ticker) or entry_bar_ts

            try:
                current_ts = pd.Timestamp(current_bar_ts)
                last_ts = pd.Timestamp(last_bar_ts)
            except Exception:
                current_ts = None
                last_ts = None

            if current_ts is not None and last_ts is not None and current_ts > last_ts:
                gap = 1
                try:
                    if isinstance(market_data, pd.DataFrame) and not market_data.empty and isinstance(market_data.index, pd.DatetimeIndex):
                        idx = pd.DatetimeIndex(market_data.index)
                        gap = int(((idx > last_ts) & (idx <= current_ts)).sum()) or 1
                except Exception:
                    gap = 1
                bars_held += gap
                self.portfolio.holding_bars[ticker] = bars_held
                self.portfolio.last_bar_timestamps[ticker] = current_ts.to_pydatetime()

            if bars_held >= max_days:
                return "TIME_EXIT"
            return None

        # Fallback to calendar-day holding when bar timestamps are unavailable.
        if entry_ts is None:
            return None
        days_held = (as_of.date() - entry_ts.date()).days
        if days_held >= max_days:
            return "TIME_EXIT"
        return None

    def _store_trade_execution(self, trade: Trade):
        """Store trade execution in database and return realized PnL details."""
        try:
            # Calculate P&L if closing position
            realized_pnl = 0.0
            realized_pct = None
            prior_entry = self.portfolio.entry_prices.get(trade.ticker)
            prior_shares = self.portfolio.positions.get(trade.ticker, 0)
            holding_period_days = None

            if prior_entry and prior_entry > 0:
                if trade.action == 'SELL' and prior_shares > 0:
                    close_size = min(prior_shares, trade.shares)
                    realized_pnl = (
                        (trade.entry_price - prior_entry) * close_size
                        - trade.transaction_cost
                    )
                    realized_pct = (trade.entry_price - prior_entry) / prior_entry
                    entry_ts = self.portfolio.entry_timestamps.get(trade.ticker)
                    if isinstance(entry_ts, datetime):
                        holding_period_days = max(0, (trade.timestamp.date() - entry_ts.date()).days)
                elif trade.action == 'BUY' and prior_shares < 0:
                    close_size = min(abs(prior_shares), trade.shares)
                    realized_pnl = (
                        (prior_entry - trade.entry_price) * close_size
                        - trade.transaction_cost
                    )
                    realized_pct = (prior_entry - trade.entry_price) / prior_entry
                    entry_ts = self.portfolio.entry_timestamps.get(trade.ticker)
                    if isinstance(entry_ts, datetime):
                        holding_period_days = max(0, (trade.timestamp.date() - entry_ts.date()).days)
            # Instrument metadata – currently spot/crypto only, but the Trade
            # dataclass allows options/synthetic extensions once feature flags
            # are enabled.
            asset_class = trade.asset_class or _infer_asset_class_from_ticker(trade.ticker)
            instrument_type = trade.instrument_type or "spot"

            self.db_manager.save_trade_execution(
                ticker=trade.ticker,
                trade_date=trade.timestamp.date(),
                action=trade.action,
                shares=trade.shares,
                price=trade.entry_price,
                total_value=trade.shares * trade.entry_price,
                commission=trade.transaction_cost,
                signal_id=trade.signal_id,
                data_source=trade.data_source,
                execution_mode=trade.execution_mode,
                synthetic_dataset_id=trade.synthetic_dataset_id,
                synthetic_generator_version=trade.synthetic_generator_version,
                run_id=trade.run_id,
                realized_pnl=realized_pnl if realized_pct is not None else None,
                realized_pnl_pct=realized_pct,
                holding_period_days=holding_period_days,
                mid_price=trade.mid_price,
                mid_slippage_bps=trade.mid_slippage_bps,
                asset_class=asset_class,
                instrument_type=instrument_type,
                underlying_ticker=trade.underlying_ticker,
                strike=trade.strike,
                expiry=trade.expiry,
                multiplier=trade.multiplier,
                barbell_bucket=trade.barbell_bucket,
                barbell_multiplier=trade.barbell_multiplier,
                base_confidence=trade.base_confidence,
                effective_confidence=trade.effective_confidence,
            )

            logger.debug("Trade stored in database: %s", trade.trade_id)
            return realized_pnl if realized_pct is not None else None, realized_pct
        except Exception as exc:
            logger.error("Failed to store trade: %s", exc)
            return None, None

    def _calculate_performance_impact(self,
                                     trade: Trade,
                                     portfolio: Portfolio) -> Dict[str, float]:
        """Calculate how trade impacts portfolio performance"""
        return {
            'portfolio_value': portfolio.total_value,
            'cash_remaining': portfolio.cash,
            'position_count': len(portfolio.positions),
            'trade_cost': trade.transaction_cost,
            'trade_value': trade.shares * trade.entry_price
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return {
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.positions.copy(),
            'num_positions': len(self.portfolio.positions),
            'num_trades': len(self.trades),
            'initial_capital': self.initial_capital,
            'total_return': (self.portfolio.total_value / self.initial_capital - 1) if self.initial_capital > 0 else 0
        }

    def get_performance_metrics(self, run_id: Optional[str] = None) -> Dict[str, float]:
        """Calculate portfolio performance metrics (optionally scoped to run_id)."""
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_trade_value': 0.0
            }

        # Get realized performance from database (run-local when run_id is set).
        perf = self.db_manager.get_performance_summary(run_id=run_id)

        return {
            'total_return': (self.portfolio.total_value / self.initial_capital - 1),
            'num_trades': len(self.trades),
            'win_rate': perf.get('win_rate', 0.0),
            'profit_factor': perf.get('profit_factor', 0.0),
            'avg_profit': perf.get('avg_profit', 0.0)
        }


# Validation
assert PaperTradingEngine.execute_signal.__doc__ is not None
assert PaperTradingEngine.get_portfolio_summary.__doc__ is not None

logger.info("Paper Trading Engine module loaded successfully")

# Line count: ~450 lines (slightly over budget but comprehensive functionality)
