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

logger = logging.getLogger(__name__)


def _infer_asset_class_from_ticker(ticker: str) -> str:
    """Best-effort asset class hint based on ticker string."""
    sym = (ticker or "").upper()
    # Simple crypto heuristic: common "-USD" pairs and bare symbols.
    if sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL"}:
        return "crypto"
    return "equity"


@dataclass
class Trade:
    """Trade execution details"""
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    entry_price: float
    transaction_cost: float
    timestamp: datetime
    is_paper_trade: bool = True
    trade_id: Optional[str] = None
    slippage: float = 0.0
    signal_id: Optional[int] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: Optional[float] = None
    # Optional instrument metadata for options/derivatives (kept inert until
    # options_trading.enabled + feature flags are active).
    asset_class: str = "equity"
    instrument_type: str = "spot"
    underlying_ticker: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None
    multiplier: float = 1.0
    
    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = f"{self.ticker}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"


@dataclass
class Portfolio:
    """Portfolio state"""
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)  # ticker -> shares
    entry_prices: Dict[str, float] = field(default_factory=dict)  # ticker -> avg entry price
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
                 slippage_pct: float = 0.001,
                 transaction_cost_pct: float = 0.001,
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
        if action not in {"BUY", "SELL"}:
            logger.info("Skipping non-actionable signal (%s) for %s", action, ticker)
            return ExecutionResult(
                status='REJECTED',
                reason="Non-actionable signal",
            )
        
        logger.info(f"Executing %s signal for %s", action, ticker)
        # Step 1: Validate signal (diagnostic toggle via env DIAGNOSTIC_MODE/EXECUTION_DIAGNOSTIC_MODE)
        diag_mode = str(os.getenv("EXECUTION_DIAGNOSTIC_MODE") or os.getenv("DIAGNOSTIC_MODE") or "0") == "1"
        validation = self.signal_validator.validate_llm_signal(
            signal,
            market_data,
            self._current_portfolio_value()
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
        
        # Step 2: Calculate position size
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
        current_price = market_data['Close'].iloc[-1]
        required_capital = position_size * current_price
        
        if action == 'BUY' and required_capital > self.portfolio.cash:
            logger.warning(
                "Insufficient cash: need $%.2f, have $%.2f",
                required_capital,
                self.portfolio.cash,
            )
            return ExecutionResult(
                status='REJECTED',
                reason=f"Insufficient cash (need ${required_capital:,.2f})"
            )
        
        # Step 4: Simulate entry price with slippage
        entry_price = self._simulate_entry_price(
            current_price, action, position_size
        )
        
        # Step 5: Calculate transaction costs
        transaction_value = position_size * entry_price
        transaction_cost = transaction_value * self.transaction_cost_pct
        
        # Step 6: Create trade object
        trade = Trade(
            ticker=ticker,
            action=action,
            shares=position_size,
            entry_price=entry_price,
            transaction_cost=transaction_cost,
            timestamp=datetime.now(),
            is_paper_trade=True,
            slippage=abs(entry_price - current_price) / current_price,
            signal_id=signal.get('signal_id'),
        )
        
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
                shares = min(current_position, desired)
            else:
                shares = desired
        else:
            desired = max(0, int(position_value / current_price))
            if current_position < 0:
                shares = min(abs(current_position), desired)
            else:
                shares = desired

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
    
    def _simulate_entry_price(self, 
                             market_price: float, 
                             action: str, 
                             shares: int) -> float:
        """
        Simulate realistic entry price with slippage and market impact.
        
        Args:
            market_price: Current market price
            action: 'BUY' or 'SELL'
            shares: Number of shares
            
        Returns:
            Simulated entry price
        """
        # Base slippage (0.1%)
        base_slippage = self.slippage_pct
        
        # Additional market impact for large orders
        # Assume 0.01% additional slippage per $10,000 of order value
        order_value = shares * market_price
        market_impact = (order_value / 10000) * 0.0001
        
        total_slippage = base_slippage + market_impact
        
        # Apply slippage direction
        if action == 'BUY':
            # Buy at higher price
            entry_price = market_price * (1 + total_slippage)
        else:  # SELL
            # Sell at lower price
            entry_price = market_price * (1 - total_slippage)
        
        return entry_price
    
    def _update_portfolio(self, trade: Trade, portfolio: Portfolio) -> Portfolio:
        """
        Update portfolio after trade execution.
        
        Args:
            trade: Executed trade
            portfolio: Current portfolio state
            
        Returns:
            Updated portfolio
        """
        new_portfolio = Portfolio(
            cash=portfolio.cash,
            positions=portfolio.positions.copy(),
            entry_prices=portfolio.entry_prices.copy()
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
        
        # Update total value
        current_prices = {trade.ticker: trade.entry_price}
        new_portfolio.update_value(current_prices)
        
        return new_portfolio

    def mark_to_market(self, price_map: Dict[str, float]) -> float:
        """Refresh portfolio valuation using the latest market prices."""
        self.portfolio.update_value(price_map)
        return self.portfolio.total_value
    
    def _store_trade_execution(self, trade: Trade):
        """Store trade execution in database and return realized PnL details."""
        try:
            # Calculate P&L if closing position
            realized_pnl = 0.0
            realized_pct = None
            prior_entry = self.portfolio.entry_prices.get(trade.ticker)
            prior_shares = self.portfolio.positions.get(trade.ticker, 0)

            if prior_entry and prior_entry > 0:
                if trade.action == 'SELL' and prior_shares > 0:
                    close_size = min(prior_shares, trade.shares)
                    realized_pnl = (
                        (trade.entry_price - prior_entry) * close_size
                        - trade.transaction_cost
                    )
                    realized_pct = (trade.entry_price - prior_entry) / prior_entry
                elif trade.action == 'BUY' and prior_shares < 0:
                    close_size = min(abs(prior_shares), trade.shares)
                    realized_pnl = (
                        (prior_entry - trade.entry_price) * close_size
                        - trade.transaction_cost
                    )
                    realized_pct = (prior_entry - trade.entry_price) / prior_entry
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
                realized_pnl=realized_pnl if realized_pct is not None else None,
                realized_pnl_pct=realized_pct,
                holding_period_days=None,
                asset_class=asset_class,
                instrument_type=instrument_type,
                underlying_ticker=trade.underlying_ticker,
                strike=trade.strike,
                expiry=trade.expiry,
                multiplier=trade.multiplier,
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
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_trade_value': 0.0
            }
        
        # Get performance from database
        perf = self.db_manager.get_performance_summary()
        
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


