#!/usr/bin/env python3
"""
LLM Signal Backtesting Framework (Phase 5.2)

Validates LLM-generated trading signals against historical data.
Per AGENT_INSTRUCTION.md: NO TRADING until >10% annual returns with 30+ days validation.

Usage:
    python scripts/backtest_llm_signals.py --signals data/llm_signals.json --data data/training/*.parquet
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import click

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.portfolio_math import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMSignalBacktester:
    """Backtest LLM trading signals against historical data.

    Mathematical Foundation:
    - Returns: R_t = ln(P_t / P_{t-1})
    - Cumulative: R_cum = ∏(1 + R_t) - 1
    - Sharpe Ratio: SR = (μ_R - r_f) / σ_R
    - Max Drawdown: MDD = max(1 - P_t / max(P_{0:t}))

    Success Criteria (per AGENT_INSTRUCTION.md):
    - Annual return > 10%
    - Beats buy-and-hold baseline
    - Validation period >= 30 days
    - Sharpe ratio > 0 (positive risk-adjusted return)
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Args:
            initial_capital: Starting portfolio value in dollars
        """
        self.initial_capital = initial_capital
        self.results = {}

    def load_signals(self, signals_path: str) -> Dict[str, List[Dict]]:
        """Load LLM-generated signals from JSON file.

        Expected format:
        {
            "AAPL": [
                {"date": "2023-01-01", "action": "BUY", "confidence": 0.75},
                ...
            ],
            ...
        }
        """
        logger.info(f"Loading signals from: {signals_path}")

        with open(signals_path, 'r') as f:
            signals = json.load(f)

        signal_counts = {ticker: len(sigs) for ticker, sigs in signals.items()}
        logger.info(f"Loaded signals: {signal_counts}")

        return signals

    def load_price_data(self, data_path: str) -> pd.DataFrame:
        """Load historical price data from parquet file."""
        logger.info(f"Loading price data from: {data_path}")

        data = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")

        return data

    def execute_signal(self, action: str, price: float, position: int,
                       confidence: float, capital: float) -> Tuple[int, float]:
        """Execute a trading signal.

        Args:
            action: 'BUY', 'SELL', or 'HOLD'
            price: Current market price
            position: Current position size (shares)
            confidence: Signal confidence (0-1)
            capital: Available cash

        Returns:
            (new_position, new_capital)
        """
        if action == 'BUY' and capital >= price:
            # Buy shares based on confidence
            shares_to_buy = int((capital * confidence * 0.95) / price)  # 95% to leave buffer
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                return position + shares_to_buy, capital - cost

        elif action == 'SELL' and position > 0:
            # Sell shares based on confidence
            shares_to_sell = int(position * confidence)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                return position - shares_to_sell, capital + proceeds

        # HOLD or unable to execute
        return position, capital

    def backtest_ticker(self, ticker: str, signals: List[Dict],
                       price_data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest signals for a single ticker.

        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"\nBacktesting {ticker}...")

        # Get ticker-specific data
        if isinstance(price_data.index, pd.MultiIndex):
            ticker_data = price_data.xs(ticker, level=0)
        else:
            ticker_data = price_data

        # Ensure we have Close prices
        if 'Close' not in ticker_data.columns:
            logger.error(f"  ✗ No 'Close' column found for {ticker}")
            return None

        # Initialize portfolio
        capital = self.initial_capital
        position = 0
        portfolio_values = []
        trade_log = []

        # Convert signals to DataFrame for easy lookup
        signal_df = pd.DataFrame(signals)
        if 'date' in signal_df.columns:
            signal_df['date'] = pd.to_datetime(signal_df['date'])
            signal_df = signal_df.set_index('date')

        # Simulate trading
        for date, row in ticker_data.iterrows():
            price = row['Close']

            # Check for signal on this date
            if date in signal_df.index:
                signal = signal_df.loc[date]
                action = signal['action']
                confidence = signal.get('confidence', 0.5)

                # Execute signal
                old_position = position
                position, capital = self.execute_signal(
                    action, price, position, confidence, capital
                )

                if position != old_position:
                    trade_log.append({
                        'date': date,
                        'action': action,
                        'price': price,
                        'shares': abs(position - old_position),
                        'confidence': confidence
                    })

            # Calculate portfolio value
            portfolio_value = capital + (position * price)
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'capital': capital,
                'position': position,
                'price': price
            })

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate performance metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Calculate buy-and-hold baseline
        initial_price = ticker_data['Close'].iloc[0]
        final_price = ticker_data['Close'].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price

        # Calculate daily returns
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        daily_returns = portfolio_df['returns'].dropna()

        # Annualized metrics
        trading_days = len(portfolio_df)
        years = trading_days / 252  # Approximate trading days per year
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

        # Risk metrics
        sharpe = calculate_sharpe_ratio(daily_returns.values) if len(daily_returns) > 0 else 0
        max_dd = calculate_max_drawdown(portfolio_df['value'].values) if len(portfolio_df) > 0 else 0

        # Win rate
        winning_trades = sum(1 for t in trade_log if t['action'] == 'SELL')
        total_trades = len(trade_log)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        results = {
            'ticker': ticker,
            'initial_value': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'buy_hold_return': buy_hold_return,
            'alpha': annual_return - buy_hold_return,  # Excess return vs buy-hold
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'validation_days': trading_days,
            'portfolio_history': portfolio_df,
            'trade_log': trade_log
        }

        # Log results
        logger.info(f"  Initial: ${self.initial_capital:,.0f}")
        logger.info(f"  Final: ${final_value:,.0f}")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Buy-Hold Return: {buy_hold_return:.2%}")
        logger.info(f"  Alpha (excess): {results['alpha']:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"  Max Drawdown: {max_dd:.2%}")
        logger.info(f"  Trades: {total_trades}, Win Rate: {win_rate:.1%}")
        logger.info(f"  Validation Period: {trading_days} days")

        # Validation check (per AGENT_INSTRUCTION.md)
        passed_validation = (
            annual_return > 0.10 and  # >10% annual return
            results['alpha'] > 0 and   # Beats buy-and-hold
            trading_days >= 30         # At least 30 days
        )

        if passed_validation:
            logger.info(f"  ✅ VALIDATION PASSED - Ready for paper trading")
        else:
            logger.warning(f"  ❌ VALIDATION FAILED - Not ready for trading")
            if annual_return <= 0.10:
                logger.warning(f"     - Annual return {annual_return:.2%} <= 10%")
            if results['alpha'] <= 0:
                logger.warning(f"     - Did not beat buy-and-hold (alpha: {results['alpha']:.2%})")
            if trading_days < 30:
                logger.warning(f"     - Validation period {trading_days} days < 30")

        results['passed_validation'] = passed_validation

        return results

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate comprehensive backtest report."""
        report = []
        report.append("=" * 80)
        report.append("LLM SIGNAL BACKTEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Initial Capital: ${self.initial_capital:,.0f}")
        report.append("")

        # Summary table
        report.append("SUMMARY BY TICKER")
        report.append("-" * 80)
        report.append(f"{'Ticker':<8} {'Total%':>10} {'Annual%':>10} {'Alpha%':>10} {'Sharpe':>8} {'MaxDD%':>10} {'Trades':>8} {'Valid':>8}")
        report.append("-" * 80)

        total_passed = 0
        for ticker, result in results.items():
            if result is None:
                continue

            report.append(
                f"{result['ticker']:<8} "
                f"{result['total_return']*100:>10.2f} "
                f"{result['annual_return']*100:>10.2f} "
                f"{result['alpha']*100:>10.2f} "
                f"{result['sharpe_ratio']:>8.3f} "
                f"{result['max_drawdown']*100:>10.2f} "
                f"{result['total_trades']:>8} "
                f"{'✅' if result['passed_validation'] else '❌':>8}"
            )

            if result['passed_validation']:
                total_passed += 1

        report.append("-" * 80)
        report.append(f"Tickers Passed Validation: {total_passed}/{len(results)}")
        report.append("")

        # Validation criteria
        report.append("VALIDATION CRITERIA (per AGENT_INSTRUCTION.md)")
        report.append("-" * 80)
        report.append("✓ Annual Return > 10%")
        report.append("✓ Alpha > 0 (beats buy-and-hold)")
        report.append("✓ Validation Period >= 30 days")
        report.append("✓ Sharpe Ratio > 0 (positive risk-adjusted return)")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)

        if total_passed > 0:
            report.append("✅ Some signals passed validation:")
            for ticker, result in results.items():
                if result and result['passed_validation']:
                    report.append(f"   - {ticker}: {result['annual_return']:.2%} annual, {result['sharpe_ratio']:.3f} Sharpe")
            report.append("")
            report.append("NEXT STEPS:")
            report.append("1. Proceed to paper trading with validated signals")
            report.append("2. Monitor performance for additional 30 days")
            report.append("3. Verify $25K+ capital requirement before live trading")
        else:
            report.append("❌ No signals passed validation")
            report.append("")
            report.append("NEXT STEPS:")
            report.append("1. Review LLM prompts and parameters")
            report.append("2. Collect more training data")
            report.append("3. Consider ensemble methods or model fine-tuning")
            report.append("4. DO NOT proceed to live trading")

        report.append("=" * 80)

        return "\n".join(report)


@click.command()
@click.option('--signals', required=True, help='Path to LLM signals JSON file')
@click.option('--data', required=True, help='Path to historical price data (parquet)')
@click.option('--capital', default=100000.0, help='Initial capital (default: $100,000)')
@click.option('--output', default='backtest_results.json', help='Output results file')
@click.option('--verbose', is_flag=True, help='Verbose logging')
@click.option('--prefer-gpu/--no-prefer-gpu', default=True, help='Prefer GPU when available (cuda/mps).')
def main(signals: str, data: str, capital: float, output: str, verbose: bool, prefer_gpu: bool):
    """Execute LLM signal backtest and generate validation report."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    device = "cpu"
    try:
        from scripts.run_etl_pipeline import _detect_device  # reuse detector

        device = _detect_device(prefer_gpu=prefer_gpu)
    except Exception:
        device = "cpu"
    os.environ["PIPELINE_DEVICE"] = device
    logger.info("LLM backtest device: %s (prefer_gpu=%s)", device, prefer_gpu)

    logger.info("=" * 80)
    logger.info("LLM SIGNAL BACKTESTING FRAMEWORK")
    logger.info("=" * 80)
    logger.info(f"Signals: {signals}")
    logger.info(f"Data: {data}")
    logger.info(f"Initial Capital: ${capital:,.0f}")
    logger.info("")

    # Initialize backtester
    backtester = LLMSignalBacktester(initial_capital=capital)

    try:
        # Load signals and data
        signal_data = backtester.load_signals(signals)
        price_data = backtester.load_price_data(data)

        # Backtest each ticker
        results = {}
        for ticker, ticker_signals in signal_data.items():
            result = backtester.backtest_ticker(ticker, ticker_signals, price_data)
            if result:
                results[ticker] = result

        # Generate and print report
        report = backtester.generate_report(results)
        print(report)

        # Save results
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': capital,
            'results': {
                ticker: {
                    k: v for k, v in result.items()
                    if k not in ['portfolio_history', 'trade_log']  # Exclude DataFrames
                }
                for ticker, result in results.items()
                if result
            }
        }

        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"\n✓ Results saved to: {output}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
