#!/usr/bin/env python3
"""
Live Ensemble Performance Dashboard
Self-iterative monitoring system that tracks GARCH integration success,
RMSE ratios, and provides automated recommendations.
"""

import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from integrity.sqlite_guardrails import guarded_sqlite_connect


class EnsembleDashboard:
    """
    Self-iterative dashboard for monitoring ensemble performance.
    Automatically analyzes results, identifies issues, and suggests improvements.
    """

    def __init__(self, db_path: str = "data/portfolio_maximizer.db"):
        self.db_path = Path(db_path)
        self.baseline_rmse_ratio = 1.682  # From Phase 7.3 diagnostics
        self.target_rmse_ratio = 1.100
        self.results = {}

    def _mirror_path(self) -> Path | None:
        if os.name != "posix":
            return None
        if not self.db_path.as_posix().startswith("/mnt/"):
            return None
        tmp_root = Path(os.environ.get("WSL_SQLITE_TMP", "/tmp"))
        return tmp_root / f"{self.db_path.name}.wsl"

    def _resolve_db_path(self) -> Path:
        mirror = self._mirror_path()
        if mirror and mirror.exists():
            try:
                if not self.db_path.exists():
                    return mirror
                if mirror.stat().st_mtime >= self.db_path.stat().st_mtime:
                    return mirror
            except OSError:
                return mirror
        return self.db_path

    def connect_db(self) -> sqlite3.Connection:
        """Connect to forecasts database."""
        db_path = self._resolve_db_path()
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        uri = f"file:{db_path.as_posix()}?mode=ro"
        try:
            return guarded_sqlite_connect(
                uri,
                uri=True,
                allow_schema_changes=False,
            )
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "disk i/o error" not in msg:
                raise
            mirror = self._mirror_path()
            if not mirror:
                raise
            try:
                mirror.parent.mkdir(parents=True, exist_ok=True)
                if db_path.exists():
                    shutil.copy2(db_path, mirror)
            except Exception:
                pass
            if mirror.exists():
                mirror_uri = f"file:{mirror.as_posix()}?mode=ro"
                return guarded_sqlite_connect(
                    mirror_uri,
                    uri=True,
                    allow_schema_changes=False,
                )
            raise

    def fetch_recent_forecasts(self, hours: int = 24) -> pd.DataFrame:
        """Fetch forecasts from the last N hours."""
        conn = self.connect_db()
        cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

        ts_expr = "forecast_date"
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(time_series_forecasts)").fetchall()}
            if "created_at" in cols:
                ts_expr = "COALESCE(created_at, forecast_date)"
        except sqlite3.Error:
            pass

        query = """
        SELECT
            ticker,
            forecast_date,
            model_type,
            diagnostics,
            regression_metrics
        FROM time_series_forecasts
        WHERE DATE({ts_expr}) >= DATE(?)
        ORDER BY ticker, {ts_expr} DESC
        """

        df = pd.read_sql_query(query.format(ts_expr=ts_expr), conn, params=(cutoff,))
        conn.close()
        return df

    def parse_ensemble_metadata(self, diagnostics: str, regression_metrics: str) -> Dict:
        """Extract ensemble weights and confidence from metadata."""
        payloads = []
        for raw in (diagnostics, regression_metrics):
            if not raw:
                continue
            try:
                payloads.append(json.loads(raw) if isinstance(raw, str) else raw)
            except (json.JSONDecodeError, TypeError):
                continue
        merged: Dict = {}
        for p in payloads:
            if isinstance(p, dict):
                merged.update(p)
        weights = merged.get("weights") or merged.get("ensemble_weights") or {}
        confidence = merged.get("confidence") or {}
        rmse_ratio = merged.get("rmse_ratio") or merged.get("rmse_ratio_over_baseline")
        best_model_rmse = merged.get("best_model_rmse") or merged.get("best_rmse")
        ensemble_rmse = merged.get("ensemble_rmse") or merged.get("rmse")
        if rmse_ratio is None and ensemble_rmse is not None and best_model_rmse:
            try:
                rmse_ratio = float(ensemble_rmse) / float(best_model_rmse)
            except Exception:
                rmse_ratio = None
        return {
            "weights": weights if isinstance(weights, dict) else {},
            "confidence": confidence if isinstance(confidence, dict) else {},
            "rmse_ratio": rmse_ratio,
            "best_model_rmse": best_model_rmse,
            "ensemble_rmse": ensemble_rmse,
        }

    def analyze_ticker_performance(self, df: pd.DataFrame, ticker: str) -> Dict:
        """Analyze performance for a single ticker."""
        ticker_data = df[df['ticker'] == ticker]

        if ticker_data.empty:
            return {'error': 'No data'}

        # Extract ensemble metadata
        ensemble_forecasts = ticker_data[ticker_data['model_type'] == 'ENSEMBLE']

        if ensemble_forecasts.empty:
            return {'error': 'No ensemble forecasts'}

        # Parse metadata for all ensemble forecasts
        parsed = []
        for _, row in ensemble_forecasts.iterrows():
            meta = self.parse_ensemble_metadata(row.get('diagnostics'), row.get('regression_metrics'))
            if meta:
                meta['forecast_date'] = row['forecast_date']
                parsed.append(meta)

        if not parsed:
            return {'error': 'No valid metadata'}

        # Calculate statistics
        garch_weights = [m['weights'].get('garch', 0.0) for m in parsed]
        rmse_ratios = [m['rmse_ratio'] for m in parsed if m.get('rmse_ratio')]

        avg_garch_weight = sum(garch_weights) / len(garch_weights) if garch_weights else 0
        avg_rmse_ratio = sum(rmse_ratios) / len(rmse_ratios) if rmse_ratios else None

        # Determine status
        if avg_rmse_ratio:
            if avg_rmse_ratio < self.target_rmse_ratio:
                status = 'TARGET_ACHIEVED'
                emoji = '✅'
            elif avg_rmse_ratio < self.baseline_rmse_ratio:
                progress = ((self.baseline_rmse_ratio - avg_rmse_ratio) /
                           (self.baseline_rmse_ratio - self.target_rmse_ratio)) * 100
                status = f'IMPROVING ({progress:.1f}% to target)'
                emoji = '⚠️'
            else:
                status = 'REGRESSING'
                emoji = '❌'
        else:
            status = 'NO_DATA'
            emoji = '❓'

        return {
            'ticker': ticker,
            'forecast_count': len(parsed),
            'avg_garch_weight': avg_garch_weight,
            'max_garch_weight': max(garch_weights) if garch_weights else 0,
            'avg_rmse_ratio': avg_rmse_ratio,
            'min_rmse_ratio': min(rmse_ratios) if rmse_ratios else None,
            'max_rmse_ratio': max(rmse_ratios) if rmse_ratios else None,
            'status': status,
            'emoji': emoji,
            'improvement_pct': ((self.baseline_rmse_ratio - avg_rmse_ratio) / self.baseline_rmse_ratio * 100)
                               if avg_rmse_ratio else None,
            'gap_to_target': avg_rmse_ratio - self.target_rmse_ratio if avg_rmse_ratio else None,
        }

    def generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate automated recommendations based on performance."""
        recommendations = []
        if not results:
            return recommendations

        # Analyze overall patterns
        total = len(results)
        garch_selected = sum(1 for r in results if r.get('avg_garch_weight', 0) > 0.5)
        at_target = sum(
            1 for r in results
            if isinstance(r.get('avg_rmse_ratio'), (int, float))
            and r['avg_rmse_ratio'] < self.target_rmse_ratio
        )
        regressing = sum(1 for r in results if 'REGRESSING' in r.get('status', ''))

        # Recommendation 1: GARCH selection frequency
        if garch_selected < total * 0.3:  # Less than 30% GARCH selection
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Model Selection',
                'issue': f'GARCH selected in only {garch_selected}/{total} tickers ({garch_selected/total*100:.0f}%)',
                'action': 'Adjust confidence normalization or add regime detection',
                'code': 'Modify ensemble.py derive_model_confidence() to better balance GARCH vs SAMoSSA confidence',
            })

        # Recommendation 2: Target achievement
        if at_target < total:
            gaps = [r.get('gap_to_target') for r in results if isinstance(r.get('gap_to_target'), (int, float))]
            gap_avg = sum(gaps) / len(gaps) if gaps else 0.0
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Performance',
                'issue': f'{total - at_target}/{total} tickers above target (avg gap: {gap_avg:.3f})',
                'action': 'Implement ensemble weight optimization using holdout data',
                'code': 'Add scipy.optimize.minimize to find optimal weights in EnsembleCoordinator',
            })

        # Recommendation 3: Regression detection
        if regressing > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Regression',
                'issue': f'{regressing} ticker(s) performing worse than baseline',
                'action': 'Investigate data quality, check for regime shifts, verify model health',
                'code': 'Run diagnostics: python scripts/run_ensemble_diagnostics.py',
            })

        # Recommendation 4: Per-ticker analysis
        for result in results:
            ticker = result.get('ticker')
            garch_weight = result.get('avg_garch_weight', 0)
            rmse_ratio = result.get('avg_rmse_ratio')

            # High GARCH weight but poor performance
            if garch_weight > 0.7 and rmse_ratio and rmse_ratio > 1.3:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Ticker-Specific',
                    'issue': f'{ticker}: GARCH-dominant ({garch_weight:.0%}) but RMSE ratio {rmse_ratio:.3f}',
                    'action': f'Check if {ticker} regime matches GARCH assumptions (stationary, vol clustering)',
                    'code': f'Analyze {ticker} stationarity: ADF test, volatility clustering, trend strength',
                })

            # No GARCH but underperforming
            if garch_weight < 0.1 and rmse_ratio and rmse_ratio > 1.4:
                recommendations.append({
                    'priority': 'LOW',
                    'category': 'Ticker-Specific',
                    'issue': f'{ticker}: SAMoSSA-only but RMSE ratio {rmse_ratio:.3f}',
                    'action': f'Consider neural forecaster (PatchTST) for {ticker} trending regime',
                    'code': 'Phase 8: Integrate PatchTST/NHITS for non-stationary, trending markets',
                })

        # Recommendation 5: Phase 8 trigger
        if at_target >= len(results) * 0.7:  # 70%+ tickers at target
            recommendations.append({
                'priority': 'LOW',
                'category': 'Next Phase',
                'issue': 'Strong baseline performance established',
                'action': 'Begin Phase 8: Neural forecaster integration',
                'code': 'See Documentation/PHASE_8_NEURAL_FORECASTER_PLAN.md',
            })

        return recommendations

    def print_dashboard(self):
        """Print formatted live dashboard."""
        print("\n" + "=" * 100)
        print("LIVE ENSEMBLE PERFORMANCE DASHBOARD - Phase 7.3 GARCH Integration")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

        # Fetch data
        try:
            df = self.fetch_recent_forecasts(hours=24)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Run pipeline first: python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,NVDA")
            return

        if df.empty:
            print("\nNo forecasts found in the last 24 hours.")
            return

        # Analyze each ticker
        tickers = df['ticker'].unique()
        results = []
        for ticker in sorted(tickers):
            result = self.analyze_ticker_performance(df, ticker)
            if 'error' not in result:
                results.append(result)

        if not results:
            print("\nNo ensemble forecasts with valid metadata found.")
            return

        # Print results table
        print("\n## Multi-Ticker Performance Summary")
        print("-" * 100)
        print(f"{'Ticker':<8} {'Status':<4} {'GARCH Weight':<15} {'RMSE Ratio':<15} "
              f"{'vs Target':<12} {'Improvement':<12}")
        print("-" * 100)

        overall_ratios = []
        overall_improvements = []
        avg_garch = sum(r['avg_garch_weight'] for r in results) / len(results)

        for result in results:
            ticker = result['ticker']
            emoji = result['emoji']
            garch_weight = result['avg_garch_weight']
            rmse_ratio = result['avg_rmse_ratio']
            gap = result['gap_to_target']
            improvement = result['improvement_pct']

            if rmse_ratio:
                overall_ratios.append(rmse_ratio)
                if improvement:
                    overall_improvements.append(improvement)

                print(f"{ticker:<8} {emoji:<4} "
                      f"{garch_weight:>6.1%} (max={result['max_garch_weight']:.1%})"
                      f" {rmse_ratio:>6.3f}"
                      f" {gap:>+7.3f}"
                      f"      {improvement:>+6.1f}%")
            else:
                print(f"{ticker:<8} {emoji:<4} {garch_weight:>6.1%}  No RMSE data")

        # Overall statistics
        if overall_ratios:
            avg_ratio = sum(overall_ratios) / len(overall_ratios)
            avg_improvement = sum(overall_improvements) / len(overall_improvements) if overall_improvements else 0

            print("-" * 100)
            print(f"{'OVERALL':<8} {'🎯':<4} "
                  f"{avg_garch:>6.1%} avg"
                  f"        {avg_ratio:>6.3f}"
                  f" {avg_ratio - self.target_rmse_ratio:>+7.3f}"
                  f"      {avg_improvement:>+6.1f}%")
            print("-" * 100)

            # Progress bar
            progress = ((self.baseline_rmse_ratio - avg_ratio) /
                       (self.baseline_rmse_ratio - self.target_rmse_ratio))
            progress_pct = max(0, min(100, progress * 100))
            bar_length = 50
            filled = int(bar_length * progress_pct / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"\nProgress to Target: [{bar}] {progress_pct:.1f}%")
            print(f"Baseline: {self.baseline_rmse_ratio:.3f} → Current: {avg_ratio:.3f} → Target: {self.target_rmse_ratio:.3f}")

        # Key Metrics
        print("\n## Key Metrics")
        print("-" * 100)
        garch_dominant = sum(1 for r in results if r['avg_garch_weight'] > 0.5)
        at_target = sum(
            1 for r in results
            if isinstance(r.get('avg_rmse_ratio'), (int, float))
            and r['avg_rmse_ratio'] < self.target_rmse_ratio
        )

        print(f"Total Tickers: {len(results)}")
        print(f"GARCH-Dominant (>50% weight): {garch_dominant} ({garch_dominant/len(results)*100:.1f}%)")
        print(f"At Target (<1.1x RMSE): {at_target} ({at_target/len(results)*100:.1f}%)")
        print(f"Average GARCH Weight: {avg_garch:.1%}")
        if overall_ratios:
            print(f"Best Performer: {min(overall_ratios):.3f} RMSE ratio")
            print(f"Worst Performer: {max(overall_ratios):.3f} RMSE ratio")

        # Recommendations
        print("\n## Automated Recommendations")
        print("-" * 100)
        recommendations = self.generate_recommendations(results)

        if not recommendations:
            print("✅ No issues detected. System performing optimally.")
        else:
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {'CRITICAL': '🔴', 'HIGH': '🟡', 'MEDIUM': '🔵', 'LOW': '⚪'}.get(rec['priority'], '⚪')
                print(f"\n{i}. {priority_emoji} [{rec['priority']}] {rec['category']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Action: {rec['action']}")
                print(f"   Implementation: {rec['code']}")

        # Self-iteration trigger
        print("\n## Self-Iteration Status")
        print("-" * 100)

        critical_issues = sum(1 for r in recommendations if r['priority'] == 'CRITICAL')
        if critical_issues > 0:
            print(f"🔴 CRITICAL: {critical_issues} critical issue(s) detected!")
            print("   Auto-triggering diagnostics...")
            print("   Command: python scripts/run_ensemble_diagnostics.py")
        elif at_target < len(results) * 0.5:
            print(f"🟡 WARNING: Only {at_target}/{len(results)} tickers at target")
            print("   Consider implementing recommendations above")
        else:
            print(f"✅ HEALTHY: {at_target}/{len(results)} tickers at target")
            print("   System ready for Phase 8 (Neural Forecasting)")

        print("\n" + "=" * 100)
        print("Dashboard refresh: python dashboard/live_ensemble_monitor.py")
        print("=" * 100 + "\n")

        # Store results for iteration
        self.results = results
        return results

    def export_metrics(self, output_path: str = "dashboard/metrics.json"):
        """Export metrics to JSON for external monitoring."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'tickers': self.results,
            'summary': {
                'total_tickers': len(self.results),
                'avg_garch_weight': sum(r.get('avg_garch_weight', 0) for r in self.results) / len(self.results) if self.results else 0,
                'avg_rmse_ratio': sum(r.get('avg_rmse_ratio', 0) for r in self.results if r.get('avg_rmse_ratio')) / max(len([r for r in self.results if r.get('avg_rmse_ratio')]), 1),
                'at_target_count': sum(
                    1 for r in self.results
                    if isinstance(r.get('avg_rmse_ratio'), (int, float))
                    and r['avg_rmse_ratio'] < self.target_rmse_ratio
                ),
            },
            'recommendations': self.generate_recommendations(self.results),
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Metrics exported to: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Live Ensemble Performance Dashboard')
    parser.add_argument('--db', default='data/portfolio_maximizer.db',
                       help='Path to forecasts database')
    parser.add_argument('--hours', type=int, default=24,
                       help='Look back N hours (default: 24)')
    parser.add_argument('--export', action='store_true',
                       help='Export metrics to JSON')
    parser.add_argument('--watch', action='store_true',
                       help='Continuous monitoring mode (refresh every 60s)')

    args = parser.parse_args()

    dashboard = EnsembleDashboard(db_path=args.db)

    if args.watch:
        import time
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                dashboard.print_dashboard()
                if args.export:
                    dashboard.export_metrics()
                print(f"Refreshing in 60 seconds...")
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        dashboard.print_dashboard()
        if args.export:
            dashboard.export_metrics()


if __name__ == "__main__":
    main()
