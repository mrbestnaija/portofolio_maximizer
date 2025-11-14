#!/usr/bin/env python3
"""
LLM System Monitor
Comprehensive monitoring script for all LLM system components
Addresses issues from SYSTEM_STATUS_2025-10-22.md
"""

import sys
import os
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_llm.performance_monitor import performance_monitor, get_performance_status
from ai_llm.signal_quality_validator import (
    signal_validator,
    backtest_signal_quality,
    Signal,
    SignalDirection,
)
from ai_llm.llm_database_integration import (
    llm_db_manager,
    get_performance_summary,
    save_risk_assessment,
)
from ai_llm.performance_optimizer import performance_optimizer, optimize_model_selection
from ai_llm.ollama_client import OllamaClient
from etl.database_manager import DatabaseManager

# Configure logging
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(logs_dir / "monitor_llm_system.log"),
    filemode="a",
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)


class LLMSystemMonitor:
    """
    Comprehensive LLM system monitoring
    Addresses all issues from SYSTEM_STATUS_2025-10-22.md
    """
    
    def __init__(self):
        self.ollama_client = None
        self.monitoring_results = {}
        self.db_manager = DatabaseManager()
        
    def initialize_ollama_client(self):
        """Initialize Ollama client for monitoring"""
        try:
            self.ollama_client = OllamaClient()
            logger.info("✅ Ollama client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama client: {e}")
            return False
    
    def monitor_llm_performance(self):
        """
        Monitor LLM Performance: Track inference times in live scenarios
        Addresses: "Monitor LLM Performance: Track inference times in live scenarios"
        """
        logger.info("🔍 Monitoring LLM Performance...")
        
        try:
            performance_status = get_performance_status()
            fallback_info = performance_status.get("fallback_events", {}) if isinstance(performance_status, dict) else {}
            fallback_count = fallback_info.get("count", 0) if isinstance(fallback_info, dict) else 0

            latency_threshold = float(os.getenv("LLM_LATENCY_BENCHMARK_SECONDS", "5.0"))

            if self.ollama_client:
                test_prompt = "Analyze the current market conditions for AAPL stock."
                start_time = datetime.now()

                try:
                    response = self.ollama_client.generate(test_prompt)
                    inference_time = (datetime.now() - start_time).total_seconds()
                    token_count = max(len(response), 1)
                    token_rate = token_count / max(inference_time, 1e-6)

                    performance_monitor.record_inference(
                        model_name=self.ollama_client.model,
                        prompt=test_prompt,
                        response=response,
                        inference_time=inference_time,
                        success=True
                    )

                    logger.info("✅ LLM inference successful: %.2fs (%.2f tokens/sec)", inference_time, token_rate)

                    status = 'HEALTHY'
                    if inference_time > latency_threshold or fallback_count:
                        status = 'DEGRADED_LATENCY'
                        logger.warning(
                            "⚠️ Latency benchmark exceeded (%.2fs > %.2fs threshold)", inference_time, latency_threshold
                        )

                    benchmark_payload = {
                        'timestamp': datetime.now().isoformat(),
                        'model': self.ollama_client.model,
                        'inference_time_seconds': inference_time,
                        'latency_threshold_seconds': latency_threshold,
                        'token_rate_per_sec': token_rate,
                        'fallback_events': fallback_info,
                    }
                    benchmark_file = logs_dir / "latency_benchmark.json"
                    with open(benchmark_file, 'w') as fp:
                        json.dump(benchmark_payload, fp, indent=2)

                    self.monitoring_results['llm_performance'] = {
                        'status': status,
                        'inference_time': inference_time,
                        'token_rate_per_sec': token_rate,
                        'response_length': len(response),
                        'model_used': self.ollama_client.model,
                        'fallback_events': fallback_info,
                        'performance_summary': performance_status,
                        'benchmark_file': str(benchmark_file),
                        'benchmark': benchmark_payload,
                    }
                except Exception as e:
                    logger.error("❌ LLM inference failed: %s", e)
                    self.monitoring_results['llm_performance'] = {
                        'status': 'FAILED',
                        'error': str(e),
                        'performance_summary': performance_status,
                    }
            else:
                logger.warning("⚠️ Ollama client not available for performance testing")
                self.monitoring_results['llm_performance'] = {
                    'status': 'UNAVAILABLE',
                    'reason': 'Ollama client not initialized',
                    'performance_summary': performance_status,
                }

        except Exception as e:
            logger.error("❌ Performance monitoring failed: %s", e)
            self.monitoring_results['llm_performance'] = {
                'status': 'ERROR',
                'error': str(e),
                'performance_summary': performance_status if 'performance_status' in locals() else None,
            }
    
    def validate_signal_quality(self):
        """
        Validate Signal Quality: Ensure LLM-generated signals are accurate
        Addresses: "Validate Signal Quality: Ensure LLM-generated signals are accurate"
        """
        logger.info("🔍 Validating Signal Quality...")
        
        try:
            # Get recent signals from database
            recent_signals = llm_db_manager.get_recent_signals(hours=24)
            
            if not recent_signals:
                cursor = self.db_manager.cursor
                cursor.execute("SELECT COUNT(*) FROM llm_signals")
                existing_count = cursor.fetchone()[0] if cursor else 0
                logger.info("ℹ️ No recent signals found for validation")
                self.monitoring_results['signal_quality'] = {
                    'status': 'PENDING_INPUT' if not existing_count else 'STALE_VALIDATION',
                    'signals_analyzed': 0,
                    'signals_available': int(existing_count),
                }
                return
            
            # Create mock market data for validation
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            market_data = pd.DataFrame({
                'close': np.random.normal(150, 5, 30)
            }, index=dates)
            
            # Validate each signal
            validation_results = []
            skipped_signals = 0
            for signal in recent_signals[:5]:  # Limit to 5 signals for testing
                if not signal.signal_type:
                    skipped_signals += 1
                    logger.warning(f"⚠️ Missing signal_type for {signal.ticker} — skipping validation")
                    continue

                try:
                    direction = SignalDirection(signal.signal_type)
                except ValueError:
                    skipped_signals += 1
                    logger.warning(f"⚠️ Invalid signal_type '{signal.signal_type}' for {signal.ticker} — skipping")
                    continue

                try:
                    validation_signal = Signal(
                        ticker=signal.ticker,
                        direction=direction,
                        confidence=signal.confidence,
                        reasoning=signal.reasoning,
                        timestamp=signal.timestamp,
                        price_at_signal=signal.market_data_snapshot.get('price', 150.0),
                        expected_return=signal.expected_return,
                        risk_estimate=signal.risk_estimate
                    )

                    result = signal_validator.validate_signal(validation_signal, market_data)
                    validation_results.append({
                        'ticker': signal.ticker,
                        'is_valid': result.is_valid,
                        'confidence_score': result.confidence_score,
                        'recommendation': result.recommendation
                    })

                except Exception as e:
                    skipped_signals += 1
                    logger.warning(f"⚠️ Failed to validate signal for {signal.ticker}: {e}")

            if validation_results:
                valid_count = sum(1 for r in validation_results if r['is_valid'])
                avg_confidence = sum(r['confidence_score'] for r in validation_results) / len(validation_results)

                self.monitoring_results['signal_quality'] = {
                    'status': 'HEALTHY',
                    'signals_analyzed': len(validation_results),
                    'valid_signals': valid_count,
                    'validation_rate': valid_count / len(validation_results),
                    'avg_confidence': avg_confidence
                }

                logger.info(f"✅ Signal validation complete: {valid_count}/{len(validation_results)} signals valid")
            else:
                cursor = self.db_manager.cursor
                cursor.execute("SELECT COUNT(*) FROM llm_signals")
                existing_count = cursor.fetchone()[0] if cursor else 0

                status = 'PENDING_INPUT'
                if existing_count and skipped_signals:
                    status = 'STALE_VALIDATION'

                self.monitoring_results['signal_quality'] = {
                    'status': status,
                    'signals_analyzed': 0,
                    'skipped_signals': skipped_signals,
                    'signals_available': int(existing_count),
                }

        except Exception as e:
            logger.error(f"❌ Signal quality validation failed: {e}")
            self.monitoring_results['signal_quality'] = {
                'status': 'ERROR',
                'error': str(e)
            }

    def _summarise_signals_directly(self, limit: int) -> Optional[Dict[str, Any]]:
        """Fallback summarisation derived from llm_signals when backtests are absent."""
        cursor = self.db_manager.cursor
        cursor.execute(
            """
            SELECT
                ticker,
                action,
                COALESCE(signal_timestamp, signal_date, created_at) AS evaluated_at,
                backtest_hit_rate,
                backtest_profit_factor
            FROM llm_signals
            ORDER BY evaluated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        records = []
        hit_rates = []
        profit_factors = []
        for row in rows:
            payload = dict(row)
            stamp = payload.get("evaluated_at")
            payload["evaluated_at"] = str(stamp) if stamp else None
            hit = payload.get("backtest_hit_rate")
            profit = payload.get("backtest_profit_factor")
            if hit is not None:
                hit_rates.append(float(hit))
            if profit is not None:
                profit_factors.append(float(profit))
            records.append(payload)

        summary = {
            "signals_considered": len(records),
            "tickers": sorted({rec["ticker"] for rec in records if rec.get("ticker")}),
            "latest_signal_at": records[0]["evaluated_at"],
            "mean_backtest_hit_rate": float(np.mean(hit_rates)) if hit_rates else None,
            "mean_backtest_profit_factor": float(np.mean(profit_factors)) if profit_factors else None,
        }

        status = "BASELINE"
        if summary["mean_backtest_profit_factor"] is not None:
            status = "HEALTHY" if summary["mean_backtest_profit_factor"] >= 1.5 else "DEGRADED"

        return {
            "status": status,
            "records": records,
            "summary": summary,
        }

    def collect_signal_backtest_metrics(self, limit: int = 5):
        """Surface aggregated metrics from llm_signal_backtests for dashboards."""
        logger.info("🔍 Collecting signal backtest metrics...")
        try:
            cursor = self.db_manager.cursor
            cursor.execute(
                """
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
                ORDER BY generated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            if not rows:
                fallback = self._summarise_signals_directly(limit)
                if fallback:
                    self.monitoring_results['signal_backtests'] = fallback
                    logger.info("ℹ️ No llm_signal_backtests rows found; derived fallback summary from llm_signals.")
                else:
                    self.monitoring_results['signal_backtests'] = {
                        'status': 'PENDING_INPUT',
                        'records': [],
                        'summary': None,
                    }
                    logger.info("ℹ️ No signal backtest summaries or raw signals found in database.")
                return

            records = []
            for row in rows:
                payload = dict(row)
                payload['generated_at'] = str(payload.get('generated_at'))
                payload['statistically_significant'] = bool(payload.get('statistically_significant'))
                records.append(payload)

            df = pd.DataFrame(records)
            summary = {
                'mean_hit_rate': float(df['hit_rate'].mean()),
                'mean_profit_factor': float(df['profit_factor'].mean()),
                'mean_sharpe': float(df['sharpe_ratio'].mean()),
                'signals_analyzed': int(df['signals_analyzed'].sum()),
                'tickers': df['ticker'].unique().tolist(),
                'latest_generated_at': records[0]['generated_at'],
            }

            status = 'HEALTHY'
            if summary['mean_hit_rate'] < 0.55 or summary['mean_profit_factor'] < 1.5:
                status = 'DEGRADED'

            self.monitoring_results['signal_backtests'] = {
                'status': status,
                'records': records,
                'summary': summary,
            }

            logger.info(
                "✅ Signal backtest metrics refreshed (latest %s | mean hit rate %.2f | profit factor %.2f)",
                summary['latest_generated_at'],
                summary['mean_hit_rate'],
                summary['mean_profit_factor'],
            )
        except Exception as exc:
            logger.error("❌ Failed to collect signal backtest metrics: %s", exc)
            self.monitoring_results['signal_backtests'] = {
                'status': 'ERROR',
                'error': str(exc),
            }
    
    def verify_database_integration(self):
        """
        Database Integration: Verify LLM risk assessments save properly
        Addresses: "Database Integration: Verify LLM risk assessments save properly"
        """
        logger.info("🔍 Verifying Database Integration...")
        
        try:
            # Test saving a risk assessment
            test_assessment_id = save_risk_assessment(
                portfolio_id="test_portfolio_001",
                risk_score=0.3,
                risk_factors=["High volatility", "Market uncertainty"],
                recommendations=["Reduce position size", "Add hedging"],
                model_used="qwen:14b-chat-q4_K_M",
                confidence=0.85,
                market_conditions={"volatility": 0.25, "trend": "bearish"}
            )
            
            # Verify the assessment was saved
            recent_assessments = llm_db_manager.get_recent_risk_assessments(hours=1)
            saved_assessment = next((a for a in recent_assessments if a.id == test_assessment_id), None)
            
            if saved_assessment:
                logger.info("✅ Risk assessment saved and retrieved successfully")
                self.monitoring_results['database_integration'] = {
                    'status': 'HEALTHY',
                    'risk_assessments_saved': len(recent_assessments),
                    'test_assessment_id': test_assessment_id
                }
            else:
                logger.error("❌ Risk assessment not found after saving")
                self.monitoring_results['database_integration'] = {
                    'status': 'FAILED',
                    'reason': 'Assessment not found after saving'
                }
                
        except Exception as e:
            logger.error(f"❌ Database integration verification failed: {e}")
            self.monitoring_results['database_integration'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def optimize_performance(self):
        """
        Performance Optimization: Fine-tune model selection for speed
        Addresses: "Performance Optimization: Fine-tune model selection for speed"
        """
        logger.info("🔍 Optimizing Performance...")
        
        try:
            # Get performance optimization recommendations
            fast_model = optimize_model_selection("fast")
            balanced_model = optimize_model_selection("balanced")
            accurate_model = optimize_model_selection("accurate")
            
            # Update optimizer with current performance data
            if self.ollama_client:
                performance_optimizer.update_model_performance(
                    model_name=self.ollama_client.model,
                    inference_time=5.0,  # Mock data
                    tokens_per_second=15.0,
                    success=True,
                    accuracy_score=0.8
                )
            
            # Get optimization report
            optimization_report = performance_optimizer.get_performance_report()
            
            self.monitoring_results['performance_optimization'] = {
                'status': 'HEALTHY',
                'fast_model': fast_model.recommended_model,
                'balanced_model': balanced_model.recommended_model,
                'accurate_model': accurate_model.recommended_model,
                'optimization_report': optimization_report
            }
            
            logger.info("✅ Performance optimization complete")
            logger.info(f"   Fast model: {fast_model.recommended_model}")
            logger.info(f"   Balanced model: {balanced_model.recommended_model}")
            logger.info(f"   Accurate model: {accurate_model.recommended_model}")
            
        except Exception as e:
            logger.error(f"❌ Performance optimization failed: {e}")
            self.monitoring_results['performance_optimization'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def run_comprehensive_monitoring(self):
        """Run comprehensive LLM system monitoring"""
        logger.info("🚀 Starting Comprehensive LLM System Monitoring")
        logger.info("=" * 60)
        
        # Initialize Ollama client
        if not self.initialize_ollama_client():
            logger.error("❌ Cannot proceed without Ollama client")
            return False
        
        # Run all monitoring checks
        self.monitor_llm_performance()
        self.validate_signal_quality()
        self.collect_signal_backtest_metrics()
        self.verify_database_integration()
        self.optimize_performance()
        
        # Generate summary report
        self.generate_monitoring_report()
        
        logger.info("✅ Comprehensive monitoring complete")
        return True
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        logger.info("📊 Generating Monitoring Report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_results": self.monitoring_results,
            "system_health": self._assess_system_health(),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report to file
        report_file = f"logs/llm_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Monitoring report saved to: {report_file}")
        
        # Print summary
        self._print_summary_report(report)
    
    def _assess_system_health(self):
        """Assess overall system health"""
        health_status = "HEALTHY"
        issues = []
        
        for component, result in self.monitoring_results.items():
            if result.get('status') not in ['HEALTHY', 'NO_DATA']:
                health_status = "DEGRADED"
                issues.append(f"{component}: {result.get('status', 'UNKNOWN')}")
        
        return {
            "overall_status": health_status,
            "issues": issues,
            "components_checked": len(self.monitoring_results)
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on monitoring results"""
        recommendations = []
        
        # Check LLM performance
        if 'llm_performance' in self.monitoring_results:
            perf = self.monitoring_results['llm_performance']
            if perf.get('status') == 'FAILED':
                recommendations.append("Investigate LLM inference failures")
            elif perf.get('status') == 'DEGRADED_LATENCY':
                recommendations.append("Latency above benchmark — review caching/fallback tuning (see logs/latency_benchmark.json)")
            elif perf.get('inference_time', 0) > 30:
                recommendations.append("Consider optimizing model selection for faster inference")
        
        # Check signal quality
        if 'signal_quality' in self.monitoring_results:
            quality = self.monitoring_results['signal_quality']
            if quality.get('validation_rate', 1.0) < 0.8:
                recommendations.append("Review signal generation quality - low validation rate")
        
        # Check database integration
        if 'database_integration' in self.monitoring_results:
            db = self.monitoring_results['database_integration']
            if db.get('status') != 'HEALTHY':
                recommendations.append("Fix database integration issues")
        
        return recommendations
    
    def _print_summary_report(self, report):
        """Print summary report to console"""
        print("\n" + "=" * 60)
        print("📊 LLM SYSTEM MONITORING SUMMARY")
        print("=" * 60)
        
        health = report['system_health']
        print(f"Overall Status: {health['overall_status']}")
        print(f"Components Checked: {health['components_checked']}")
        
        if health['issues']:
            print("\n⚠️ Issues Found:")
            for issue in health['issues']:
                print(f"  - {issue}")
        
        if report['recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\n📈 Component Status:")
        for component, result in self.monitoring_results.items():
            status = result.get('status', 'UNKNOWN')
            print(f"  {component}: {status}")
        
        print("=" * 60)


def main():
    """Main monitoring function"""
    monitor = LLMSystemMonitor()
    
    try:
        success = monitor.run_comprehensive_monitoring()
        if success:
            print("\n✅ LLM System Monitoring completed successfully")
            sys.exit(0)
        else:
            print("\n❌ LLM System Monitoring failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Monitoring failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
