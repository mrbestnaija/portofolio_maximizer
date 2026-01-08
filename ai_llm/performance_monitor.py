"""
LLM Performance Monitor
Tracks inference times and performance metrics for production monitoring
"""

import logging
import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics for a single LLM inference"""
    model_name: str
    prompt_length: int
    response_length: int
    inference_time: float
    tokens_per_second: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    stage: Optional[str] = None
    ticker: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class LLMPerformanceMonitor:
    """
    Monitor LLM performance in real-time
    Tracks inference times, token rates, error rates, and fallback usage
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: List[InferenceMetrics] = []
        self.fallback_events: List[Dict[str, Any]] = []
        self.performance_thresholds = {
            'max_inference_time': 30.0,  # 30 seconds max
            'min_tokens_per_second': 5.0,  # 5 tokens/sec minimum
            'max_error_rate': 0.05  # 5% max error rate
        }
        
    def record_inference(self, 
                        model_name: str,
                        prompt: str,
                        response: str,
                        inference_time: float,
                        success: bool = True,
                        error_message: Optional[str] = None,
                        stage: Optional[str] = None,
                        ticker: Optional[str] = None,
                        fallback_used: bool = False,
                        fallback_reason: Optional[str] = None) -> InferenceMetrics:
        """Record a single LLM inference"""
        
        tokens_per_second = len(response.split()) / inference_time if inference_time > 0 else 0
        
        metrics = InferenceMetrics(
            model_name=model_name,
            prompt_length=len(prompt),
            response_length=len(response),
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message,
            stage=stage,
            ticker=ticker,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Log performance warning if needed
        self._check_performance_warnings(metrics)
        
        if fallback_used:
            self._record_fallback_event_internal(
                stage=stage,
                ticker=ticker,
                reason=fallback_reason or "unspecified",
                metrics=metrics,
            )
        
        return metrics
    
    def _record_fallback_event_internal(
        self,
        stage: Optional[str],
        ticker: Optional[str],
        reason: str,
        metrics: Optional[InferenceMetrics] = None,
    ) -> Dict[str, Any]:
        event = {
            "timestamp": datetime.now(),
            "stage": stage,
            "ticker": ticker,
            "reason": reason,
            "inference_time": getattr(metrics, "inference_time", None),
            "tokens_per_second": getattr(metrics, "tokens_per_second", None),
            "model_name": getattr(metrics, "model_name", None),
        }
        self.fallback_events.append(event)
        if len(self.fallback_events) > self.max_history:
            self.fallback_events = self.fallback_events[-self.max_history:]

        logger.warning(
            "LLM latency guard activated (stage=%s, ticker=%s, reason=%s)",
            stage or "unknown",
            ticker or "n/a",
            reason,
        )
        return event

    def record_latency_fallback(
        self,
        stage: str,
        ticker: Optional[str],
        reason: str,
        inference_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a latency-triggered fallback event for monitoring dashboards."""
        metrics = None
        if inference_stats:
            metrics = InferenceMetrics(
                model_name=inference_stats.get("model_name", "unknown"),
                prompt_length=inference_stats.get("prompt_length", 0),
                response_length=inference_stats.get("response_length", 0),
                inference_time=inference_stats.get("inference_time", 0.0),
                tokens_per_second=inference_stats.get("tokens_per_second", 0.0),
                timestamp=inference_stats.get("timestamp", datetime.now()),
                success=False,
                error_message=inference_stats.get("error"),
                stage=stage,
                ticker=ticker,
                fallback_used=True,
                fallback_reason=reason,
            )
        return self._record_fallback_event_internal(stage, ticker, reason, metrics)
    
    def _check_performance_warnings(self, metrics: InferenceMetrics):
        """Check for performance issues and log warnings"""
        
        if metrics.inference_time > self.performance_thresholds['max_inference_time']:
            logger.warning(f"Slow inference detected: {metrics.inference_time:.2f}s "
                          f"(threshold: {self.performance_thresholds['max_inference_time']}s)")
        
        if metrics.tokens_per_second < self.performance_thresholds['min_tokens_per_second']:
            logger.warning(f"Low token rate detected: {metrics.tokens_per_second:.2f} tokens/sec "
                          f"(threshold: {self.performance_thresholds['min_tokens_per_second']} tokens/sec)")
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {
                "time_period_hours": hours,
                "total_inferences": 0,
                "successful_inferences": 0,
                "failed_inferences": 0,
                "fallback_events": {"count": 0, "by_stage": {}},
                "success_rate": 0.0,
                "avg_inference_time": 0.0,
                "median_inference_time": 0.0,
                "max_inference_time": 0.0,
                "avg_tokens_per_second": 0.0,
                "median_tokens_per_second": 0.0,
                "model_breakdown": {},
                "performance_status": "NO_DATA",
            }
        
        successful_metrics = [m for m in recent_metrics if m.success]
        failed_metrics = [m for m in recent_metrics if not m.success]
        fallback_recent = [e for e in self.fallback_events if e["timestamp"] >= cutoff_time]
        
        # Calculate statistics
        inference_times = [m.inference_time for m in successful_metrics]
        token_rates = [m.tokens_per_second for m in successful_metrics]
        
        summary = {
            "time_period_hours": hours,
            "total_inferences": len(recent_metrics),
            "successful_inferences": len(successful_metrics),
            "failed_inferences": len(failed_metrics),
            "fallback_events": {
                "count": len(fallback_recent),
                "by_stage": dict(Counter((e["stage"] or "unknown") for e in fallback_recent)),
            },
            "success_rate": len(successful_metrics) / len(recent_metrics) if recent_metrics else 0,
            "avg_inference_time": statistics.mean(inference_times) if inference_times else 0,
            "median_inference_time": statistics.median(inference_times) if inference_times else 0,
            "max_inference_time": max(inference_times) if inference_times else 0,
            "avg_tokens_per_second": statistics.mean(token_rates) if token_rates else 0,
            "median_tokens_per_second": statistics.median(token_rates) if token_rates else 0,
            "model_breakdown": self._get_model_breakdown(recent_metrics),
            "performance_status": self._assess_performance_status(recent_metrics, fallback_recent)
        }
        
        return summary
    
    def _get_model_breakdown(self, metrics: List[InferenceMetrics]) -> Dict:
        """Get performance breakdown by model"""
        
        model_stats = {}
        for model_name in set(m.model_name for m in metrics):
            model_metrics = [m for m in metrics if m.model_name == model_name]
            successful = [m for m in model_metrics if m.success]
            
            model_stats[model_name] = {
                "total_inferences": len(model_metrics),
                "successful": len(successful),
                "success_rate": len(successful) / len(model_metrics) if model_metrics else 0,
                "avg_inference_time": statistics.mean([m.inference_time for m in successful]) if successful else 0,
                "avg_tokens_per_second": statistics.mean([m.tokens_per_second for m in successful]) if successful else 0
            }
        
        return model_stats
    
    def _assess_performance_status(
        self,
        metrics: List[InferenceMetrics],
        fallback_events: List[Dict[str, Any]],
    ) -> str:
        """Assess overall performance status"""
        
        if not metrics:
            return "NO_DATA"
        
        successful = [m for m in metrics if m.success]
        if not successful:
            return "CRITICAL"
        
        # Check error rate
        error_rate = (len(metrics) - len(successful)) / len(metrics)
        if error_rate > self.performance_thresholds['max_error_rate']:
            return "DEGRADED"
        
        # Check average inference time
        avg_time = statistics.mean([m.inference_time for m in successful])
        if avg_time > self.performance_thresholds['max_inference_time']:
            return "SLOW"
        
        # Check average token rate
        avg_tokens = statistics.mean([m.tokens_per_second for m in successful])
        if avg_tokens < self.performance_thresholds['min_tokens_per_second']:
            return "SLOW"

        # If too many fallbacks within window, mark as DEGRADED
        if len(fallback_events) / max(len(metrics), 1) > 0.25:
            return "DEGRADED"
        
        return "HEALTHY"
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.metrics_history),
            "performance_summary": self.get_performance_summary(24),
            "recent_metrics": [
                {
                    "model_name": m.model_name,
                    "inference_time": m.inference_time,
                    "tokens_per_second": m.tokens_per_second,
                    "success": m.success,
                    "timestamp": m.timestamp.isoformat(),
                    "stage": m.stage,
                    "ticker": m.ticker,
                    "fallback_used": m.fallback_used,
                    "fallback_reason": m.fallback_reason,
                }
                for m in self.metrics_history[-100:]  # Last 100 metrics
            ],
            "recent_fallbacks": [
                {
                    "timestamp": e["timestamp"].isoformat(),
                    "stage": e["stage"],
                    "ticker": e["ticker"],
                    "reason": e["reason"],
                    "inference_time": e["inference_time"],
                    "tokens_per_second": e["tokens_per_second"],
                    "model_name": e["model_name"],
                }
                for e in self.fallback_events[-100:]
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


# Global performance monitor instance
performance_monitor = LLMPerformanceMonitor()


def monitor_inference(model_name: str, prompt: str, response: str, 
                     inference_time: float, success: bool = True, 
                     error_message: Optional[str] = None,
                     stage: Optional[str] = None,
                     ticker: Optional[str] = None,
                     fallback_used: bool = False,
                     fallback_reason: Optional[str] = None) -> InferenceMetrics:
    """Convenience function to record inference metrics"""
    return performance_monitor.record_inference(
        model_name=model_name,
        prompt=prompt,
        response=response,
        inference_time=inference_time,
        success=success,
        error_message=error_message,
        stage=stage,
        ticker=ticker,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def record_latency_fallback(stage: str,
                            ticker: Optional[str],
                            reason: str,
                            inference_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record a latency fallback event via the global monitor."""
    return performance_monitor.record_latency_fallback(stage, ticker, reason, inference_stats)


def get_performance_status() -> Dict:
    """Get current performance status"""
    return performance_monitor.get_performance_summary(24)
