"""
LLM Performance Optimizer
Fine-tunes model selection for optimal speed and accuracy
"""

import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    FAST = "fast"      # For quick iterations
    BALANCED = "balanced"  # For production use
    ACCURATE = "accurate"  # For critical analysis


@dataclass
class ModelPerformance:
    """Performance metrics for a specific model"""
    model_name: str
    avg_inference_time: float
    avg_tokens_per_second: float
    success_rate: float
    accuracy_score: float
    memory_usage: float
    last_updated: datetime


@dataclass
class OptimizationResult:
    """Result of performance optimization"""
    recommended_model: str
    expected_inference_time: float
    expected_accuracy: float
    optimization_reason: str
    alternative_models: List[str]


class LLMPerformanceOptimizer:
    """
    Optimizes LLM model selection based on performance metrics
    """
    
    def __init__(self):
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.optimization_thresholds = {
            'max_inference_time': 30.0,  # 30 seconds
            'min_tokens_per_second': 5.0,  # 5 tokens/sec
            'min_success_rate': 0.95,  # 95% success rate
            'min_accuracy': 0.60  # 60% accuracy
        }
        
        # Model characteristics (based on actual performance)
        self.model_characteristics = {
            'qwen:14b-chat-q4_K_M': {
                'type': ModelType.ACCURATE,
                'typical_inference_time': 20.0,
                'typical_tokens_per_second': 8.0,
                'memory_usage': 9.4,  # GB
                'use_case': 'Complex financial reasoning'
            },
            'deepseek-coder:6.7b-instruct-q4_K_M': {
                'type': ModelType.FAST,
                'typical_inference_time': 8.0,
                'typical_tokens_per_second': 15.0,
                'memory_usage': 4.1,  # GB
                'use_case': 'Fast inference backup'
            },
            'codellama:13b-instruct-q4_K_M': {
                'type': ModelType.BALANCED,
                'typical_inference_time': 12.0,
                'typical_tokens_per_second': 12.0,
                'memory_usage': 7.9,  # GB
                'use_case': 'Balanced performance'
            }
        }
    
    def update_model_performance(self, model_name: str, 
                               inference_time: float,
                               tokens_per_second: float,
                               success: bool,
                               accuracy_score: Optional[float] = None):
        """Update performance metrics for a model"""
        
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformance(
                model_name=model_name,
                avg_inference_time=inference_time,
                avg_tokens_per_second=tokens_per_second,
                success_rate=1.0 if success else 0.0,
                accuracy_score=accuracy_score or 0.0,
                memory_usage=self.model_characteristics.get(model_name, {}).get('memory_usage', 0.0),
                last_updated=datetime.now()
            )
        else:
            # Update with exponential moving average
            performance = self.model_performance[model_name]
            alpha = 0.1  # Learning rate
            
            performance.avg_inference_time = (
                alpha * inference_time + (1 - alpha) * performance.avg_inference_time
            )
            performance.avg_tokens_per_second = (
                alpha * tokens_per_second + (1 - alpha) * performance.avg_tokens_per_second
            )
            
            # Update success rate
            if success:
                performance.success_rate = min(1.0, performance.success_rate + 0.01)
            else:
                performance.success_rate = max(0.0, performance.success_rate - 0.05)
            
            # Update accuracy if provided
            if accuracy_score is not None:
                performance.accuracy_score = (
                    alpha * accuracy_score + (1 - alpha) * performance.accuracy_score
                )
            
            performance.last_updated = datetime.now()
    
    def get_optimal_model(self, 
                         use_case: str = "balanced",
                         max_inference_time: Optional[float] = None,
                         min_accuracy: Optional[float] = None) -> OptimizationResult:
        """
        Get the optimal model for a specific use case
        
        Args:
            use_case: "fast", "balanced", "accurate", or "real_time"
            max_inference_time: Maximum allowed inference time
            min_accuracy: Minimum required accuracy
            
        Returns:
            OptimizationResult with recommended model
        """
        
        # Set default constraints
        max_time = max_inference_time or self.optimization_thresholds['max_inference_time']
        min_acc = min_accuracy or self.optimization_thresholds['min_accuracy']
        
        # Filter models by constraints
        suitable_models = []
        
        for model_name, performance in self.model_performance.items():
            if (performance.avg_inference_time <= max_time and
                performance.accuracy_score >= min_acc and
                performance.success_rate >= self.optimization_thresholds['min_success_rate']):
                suitable_models.append((model_name, performance))
        
        if not suitable_models:
            # Fallback to model characteristics if no performance data
            suitable_models = self._get_fallback_models(max_time, min_acc)
        
        if not suitable_models:
            return OptimizationResult(
                recommended_model="qwen:14b-chat-q4_K_M",  # Default fallback
                expected_inference_time=20.0,
                expected_accuracy=0.60,
                optimization_reason="No suitable models found, using default",
                alternative_models=[]
            )
        
        # Select model based on use case
        if use_case == "fast":
            # Prioritize speed
            best_model = min(suitable_models, key=lambda x: x[1].avg_inference_time)
        elif use_case == "accurate":
            # Prioritize accuracy
            best_model = max(suitable_models, key=lambda x: x[1].accuracy_score)
        elif use_case == "real_time":
            # Prioritize both speed and accuracy
            best_model = self._optimize_for_real_time(suitable_models)
        else:  # balanced
            # Balance speed and accuracy
            best_model = self._optimize_balanced(suitable_models)
        
        model_name, performance = best_model
        
        # Get alternatives
        alternatives = [m[0] for m in suitable_models if m[0] != model_name]
        
        return OptimizationResult(
            recommended_model=model_name,
            expected_inference_time=performance.avg_inference_time,
            expected_accuracy=performance.accuracy_score,
            optimization_reason=f"Optimized for {use_case} use case",
            alternative_models=alternatives[:2]  # Top 2 alternatives
        )
    
    def _get_fallback_models(self, max_time: float, min_accuracy: float) -> List[Tuple[str, ModelPerformance]]:
        """Get fallback models based on characteristics when no performance data available"""
        fallback_models = []
        
        for model_name, characteristics in self.model_characteristics.items():
            if (characteristics['typical_inference_time'] <= max_time and
                characteristics['typical_tokens_per_second'] >= self.optimization_thresholds['min_tokens_per_second']):
                
                performance = ModelPerformance(
                    model_name=model_name,
                    avg_inference_time=characteristics['typical_inference_time'],
                    avg_tokens_per_second=characteristics['typical_tokens_per_second'],
                    success_rate=0.95,  # Assume good success rate
                    accuracy_score=0.60,  # Assume baseline accuracy
                    memory_usage=characteristics['memory_usage'],
                    last_updated=datetime.now()
                )
                fallback_models.append((model_name, performance))
        
        return fallback_models
    
    def _optimize_for_real_time(self, models: List[Tuple[str, ModelPerformance]]) -> Tuple[str, ModelPerformance]:
        """Optimize for real-time trading (speed + accuracy)"""
        # Score based on speed and accuracy
        scored_models = []
        
        for model_name, performance in models:
            # Speed score (inverse of inference time)
            speed_score = 1.0 / max(performance.avg_inference_time, 0.1)
            
            # Accuracy score
            accuracy_score = performance.accuracy_score
            
            # Combined score (weighted)
            combined_score = 0.6 * speed_score + 0.4 * accuracy_score
            scored_models.append((combined_score, model_name, performance))
        
        # Return best scoring model
        scored_models.sort(reverse=True)
        return scored_models[0][1], scored_models[0][2]
    
    def _optimize_balanced(self, models: List[Tuple[str, ModelPerformance]]) -> Tuple[str, ModelPerformance]:
        """Optimize for balanced performance"""
        # Score based on multiple factors
        scored_models = []
        
        for model_name, performance in models:
            # Speed score (inverse of inference time)
            speed_score = 1.0 / max(performance.avg_inference_time, 0.1)
            
            # Token rate score
            token_score = performance.avg_tokens_per_second / 20.0  # Normalize to 20 tokens/sec
            
            # Accuracy score
            accuracy_score = performance.accuracy_score
            
            # Success rate score
            success_score = performance.success_rate
            
            # Combined score (equal weights)
            combined_score = (speed_score + token_score + accuracy_score + success_score) / 4.0
            scored_models.append((combined_score, model_name, performance))
        
        # Return best scoring model
        scored_models.sort(reverse=True)
        return scored_models[0][1], scored_models[0][2]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.model_performance:
            return {"error": "No performance data available"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "models_analyzed": len(self.model_performance),
            "model_performance": {},
            "optimization_recommendations": {}
        }
        
        # Individual model performance
        for model_name, performance in self.model_performance.items():
            report["model_performance"][model_name] = {
                "avg_inference_time": performance.avg_inference_time,
                "avg_tokens_per_second": performance.avg_tokens_per_second,
                "success_rate": performance.success_rate,
                "accuracy_score": performance.accuracy_score,
                "memory_usage_gb": performance.memory_usage,
                "last_updated": performance.last_updated.isoformat(),
                "performance_status": self._assess_model_status(performance)
            }
        
        # Optimization recommendations
        for use_case in ["fast", "balanced", "accurate", "real_time"]:
            try:
                result = self.get_optimal_model(use_case)
                report["optimization_recommendations"][use_case] = {
                    "recommended_model": result.recommended_model,
                    "expected_inference_time": result.expected_inference_time,
                    "expected_accuracy": result.expected_accuracy,
                    "reason": result.optimization_reason,
                    "alternatives": result.alternative_models
                }
            except Exception as e:
                report["optimization_recommendations"][use_case] = {"error": str(e)}
        
        return report
    
    def _assess_model_status(self, performance: ModelPerformance) -> str:
        """Assess model performance status"""
        if performance.success_rate < 0.9:
            return "DEGRADED"
        elif performance.avg_inference_time > 30.0:
            return "SLOW"
        elif performance.accuracy_score < 0.5:
            return "LOW_ACCURACY"
        else:
            return "HEALTHY"
    
    def optimize_for_task(self, task_description: str) -> OptimizationResult:
        """
        Optimize model selection based on task description
        
        Args:
            task_description: Description of the task to optimize for
            
        Returns:
            OptimizationResult with recommended model
        """
        task_lower = task_description.lower()
        
        # Determine use case from task description
        if any(word in task_lower for word in ["real-time", "live", "streaming", "immediate"]):
            use_case = "real_time"
        elif any(word in task_lower for word in ["analysis", "research", "detailed", "comprehensive"]):
            use_case = "accurate"
        elif any(word in task_lower for word in ["quick", "fast", "rapid", "batch"]):
            use_case = "fast"
        else:
            use_case = "balanced"
        
        # Adjust constraints based on task
        max_time = 15.0 if "real-time" in task_lower else 30.0
        min_accuracy = 0.7 if "analysis" in task_lower else 0.6
        
        return self.get_optimal_model(use_case, max_time, min_accuracy)


# Global optimizer instance
performance_optimizer = LLMPerformanceOptimizer()


def optimize_model_selection(use_case: str = "balanced") -> OptimizationResult:
    """Convenience function to get optimal model"""
    return performance_optimizer.get_optimal_model(use_case)


def update_model_performance(model_name: str, inference_time: float,
                           tokens_per_second: float, success: bool,
                           accuracy_score: Optional[float] = None):
    """Convenience function to update model performance"""
    performance_optimizer.update_model_performance(
        model_name, inference_time, tokens_per_second, success, accuracy_score
    )
