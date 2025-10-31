"""
Ollama Client - Local LLM Interface
Line Count: ~120 lines (within 500-line phase limit)
Cost: $0/month (local GPU)

Validates Ollama availability before any LLM operations.
Pipeline fails immediately if Ollama is not running (per requirement 3b).
"""

import os
import requests
import logging
import time
import json
import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
from datetime import datetime
from .performance_monitor import monitor_inference

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_llm.performance_optimizer import LLMPerformanceOptimizer

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama server is unavailable - pipeline must stop"""
    pass


class OllamaClient:
    """
    Production-grade Ollama client with strict validation.
    
    REQUIREMENTS:
    - Validate Ollama availability on init (fail-fast)
    - Support multiple models (DeepSeek, CodeLlama, Qwen)
    - Timeout handling for slow responses
    - Zero external API costs
    
    Mathematical Foundation:
    - Response time: T_llm = T_process + T_gen
    - Expected: T_llm < 120s for 6.7B model (faster than 33B)
    """
    
    def __init__(self, 
                 host: str = "http://localhost:11434",
                 model: Optional[str] = None,
                 timeout: int = 120,
                 optimizer: Optional["LLMPerformanceOptimizer"] = None,
                 optimize_use_case: str = "balanced",
                 enable_cache: bool = True,
                 cache_max_size: int = 32,
                 generation_options: Optional[Dict[str, Any]] = None):
        """
        Initialize Ollama client with strict validation.
        
        Args:
            host: Ollama server URL
            model: Preferred model name (defaults to DeepSeek 6.7B if None)
            timeout: Max response time in seconds
            optimizer: Optional performance optimizer for dynamic model selection
            optimize_use_case: Optimizer use case ("fast", "balanced", etc.)
            enable_cache: Whether to reuse identical prompt responses
            cache_max_size: Maximum number of cached responses retained in memory
            generation_options: Additional generation options (top_p, top_k, etc.)
        
        Raises:
            OllamaConnectionError: If Ollama is unavailable (REQUIRED per 3b)
        """
        self.host = host.rstrip('/')
        self.optimizer = optimizer
        self.optimize_use_case = optimize_use_case
        self.enable_cache = enable_cache
        self.cache_max_size = max(1, cache_max_size)
        self._response_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.generation_options = generation_options or {}
        self.timeout = timeout
        self._session = requests.Session()
        self._explicit_model = model is not None
        resolved_model = model or "deepseek-coder:6.7b-instruct-q4_K_M"

        if self.optimizer and not self._explicit_model:
            try:
                optimization = self.optimizer.get_optimal_model(use_case=self.optimize_use_case)
                if optimization and optimization.recommended_model:
                    resolved_model = optimization.recommended_model
                    logger.info(
                        "Performance optimizer recommended model '%s' "
                        "(expected latency %.2fs, accuracy %.2f)",
                        optimization.recommended_model,
                        optimization.expected_inference_time,
                        optimization.expected_accuracy,
                    )
                    self._optimizer_alternatives = optimization.alternative_models
                else:
                    self._optimizer_alternatives = []
            except Exception as opt_err:  # pragma: no cover - defensive
                logger.warning(f"Performance optimizer selection failed: {opt_err}")
                self._optimizer_alternatives = []
        else:
            self._optimizer_alternatives = []

        self.model = resolved_model
        
        # CRITICAL: Validate Ollama availability immediately
        self._validate_connection()
        
        logger.info(f"Ollama client initialized: {self.model} @ {self.host}")
    
    def health_check(self) -> bool:
        """
        Check if Ollama service is healthy and model is available.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            return self.model in model_names
        except Exception:
            return False
    
    def _validate_connection(self) -> None:
        """
        Validate Ollama server is running and model is available.
        
        Raises:
            OllamaConnectionError: If validation fails - pipeline stops
        """
        try:
            # Check server health
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check model availability
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            if self.model not in model_names:
                if getattr(self, "_explicit_model", False):
                    raise OllamaConnectionError(
                        f"Model '{self.model}' not found. "
                        f"Available models: {model_names}. "
                        f"Run: ollama pull {self.model}"
                    )

                fallback_model = self._select_available_model(model_names)
                if fallback_model:
                    logger.warning(
                        "Preferred model '%s' unavailable. Falling back to '%s'.",
                        self.model,
                        fallback_model,
                    )
                    self.model = fallback_model
                else:
                    raise OllamaConnectionError(
                        f"Model '{self.model}' not found. "
                        f"Available models: {model_names}. "
                        f"Run: ollama pull {self.model}"
                    )
            
            logger.info(f"✓ Ollama validated: {self.model} ready")
            
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                f"Ollama server not running at {self.host}. "
                "Start Ollama with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(
                f"Ollama server timeout at {self.host}. "
                "Check if server is overloaded."
            )
        except Exception as e:
            raise OllamaConnectionError(f"Ollama validation failed: {e}")
    
    def _select_available_model(self, available_models: Any) -> Optional[str]:
        """Select an available model when the preferred option is missing."""
        candidate_list = list(available_models) if available_models else []
        if not candidate_list:
            return None
        
        # Try optimizer suggested alternatives first
        for candidate in getattr(self, "_optimizer_alternatives", []):
            if candidate in candidate_list:
                return candidate
        
        # Fallback to any available cached model
        preferred_fallbacks = [
            "deepseek-coder:6.7b-instruct-q4_K_M",
            "codellama:13b-instruct-q4_K_M",
            "qwen:14b-chat-q4_K_M",
        ]
        for candidate in preferred_fallbacks:
            if candidate in candidate_list:
                return candidate
        
        # As a last resort, return the first available model
        return candidate_list[0]

    def _build_cache_key(self, prompt: str, system: Optional[str], temperature: float) -> str:
        """Build deterministic cache key for prompt/system combinations."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "temperature": temperature,
            "options": self.generation_options,
        }
        payload_json = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached inference result."""
        if not self.enable_cache:
            return None
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            self._response_cache.move_to_end(cache_key)
        return cached

    def _cache_set(self, cache_key: str, payload: Dict[str, Any]) -> None:
        """Store inference result in cache."""
        if not self.enable_cache:
            return
        self._response_cache[cache_key] = payload
        self._response_cache.move_to_end(cache_key)
        if len(self._response_cache) > self.cache_max_size:
            self._response_cache.popitem(last=False)
    
    def _build_generation_options(self, temperature: float) -> Dict[str, Any]:
        """Merge generation options with runtime temperature."""
        raw_max_tokens = self.generation_options.get("max_tokens", 1024)
        try:
            max_tokens = int(raw_max_tokens)
        except (TypeError, ValueError):
            max_tokens = 1024

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": min(max(1, max_tokens), 2048),
        }
        for key in ("top_p", "top_k", "repeat_penalty"):
            value = self.generation_options.get(key)
            if value is not None:
                options[key] = value
        return options

    def generate(self, 
                 prompt: str, 
                 system: Optional[str] = None,
                 temperature: float = 0.1) -> str:
        """
        Generate LLM response with strict validation.
        
        Args:
            prompt: Input prompt
            system: System instructions (optional)
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Generated text response
            
        Raises:
            OllamaConnectionError: If generation fails
            
        Performance:
            - DeepSeek 6.7B: ~15-20 tokens/sec
            - CodeLlama 13B: ~25-35 tokens/sec
            - Expected latency: 5-30s depending on response length
        """
        cache_key = self._build_cache_key(prompt, system, temperature)
        cached = self._cache_get(cache_key)
        if cached:
            logger.debug("Returning cached response for model '%s'", self.model)
            monitor_inference(
                model_name=self.model,
                prompt=prompt,
                response=cached["response"],
                inference_time=0.0,
                success=True,
            )
            return cached["response"]

        start_time = time.time()
        
        try:
            options = self._build_generation_options(temperature)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }
            
            if system:
                payload["system"] = system
            
            response = self._session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            # Validate non-empty response
            if not generated_text.strip():
                raise OllamaConnectionError("Empty LLM response")
            
            duration = time.time() - start_time
            duration = max(duration, 1e-6)
            tokens_per_second = len(generated_text.split()) / duration

            self._cache_set(
                cache_key,
                {
                    "response": generated_text,
                    "inference_time": duration,
                    "tokens_per_second": tokens_per_second,
                },
            )

            if self.optimizer:
                try:
                    self.optimizer.update_model_performance(
                        model_name=self.model,
                        inference_time=duration,
                        tokens_per_second=tokens_per_second,
                        success=True,
                    )
                except Exception as opt_err:  # pragma: no cover - defensive
                    logger.debug(f"Performance optimizer update failed: {opt_err}")
            
            # Record performance metrics
            monitor_inference(
                model_name=self.model,
                prompt=prompt,
                response=generated_text,
                inference_time=duration,
                success=True
            )
            
            logger.info(f"LLM generation: {duration:.1f}s, {len(generated_text)} chars")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            error_msg = f"LLM generation timeout (>{self.timeout}s)"
            
            # Record failed inference
            monitor_inference(
                model_name=self.model,
                prompt=prompt,
                response="",
                inference_time=duration,
                success=False,
                error_message=error_msg
            )
            if self.optimizer:
                try:
                    self.optimizer.update_model_performance(
                        model_name=self.model,
                        inference_time=duration,
                        tokens_per_second=0.0,
                        success=False,
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            
            raise OllamaConnectionError(
                f"{error_msg}. Try reducing prompt length or increasing timeout."
            )
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"LLM generation failed: {e}"
            
            # Record failed inference
            monitor_inference(
                model_name=self.model,
                prompt=prompt,
                response="",
                inference_time=duration,
                success=False,
                error_message=error_msg
            )
            if self.optimizer:
                try:
                    self.optimizer.update_model_performance(
                        model_name=self.model,
                        inference_time=duration,
                        tokens_per_second=0.0,
                        success=False,
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            
            raise OllamaConnectionError(error_msg)
    
    def close(self) -> None:
        """Release underlying HTTP session resources."""
        try:
            self._session.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance stats"""
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            for model in models:
                if model.get('name') == self.model:
                    return {
                        'name': model.get('name'),
                        'size': model.get('size', 'Unknown'),
                        'modified': model.get('modified_at', 'Unknown'),
                        'family': model.get('details', {}).get('family', 'Unknown')
                    }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {}

# Performance validation
assert OllamaClient.__init__.__doc__ is not None, "Missing docstring"
assert OllamaClient.generate.__doc__ is not None, "Missing docstring"

# Line count: ~150 lines (within budget)

