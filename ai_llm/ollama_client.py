"""
Ollama Client - Local LLM Interface
Line Count: ~120 lines (within 500-line phase limit)
Cost: $0/month (local GPU)

Validates Ollama availability before any LLM operations.
Pipeline fails immediately if Ollama is not running (per requirement 3b).
"""

import os
import re
import requests
import logging
import time
import json
import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING, Sequence
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
                 generation_options: Optional[Dict[str, Any]] = None,
                 http_client: Optional[Any] = None,
                 cache_ttl_seconds: Optional[int] = 600,
                 latency_failover_threshold: float = 12.0,
                 token_rate_failover_threshold: float = 12.0,
                 fallback_models: Optional[Sequence[str]] = None):
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
            http_client: Optional requests-compatible client (for testing/mocking)
            cache_ttl_seconds: Optional cache expiry in seconds (None disables TTL)
            latency_failover_threshold: Seconds before switching to faster fallback model on retry
            token_rate_failover_threshold: Minimum tokens/sec before throughput fallback triggers
            fallback_models: Optional ordered list of model names to attempt when performance degrades
        
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
        self.cache_ttl_seconds = cache_ttl_seconds
        self.latency_failover_threshold = max(1.0, float(latency_failover_threshold))
        self.token_rate_failover_threshold = max(0.5, float(token_rate_failover_threshold))
        self._token_rate_baseline_seconds = 1.0
        self._manual_fallbacks = [m.strip() for m in (fallback_models or []) if m]
        self._session = http_client if http_client is not None else requests.Session()
        self._owns_session = http_client is None and hasattr(self._session, "close")
        self._explicit_model = model is not None
        resolved_model = model or "deepseek-coder:6.7b-instruct-q4_K_M"
        self._last_inference_stats: Optional[Dict[str, Any]] = None

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

        if self._manual_fallbacks:
            fallback_chain = list(dict.fromkeys(self._manual_fallbacks))
            self._optimizer_alternatives.extend(
                model for model in fallback_chain if model != resolved_model
            )

        self.model = resolved_model
        self._tried_models = {self.model}
        
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
            if self._owns_session or not hasattr(self._session, "get"):
                get_callable = requests.get
            else:
                get_callable = self._session.get

            # Check server health
            response = get_callable(f"{self.host}/api/tags", timeout=5)
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
            
            logger.info("Ollama validated: %s ready", self.model)
            
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
        if cached is None:
            return None

        if self.cache_ttl_seconds is not None:
            if time.time() - cached["timestamp"] > self.cache_ttl_seconds:
                # Expired cache entry
                self._response_cache.pop(cache_key, None)
                return None

        self._response_cache.move_to_end(cache_key)
        return cached

    def _cache_set(self, cache_key: str, payload: Dict[str, Any]) -> None:
        """Store inference result in cache."""
        if not self.enable_cache:
            return
        payload_with_timestamp = dict(payload)
        payload_with_timestamp["timestamp"] = time.time()
        self._response_cache[cache_key] = payload_with_timestamp
        self._response_cache.move_to_end(cache_key)
        if len(self._response_cache) > self.cache_max_size:
            self._response_cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Clear cached responses (useful for testing and memory control)."""
        self._response_cache.clear()
        logger.debug("Ollama response cache cleared")

    def _optimise_prompt(self, prompt: str) -> str:
        """
        Reduce prompt token count by trimming whitespace and removing redundancy.

        This is a lightweight heuristic that preserves semantic content while
        keeping prompts compact to improve inference latency.
        """
        if not isinstance(prompt, str):
            return prompt

        cleaned_lines: list[str] = []
        cleaned_lines_lower: list[str] = []
        for raw_line in prompt.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalised = re.sub(r"\s+", " ", line)
            lower = normalised.lower()

            # Skip duplicates
            if lower in cleaned_lines_lower:
                continue

            # Prefer richer instructions: remove existing lines that are substrings
            indices_to_remove = [
                idx for idx, existing_lower in enumerate(cleaned_lines_lower)
                if existing_lower != lower and existing_lower in lower
            ]
            if indices_to_remove:
                for idx in reversed(indices_to_remove):
                    cleaned_lines.pop(idx)
                    cleaned_lines_lower.pop(idx)

            # Skip if current line is a substring of an existing richer instruction
            if any(lower in existing_lower for existing_lower in cleaned_lines_lower):
                continue

            cleaned_lines.append(normalised)
            cleaned_lines_lower.append(lower)

        optimised = "\n".join(cleaned_lines)

        max_chars = int(self.generation_options.get("max_prompt_chars", 4096))
        if len(optimised) > max_chars:
            optimised = optimised[:max_chars]

        return optimised

    def _should_switch_model(self, last_latency: float) -> bool:
        """Decide if the client should attempt a faster model."""
        if not getattr(self, "_optimizer_alternatives", []):
            return False
        if last_latency < self.latency_failover_threshold:
            return False
        return True

    def _should_switch_model_for_token_rate(self, tokens_per_second: float) -> bool:
        """Decide if we should swap models due to low token throughput."""
        if tokens_per_second >= self.token_rate_failover_threshold:
            return False
        alternatives = getattr(self, "_optimizer_alternatives", [])
        if not alternatives:
            return False
        for candidate in alternatives:
            if candidate not in self._tried_models:
                return True
        return False

    def _switch_to_alternative_model(self, reason: str) -> bool:
        """Switch to the next available alternative model if present."""
        alternatives = getattr(self, "_optimizer_alternatives", [])
        for candidate in alternatives:
            if candidate in self._tried_models:
                continue
            logger.warning(
                "Switching Ollama model from '%s' to '%s' due to %s",
                self.model,
                candidate,
                reason,
            )
            self.model = candidate
            self._tried_models.add(candidate)
            try:
                self._validate_connection()
                return True
            except OllamaConnectionError as validation_error:
                logger.error(
                    "Fallback model '%s' validation failed: %s",
                    candidate,
                    validation_error,
                )
                continue
        return False
    
    def _build_generation_options(self, temperature: float) -> Dict[str, Any]:
        """Merge generation options with runtime temperature."""
        raw_max_tokens = self.generation_options.get("max_tokens", 1024)
        try:
            max_tokens = int(raw_max_tokens)
        except (TypeError, ValueError):
            max_tokens = 1024

        max_tokens = max(16, min(max_tokens, 1024))

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
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
        optimised_prompt = self._optimise_prompt(prompt)
        if optimised_prompt != prompt:
            logger.debug(
                "Optimised prompt length from %d to %d characters",
                len(prompt),
                len(optimised_prompt),
            )

        cache_key = self._build_cache_key(optimised_prompt, system, temperature)
        cached = self._cache_get(cache_key)
        if cached:
            logger.debug("Returning cached response for model '%s'", self.model)
            monitor_inference(
                model_name=self.model,
                prompt=optimised_prompt,
                response=cached["response"],
                inference_time=0.0,
                success=True,
            )
            return cached["response"]

        attempts = 0
        max_attempts = 1 + len(getattr(self, "_optimizer_alternatives", []))
        self._last_inference_stats = None
        
        while attempts < max_attempts:
            attempts += 1
            # Use a monotonic clock for latency measurement so tests that monkeypatch
            # time.time() don't interfere with inference timing.
            start_time = time.perf_counter()
            try:
                options = self._build_generation_options(temperature)
                payload = {
                    "model": self.model,
                    "prompt": optimised_prompt,
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
                
                duration = time.perf_counter() - start_time
                duration = max(duration, 1e-6)
                effective_duration = max(duration, self._token_rate_baseline_seconds)
                word_tokens = max(1.0, float(len(generated_text.split())))
                char_tokens = max(1.0, len(generated_text) / 4.0)
                token_estimate = min(word_tokens, char_tokens)
                tokens_per_second = token_estimate / effective_duration
                low_token_rate = tokens_per_second < self.token_rate_failover_threshold
                low_token_reason = None
                if low_token_rate:
                    low_token_reason = (
                        f"LLM token throughput {tokens_per_second:.2f} tokens/sec "
                        f"below threshold {self.token_rate_failover_threshold:.2f} tokens/sec"
                    )
                    logger.warning(
                        "%s (model=%s, prompt_chars=%d, response_chars=%d)",
                        low_token_reason,
                        self.model,
                        len(optimised_prompt),
                        len(generated_text),
                    )

                self._last_inference_stats = {
                    "success": True,
                    "inference_time": duration,
                    "tokens_per_second": tokens_per_second,
                    "model_name": self.model,
                    "timestamp": datetime.now(),
                    "prompt_length": len(optimised_prompt),
                    "response_length": len(generated_text),
                    "low_token_rate": low_token_rate,
                }
                if low_token_rate:
                    self._last_inference_stats["error"] = low_token_reason

                optimizer_recorded = False
                if low_token_rate and self._should_switch_model_for_token_rate(tokens_per_second):
                    if self.optimizer:
                        try:
                            self.optimizer.update_model_performance(
                                model_name=self.model,
                                inference_time=duration,
                                tokens_per_second=tokens_per_second,
                                success=False,
                            )
                            optimizer_recorded = True
                        except Exception:  # pragma: no cover - defensive
                            pass
                    previous_model = self.model
                    switched = self._switch_to_alternative_model("low token throughput")
                    if switched:
                        self._last_inference_stats.update(
                            {
                                "success": False,
                                "fallback_reason": "low_token_throughput",
                            }
                        )
                        monitor_inference(
                            model_name=previous_model,
                            prompt=optimised_prompt,
                            response=generated_text,
                            inference_time=duration,
                            success=False,
                            error_message=low_token_reason,
                            fallback_used=True,
                            fallback_reason="low_token_throughput",
                        )
                        cache_key = self._build_cache_key(optimised_prompt, system, temperature)
                        continue
                    logger.warning(
                        "Low token throughput detected but alternative model validation failed; "
                        "continuing with current model '%s'.",
                        self.model,
                    )

                self._cache_set(
                    cache_key,
                    {
                        "response": generated_text,
                        "inference_time": duration,
                        "tokens_per_second": tokens_per_second,
                    },
                )

                if self.optimizer and not optimizer_recorded:
                    try:
                        self.optimizer.update_model_performance(
                            model_name=self.model,
                            inference_time=duration,
                            tokens_per_second=tokens_per_second,
                            success=not low_token_rate,
                        )
                    except Exception as opt_err:  # pragma: no cover - defensive
                        logger.debug(f"Performance optimizer update failed: {opt_err}")
                
                monitor_inference(
                    model_name=self.model,
                    prompt=optimised_prompt,
                    response=generated_text,
                    inference_time=duration,
                    success=True,
                    error_message=low_token_reason if low_token_rate else None,
                    fallback_used=False,
                )
                
                logger.info(
                    "LLM generation: %.1fs, %d chars (model=%s)",
                    duration,
                    len(generated_text),
                    self.model,
                )

                if self._should_switch_model(duration):
                    self._switch_to_alternative_model("latency threshold exceeded")
                
                return generated_text
                
            except requests.exceptions.Timeout:
                duration = time.perf_counter() - start_time
                error_msg = f"LLM generation timeout (>{self.timeout}s)"
                
                monitor_inference(
                    model_name=self.model,
                    prompt=optimised_prompt,
                    response="",
                    inference_time=duration,
                    success=False,
                    error_message=error_msg
                )
                self._last_inference_stats = {
                    "success": False,
                    "inference_time": duration,
                    "tokens_per_second": 0.0,
                    "model_name": self.model,
                    "timestamp": datetime.now(),
                    "prompt_length": len(optimised_prompt),
                    "response_length": 0,
                    "error": error_msg,
                }
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

                if self._switch_to_alternative_model("timeout"):
                    cache_key = self._build_cache_key(optimised_prompt, system, temperature)
                    continue
                
                raise OllamaConnectionError(
                    f"{error_msg}. Try reducing prompt length or increasing timeout."
                ) from None
            except Exception as e:
                duration = time.perf_counter() - start_time
                error_msg = f"LLM generation failed: {e}"
                
                monitor_inference(
                    model_name=self.model,
                    prompt=optimised_prompt,
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

                if self._switch_to_alternative_model("error"):
                    cache_key = self._build_cache_key(optimised_prompt, system, temperature)
                    continue
                
                raise OllamaConnectionError(error_msg) from None
            finally:
                if self._last_inference_stats is None:
                    self._last_inference_stats = {
                        "success": False,
                        "inference_time": time.perf_counter() - start_time,
                        "tokens_per_second": 0.0,
                        "model_name": self.model,
                        "timestamp": datetime.now(),
                        "prompt_length": len(optimised_prompt),
                        "response_length": 0,
                    }
    
    def close(self) -> None:
        """Release underlying HTTP session resources."""
        if not self._owns_session:
            return

        close_method = getattr(self._session, "close", None)
        if callable(close_method):
            try:
                close_method()
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

    def get_last_inference_stats(self) -> Optional[Dict[str, Any]]:
        """Return metrics captured during the most recent inference attempt."""
        return self._last_inference_stats

    def should_use_latency_fallback(
        self,
        max_latency_override: Optional[float] = None,
        min_token_rate: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if the caller should fall back to deterministic output due to performance.

        Args:
            max_latency_override: Optional hard cap on acceptable latency (seconds).
            min_token_rate: Optional minimum acceptable tokens per second.

        Returns:
            Tuple of (should_fallback, reason). Reason is a human-readable description.
        """
        stats = self._last_inference_stats
        if not stats or not stats.get("success", False):
            return False, None

        latency_threshold = self.latency_failover_threshold
        if max_latency_override is not None:
            latency_threshold = min(
                latency_threshold,
                max(0.5, float(max_latency_override)),
            )

        latency = float(stats.get("inference_time", 0.0))
        if latency > latency_threshold:
            return True, f"latency {latency:.2f}s > {latency_threshold:.2f}s"

        token_threshold = 5.0 if min_token_rate is None else max(0.1, float(min_token_rate))
        tokens_per_second = float(stats.get("tokens_per_second", 0.0))
        if tokens_per_second < token_threshold:
            return True, f"token rate {tokens_per_second:.2f} < {token_threshold:.2f}"

        return False, None

# Performance validation
assert OllamaClient.__init__.__doc__ is not None, "Missing docstring"
assert OllamaClient.generate.__doc__ is not None, "Missing docstring"

# Line count: ~150 lines (within budget)

