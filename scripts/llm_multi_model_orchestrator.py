#!/usr/bin/env python3
"""
Multi-model LLM orchestrator for Portfolio Maximizer.

Fully self-contained orchestration layer that manages the 3-model local LLM
strategy and integrates with OpenClaw for message-driven interactions.

Strategy:
  - qwen3:8b       -> tool-calling orchestrator (structured output, function dispatch)
  - deepseek-r1:8b -> fast reasoning (quant analysis, signal validation)
  - deepseek-r1:32b -> heavy reasoning (adversarial audits, complex analysis)

OpenClaw Integration:
  - Registers as an OpenClaw agent skill for message-driven orchestration
  - Routes incoming messages through the 3-model pipeline
  - Delivers responses back via OpenClaw channels (WhatsApp, Telegram, Discord)
  - Health-checks models before each orchestration cycle
  - Auto-fallback when models are unavailable or slow

Usage:
  python scripts/llm_multi_model_orchestrator.py status
  python scripts/llm_multi_model_orchestrator.py health
  python scripts/llm_multi_model_orchestrator.py route --task "Analyze AAPL regime"
  python scripts/llm_multi_model_orchestrator.py orchestrate --prompt "Summarize gate status"
  python scripts/llm_multi_model_orchestrator.py openclaw-bridge --channel whatsapp --message "Check PnL"
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap .env before any config/secret access
from etl.secret_loader import bootstrap_dotenv
bootstrap_dotenv()

from ai_llm.llm_activity_logger import get_logger as _get_activity_logger
from scripts.llm_operator_tools import (
    review_changed_files as _review_changed_files_helper,
    summarize_test_failure as _summarize_test_failure_helper,
    triage_gate_failure as _triage_gate_failure_helper,
)
import yaml

logger = logging.getLogger("pmx.orchestrator")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    role: str  # "orchestrator" | "reasoning" | "heavy_reasoning"
    capabilities: list[str] = field(default_factory=list)
    vram_gb: float = 0.0
    speed: str = "fast"  # "fast" | "medium" | "slow"
    supports_tools: bool = False
    max_tokens: int = 8192
    context_window: int = 65536
    # Runtime state
    available: bool = False
    last_health_check: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    error_streak: int = 0
    cooldown_until: float = 0.0


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen3:8b": ModelSpec(
        name="qwen3:8b",
        role="orchestrator",
        capabilities=["tool-calling", "function-calling", "structured-output", "thinking-mode"],
        vram_gb=5.5,
        speed="fast",
        supports_tools=True,
        context_window=32768,
    ),
    "deepseek-r1:8b": ModelSpec(
        name="deepseek-r1:8b",
        role="reasoning",
        capabilities=["chain-of-thought", "math", "code-generation", "quantitative-analysis"],
        vram_gb=5.5,
        speed="fast",
        supports_tools=False,
    ),
    "deepseek-r1:32b": ModelSpec(
        name="deepseek-r1:32b",
        role="heavy_reasoning",
        capabilities=["chain-of-thought", "math", "code-generation", "long-context", "adversarial-analysis"],
        vram_gb=20,
        speed="slow",
        supports_tools=False,
    ),
}

# Task -> model routing table
TASK_ROUTING = {
    "market_analysis": "deepseek-r1:8b",
    "signal_generation": "deepseek-r1:8b",
    "risk_assessment": "deepseek-r1:8b",
    "regime_detection": "deepseek-r1:8b",
    "portfolio_optimization": "deepseek-r1:32b",
    "adversarial_audit": "deepseek-r1:32b",
    "complex_forecasting": "deepseek-r1:32b",
    "api_orchestration": "qwen3:8b",
    "notification_formatting": "qwen3:8b",
    "data_extraction": "qwen3:8b",
    "social_media_automation": "qwen3:8b",
    "structured_output": "qwen3:8b",
    "tool_dispatch": "qwen3:8b",
}

# Routing keywords for task classification
ROUTING_KEYWORDS = {
    "market_analysis": ["market", "price", "trend", "regime", "volatility"],
    "signal_generation": ["signal", "buy", "sell", "hold", "trade signal"],
    "risk_assessment": ["risk", "drawdown", "var", "exposure"],
    "regime_detection": ["regime", "regime detect", "market state"],
    "portfolio_optimization": ["portfolio", "optimize", "allocation", "rebalance"],
    "adversarial_audit": ["adversarial", "audit", "weakness", "vulnerability"],
    "complex_forecasting": ["forecast", "predict", "long-term", "multi-step"],
    "api_orchestration": ["api", "endpoint", "webhook", "integration"],
    "notification_formatting": ["notify", "alert", "message", "format"],
    "data_extraction": ["extract", "scrape", "parse", "etl"],
    "social_media_automation": ["social", "telegram", "discord", "whatsapp"],
    "structured_output": ["json", "structured", "schema", "format output"],
    "tool_dispatch": ["tool", "function", "call", "dispatch", "orchestrate"],
}

# ---------------------------------------------------------------------------
# Ollama client (self-contained, no external deps)
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
_HEALTH_CHECK_INTERVAL = 60.0  # seconds between health checks
_health_lock = threading.Lock()

LLM_CONFIG_PATH = Path(os.getenv("PMX_LLM_CONFIG_PATH", str(PROJECT_ROOT / "config" / "llm_config.yml")))
_openclaw_templates_raw = (
    os.getenv("PMX_OPENCLAW_PROMPT_TEMPLATES_PATH", "config/openclaw_prompt_templates.yml").strip()
)
OPENCLAW_PROMPT_TEMPLATES_PATH = Path(_openclaw_templates_raw)
if not OPENCLAW_PROMPT_TEMPLATES_PATH.is_absolute():
    OPENCLAW_PROMPT_TEMPLATES_PATH = (PROJECT_ROOT / OPENCLAW_PROMPT_TEMPLATES_PATH).resolve()


def _safe_load_llm_config(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load llm config (%s): %s", path, exc)
        return {}


def _pick_production_profile(config_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    llm_cfg = config_payload.get("llm", {}) if isinstance(config_payload, dict) else {}
    profiles_cfg = llm_cfg.get("production_profiles", {}) if isinstance(llm_cfg, dict) else {}
    profiles = profiles_cfg.get("profiles", {}) if isinstance(profiles_cfg, dict) else {}
    aliases = profiles_cfg.get("profile_aliases", {}) if isinstance(profiles_cfg, dict) else {}
    alias_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.items()}

    default_profile = str(profiles_cfg.get("selected_profile", "")).strip().lower()
    if not default_profile:
        # Backward-compatible hint from older llm_config format.
        use_case = str(llm_cfg.get("performance", {}).get("default_use_case", "")).strip().lower()
        default_profile = "high_accuracy" if use_case in {"accurate", "accuracy"} else "low_latency"

    env_profile = str(os.getenv("PMX_QWEN_PRODUCTION_PROFILE", "")).strip().lower()
    requested = env_profile or default_profile or "low_latency"
    selected = alias_map.get(requested, requested)
    selected_payload = profiles.get(selected)

    if isinstance(selected_payload, dict):
        return selected, selected_payload

    fallback = profiles.get("low_latency")
    if isinstance(fallback, dict):
        return "low_latency", fallback

    # If config is missing, use built-in hard defaults.
    return "builtin_default", {}


_LLM_CONFIG = _safe_load_llm_config(LLM_CONFIG_PATH)
ACTIVE_PRODUCTION_PROFILE_NAME, ACTIVE_PRODUCTION_PROFILE = _pick_production_profile(_LLM_CONFIG)
ACTIVE_PRODUCTION_SLA = (
    ACTIVE_PRODUCTION_PROFILE.get("sla", {})
    if isinstance(ACTIVE_PRODUCTION_PROFILE, dict)
    else {}
)


def _safe_load_openclaw_prompt_templates(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load OpenClaw prompt templates (%s): %s", path, exc)
        return {}


_OPENCLAW_PROMPT_TEMPLATES = _safe_load_openclaw_prompt_templates(OPENCLAW_PROMPT_TEMPLATES_PATH)


def _profile_env_default(key: str, fallback: Any) -> Any:
    if isinstance(ACTIVE_PRODUCTION_PROFILE, dict):
        env_defaults = ACTIVE_PRODUCTION_PROFILE.get("env_defaults", {})
        if isinstance(env_defaults, dict) and key in env_defaults:
            return env_defaults[key]
    return fallback


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return bool(fallback)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(fallback)


def _csv_env_tokens(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return tuple(default)
    tokens: list[str] = []
    for part in re.split(r"[,\n;|]", raw):
        token = str(part or "").strip().lower()
        if token:
            tokens.append(token)
    if not tokens:
        return tuple(default)
    # Keep order deterministic while removing duplicates.
    return tuple(dict.fromkeys(tokens))


def _text_contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    return any(tok in low for tok in tokens if tok)


MAX_ROUNDS_DEFAULT = max(
    1,
    _int_env("PMX_QWEN_MAX_ROUNDS", _as_int(_profile_env_default("PMX_QWEN_MAX_ROUNDS", 3), 3)),
)
MAX_TOOL_CALLS_DEFAULT = max(
    1,
    _int_env("PMX_QWEN_MAX_TOOL_CALLS", _as_int(_profile_env_default("PMX_QWEN_MAX_TOOL_CALLS", 12), 12)),
)
FORCE_TOOL_PRIMER_DEFAULT = _truthy_env(
    "PMX_QWEN_FORCE_TOOL_PRIMER",
    _as_bool(_profile_env_default("PMX_QWEN_FORCE_TOOL_PRIMER", False), False),
)
SUBAGENT_WORKFLOW_DEFAULT = _truthy_env(
    "PMX_QWEN_SUBAGENT_WORKFLOW",
    _as_bool(_profile_env_default("PMX_QWEN_SUBAGENT_WORKFLOW", False), False),
)
ORCHESTRATION_TIMEOUT_SECONDS_DEFAULT = max(
    20,
    _int_env("PMX_QWEN_ORCH_TIMEOUT_SECONDS", _as_int(_profile_env_default("PMX_QWEN_ORCH_TIMEOUT_SECONDS", 120), 120)),
)
MODEL_ERROR_STREAK_THRESHOLD = max(
    1,
    _int_env("PMX_QWEN_MODEL_ERROR_STREAK", _as_int(_profile_env_default("PMX_QWEN_MODEL_ERROR_STREAK", 3), 3)),
)
MODEL_COOLDOWN_SECONDS = float(
    max(
        30,
        _int_env("PMX_QWEN_MODEL_COOLDOWN_SECONDS", _as_int(_profile_env_default("PMX_QWEN_MODEL_COOLDOWN_SECONDS", 180), 180)),
    )
)
SUBAGENT_MAX_TASKS_DEFAULT = max(
    1,
    _int_env("PMX_QWEN_SUBAGENT_MAX_TASKS", _as_int(_profile_env_default("PMX_QWEN_SUBAGENT_MAX_TASKS", 3), 3)),
)
SUBAGENT_DEEP_LATENCY_CUTOFF_MS = float(
    max(
        1000,
        _int_env(
            "PMX_QWEN_SUBAGENT_DEEP_LATENCY_CUTOFF_MS",
            _as_int(_profile_env_default("PMX_QWEN_SUBAGENT_DEEP_LATENCY_CUTOFF_MS", 16000), 16000),
        ),
    )
)
TOOL_CACHE_TTL_SECONDS = float(
    max(
        5,
        _int_env("PMX_QWEN_TOOL_CACHE_TTL_SECONDS", _as_int(_profile_env_default("PMX_QWEN_TOOL_CACHE_TTL_SECONDS", 45), 45)),
    )
)
TOOL_CACHE_MAX_ENTRIES = max(
    16,
    _int_env("PMX_QWEN_TOOL_CACHE_MAX_ENTRIES", _as_int(_profile_env_default("PMX_QWEN_TOOL_CACHE_MAX_ENTRIES", 96), 96)),
)
CHAT_NUM_PREDICT_DEFAULT = max(
    128,
    _int_env("PMX_QWEN_CHAT_NUM_PREDICT", _as_int(_profile_env_default("PMX_QWEN_CHAT_NUM_PREDICT", 768), 768)),
)
REASONING_MAX_PROMPT_CHARS = max(
    800,
    _int_env(
        "PMX_QWEN_REASONING_MAX_PROMPT_CHARS",
        _as_int(_profile_env_default("PMX_QWEN_REASONING_MAX_PROMPT_CHARS", 3600), 3600),
    ),
)
CHAT_ROUND_TIMEOUT_CAP_SECONDS = float(
    max(
        10,
        _int_env(
            "PMX_QWEN_CHAT_ROUND_TIMEOUT_CAP_SECONDS",
            _as_int(_profile_env_default("PMX_QWEN_CHAT_ROUND_TIMEOUT_CAP_SECONDS", 25), 25),
        ),
    )
)
WHATSAPP_BRIDGE_TURN_LOCK_TIMEOUT_SECONDS = float(
    max(
        10,
        _int_env(
            "PMX_OPENCLAW_WHATSAPP_TURN_LOCK_TIMEOUT_SECONDS",
            _as_int(_profile_env_default("PMX_OPENCLAW_WHATSAPP_TURN_LOCK_TIMEOUT_SECONDS", 90), 90),
        ),
    )
)
WHATSAPP_BRIDGE_TURN_LOCK_STALE_SECONDS = float(
    max(
        60,
        _int_env(
            "PMX_OPENCLAW_WHATSAPP_TURN_LOCK_STALE_SECONDS",
            _as_int(_profile_env_default("PMX_OPENCLAW_WHATSAPP_TURN_LOCK_STALE_SECONDS", 600), 600),
        ),
    )
)
WHATSAPP_BRIDGE_TURN_LOCK_POLL_SECONDS = float(
    max(
        0.05,
        _float_env(
            "PMX_OPENCLAW_WHATSAPP_TURN_LOCK_POLL_SECONDS",
            _as_float(_profile_env_default("PMX_OPENCLAW_WHATSAPP_TURN_LOCK_POLL_SECONDS", 0.25), 0.25),
        ),
    )
)
PROGRESS_UPDATES_DEFAULT = _truthy_env(
    "PMX_QWEN_PROGRESS_UPDATES",
    _as_bool(_profile_env_default("PMX_QWEN_PROGRESS_UPDATES", True), True),
)
PROGRESS_MIN_INTERVAL_SECONDS = float(
    max(
        2.0,
        _float_env(
            "PMX_QWEN_PROGRESS_MIN_INTERVAL_SECONDS",
            _as_float(_profile_env_default("PMX_QWEN_PROGRESS_MIN_INTERVAL_SECONDS", 8.0), 8.0),
        ),
    )
)
PROGRESS_MAX_MESSAGE_CHARS = max(
    80,
    _int_env(
        "PMX_QWEN_PROGRESS_MAX_MESSAGE_CHARS",
        _as_int(_profile_env_default("PMX_QWEN_PROGRESS_MAX_MESSAGE_CHARS", 220), 220),
    ),
)

TRADING_OBJECTIVE_MIN_TRADES = max(
    10,
    _int_env("PMX_QWEN_OBJECTIVE_MIN_TRADES", _as_int(_profile_env_default("PMX_QWEN_OBJECTIVE_MIN_TRADES", 40), 40)),
)
TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE = min(
    0.49,
    max(
        0.01,
        _float_env(
            "PMX_QWEN_OBJECTIVE_TARGET_MAX_ERROR_RATE",
            _as_float(_profile_env_default("PMX_QWEN_OBJECTIVE_TARGET_MAX_ERROR_RATE", 0.42), 0.42),
        ),
    ),
)
TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE = min(
    0.99,
    max(
        0.5,
        _float_env(
            "PMX_QWEN_OBJECTIVE_MIN_WILSON_WIN_RATE",
            _as_float(_profile_env_default("PMX_QWEN_OBJECTIVE_MIN_WILSON_WIN_RATE", 0.55), 0.55),
        ),
    ),
)
TRADING_OBJECTIVE_PVALUE_MAX = min(
    0.5,
    max(
        0.0001,
        _float_env(
            "PMX_QWEN_OBJECTIVE_PVALUE_MAX",
            _as_float(_profile_env_default("PMX_QWEN_OBJECTIVE_PVALUE_MAX", 0.05), 0.05),
        ),
    ),
)
TRADING_OBJECTIVE_WILSON_Z = min(
    4.0,
    max(
        1.0,
        _float_env(
            "PMX_QWEN_OBJECTIVE_WILSON_Z",
            _as_float(_profile_env_default("PMX_QWEN_OBJECTIVE_WILSON_Z", 1.96), 1.96),
        ),
    ),
)
TRADING_CRITICAL_KEYWORDS = (
    "trade",
    "buy",
    "sell",
    "position",
    "portfolio",
    "pnl",
    "profit",
    "risk",
    "drawdown",
    "execution",
    "allocation",
    "rebalance",
)
TORCH_DEFAULT_VERSION = (os.getenv("PMX_TORCH_DEFAULT_VERSION") or "2.9.1").strip() or "2.9.1"
TORCH_DEFAULT_CUDA_VARIANT = (os.getenv("PMX_TORCH_DEFAULT_CUDA_VARIANT") or "cu128").strip().lower() or "cu128"

REPO_STATUS_SUBJECT_TOKENS = _csv_env_tokens(
    "PMX_REPO_STATUS_SUBJECT_TOKENS",
    (
        "github",
        "repository",
        "repo",
        "origin",
        "remote",
        "branch",
        "commit",
        "ahead",
        "behind",
        "pull request",
    ),
)
REPO_STATUS_ACTION_TOKENS = _csv_env_tokens(
    "PMX_REPO_STATUS_ACTION_TOKENS",
    (
        "check",
        "status",
        "verify",
        "health",
        "state",
        "sync",
        "latest",
    ),
)
STATUS_FAST_PATH_BLOCKED_TOKENS = _csv_env_tokens(
    "PMX_STATUS_FAST_PATH_BLOCKED_TOKENS",
    (
        "install ",
        "pip ",
        "torch",
        "reconcile",
        "gate",
        "audit",
        "pnl",
        "profit",
        "trade",
        "buy",
        "sell",
        "portfolio",
    ),
)
STATUS_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_STATUS_FAST_PATH_TRIGGER_TOKENS",
    (
        "health",
        "status",
        "system up",
        "gateway",
        "model availability",
        "models available",
    ),
)

SOURCE_INDEX_FILE = PROJECT_ROOT / "logs" / "llm_activity" / "source_index.json"
SOURCE_READ_ALLOWED_PREFIXES = _csv_env_tokens(
    "PMX_SOURCE_READ_ALLOWED_PREFIXES",
    (
        "ai_llm/",
        "config/",
        "etl/",
        "execution/",
        "forcester_ts/",
        "integrity/",
        "models/",
        "risk/",
        "scripts/",
        "tests/",
        "Documentation/",
    ),
)
SOURCE_READ_BLOCKED_TOKENS = _csv_env_tokens(
    "PMX_SOURCE_READ_BLOCKED_TOKENS",
    (
        ".env",
        "credentials.json",
        "auth-profiles.json",
        ".pem",
        ".key",
    ),
)
SOURCE_QUERY_STOPWORDS = _csv_env_tokens(
    "PMX_SOURCE_QUERY_STOPWORDS",
    (
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "code",
        "source",
        "file",
        "files",
        "project",
        "repo",
        "repository",
        "please",
        "show",
        "check",
    ),
)
SOURCE_READ_DEFAULT_MAX_FILES = max(1, _int_env("PMX_SOURCE_READ_DEFAULT_MAX_FILES", 4))
SOURCE_READ_DEFAULT_MAX_CHARS = max(1200, _int_env("PMX_SOURCE_READ_DEFAULT_MAX_CHARS", 12000))
SOURCE_READ_MAX_FILES_CAP = max(2, _int_env("PMX_SOURCE_READ_MAX_FILES_CAP", 12))
SOURCE_READ_MAX_CHARS_CAP = max(4000, _int_env("PMX_SOURCE_READ_MAX_CHARS_CAP", 30000))
SOURCE_READ_MAX_SNIPPETS_PER_FILE = max(1, _int_env("PMX_SOURCE_READ_MAX_SNIPPETS_PER_FILE", 6))
SOURCE_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_SOURCE_FAST_PATH_TRIGGER_TOKENS",
    (
        "source code",
        "inspect source",
        "read source",
        "self attribution",
        "self-attribution",
        "cite",
        "citation",
        "code evidence",
        "self-improvement",
    ),
)
REVIEW_CHANGED_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_REVIEW_CHANGED_FAST_PATH_TRIGGER_TOKENS",
    (
        "review changes",
        "review current changes",
        "review diff",
        "review my diff",
        "review changed files",
        "changed files",
        "worktree review",
        "working tree review",
        "code review",
    ),
)
REVIEW_CHANGED_FAST_PATH_BLOCKED_TOKENS = _csv_env_tokens(
    "PMX_REVIEW_CHANGED_FAST_PATH_BLOCKED_TOKENS",
    (
        "github status",
        "repo status",
        "web search",
        "gate failure",
        "pytest",
        "test failure",
    ),
)
TEST_FAILURE_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_TEST_FAILURE_FAST_PATH_TRIGGER_TOKENS",
    (
        "test failure",
        "tests failed",
        "failing test",
        "pytest failed",
        "pytest failure",
        "failed tests",
        "test log",
        "failure log",
    ),
)
GATE_TRIAGE_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_GATE_TRIAGE_FAST_PATH_TRIGGER_TOKENS",
    (
        "gate failure",
        "production gate failed",
        "why did the gate fail",
        "triage gate",
        "decompose gate",
        "gate decomposition",
        "explain gate failure",
    ),
)
WEB_SEARCH_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_WEB_SEARCH_FAST_PATH_TRIGGER_TOKENS",
    (
        "web search",
        "online search",
        "search the web",
        "search online",
        "search internet",
        "search on the internet",
        "internet search",
        "web lookup",
        "online lookup",
        "internet lookup",
        "look up online",
        "lookup online",
        "look up on the web",
        "lookup on the web",
        "find online",
        "find on the web",
        "browse the web",
        "browse internet",
        "research online",
        "check online",
        "google ",
        "duckduckgo",
        "wikipedia",
        "tavily",
    ),
)
WEB_SEARCH_FAST_PATH_BLOCKED_TOKENS = _csv_env_tokens(
    "PMX_WEB_SEARCH_FAST_PATH_BLOCKED_TOKENS",
    (
        "source code",
        "inspect source",
        "read source",
        "repository status",
        "repo status",
        "github repo status",
        "system health",
        "gateway status",
    ),
)
QUANT_VALIDATION_FAST_PATH_TRIGGER_TOKENS = _csv_env_tokens(
    "PMX_QUANT_VALIDATION_FAST_PATH_TRIGGER_TOKENS",
    (
        "quant validation",
        "quant health",
        "validation health",
        "fail rate",
        "headroom",
        "95% gate",
        "90% gate",
        "check quant",
        "quant gate",
    ),
)
QUANT_VALIDATION_FAST_PATH_BLOCKED_TOKENS = _csv_env_tokens(
    "PMX_QUANT_VALIDATION_FAST_PATH_BLOCKED_TOKENS",
    (
        "install ",
        "pip ",
        "source code",
        "repo status",
        "github",
        "web search",
        "search online",
    ),
)

_TOOL_CACHEABLE = {
    "fast_reasoning",
    "deep_reasoning",
    "read_gate_status",
    "query_trade_metrics",
    "check_quant_validation_headroom",
    "search_web_tavily",
    "inspect_source_code",
    "check_github_repo_status",
    "check_system_health",
    "run_sub_agent_batch",
    "triage_gate_failure",
}
_tool_cache_lock = threading.Lock()
_tool_cache: dict[str, tuple[float, str]] = {}
_tool_failure_lock = threading.Lock()
_tool_failure_memory: list[dict[str, Any]] = []
TOOL_FAILURE_MEMORY_MAX = max(4, _int_env("PMX_QWEN_TOOL_FAILURE_MEMORY_MAX", 16))
TOOL_FAILURE_HINTS_MAX = max(1, _int_env("PMX_QWEN_TOOL_FAILURE_HINTS_MAX", 4))
TOOL_REPEAT_SIGNATURE_LIMIT = max(1, _int_env("PMX_QWEN_TOOL_REPEAT_SIGNATURE_LIMIT", 2))
TOOL_ERROR_ROUND_STREAK_LIMIT = max(1, _int_env("PMX_QWEN_TOOL_ERROR_ROUND_STREAK_LIMIT", 2))
_tool_failure_memory_path_raw = (
    os.getenv("PMX_QWEN_TOOL_FAILURE_MEMORY_PATH", "logs/automation/qwen_tool_failures.jsonl").strip()
)
TOOL_FAILURE_MEMORY_PATH = Path(_tool_failure_memory_path_raw)
if not TOOL_FAILURE_MEMORY_PATH.is_absolute():
    TOOL_FAILURE_MEMORY_PATH = (PROJECT_ROOT / TOOL_FAILURE_MEMORY_PATH).resolve()
TOOL_FAILURE_FILE_MAX_LINES = max(
    TOOL_FAILURE_MEMORY_MAX,
    _int_env("PMX_QWEN_TOOL_FAILURE_FILE_MAX_LINES", 2048),
)
_tool_failure_loaded = False

_TOOL_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "fast_reasoning": ("task",),
    "deep_reasoning": ("task",),
    "read_gate_status": ("gate",),
    "query_trade_metrics": ("metric",),
    "check_quant_validation_headroom": (),
    "search_web_tavily": ("query",),
    "inspect_source_code": (),
    "check_github_repo_status": (),
    "send_notification": ("channel", "message"),
    "run_sub_agent_batch": ("tasks",),
    "check_system_health": (),
    "run_production_audit_gate": (),
    "review_changed_files": (),
    "summarize_test_failure": (),
    "triage_gate_failure": (),
    "install_torch_runtime": (),
    "install_python_package": ("package",),
}

_TOOL_NAME_ALIASES: dict[str, str] = {
    "web_search": "search_web_tavily",
    "search_web": "search_web_tavily",
    "brave_web_search": "search_web_tavily",
    "source_read": "inspect_source_code",
    "read_source_code": "inspect_source_code",
    "code_search": "inspect_source_code",
    "repo_status": "check_github_repo_status",
    "github_repo_status": "check_github_repo_status",
    "check_repo_status": "check_github_repo_status",
    "review_diff": "review_changed_files",
    "review_changes": "review_changed_files",
    "review_pr": "review_changed_files",
    "check_quant_health": "check_quant_validation_headroom",
    "quant_validation_health": "check_quant_validation_headroom",
    "quant_health": "check_quant_validation_headroom",
    "test_failure_summary": "summarize_test_failure",
    "gate_failure_triage": "triage_gate_failure",
    "gate_decomposition": "triage_gate_failure",
}


def _canonical_tool_name(tool_name: str) -> str:
    raw = str(tool_name or "").strip()
    if not raw:
        return ""
    return _TOOL_NAME_ALIASES.get(raw.lower(), raw)


def _cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        normalized = json.dumps(arguments, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        normalized = str(arguments)
    return f"{tool_name}|{normalized}"


def _cache_get(key: str) -> Optional[str]:
    now = time.monotonic()
    with _tool_cache_lock:
        row = _tool_cache.get(key)
        if not row:
            return None
        expires_at, value = row
        if expires_at < now:
            _tool_cache.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: str, ttl_seconds: float = TOOL_CACHE_TTL_SECONDS) -> None:
    now = time.monotonic()
    with _tool_cache_lock:
        _tool_cache[key] = (now + max(1.0, float(ttl_seconds)), value)
        if len(_tool_cache) <= TOOL_CACHE_MAX_ENTRIES:
            return
        # Evict oldest-expiring entries first.
        stale_keys = sorted(_tool_cache.items(), key=lambda kv: kv[1][0])[: max(1, len(_tool_cache) - TOOL_CACHE_MAX_ENTRIES)]
        for k, _ in stale_keys:
            _tool_cache.pop(k, None)


def _record_tool_failure(tool_name: str, arguments: dict[str, Any], error: str) -> None:
    _load_tool_failure_memory_once()
    item = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "tool": str(tool_name or "").strip() or "unknown",
        "arguments": arguments if isinstance(arguments, dict) else {},
        "error": str(error or "").strip()[:400],
    }
    with _tool_failure_lock:
        _tool_failure_memory.append(item)
        if len(_tool_failure_memory) > TOOL_FAILURE_MEMORY_MAX:
            del _tool_failure_memory[: len(_tool_failure_memory) - TOOL_FAILURE_MEMORY_MAX]
    _append_tool_failure_item(item)


def _recent_tool_failure_hints(limit: int = 3) -> str:
    _load_tool_failure_memory_once()
    with _tool_failure_lock:
        rows = list(_tool_failure_memory[-max(0, int(limit)):])
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        tool = row.get("tool", "unknown")
        err = row.get("error", "")
        lines.append(f"- {tool}: {err}")
    return "\n".join(lines)


def _sanitize_failure_item(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    tool = str(raw.get("tool") or "").strip() or "unknown"
    err = str(raw.get("error") or "").strip()
    if not err:
        return None
    ts_utc = str(raw.get("ts_utc") or "").strip() or datetime.now(timezone.utc).isoformat()
    args_raw = raw.get("arguments")
    args = args_raw if isinstance(args_raw, dict) else {}
    return {
        "ts_utc": ts_utc[:64],
        "tool": tool[:120],
        "arguments": args,
        "error": err[:400],
    }


def _load_tool_failure_memory_once() -> None:
    global _tool_failure_loaded
    if _tool_failure_loaded:
        return
    with _tool_failure_lock:
        if _tool_failure_loaded:
            return
        _tool_failure_loaded = True

    path = TOOL_FAILURE_MEMORY_PATH
    if not path.exists():
        return

    rows: deque[dict[str, Any]] = deque(maxlen=max(TOOL_FAILURE_MEMORY_MAX, TOOL_FAILURE_FILE_MAX_LINES))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                raw = (line or "").strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except Exception:
                    continue
                item = _sanitize_failure_item(parsed)
                if item:
                    rows.append(item)
    except Exception as exc:
        logger.warning("Failed to load tool failure memory from %s: %s", path, exc)
        return

    with _tool_failure_lock:
        _tool_failure_memory.clear()
        _tool_failure_memory.extend(list(rows)[-TOOL_FAILURE_MEMORY_MAX:])


def _append_tool_failure_item(item: dict[str, Any]) -> None:
    path = TOOL_FAILURE_MEMORY_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("Failed to append tool failure memory to %s: %s", path, exc)
        return

    _compact_tool_failure_file_if_needed(path)


def _compact_tool_failure_file_if_needed(path: Path) -> None:
    try:
        stat = path.stat()
    except Exception:
        return
    if int(stat.st_size) <= 2 * 1024 * 1024:
        return

    rows: deque[str] = deque(maxlen=TOOL_FAILURE_FILE_MAX_LINES)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = (line or "").strip()
                if text:
                    rows.append(text)
    except Exception:
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(row + "\n")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _tool_error_json(
    tool_name: str,
    error: str,
    *,
    arguments: Optional[dict[str, Any]] = None,
    include_available: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "error": str(error or "tool execution failed"),
        "tool": str(tool_name or "unknown"),
    }
    if isinstance(arguments, dict):
        payload["arguments"] = arguments
    if include_available:
        payload["available_tools"] = sorted(_TOOL_REQUIRED_FIELDS.keys())
    return json.dumps(payload)


def _tool_error_message(tool_result: str) -> str:
    text = (tool_result or "").strip()
    if not text:
        return "empty tool response"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if err:
                return str(err)
            status = str(parsed.get("status", "")).strip().upper()
            if status == "FAIL":
                return "tool reported status=FAIL"
    except Exception:
        pass
    if text.startswith("[ERROR]"):
        return text[:200]
    return ""


def _json_object_or_none(raw_text: str) -> Optional[dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _maybe_float(value: Any, *, allow_inf: bool = False) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed):
        return None
    if math.isinf(parsed) and not allow_inf:
        return None
    return parsed


def _maybe_ratio(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    scale = 1.0
    if text.endswith("%"):
        text = text[:-1].strip()
        scale = 0.01
    parsed = _maybe_float(text)
    if parsed is None:
        return None
    return parsed * scale


def _tool_status_json(
    tool_name: str,
    *,
    status: str,
    message: str,
    action: str = "",
    validation_code: str = "",
    details: Optional[dict[str, Any]] = None,
) -> str:
    payload: dict[str, Any] = {
        "status": str(status or "FAIL").upper(),
        "tool": str(tool_name or "unknown"),
        "message": str(message or "").strip() or "tool status unavailable",
    }
    if action:
        payload["action"] = action
    if validation_code:
        payload["validation_code"] = validation_code
    if isinstance(details, dict) and details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=True)


def _validate_query_trade_metrics_result(arguments: dict[str, Any], raw_result: str) -> str:
    metric = str(arguments.get("metric") or "summary").strip().lower() or "summary"
    parsed = _json_object_or_none(raw_result)
    if parsed is None or metric not in {"summary", "objective"}:
        return raw_result

    if metric == "summary":
        total_trades = _maybe_int(parsed.get("total_trades"))
        wins = _maybe_int(parsed.get("wins"))
        losses = _maybe_int(parsed.get("losses"))
        win_rate = _maybe_ratio(parsed.get("win_rate"))
        profit_factor = _maybe_float(parsed.get("profit_factor"), allow_inf=True)
        if total_trades is None or wins is None or losses is None:
            return _tool_status_json(
                "query_trade_metrics",
                status="FAIL",
                action="query_trade_metrics",
                validation_code="trade_metrics_invalid_payload",
                message="Trade metrics summary is missing required count fields.",
                details={"metric": metric},
            )
        if (
            total_trades < 0
            or wins < 0
            or losses < 0
            or wins + losses > total_trades
            or win_rate is None
            or win_rate < 0.0
            or win_rate > 1.0
            or profit_factor is None
            or profit_factor < 0.0
        ):
            return _tool_status_json(
                "query_trade_metrics",
                status="FAIL",
                action="query_trade_metrics",
                validation_code="trade_metrics_out_of_range",
                message="Trade metrics summary contains out-of-range or internally inconsistent values.",
                details={
                    "metric": metric,
                    "total_trades": total_trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": parsed.get("win_rate"),
                    "profit_factor": parsed.get("profit_factor"),
                },
            )
        updated = dict(parsed)
        if total_trades == 0:
            updated["status"] = "LIMITED"
            updated["message"] = (
                "No production round-trip trades were found; realized trade metrics are unavailable."
            )
            updated["no_data_reason"] = "no_round_trip_data"
        else:
            updated["status"] = "PASS"
        return json.dumps(updated, ensure_ascii=True)

    sample_size = _maybe_int(parsed.get("sample_size"))
    objective_value = parsed.get("objective_value")
    objective_float = None if objective_value in {None, ""} else _maybe_float(objective_value)
    observed = parsed.get("observed") if isinstance(parsed.get("observed"), dict) else {}
    bounds = parsed.get("confidence_bounds") if isinstance(parsed.get("confidence_bounds"), dict) else {}
    wins = _maybe_int(observed.get("wins"))
    losses = _maybe_int(observed.get("losses"))
    observed_win_rate = _maybe_float(observed.get("win_rate"))
    observed_error_rate = _maybe_float(observed.get("error_rate"))
    wilson = _maybe_float(bounds.get("wilson_win_rate_lower"))
    conservative = _maybe_float(bounds.get("conservative_error_rate_upper"))
    p_value = _maybe_float(bounds.get("p_value"))
    if sample_size is None or sample_size < 0:
        return _tool_status_json(
            "query_trade_metrics",
            status="FAIL",
            action="query_trade_metrics",
            validation_code="trade_objective_invalid_payload",
            message="Trade objective payload is missing a valid sample size.",
            details={"metric": metric, "sample_size": parsed.get("sample_size")},
        )
    if sample_size == 0:
        updated = dict(parsed)
        updated["status"] = str(updated.get("status") or "LIMITED").upper()
        updated.setdefault(
            "message",
            "No production round-trip trades were found; the trading objective cannot be evaluated yet.",
        )
        updated["no_data_reason"] = "no_round_trip_data"
        return json.dumps(updated, ensure_ascii=True)
    if (
        wins is None
        or losses is None
        or wins < 0
        or losses < 0
        or wins + losses > sample_size
        or observed_win_rate is None
        or observed_win_rate < 0.0
        or observed_win_rate > 1.0
        or observed_error_rate is None
        or observed_error_rate < 0.0
        or observed_error_rate > 1.0
        or wilson is None
        or wilson < 0.0
        or wilson > 1.0
        or conservative is None
        or conservative < 0.0
        or conservative > 1.0
        or p_value is None
        or p_value < 0.0
        or p_value > 1.0
        or (objective_value not in {None, ""} and objective_float is None)
    ):
        return _tool_status_json(
            "query_trade_metrics",
            status="FAIL",
            action="query_trade_metrics",
            validation_code="trade_objective_out_of_range",
            message="Trade objective payload contains out-of-range or internally inconsistent values.",
            details={"metric": metric, "sample_size": sample_size},
        )
    updated = dict(parsed)
    updated["status"] = str(updated.get("status") or "PASS").upper()
    return json.dumps(updated, ensure_ascii=True)


def _validate_quant_validation_headroom_result(arguments: dict[str, Any], raw_result: str) -> str:
    del arguments
    parsed = _json_object_or_none(raw_result)
    if parsed is None:
        return raw_result
    status = str(parsed.get("status") or "").upper()
    if status == "FAIL" and str(parsed.get("error") or "").strip():
        return raw_result
    summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
    if not summary:
        return raw_result

    total = _maybe_int(summary.get("total"))
    fail_count = _maybe_int(summary.get("fail_count"))
    fail_rate_pct = _maybe_float(summary.get("fail_rate_pct"))
    red_gate_pct = _maybe_float(summary.get("red_gate_pct"))
    warn_gate_pct = _maybe_float(summary.get("warn_gate_pct"))
    headroom_pct = _maybe_float(summary.get("headroom_to_red_gate_pct"))
    if (
        total is None
        or fail_count is None
        or total < 0
        or fail_count < 0
        or fail_count > total
        or fail_rate_pct is None
        or fail_rate_pct < 0.0
        or fail_rate_pct > 100.0
        or red_gate_pct is None
        or red_gate_pct <= 0.0
        or red_gate_pct > 100.0
        or warn_gate_pct is None
        or warn_gate_pct < 0.0
        or warn_gate_pct > red_gate_pct
        or headroom_pct is None
        or headroom_pct < -100.0
        or headroom_pct > 100.0
    ):
        return _tool_status_json(
            "check_quant_validation_headroom",
            status="FAIL",
            action="check_quant_validation_headroom",
            validation_code="quant_validation_out_of_range",
            message="Quant-validation summary contains out-of-range or internally inconsistent values.",
            details={"summary": summary},
        )
    if total == 0:
        updated = dict(parsed)
        updated["status"] = "NO_DATA"
        updated["message"] = "Quant-validation summary returned no usable entries."
        updated["no_data_reason"] = "no_quant_validation_rows"
        return json.dumps(updated, ensure_ascii=True)

    per_ticker = summary.get("per_ticker") if isinstance(summary.get("per_ticker"), list) else []
    for row in per_ticker:
        if not isinstance(row, dict):
            return _tool_status_json(
                "check_quant_validation_headroom",
                status="FAIL",
                action="check_quant_validation_headroom",
                validation_code="quant_validation_invalid_payload",
                message="Quant-validation per-ticker rows must be structured objects.",
            )
        row_total = _maybe_int(row.get("total"))
        row_fail = _maybe_int(row.get("fail_count"))
        row_rate = _maybe_float(row.get("fail_rate_pct"))
        if (
            row_total is None
            or row_fail is None
            or row_total < 0
            or row_fail < 0
            or row_fail > row_total
            or row_rate is None
            or row_rate < 0.0
            or row_rate > 100.0
        ):
            return _tool_status_json(
                "check_quant_validation_headroom",
                status="FAIL",
                action="check_quant_validation_headroom",
                validation_code="quant_validation_out_of_range",
                message="Quant-validation per-ticker rows contain out-of-range or inconsistent values.",
                details={"row": row},
            )
    return raw_result


def _validate_search_web_tavily_result(arguments: dict[str, Any], raw_result: str) -> str:
    del arguments
    parsed = _json_object_or_none(raw_result)
    if parsed is None:
        return raw_result
    latency_seconds = parsed.get("latency_seconds")
    if latency_seconds not in {None, ""}:
        latency = _maybe_float(latency_seconds)
        if latency is None or latency < 0.0 or latency > 600.0:
            return _tool_status_json(
                "search_web_tavily",
                status="FAIL",
                action="search_web_tavily",
                validation_code="search_latency_out_of_range",
                message="Search result latency was out of expected range.",
                details={"latency_seconds": latency_seconds},
            )

    status = str(parsed.get("status") or "").upper()
    answer = str(parsed.get("answer") or "").strip()
    results = parsed.get("results") if isinstance(parsed.get("results"), list) else []
    if status == "PASS" and not answer and not results:
        updated = dict(parsed)
        updated["status"] = "NO_DATA"
        updated["message"] = "Search completed but returned no answer or search results."
        updated["no_data_reason"] = "empty_search_result"
        return json.dumps(updated, ensure_ascii=True)
    return raw_result


def _validate_tool_result(tool_name: str, arguments: dict[str, Any], raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return _tool_status_json(
            tool_name,
            status="FAIL",
            action=tool_name,
            validation_code="empty_tool_result",
            message="Tool returned no data or empty output.",
        )

    validators: dict[str, Callable[[dict[str, Any], str], str]] = {
        "query_trade_metrics": _validate_query_trade_metrics_result,
        "check_quant_validation_headroom": _validate_quant_validation_headroom_result,
        "search_web_tavily": _validate_search_web_tavily_result,
    }
    validator = validators.get(str(tool_name or "").strip())
    if validator is None:
        return raw_result
    try:
        validated = validator(arguments, raw_result)
    except Exception as exc:
        return _tool_status_json(
            tool_name,
            status="FAIL",
            action=tool_name,
            validation_code="tool_result_validation_error",
            message=f"Tool result validation failed: {exc}",
        )
    return str(validated or "").strip() or text


def _tool_result_guidance_message(tool_result: str) -> str:
    parsed = _json_object_or_none(tool_result)
    if not parsed:
        return ""
    status = str(parsed.get("status") or "").upper()
    validation_code = str(parsed.get("validation_code") or "").strip().lower()
    if status in {"NO_DATA", "LIMITED"}:
        return (
            "Previous tool call returned limited or missing data. Do not infer unavailable values, "
            "do not invent numbers, and state the limitation explicitly in the final answer."
        )
    if validation_code.endswith("out_of_range") or validation_code.endswith("invalid_payload"):
        return (
            "Previous tool call detected out-of-range or internally inconsistent data. Treat that source "
            "as invalid, avoid numerical conclusions from it, and explain that the data needs repair or regeneration."
        )
    return ""


def _sanitize_tool_arguments(tool_name: str, raw_arguments: Any) -> tuple[dict[str, Any], Optional[str]]:
    canonical_tool_name = _canonical_tool_name(tool_name)
    args: dict[str, Any]
    if isinstance(raw_arguments, dict):
        args = dict(raw_arguments)
    elif isinstance(raw_arguments, str):
        raw = raw_arguments.strip()
        if not raw:
            args = {}
        else:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    args = {"task": raw}
            except Exception:
                args = {"task": raw}
    else:
        args = {}

    if canonical_tool_name in {"read", "read_file"}:
        path = str(args.get("path") or "").strip()
        if not path:
            return args, "Missing required field: path"
        return args, "Unsupported tool name in this orchestrator. Use read_gate_status instead."

    required = _TOOL_REQUIRED_FIELDS.get(canonical_tool_name)
    if required is None:
        unknown = canonical_tool_name or str(tool_name or "").strip() or "unknown"
        return args, f"Unknown tool: {unknown}"
    missing = [field for field in required if not str(args.get(field) or "").strip()]
    if missing:
        return args, f"Missing required field(s): {', '.join(missing)}"
    return args, None


def _bounded_timeout(default_seconds: float, budget_seconds: Optional[float]) -> float:
    default_val = max(6.0, float(default_seconds))
    if budget_seconds is None:
        return default_val
    try:
        budget_val = float(budget_seconds)
    except Exception:
        return default_val
    return max(6.0, min(default_val, max(6.0, budget_val - 1.0)))


def _ollama_get(path: str, timeout: float = 5.0) -> Any:
    url = f"{OLLAMA_BASE.rstrip('/')}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _ollama_post(path: str, payload: dict, timeout: float = 120.0) -> Any:
    url = f"{OLLAMA_BASE.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def discover_models() -> list[str]:
    """Discover models available in local Ollama instance."""
    try:
        tags = _ollama_get("/api/tags")
        return [m["name"] for m in tags.get("models", []) if isinstance(m, dict)]
    except (OSError, urllib.error.URLError, ValueError):
        return []


def check_model_available(model: str) -> bool:
    return model in discover_models()


def refresh_health() -> dict[str, bool]:
    """Check all registered models and update availability status."""
    now = time.monotonic()
    available_models = discover_models()

    with _health_lock:
        results = {}
        for name, spec in MODEL_REGISTRY.items():
            was_available = spec.available
            spec.available = name in available_models
            spec.last_health_check = now
            results[name] = spec.available
            if was_available and not spec.available:
                logger.warning("Model %s became UNAVAILABLE", name)
            elif not was_available and spec.available:
                logger.info("Model %s is now AVAILABLE", name)

    return results


def _ensure_health_fresh() -> None:
    """Refresh health if stale (older than _HEALTH_CHECK_INTERVAL)."""
    now = time.monotonic()
    oldest = min(
        (s.last_health_check for s in MODEL_REGISTRY.values()),
        default=0.0,
    )
    if now - oldest > _HEALTH_CHECK_INTERVAL:
        refresh_health()


def get_best_model_for_role(role: str) -> Optional[str]:
    """Get the best available model for a given role, with fallback."""
    _ensure_health_fresh()

    # Primary: exact role match
    for name, spec in MODEL_REGISTRY.items():
        if spec.role == role and spec.available and not _model_in_cooldown(spec):
            return name

    # Fallback: any available model (prefer fast over slow)
    speed_order = {"fast": 0, "medium": 1, "slow": 2}
    candidates = [
        (name, spec) for name, spec in MODEL_REGISTRY.items()
        if spec.available and not _model_in_cooldown(spec)
    ]
    candidates.sort(key=lambda x: speed_order.get(x[1].speed, 99))
    return candidates[0][0] if candidates else None


def _record_model_stats(model: str, latency_ms: float, success: bool) -> None:
    """Update running statistics for a model."""
    spec = MODEL_REGISTRY.get(model)
    if not spec:
        return
    with _health_lock:
        if success:
            spec.success_count += 1
            spec.error_streak = 0
            # Exponential moving average of latency
            alpha = 0.3
            spec.avg_latency_ms = (alpha * latency_ms) + ((1 - alpha) * spec.avg_latency_ms) if spec.avg_latency_ms > 0 else latency_ms
        else:
            spec.error_count += 1
            spec.error_streak += 1
            if spec.error_streak >= MODEL_ERROR_STREAK_THRESHOLD:
                spec.cooldown_until = time.monotonic() + MODEL_COOLDOWN_SECONDS
                spec.error_streak = 0


def _model_in_cooldown(spec: Optional[ModelSpec]) -> bool:
    if spec is None:
        return False
    return float(spec.cooldown_until or 0.0) > time.monotonic()


def _system_degraded() -> bool:
    # Conservative degrade condition: too many errors/cooldowns or high heavy-model latency.
    with _health_lock:
        cooldowns = sum(1 for s in MODEL_REGISTRY.values() if _model_in_cooldown(s))
        heavy = MODEL_REGISTRY.get("deepseek-r1:32b")
        heavy_slow = bool(heavy and heavy.avg_latency_ms > SUBAGENT_DEEP_LATENCY_CUTOFF_MS)
        recent_errors = sum(1 for s in MODEL_REGISTRY.values() if s.error_count > 0 and s.success_count == 0)
    return cooldowns > 0 or heavy_slow or recent_errors >= 2


@dataclass
class _BridgeTurnLock:
    path: Path
    token: str
    waited_seconds: float
    payload: dict[str, Any] = field(default_factory=dict)


def _bridge_turn_lock_path(channel: Optional[str]) -> Path:
    channel_text = re.sub(r"[^a-z0-9_-]+", "_", str(channel or "unknown").strip().lower()) or "unknown"
    return PROJECT_ROOT / "logs" / "automation" / f"openclaw_{channel_text}_bridge_turn.lock"


def _bridge_requires_forced_orchestration(channel: Optional[str]) -> bool:
    return str(channel or "").strip().lower() == "whatsapp"


def _read_lock_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _acquire_bridge_turn_lock(
    *,
    channel: Optional[str],
    session_id: Optional[str],
    wait_seconds: float,
    stale_seconds: float,
    poll_seconds: float,
) -> Optional[_BridgeTurnLock]:
    lock_path = _bridge_turn_lock_path(channel)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    started = time.monotonic()

    while True:
        token = f"{os.getpid()}-{int(time.time() * 1000)}"
        payload = {
            "pid": int(os.getpid()),
            "channel": str(channel or "").strip().lower() or "unknown",
            "session_id": str(session_id or "").strip() or None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "token": token,
            "command": " ".join(sys.argv),
        }
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
            finally:
                os.close(fd)
            waited = max(0.0, time.monotonic() - started)
            return _BridgeTurnLock(path=lock_path, token=token, waited_seconds=waited, payload=payload)
        except FileExistsError:
            try:
                age_seconds = max(0.0, time.time() - float(lock_path.stat().st_mtime))
            except Exception:
                age_seconds = 0.0
            if age_seconds >= max(60.0, float(stale_seconds)):
                try:
                    lock_path.unlink(missing_ok=True)
                    continue
                except Exception:
                    pass
            if time.monotonic() >= deadline:
                return None
            time.sleep(max(0.01, float(poll_seconds)))
        except Exception:
            return None


def _release_bridge_turn_lock(lock: Optional[_BridgeTurnLock]) -> None:
    if lock is None:
        return
    try:
        holder = _read_lock_payload(lock.path)
        if str(holder.get("token") or "") != str(lock.token):
            return
        lock.path.unlink(missing_ok=True)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Tool definitions (for qwen3 orchestrator)
# ---------------------------------------------------------------------------

REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fast_reasoning",
            "description": "Use deepseek-r1:8b for fast chain-of-thought reasoning on quantitative/financial analysis tasks. Use for market analysis, signal validation, regime detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The reasoning task to perform"},
                    "context": {"type": "string", "description": "Relevant data/context for the reasoning task"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_reasoning",
            "description": "Use deepseek-r1:32b for deep multi-step reasoning on complex analysis. Use for portfolio optimization, adversarial audits, complex forecasting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The complex reasoning task to perform"},
                    "context": {"type": "string", "description": "Relevant data/context for the reasoning task"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_gate_status",
            "description": "Read the latest production audit gate results from logs/audit_sprint/",
            "parameters": {
                "type": "object",
                "properties": {
                    "gate": {
                        "type": "string",
                        "enum": ["gate_1_forecaster_audit", "gate_2_quant_health", "gate_3_quant_health", "all"],
                        "description": "Which gate log to read",
                    },
                },
                "required": ["gate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_trade_metrics",
            "description": "Query PnL integrity metrics and trading-error objective stats from the trading database",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["canonical_pnl", "win_rate", "profit_factor", "round_trips", "summary", "objective"],
                        "description": "Which metric to retrieve",
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_quant_validation_headroom",
            "description": (
                "Summarize recent quant-validation fail rate and headroom to gate thresholds "
                "from logs/signals/quant_validation.jsonl. Use this instead of inline python -c."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {
                        "type": "integer",
                        "description": "Number of most recent entries to analyze (default 120).",
                    },
                    "red_gate_pct": {
                        "type": "number",
                        "description": "RED threshold in percent (default 95).",
                    },
                    "warn_gate_pct": {
                        "type": "number",
                        "description": "YELLOW threshold in percent (default 90).",
                    },
                    "log_path": {
                        "type": "string",
                        "description": "Optional quant_validation.jsonl path override.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web_tavily",
            "description": (
                "Search the web robustly using Tavily first with free-provider fallback "
                "(DuckDuckGo, Wikipedia) for current external information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (1-10, default 5).",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced", "fast", "ultra-fast"],
                        "description": "Search depth.",
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "description": "Optional topic (for example: general, news).",
                    },
                    "include_domains": {
                        "type": "string",
                        "description": "Optional comma-separated domain allowlist.",
                    },
                    "exclude_domains": {
                        "type": "string",
                        "description": "Optional comma-separated domain blocklist.",
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Optional Tavily base URL override.",
                    },
                    "providers": {
                        "type": "string",
                        "description": (
                            "Optional provider order as CSV. "
                            "Allowed: tavily,duckduckgo,wikipedia."
                        ),
                    },
                    "allow_fallback": {
                        "type": "boolean",
                        "description": "When false, force Tavily-only lookup.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_source_code",
            "description": (
                "Read PMX source files with safe allowlist constraints and return attributable "
                "snippets with file:line citations for evidence-backed reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language intent to search relevant source files.",
                    },
                    "paths": {
                        "type": "array",
                        "description": "Optional explicit relative file paths to inspect first.",
                        "items": {"type": "string"},
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum files to inspect (bounded).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum aggregate characters returned (bounded).",
                    },
                    "refresh_index": {
                        "type": "boolean",
                        "description": "When true, rebuild source index before reading.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_github_repo_status",
            "description": (
                "Check local repository state versus GitHub remote for the current workspace. "
                "Use for requests about branch/head/dirty state/upstream ahead-behind and repo sync."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_changed_files",
            "description": (
                "Review the current git worktree or a selected path set for code-review context. "
                "Returns changed files, diff preview, and risk tags without modifying anything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "description": "Optional relative paths to narrow the review scope.",
                        "items": {"type": "string"},
                    },
                    "base_ref": {
                        "type": "string",
                        "description": "Optional git ref to diff against (default current HEAD).",
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum changed files to include.",
                    },
                    "max_diff_chars": {
                        "type": "integer",
                        "description": "Maximum diff preview characters to return.",
                    },
                    "diff_context": {
                        "type": "integer",
                        "description": "Unified diff context lines per hunk.",
                    },
                    "include_untracked": {
                        "type": "boolean",
                        "description": "Include previews for untracked text files when available.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_test_failure",
            "description": (
                "Summarize pytest failures from inline output text or a log file path. "
                "Use for operator triage and reviewer explanations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Optional path to a pytest log/output file.",
                    },
                    "output_text": {
                        "type": "string",
                        "description": "Optional inline pytest output text to summarize.",
                    },
                    "max_failures": {
                        "type": "integer",
                        "description": "Maximum failure records to include.",
                    },
                    "max_suspect_paths": {
                        "type": "integer",
                        "description": "Maximum traceback file paths to include.",
                    },
                    "max_error_types": {
                        "type": "integer",
                        "description": "Maximum distinct error types to include.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "triage_gate_failure",
            "description": (
                "Decompose the latest production gate artifact into blocker components and "
                "reason-code breakdown for operator action planning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_artifact": {
                        "type": "string",
                        "description": "Optional production gate artifact path override.",
                    },
                    "summary_cache": {
                        "type": "string",
                        "description": "Optional forecast-audit summary cache path override.",
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Optional output path for decomposition JSON.",
                    },
                    "output_md_path": {
                        "type": "string",
                        "description": "Optional output path for decomposition markdown.",
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": "Force regeneration even when the decomposition report looks fresh.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_production_audit_gate",
            "description": (
                "Safely run scripts/production_audit_gate.py without shell chaining. "
                "Use this when the user asks to run/reconcile production gates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reconcile_close_ids": {
                        "type": "array",
                        "description": "Optional close IDs for pre-gate reconciliation",
                        "items": {"type": "integer"},
                    },
                    "reconcile_apply": {
                        "type": "boolean",
                        "description": "Apply reconciliation changes; default false (dry-run).",
                    },
                    "require_holding_period": {"type": "boolean"},
                    "allow_inconclusive_lift": {"type": "boolean"},
                    "require_profitable": {"type": "boolean"},
                    "max_files": {"type": "integer"},
                    "notify_openclaw": {"type": "boolean"},
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Subprocess timeout for the gate command.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_torch_runtime",
            "description": (
                "Install or verify torch for the current Python runtime with structured reporting. "
                "Use this instead of generic pip shell commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {
                        "type": "string",
                        "description": f"Optional torch version (default: {TORCH_DEFAULT_VERSION})",
                    },
                     "variant": {
                         "type": "string",
                         "enum": ["auto", "default", "cpu", "cu118", "cu121", "cu124", "cu128"],
                         "description": (
                             "Install source variant. "
                             "auto selects a CUDA build (default cu128) when an NVIDIA GPU is detected, else default. "
                             "default installs from PyPI, cpu forces the CPU-only index, cu* forces a specific CUDA index."
                         ),
                     },
                    "scope": {
                        "type": "string",
                        "enum": ["auto", "system", "user"],
                        "description": "Install scope. auto uses venv/system first and falls back to --user on permission errors.",
                    },
                     "upgrade": {
                         "type": "boolean",
                         "description": "When true, pass --upgrade to pip install.",
                     },
                     "force": {
                         "type": "boolean",
                         "description": "When true, attempt installation even if torch is already importable (default false).",
                     },
                     "verify_only": {
                         "type": "boolean",
                         "description": "Only verify torch import/version without installing.",
                     },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Install timeout budget per attempt.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_python_package",
            "description": (
                "Install or verify a Python package for the current runtime with structured reporting. "
                "Use this for non-torch dependencies instead of generic pip shell commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package specifier (e.g. pandas, pydantic, uvicorn[standard]).",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional version pin when package has no comparator.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["auto", "system", "user"],
                        "description": "Install scope. auto uses venv/system first and falls back to --user on permission errors.",
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "When true, pass --upgrade to pip install.",
                    },
                    "pre_release": {
                        "type": "boolean",
                        "description": "When true, pass --pre to pip install.",
                    },
                    "verify_import": {
                        "type": "string",
                        "description": "Optional module name to import for verification (defaults to package-based guess).",
                    },
                    "verify_only": {
                        "type": "boolean",
                        "description": "Only verify import/version without installing.",
                    },
                    "index_url": {
                        "type": "string",
                        "description": "Optional custom pip index-url (https://..., allowed host only).",
                    },
                    "extra_index_url": {
                        "type": "string",
                        "description": "Optional extra pip index-url (https://..., allowed host only).",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Install timeout budget per attempt.",
                    },
                },
                "required": ["package"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a message via OpenClaw to a channel (whatsapp, telegram, discord). Use for delivering results, alerts, or summaries to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["whatsapp", "telegram", "discord"], "description": "Channel to send on"},
                    "message": {"type": "string", "description": "Message text to send"},
                },
                "required": ["channel", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sub_agent_batch",
            "description": (
                "Run a batch of sub-agent tasks using deepseek-r1 models. "
                "Use this to decompose complex user goals into smaller reasoning jobs "
                "and aggregate their outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "List of sub-tasks to run",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "context": {"type": "string"},
                                "complexity": {"type": "string", "enum": ["auto", "fast", "deep"]},
                            },
                            "required": ["task"],
                        },
                    },
                    "default_context": {"type": "string"},
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_system_health",
            "description": "Check the health of all LLM models and the OpenClaw gateway",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allow_cache: bool = True,
    budget_seconds: Optional[float] = None,
) -> str:
    """Execute a tool call requested by the orchestrator model."""
    safe_tool_name = _canonical_tool_name(tool_name)
    args, arg_error = _sanitize_tool_arguments(safe_tool_name, arguments)
    if arg_error:
        _record_tool_failure(safe_tool_name, args, arg_error)
        return _tool_error_json(
            safe_tool_name,
            arg_error,
            arguments=args,
            include_available=True,
        )

    cacheable = allow_cache and safe_tool_name in _TOOL_CACHEABLE
    cache_key = _cache_key(safe_tool_name, args) if cacheable else ""
    if cacheable:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    def _finish(result: str, *, ttl_seconds: float = TOOL_CACHE_TTL_SECONDS) -> str:
        validated = _validate_tool_result(safe_tool_name, args, result)
        if cacheable:
            _cache_set(cache_key, validated, ttl_seconds=ttl_seconds)
        return validated

    try:
        if safe_tool_name == "fast_reasoning":
            model = get_best_model_for_role("reasoning") or "deepseek-r1:8b"
            result = _run_reasoning_model(
                model=model,
                task=args.get("task", ""),
                context=args.get("context", ""),
                max_predict=360,
                timeout_seconds=_bounded_timeout(80.0, budget_seconds),
            )
            return _finish(result, ttl_seconds=30.0)

        if safe_tool_name == "deep_reasoning":
            model = get_best_model_for_role("heavy_reasoning") or "deepseek-r1:32b"
            # Auto-degrade deep reasoning to fast reasoning under stress.
            if _system_degraded():
                model = get_best_model_for_role("reasoning") or "deepseek-r1:8b"
            default_timeout = 120.0 if model.endswith(":32b") else 75.0
            result = _run_reasoning_model(
                model=model,
                task=args.get("task", ""),
                context=args.get("context", ""),
                max_predict=560,
                timeout_seconds=_bounded_timeout(default_timeout, budget_seconds),
            )
            return _finish(result, ttl_seconds=30.0)

        if safe_tool_name == "read_gate_status":
            result = _read_gate_status(args.get("gate", "all"))
            return _finish(result, ttl_seconds=25.0)

        if safe_tool_name == "query_trade_metrics":
            result = _query_trade_metrics(args.get("metric", "summary"))
            return _finish(result, ttl_seconds=15.0)

        if safe_tool_name == "check_quant_validation_headroom":
            result = _check_quant_validation_headroom_tool(args, budget_seconds=budget_seconds)
            return _finish(result, ttl_seconds=15.0)

        if safe_tool_name == "search_web_tavily":
            result = _search_web_tavily_tool(args, budget_seconds=budget_seconds)
            return _finish(result, ttl_seconds=20.0)

        if safe_tool_name == "inspect_source_code":
            result = _inspect_source_code_tool(args, budget_seconds=budget_seconds)
            return _finish(result, ttl_seconds=20.0)

        if safe_tool_name == "check_github_repo_status":
            result = _check_github_repo_status_tool(args, budget_seconds=budget_seconds)
            return _finish(result, ttl_seconds=20.0)

        if safe_tool_name == "review_changed_files":
            return _finish(_review_changed_files_tool(args, budget_seconds=budget_seconds))

        if safe_tool_name == "summarize_test_failure":
            return _finish(_summarize_test_failure_tool(args, budget_seconds=budget_seconds))

        if safe_tool_name == "triage_gate_failure":
            result = _triage_gate_failure_tool(args, budget_seconds=budget_seconds)
            return _finish(result, ttl_seconds=15.0)

        if safe_tool_name == "run_production_audit_gate":
            return _finish(_run_production_audit_gate_tool(args, budget_seconds=budget_seconds))

        if safe_tool_name == "install_torch_runtime":
            return _finish(_install_torch_runtime_tool(args, budget_seconds=budget_seconds))

        if safe_tool_name == "install_python_package":
            return _finish(_install_python_package_tool(args, budget_seconds=budget_seconds))

        if safe_tool_name == "send_notification":
            return _finish(_send_notification(
                channel=args.get("channel", "whatsapp"),
                message=args.get("message", ""),
            ))

        if safe_tool_name == "run_sub_agent_batch":
            result = _run_sub_agent_batch(
                tasks=args.get("tasks", []),
                default_context=args.get("default_context", ""),
            )
            return _finish(result, ttl_seconds=30.0)

        if safe_tool_name == "check_system_health":
            result = _get_system_health_json()
            return _finish(result, ttl_seconds=10.0)

        unknown = f"Unknown tool: {safe_tool_name}"
        _record_tool_failure(safe_tool_name, args, unknown)
        return _tool_error_json(
            safe_tool_name,
            unknown,
            arguments=args,
            include_available=True,
        )
    except Exception as exc:
        _record_tool_failure(safe_tool_name, args, str(exc))
        return _tool_error_json(
            safe_tool_name,
            f"Tool execution failed: {exc}",
            arguments=args,
        )


def _run_reasoning_model(
    model: str,
    task: str,
    context: str = "",
    *,
    max_predict: int = 1024,
    timeout_seconds: float = 180.0,
    allow_fallback: bool = True,
) -> str:
    """Dispatch a reasoning task to a deepseek-r1 model with fallback."""
    prompt = task
    if context:
        prompt = f"Context:\n{context}\n\nTask:\n{task}"
    if len(prompt) > REASONING_MAX_PROMPT_CHARS:
        prompt = prompt[:REASONING_MAX_PROMPT_CHARS]

    # Try primary model, then fallback if allowed.
    models_to_try = [model]
    if allow_fallback:
        if model == "deepseek-r1:8b":
            models_to_try.append("deepseek-r1:32b")
        elif model == "deepseek-r1:32b":
            models_to_try.append("deepseek-r1:8b")
        # Last resort: any available model
        for m in MODEL_REGISTRY:
            if m not in models_to_try:
                models_to_try.append(m)

    activity = _get_activity_logger()

    for try_model in models_to_try:
        spec = MODEL_REGISTRY.get(try_model)
        if spec and (not spec.available or _model_in_cooldown(spec)):
            continue

        t0 = time.time()
        try:
            result = _ollama_post("/api/generate", {
                "model": try_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": int(max_predict)},
            }, timeout=float(timeout_seconds))
            response = result.get("response", "(no response)")
            latency_ms = (time.time() - t0) * 1000
            _record_model_stats(try_model, latency_ms, True)
            activity.log_request(
                model=try_model, prompt=prompt, response=response,
                latency_ms=latency_ms, task_type="reasoning",
            )
            if try_model != model:
                logger.info("Reasoning task fell back from %s to %s", model, try_model)
            return response
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            _record_model_stats(try_model, latency_ms, False)
            activity.log_request(
                model=try_model, prompt=prompt, response="",
                latency_ms=latency_ms, success=False, error=str(e),
            )
            logger.warning("Model %s failed: %s -- trying next fallback", try_model, e)
            continue

    return f"[ERROR] All models failed for reasoning task"


def _read_gate_status(gate: str) -> str:
    """Read latest audit gate logs."""
    action = "read_gate_status"
    audit_dir = PROJECT_ROOT / "logs" / "audit_sprint"
    if not audit_dir.exists():
        return json.dumps(
            {
                "status": "NO_DATA",
                "action": action,
                "gate": gate,
                "message": "No audit_sprint directory found.",
                "no_data_reason": "missing_audit_directory",
            },
            ensure_ascii=True,
        )

    runs = sorted([d for d in audit_dir.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        return json.dumps(
            {
                "status": "NO_DATA",
                "action": action,
                "gate": gate,
                "message": "No audit runs found.",
                "no_data_reason": "missing_audit_runs",
            },
            ensure_ascii=True,
        )

    latest = runs[0]
    results: dict[str, str] = {}

    if gate == "all":
        for f in sorted(latest.glob("gate_*.log")):
            results[f.stem] = f.read_text(encoding="utf-8", errors="replace")[:2000]
        if not results:
            return json.dumps(
                {
                    "status": "NO_DATA",
                    "action": action,
                    "gate": gate,
                    "run_dir": str(latest),
                    "message": "No gate logs were found in the latest audit run.",
                    "no_data_reason": "missing_gate_logs",
                },
                ensure_ascii=True,
            )
    else:
        target = latest / f"{gate}.log"
        if target.exists():
            results[gate] = target.read_text(encoding="utf-8", errors="replace")[:2000]
        else:
            return json.dumps(
                {
                    "status": "NO_DATA",
                    "action": action,
                    "gate": gate,
                    "run_dir": str(latest),
                    "message": f"Gate log not found: {target}",
                    "no_data_reason": "missing_gate_log",
                },
                ensure_ascii=True,
            )

    return json.dumps(
        {
            "status": "PASS",
            "action": action,
            "gate": gate,
            "run_dir": str(latest),
            "results": results,
        },
        ensure_ascii=True,
        indent=2,
    )


def _wilson_lower_bound(successes: int, total: int, z_score: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p_hat = max(0.0, min(1.0, float(successes) / float(total)))
    z2 = float(z_score) * float(z_score)
    denom = 1.0 + (z2 / float(total))
    centre = p_hat + (z2 / (2.0 * float(total)))
    variance = (p_hat * (1.0 - p_hat) / float(total)) + (z2 / (4.0 * float(total) * float(total)))
    margin = float(z_score) * math.sqrt(max(0.0, variance))
    return max(0.0, min(1.0, (centre - margin) / denom))


def _normal_two_sided_pvalue_from_z(z_score: float) -> float:
    # two-sided normal approximation, no external stats deps required
    return max(0.0, min(1.0, math.erfc(abs(float(z_score)) / math.sqrt(2.0))))


def _build_trading_objective_report(total_trades: int, wins: int, losses: int) -> dict[str, Any]:
    total = max(0, int(total_trades))
    win_count = max(0, int(wins))
    loss_count = max(0, int(losses))
    if total <= 0:
        return {
            "status": "LIMITED",
            "objective_function": "minimize conservative_error_rate_upper_bound - target_max_error_rate",
            "objective_value": None,
            "sample_size": 0,
            "targets": {
                "target_max_error_rate": TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE,
                "min_trades_for_significance": TRADING_OBJECTIVE_MIN_TRADES,
                "min_wilson_win_rate": TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE,
                "max_p_value": TRADING_OBJECTIVE_PVALUE_MAX,
                "wilson_z_score": TRADING_OBJECTIVE_WILSON_Z,
            },
            "limitations": [
                "No production round-trip trades available for significance testing.",
            ],
            "openclaw_self_improve": [
                "Keep recommendations in advisory/read-only mode until enough live-paper evidence is collected.",
                "Run confidence calibration (isotonic or Platt) before using confidence for sizing decisions.",
                "Force tool-backed responses and gate/metric checks for every trading-critical prompt.",
            ],
        }

    # Keep counts internally consistent if malformed source data appears.
    if win_count + loss_count != total:
        loss_count = max(0, total - win_count)

    observed_win_rate = win_count / total
    observed_error_rate = loss_count / total
    wilson_win_rate_lower = _wilson_lower_bound(win_count, total, z_score=TRADING_OBJECTIVE_WILSON_Z)
    conservative_error_rate_upper = max(0.0, 1.0 - wilson_win_rate_lower)
    objective_value = conservative_error_rate_upper - TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE

    # Binomial test approximation against random 50/50 directional accuracy.
    z_score = 0.0
    p_value = 1.0
    if total > 0:
        denom = math.sqrt(0.25 * total)
        if denom > 0:
            z_score = (win_count - (0.5 * total)) / denom
            p_value = _normal_two_sided_pvalue_from_z(z_score)

    sufficient_sample = total >= TRADING_OBJECTIVE_MIN_TRADES
    statistically_significant = sufficient_sample and (p_value <= TRADING_OBJECTIVE_PVALUE_MAX)
    passes_objective = bool(
        statistically_significant
        and (wilson_win_rate_lower >= TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE)
        and (conservative_error_rate_upper <= TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE)
    )

    limitations: list[str] = []
    if not sufficient_sample:
        limitations.append(
            f"Sample size too small for robust significance (have {total}, need {TRADING_OBJECTIVE_MIN_TRADES})."
        )
    if p_value > TRADING_OBJECTIVE_PVALUE_MAX:
        limitations.append(
            f"Directional edge is not statistically significant (p_value={p_value:.4f}, max={TRADING_OBJECTIVE_PVALUE_MAX:.4f})."
        )
    if wilson_win_rate_lower < TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE:
        limitations.append(
            f"Conservative win-rate bound is below target ({wilson_win_rate_lower:.3f} < {TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE:.3f})."
        )
    if conservative_error_rate_upper > TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE:
        limitations.append(
            f"Conservative error-rate upper bound exceeds target ({conservative_error_rate_upper:.3f} > {TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE:.3f})."
        )

    status = "PASS" if passes_objective else ("LIMITED" if sufficient_sample else "FAIL")
    return {
        "status": status,
        "objective_function": "minimize conservative_error_rate_upper_bound - target_max_error_rate",
        "objective_value": round(objective_value, 6),
        "sample_size": total,
        "observed": {
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(observed_win_rate, 6),
            "error_rate": round(observed_error_rate, 6),
        },
        "confidence_bounds": {
            "wilson_win_rate_lower": round(wilson_win_rate_lower, 6),
            "conservative_error_rate_upper": round(conservative_error_rate_upper, 6),
            "z_score": round(z_score, 6),
            "p_value": round(p_value, 6),
        },
        "targets": {
            "target_max_error_rate": TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE,
            "min_trades_for_significance": TRADING_OBJECTIVE_MIN_TRADES,
            "min_wilson_win_rate": TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE,
            "max_p_value": TRADING_OBJECTIVE_PVALUE_MAX,
            "wilson_z_score": TRADING_OBJECTIVE_WILSON_Z,
        },
        "statistical_significance": {
            "sufficient_sample_size": sufficient_sample,
            "significant_vs_random_baseline": statistically_significant,
        },
        "limitations": limitations,
        "openclaw_self_improve": [
            "Use high-accuracy profile for trading-critical prompts and keep tool-primer forced.",
            "Fallback to evidence-first snapshots on model timeout instead of speculative recommendations.",
            "Hold sizing/aggressive actions until objective status is PASS with significance.",
        ],
    }


def _query_trade_metrics(metric: str) -> str:
    """Query canonical PnL metrics via PnLIntegrityEnforcer."""
    db_path = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
    if not db_path.exists():
        if metric == "objective":
            return json.dumps(
                {
                    "status": "LIMITED",
                    "objective_function": "minimize conservative_error_rate_upper_bound - target_max_error_rate",
                    "objective_value": None,
                    "sample_size": 0,
                    "limitations": ["Database not found; cannot evaluate trading error objective."],
                    "openclaw_self_improve": [
                        "Run paper trading to generate canonical round-trip metrics.",
                        "Re-run objective check once portfolio_maximizer.db is available.",
                    ],
                },
                indent=2,
            )
        return json.dumps({"error": "Database not found"})

    try:
        from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer

        with PnLIntegrityEnforcer(str(db_path)) as enforcer:
            metrics = enforcer.get_canonical_metrics()
            total_trades = int(
                getattr(metrics, "total_round_trips", getattr(metrics, "total_trades", 0)) or 0
            )
            wins = int(getattr(metrics, "win_count", getattr(metrics, "wins", 0)) or 0)
            losses = int(getattr(metrics, "loss_count", getattr(metrics, "losses", 0)) or 0)
            objective = _build_trading_objective_report(total_trades=total_trades, wins=wins, losses=losses)
            if metric == "summary":
                return json.dumps({
                    "total_pnl": f"${metrics.total_realized_pnl:+,.2f}",
                    "win_rate": f"{metrics.win_rate:.1%}",
                    "error_rate": f"{(1.0 - float(metrics.win_rate)):.1%}",
                    "profit_factor": f"{metrics.profit_factor:.2f}",
                    "total_trades": total_trades,
                    "wins": wins,
                    "losses": losses,
                    "trading_error_objective": {
                        "status": objective.get("status"),
                        "objective_value": objective.get("objective_value"),
                        "sample_size": objective.get("sample_size"),
                    },
                })
            elif metric == "objective":
                return json.dumps(objective, indent=2)
            elif metric == "canonical_pnl":
                return f"${metrics.total_realized_pnl:+,.2f}"
            elif metric == "win_rate":
                return f"{metrics.win_rate:.1%}"
            elif metric == "profit_factor":
                return f"{metrics.profit_factor:.2f}"
            elif metric == "round_trips":
                return str(total_trades)
            else:
                return json.dumps({"error": f"Unknown metric: {metric}"})
    except Exception as e:
        if metric == "objective":
            return json.dumps(
                {
                    "status": "LIMITED",
                    "objective_function": "minimize conservative_error_rate_upper_bound - target_max_error_rate",
                    "objective_value": None,
                    "sample_size": 0,
                    "limitations": [f"Objective evaluation failed: {e}"],
                    "openclaw_self_improve": [
                        "Run integrity checks and ensure DB schema is healthy before objective evaluation.",
                        "Keep trading-critical responses evidence-only until objective checks are healthy.",
                    ],
                },
                indent=2,
            )
        return json.dumps({"error": str(e)})


def _check_quant_validation_headroom_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    script = PROJECT_ROOT / "scripts" / "quant_validation_headroom.py"
    if not script.exists():
        return json.dumps(
            {
                "status": "FAIL",
                "action": "check_quant_validation_headroom",
                "error": f"missing_script: {script}",
            },
            ensure_ascii=True,
        )

    try:
        window = int(args.get("window", 120))
    except Exception:
        window = 120
    window = max(0, min(10000, window))

    red_gate_pct = _as_float(args.get("red_gate_pct"), 95.0)
    red_gate_pct = max(1.0, min(100.0, red_gate_pct))
    warn_gate_pct = _as_float(args.get("warn_gate_pct"), 90.0)
    warn_gate_pct = max(0.0, min(red_gate_pct, warn_gate_pct))

    cmd = [
        sys.executable,
        str(script),
        "--window",
        str(window),
        "--red-gate-pct",
        f"{red_gate_pct:.3f}",
        "--warn-gate-pct",
        f"{warn_gate_pct:.3f}",
        "--json",
    ]

    log_path = str(args.get("log_path") or "").strip()
    if log_path:
        cmd.extend(["--log-path", log_path])

    timeout_seconds = _bounded_timeout(
        _as_float(args.get("timeout_seconds"), 45.0),
        budget_seconds,
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "check_quant_validation_headroom",
                "error": f"quant_validation_headroom timed out after {timeout_seconds:.1f}s",
                "command": cmd,
            },
            ensure_ascii=True,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "check_quant_validation_headroom",
                "error": f"quant_validation_headroom execution error: {exc}",
                "command": cmd,
            },
            ensure_ascii=True,
        )

    stdout_text = str(proc.stdout or "").strip()
    parsed: Optional[dict[str, Any]] = None
    if stdout_text:
        try:
            obj = json.loads(stdout_text)
            if isinstance(obj, dict):
                parsed = obj
        except Exception:
            start = stdout_text.find("{")
            end = stdout_text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(stdout_text[start : end + 1])
                    if isinstance(obj, dict):
                        parsed = obj
                except Exception:
                    parsed = None

    classification = str((parsed or {}).get("status") or "").upper()
    if classification == "RED":
        status = "FAIL"
    elif classification == "YELLOW":
        status = "WARN"
    elif classification == "GREEN":
        status = "PASS"
    else:
        status = "PASS" if int(proc.returncode) == 0 else "FAIL"

    payload: dict[str, Any] = {
        "status": status,
        "action": "check_quant_validation_headroom",
        "exit_code": int(proc.returncode),
        "classification": classification or None,
        "summary": parsed,
        "command": cmd,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
    }
    if parsed is None and stdout_text:
        payload["raw_output"] = stdout_text[:4000]
    return json.dumps(payload, ensure_ascii=True)


def _search_web_tavily_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    query = str(arguments.get("query") or "").strip()
    if not query:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "search_web_tavily",
                "error": "missing_query",
            }
        )

    try:
        max_results = int(arguments.get("max_results", 5))
    except Exception:
        max_results = 5
    max_results = max(1, min(10, max_results))

    search_depth = str(arguments.get("search_depth") or "basic").strip().lower() or "basic"
    if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
        search_depth = "basic"

    topic = str(arguments.get("topic") or "general").strip() or "general"
    topic = topic.lower()
    if topic not in {"general", "news", "finance"}:
        topic = "general"
    include_domains = str(arguments.get("include_domains") or "").strip() or None
    exclude_domains = str(arguments.get("exclude_domains") or "").strip() or None
    tavily_base_url = str(arguments.get("base_url") or "").strip() or None
    providers_csv = str(arguments.get("providers") or "").strip() or None
    allow_fallback = _as_bool(arguments.get("allow_fallback"), True)
    if not allow_fallback and not providers_csv:
        providers_csv = "tavily"

    try:
        from utils.web_search_client import search_web_multi
    except Exception as exc:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "search_web_tavily",
                "query": query,
                "error": f"web_search_client_unavailable: {exc}",
            }
        )

    timeout_seconds = _bounded_timeout(20.0, budget_seconds)
    result = search_web_multi(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        include_domains_csv=include_domains,
        exclude_domains_csv=exclude_domains,
        tavily_base_url=tavily_base_url,
        timeout_seconds=timeout_seconds,
        providers_csv=providers_csv,
    )

    payload = {
        "status": "PASS" if bool(result.ok) else "FAIL",
        "action": "search_web_tavily",
        "query": result.query or query,
        "provider": result.provider,
        "attempts": result.attempts,
        "answer": result.answer,
        "results": result.results,
        "error": result.error,
        "status_code": result.status_code,
        "latency_seconds": result.latency_seconds,
    }
    return json.dumps(payload, ensure_ascii=True)


def _normalize_rel_path(path_text: str) -> str:
    text = str(path_text or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _split_source_path_args(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_normalize_rel_path(v) for v in value if str(v or "").strip()]
    if isinstance(value, str):
        parts = re.split(r"[,;\n]", value)
        return [_normalize_rel_path(v) for v in parts if str(v or "").strip()]
    return []


def _is_allowed_source_rel_path(rel_path: str) -> bool:
    rel = _normalize_rel_path(rel_path)
    if not rel:
        return False
    low = rel.lower()
    if any(token and token in low for token in SOURCE_READ_BLOCKED_TOKENS):
        return False
    return any(rel.startswith(prefix.rstrip("/")) for prefix in SOURCE_READ_ALLOWED_PREFIXES)


def _safe_read_source_index(refresh: bool = False, timeout_seconds: float = 45.0) -> dict[str, Any]:
    if refresh or not SOURCE_INDEX_FILE.exists():
        setup_script = PROJECT_ROOT / "scripts" / "openclaw_self_improve.py"
        if setup_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(setup_script), "index"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=max(10.0, timeout_seconds),
                    check=False,
                )
            except Exception:
                pass
    if not SOURCE_INDEX_FILE.exists():
        return {"files": []}
    try:
        payload = json.loads(SOURCE_INDEX_FILE.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {"files": []}


def _source_query_tokens(query: str) -> list[str]:
    low = str(query or "").lower()
    tokens = [tok for tok in re.findall(r"[a-z][a-z0-9_]{2,}", low) if tok not in SOURCE_QUERY_STOPWORDS]
    # Deduplicate while preserving order.
    return list(dict.fromkeys(tokens))[:24]


def _source_candidate_paths(index_payload: dict[str, Any], query: str, requested_paths: list[str], max_files: int) -> list[str]:
    files = index_payload.get("files") if isinstance(index_payload.get("files"), list) else []
    index_paths: list[str] = []
    for row in files:
        if not isinstance(row, dict):
            continue
        path = _normalize_rel_path(str(row.get("path") or ""))
        if path and _is_allowed_source_rel_path(path):
            index_paths.append(path)

    seen: set[str] = set()
    selected: list[str] = []
    for raw in requested_paths:
        p = _normalize_rel_path(raw)
        if not p or p in seen or p not in index_paths:
            continue
        selected.append(p)
        seen.add(p)
        if len(selected) >= max_files:
            return selected

    query_tokens = _source_query_tokens(query)
    scored: list[tuple[int, str]] = []
    for p in index_paths:
        if p in seen:
            continue
        low = p.lower()
        score = 0
        for tok in query_tokens:
            if tok in low:
                score += 3
        basename = Path(p).name.lower()
        for tok in query_tokens:
            if tok in basename:
                score += 2
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda row: (-row[0], row[1]))
    for _, p in scored:
        if p in seen:
            continue
        selected.append(p)
        seen.add(p)
        if len(selected) >= max_files:
            break
    return selected


def _snippet_rows_for_content(content: str, query_tokens: list[str], max_snippets: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = content.splitlines()
    if not lines:
        return rows

    def _append_line(line_no: int, text: str) -> None:
        trimmed = str(text or "").strip()
        if not trimmed:
            return
        rows.append(
            {
                "line": int(line_no),
                "text": trimmed[:220],
            }
        )

    if query_tokens:
        for idx, line in enumerate(lines, start=1):
            low = line.lower()
            if any(tok in low for tok in query_tokens):
                _append_line(idx, line)
                if len(rows) >= max_snippets:
                    break

    if rows:
        return rows

    for idx, line in enumerate(lines, start=1):
        text = str(line or "").lstrip()
        if text.startswith(("def ", "class ", "async def ", "import ", "from ")):
            _append_line(idx, line)
            if len(rows) >= max_snippets:
                break

    if rows:
        return rows

    _append_line(1, lines[0])
    return rows


def _inspect_source_code_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    query = str(args.get("query") or "").strip()
    requested_paths = _split_source_path_args(args.get("paths"))
    refresh_index = _as_bool(args.get("refresh_index"), False)

    if not query and not requested_paths:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "inspect_source_code",
                "error": "Provide query and/or paths.",
            },
            ensure_ascii=True,
        )

    max_files = _as_int(args.get("max_files"), SOURCE_READ_DEFAULT_MAX_FILES)
    max_files = max(1, min(SOURCE_READ_MAX_FILES_CAP, max_files))
    max_chars = _as_int(args.get("max_chars"), SOURCE_READ_DEFAULT_MAX_CHARS)
    max_chars = max(1200, min(SOURCE_READ_MAX_CHARS_CAP, max_chars))
    timeout_seconds = _bounded_timeout(30.0, budget_seconds)

    index_payload = _safe_read_source_index(refresh=refresh_index, timeout_seconds=timeout_seconds)
    selected_paths = _source_candidate_paths(
        index_payload=index_payload,
        query=query,
        requested_paths=requested_paths,
        max_files=max_files,
    )

    total_chars = 0
    matches: list[dict[str, Any]] = []
    citations: list[str] = []
    query_tokens = _source_query_tokens(query)

    for rel in selected_paths:
        full_path = (PROJECT_ROOT / rel).resolve()
        if not full_path.exists() or not full_path.is_file():
            continue
        if total_chars >= max_chars:
            break
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not content:
            continue

        remaining = max_chars - total_chars
        excerpt = content[:remaining]
        total_chars += len(excerpt)

        snippets = _snippet_rows_for_content(
            excerpt,
            query_tokens=query_tokens,
            max_snippets=SOURCE_READ_MAX_SNIPPETS_PER_FILE,
        )
        for row in snippets:
            line_no = int(row.get("line") or 1)
            citations.append(f"{rel}:{line_no}")

        matches.append(
            {
                "path": rel,
                "chars_read": len(excerpt),
                "snippets": snippets,
            }
        )

    status = "PASS" if matches else "FAIL"
    payload: dict[str, Any] = {
        "status": status,
        "action": "inspect_source_code",
        "query": query,
        "requested_paths": requested_paths,
        "selected_paths": [m.get("path") for m in matches],
        "matches": matches,
        "citations": list(dict.fromkeys(citations))[:64],
        "limits": {
            "max_files": max_files,
            "max_chars": max_chars,
            "chars_read": total_chars,
        },
    }
    if not matches:
        payload["error"] = "No eligible source files matched query/paths."
    return json.dumps(payload, ensure_ascii=True)


def _check_github_repo_status_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    _ = arguments if isinstance(arguments, dict) else {}
    script = PROJECT_ROOT / "scripts" / "check_github_repo_status.py"
    if not script.exists():
        return json.dumps(
            {
                "status": "FAIL",
                "action": "check_github_repo_status",
                "error": f"missing_script: {script}",
            },
            ensure_ascii=True,
        )

    timeout_seconds = _bounded_timeout(25.0, budget_seconds)
    cmd = [sys.executable, str(script), "--pretty"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "check_github_repo_status",
                "error": f"check_github_repo_status timed out after {timeout_seconds:.1f}s",
                "command": cmd,
            },
            ensure_ascii=True,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "check_github_repo_status",
                "error": f"check_github_repo_status execution error: {exc}",
                "command": cmd,
            },
            ensure_ascii=True,
        )

    stdout_text = str(proc.stdout or "").strip()
    parsed_report: Optional[dict[str, Any]] = None
    if stdout_text:
        try:
            report_obj = json.loads(stdout_text)
            if isinstance(report_obj, dict):
                parsed_report = report_obj
        except Exception:
            parsed_report = None

    status = "PASS" if int(proc.returncode) == 0 else "FAIL"
    if parsed_report:
        report_status = str(parsed_report.get("status", "")).upper()
        if report_status in {"PASS", "WARN", "FAIL"}:
            status = report_status if int(proc.returncode) == 0 else "FAIL"

    payload: dict[str, Any] = {
        "status": status,
        "action": "check_github_repo_status",
        "exit_code": int(proc.returncode),
        "command": cmd,
        "report": parsed_report,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
    }
    if parsed_report is None and stdout_text:
        payload["raw_output"] = stdout_text[:4000]
    return json.dumps(payload, ensure_ascii=True)


def _review_changed_files_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    payload = _review_changed_files_helper(
        project_root=PROJECT_ROOT,
        paths=args.get("paths") if isinstance(args.get("paths"), list) else None,
        base_ref=str(args.get("base_ref") or "").strip(),
        max_files=max(1, min(40, _as_int(args.get("max_files"), 12))),
        max_diff_chars=max(800, min(40000, _as_int(args.get("max_diff_chars"), 16000))),
        diff_context=max(0, min(6, _as_int(args.get("diff_context"), 1))),
        include_untracked=_as_bool(args.get("include_untracked"), True),
        timeout_seconds=_bounded_timeout(25.0, budget_seconds),
    )
    return json.dumps(payload, ensure_ascii=True)


def _summarize_test_failure_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    del budget_seconds
    args = arguments if isinstance(arguments, dict) else {}
    payload = _summarize_test_failure_helper(
        log_path=str(args.get("log_path") or "").strip(),
        output_text=str(args.get("output_text") or ""),
        max_failures=max(1, min(25, _as_int(args.get("max_failures"), 10))),
        max_suspect_paths=max(1, min(20, _as_int(args.get("max_suspect_paths"), 8))),
        max_error_types=max(1, min(20, _as_int(args.get("max_error_types"), 8))),
    )
    return json.dumps(payload, ensure_ascii=True)


def _triage_gate_failure_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    del budget_seconds
    args = arguments if isinstance(arguments, dict) else {}
    payload = _triage_gate_failure_helper(
        gate_artifact=str(args.get("gate_artifact") or "").strip(),
        summary_cache=str(args.get("summary_cache") or "").strip(),
        output_json_path=str(args.get("output_json_path") or "").strip(),
        output_md_path=str(args.get("output_md_path") or "").strip(),
        force_refresh=_as_bool(args.get("force_refresh"), False),
    )
    return json.dumps(payload, ensure_ascii=True)


def _run_production_audit_gate_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    """Run production gate via structured args (no shell concatenation)."""
    args = arguments if isinstance(arguments, dict) else {}
    script = PROJECT_ROOT / "scripts" / "production_audit_gate.py"
    artifact = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
    cmd = [sys.executable, str(script), "--output-json", str(artifact)]

    close_ids_raw = args.get("reconcile_close_ids")
    close_ids: list[int] = []
    if isinstance(close_ids_raw, list):
        for raw in close_ids_raw:
            try:
                val = int(raw)
            except Exception:
                continue
            if val > 0:
                close_ids.append(val)
    if close_ids or ("reconcile_close_ids" in args):
        cmd.append("--reconcile")
        cmd.extend(str(x) for x in close_ids)

    if _as_bool(args.get("reconcile_apply"), False):
        cmd.append("--reconcile-apply")
    if _as_bool(args.get("require_holding_period"), False):
        cmd.append("--require-holding-period")
    if _as_bool(args.get("allow_inconclusive_lift"), False):
        cmd.append("--allow-inconclusive-lift")
    if _as_bool(args.get("require_profitable"), False):
        cmd.append("--require-profitable")
    if _as_bool(args.get("notify_openclaw"), False):
        cmd.append("--notify-openclaw")

    max_files = _as_int(args.get("max_files"), 0)
    if max_files > 0:
        cmd.extend(["--max-files", str(max_files)])

    timeout_seconds = _bounded_timeout(
        _as_float(args.get("timeout_seconds"), 120.0),
        budget_seconds,
    )

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "FAIL",
                "error": f"production_audit_gate timed out after {timeout_seconds:.1f}s",
                "command": cmd,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "FAIL",
                "error": f"Failed to run production gate: {exc}",
                "command": cmd,
            },
            indent=2,
        )

    summary: dict[str, Any] = {}
    if artifact.exists():
        try:
            parsed = json.loads(artifact.read_text(encoding="utf-8", errors="replace"))
            if isinstance(parsed, dict):
                summary = {
                    "timestamp_utc": parsed.get("timestamp_utc"),
                    "gate_status": ((parsed.get("production_profitability_gate") or {}).get("status")),
                    "lift_status": ((parsed.get("lift_gate") or {}).get("status")),
                    "proof_status": ((parsed.get("profitability_proof") or {}).get("status")),
                    "reconciliation": parsed.get("reconciliation"),
                }
        except Exception:
            summary = {}

    return json.dumps(
        {
            "status": "PASS" if int(proc.returncode) == 0 else "FAIL",
            "exit_code": int(proc.returncode),
            "command": cmd,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
            "artifact": str(artifact),
            "summary": summary,
        },
        indent=2,
    )


def _probe_torch_runtime(timeout_seconds: float = 25.0) -> dict[str, Any]:
    probe_cmd = [
        sys.executable,
        "-c",
        (
            "import json; "
            "import torch; "
            "print(json.dumps({"
            "'version': getattr(torch, '__version__', ''), "
            "'cuda': getattr(getattr(torch, 'version', object()), 'cuda', None), "
            "'cuda_available': bool(torch.cuda.is_available())"
            "}, ensure_ascii=True))"
        ),
    ]
    try:
        proc = subprocess.run(
            probe_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(5.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"installed": False, "error": "torch probe timed out"}
    except Exception as exc:
        return {"installed": False, "error": f"torch probe failed: {exc}"}

    if int(proc.returncode) != 0:
        err = "\n".join((proc.stderr or "").splitlines()[-20:])
        return {"installed": False, "error": err or "torch import failed"}

    payload = {}
    out = (proc.stdout or "").strip()
    for line in reversed(out.splitlines()):
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
                break
        except Exception:
            continue
    if not payload:
        payload = {"raw_stdout": "\n".join(out.splitlines()[-20:])}
    payload["installed"] = True
    return payload


def _detect_nvidia_gpu(timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Best-effort GPU detection for torch install decisions (no hard dependency)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"gpu_detected": False, "method": "nvidia-smi", "available": False}
    try:
        proc = subprocess.run(
            [smi, "-L"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"gpu_detected": False, "method": "nvidia-smi", "available": True, "error": "timeout"}
    except Exception as exc:
        return {
            "gpu_detected": False,
            "method": "nvidia-smi",
            "available": True,
            "error": str(exc)[:200],
        }

    out = (proc.stdout or "").strip()
    stdout_tail_lines = [
        re.sub(r"\(UUID:\s*[^)]+\)", "(UUID: [REDACTED])", line)
        for line in out.splitlines()[-5:]
    ]
    detected = int(proc.returncode) == 0 and ("GPU" in out or "NVIDIA" in out)
    return {
        "gpu_detected": bool(detected),
        "method": "nvidia-smi",
        "available": True,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(stdout_tail_lines),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-5:]),
    }


def _is_permission_error_text(text: str) -> bool:
    low = str(text or "").lower()
    return any(
        tok in low
        for tok in (
            "permission denied",
            "access is denied",
            "errno 13",
            "not permitted",
            "requires administrator",
        )
    )


def _run_pip_install_attempt(cmd: list[str], timeout_seconds: float) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(30.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "FAIL",
            "error": f"pip install timed out after {timeout_seconds:.1f}s",
            "command": cmd,
            "started_at_utc": started,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "error": f"pip install failed to start: {exc}",
            "command": cmd,
            "started_at_utc": started,
        }

    return {
        "status": "PASS" if int(proc.returncode) == 0 else "FAIL",
        "exit_code": int(proc.returncode),
        "command": cmd,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-30:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-30:]),
    }


def _normalize_package_spec(package_raw: Any, version_raw: Any) -> tuple[str, Optional[str]]:
    package = str(package_raw or "").strip()
    version = str(version_raw or "").strip()
    if not package:
        return "", "Missing package name"
    if package.startswith("-"):
        return "", "Invalid package spec: cannot start with '-'"
    if any(ch.isspace() for ch in package):
        return "", "Invalid package spec: whitespace is not allowed"
    if "@" in package:
        return "", "Direct URL/VCS package specs are disabled for runtime installs"

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.[]<>=!~,+")
    for ch in package:
        if ch not in allowed:
            return "", f"Invalid package spec character: {ch}"

    has_comparator = any(tok in package for tok in ("==", ">=", "<=", "!=", "~=", ">", "<", "@"))
    if version:
        if has_comparator:
            return "", "Do not set version when package already contains a comparator/specifier"
        if any(ch.isspace() for ch in version):
            return "", "Invalid version: whitespace is not allowed"
        version_allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+")
        for ch in version:
            if ch not in version_allowed:
                return "", f"Invalid version character: {ch}"
        package = f"{package}=={version}"

    return package, None


def _guess_import_module_name(package_spec: str) -> str:
    raw = str(package_spec or "").strip()
    if not raw:
        return ""
    cut = len(raw)
    for token in ("[", "=", "<", ">", "!", "~", "@", ","):
        idx = raw.find(token)
        if idx >= 0:
            cut = min(cut, idx)
    base = raw[:cut].strip()
    if not base:
        return ""
    return base.replace("-", "_").replace(".", "_")


def _normalize_module_name(module_raw: Any) -> tuple[str, Optional[str]]:
    module = str(module_raw or "").strip()
    if not module:
        return "", "verify_import is empty after normalization"
    if any(ch.isspace() for ch in module):
        return "", "verify_import must not contain whitespace"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
    for ch in module:
        if ch not in allowed:
            return "", f"Invalid verify_import character: {ch}"
    if module.startswith(".") or module.endswith(".") or ".." in module:
        return "", "verify_import must be a valid dotted module path"
    return module, None


def _normalize_index_url(url_raw: Any, field_name: str) -> tuple[str, Optional[str]]:
    raw = str(url_raw or "").strip()
    if not raw:
        return "", None
    if any(ch.isspace() for ch in raw):
        return "", f"{field_name} must not contain whitespace"
    if not raw.startswith("https://"):
        return "", f"{field_name} must start with https://"
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        return "", f"{field_name} must be a valid https URL"
    if parsed.username or parsed.password:
        return "", f"{field_name} must not include credentials"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "", f"{field_name} host is missing"

    default_allowed_hosts = {
        "pypi.org",
        "files.pythonhosted.org",
        "download.pytorch.org",
        "test.pypi.org",
    }
    extra_hosts_raw = str(os.getenv("PMX_RUNTIME_PIP_ALLOWED_INDEX_HOSTS") or "").strip()
    extra_hosts = {h.strip().lower() for h in extra_hosts_raw.split(",") if h.strip()}
    allowed_hosts = default_allowed_hosts | extra_hosts
    if host not in allowed_hosts:
        allowed_str = ", ".join(sorted(allowed_hosts))
        return "", f"{field_name} host '{host}' is not allowed (allowed: {allowed_str})"
    return raw, None


def _probe_python_module(module_name: str, timeout_seconds: float = 25.0) -> dict[str, Any]:
    module, err = _normalize_module_name(module_name)
    if err:
        return {"installed": False, "error": err}

    probe_cmd = [
        sys.executable,
        "-c",
        (
            "import json, importlib; "
            f"m = importlib.import_module('{module}'); "
            "print(json.dumps({'module': '" + module + "', 'version': getattr(m, '__version__', '')}, ensure_ascii=True))"
        ),
    ]
    try:
        proc = subprocess.run(
            probe_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(5.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"installed": False, "error": f"module probe timed out ({module})"}
    except Exception as exc:
        return {"installed": False, "error": f"module probe failed ({module}): {exc}"}

    if int(proc.returncode) != 0:
        err_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        return {"installed": False, "module": module, "error": err_tail or "module import failed"}

    payload: dict[str, Any] = {}
    out = (proc.stdout or "").strip()
    for line in reversed(out.splitlines()):
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
                break
        except Exception:
            continue
    if not payload:
        payload = {"module": module, "raw_stdout": "\n".join(out.splitlines()[-20:])}
    payload["installed"] = True
    return payload


def _install_python_package_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    if not _truthy_env("PMX_ALLOW_RUNTIME_PIP_INSTALL", default=False):
        return json.dumps(
            {
                "status": "FAIL",
                "action": "install_python_package",
                "error": (
                    "Runtime pip installation is disabled by policy. "
                    "Set PMX_ALLOW_RUNTIME_PIP_INSTALL=1 to enable explicitly."
                ),
            },
            indent=2,
        )

    pkg_spec, pkg_err = _normalize_package_spec(
        args.get("package", ""),
        args.get("version", ""),
    )
    if pkg_err:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "install_python_package",
                "error": pkg_err,
            },
            indent=2,
        )

    verify_import_raw = str(args.get("verify_import") or "").strip()
    verify_import = verify_import_raw or _guess_import_module_name(pkg_spec)
    module_name, module_err = _normalize_module_name(verify_import)
    if module_err:
        return json.dumps(
            {
                "status": "FAIL",
                "action": "install_python_package",
                "error": module_err,
                "requested": {"package": pkg_spec, "verify_import": verify_import_raw},
            },
            indent=2,
        )

    index_url, index_err = _normalize_index_url(args.get("index_url"), "index_url")
    if index_err:
        return json.dumps(
            {"status": "FAIL", "action": "install_python_package", "error": index_err},
            indent=2,
        )
    extra_index_url, extra_index_err = _normalize_index_url(args.get("extra_index_url"), "extra_index_url")
    if extra_index_err:
        return json.dumps(
            {"status": "FAIL", "action": "install_python_package", "error": extra_index_err},
            indent=2,
        )

    scope = str(args.get("scope") or "auto").strip().lower()
    if scope not in {"auto", "system", "user"}:
        scope = "auto"
    verify_only = _as_bool(args.get("verify_only"), False)
    upgrade = _as_bool(args.get("upgrade"), True)
    pre_release = _as_bool(args.get("pre_release"), False)

    timeout_seconds = _bounded_timeout(
        _as_float(args.get("timeout_seconds"), 900.0),
        budget_seconds,
    )
    in_venv = bool((os.getenv("VIRTUAL_ENV") or "").strip()) or (sys.prefix != getattr(sys, "base_prefix", sys.prefix))

    initial_probe = _probe_python_module(module_name, timeout_seconds=min(25.0, timeout_seconds))
    if verify_only:
        return json.dumps(
            {
                "status": "PASS" if initial_probe.get("installed") else "FAIL",
                "action": "verify_python_package",
                "python_executable": sys.executable,
                "in_virtualenv": in_venv,
                "requested": {"package": pkg_spec, "verify_import": module_name},
                "probe": initial_probe,
            },
            indent=2,
        )

    if initial_probe.get("installed") and (not upgrade) and (not index_url) and (not extra_index_url):
        return json.dumps(
            {
                "status": "PASS",
                "action": "install_python_package",
                "python_executable": sys.executable,
                "in_virtualenv": in_venv,
                "requested": {"package": pkg_spec, "verify_import": module_name, "scope": scope, "upgrade": upgrade},
                "initial_probe": initial_probe,
                "attempts": [],
                "probe": initial_probe,
                "limitations": [],
                "note": "Package already importable and upgrade=false; skipped install.",
            },
            indent=2,
        )

    def _pip_cmd(*, user_flag: bool) -> list[str]:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        if upgrade:
            cmd.append("--upgrade")
        if pre_release:
            cmd.append("--pre")
        if user_flag:
            cmd.append("--user")
        if index_url:
            cmd.extend(["--index-url", index_url])
        if extra_index_url:
            cmd.extend(["--extra-index-url", extra_index_url])
        cmd.append(pkg_spec)
        return cmd

    use_user_requested = scope == "user"
    use_system_requested = scope == "system"
    user_fallback_allowed = scope == "auto" and not in_venv

    plan: list[tuple[bool, str]] = []
    if use_user_requested:
        plan.append((True, "user_scope_requested"))
    else:
        plan.append((False, "primary"))
    if user_fallback_allowed:
        plan.append((True, "auto_user_fallback"))

    attempts: list[dict[str, Any]] = []
    chosen_result: dict[str, Any] | None = None
    seen_cmd: set[str] = set()

    for use_user, label in plan:
        if use_system_requested and use_user:
            continue
        cmd = _pip_cmd(user_flag=use_user)
        key = json.dumps(cmd, ensure_ascii=True)
        if key in seen_cmd:
            continue
        seen_cmd.add(key)

        row = _run_pip_install_attempt(cmd, timeout_seconds=timeout_seconds)
        row["attempt_label"] = label
        row["scope"] = "user" if use_user else "system_or_venv"
        attempts.append(row)

        if row.get("status") == "PASS":
            chosen_result = row
            break

        if not user_fallback_allowed and not use_user_requested:
            err_text = f"{row.get('stdout_tail', '')}\n{row.get('stderr_tail', '')}"
            if _is_permission_error_text(err_text):
                break

    final_probe = _probe_python_module(module_name, timeout_seconds=min(30.0, timeout_seconds))
    ok = bool(chosen_result and chosen_result.get("status") == "PASS" and final_probe.get("installed"))

    limitations: list[str] = []
    if not final_probe.get("installed"):
        limitations.append("module import verification failed after install attempts")
    if not ok and chosen_result and chosen_result.get("status") == "PASS":
        limitations.append("installation reported PASS but verification import failed; set verify_import to exact module path")
    if not ok and use_system_requested:
        limitations.append("scope=system was requested; no --user fallback applied")

    return json.dumps(
        {
            "status": "PASS" if ok else "FAIL",
            "action": "install_python_package",
            "python_executable": sys.executable,
            "in_virtualenv": in_venv,
            "requested": {
                "package": pkg_spec,
                "verify_import": module_name,
                "scope": scope,
                "upgrade": upgrade,
                "pre_release": pre_release,
                "index_url": index_url,
                "extra_index_url": extra_index_url,
            },
            "initial_probe": initial_probe,
            "attempts": attempts,
            "probe": final_probe,
            "limitations": limitations,
        },
        indent=2,
    )


def _install_torch_runtime_tool(arguments: dict[str, Any], *, budget_seconds: Optional[float] = None) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    requested_version = str(args.get("version") or TORCH_DEFAULT_VERSION).strip() or TORCH_DEFAULT_VERSION
    variant_raw = str(args.get("variant") or "default").strip().lower()
    if variant_raw in {"auto", "gpu", "cuda"}:
        variant = "auto"
    elif variant_raw in {"default", "pypi"}:
        variant = "default"
    elif variant_raw == "cpu":
        variant = "cpu"
    elif re.fullmatch(r"cu\d+", variant_raw or ""):
        variant = variant_raw
    else:
        variant = "default"
    scope = str(args.get("scope") or "auto").strip().lower()
    if scope not in {"auto", "system", "user"}:
        scope = "auto"
    verify_only = _as_bool(args.get("verify_only"), False)
    upgrade = _as_bool(args.get("upgrade"), True)
    force = _as_bool(args.get("force"), False)

    timeout_seconds = _bounded_timeout(
        _as_float(args.get("timeout_seconds"), 900.0),
        budget_seconds,
    )
    in_venv = bool((os.getenv("VIRTUAL_ENV") or "").strip()) or (sys.prefix != getattr(sys, "base_prefix", sys.prefix))

    initial_probe = _probe_torch_runtime(timeout_seconds=min(25.0, timeout_seconds))
    if verify_only:
        return json.dumps(
            {
                "status": "PASS" if initial_probe.get("installed") else "FAIL",
                "action": "verify_torch_runtime",
                "python_executable": sys.executable,
                "in_virtualenv": in_venv,
                "probe": initial_probe,
            },
            indent=2,
        )
    if not _truthy_env("PMX_ALLOW_RUNTIME_PIP_INSTALL", default=False):
        return json.dumps(
            {
                "status": "FAIL",
                "action": "install_torch_runtime",
                "error": (
                    "Runtime pip installation is disabled by policy. "
                    "Set PMX_ALLOW_RUNTIME_PIP_INSTALL=1 to enable explicitly."
                ),
            },
            indent=2,
        )

    auto_detection: dict[str, Any] = {"status": "skipped"}
    desired_variant = variant
    if desired_variant == "auto":
        auto_detection = _detect_nvidia_gpu(timeout_seconds=min(2.0, timeout_seconds))
        desired_variant = TORCH_DEFAULT_CUDA_VARIANT if auto_detection.get("gpu_detected") else "default"

    def _cuda_version_from_variant(v: str) -> Optional[str]:
        m = re.fullmatch(r"cu(\d+)", str(v or "").strip().lower())
        if not m:
            return None
        digits = m.group(1)
        if len(digits) == 3:
            return f"{digits[0:2]}.{digits[2:]}"
        if len(digits) == 4:
            return f"{digits[0:2]}.{digits[2:]}"
        return None

    def _variant_satisfied(probe: dict[str, Any], want: str) -> bool:
        if not probe.get("installed"):
            return False
        want_norm = str(want or "default").strip().lower() or "default"
        if want_norm == "default":
            return True
        cuda_text = str(probe.get("cuda") or "").strip()
        if want_norm == "cpu":
            return not cuda_text
        if want_norm.startswith("cu"):
            expected = _cuda_version_from_variant(want_norm)
            if expected and cuda_text:
                return cuda_text.startswith(expected)
            return bool(cuda_text)
        return True

    if initial_probe.get("installed") and _variant_satisfied(initial_probe, desired_variant) and not force:
        return json.dumps(
            {
                "status": "PASS",
                "action": "install_torch_runtime",
                "note": "torch already importable for this runtime; skipping install (set force=true to reinstall)",
                "python_executable": sys.executable,
                "in_virtualenv": in_venv,
                "requested": {
                    "version": requested_version,
                    "variant": variant,
                    "resolved_variant": desired_variant,
                    "scope": scope,
                    "upgrade": upgrade,
                    "force": force,
                },
                "auto_detection": auto_detection,
                "initial_probe": initial_probe,
                "attempts": [],
                "probe": initial_probe,
                "limitations": [],
            },
            indent=2,
        )

    pkg_spec = "torch"
    if requested_version and requested_version.lower() not in {"latest", "any"}:
        pkg_spec = f"torch=={requested_version}"

    def _index_url_for_variant(want: str) -> Optional[str]:
        want_norm = str(want or "").strip().lower()
        if want_norm == "cpu":
            return "https://download.pytorch.org/whl/cpu"
        if want_norm.startswith("cu"):
            return f"https://download.pytorch.org/whl/{want_norm}"
        return None

    def _pip_cmd(*, user_flag: bool, index_url: Optional[str]) -> list[str]:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        if upgrade:
            cmd.append("--upgrade")
        if user_flag:
            cmd.append("--user")
        if index_url:
            cmd.extend(["--index-url", index_url])
        cmd.append(pkg_spec)
        return cmd

    install_attempts: list[dict[str, Any]] = []
    use_user_requested = scope == "user"
    use_system_requested = scope == "system"
    user_fallback_allowed = scope == "auto" and not in_venv

    plan: list[tuple[bool, Optional[str], str, str]] = []
    primary_index = _index_url_for_variant(desired_variant)
    if use_user_requested:
        plan.append((True, primary_index, "user_scope_requested", desired_variant))
    else:
        plan.append((False, primary_index, "primary", desired_variant))
    if desired_variant != "cpu":
        plan.append((use_user_requested, _index_url_for_variant("cpu"), "cpu_fallback", "cpu"))
    if user_fallback_allowed:
        plan.append((True, primary_index, "auto_user_fallback", desired_variant))
        if desired_variant != "cpu":
            plan.append((True, _index_url_for_variant("cpu"), "auto_user_cpu_fallback", "cpu"))

    seen_cmd: set[str] = set()
    chosen_result: dict[str, Any] | None = None
    for use_user, index_url, label, attempt_variant in plan:
        if use_system_requested and use_user:
            continue
        cmd = _pip_cmd(user_flag=use_user, index_url=index_url)
        key = json.dumps(cmd, ensure_ascii=True)
        if key in seen_cmd:
            continue
        seen_cmd.add(key)

        row = _run_pip_install_attempt(cmd, timeout_seconds=timeout_seconds)
        row["attempt_label"] = label
        row["scope"] = "user" if use_user else "system_or_venv"
        row["variant"] = attempt_variant
        row["index_url"] = index_url or ""
        install_attempts.append(row)

        if row.get("status") == "PASS":
            chosen_result = row
            break

        if not user_fallback_allowed and not use_user_requested:
            err_text = f"{row.get('stdout_tail', '')}\n{row.get('stderr_tail', '')}"
            if _is_permission_error_text(err_text):
                break

    final_probe = _probe_torch_runtime(timeout_seconds=min(30.0, timeout_seconds))
    ok = bool(
        chosen_result
        and chosen_result.get("status") == "PASS"
        and final_probe.get("installed")
        and _variant_satisfied(final_probe, desired_variant)
    )
    status = "PASS" if ok else "FAIL"
    limitations: list[str] = []
    if not ok and not final_probe.get("installed"):
        limitations.append("torch import verification failed after install attempts")
    if not ok and final_probe.get("installed") and not _variant_satisfied(final_probe, desired_variant):
        limitations.append(f"torch build does not match requested variant: {desired_variant}")
    if not ok and use_system_requested:
        limitations.append("scope=system was requested; no --user fallback applied")

    return json.dumps(
        {
            "status": status,
            "action": "install_torch_runtime",
            "python_executable": sys.executable,
            "in_virtualenv": in_venv,
            "requested": {
                "version": requested_version,
                "variant": variant,
                "resolved_variant": desired_variant,
                "scope": scope,
                "upgrade": upgrade,
                "force": force,
            },
            "auto_detection": auto_detection,
            "initial_probe": initial_probe,
            "attempts": install_attempts,
            "probe": final_probe,
            "limitations": limitations,
        },
        indent=2,
    )


def _send_notification(channel: str, message: str) -> str:
    """Send a message via OpenClaw."""
    if not message.strip():
        return json.dumps({"error": "Empty message"})

    try:
        from utils.openclaw_cli import send_message, resolve_openclaw_targets

        targets = resolve_openclaw_targets(
            env_targets=os.getenv("OPENCLAW_TARGETS"),
            env_to=os.getenv("OPENCLAW_TO"),
            default_channel=channel,
        )
        if not targets:
            return json.dumps({"error": f"No targets configured for channel={channel}"})

        results = []
        for ch, to in targets:
            if ch and ch != channel:
                continue
            res = send_message(
                to=to, message=message,
                command=os.getenv("OPENCLAW_COMMAND", "openclaw"),
                channel=ch,
            )
            results.append({"to": to, "ok": res.ok, "channel": ch or "default"})

        _get_activity_logger().log_openclaw_event(
            channel=channel, event_type="notification_sent",
            payload={"message_preview": message[:100], "targets": len(results)},
        )
        return json.dumps({"sent": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_system_health_json() -> str:
    """Get system health as JSON for tool calls."""
    health = refresh_health()
    ollama_up = any(health.values())

    # Check OpenClaw gateway
    gateway_ok = False
    try:
        oc_status = subprocess.run(
            ["cmd", "/d", "/s", "/c", "openclaw", "status", "--json"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        gateway_ok = oc_status.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    model_stats = {}
    for name, spec in MODEL_REGISTRY.items():
        model_stats[name] = {
            "available": spec.available,
            "role": spec.role,
            "avg_latency_ms": round(spec.avg_latency_ms, 1),
            "success_count": spec.success_count,
            "error_count": spec.error_count,
        }

    return json.dumps({
        "ollama_up": ollama_up,
        "gateway_ok": gateway_ok,
        "profile": {
            "name": ACTIVE_PRODUCTION_PROFILE_NAME,
            "sla": ACTIVE_PRODUCTION_SLA,
        },
        "models": model_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


def _is_data_heavy_prompt(prompt: str) -> bool:
    text = (prompt or "").lower()
    return any(
        kw in text
        for kw in (
            "pnl",
            "gate",
            "audit",
            "profit",
            "trade",
            "win rate",
            "profit factor",
            "integrity",
            "objective",
            "reconcile",
        )
    )


def _bridge_force_tool_primer(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text:
        return False
    return any(
        (
            _is_trading_critical_prompt(text),
            _is_gate_execution_prompt(text),
            _is_runtime_setup_prompt(text),
            _is_data_heavy_prompt(text),
        )
    )


def _bridge_timeout_seconds(prompt: str, channel: Optional[str]) -> int:
    base = max(20, int(ORCHESTRATION_TIMEOUT_SECONDS_DEFAULT))
    chan = str(channel or "").strip().lower()
    interactive_channel = chan in {"whatsapp", "telegram", "discord"}

    # Keep interactive channels responsive by default.
    timeout_budget = min(base, 75) if interactive_channel else min(base, 90)
    # Preserve wider budget for trading-critical/gate operations.
    if _is_trading_critical_prompt(prompt) or _is_gate_execution_prompt(prompt):
        timeout_budget = min(base, 120)
    return max(20, int(timeout_budget))


def _is_gate_execution_prompt(prompt: str) -> bool:
    text = (prompt or "").lower()
    gate_tokens = (
        "production gate",
        "audit gate",
        "run gate",
        "gate check",
        "reconcile",
        "unlinked close",
        "close ids",
    )
    return any(tok in text for tok in gate_tokens)


def _is_runtime_setup_prompt(prompt: str) -> bool:
    text = (prompt or "").lower()
    runtime_tokens = (
        "install torch",
        "pip install torch",
        "setup torch",
        "install pytorch",
        "pip install ",
        "install package",
        "install python package",
        "dependency install",
        "install dependencies",
        "missing module named torch",
        "module not found",
        "modulenotfounderror",
    )
    if any(tok in text for tok in runtime_tokens):
        return True
    if "install " in text and any(tok in text for tok in ("pip", "python", "dependency", "package", "module", "library")):
        return True
    return False


def _runtime_scope_from_text(text: str) -> str:
    low = (text or "").lower()
    if "scope=user" in low or " user " in f" {low} ":
        return "user"
    if "scope=system" in low or "system-wide" in low or "system wise" in low or " system " in f" {low} ":
        return "system"
    return "auto"


def _extract_runtime_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = (message or "").strip()
    low = text.lower()
    if not text or not _is_runtime_setup_prompt(text):
        return None

    scope = _runtime_scope_from_text(low)
    verify_only = any(tok in low for tok in ("verify", "check", "confirm", "status only", "dry run"))

    if "torch" in low or "pytorch" in low:
        version = ""
        m = re.search(r"torch\s*==\s*([A-Za-z0-9\.\-\+_]+)", text, flags=re.IGNORECASE)
        if m:
            version = m.group(1).strip()
        variant = "cpu" if " cpu" in f" {low} " else "default"
        return (
            "install_torch_runtime",
            {
                "scope": scope,
                "verify_only": verify_only,
                "variant": variant,
                "upgrade": True,
                "version": version,
            },
        )

    pkg = ""
    verify_import = ""
    pip_match = re.search(
        r"pip\s+install\s+([A-Za-z0-9\._\-\[\]<>=!~,+@]+)",
        text,
        flags=re.IGNORECASE,
    )
    if pip_match:
        pkg = pip_match.group(1).strip()
    else:
        pkg_match = re.search(
            r"install(?:\s+python)?\s+package\s+([A-Za-z0-9\._\-\[\]<>=!~,+@]+)",
            text,
            flags=re.IGNORECASE,
        )
        if pkg_match:
            pkg = pkg_match.group(1).strip()
        else:
            install_match = re.search(
                r"\binstall\s+([A-Za-z][A-Za-z0-9\._\-]{1,80})\b",
                text,
                flags=re.IGNORECASE,
            )
            if install_match:
                candidate = install_match.group(1).strip()
                if candidate.lower() not in {"python", "package", "dependency", "dependencies", "module", "library"}:
                    pkg = candidate

    verify_match = re.search(r"verify[_\s-]*import\s*[:=]?\s*([A-Za-z0-9_\.]+)", text, flags=re.IGNORECASE)
    if verify_match:
        verify_import = verify_match.group(1).strip()

    if not pkg:
        return None

    args: dict[str, Any] = {
        "package": pkg,
        "scope": scope,
        "verify_only": verify_only,
        "upgrade": True,
    }
    if verify_import:
        args["verify_import"] = verify_import
    return ("install_python_package", args)


def _extract_source_fast_path_paths(message: str) -> list[str]:
    text = str(message or "")
    candidates = re.findall(r"[A-Za-z0-9_./\\-]+\.(?:py|yml|yaml|json|md|toml)", text)
    normalized: list[str] = []
    for row in candidates:
        rel = _normalize_rel_path(row)
        if rel and _is_allowed_source_rel_path(rel):
            normalized.append(rel)
    return list(dict.fromkeys(normalized))[:6]


def _extract_source_inspection_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = str(message or "").strip()
    if not text:
        return None
    low = text.lower()
    if not _text_contains_any(low, SOURCE_FAST_PATH_TRIGGER_TOKENS):
        return None
    return (
        "inspect_source_code",
        {
            "query": text,
            "paths": _extract_source_fast_path_paths(text),
            "max_files": min(4, SOURCE_READ_MAX_FILES_CAP),
            "max_chars": min(12000, SOURCE_READ_MAX_CHARS_CAP),
        },
    )


def _extract_review_changed_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = str(message or "").strip()
    if not text:
        return None
    low = text.lower()
    if _text_contains_any(low, REVIEW_CHANGED_FAST_PATH_BLOCKED_TOKENS):
        return None
    if not _text_contains_any(low, REVIEW_CHANGED_FAST_PATH_TRIGGER_TOKENS):
        return None

    args: dict[str, Any] = {
        "include_untracked": True,
    }
    paths = _extract_source_fast_path_paths(text)
    if paths:
        args["paths"] = paths

    ref_match = re.search(r"\b(?:against|vs\.?|versus|base)\s+([A-Za-z0-9._/\-]+)\b", text, flags=re.IGNORECASE)
    if ref_match:
        args["base_ref"] = ref_match.group(1).strip()
    return ("review_changed_files", args)


def _is_repo_status_prompt(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    if not _text_contains_any(text, REPO_STATUS_SUBJECT_TOKENS):
        return False
    return _text_contains_any(text, REPO_STATUS_ACTION_TOKENS)


def _extract_repo_status_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    if not _is_repo_status_prompt(message):
        return None
    return ("check_github_repo_status", {})


def _extract_quant_window(message: str) -> Optional[int]:
    text = str(message or "")
    patterns = (
        r"\blast\s+(\d{1,5})\b",
        r"\brecent\s+(\d{1,5})\b",
        r"\bwindow\s*[:=]?\s*(\d{1,5})\b",
        r"\bentries?\s*[:=]?\s*(\d{1,5})\b",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _extract_quant_validation_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = str(message or "").strip()
    if not text:
        return None
    low = text.lower()
    if _text_contains_any(low, QUANT_VALIDATION_FAST_PATH_BLOCKED_TOKENS):
        return None
    if not _text_contains_any(low, QUANT_VALIDATION_FAST_PATH_TRIGGER_TOKENS):
        return None

    args: dict[str, Any] = {}
    window = _extract_quant_window(text)
    if window is not None:
        args["window"] = max(1, min(10000, window))

    if "95%" in low or "95 % " in f" {low} ":
        args["red_gate_pct"] = 95.0
    if "90%" in low or "90 % " in f" {low} ":
        args["warn_gate_pct"] = 90.0

    return ("check_quant_validation_headroom", args)


def _extract_test_log_path(message: str) -> str:
    text = str(message or "")
    candidates = re.findall(r"[A-Za-z0-9_./\\-]+\.(?:log|txt|out|xml)", text, flags=re.IGNORECASE)
    for candidate in candidates:
        cleaned = str(candidate).strip().strip("'\"")
        if cleaned:
            return cleaned
    return ""


def _extract_test_failure_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = str(message or "").strip()
    if not text:
        return None
    low = text.lower()
    if not _text_contains_any(low, TEST_FAILURE_FAST_PATH_TRIGGER_TOKENS):
        return None

    args: dict[str, Any] = {}
    log_path = _extract_test_log_path(text)
    if log_path:
        args["log_path"] = log_path
    elif "failed " in low or "\n" in text or text.startswith("FAILED "):
        args["output_text"] = text
    else:
        return None
    return ("summarize_test_failure", args)


def _extract_gate_triage_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = str(message or "").strip()
    if not text:
        return None
    low = text.lower()
    triggered = _text_contains_any(low, GATE_TRIAGE_FAST_PATH_TRIGGER_TOKENS) or (
        "gate" in low and any(token in low for token in ("fail", "failed", "failure", "decompose", "triage"))
    )
    if not triggered:
        return None

    args: dict[str, Any] = {"force_refresh": False}
    json_path = ""
    for candidate in re.findall(r"[A-Za-z0-9_./\\-]+\.json", text, flags=re.IGNORECASE):
        cleaned = str(candidate).strip().strip("'\"")
        if "gate" in cleaned.lower() or "audit" in cleaned.lower():
            json_path = cleaned
            break
    if json_path:
        args["gate_artifact"] = json_path
    return ("triage_gate_failure", args)


def _infer_web_search_topic(message: str) -> str:
    low = str(message or "").lower()
    if _text_contains_any(
        low,
        (
            "stock",
            "ticker",
            "market",
            "price",
            "crypto",
            "forex",
            "bond",
            "etf",
            "economy",
            "earnings",
        ),
    ):
        return "finance"
    if _text_contains_any(
        low,
        (
            "news",
            "latest",
            "today",
            "yesterday",
            "breaking",
            "headline",
            "update",
        ),
    ):
        return "news"
    return "general"


def _extract_web_search_query(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        return ""
    patterns = (
        r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|kindly\s+)?(?:look\s*up|lookup|find|search|research|check|browse)\s+(?:it\s+)?(?:online|on\s+the\s+web|the\s+web|internet)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|kindly\s+)?(?:online|web|internet)\s+(?:search|lookup|look\s*up|research)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:web\s+search)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:search\s+the\s+web)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:search\s+online)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:search\s+internet)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:internet\s+search)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:web\s+lookup|online\s+lookup|internet\s+lookup)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:look\s*up|lookup)\s+(?:online|on\s+the\s+web)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:browse\s+the\s+web|browse\s+internet|research\s+online|check\s+online)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:find\s+online)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:find\s+on\s+the\s+web)(?:\s+for)?\s*[:\-]?\s*",
        r"^\s*(?:please\s+)?(?:google|duckduckgo|wikipedia|tavily)(?:\s+for)?\s*[:\-]?\s*",
    )
    query = text
    for pattern in patterns:
        stripped = re.sub(pattern, "", query, count=1, flags=re.IGNORECASE).strip()
        if stripped != query:
            query = stripped
            break
    return query.strip(" \t\r\n?:,-")


def _extract_web_search_fast_path_request(message: str) -> Optional[tuple[str, dict[str, Any]]]:
    text = str(message or "").strip()
    if not text:
        return None
    low = text.lower()
    if _text_contains_any(low, WEB_SEARCH_FAST_PATH_BLOCKED_TOKENS):
        return None
    if not _text_contains_any(low, WEB_SEARCH_FAST_PATH_TRIGGER_TOKENS):
        return None

    query = _extract_web_search_query(text)
    if not query:
        return None

    args: dict[str, Any] = {
        "query": query,
        "allow_fallback": True,
    }
    topic = _infer_web_search_topic(query)
    if topic != "general":
        args["topic"] = topic

    max_results = None
    max_match = re.search(
        r"\b(?:top|max(?:imum)?\s+results?)\s*[:=]?\s*(\d{1,2})\b",
        low,
        flags=re.IGNORECASE,
    )
    if max_match:
        try:
            max_results = int(max_match.group(1))
        except Exception:
            max_results = None
    if max_results is not None:
        args["max_results"] = max(1, min(10, max_results))

    provider_order: list[str] = []
    for provider in ("tavily", "duckduckgo", "wikipedia"):
        if re.search(rf"\b{provider}\b", low):
            provider_order.append(provider)
    if provider_order:
        args["providers"] = ",".join(dict.fromkeys(provider_order))

    return ("search_web_tavily", args)


def _summarize_runtime_tool_result(tool_name: str, raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return f"{tool_name}: no output"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 700)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 700)

    status = str(parsed.get("status", "UNKNOWN")).upper()
    action = str(parsed.get("action", tool_name))
    requested = parsed.get("requested") if isinstance(parsed.get("requested"), dict) else {}
    probe = parsed.get("probe") if isinstance(parsed.get("probe"), dict) else {}
    attempts = parsed.get("attempts") if isinstance(parsed.get("attempts"), list) else []
    limitations = parsed.get("limitations") if isinstance(parsed.get("limitations"), list) else []
    module = str((requested or {}).get("verify_import") or "")
    package = str((requested or {}).get("package") or (requested or {}).get("version") or "")
    version = str((probe or {}).get("version") or "")

    lines = [
        f"{action}: {status}",
        f"package={package or 'n/a'} module={module or 'n/a'}",
        f"attempts={len(attempts)} version={version or 'unknown'}",
    ]
    if limitations:
        lines.append("limitations: " + "; ".join(str(x) for x in limitations[:2]))
    return _truncate_progress_text("\n".join(lines), 900)


def _summarize_repo_status_tool_result(raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "repo status: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    report = parsed.get("report") if isinstance(parsed.get("report"), dict) else {}
    if not report and isinstance(parsed.get("checks"), dict):
        report = parsed

    status = str((report or {}).get("status") or parsed.get("status") or "UNKNOWN").upper()
    checks = report.get("checks") if isinstance(report, dict) and isinstance(report.get("checks"), dict) else {}
    local = checks.get("local") if isinstance(checks.get("local"), dict) else {}
    tracking = checks.get("tracking") if isinstance(checks.get("tracking"), dict) else {}
    issues = report.get("issues") if isinstance(report.get("issues"), list) else []
    slug = str(checks.get("github_slug") or "").strip()
    branch = str(local.get("branch") or "unknown")
    head_short = str(local.get("head_short") or local.get("head") or "unknown")
    upstream = str(tracking.get("upstream") or "none")
    ahead = tracking.get("ahead")
    behind = tracking.get("behind")
    dirty = bool(local.get("dirty"))

    lines = [
        f"repo status: {status}" + (f" | {slug}" if slug else ""),
        f"branch={branch} head={head_short} worktree={'dirty' if dirty else 'clean'}",
        f"tracking={upstream} ahead={ahead if ahead is not None else 'n/a'} behind={behind if behind is not None else 'n/a'}",
    ]
    if issues:
        lines.append("issues: " + "; ".join(str(x) for x in issues[:3]))
    return _truncate_progress_text("\n".join(lines), 900)


def _summarize_web_search_tool_result(raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "web search: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    status = str(parsed.get("status") or "UNKNOWN").upper()
    provider = str(parsed.get("provider") or "none")
    attempts = parsed.get("attempts") if isinstance(parsed.get("attempts"), list) else []
    answer = str(parsed.get("answer") or "").strip()
    results = parsed.get("results") if isinstance(parsed.get("results"), list) else []
    reason = str(parsed.get("message") or parsed.get("error") or "").strip()

    lines = [f"web search: {status} | provider={provider} | attempts={len(attempts)}"]
    if status != "PASS":
        lines.append(f"reason: {reason or 'unknown_error'}")
        return _truncate_progress_text("\n".join(lines), 900)

    if answer:
        lines.append("answer: " + _truncate_progress_text(answer, 260))
    for idx, row in enumerate(results[:3], start=1):
        if not isinstance(row, dict):
            continue
        title = _truncate_progress_text(str(row.get("title") or "result"), 130)
        url = _truncate_progress_text(str(row.get("url") or "").strip(), 160)
        lines.append(f"{idx}. {title}{f' - {url}' if url else ''}")
    return _truncate_progress_text("\n".join(lines), 900)


def _summarize_quant_validation_tool_result(raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "quant validation: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    status = str(parsed.get("status") or "UNKNOWN").upper()
    classification = str(parsed.get("classification") or "UNKNOWN").upper()
    summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
    message = str(parsed.get("message") or parsed.get("error") or "").strip()
    fail_count = summary.get("fail_count", "n/a")
    total = summary.get("total", "n/a")
    fail_rate = summary.get("fail_rate_pct", "n/a")
    headroom = summary.get("headroom_to_red_gate_pct", "n/a")
    red_gate = summary.get("red_gate_pct", "n/a")

    lines = [
        f"quant validation: {status} | class={classification}",
        f"fail={fail_count}/{total} ({fail_rate}%), headroom_to_{red_gate}%={headroom}%",
    ]
    if message:
        lines.append("note: " + _truncate_progress_text(message, 220))
    per_ticker = summary.get("per_ticker") if isinstance(summary.get("per_ticker"), list) else []
    if per_ticker:
        top = []
        for row in per_ticker[:3]:
            if not isinstance(row, dict):
                continue
            top.append(
                f"{row.get('ticker', '?')}:{row.get('fail_count', '?')}/{row.get('total', '?')}({row.get('fail_rate_pct', '?')}%)"
            )
        if top:
            lines.append("tickers: " + ", ".join(top))
    return _truncate_progress_text("\n".join(lines), 900)


@dataclass(frozen=True)
class BridgeFastPathSpec:
    key: str
    extractor: Callable[[str], Optional[tuple[str, dict[str, Any]]]]
    summarizer: Callable[[str, str], str]
    progress_label: str
    budget_seconds: float


def _summarize_runtime_fast_path(tool_name: str, raw_result: str) -> str:
    return _summarize_runtime_tool_result(tool_name, raw_result)


def _summarize_source_fast_path(_tool_name: str, raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "source evidence: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    status = str(parsed.get("status", "UNKNOWN")).upper()
    selected = parsed.get("selected_paths") if isinstance(parsed.get("selected_paths"), list) else []
    citations = parsed.get("citations") if isinstance(parsed.get("citations"), list) else []
    lines = [
        f"source evidence: {status}",
        f"files={len(selected)} citations={len(citations)}",
    ]
    if selected:
        lines.append("paths: " + ", ".join(str(x) for x in selected[:3]))
    if citations:
        lines.append("cite: " + ", ".join(str(x) for x in citations[:5]))
    return _truncate_progress_text("\n".join(lines), 900)


def _summarize_review_changed_fast_path(_tool_name: str, raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "change review: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    status = str(parsed.get("status") or "UNKNOWN").upper()
    state = str(parsed.get("worktree_state") or "unknown")
    total = parsed.get("total_changed_files", "n/a")
    files = parsed.get("changed_files") if isinstance(parsed.get("changed_files"), list) else []
    tags = parsed.get("tag_counts") if isinstance(parsed.get("tag_counts"), dict) else {}

    lines = [
        f"change review: {status} | worktree={state} | files={total}",
    ]
    if files:
        preview = ", ".join(str(row.get("path") or "?") for row in files[:4] if isinstance(row, dict))
        if preview:
            lines.append("paths: " + preview)
    if tags:
        top_tags = ", ".join(f"{k}:{v}" for k, v in list(tags.items())[:4])
        if top_tags:
            lines.append("tags: " + top_tags)
    return _truncate_progress_text("\n".join(lines), 900)


def _summarize_repo_fast_path(_tool_name: str, raw_result: str) -> str:
    return _summarize_repo_status_tool_result(raw_result)


def _summarize_web_search_fast_path(_tool_name: str, raw_result: str) -> str:
    return _summarize_web_search_tool_result(raw_result)


def _summarize_quant_validation_fast_path(_tool_name: str, raw_result: str) -> str:
    return _summarize_quant_validation_tool_result(raw_result)


def _summarize_test_failure_fast_path(_tool_name: str, raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "test failure: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    status = str(parsed.get("status") or "UNKNOWN").upper()
    total = parsed.get("total_failures", "n/a")
    error_types = parsed.get("error_types") if isinstance(parsed.get("error_types"), list) else []
    failures = parsed.get("failures") if isinstance(parsed.get("failures"), list) else []
    lines = [f"test failure: {status} | failures={total}"]
    if failures:
        lines.append("nodes: " + ", ".join(str(row.get("nodeid") or "?") for row in failures[:3] if isinstance(row, dict)))
    if error_types:
        lines.append("errors: " + ", ".join(f"{row.get('type')}:{row.get('count')}" for row in error_types[:3] if isinstance(row, dict)))
    return _truncate_progress_text("\n".join(lines), 900)


def _summarize_gate_triage_fast_path(_tool_name: str, raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "gate triage: FAIL | empty result"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 900)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 900)

    status = str(parsed.get("status") or "UNKNOWN").upper()
    gate_status = str(parsed.get("gate_status") or "UNKNOWN").upper()
    reason = str(parsed.get("phase3_reason") or "unknown")
    failed = parsed.get("failed_components") if isinstance(parsed.get("failed_components"), list) else []
    lines = [f"gate triage: {status} | gate={gate_status} | reason={reason}"]
    if failed:
        lines.append("failed: " + ", ".join(str(x) for x in failed[:4]))
    return _truncate_progress_text("\n".join(lines), 900)


_BRIDGE_FAST_PATH_SPECS: tuple[BridgeFastPathSpec, ...] = (
    BridgeFastPathSpec(
        key="runtime",
        extractor=_extract_runtime_fast_path_request,
        summarizer=_summarize_runtime_fast_path,
        progress_label="runtime_fast_path",
        budget_seconds=1200.0,
    ),
    BridgeFastPathSpec(
        key="change_review",
        extractor=_extract_review_changed_fast_path_request,
        summarizer=_summarize_review_changed_fast_path,
        progress_label="change_review_fast_path",
        budget_seconds=35.0,
    ),
    BridgeFastPathSpec(
        key="gate_triage",
        extractor=_extract_gate_triage_fast_path_request,
        summarizer=_summarize_gate_triage_fast_path,
        progress_label="gate_triage_fast_path",
        budget_seconds=45.0,
    ),
    BridgeFastPathSpec(
        key="test_failure",
        extractor=_extract_test_failure_fast_path_request,
        summarizer=_summarize_test_failure_fast_path,
        progress_label="test_failure_fast_path",
        budget_seconds=20.0,
    ),
    BridgeFastPathSpec(
        key="repo",
        extractor=_extract_repo_status_fast_path_request,
        summarizer=_summarize_repo_fast_path,
        progress_label="repo_fast_path",
        budget_seconds=45.0,
    ),
    BridgeFastPathSpec(
        key="source",
        extractor=_extract_source_inspection_fast_path_request,
        summarizer=_summarize_source_fast_path,
        progress_label="source_fast_path",
        budget_seconds=50.0,
    ),
    BridgeFastPathSpec(
        key="quant",
        extractor=_extract_quant_validation_fast_path_request,
        summarizer=_summarize_quant_validation_fast_path,
        progress_label="quant_fast_path",
        budget_seconds=45.0,
    ),
    BridgeFastPathSpec(
        key="web",
        extractor=_extract_web_search_fast_path_request,
        summarizer=_summarize_web_search_fast_path,
        progress_label="web_search_fast_path",
        budget_seconds=45.0,
    ),
)


def _has_registered_fast_path_match(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    for spec in _BRIDGE_FAST_PATH_SPECS:
        try:
            if spec.extractor(text):
                return True
        except Exception:
            continue
    return False


def _run_bridge_fast_path(
    *,
    spec: BridgeFastPathSpec,
    message: str,
    channel: Optional[str],
    reply_to: Optional[str],
    activity: Any,
) -> Optional[str]:
    fast_path = spec.extractor(message)
    if not fast_path:
        return None
    tool_name, tool_args = fast_path
    arg_keys = sorted(tool_args.keys()) if isinstance(tool_args, dict) else []
    activity.log_openclaw_event(
        channel=channel or "unknown",
        event_type=f"bridge_{spec.key}_fast_path_start",
        payload={"tool": tool_name, "arg_keys": arg_keys},
    )
    if channel and reply_to and PROGRESS_UPDATES_DEFAULT:
        _deliver_response(
            channel=channel,
            to=reply_to,
            message=f"[progress 0.1s] {spec.progress_label} | executing {tool_name}",
        )
    tool_raw = execute_tool_call(tool_name, tool_args, allow_cache=False, budget_seconds=spec.budget_seconds)
    response = spec.summarizer(tool_name, tool_raw)
    if reply_to and channel:
        _deliver_response(channel=channel, to=reply_to, message=response)
    activity.log_openclaw_event(
        channel=channel or "unknown",
        event_type=f"bridge_{spec.key}_fast_path_complete",
        payload={"tool": tool_name, "response_preview": response[:120]},
    )
    return response


def _is_status_fast_path_prompt(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    if _has_registered_fast_path_match(text):
        return False
    if _text_contains_any(text, STATUS_FAST_PATH_BLOCKED_TOKENS):
        return False
    return _text_contains_any(text, STATUS_FAST_PATH_TRIGGER_TOKENS)


def _summarize_health_tool_result(raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return "health: FAIL | empty health payload"
    try:
        parsed = json.loads(text)
    except Exception:
        return _truncate_progress_text(text, 700)
    if not isinstance(parsed, dict):
        return _truncate_progress_text(text, 700)

    ollama_up = bool(parsed.get("ollama_up"))
    gateway_ok = bool(parsed.get("gateway_ok"))
    profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
    profile_name = str(profile.get("name") or "unknown")
    models = parsed.get("models") if isinstance(parsed.get("models"), dict) else {}

    available_models = []
    unavailable_models = []
    for model_name, row in models.items():
        if isinstance(row, dict) and bool(row.get("available")):
            available_models.append(str(model_name))
        else:
            unavailable_models.append(str(model_name))

    status = "PASS" if (ollama_up and gateway_ok and not unavailable_models) else "DEGRADED"
    lines = [
        f"health: {status}",
        f"gateway={'up' if gateway_ok else 'down'} ollama={'up' if ollama_up else 'down'} profile={profile_name}",
        f"models_up={len(available_models)}/{len(models) or len(available_models)}",
    ]
    if unavailable_models:
        lines.append("models_down: " + ", ".join(unavailable_models[:4]))
    return _truncate_progress_text("\n".join(lines), 900)


def _is_trading_critical_prompt(prompt: str) -> bool:
    text = (prompt or "").lower()
    return any(kw in text for kw in TRADING_CRITICAL_KEYWORDS)


def _estimate_prompt_complexity(prompt: str) -> str:
    text = (prompt or "").lower()
    score = 0
    if len(text) > 1200:
        score += 2
    elif len(text) > 400:
        score += 1
    if any(k in text for k in ("adversarial", "optimization", "multi-step", "portfolio", "risk")):
        score += 2
    if any(k in text for k in ("explain", "summarize", "status", "health")):
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _runtime_plan(
    *,
    prompt: str,
    max_rounds: int,
    max_tool_calls: int,
    force_tool_primer: bool,
    subagent_workflow: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    complexity = _estimate_prompt_complexity(prompt)
    trading_critical = _is_trading_critical_prompt(prompt)
    data_heavy = _is_data_heavy_prompt(prompt)
    degraded = _system_degraded()
    sla_timeout_budget = _as_int(
        (ACTIVE_PRODUCTION_SLA or {}).get("target_timeout_budget_seconds", timeout_seconds),
        timeout_seconds,
    )

    rounds = max(1, int(max_rounds))
    calls = max(1, int(max_tool_calls))
    timeout_budget = max(20, int(timeout_seconds))
    if sla_timeout_budget > 0:
        timeout_budget = min(timeout_budget, max(20, sla_timeout_budget))
    use_subagent = bool(subagent_workflow)
    use_primer = bool(force_tool_primer)
    chat_num_predict = int(CHAT_NUM_PREDICT_DEFAULT)

    if complexity == "low":
        rounds = min(rounds, 1)
        calls = min(calls, 6)
        timeout_budget = min(timeout_budget, 90)
        chat_num_predict = min(chat_num_predict, 300)
    elif complexity == "medium":
        rounds = min(rounds, 2)
        calls = min(calls, 10)
        timeout_budget = min(timeout_budget, 140)
        chat_num_predict = min(chat_num_predict, 420)
    else:
        rounds = min(rounds, 3)
        calls = min(calls, 12)
        timeout_budget = min(timeout_budget, 180)
        chat_num_predict = min(chat_num_predict, 640)

    if degraded:
        # Fast-degrade under stress to avoid cascaded failures.
        use_subagent = False
        rounds = min(rounds, 1)
        calls = min(calls, 6)
        timeout_budget = min(timeout_budget, 75)
        chat_num_predict = min(chat_num_predict, 260)

    if trading_critical or data_heavy:
        use_primer = True
        chat_num_predict = min(chat_num_predict, 420)

    return {
        "complexity": complexity,
        "trading_critical": trading_critical,
        "data_heavy": data_heavy,
        "degraded": degraded,
        "max_rounds": rounds,
        "max_tool_calls": calls,
        "subagent_workflow": use_subagent,
        "force_tool_primer": use_primer,
        "timeout_seconds": timeout_budget,
        "chat_num_predict": max(180, int(chat_num_predict)),
    }


def _base_system_prompt() -> str:
    return (
        "You are Best-Anime, a quantitative portfolio optimization assistant for Bestman. "
        "You have access to reasoning tools (deepseek-r1 models) for chain-of-thought analysis, "
        "data tools for reading gate status/trade metrics and inspecting source code with citations, "
        "review tools for changed-file review, pytest failure summarization, and production gate triage, "
        "quant validation health/headroom tools for fail-rate monitoring, "
        "robust web search (Tavily with free-provider fallback) for external lookups, "
        "notification tools for sending "
        "messages via OpenClaw, runtime dependency tools (including torch and generic package install/verification), "
        "system health checks, and sub-agent batch execution. "
        "Use tools aggressively for evidence-backed answers and execute sub-agent batches for complex tasks. "
        "Prefer tool-backed conclusions over unsupported assumptions. "
        "When claims depend on repository code, call inspect_source_code and cite path:line evidence. "
        "For trading-critical prompts, rely on gate/metrics evidence before recommending actions. "
        "Be concise, data-driven, and profit-focused."
    )


def _openclaw_prompt_template_for_message(message: str) -> str:
    root = _OPENCLAW_PROMPT_TEMPLATES if isinstance(_OPENCLAW_PROMPT_TEMPLATES, dict) else {}
    section = root.get("openclaw_prompt_templates", {}) if isinstance(root.get("openclaw_prompt_templates"), dict) else {}
    templates = section.get("templates", {}) if isinstance(section.get("templates"), dict) else {}
    default_template = str(templates.get("default", "") or "").strip()
    gate_template = str(templates.get("production_gate", "") or "").strip()
    runtime_template = str(templates.get("runtime_setup", "") or "").strip()

    chunks: list[str] = []
    if default_template:
        chunks.append(default_template)
    if _is_gate_execution_prompt(message):
        if gate_template:
            chunks.append(gate_template)
        else:
            chunks.append(
                "For production gate or reconciliation requests, call run_production_audit_gate "
                "instead of generic exec/shell style commands."
            )
    if _is_runtime_setup_prompt(message):
        if runtime_template:
            chunks.append(runtime_template)
        else:
            chunks.append(
                "For runtime dependency setup requests, call install_torch_runtime or install_python_package and report "
                "status, attempts, and verification results."
            )
    return "\n\n".join(c for c in chunks if c.strip())


def _auto_subagent_tasks(prompt: str) -> list[dict[str, str]]:
    base = (prompt or "").strip()
    if not base:
        return []
    return [
        {
            "task": "Extract the exact objective, constraints, and success criteria from the request.",
            "context": base,
            "complexity": "fast",
        },
        {
            "task": "Identify key execution risks and checks required before action.",
            "context": base,
            "complexity": "auto",
        },
    ]


def _pick_subagent_model(task: str, complexity: str) -> str:
    if _system_degraded():
        return get_best_model_for_role("reasoning") or "deepseek-r1:8b"

    c = (complexity or "auto").strip().lower()
    if c == "fast":
        return get_best_model_for_role("reasoning") or "deepseek-r1:8b"
    if c == "deep":
        heavy = MODEL_REGISTRY.get("deepseek-r1:32b")
        if heavy and heavy.avg_latency_ms > SUBAGENT_DEEP_LATENCY_CUTOFF_MS:
            return get_best_model_for_role("reasoning") or "deepseek-r1:8b"
        return get_best_model_for_role("heavy_reasoning") or "deepseek-r1:32b"

    text = (task or "").lower()
    deep_keywords = (
        "adversarial",
        "audit",
        "portfolio",
        "optimization",
        "multi-step",
        "complex",
        "risk",
        "validation",
    )
    if any(k in text for k in deep_keywords):
        return get_best_model_for_role("heavy_reasoning") or "deepseek-r1:32b"
    return get_best_model_for_role("reasoning") or "deepseek-r1:8b"


def _run_sub_agent_batch(tasks: Any, default_context: str = "") -> str:
    """Execute sub-agent tasks and return structured results."""
    if not isinstance(tasks, list) or not tasks:
        return json.dumps({"error": "tasks must be a non-empty list"})

    results: list[dict[str, Any]] = []
    max_tasks = SUBAGENT_MAX_TASKS_DEFAULT
    if _system_degraded():
        max_tasks = 1

    for idx, raw in enumerate(tasks[:max_tasks], start=1):
        if isinstance(raw, str):
            task = raw.strip()
            context = default_context
            complexity = "auto"
        elif isinstance(raw, dict):
            task = str(raw.get("task") or "").strip()
            context = str(raw.get("context") or default_context or "")
            complexity = str(raw.get("complexity") or "auto")
        else:
            continue

        if not task:
            continue

        model = _pick_subagent_model(task=task, complexity=complexity)
        output = _run_reasoning_model(
            model=model,
            task=task,
            context=context,
            max_predict=220,
            timeout_seconds=40.0 if model.endswith(":8b") else 60.0,
        )
        results.append(
            {
                "index": idx,
                "model": model,
                "complexity": complexity,
                "task": task,
                "output": output[:2500],
            }
        )

    return json.dumps(
        {
            "sub_agent_count": len(results),
            "results": results,
        },
        indent=2,
    )


def _truncate_progress_text(text: str, max_chars: int = PROGRESS_MAX_MESSAGE_CHARS) -> str:
    payload = str(text or "").strip()
    if len(payload) <= max_chars:
        return payload
    return payload[: max_chars - 3].rstrip() + "..."


def _tool_result_status_line(tool_name: str, tool_result: str) -> str:
    err = _tool_error_message(tool_result)
    if err:
        return f"{tool_name}: FAIL ({_truncate_progress_text(err, 90)})"
    try:
        parsed = json.loads(tool_result)
    except Exception:
        parsed = {}
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "")).strip().upper()
        if status:
            return f"{tool_name}: {status}"
    return f"{tool_name}: PASS"


class _ProgressReporter:
    def __init__(
        self,
        *,
        activity_logger: Any,
        channel: Optional[str],
        to: Optional[str],
        enabled: bool,
        min_interval_seconds: float,
        start_time: float,
    ):
        self._activity = activity_logger
        self._channel = (channel or "").strip()
        self._to = (to or "").strip()
        self._enabled = bool(enabled)
        self._min_interval_seconds = max(1.0, float(min_interval_seconds))
        self._start_time = float(start_time)
        self._last_emit_ts = 0.0
        self._last_message = ""

    def emit(self, stage: str, detail: str = "", *, force: bool = False) -> None:
        stage_text = _truncate_progress_text(stage, 100)
        detail_text = _truncate_progress_text(detail, 160)
        elapsed = max(0.0, time.time() - self._start_time)

        payload = {
            "stage": stage_text,
            "detail": detail_text,
            "elapsed_seconds": round(elapsed, 1),
        }
        try:
            self._activity.log_openclaw_event(
                channel=self._channel or "unknown",
                event_type="orchestration_progress",
                payload=payload,
            )
        except Exception:
            pass

        if not self._enabled or not self._channel or not self._to:
            return

        now = time.monotonic()
        if not force and (now - self._last_emit_ts) < self._min_interval_seconds:
            return

        base = f"[progress {elapsed:.1f}s] {stage_text}"
        if detail_text:
            base = f"{base} | {detail_text}"
        message = _truncate_progress_text(base, PROGRESS_MAX_MESSAGE_CHARS)
        if not force and message == self._last_message:
            return
        try:
            _deliver_response(channel=self._channel, to=self._to, message=message)
            self._last_emit_ts = now
            self._last_message = message
        except Exception as exc:
            logger.warning("Progress update delivery failed: %s", exc)


def _build_precomputed_tool_context(
    prompt: str,
    subagent_workflow: bool,
    progress: Optional[_ProgressReporter] = None,
) -> str:
    """
    Pre-run high-value tool calls so qwen starts with concrete context.
    This increases tool-backed responses even on the first round.
    """
    context_parts: list[str] = []
    trading_critical = _is_trading_critical_prompt(prompt)

    base_calls: list[tuple[str, dict[str, Any]]] = [("check_system_health", {})]
    if _is_data_heavy_prompt(prompt) or trading_critical:
        base_calls.extend(
            [
                ("query_trade_metrics", {"metric": "summary"}),
                ("query_trade_metrics", {"metric": "objective"}),
                ("read_gate_status", {"gate": "all"}),
            ]
        )
    if _is_gate_execution_prompt(prompt):
        base_calls.append(
            (
                "run_production_audit_gate",
                {"reconcile_apply": False, "max_files": 200},
            )
        )
    if subagent_workflow and not _system_degraded():
        base_calls.append(
            (
                "run_sub_agent_batch",
                {"tasks": _auto_subagent_tasks(prompt), "default_context": prompt},
            )
        )

    for tool_name, args in base_calls:
        if progress:
            progress.emit("Priming tools", f"running {tool_name}")
        try:
            result = execute_tool_call(tool_name, args)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        if progress:
            progress.emit("Priming tools", _tool_result_status_line(tool_name, result))
        context_parts.append(f"[{tool_name}]\n{result[:3000]}")

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Orchestration engine
# ---------------------------------------------------------------------------

def orchestrate(
    prompt: str,
    max_rounds: int = MAX_ROUNDS_DEFAULT,
    system_prompt: Optional[str] = None,
    reply_channel: Optional[str] = None,
    reply_to: Optional[str] = None,
    force_tool_primer: bool = FORCE_TOOL_PRIMER_DEFAULT,
    subagent_workflow: bool = SUBAGENT_WORKFLOW_DEFAULT,
    max_tool_calls: int = MAX_TOOL_CALLS_DEFAULT,
    timeout_seconds: int = ORCHESTRATION_TIMEOUT_SECONDS_DEFAULT,
) -> str:
    """
    Use qwen3:8b as the orchestrator with tool-calling.

    qwen3 receives the prompt + tool definitions. If it calls tools,
    we execute them (potentially dispatching to deepseek-r1 models)
    and feed results back. This continues for up to max_rounds.

    Args:
        prompt: User prompt to process.
        max_rounds: Max tool-calling rounds.
        system_prompt: Override system prompt.
        reply_channel: OpenClaw channel to deliver result to (optional).
        reply_to: OpenClaw target to deliver result to (optional).

    Returns:
        Final response text.
    """
    _ensure_health_fresh()
    activity = _get_activity_logger()
    tools_called_log: list[str] = []
    t0_total = time.time()
    progress = _ProgressReporter(
        activity_logger=activity,
        channel=reply_channel,
        to=reply_to,
        enabled=bool(PROGRESS_UPDATES_DEFAULT),
        min_interval_seconds=PROGRESS_MIN_INTERVAL_SECONDS,
        start_time=t0_total,
    )

    # Pick orchestrator model (prefer qwen3:8b, fallback to any tool-capable)
    orch_model = get_best_model_for_role("orchestrator")
    if not orch_model:
        # No tool-capable model available; fall back to direct reasoning
        logger.warning("No orchestrator model available; falling back to direct reasoning")
        reasoning_model = get_best_model_for_role("reasoning")
        if reasoning_model:
            return _run_reasoning_model(model=reasoning_model, task=prompt)
        return "[ERROR] No LLM models available. Check Ollama: ollama list"

    plan = _runtime_plan(
        prompt=prompt,
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
        force_tool_primer=force_tool_primer,
        subagent_workflow=subagent_workflow,
        timeout_seconds=timeout_seconds,
    )
    max_rounds = int(plan["max_rounds"])
    max_tool_calls = int(plan["max_tool_calls"])
    timeout_seconds = int(plan["timeout_seconds"])
    force_tool_primer = bool(plan["force_tool_primer"])
    subagent_workflow = bool(plan["subagent_workflow"])
    chat_num_predict = int(plan["chat_num_predict"])
    trading_critical = bool(plan.get("trading_critical", False))
    progress.emit(
        "orchestration_started",
        (
            f"model={orch_model} rounds<={max_rounds} tools<={max_tool_calls} "
            f"timeout={timeout_seconds}s profile={ACTIVE_PRODUCTION_PROFILE_NAME}"
        ),
        force=True,
    )

    sys_content = system_prompt or _base_system_prompt()

    messages: list[dict] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt},
    ]
    startup_failure_hints = _recent_tool_failure_hints(limit=TOOL_FAILURE_HINTS_MAX)
    if startup_failure_hints:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Recent tool failures from earlier sessions (adapt call arguments, do not repeat them):\n"
                    f"{startup_failure_hints}"
                ),
            }
        )

    primed_context = ""
    if force_tool_primer:
        progress.emit("tool_primer", "building precomputed tool context")
        primed = _build_precomputed_tool_context(
            prompt=prompt,
            subagent_workflow=subagent_workflow,
            progress=progress,
        )
        if primed:
            primed_context = primed
            progress.emit("tool_primer", "precomputed context ready")
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Precomputed tool context is available below. Reuse it and still call extra tools if needed.\n\n"
                        + primed
                    ),
                }
            )

    content = ""
    total_tool_calls = 0
    repeated_tool_signatures: dict[str, int] = {}
    consecutive_tool_error_rounds = 0
    for round_num in range(max_rounds):
        elapsed = time.time() - t0_total
        remaining_budget = float(timeout_seconds) - elapsed
        if elapsed >= timeout_seconds:
            logger.warning(
                "Orchestration timeout reached (elapsed=%.1fs, budget=%ss). Returning best available result.",
                elapsed,
                timeout_seconds,
            )
            progress.emit("timeout_guard", f"elapsed={elapsed:.1f}s budget={timeout_seconds}s", force=True)
            break
        if remaining_budget <= 8.0:
            logger.warning(
                "Orchestration remaining budget too small (remaining=%.1fs). Stopping new rounds.",
                remaining_budget,
            )
            progress.emit("budget_guard", f"remaining={remaining_budget:.1f}s; stopping new rounds", force=True)
            break

        tools_enabled = total_tool_calls < max_tool_calls
        progress.emit(
            f"round_{round_num + 1}_start",
            f"tools_enabled={tools_enabled} used={total_tool_calls}/{max_tool_calls} remaining={remaining_budget:.1f}s",
        )
        payload = {
            "model": orch_model,
            "messages": messages,
            "tools": REASONING_TOOLS if tools_enabled else [],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": chat_num_predict},
        }

        t0_round = time.time()
        try:
            round_timeout = max(5.0, min(CHAT_ROUND_TIMEOUT_CAP_SECONDS, remaining_budget - 2.0))
            result = _ollama_post("/api/chat", payload, timeout=round_timeout)
            _record_model_stats(orch_model, (time.time() - t0_round) * 1000, True)
        except Exception as e:
            _record_model_stats(orch_model, (time.time() - t0_round) * 1000, False)
            logger.error("Orchestration round %d failed: %s", round_num + 1, e)
            progress.emit(f"round_{round_num + 1}_error", str(e), force=True)

            # Try fallback to direct reasoning
            fallback = get_best_model_for_role("reasoning")
            if fallback:
                logger.info("Falling back to direct reasoning with %s", fallback)
                progress.emit("fallback_reasoning", f"switching to {fallback}", force=True)
                remaining = max(8.0, float(timeout_seconds) - (time.time() - t0_total))
                if remaining <= 10.0:
                    content = content or f"[ERROR] Orchestration failed near timeout: {e}"
                    break
                return _run_reasoning_model(
                    model=fallback,
                    task=prompt,
                    max_predict=320,
                    timeout_seconds=min(12.0, remaining),
                    allow_fallback=False,
                )
            return f"[ERROR] Orchestration failed: {e}"

        message = result.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            progress.emit(
                f"round_{round_num + 1}_complete",
                f"no tool calls requested; finalizing response (tool_calls={total_tool_calls})",
            )
            activity.log_orchestration(
                prompt=prompt, final_response=content,
                rounds=round_num + 1,
                total_latency_ms=(time.time() - t0_total) * 1000,
                tools_called=tools_called_log, success=True,
            )
            break

        # Execute tool calls
        messages.append(message)
        tool_errors_this_round = 0
        tool_success_this_round = 0

        for tc in tool_calls:
            if total_tool_calls >= max_tool_calls:
                progress.emit("tool_budget_reached", f"tool_calls={total_tool_calls}/{max_tool_calls}", force=True)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Tool budget reached. Produce final answer from available evidence without further tool calls."
                        ),
                    }
                )
                break

            fn = tc.get("function", {})
            raw_tool_name = str(fn.get("name") or "").strip()
            tool_name = _canonical_tool_name(raw_tool_name)
            args_raw = fn.get("arguments", {})
            args, arg_error = _sanitize_tool_arguments(tool_name, args_raw)
            if arg_error:
                tool_errors_this_round += 1
                _record_tool_failure(tool_name or "unknown", args, arg_error)
                tool_result = _tool_error_json(
                    tool_name or "unknown",
                    arg_error,
                    arguments=args,
                    include_available=True,
                )
                tool_message = {
                    "role": "tool",
                    "name": raw_tool_name or tool_name or "unknown",
                    "content": tool_result[:4000],
                }
                if tc.get("id"):
                    tool_message["tool_call_id"] = tc.get("id")
                messages.append(tool_message)
                continue

            sig = _cache_key(tool_name, args)
            repeated_tool_signatures[sig] = repeated_tool_signatures.get(sig, 0) + 1
            if repeated_tool_signatures[sig] > TOOL_REPEAT_SIGNATURE_LIMIT:
                repeat_error = (
                    f"Repeated identical tool call blocked after {TOOL_REPEAT_SIGNATURE_LIMIT} attempts."
                )
                tool_errors_this_round += 1
                _record_tool_failure(tool_name, args, repeat_error)
                tool_result = _tool_error_json(tool_name, repeat_error, arguments=args)
                tool_message = {
                    "role": "tool",
                    "name": raw_tool_name or tool_name,
                    "content": tool_result[:4000],
                }
                if tc.get("id"):
                    tool_message["tool_call_id"] = tc.get("id")
                messages.append(tool_message)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You repeated the same tool call with same arguments. "
                            "Choose a different tool or change arguments."
                        ),
                    }
                )
                continue

            if raw_tool_name and raw_tool_name != tool_name:
                logger.info(
                    "[round %d] Tool alias: %s -> %s",
                    round_num + 1,
                    raw_tool_name,
                    tool_name,
                )
            logger.info("[round %d] Tool call: %s(%s)", round_num + 1, tool_name, json.dumps(args)[:80])
            progress.emit(
                f"round_{round_num + 1}_tool_start",
                f"{tool_name} call {total_tool_calls + 1}/{max_tool_calls}",
            )
            t0_tool = time.time()
            budget_left = max(6.0, float(timeout_seconds) - (time.time() - t0_total))
            tool_result = execute_tool_call(
                tool_name,
                args,
                allow_cache=True,
                budget_seconds=budget_left,
            )
            tools_called_log.append(tool_name)
            total_tool_calls += 1
            progress.emit(
                f"round_{round_num + 1}_tool_done",
                _tool_result_status_line(tool_name, tool_result),
            )
            activity.log_tool_call(
                orchestrator=orch_model, tool=tool_name,
                arguments=args, result=tool_result,
                latency_ms=(time.time() - t0_tool) * 1000,
                round_num=round_num + 1,
            )

            err = _tool_error_message(tool_result)
            guidance = _tool_result_guidance_message(tool_result)
            if err:
                tool_errors_this_round += 1
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Previous tool call failed. Correct the arguments or choose a different tool "
                            "before proceeding."
                        ),
                    }
                )
                if guidance:
                    messages.append({"role": "system", "content": guidance})
            else:
                tool_success_this_round += 1
                if guidance:
                    messages.append({"role": "system", "content": guidance})

            tool_message = {
                "role": "tool",
                "name": raw_tool_name or tool_name,
                "content": tool_result[:4000],
            }
            if tc.get("id"):
                tool_message["tool_call_id"] = tc.get("id")
            messages.append(tool_message)

        if tool_errors_this_round > 0 and tool_success_this_round == 0:
            consecutive_tool_error_rounds += 1
        else:
            consecutive_tool_error_rounds = 0

        if consecutive_tool_error_rounds >= TOOL_ERROR_ROUND_STREAK_LIMIT:
            progress.emit(
                "tool_error_streak_limit",
                f"consecutive_error_rounds={consecutive_tool_error_rounds}",
                force=True,
            )
            hint_text = _recent_tool_failure_hints(limit=TOOL_FAILURE_HINTS_MAX)
            if hint_text:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Stopping repeated failing tool loops. Summarize best-known evidence and limitations.\n"
                            f"Recent failures:\n{hint_text}"
                        ),
                    }
                )
            break

    if not content.strip():
        fallback = get_best_model_for_role("reasoning") or "deepseek-r1:8b"
        remaining = max(8.0, float(timeout_seconds) - (time.time() - t0_total))
        if remaining > 10.0:
            progress.emit("final_fallback", f"using {fallback} for concise final answer", force=True)
            content = _run_reasoning_model(
                model=fallback,
                task=f"Provide a concise answer using available evidence. Task: {prompt}",
                max_predict=240,
                timeout_seconds=min(12.0, remaining),
                allow_fallback=False,
            )
        else:
            content = "[ERROR] Orchestration budget exhausted before final answer."

    if ("[ERROR]" in content or "timed out" in content.lower()) and primed_context:
        compact = primed_context[:1400].strip()
        content = (
            "Model inference timed out; returning evidence-first snapshot from tools.\n\n"
            + compact
        )

    if trading_critical:
        progress.emit("objective_guard", "evaluating trading objective constraints")
        objective_raw = execute_tool_call(
            "query_trade_metrics",
            {"metric": "objective"},
            allow_cache=True,
        )
        try:
            objective_payload = json.loads(objective_raw)
        except Exception:
            objective_payload = {}
        if isinstance(objective_payload, dict):
            objective_status = str(objective_payload.get("status", "")).upper()
            limitations = objective_payload.get("limitations", [])
            if objective_status != "PASS" and isinstance(limitations, list) and limitations:
                short_limitations = "\n".join(f"- {str(x)}" for x in limitations[:3])
                content = (
                    f"{content}\n\n"
                    "Trading objective guard (non-PASS):\n"
                    f"{short_limitations}\n"
                    "Mode: advisory/evidence-first; apply conservative position sizing until objective is PASS."
                )

    # Deliver result via OpenClaw if requested
    if reply_channel and reply_to and content:
        progress.emit(
            "orchestration_complete",
            f"sending final response (tool_calls={total_tool_calls}, elapsed={time.time() - t0_total:.1f}s)",
            force=True,
        )
        _deliver_response(channel=reply_channel, to=reply_to, message=content)

    return content or "(orchestration completed without final response)"


def _deliver_response(channel: str, to: str, message: str) -> None:
    """Deliver orchestration result via OpenClaw."""
    try:
        from utils.openclaw_cli import send_message

        res = send_message(
            to=to, message=message,
            command=os.getenv("OPENCLAW_COMMAND", "openclaw"),
            channel=channel,
        )
        if res.ok:
            logger.info("Delivered response to %s:%s", channel, to)
        else:
            logger.warning("Failed to deliver to %s:%s: %s", channel, to, res.stderr[:200])
    except Exception as e:
        logger.error("OpenClaw delivery failed: %s", e)


# ---------------------------------------------------------------------------
# OpenClaw bridge - message-driven orchestration
# ---------------------------------------------------------------------------

def openclaw_bridge(
    message: str,
    channel: Optional[str] = None,
    reply_to: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    OpenClaw bridge: accept an incoming message and route through the
    multi-model orchestration pipeline.

    This is the entry point for OpenClaw agent integration. OpenClaw sends
    user messages here, and the orchestrator processes them using the
    3-model pipeline, then returns the response.

    Args:
        message: Incoming user message text.
        channel: Source channel (whatsapp, telegram, discord).
        reply_to: Target to reply to via OpenClaw.
        session_id: OpenClaw session ID for context continuity.

    Returns:
        Orchestrated response text.
    """
    activity = _get_activity_logger()
    bridge_channel = str(channel or "").strip().lower() or None
    force_orchestration = _bridge_requires_forced_orchestration(bridge_channel)
    activity.log_openclaw_event(
        channel=bridge_channel or "unknown",
        event_type="bridge_incoming",
        payload={"message_preview": message[:100], "session_id": session_id},
    )
    if channel and reply_to and PROGRESS_UPDATES_DEFAULT:
        _deliver_response(
            channel=channel,
            to=reply_to,
            message="[progress 0.0s] request_received | analyzing request and selecting tools",
        )

    bridge_force_tool_primer = _bridge_force_tool_primer(message)
    bridge_timeout = _bridge_timeout_seconds(message, bridge_channel)
    bridge_max_rounds = max(1, min(MAX_ROUNDS_DEFAULT, 2))
    bridge_max_tool_calls = max(1, min(MAX_TOOL_CALLS_DEFAULT, 8))
    openclaw_template = _openclaw_prompt_template_for_message(message)
    bridge_system_prompt = _base_system_prompt()
    if openclaw_template:
        bridge_system_prompt = f"{bridge_system_prompt}\n\n{openclaw_template}"

    bridge_lock: Optional[_BridgeTurnLock] = None
    if force_orchestration:
        lock_timeout = max(float(bridge_timeout), float(WHATSAPP_BRIDGE_TURN_LOCK_TIMEOUT_SECONDS))
        bridge_lock = _acquire_bridge_turn_lock(
            channel=bridge_channel,
            session_id=session_id,
            wait_seconds=lock_timeout,
            stale_seconds=WHATSAPP_BRIDGE_TURN_LOCK_STALE_SECONDS,
            poll_seconds=WHATSAPP_BRIDGE_TURN_LOCK_POLL_SECONDS,
        )
        if bridge_lock is None:
            response = (
                "[ERROR] WhatsApp bridge busy: unable to acquire the exclusive qwen orchestration slot "
                f"within {int(lock_timeout)}s."
            )
            activity.log_openclaw_event(
                channel=bridge_channel or "unknown",
                event_type="bridge_turn_lock_timeout",
                payload={"wait_seconds": round(lock_timeout, 1)},
            )
            if channel and reply_to:
                _deliver_response(channel=channel, to=reply_to, message=response)
            return response
        activity.log_openclaw_event(
            channel=bridge_channel or "unknown",
            event_type="bridge_turn_lock_acquired",
            payload={"waited_seconds": round(float(bridge_lock.waited_seconds), 3)},
        )
        if channel and reply_to and bridge_lock.waited_seconds >= 1.0 and PROGRESS_UPDATES_DEFAULT:
            _deliver_response(
                channel=channel,
                to=reply_to,
                message=(
                    "[progress 0.1s] whatsapp_turn_queue | "
                    f"exclusive qwen slot acquired after {bridge_lock.waited_seconds:.1f}s"
                ),
            )

    try:
        if force_orchestration:
            response = orchestrate(
                prompt=message,
                system_prompt=bridge_system_prompt,
                reply_channel=channel,
                reply_to=reply_to,
                force_tool_primer=bridge_force_tool_primer,
                max_rounds=bridge_max_rounds,
                max_tool_calls=bridge_max_tool_calls,
                timeout_seconds=bridge_timeout,
            )
            activity.log_openclaw_event(
                channel=bridge_channel or "unknown",
                event_type="bridge_response",
                payload={"response_preview": response[:100]},
            )
            return response

        for fast_path_spec in _BRIDGE_FAST_PATH_SPECS:
            fast_path_response = _run_bridge_fast_path(
                spec=fast_path_spec,
                message=message,
                channel=channel,
                reply_to=reply_to,
                activity=activity,
            )
            if fast_path_response is not None:
                return fast_path_response

        if _is_status_fast_path_prompt(message):
            activity.log_openclaw_event(
                channel=channel or "unknown",
                event_type="bridge_status_fast_path_start",
                payload={},
            )
            if channel and reply_to and PROGRESS_UPDATES_DEFAULT:
                _deliver_response(
                    channel=channel,
                    to=reply_to,
                    message="[progress 0.1s] status_fast_path | checking system health",
                )
            health_raw = execute_tool_call("check_system_health", {}, allow_cache=True, budget_seconds=20.0)
            response = _summarize_health_tool_result(health_raw)
            if reply_to and channel:
                _deliver_response(channel=channel, to=reply_to, message=response)
            activity.log_openclaw_event(
                channel=channel or "unknown",
                event_type="bridge_status_fast_path_complete",
                payload={"response_preview": response[:120]},
            )
            return response

        # Determine if this needs tool orchestration or direct reasoning
        routed_model = route_task(message)
        routed_spec = MODEL_REGISTRY.get(routed_model)
        trading_critical = _is_trading_critical_prompt(message)

        # If the task routes to a tool-capable model, use full orchestration
        if trading_critical or (routed_spec and routed_spec.supports_tools):
            response = orchestrate(
                prompt=message,
                system_prompt=bridge_system_prompt,
                reply_channel=channel,
                reply_to=reply_to,
                force_tool_primer=bridge_force_tool_primer,
                max_rounds=bridge_max_rounds,
                max_tool_calls=bridge_max_tool_calls,
                timeout_seconds=bridge_timeout,
            )
        else:
            # For reasoning tasks, check if orchestration would add value
            task_lower = message.lower()
            needs_data = any(kw in task_lower for kw in [
                "pnl", "gate", "metric", "trade", "status", "health", "check",
            ])
            if needs_data:
                # Use orchestrator to access data tools
                response = orchestrate(
                    prompt=message,
                    system_prompt=bridge_system_prompt,
                    reply_channel=channel,
                    reply_to=reply_to,
                    force_tool_primer=bridge_force_tool_primer,
                    max_rounds=bridge_max_rounds,
                    max_tool_calls=bridge_max_tool_calls,
                    timeout_seconds=bridge_timeout,
                )
            else:
                # Direct reasoning without orchestration overhead
                response = _run_reasoning_model(model=routed_model, task=message)
                if reply_to and channel:
                    _deliver_response(channel=channel, to=reply_to, message=response)

        activity.log_openclaw_event(
            channel=channel or "unknown",
            event_type="bridge_response",
            payload={"response_preview": response[:100]},
        )

        return response
    finally:
        if bridge_lock is not None:
            _release_bridge_turn_lock(bridge_lock)


# ---------------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------------

def route_task(task_description: str) -> str:
    """Determine which model should handle a task."""
    task_lower = task_description.lower()

    best_task = "market_analysis"
    best_score = 0

    for task_type, keywords in ROUTING_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > best_score:
            best_score = score
            best_task = task_type

    model = TASK_ROUTING.get(best_task, "deepseek-r1:8b")

    # Check availability and fallback
    spec = MODEL_REGISTRY.get(model)
    if spec and not spec.available:
        _ensure_health_fresh()
        spec = MODEL_REGISTRY.get(model)
        if spec and not spec.available:
            fallback = get_best_model_for_role(spec.role)
            if fallback:
                logger.info("Routing fallback: %s -> %s", model, fallback)
                return fallback

    return model


# ---------------------------------------------------------------------------
# OpenClaw config sync
# ---------------------------------------------------------------------------

def sync_openclaw_config(dry_run: bool = False) -> list[str]:
    """
    Sync the multi-model orchestrator config into OpenClaw.

    Ensures:
    - All 3 models are registered in models.providers.ollama
    - qwen3:8b is set as tool-calling model
    - Agent defaults use the correct primary + fallback chain
    - The Ollama API mode is set to openai-completions (not native ollama)
      to prevent 'does not support tools' errors with deepseek-r1

    Returns list of status messages.
    """
    msgs: list[str] = []

    available = discover_models()
    registered = [m for m in MODEL_REGISTRY if m in available]
    missing = [m for m in MODEL_REGISTRY if m not in available]

    if missing:
        msgs.append(f"Missing models ({len(missing)}): {', '.join(missing)}")
        cmds = [f"ollama pull {m}" for m in missing]
        msgs.append("Run each command separately:\n  " + "\n  ".join(cmds))

    if not registered:
        msgs.append("[ERROR] No registered models available in Ollama")
        return msgs

    msgs.append(f"Available models ({len(registered)}): {', '.join(registered)}")

    if dry_run:
        msgs.append("[DRY-RUN] Would update OpenClaw config")
        return msgs

    # Build Ollama provider config with openai-completions API mode
    # (prevents 'does not support tools' errors for deepseek-r1)
    ollama_provider = {
        "baseUrl": f"{OLLAMA_BASE.rstrip('/')}/v1",
        "api": "openai-completions",
        "models": [],
    }
    for name in registered:
        spec = MODEL_REGISTRY[name]
        ollama_provider["models"].append({
            "id": name,
            "name": name,
            "reasoning": "r1" in name.lower() or "r2" in name.lower(),
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": spec.context_window,
            "maxTokens": spec.max_tokens,
        })

    # Set via openclaw CLI
    def _oc_set(path: str, value: Any) -> bool:
        payload = json.dumps(value, ensure_ascii=True)
        try:
            proc = subprocess.run(
                ["cmd", "/d", "/s", "/c", "openclaw", "--no-color", "config", "set", path, payload, "--json"],
                capture_output=True, text=True, timeout=20,
                env={**os.environ, "NODE_NO_WARNINGS": "1"},
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    # 1. Set Ollama provider
    if _oc_set("models.providers.ollama", ollama_provider):
        msgs.append("Set models.providers.ollama (openai-completions mode)")
    else:
        msgs.append("[ERROR] Failed to set models.providers.ollama")

    # 2. Set agent model defaults: primary + fallbacks
    # CRITICAL: Primary MUST be qwen3:8b (only tool-capable model).
    # OpenClaw agent chat requires tool-calling; deepseek-r1 does NOT support tools.
    # deepseek-r1 models are used as reasoning backends called BY qwen3:8b
    # through the orchestrator -- they must NOT be in the agent fallback chain.
    primary = "ollama/qwen3:8b" if "qwen3:8b" in registered else f"ollama/{registered[0]}"
    # Only remote fallback -- deepseek models are NOT tool-capable and must not
    # be in the agent fallback chain (they cause "does not support tools" errors).
    fallbacks = ["qwen-portal/coder-model"]

    model_block = {"primary": primary, "fallbacks": fallbacks}
    if _oc_set("agents.defaults.model", model_block):
        msgs.append(f"Set agents.defaults.model primary={primary} fallbacks={len(fallbacks)}")
    else:
        msgs.append("[ERROR] Failed to set agents.defaults.model")

    # 3. Update model allowlist (agent-accessible models only)
    # Only include tool-capable models; deepseek stays in the provider
    # for direct Ollama API calls but is NOT offered to the agent.
    models_allowlist = {"ollama/qwen3:8b": {}}
    models_allowlist["qwen-portal/coder-model"] = {"alias": "qwen"}
    models_allowlist["qwen-portal/vision-model"] = {}
    if _oc_set("agents.defaults.models", models_allowlist):
        msgs.append(f"Updated agents.defaults.models allowlist ({len(models_allowlist)} refs)")
    else:
        msgs.append("[ERROR] Failed to update model allowlist")

    # 4. Restart gateway
    try:
        proc = subprocess.run(
            ["cmd", "/d", "/s", "/c", "openclaw", "gateway", "restart"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        if proc.returncode == 0:
            msgs.append("Gateway restarted successfully")
        else:
            msgs.append(f"Gateway restart failed (exit={proc.returncode})")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        msgs.append(f"Gateway restart error: {e}")

    return msgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_status(_args) -> int:
    health = refresh_health()
    print("[orchestrator] Multi-Model LLM Orchestration Status")
    print(f"  Ollama endpoint: {OLLAMA_BASE}")
    print(f"  LLM config path: {LLM_CONFIG_PATH}")
    print(f"  Active production profile: {ACTIVE_PRODUCTION_PROFILE_NAME}")
    if isinstance(ACTIVE_PRODUCTION_SLA, dict) and ACTIVE_PRODUCTION_SLA:
        p95 = ACTIVE_PRODUCTION_SLA.get("target_p95_latency_seconds")
        budget = ACTIVE_PRODUCTION_SLA.get("target_timeout_budget_seconds")
        min_wr = ACTIVE_PRODUCTION_SLA.get("trading_min_wilson_win_rate")
        max_err = ACTIVE_PRODUCTION_SLA.get("trading_target_max_error_rate")
        print(
            "  SLA targets: "
            f"p95_latency<={p95}s timeout_budget<={budget}s "
            f"wilson_win_rate>={min_wr} max_error<={max_err}"
        )
    print(f"  Models registered: {len(MODEL_REGISTRY)}")
    print(f"  System degraded: {_system_degraded()}")
    print(
        "  Trading objective defaults: "
        f"min_trades={TRADING_OBJECTIVE_MIN_TRADES}, "
        f"max_error={TRADING_OBJECTIVE_TARGET_MAX_ERROR_RATE:.2f}, "
        f"min_wilson_win={TRADING_OBJECTIVE_MIN_WILSON_WIN_RATE:.2f}, "
        f"max_p={TRADING_OBJECTIVE_PVALUE_MAX:.4f}"
    )
    print()

    for name, spec in MODEL_REGISTRY.items():
        status = "[OK]" if spec.available else "[MISSING]"
        if _model_in_cooldown(spec):
            status = "[COOLDOWN]"
        tool_tag = " [tools]" if spec.supports_tools else ""
        print(f"  {status} {name} (role={spec.role}, speed={spec.speed}, vram={spec.vram_gb}GB{tool_tag})")
        if spec.success_count or spec.error_count:
            total = spec.success_count + spec.error_count
            rate = spec.success_count / total if total > 0 else 0
            print(
                f"        success={spec.success_count} error={spec.error_count} "
                f"rate={rate:.0%} avg_latency={spec.avg_latency_ms:.0f}ms"
            )

    missing = [m for m in MODEL_REGISTRY if not MODEL_REGISTRY[m].available]
    if missing:
        print(f"\n  To install missing models:")
        for m in missing:
            print(f"    ollama pull {m}")

    return 0


def cmd_health(_args) -> int:
    print(_get_system_health_json())
    return 0


def cmd_route(args) -> int:
    _ensure_health_fresh()
    model = route_task(args.task)
    spec = MODEL_REGISTRY.get(model)
    print(f"[orchestrator] Task: {args.task}")
    print(f"  Routed to: {model}")
    print(f"  Role: {spec.role if spec else 'unknown'}")
    print(f"  Speed: {spec.speed if spec else 'unknown'}")
    print(f"  Available: {spec.available if spec else False}")
    return 0


def cmd_orchestrate(args) -> int:
    orch = get_best_model_for_role("orchestrator")
    if not orch:
        print("[ERROR] No orchestrator model available. Run: ollama pull qwen3:8b")
        return 1

    print(f"[orchestrator] Orchestrating with {orch} (max {args.max_rounds} rounds)")
    print(
        f"  Tool strategy: max_tool_calls={args.max_tool_calls}, "
        f"force_tool_primer={args.force_tool_primer}, subagentic={args.subagentic}, "
        f"timeout_seconds={args.timeout_seconds}"
    )
    print(f"  Prompt: {args.prompt[:100]}...")
    print()

    result = orchestrate(
        args.prompt,
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
        force_tool_primer=bool(args.force_tool_primer),
        subagent_workflow=bool(args.subagentic),
        timeout_seconds=args.timeout_seconds,
    )
    print("\n--- Result ---")
    print(result)
    return 0


def cmd_openclaw_bridge(args) -> int:
    print(f"[orchestrator] OpenClaw bridge: channel={args.channel}, reply_to={args.reply_to or 'none'}")
    result = openclaw_bridge(
        message=args.message,
        channel=args.channel,
        reply_to=args.reply_to,
    )
    print("\n--- Response ---")
    print(result)
    return 0


def cmd_sync(args) -> int:
    msgs = sync_openclaw_config(dry_run=bool(args.dry_run))
    for m in msgs:
        print(f"[orchestrator] {m}")
    return 0


def main(argv: list[str]) -> int:
    # Avoid UnicodeEncodeError in Windows terminals when model output includes symbols.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="Multi-model LLM orchestrator with OpenClaw integration"
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Show model availability and routing stats")
    sub.add_parser("health", help="JSON health check for all models + gateway")

    pr = sub.add_parser("route", help="Route a task to the best model")
    pr.add_argument("--task", required=True, help="Task description to route")

    po = sub.add_parser("orchestrate", help="Run multi-model orchestration with tool-calling")
    po.add_argument("--prompt", required=True, help="Prompt for the orchestrator")
    po.add_argument("--max-rounds", type=int, default=MAX_ROUNDS_DEFAULT, help="Max tool-calling rounds")
    po.add_argument(
        "--timeout-seconds",
        type=int,
        default=ORCHESTRATION_TIMEOUT_SECONDS_DEFAULT,
        help=f"Hard orchestration wall-clock timeout (default: {ORCHESTRATION_TIMEOUT_SECONDS_DEFAULT}s)",
    )
    po.add_argument(
        "--max-tool-calls",
        type=int,
        default=MAX_TOOL_CALLS_DEFAULT,
        help=f"Hard cap on executed tool calls (default: {MAX_TOOL_CALLS_DEFAULT})",
    )
    po.add_argument(
        "--force-tool-primer",
        dest="force_tool_primer",
        action="store_true",
        default=FORCE_TOOL_PRIMER_DEFAULT,
        help="Precompute tool context before qwen orchestration.",
    )
    po.add_argument(
        "--no-force-tool-primer",
        dest="force_tool_primer",
        action="store_false",
        help="Disable precomputed tool context priming.",
    )
    po.add_argument(
        "--subagentic",
        dest="subagentic",
        action="store_true",
        default=SUBAGENT_WORKFLOW_DEFAULT,
        help="Enable sub-agent batch workflow for complex tasks.",
    )
    po.add_argument(
        "--no-subagentic",
        dest="subagentic",
        action="store_false",
        help="Disable sub-agent batch workflow.",
    )

    pb = sub.add_parser("openclaw-bridge", help="Process a message via OpenClaw bridge")
    pb.add_argument("--message", required=True, help="Message to process")
    pb.add_argument("--channel", default="whatsapp", help="Source channel")
    pb.add_argument("--reply-to", default=None, help="Target to reply to")

    ps = sub.add_parser("sync", help="Sync orchestrator config into OpenClaw")
    ps.add_argument("--dry-run", action="store_true", help="Show changes without applying")

    args = p.parse_args(argv)

    handlers = {
        "status": cmd_status,
        "health": cmd_health,
        "route": cmd_route,
        "orchestrate": cmd_orchestrate,
        "openclaw-bridge": cmd_openclaw_bridge,
        "sync": cmd_sync,
    }

    handler = handlers.get(args.cmd)
    if handler:
        return handler(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
