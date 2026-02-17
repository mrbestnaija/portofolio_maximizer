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
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_llm.llm_activity_logger import get_logger as _get_activity_logger

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


MAX_TOOL_CALLS_DEFAULT = max(1, _int_env("PMX_QWEN_MAX_TOOL_CALLS", 12))
FORCE_TOOL_PRIMER_DEFAULT = _truthy_env("PMX_QWEN_FORCE_TOOL_PRIMER", False)
SUBAGENT_WORKFLOW_DEFAULT = _truthy_env("PMX_QWEN_SUBAGENT_WORKFLOW", False)


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
        if spec.role == role and spec.available:
            return name

    # Fallback: any available model (prefer fast over slow)
    speed_order = {"fast": 0, "medium": 1, "slow": 2}
    candidates = [
        (name, spec) for name, spec in MODEL_REGISTRY.items()
        if spec.available
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
            # Exponential moving average of latency
            alpha = 0.3
            spec.avg_latency_ms = (alpha * latency_ms) + ((1 - alpha) * spec.avg_latency_ms) if spec.avg_latency_ms > 0 else latency_ms
        else:
            spec.error_count += 1


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
            "description": "Query PnL integrity metrics from the trading database",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["canonical_pnl", "win_rate", "profit_factor", "round_trips", "summary"],
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

def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool call requested by the orchestrator model."""

    if tool_name == "fast_reasoning":
        model = get_best_model_for_role("reasoning") or "deepseek-r1:8b"
        return _run_reasoning_model(
            model=model,
            task=arguments.get("task", ""),
            context=arguments.get("context", ""),
        )

    if tool_name == "deep_reasoning":
        model = get_best_model_for_role("heavy_reasoning") or "deepseek-r1:32b"
        return _run_reasoning_model(
            model=model,
            task=arguments.get("task", ""),
            context=arguments.get("context", ""),
        )

    if tool_name == "read_gate_status":
        return _read_gate_status(arguments.get("gate", "all"))

    if tool_name == "query_trade_metrics":
        return _query_trade_metrics(arguments.get("metric", "summary"))

    if tool_name == "send_notification":
        return _send_notification(
            channel=arguments.get("channel", "whatsapp"),
            message=arguments.get("message", ""),
        )

    if tool_name == "run_sub_agent_batch":
        return _run_sub_agent_batch(
            tasks=arguments.get("tasks", []),
            default_context=arguments.get("default_context", ""),
        )

    if tool_name == "check_system_health":
        return _get_system_health_json()

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _run_reasoning_model(
    model: str,
    task: str,
    context: str = "",
    *,
    max_predict: int = 1024,
    timeout_seconds: float = 180.0,
) -> str:
    """Dispatch a reasoning task to a deepseek-r1 model with fallback."""
    prompt = task
    if context:
        prompt = f"Context:\n{context}\n\nTask:\n{task}"

    # Try primary model, then fallback
    models_to_try = [model]
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
        if spec and not spec.available:
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
    audit_dir = PROJECT_ROOT / "logs" / "audit_sprint"
    if not audit_dir.exists():
        return json.dumps({"error": "No audit_sprint directory found"})

    runs = sorted([d for d in audit_dir.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        return json.dumps({"error": "No audit runs found"})

    latest = runs[0]
    results = {}

    if gate == "all":
        for f in sorted(latest.glob("gate_*.log")):
            results[f.stem] = f.read_text(encoding="utf-8", errors="replace")[:2000]
    else:
        target = latest / f"{gate}.log"
        if target.exists():
            results[gate] = target.read_text(encoding="utf-8", errors="replace")[:2000]
        else:
            results["error"] = f"Gate log not found: {target}"

    return json.dumps(results, indent=2)


def _query_trade_metrics(metric: str) -> str:
    """Query canonical PnL metrics via PnLIntegrityEnforcer."""
    db_path = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
    if not db_path.exists():
        return json.dumps({"error": "Database not found"})

    try:
        from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer

        with PnLIntegrityEnforcer(str(db_path)) as enforcer:
            metrics = enforcer.get_canonical_metrics()
            if metric == "summary":
                return json.dumps({
                    "total_pnl": f"${metrics.total_realized_pnl:+,.2f}",
                    "win_rate": f"{metrics.win_rate:.1%}",
                    "profit_factor": f"{metrics.profit_factor:.2f}",
                    "total_trades": metrics.total_trades,
                    "wins": metrics.wins,
                    "losses": metrics.losses,
                })
            elif metric == "canonical_pnl":
                return f"${metrics.total_realized_pnl:+,.2f}"
            elif metric == "win_rate":
                return f"{metrics.win_rate:.1%}"
            elif metric == "profit_factor":
                return f"{metrics.profit_factor:.2f}"
            elif metric == "round_trips":
                return str(metrics.total_trades)
            else:
                return json.dumps({"error": f"Unknown metric: {metric}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


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
            "status",
        )
    )


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
    c = (complexity or "auto").strip().lower()
    if c == "fast":
        return get_best_model_for_role("reasoning") or "deepseek-r1:8b"
    if c == "deep":
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
    for idx, raw in enumerate(tasks[:8], start=1):
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
            timeout_seconds=45.0,
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


def _build_precomputed_tool_context(prompt: str, subagent_workflow: bool) -> str:
    """
    Pre-run high-value tool calls so qwen starts with concrete context.
    This increases tool-backed responses even on the first round.
    """
    context_parts: list[str] = []

    base_calls: list[tuple[str, dict[str, Any]]] = [("check_system_health", {})]
    if _is_data_heavy_prompt(prompt):
        base_calls.extend(
            [
                ("query_trade_metrics", {"metric": "summary"}),
                ("read_gate_status", {"gate": "all"}),
            ]
        )
    if subagent_workflow:
        base_calls.append(
            (
                "run_sub_agent_batch",
                {"tasks": _auto_subagent_tasks(prompt), "default_context": prompt},
            )
        )

    for tool_name, args in base_calls:
        try:
            result = execute_tool_call(tool_name, args)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        context_parts.append(f"[{tool_name}]\n{result[:3000]}")

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Orchestration engine
# ---------------------------------------------------------------------------

def orchestrate(
    prompt: str,
    max_rounds: int = 3,
    system_prompt: Optional[str] = None,
    reply_channel: Optional[str] = None,
    reply_to: Optional[str] = None,
    force_tool_primer: bool = FORCE_TOOL_PRIMER_DEFAULT,
    subagent_workflow: bool = SUBAGENT_WORKFLOW_DEFAULT,
    max_tool_calls: int = MAX_TOOL_CALLS_DEFAULT,
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

    # Pick orchestrator model (prefer qwen3:8b, fallback to any tool-capable)
    orch_model = get_best_model_for_role("orchestrator")
    if not orch_model:
        # No tool-capable model available; fall back to direct reasoning
        logger.warning("No orchestrator model available; falling back to direct reasoning")
        reasoning_model = get_best_model_for_role("reasoning")
        if reasoning_model:
            return _run_reasoning_model(model=reasoning_model, task=prompt)
        return "[ERROR] No LLM models available. Check Ollama: ollama list"

    max_tool_calls = max(1, int(max_tool_calls))

    sys_content = system_prompt or (
        "You are Best-Anime, a quantitative portfolio optimization assistant for Bestman. "
        "You have access to reasoning tools (deepseek-r1 models) for chain-of-thought analysis, "
        "data tools for reading gate status and trade metrics, notification tools for sending "
        "messages via OpenClaw, system health checks, and sub-agent batch execution. "
        "Use tools aggressively for evidence-backed answers and execute sub-agent batches for complex tasks. "
        "Prefer tool-backed conclusions over unsupported assumptions. "
        "Be concise, data-driven, and profit-focused."
    )

    messages: list[dict] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt},
    ]

    if force_tool_primer:
        primed = _build_precomputed_tool_context(prompt=prompt, subagent_workflow=subagent_workflow)
        if primed:
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
    for round_num in range(max_rounds):
        tools_enabled = total_tool_calls < max_tool_calls
        payload = {
            "model": orch_model,
            "messages": messages,
            "tools": REASONING_TOOLS if tools_enabled else [],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        t0_round = time.time()
        try:
            result = _ollama_post("/api/chat", payload, timeout=120.0)
            _record_model_stats(orch_model, (time.time() - t0_round) * 1000, True)
        except Exception as e:
            _record_model_stats(orch_model, (time.time() - t0_round) * 1000, False)
            logger.error("Orchestration round %d failed: %s", round_num + 1, e)

            # Try fallback to direct reasoning
            fallback = get_best_model_for_role("reasoning")
            if fallback:
                logger.info("Falling back to direct reasoning with %s", fallback)
                return _run_reasoning_model(model=fallback, task=prompt)
            return f"[ERROR] Orchestration failed: {e}"

        message = result.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            activity.log_orchestration(
                prompt=prompt, final_response=content,
                rounds=round_num + 1,
                total_latency_ms=(time.time() - t0_total) * 1000,
                tools_called=tools_called_log, success=True,
            )
            break

        # Execute tool calls
        messages.append(message)

        for tc in tool_calls:
            if total_tool_calls >= max_tool_calls:
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
            tool_name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {"task": args}

            logger.info("[round %d] Tool call: %s(%s)", round_num + 1, tool_name, json.dumps(args)[:80])
            t0_tool = time.time()
            tool_result = execute_tool_call(tool_name, args)
            tools_called_log.append(tool_name)
            total_tool_calls += 1
            activity.log_tool_call(
                orchestrator=orch_model, tool=tool_name,
                arguments=args, result=tool_result,
                latency_ms=(time.time() - t0_tool) * 1000,
                round_num=round_num + 1,
            )

            messages.append({
                "role": "tool",
                "content": tool_result[:4000],
            })

    # Deliver result via OpenClaw if requested
    if reply_channel and reply_to and content:
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
    activity.log_openclaw_event(
        channel=channel or "unknown",
        event_type="bridge_incoming",
        payload={"message_preview": message[:100], "session_id": session_id},
    )

    # Determine if this needs tool orchestration or direct reasoning
    routed_model = route_task(message)
    routed_spec = MODEL_REGISTRY.get(routed_model)

    # If the task routes to a tool-capable model, use full orchestration
    if routed_spec and routed_spec.supports_tools:
        response = orchestrate(
            prompt=message,
            reply_channel=channel,
            reply_to=reply_to,
        )
    else:
        # For reasoning tasks, check if orchestration would add value
        task_lower = message.lower()
        needs_data = any(kw in task_lower for kw in [
            "pnl", "gate", "metric", "trade", "status", "health", "check",
        ])
        if needs_data:
            # Use orchestrator to access data tools
            response = orchestrate(prompt=message, reply_channel=channel, reply_to=reply_to)
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
        msgs.append("Run: " + " && ".join(f"ollama pull {m}" for m in missing))

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
    print(f"  Models registered: {len(MODEL_REGISTRY)}")
    print()

    for name, spec in MODEL_REGISTRY.items():
        status = "[OK]" if spec.available else "[MISSING]"
        tool_tag = " [tools]" if spec.supports_tools else ""
        print(f"  {status} {name} (role={spec.role}, speed={spec.speed}, vram={spec.vram_gb}GB{tool_tag})")
        if spec.success_count or spec.error_count:
            total = spec.success_count + spec.error_count
            rate = spec.success_count / total if total > 0 else 0
            print(f"        success={spec.success_count} error={spec.error_count} rate={rate:.0%} avg_latency={spec.avg_latency_ms:.0f}ms")

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
        f"force_tool_primer={args.force_tool_primer}, subagentic={args.subagentic}"
    )
    print(f"  Prompt: {args.prompt[:100]}...")
    print()

    result = orchestrate(
        args.prompt,
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
        force_tool_primer=bool(args.force_tool_primer),
        subagent_workflow=bool(args.subagentic),
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
    po.add_argument("--max-rounds", type=int, default=3, help="Max tool-calling rounds")
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
