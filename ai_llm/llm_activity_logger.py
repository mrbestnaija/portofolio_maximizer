#!/usr/bin/env python3
"""
LLM Activity Persistence Logger for Portfolio Maximizer.

Logs all local LLM interactions (Ollama, OpenClaw) to structured JSONL files
for audit trails, performance tracking, and self-improvement analysis.

Logs persist to: logs/llm_activity/YYYY-MM-DD.jsonl
Summary index:   logs/llm_activity/activity_index.json

Usage:
    from ai_llm.llm_activity_logger import LLMActivityLogger

    logger = LLMActivityLogger()
    logger.log_request(model="deepseek-r1:8b", prompt="...", response="...", latency_ms=320)
    logger.log_tool_call(orchestrator="qwen3:8b", tool="fast_reasoning", args={...}, result="...")
    logger.log_openclaw_event(channel="whatsapp", event_type="message_sent", payload={...})

    # Query recent activity
    recent = logger.get_recent(hours=24)
    summary = logger.get_summary()
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs" / "llm_activity"
INDEX_FILE = LOG_DIR / "activity_index.json"


class LLMActivityLogger:
    """Thread-safe JSONL logger for all LLM activity."""

    def __init__(self, log_dir: Optional[Path] = None):
        self._log_dir = log_dir or LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._session_id = f"sess_{int(time.time())}_{os.getpid()}"

    def _today_file(self) -> Path:
        return self._log_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

    def _write_entry(self, entry: dict) -> None:
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        entry.setdefault("session_id", self._session_id)
        line = json.dumps(entry, default=str, ensure_ascii=False)
        with self._lock:
            with open(self._today_file(), "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_request(
        self,
        model: str,
        prompt: str,
        response: str,
        latency_ms: float = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        temperature: float = 0.1,
        task_type: str = "",
        success: bool = True,
        error: str = "",
    ) -> None:
        """Log a direct LLM inference request."""
        self._write_entry({
            "type": "llm_request",
            "model": model,
            "prompt_len": len(prompt),
            "prompt_preview": prompt[:200],
            "response_len": len(response),
            "response_preview": response[:300],
            "latency_ms": round(latency_ms, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "temperature": temperature,
            "task_type": task_type,
            "success": success,
            "error": error,
        })

    def log_tool_call(
        self,
        orchestrator: str,
        tool: str,
        arguments: dict,
        result: str,
        latency_ms: float = 0,
        round_num: int = 0,
    ) -> None:
        """Log a tool call dispatched by qwen3:8b orchestrator."""
        self._write_entry({
            "type": "tool_call",
            "orchestrator": orchestrator,
            "tool": tool,
            "arguments": arguments,
            "result_len": len(result),
            "result_preview": result[:300],
            "latency_ms": round(latency_ms, 1),
            "round_num": round_num,
        })

    def log_orchestration(
        self,
        prompt: str,
        final_response: str,
        rounds: int,
        total_latency_ms: float,
        tools_called: list,
        success: bool = True,
    ) -> None:
        """Log a complete orchestration session."""
        self._write_entry({
            "type": "orchestration",
            "prompt_preview": prompt[:200],
            "response_preview": final_response[:300],
            "rounds": rounds,
            "total_latency_ms": round(total_latency_ms, 1),
            "tools_called": tools_called,
            "success": success,
        })

    def log_openclaw_event(
        self,
        channel: str,
        event_type: str,
        payload: Optional[dict] = None,
        model_used: str = "",
        latency_ms: float = 0,
    ) -> None:
        """Log an OpenClaw gateway event (message send/receive, model routing)."""
        self._write_entry({
            "type": "openclaw_event",
            "channel": channel,
            "event_type": event_type,
            "model_used": model_used,
            "payload_keys": list((payload or {}).keys()),
            "latency_ms": round(latency_ms, 1),
        })

    def log_self_improvement(
        self,
        action: str,
        target_file: str,
        description: str,
        diff_preview: str = "",
        approved: bool = False,
        applied: bool = False,
    ) -> None:
        """Log a self-improvement action (code modification, config change)."""
        self._write_entry({
            "type": "self_improvement",
            "action": action,
            "target_file": target_file,
            "description": description,
            "diff_preview": diff_preview[:500],
            "approved": approved,
            "applied": applied,
        })

    def get_recent(self, hours: int = 24) -> list[dict]:
        """Get recent activity entries within the last N hours."""
        cutoff = time.time() - (hours * 3600)
        entries = []
        for log_file in sorted(self._log_dir.glob("*.jsonl"), reverse=True)[:3]:
            try:
                for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    ts = entry.get("timestamp", "")
                    if ts:
                        entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                        if entry_time >= cutoff:
                            entries.append(entry)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
        return entries

    def get_summary(self) -> dict:
        """Get summary statistics of all logged activity."""
        recent = self.get_recent(hours=24)
        type_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        total_latency = 0.0
        for e in recent:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            m = e.get("model") or e.get("orchestrator") or e.get("model_used") or ""
            if m:
                model_counts[m] = model_counts.get(m, 0) + 1
            total_latency += e.get("latency_ms", 0) or e.get("total_latency_ms", 0)

        return {
            "period": "last_24h",
            "total_events": len(recent),
            "by_type": type_counts,
            "by_model": model_counts,
            "total_latency_ms": round(total_latency, 1),
            "self_improvements": sum(
                1 for e in recent if e.get("type") == "self_improvement"
            ),
        }


# Module-level singleton for easy import
_default_logger: Optional[LLMActivityLogger] = None


def get_logger() -> LLMActivityLogger:
    """Get or create the default activity logger singleton."""
    global _default_logger
    if _default_logger is None:
        _default_logger = LLMActivityLogger()
    return _default_logger
