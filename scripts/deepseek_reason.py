#!/usr/bin/env python3
"""
DeepSeek Reasoning Delegate -- called by the qwen3 OpenClaw agent via `exec`.

This script sends a reasoning prompt to a DeepSeek R1 model via the local
Ollama API and returns the response.  It is designed to be invoked by the
qwen3:8b orchestrator agent when a task requires deeper reasoning, longer
context, or chain-of-thought analysis that exceeds qwen3's capabilities.

Usage (from the OpenClaw agent's exec tool):
    python scripts/deepseek_reason.py "Analyze why AAPL lost $325 in the last sprint"
    python scripts/deepseek_reason.py --model deepseek-r1:32b "Complex analysis prompt"
    python scripts/deepseek_reason.py --context-file data/some_data.json "Summarize this"
    echo "long prompt" | python scripts/deepseek_reason.py --stdin

Environment:
    OLLAMA_HOST        Ollama server URL (default: http://localhost:11434)
    DEEPSEEK_MODEL     Default model (default: deepseek-r1:8b)
    DEEPSEEK_TIMEOUT   Request timeout in seconds (default: 120)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Bootstrap .env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from etl.secret_loader import bootstrap_dotenv
    bootstrap_dotenv()
except Exception:
    pass


def _ollama_host() -> str:
    return (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def _default_model() -> str:
    return os.getenv("DEEPSEEK_MODEL") or "deepseek-r1:8b"


def _timeout() -> float:
    try:
        return max(10.0, float(os.getenv("DEEPSEEK_TIMEOUT") or "120"))
    except (ValueError, TypeError):
        return 120.0


def reason(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    context_text: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
) -> dict:
    """Send a reasoning request to DeepSeek via Ollama and return the result."""
    import requests

    model = model or _default_model()
    host = _ollama_host()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if context_text:
        messages.append({"role": "user", "content": f"Context:\n{context_text}"})
        messages.append({"role": "assistant", "content": "I've read the context. What would you like me to analyze?"})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(
            f"{host}/api/chat",
            json=payload,
            timeout=_timeout(),
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        # Strip <think>...</think> blocks from reasoning models
        import re
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return {
            "ok": True,
            "model": model,
            "response": content_clean,
            "thinking": _extract_thinking(content),
            "total_duration_ms": data.get("total_duration", 0) // 1_000_000,
            "eval_count": data.get("eval_count", 0),
        }
    except requests.ConnectionError:
        return {"ok": False, "error": f"Cannot connect to Ollama at {host}", "model": model}
    except requests.Timeout:
        return {"ok": False, "error": f"Request timed out after {_timeout()}s", "model": model}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "model": model}


def _extract_thinking(content: str) -> str | None:
    """Extract <think> block content if present."""
    import re
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return match.group(1).strip() if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delegate reasoning tasks to DeepSeek R1 via Ollama",
        epilog="Called by the qwen3 OpenClaw agent for tasks requiring deep reasoning.",
    )
    parser.add_argument("prompt", nargs="?", default="", help="The reasoning prompt")
    parser.add_argument("--model", default=None, help=f"Model name (default: {_default_model()})")
    parser.add_argument("--system", default=None, help="System prompt for the model")
    parser.add_argument(
        "--context-file", default=None,
        help="Path to a file whose contents are injected as context before the prompt",
    )
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max response tokens")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    args = parser.parse_args(argv)

    prompt = args.prompt
    if args.stdin or (not prompt and not sys.stdin.isatty()):
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: No prompt provided. Use positional arg, --stdin, or pipe input.", file=sys.stderr)
        return 1

    context_text = None
    if args.context_file:
        ctx_path = Path(args.context_file)
        if not ctx_path.exists():
            print(f"Error: Context file not found: {ctx_path}", file=sys.stderr)
            return 1
        context_text = ctx_path.read_text(encoding="utf-8", errors="replace")

    result = reason(
        prompt,
        model=args.model,
        system=args.system,
        context_text=context_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif result["ok"]:
        print(result["response"])
    else:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
