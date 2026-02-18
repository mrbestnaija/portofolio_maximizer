#!/usr/bin/env python3
"""
OpenClaw model provider + failover manager (local + remote).

Goals:
- Avoid "paywall lock-in" by offering a zero-cost local fallback via Ollama.
- Make model provider swapping robust: configure primary + fallbacks so OpenClaw
  can fail over when a provider is down, keys are missing, or network is flaky.
- Keep credentials canonical in repo `.env` (git-ignored) while supporting the
  OpenClaw Gateway Scheduled Task which does NOT automatically inherit repo env.

What this does (optional, explicit):
- Configure `models.providers.ollama` based on local Ollama `/api/tags` discovery
  (or a configured list).
- Optionally configure `models.providers.openai` / `models.providers.anthropic`.
- Configure `agents.defaults.model` (primary + fallbacks).
- Configure `agents.defaults.imageModel` (primary + fallbacks).
- Optionally sync API keys from `.env` into OpenClaw's auth store:
  `~/.openclaw/agents/<id>/agent/auth-profiles.json`

Security:
- Never prints secret values.
- Only writes secrets to OpenClaw's local auth store when you run `sync-auth`
  or `apply --sync-auth`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_dotenv() -> None:
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        return


def _load_secret(name: str) -> Optional[str]:
    try:
        from etl.secret_loader import load_secret

        return load_secret(name)
    except Exception:
        return (os.getenv(name) or "").strip() or None


def _split_command(command: str) -> list[str]:
    # Reuse the repo helper (handles Windows .cmd shims).
    try:
        from utils.openclaw_cli import _split_command as _split  # type: ignore

        return _split(command)
    except Exception:
        return [str(command or "openclaw").strip() or "openclaw"]


@dataclass(frozen=True)
class _CmdResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str


def _run_openclaw(*, base: list[str], args: list[str], timeout_seconds: float = 20.0) -> _CmdResult:
    import subprocess

    cmd = [*base, *args]
    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            env=env,
        )
        return _CmdResult(
            ok=int(proc.returncode) == 0,
            returncode=int(proc.returncode),
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except FileNotFoundError as exc:
        return _CmdResult(ok=False, returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return _CmdResult(ok=False, returncode=124, stdout=stdout, stderr=stderr or "timeout")


def _parse_json_best_effort(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _oc_config_get_json(*, oc_base: list[str], path: str, timeout_seconds: float = 10.0) -> Optional[Any]:
    res = _run_openclaw(base=oc_base, args=["--no-color", "config", "get", path, "--json"], timeout_seconds=timeout_seconds)
    if not res.ok:
        return None
    try:
        return _parse_json_best_effort(res.stdout)
    except Exception:
        return None


def _oc_config_set_json(
    *,
    oc_base: list[str],
    path: str,
    value: Any,
    timeout_seconds: float = 20.0,
    dry_run: bool,
) -> _CmdResult:
    payload = json.dumps(value, ensure_ascii=True)
    if dry_run:
        return _CmdResult(ok=True, returncode=0, stdout="", stderr="")
    return _run_openclaw(
        base=oc_base,
        args=["--no-color", "config", "set", path, payload, "--json"],
        timeout_seconds=timeout_seconds,
    )


def _parse_csv(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    normalized = raw.replace("\r\n", "\n").replace("\n", ",").replace(";", ",")
    return [p.strip() for p in normalized.split(",") if p and p.strip()]


def _ollama_api_base_from_configured(base_url: str) -> str:
    # OpenClaw strips /v1 when talking to native endpoints; do the same here for discovery.
    u = (base_url or "").strip() or "http://127.0.0.1:11434/v1"
    u = u.rstrip("/")
    if u.lower().endswith("/v1"):
        u = u[: -len("/v1")]
    return u


def _discover_ollama_models(base_url: str, timeout_seconds: float = 3.0) -> list[str]:
    api_base = _ollama_api_base_from_configured(base_url)
    url = f"{api_base}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        models = payload.get("models") if isinstance(payload, dict) else None
        out: list[str] = []
        if isinstance(models, list):
            for m in models:
                if not isinstance(m, dict):
                    continue
                name = (m.get("name") or "").strip()
                if name:
                    out.append(name)
        return out
    except (OSError, urllib.error.URLError, ValueError):
        return []


def _default_ollama_model_order() -> list[str]:
    # Keep aligned with config/llm_config.yml defaults.
    # OpenClaw agent turns require tool-calling support, so keep qwen3 first.
    return [
        "qwen3:8b",             # tool-calling / function-calling orchestrator (primary for OpenClaw agent)
        "deepseek-r1:8b",       # fast reasoning (primary)
        "deepseek-r1:32b",      # heavy reasoning (fallback)
        "qwen:14b-chat-q4_K_M",
        "deepseek-coder:6.7b-instruct-q4_K_M",
    ]


def _pick_primary(preferred: list[str], available: list[str]) -> Optional[str]:
    avail = set(available)
    for cand in preferred:
        if cand in avail:
            return cand
    return available[0] if available else None


def _promote_tool_primary(order: list[str]) -> list[str]:
    """
    Keep a tool-capable model first for OpenClaw agent turns.
    """
    models = [str(x).strip() for x in order if str(x).strip()]
    if not models:
        return models
    if (os.getenv("OPENCLAW_RESPECT_ENV_MODEL_ORDER") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return models

    tool_candidates: list[str] = []
    for m in models:
        low = m.lower()
        if "qwen3" in low or "qwen2.5" in low:
            tool_candidates.append(m)

    if not tool_candidates:
        return models

    first_tool = tool_candidates[0]
    return [first_tool, *[m for m in models if m != first_tool]]


def _ollama_model_def(model_id: str) -> dict[str, Any]:
    low = (model_id or "").lower()
    is_reasoning = ("r1" in low) or ("r2" in low) or ("reasoning" in low)
    is_tool_capable = ("qwen3" in low) or ("qwen2.5" in low) or ("llama3" in low and "tool" in low)
    return {
        "id": model_id,
        "name": model_id,
        "reasoning": is_reasoning,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 65536 if not is_tool_capable else 32768,
        "maxTokens": 8192,
    }


def _simple_model_def(*, model_id: str, name: Optional[str] = None, input_types: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": name or model_id,
        "reasoning": False,
        "input": input_types or ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 8192,
    }


def _auth_store_path_for_agent(agent_id: str) -> Path:
    return Path.home() / ".openclaw" / "agents" / agent_id / "agent" / "auth-profiles.json"


def _detect_default_agent_id(*, oc_base: list[str]) -> str:
    payload = _oc_config_get_json(oc_base=oc_base, path="agents", timeout_seconds=10.0)
    if isinstance(payload, dict):
        defaults = payload.get("defaults")
        if isinstance(defaults, dict):
            # No agent id in this branch; fall through.
            pass

    status = _run_openclaw(base=oc_base, args=["--no-color", "status", "--json"], timeout_seconds=10.0)
    if status.ok:
        try:
            obj = _parse_json_best_effort(status.stdout)
            if isinstance(obj, dict):
                agents = obj.get("agents")
                if isinstance(agents, dict):
                    default_id = (agents.get("defaultId") or "").strip()
                    if default_id:
                        return default_id
        except Exception:
            pass
    return "main"


def _auth_has_provider_key(*, store_path: Path, provider: str) -> bool:
    try:
        obj = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    profiles = obj.get("profiles")
    if not isinstance(profiles, dict):
        return False
    for entry in profiles.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("provider") or "").strip().lower() != provider.lower():
            continue
        if str(entry.get("type") or "").strip() != "api_key":
            continue
        if str(entry.get("key") or "").strip():
            return True
    return False


def _sync_auth_store(
    *,
    store_path: Path,
    openai_key: Optional[str],
    anthropic_key: Optional[str],
    ollama_key: Optional[str] = None,
    dry_run: bool,
) -> tuple[bool, list[str]]:
    """
    Returns: (changed, messages)
    Messages are safe (no secrets).
    """
    msgs: list[str] = []
    changed = False

    if not openai_key and not anthropic_key:
        msgs.append("No OpenAI/Anthropic keys found in env; nothing to sync.")
        # Still allow a non-secret Ollama placeholder key if requested.
        if not (ollama_key or "").strip():
            return False, msgs

    try:
        obj = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}
    except Exception:
        obj = {}

    if not isinstance(obj, dict):
        obj = {}
    profiles = obj.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
        obj["profiles"] = profiles

    # Preserve store version if present; otherwise set a sane default.
    if "version" not in obj:
        obj["version"] = 1

    if openai_key:
        pid = "openai:default"
        profiles[pid] = {"type": "api_key", "provider": "openai", "key": openai_key}
        msgs.append("Synced OpenAI key into OpenClaw auth store (openai:default).")
        changed = True

    if anthropic_key:
        pid = "anthropic:default"
        profiles[pid] = {"type": "api_key", "provider": "anthropic", "key": anthropic_key}
        msgs.append("Synced Anthropic key into OpenClaw auth store (anthropic:default).")
        changed = True

    # OpenClaw currently requires an apiKey to be resolvable for the Ollama
    # provider even though Ollama usually does not require auth. Use a harmless
    # placeholder so local failover works without additional steps.
    if (ollama_key or "").strip():
        pid = "ollama:default"
        profiles.setdefault(pid, {"type": "api_key", "provider": "ollama", "key": str(ollama_key)})
        # Ensure key exists and is non-empty.
        try:
            if not str(profiles.get(pid, {}).get("key") or "").strip():
                profiles[pid] = {"type": "api_key", "provider": "ollama", "key": str(ollama_key)}
        except Exception:
            profiles[pid] = {"type": "api_key", "provider": "ollama", "key": str(ollama_key)}
        msgs.append("Ensured Ollama placeholder key in OpenClaw auth store (ollama:default).")
        changed = True

    if dry_run:
        return changed, ["DRY-RUN: " + m for m in msgs]

    if changed:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    return changed, msgs


def _build_fallbacks(
    *,
    local_models_ordered: list[str],
    include_remote_qwen: bool,
    include_openai: bool,
    include_anthropic: bool,
    openai_model: str,
    anthropic_model: str,
) -> list[str]:
    fallbacks: list[str] = []
    for m in local_models_ordered:
        fallbacks.append(f"ollama/{m}")
    if include_remote_qwen:
        fallbacks.append("qwen-portal/coder-model")
    if include_openai:
        fallbacks.append(f"openai/{openai_model}")
    if include_anthropic:
        fallbacks.append(f"anthropic/{anthropic_model}")
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for x in fallbacks:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _restart_gateway(*, oc_base: list[str], dry_run: bool) -> _CmdResult:
    if dry_run:
        return _CmdResult(ok=True, returncode=0, stdout="", stderr="")
    return _run_openclaw(base=oc_base, args=["gateway", "restart"], timeout_seconds=30.0)


def _cmd_status(args) -> int:
    _bootstrap_dotenv()

    oc_base = _split_command(args.command)
    agent_id = _detect_default_agent_id(oc_base=oc_base)
    store_path = _auth_store_path_for_agent(agent_id)

    model_cfg = _oc_config_get_json(oc_base=oc_base, path="agents.defaults.model", timeout_seconds=10.0) or {}
    img_cfg = _oc_config_get_json(oc_base=oc_base, path="agents.defaults.imageModel", timeout_seconds=10.0) or {}
    providers = _oc_config_get_json(oc_base=oc_base, path="models.providers", timeout_seconds=10.0) or {}

    primary = ""
    fallbacks: list[str] = []
    if isinstance(model_cfg, dict):
        primary = str(model_cfg.get("primary") or "")
        fbs = model_cfg.get("fallbacks")
        if isinstance(fbs, list):
            fallbacks = [str(x) for x in fbs if str(x).strip()]

    img_primary = ""
    img_fallbacks: list[str] = []
    if isinstance(img_cfg, dict):
        img_primary = str(img_cfg.get("primary") or "")
        fbs = img_cfg.get("fallbacks")
        if isinstance(fbs, list):
            img_fallbacks = [str(x) for x in fbs if str(x).strip()]

    provider_ids = sorted([str(k) for k in providers.keys()]) if isinstance(providers, dict) else []

    # Credential presence (values never printed)
    # load_secret(...) already honors the *_FILE convention; do not probe *_FILE directly.
    has_openai_env = bool(_load_secret("OPENAI_API_KEY"))
    has_anth_env = bool(_load_secret("ANTHROPIC_API_KEY") or _load_secret("CLAUDE_API_KEY"))
    has_openai_store = _auth_has_provider_key(store_path=store_path, provider="openai")
    has_anth_store = _auth_has_provider_key(store_path=store_path, provider="anthropic")
    has_ollama_store = _auth_has_provider_key(store_path=store_path, provider="ollama")

    # Ollama reachability + model inventory
    ollama_base = (os.getenv("OPENCLAW_OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434/v1").strip()
    discovered = _discover_ollama_models(ollama_base, timeout_seconds=2.0)

    print("[openclaw_models] OpenClaw model status")
    print(f"  command: {args.command}")
    print(f"  text primary: {primary or '(unset)'}")
    print(f"  text fallbacks ({len(fallbacks)}): {', '.join(fallbacks) if fallbacks else '-'}")
    print(f"  image primary: {img_primary or '(unset)'}")
    print(f"  image fallbacks ({len(img_fallbacks)}): {', '.join(img_fallbacks) if img_fallbacks else '-'}")
    print(f"  configured providers ({len(provider_ids)}): {', '.join(provider_ids) if provider_ids else '-'}")
    print(f"  auth store: {store_path}")
    print(f"  OPENAI_API_KEY present: {'yes' if has_openai_env else 'no'}")
    print(f"  ANTHROPIC_API_KEY/CLAUDE_API_KEY present: {'yes' if has_anth_env else 'no'}")
    print(f"  OpenClaw auth openai: {'yes' if has_openai_store else 'no'}")
    print(f"  OpenClaw auth anthropic: {'yes' if has_anth_store else 'no'}")
    print(f"  OpenClaw auth ollama: {'yes' if has_ollama_store else 'no'}")
    print(f"  Ollama reachable: {'yes' if bool(discovered) else 'no'} (models={len(discovered)})")
    if bool(args.list_ollama_models) and discovered:
        for name in discovered[:50]:
            print(f"    - {name}")
        if len(discovered) > 50:
            print("    ...(truncated)")
    return 0


def _cmd_sync_auth(args) -> int:
    _bootstrap_dotenv()

    oc_base = _split_command(args.command)
    agent_id = (args.agent_id or "").strip() or _detect_default_agent_id(oc_base=oc_base)
    store_path = _auth_store_path_for_agent(agent_id)

    openai_key = _load_secret("OPENAI_API_KEY")
    anthropic_key = _load_secret("ANTHROPIC_API_KEY") or _load_secret("CLAUDE_API_KEY")
    # OpenClaw currently requires a resolvable apiKey for the ollama provider.
    # This is a harmless placeholder for local-only use.
    ollama_key = (os.getenv("OPENCLAW_OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY") or "local").strip() or "local"

    changed, msgs = _sync_auth_store(
        store_path=store_path,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        ollama_key=ollama_key,
        dry_run=bool(args.dry_run),
    )
    print(f"[openclaw_models] auth store: {store_path}")
    for m in msgs:
        print(f"[openclaw_models] {m}")
    if changed and bool(args.restart_gateway):
        res = _restart_gateway(oc_base=oc_base, dry_run=bool(args.dry_run))
        if res.ok:
            print("[openclaw_models] gateway restart: OK")
            return 0
        print(f"[openclaw_models] gateway restart: FAILED (exit={res.returncode})", file=sys.stderr)
        return 1
    return 0


def _cmd_apply(args) -> int:
    _bootstrap_dotenv()

    oc_base = _split_command(args.command)
    agent_id = (args.agent_id or "").strip() or _detect_default_agent_id(oc_base=oc_base)
    store_path = _auth_store_path_for_agent(agent_id)

    dry_run = bool(args.dry_run)

    openai_key = _load_secret("OPENAI_API_KEY")
    anthropic_key = _load_secret("ANTHROPIC_API_KEY") or _load_secret("CLAUDE_API_KEY")
    ollama_key = (os.getenv("OPENCLAW_OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY") or "local").strip() or "local"

    if bool(args.sync_auth):
        _, msgs = _sync_auth_store(
            store_path=store_path,
            openai_key=openai_key,
            anthropic_key=anthropic_key,
            ollama_key=ollama_key,
            dry_run=dry_run,
        )
        for m in msgs:
            print(f"[openclaw_models] {m}")
    else:
        # Ensure local failover can actually run by giving the ollama provider a
        # resolvable apiKey (placeholder). This does not grant access to any paid
        # provider and is safe for localhost setups.
        _, msgs = _sync_auth_store(
            store_path=store_path,
            openai_key=None,
            anthropic_key=None,
            ollama_key=ollama_key,
            dry_run=dry_run,
        )
        for m in msgs:
            if "Ollama placeholder" in m:
                print(f"[openclaw_models] {m}")

    # Provider configs
    ollama_base_url = (args.ollama_base_url or "").strip() or (os.getenv("OPENCLAW_OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434/v1").strip()
    ollama_models = _parse_csv(args.ollama_models or "") or _parse_csv(os.getenv("OPENCLAW_OLLAMA_MODELS", ""))
    discovered = _discover_ollama_models(ollama_base_url, timeout_seconds=2.0)
    if not ollama_models:
        ollama_models = discovered or _default_ollama_model_order()

    if bool(args.enable_ollama_provider):
        ollama_provider = {
            "baseUrl": ollama_base_url,
            "api": "ollama",
            "models": [_ollama_model_def(m) for m in ollama_models],
        }
        set_res = _oc_config_set_json(
            oc_base=oc_base,
            path="models.providers.ollama",
            value=ollama_provider,
            timeout_seconds=20.0,
            dry_run=dry_run,
        )
        if not set_res.ok:
            print(f"[openclaw_models] FAILED setting models.providers.ollama (exit={set_res.returncode})", file=sys.stderr)
            tail = "\n".join((set_res.stderr or "").splitlines()[-10:])
            if tail:
                print(tail, file=sys.stderr)
            return 1
        print(f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}set models.providers.ollama (models={len(ollama_models)})")

    # Optional remote providers (do not set apiKey in config; auth comes from profiles/env)
    openai_models = _parse_csv(args.openai_models or "") or _parse_csv(os.getenv("OPENCLAW_OPENAI_MODELS", "gpt-4o-mini,gpt-4o"))
    if bool(args.enable_openai_provider):
        openai_provider = {
            "baseUrl": "https://api.openai.com/v1",
            "api": "openai-responses",
            "auth": "api-key",
            "models": [
                _simple_model_def(model_id=m, name=m, input_types=["text", "image"] if "4o" in m else ["text"]) for m in openai_models
            ],
        }
        set_res = _oc_config_set_json(
            oc_base=oc_base,
            path="models.providers.openai",
            value=openai_provider,
            timeout_seconds=20.0,
            dry_run=dry_run,
        )
        if not set_res.ok:
            print(f"[openclaw_models] FAILED setting models.providers.openai (exit={set_res.returncode})", file=sys.stderr)
            tail = "\n".join((set_res.stderr or "").splitlines()[-10:])
            if tail:
                print(tail, file=sys.stderr)
            return 1
        print(f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}set models.providers.openai (models={len(openai_models)})")

    anthropic_models = _parse_csv(args.anthropic_models or "") or _parse_csv(
        os.getenv("OPENCLAW_ANTHROPIC_MODELS", "claude-opus-4-6,claude-sonnet-4-6")
    )
    if bool(args.enable_anthropic_provider):
        anthropic_provider = {
            "baseUrl": "https://api.anthropic.com",
            "api": "anthropic-messages",
            "auth": "api-key",
            "models": [_simple_model_def(model_id=m, name=m) for m in anthropic_models],
        }
        set_res = _oc_config_set_json(
            oc_base=oc_base,
            path="models.providers.anthropic",
            value=anthropic_provider,
            timeout_seconds=20.0,
            dry_run=dry_run,
        )
        if not set_res.ok:
            print(f"[openclaw_models] FAILED setting models.providers.anthropic (exit={set_res.returncode})", file=sys.stderr)
            tail = "\n".join((set_res.stderr or "").splitlines()[-10:])
            if tail:
                print(tail, file=sys.stderr)
            return 1
        print(f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}set models.providers.anthropic (models={len(anthropic_models)})")

    # Decide primary + fallbacks
    env_primary = (os.getenv("OPENCLAW_MODEL_PRIMARY") or "").strip()
    env_fallbacks = _parse_csv(os.getenv("OPENCLAW_MODEL_FALLBACKS", ""))
    strategy = (args.strategy or "").strip().lower()
    requested_strategy = strategy
    # Hard safety: block remote/cloud models unless explicitly opted in.
    # This prevents accidental 429 "free quota exceeded" errors.
    local_only = (os.getenv("OPENCLAW_LOCAL_ONLY") or "1").strip().lower() in {"1", "true", "yes", "on"}
    if local_only and strategy in {"auto", "remote-first"}:
        strategy = "local-first"
        print(
            f"[openclaw_models] OPENCLAW_LOCAL_ONLY=1 forced strategy {requested_strategy!r} -> 'local-first' (remote/cloud disabled)"
        )
    if env_primary and env_fallbacks:
        strategy = "custom"

    preferred_local = _default_ollama_model_order()
    preferred_local = _parse_csv(os.getenv("OPENCLAW_OLLAMA_MODEL_ORDER", "")) or preferred_local
    preferred_local = _promote_tool_primary(preferred_local)
    local_primary = _pick_primary(preferred_local, discovered or ollama_models)

    include_openai = bool(args.enable_openai_provider) and (
        _auth_has_provider_key(store_path=store_path, provider="openai") or bool(openai_key) or dry_run
    )
    include_anthropic = bool(args.enable_anthropic_provider) and (
        _auth_has_provider_key(store_path=store_path, provider="anthropic") or bool(anthropic_key) or dry_run
    )
    openai_primary_model = (args.openai_primary_model or "").strip() or (openai_models[0] if openai_models else "gpt-4o-mini")
    anthropic_primary_model = (args.anthropic_primary_model or "").strip() or (anthropic_models[0] if anthropic_models else "claude-opus-4-6")

    if strategy not in {"auto", "local-first", "remote-first", "custom"}:
        print(f"[openclaw_models] Unknown strategy: {strategy!r} (use auto|local-first|remote-first|custom)", file=sys.stderr)
        return 2

    if strategy == "custom":
        primary = env_primary
        fallbacks = env_fallbacks
    elif strategy == "remote-first":
        primary = "qwen-portal/coder-model"
        local_ordered = [m for m in preferred_local if m in (discovered or ollama_models)] + [
            m for m in (discovered or ollama_models) if m not in set(preferred_local)
        ]
        fallbacks = _build_fallbacks(
            local_models_ordered=local_ordered,
            include_remote_qwen=False,
            include_openai=include_openai,
            include_anthropic=include_anthropic,
            openai_model=openai_primary_model,
            anthropic_model=anthropic_primary_model,
        )
    else:
        # auto / local-first
        # local-first NEVER uses remote models -- prevents 429 quota errors.
        include_remote_qwen = False
        if strategy == "auto" and not local_only:
            include_remote_qwen = True
        include_remote_qwen_env = (os.getenv("OPENCLAW_INCLUDE_REMOTE_QWEN_FALLBACK") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if include_remote_qwen_env:
            if not local_only:
                include_remote_qwen = True
            else:
                print(
                    "[openclaw_models] OPENCLAW_LOCAL_ONLY=1 ignoring OPENCLAW_INCLUDE_REMOTE_QWEN_FALLBACK=1"
                )
        have_local_now = bool(discovered)
        if strategy == "auto" and not have_local_now and not local_only:
            primary = "qwen-portal/coder-model"
        elif local_primary:
            primary = f"ollama/{local_primary}"
        else:
            # Even if Ollama is unreachable, set local model as primary.
            # OpenClaw will retry when Ollama comes back online.
            primary = f"ollama/{preferred_local[0]}" if preferred_local else "ollama/qwen3:8b"

        local_ordered = [m for m in preferred_local if m in (discovered or ollama_models)] + [
            m for m in (discovered or ollama_models) if m not in set(preferred_local)
        ]
        fallbacks = _build_fallbacks(
            local_models_ordered=[m for m in local_ordered if f"ollama/{m}" != primary],
            include_remote_qwen=include_remote_qwen,
            include_openai=include_openai,
            include_anthropic=include_anthropic,
            openai_model=openai_primary_model,
            anthropic_model=anthropic_primary_model,
        )

    # Never include primary as a fallback.
    fallbacks = [fb for fb in fallbacks if fb and fb != primary]

    # Hard safety: strip ALL remote/cloud models when local_only is active.
    if local_only:
        fallbacks = [fb for fb in fallbacks if fb.startswith("ollama/")]

    model_block: dict[str, Any] = {"primary": primary}
    if fallbacks:
        model_block["fallbacks"] = fallbacks

    # Ensure the allowlist includes the full primary+fallback chain; otherwise
    # OpenClaw will ignore fallbacks even if configured.
    allow_refs: set[str] = set([primary, *fallbacks])

    set_res = _oc_config_set_json(
        oc_base=oc_base,
        path="agents.defaults.model",
        value=model_block,
        timeout_seconds=20.0,
        dry_run=dry_run,
    )
    if not set_res.ok:
        print(f"[openclaw_models] FAILED setting agents.defaults.model (exit={set_res.returncode})", file=sys.stderr)
        tail = "\n".join((set_res.stderr or "").splitlines()[-10:])
        if tail:
            print(tail, file=sys.stderr)
        return 1
    print(f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}set agents.defaults.model primary={primary} fallbacks={len(fallbacks)}")

    # Image model defaults: Qwen Vision, plus optional OpenAI image-capable model.
    if bool(args.set_image_defaults):
        explicit_image_primary = (args.image_primary or "").strip()
        if local_only and not explicit_image_primary:
            print(
                "[openclaw_models] OPENCLAW_LOCAL_ONLY=1 skipping agents.defaults.imageModel defaults "
                "(remote vision models disabled)."
            )
        else:
            image_primary = explicit_image_primary or "qwen-portal/vision-model"
            if local_only and not image_primary.startswith("ollama/"):
                print(
                    "[openclaw_models] OPENCLAW_LOCAL_ONLY=1 forcing imageModel primary "
                    f"{image_primary!r} -> {primary!r}"
                )
                image_primary = primary
            image_fallbacks: list[str] = []
            if include_openai and not local_only:
                # Prefer a "4o" family model for image inputs.
                image_openai = (args.openai_image_model or "").strip() or "gpt-4o"
                image_fallbacks.append(f"openai/{image_openai}")
            allow_refs.add(image_primary)
            allow_refs.update(image_fallbacks)
            img_block: dict[str, Any] = {"primary": image_primary}
            if image_fallbacks:
                img_block["fallbacks"] = image_fallbacks
            set_res = _oc_config_set_json(
                oc_base=oc_base,
                path="agents.defaults.imageModel",
                value=img_block,
                timeout_seconds=20.0,
                dry_run=dry_run,
            )
            if not set_res.ok:
                print(f"[openclaw_models] FAILED setting agents.defaults.imageModel (exit={set_res.returncode})", file=sys.stderr)
                tail = "\n".join((set_res.stderr or "").splitlines()[-10:])
                if tail:
                    print(tail, file=sys.stderr)
                return 1
            print(
                f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}set agents.defaults.imageModel "
                f"primary={image_primary} fallbacks={len(image_fallbacks)}"
            )

    # Update model allowlist (agents.defaults.models) to include all referenced refs.
    existing_models = _oc_config_get_json(oc_base=oc_base, path="agents.defaults.models", timeout_seconds=10.0)
    if not isinstance(existing_models, dict):
        existing_models = {}
    changed_models = False
    for ref in sorted([r for r in allow_refs if str(r).strip()]):
        if ref in existing_models:
            continue
        existing_models[ref] = {}
        changed_models = True
    if changed_models:
        set_res = _oc_config_set_json(
            oc_base=oc_base,
            path="agents.defaults.models",
            value=existing_models,
            timeout_seconds=20.0,
            dry_run=dry_run,
        )
        if not set_res.ok:
            print(f"[openclaw_models] FAILED setting agents.defaults.models allowlist (exit={set_res.returncode})", file=sys.stderr)
            tail = "\n".join((set_res.stderr or "").splitlines()[-10:])
            if tail:
                print(tail, file=sys.stderr)
            return 1
        print(f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}updated agents.defaults.models allowlist (+{len(allow_refs)} refs)")

    if bool(args.restart_gateway):
        res = _restart_gateway(oc_base=oc_base, dry_run=dry_run)
        if res.ok:
            print(f"[openclaw_models] {'DRY-RUN ' if dry_run else ''}gateway restart: OK")
            return 0
        print(f"[openclaw_models] gateway restart: FAILED (exit={res.returncode})", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Configure OpenClaw model providers + failover (local Ollama + remote).")
    p.add_argument("--command", default=os.getenv("OPENCLAW_COMMAND", "openclaw"), help='OpenClaw command (default: "openclaw").')
    sub = p.add_subparsers(dest="cmd", required=False)

    ps = sub.add_parser("status", help="Show current OpenClaw model + provider status.")
    ps.add_argument("--list-ollama-models", action="store_true", help="List up to 50 discovered Ollama models.")
    ps.set_defaults(func=_cmd_status)

    pa = sub.add_parser("apply", help="Apply provider + failover configuration.")
    pa.add_argument("--dry-run", action="store_true", help="Compute changes but do not write config/auth.")
    pa.add_argument("--agent-id", default="", help="Agent id for auth store path (default: autodetect).")
    pa.add_argument(
        "--strategy",
        default=os.getenv("OPENCLAW_MODEL_STRATEGY", "local-first"),
        help="Failover strategy: local-first|auto|remote-first|custom. Default: local-first (never uses remote/cloud models).",
    )
    pa.add_argument("--restart-gateway", action="store_true", help="Restart OpenClaw gateway after applying.")

    pa.add_argument("--sync-auth", action="store_true", help="Sync OpenAI/Anthropic keys from .env into OpenClaw auth store.")

    pa.add_argument("--enable-ollama-provider", action="store_true", default=True, help="Ensure models.providers.ollama is configured.")
    pa.add_argument("--ollama-base-url", default="", help="Ollama baseUrl (default: OPENCLAW_OLLAMA_BASE_URL / OLLAMA_HOST / http://127.0.0.1:11434/v1).")
    pa.add_argument("--ollama-models", default="", help="Comma-separated Ollama model names to register (default: discover / defaults).")

    pa.add_argument("--enable-openai-provider", action="store_true", default=False, help="Configure OpenAI provider (requires OPENAI_API_KEY in .env/auth store).")
    pa.add_argument("--openai-models", default="", help="Comma-separated OpenAI model ids to register (default: OPENCLAW_OPENAI_MODELS).")
    pa.add_argument("--openai-primary-model", default="", help="Primary OpenAI model id (for fallbacks).")
    pa.add_argument("--openai-image-model", default="", help="Image-capable OpenAI model id for image fallbacks (default: gpt-4o).")

    pa.add_argument("--enable-anthropic-provider", action="store_true", default=False, help="Configure Anthropic provider (requires ANTHROPIC_API_KEY/CLAUDE_API_KEY).")
    pa.add_argument("--anthropic-models", default="", help="Comma-separated Anthropic model ids to register (default: OPENCLAW_ANTHROPIC_MODELS).")
    pa.add_argument("--anthropic-primary-model", default="", help="Primary Anthropic model id (for fallbacks).")

    pa.add_argument("--set-image-defaults", action="store_true", default=True, help="Set agents.defaults.imageModel (Qwen Vision primary).")
    pa.add_argument("--image-primary", default="", help="Image primary model (default: qwen-portal/vision-model).")
    pa.set_defaults(func=_cmd_apply)

    pu = sub.add_parser("sync-auth", help="Sync OpenAI/Anthropic keys from repo .env into OpenClaw auth store (for gateway service).")
    pu.add_argument("--dry-run", action="store_true", help="Show what would change, without writing.")
    pu.add_argument("--agent-id", default="", help="Agent id for auth store path (default: autodetect).")
    pu.add_argument("--restart-gateway", action="store_true", help="Restart OpenClaw gateway after syncing.")
    pu.set_defaults(func=_cmd_sync_auth)

    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        args.cmd = "status"
        args.func = _cmd_status
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
