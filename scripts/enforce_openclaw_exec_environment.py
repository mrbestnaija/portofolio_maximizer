#!/usr/bin/env python3
"""
Enforce OpenClaw exec host + sandbox + ACP defaults.

Purpose:
- Prevent "requested execution hosts are unavailable/not permitted" failures.
- Keep host selection stable across restarts.
- Ensure ACP sessions can resolve a default agent id.

This script only edits ~/.openclaw/openclaw.json (or a custom --config path).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENCLAW_JSON = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_EXEC_HOST_CONF = PROJECT_ROOT / "config" / "exec_host.conf"

VALID_EXEC_HOSTS = {"sandbox", "gateway", "node"}
VALID_SANDBOX_MODES = {"off", "main", "non-main", "all"}
VALID_SANDBOX_MODES_FOR_SANDBOX_HOST = {"non-main", "all"}
DEFAULT_SANDBOX_FALLBACK_HOST = "gateway"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    child = parent.get(key)
    if isinstance(child, dict):
        return child
    child = {}
    parent[key] = child
    return child


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_exec_host_conf(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != "tools.exec.host":
            continue
        host = str(value or "").strip().lower()
        if host in VALID_EXEC_HOSTS:
            return host
        return ""
    return ""


def _docker_sandbox_available(timeout_seconds: float = 5.0) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return int(proc.returncode) == 0


def _node_host_available(timeout_seconds: float = 6.0) -> bool:
    """Return True if at least one OpenClaw node is paired and reachable."""
    try:
        proc = subprocess.run(
            ["openclaw", "nodes", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    if int(proc.returncode) != 0:
        return False
    try:
        payload = json.loads(proc.stdout or "")
    except Exception:
        return False
    if isinstance(payload, list):
        nodes = payload
    elif isinstance(payload, dict):
        nodes = payload.get("nodes") or []
    else:
        return False
    return bool(nodes)


def _agent_ids(cfg: dict[str, Any]) -> list[str]:
    agents = _as_dict(cfg.get("agents"))
    rows = agents.get("list")
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        aid = str(row.get("id") or "").strip()
        if not aid:
            continue
        if aid not in out:
            out.append(aid)
    return out


def _default_agent_from_list(cfg: dict[str, Any]) -> str:
    agents = _as_dict(cfg.get("agents"))
    rows = agents.get("list")
    if not isinstance(rows, list):
        return ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("default")):
            aid = str(row.get("id") or "").strip()
            if aid:
                return aid
    return ""


def _agent_allows_exec(agent: dict[str, Any]) -> bool:
    tools = _as_dict(agent.get("tools"))
    deny = {str(item).strip().lower() for item in _as_list(tools.get("deny")) if str(item).strip()}
    if "exec" in deny or "group:runtime" in deny:
        return False

    allow = {str(item).strip().lower() for item in _as_list(tools.get("allow")) if str(item).strip()}
    if allow:
        return "exec" in allow or "group:runtime" in allow

    profile = str(tools.get("profile") or "").strip().lower()
    return profile != "messaging"


def _pick_acp_default_agent(cfg: dict[str, Any], preferred_agent: str) -> str:
    agent_ids = _agent_ids(cfg)
    valid_ids = set(agent_ids)

    preferred = str(preferred_agent or "").strip()
    if preferred and (not valid_ids or preferred in valid_ids):
        return preferred

    acp = _as_dict(cfg.get("acp"))
    current = str(acp.get("defaultAgent") or "").strip()
    if current and (not valid_ids or current in valid_ids):
        return current

    listed_default = _default_agent_from_list(cfg)
    if listed_default:
        return listed_default

    if "ops" in valid_ids:
        return "ops"
    if "training" in valid_ids:
        return "training"
    if agent_ids:
        return agent_ids[0]
    return "ops"


def enforce_exec_environment(
    cfg: dict[str, Any],
    *,
    preferred_host: str,
    sandbox_mode_for_sandbox_host: str,
    ensure_acp_default_agent: bool,
    preferred_agent: str,
) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(_as_dict(cfg))
    changes: list[str] = []

    host = str(preferred_host or "").strip().lower()
    if host not in VALID_EXEC_HOSTS:
        host = "sandbox"

    tools = _ensure_dict(out, "tools")
    exec_cfg = _ensure_dict(tools, "exec")
    current_host = str(exec_cfg.get("host") or "").strip().lower()
    if current_host != host:
        exec_cfg["host"] = host
        changes.append(f"tools.exec.host:{current_host or '<unset>'}->{host}")

    if host == "sandbox":
        agents = _ensure_dict(out, "agents")
        defaults = _ensure_dict(agents, "defaults")
        sandbox = _ensure_dict(defaults, "sandbox")

        requested_mode = str(sandbox_mode_for_sandbox_host or "").strip().lower()
        mode_target = (
            requested_mode if requested_mode in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST else "non-main"
        )
        current_mode = str(sandbox.get("mode") or "").strip().lower()
        if current_mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
            sandbox["mode"] = mode_target
            changes.append(f"agents.defaults.sandbox.mode:{current_mode or '<unset>'}->{mode_target}")
        elif current_mode != mode_target:
            sandbox["mode"] = mode_target
            changes.append(f"agents.defaults.sandbox.mode:{current_mode}->{mode_target}")

        current_scope = str(sandbox.get("scope") or "").strip().lower()
        if not current_scope:
            sandbox["scope"] = "agent"
            changes.append("agents.defaults.sandbox.scope:<unset>->agent")

        for agent in _as_list(agents.get("list")):
            if not isinstance(agent, dict) or not _agent_allows_exec(agent):
                continue
            aid = str(agent.get("id") or "?").strip() or "?"
            agent_sandbox = _ensure_dict(agent, "sandbox")
            current_agent_mode = str(agent_sandbox.get("mode") or "").strip().lower()
            if current_agent_mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
                agent_sandbox["mode"] = mode_target
                changes.append(
                    f"agents.list[{aid}].sandbox.mode:{current_agent_mode or '<unset>'}->{mode_target}"
                )
            elif current_agent_mode != mode_target:
                agent_sandbox["mode"] = mode_target
                changes.append(f"agents.list[{aid}].sandbox.mode:{current_agent_mode}->{mode_target}")

            current_agent_scope = str(agent_sandbox.get("scope") or "").strip().lower()
            if not current_agent_scope:
                agent_sandbox["scope"] = "agent"
                changes.append(f"agents.list[{aid}].sandbox.scope:<unset>->agent")

    if ensure_acp_default_agent:
        acp = _ensure_dict(out, "acp")
        agent = _pick_acp_default_agent(out, preferred_agent)
        current = str(acp.get("defaultAgent") or "").strip()
        if current != agent:
            acp["defaultAgent"] = agent
            changes.append(f"acp.defaultAgent:{current or '<unset>'}->{agent}")

    return out, changes


def enforce_config_file(
    *,
    config_path: Path,
    preferred_host: str,
    sandbox_mode_for_sandbox_host: str,
    ensure_acp_default_agent: bool,
    preferred_agent: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not config_path.exists():
        return {
            "ok": False,
            "changed": False,
            "error": f"config_missing:{config_path}",
            "config_path": str(config_path),
            "changes": [],
        }

    try:
        cfg = _read_json(config_path)
    except Exception as exc:
        return {
            "ok": False,
            "changed": False,
            "error": f"config_parse_error:{exc}",
            "config_path": str(config_path),
            "changes": [],
        }

    requested_host = str(preferred_host or "").strip().lower()
    if requested_host not in VALID_EXEC_HOSTS:
        requested_host = _parse_exec_host_conf(DEFAULT_EXEC_HOST_CONF) or "sandbox"

    effective_host = requested_host
    fallback_reason = ""
    if effective_host == "sandbox" and not _docker_sandbox_available():
        effective_host = DEFAULT_SANDBOX_FALLBACK_HOST
        fallback_reason = "docker_sandbox_unavailable"
    elif effective_host == "node" and not _node_host_available():
        effective_host = "gateway"
        fallback_reason = "node_host_unavailable"

    updated, changes = enforce_exec_environment(
        cfg,
        preferred_host=effective_host,
        sandbox_mode_for_sandbox_host=sandbox_mode_for_sandbox_host,
        ensure_acp_default_agent=ensure_acp_default_agent,
        preferred_agent=preferred_agent,
    )
    changed = bool(changes)

    if changed and not dry_run:
        try:
            _write_json(config_path, updated)
        except Exception as exc:
            return {
                "ok": False,
                "changed": False,
                "error": f"config_write_error:{exc}",
                "config_path": str(config_path),
                "changes": changes,
            }

    return {
        "ok": True,
        "changed": changed,
        "dry_run": bool(dry_run),
        "config_path": str(config_path),
        "requested_host": requested_host,
        "host": effective_host,
        "fallback_reason": fallback_reason,
        "changes": changes,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_OPENCLAW_JSON), help="Path to openclaw.json.")
    parser.add_argument(
        "--host",
        default="",
        choices=sorted(list(VALID_EXEC_HOSTS)) + [""],
        help="Preferred tools.exec.host. Default: from config/exec_host.conf, else sandbox.",
    )
    parser.add_argument(
        "--sandbox-mode",
        default="non-main",
        choices=sorted(VALID_SANDBOX_MODES),
        help="Sandbox mode to enforce when host=sandbox.",
    )
    parser.add_argument(
        "--agent-id",
        default="",
        help="Preferred ACP default agent id (must exist in agents.list when present).",
    )
    parser.add_argument(
        "--no-acp-default-agent",
        action="store_true",
        help="Do not enforce acp.defaultAgent.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    report = enforce_config_file(
        config_path=Path(str(args.config)).expanduser(),
        preferred_host=str(args.host or "").strip().lower(),
        sandbox_mode_for_sandbox_host=str(args.sandbox_mode or "non-main").strip().lower(),
        ensure_acp_default_agent=not bool(args.no_acp_default_agent),
        preferred_agent=str(args.agent_id or "").strip(),
        dry_run=bool(args.dry_run),
    )

    if bool(args.json):
        print(json.dumps(report, indent=2))
    else:
        if not report.get("ok"):
            print(f"[enforce_openclaw_exec_environment] ERROR: {report.get('error')}")
            return 1
        print(
            "[enforce_openclaw_exec_environment] "
            f"{'DRY-RUN ' if report.get('dry_run') else ''}"
            f"changed={int(bool(report.get('changed')))} host={report.get('host')} "
            f"path={report.get('config_path')}"
        )
        if report.get("fallback_reason"):
            print(
                "[enforce_openclaw_exec_environment] "
                f"requested_host={report.get('requested_host')} "
                f"fallback_reason={report.get('fallback_reason')}"
            )
        for item in report.get("changes", []):
            print(f"[enforce_openclaw_exec_environment] {item}")
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
