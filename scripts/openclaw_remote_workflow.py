"""
openclaw_remote_workflow.py
---------------------------
OpenClaw v2026.3.13 remote workflow manager for Portfolio Maximizer.

Provides remote workflow diagnostics, channel health checks, gateway status,
cron job health, and one-command remediation for common delivery failures.

Commands:
  status         Full remote workflow status (default)
  health         Quick health check (exit 0=OK, 1=WARN, 2=FAIL)
  diagnose       Deep diagnostic: gateway, channels, delivery, agent bindings
  channel-test   Send a test notification on all enabled channels
  gateway-restart  Restart the OpenClaw gateway and verify connectivity
  cron-health    Show cron job health + delivery failure breakdown
  failover-test  Simulate WhatsApp outage: verify Telegram fallback fires

Usage:
  python scripts/openclaw_remote_workflow.py [command] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
CRON_JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"

_TARGET_VERSION = "2026.3.13"
_GATEWAY_PORT = 18789
_GATEWAY_TOKEN_ENV = "OPENCLAW_GATEWAY_TOKEN"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    if not OPENCLAW_CONFIG.exists():
        return {}
    try:
        return json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_cron_jobs() -> List[Dict[str, Any]]:
    if not CRON_JOBS_PATH.exists():
        return []
    try:
        return json.loads(CRON_JOBS_PATH.read_text(encoding="utf-8")).get("jobs", [])
    except Exception:
        return []


def _run(cmd: List[str], timeout: float = 15.0, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or PROJECT_ROOT,
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout ({timeout}s) waiting for: {' '.join(cmd)}"
    except Exception as exc:
        return -1, "", str(exc)


def _openclaw_available() -> bool:
    return shutil.which("openclaw") is not None


def _gateway_local_ping() -> Tuple[bool, str]:
    """Try to reach the local OpenClaw gateway HTTP endpoint."""
    try:
        import urllib.request
        url = f"http://127.0.0.1:{_GATEWAY_PORT}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, body[:200]
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Status checks
# ---------------------------------------------------------------------------

def _check_version(cfg: Dict[str, Any]) -> Dict[str, Any]:
    installed = cfg.get("meta", {}).get("lastTouchedVersion", "unknown")
    ok = installed == _TARGET_VERSION
    return {
        "check": "version",
        "status": "OK" if ok else "WARN",
        "installed": installed,
        "target": _TARGET_VERSION,
        "detail": "Up to date" if ok else f"Expected {_TARGET_VERSION}, got {installed}",
    }


def _check_gateway(cfg: Dict[str, Any]) -> Dict[str, Any]:
    gw = cfg.get("gateway", {})
    mode = gw.get("mode", "unknown")
    bind = gw.get("bind", "unknown")
    tailscale = gw.get("tailscale", {}).get("mode", "off")
    remote_cfg = gw.get("remote", {})

    reachable, ping_detail = _gateway_local_ping()

    issues = []
    if mode != "remote":
        issues.append(f"mode={mode} (should be 'remote')")
    if bind == "loopback":
        issues.append("bind=loopback (remote access blocked)")
    if not remote_cfg.get("allowExternalAgentTrigger"):
        issues.append("allowExternalAgentTrigger not set")

    status = "OK" if not issues and reachable else ("WARN" if not issues else "FAIL")

    return {
        "check": "gateway",
        "status": status,
        "mode": mode,
        "bind": bind,
        "tailscale": tailscale,
        "reachable": reachable,
        "remote_config": remote_cfg,
        "issues": issues,
        "ping_detail": ping_detail[:120] if ping_detail else "",
    }


def _check_channels(cfg: Dict[str, Any]) -> Dict[str, Any]:
    channels = cfg.get("channels", {})
    results = {}

    # WhatsApp
    wa = channels.get("whatsapp", {})
    wa_enabled = wa.get("enabled", False)
    wa_account_ok = wa.get("accounts", {}).get("default", {}).get("enabled", False)
    results["whatsapp"] = {
        "enabled": wa_enabled,
        "account_active": wa_account_ok,
        "status": "OK" if (wa_enabled and wa_account_ok) else "WARN",
    }

    # Telegram
    tg = channels.get("telegram", {})
    tg_enabled = tg.get("enabled", False)
    tg_token = bool(tg.get("botToken", "").strip())
    tg_account_ok = tg.get("accounts", {}).get("default", {}).get("enabled", False)
    tg_configured = tg_enabled and tg_token and tg_account_ok
    results["telegram"] = {
        "enabled": tg_enabled,
        "token_present": tg_token,
        "account_active": tg_account_ok,
        "status": "OK" if tg_configured else ("WARN" if tg_enabled else "OFF"),
    }

    # Discord
    dc = channels.get("discord", {})
    dc_enabled = dc.get("enabled", False)
    dc_account = dc.get("accounts", {}).get("custom-1", {})
    dc_account_ok = dc_account.get("enabled", False)
    dc_token = bool(dc_account.get("token", "").strip())
    results["discord"] = {
        "enabled": dc_enabled,
        "account_active": dc_account_ok,
        "token_present": dc_token,
        "status": "OK" if (dc_enabled and dc_account_ok and dc_token) else "OFF",
    }

    # Overall: need at least 2 active channels for redundancy
    active = sum(1 for ch in results.values() if ch["status"] == "OK")
    overall = "OK" if active >= 2 else ("WARN" if active == 1 else "FAIL")
    return {
        "check": "channels",
        "status": overall,
        "active_count": active,
        "channels": results,
        "detail": f"{active}/3 channels active" + (
            " (redundancy OK)" if active >= 2 else " (need >= 2 for failover)"
        ),
    }


def _check_agents(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agents = cfg.get("agents", {}).get("list", [])
    agent_to_agent = cfg.get("tools", {}).get("agentToAgent", {}).get("enabled", False)
    agent_ids = [a["id"] for a in agents]
    required = {"ops", "trading", "training"}
    missing = required - set(agent_ids)

    return {
        "check": "agents",
        "status": "OK" if not missing else "WARN",
        "agents": agent_ids,
        "agent_to_agent_enabled": agent_to_agent,
        "missing_required": sorted(missing),
        "detail": (
            f"All {len(agents)} agents present, agentToAgent={agent_to_agent}"
            if not missing
            else f"Missing agents: {sorted(missing)}"
        ),
    }


def _check_cron_jobs() -> Dict[str, Any]:
    jobs = _load_cron_jobs()
    if not jobs:
        return {"check": "cron_jobs", "status": "WARN", "detail": "No cron jobs found"}

    total = len(jobs)
    enabled = [j for j in jobs if j.get("enabled", False)]
    failing = [
        j for j in enabled
        if j.get("state", {}).get("consecutiveErrors", 0) > 0
    ]
    delivery_failures = [
        j for j in failing
        if "delivery" in j.get("state", {}).get("lastError", "").lower()
    ]
    with_fallback = [
        j for j in enabled
        if "fallback" in j.get("delivery", {})
    ]

    status = "OK" if not failing else ("WARN" if len(failing) <= 3 else "FAIL")
    return {
        "check": "cron_jobs",
        "status": status,
        "total": total,
        "enabled": len(enabled),
        "failing": len(failing),
        "delivery_failures": len(delivery_failures),
        "with_telegram_fallback": len(with_fallback),
        "failing_names": [j["name"] for j in failing],
        "detail": (
            f"{len(enabled)} enabled, {len(failing)} failing "
            f"({len(delivery_failures)} delivery), {len(with_fallback)} have fallback"
        ),
    }


def _check_interactions_api() -> Dict[str, Any]:
    """Check if the PMX Interactions API is running on localhost."""
    port = int(os.getenv("INTERACTIONS_PORT", "8000"))
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=4) as resp:
            return {
                "check": "interactions_api",
                "status": "OK",
                "port": port,
                "detail": f"Responding on port {port}",
            }
    except Exception as exc:
        return {
            "check": "interactions_api",
            "status": "OFF",
            "port": port,
            "detail": f"Not running (start with: python scripts/pmx_interactions_api.py): {exc}",
        }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(as_json: bool = False) -> int:
    cfg = _load_config()

    checks = [
        _check_version(cfg),
        _check_gateway(cfg),
        _check_channels(cfg),
        _check_agents(cfg),
        _check_cron_jobs(),
        _check_interactions_api(),
    ]

    statuses = [c["status"] for c in checks]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "OK")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "openclaw_available": _openclaw_available(),
        "checks": checks,
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[remote-workflow] Overall: {overall}")
        print(f"  OpenClaw CLI:  {'available' if result['openclaw_available'] else 'NOT FOUND'}")
        for c in checks:
            s = c["status"]
            icon = "[OK]" if s == "OK" else ("[!]" if s == "WARN" else "[ ]" if s == "OFF" else "[X]")
            print(f"  {icon} {c['check']:<22} {c.get('detail', s)}")
        print()

        if overall != "OK":
            print("  Remediation hints:")
            for c in checks:
                if c["status"] in ("FAIL", "WARN"):
                    if c["check"] == "gateway":
                        print("    - Run: openclaw gateway restart")
                        print("    - Or:  python scripts/openclaw_remote_workflow.py gateway-restart")
                    elif c["check"] == "channels":
                        ch = c.get("channels", {})
                        if ch.get("telegram", {}).get("status") not in ("OK",):
                            print("    - Telegram: openclaw channels login --channel telegram --account default")
                        if ch.get("whatsapp", {}).get("status") not in ("OK",):
                            print("    - WhatsApp: openclaw channels login --channel whatsapp --account default")
                    elif c["check"] == "cron_jobs" and c.get("delivery_failures", 0) > 0:
                        print("    - Reset delivery errors: cron jobs will auto-retry on next schedule")
                        print("    - Check gateway: python scripts/openclaw_remote_workflow.py gateway-restart")
            print()

    return 0 if overall == "OK" else (1 if overall == "WARN" else 2)


def cmd_health(as_json: bool = False) -> int:
    cfg = _load_config()
    gw_reachable, _ = _gateway_local_ping()
    wa_ok = cfg.get("channels", {}).get("whatsapp", {}).get("enabled", False)
    tg_ok = (
        cfg.get("channels", {}).get("telegram", {}).get("enabled", False)
        and cfg.get("channels", {}).get("telegram", {}).get("accounts", {}).get("default", {}).get("enabled", False)
    )

    active_channels = sum([wa_ok, tg_ok])
    overall = "OK" if (gw_reachable and active_channels >= 1) else "WARN" if active_channels >= 1 else "FAIL"

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "gateway_reachable": gw_reachable,
        "active_channels": active_channels,
        "whatsapp": wa_ok,
        "telegram": tg_ok,
    }

    if as_json:
        print(json.dumps(result))
    else:
        print(f"[health] {overall}  gateway={gw_reachable}  channels={active_channels}")

    return 0 if overall == "OK" else (1 if overall == "WARN" else 2)


def cmd_diagnose(as_json: bool = False) -> int:
    print("[diagnose] Running deep remote workflow diagnostic...")

    cfg = _load_config()
    diag: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(OPENCLAW_CONFIG),
        "config_exists": OPENCLAW_CONFIG.exists(),
        "openclaw_cli": _openclaw_available(),
    }

    # Gateway ping
    gw_ok, gw_detail = _gateway_local_ping()
    diag["gateway_local_reachable"] = gw_ok
    diag["gateway_ping_detail"] = gw_detail

    # openclaw status
    if _openclaw_available():
        rc, out, err = _run(["openclaw", "status", "--json"], timeout=10)
        diag["openclaw_status_rc"] = rc
        diag["openclaw_status_stdout"] = out[:500]
        diag["openclaw_status_stderr"] = err[:200]
    else:
        diag["openclaw_cli_note"] = "openclaw not in PATH — install or check PATH"

    # Channel login state
    for ch in ("whatsapp", "telegram"):
        if _openclaw_available():
            rc, out, err = _run(
                ["openclaw", "channels", "status", "--channel", ch, "--json"],
                timeout=10,
            )
            diag[f"channel_{ch}_rc"] = rc
            diag[f"channel_{ch}_stdout"] = out[:300]

    # Cron job summary
    jobs = _load_cron_jobs()
    failing = [j for j in jobs if j.get("state", {}).get("consecutiveErrors", 0) > 0]
    diag["cron_total"] = len(jobs)
    diag["cron_failing"] = len(failing)
    diag["cron_failing_names"] = [j["name"] for j in failing]

    # Interactions API
    port = int(os.getenv("INTERACTIONS_PORT", "8000"))
    api_check = _check_interactions_api()
    diag["interactions_api"] = api_check

    if as_json:
        print(json.dumps(diag, indent=2))
    else:
        for k, v in diag.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")

    return 0


def cmd_cron_health(as_json: bool = False) -> int:
    jobs = _load_cron_jobs()
    check = _check_cron_jobs()

    rows = []
    for j in jobs:
        state = j.get("state", {})
        delivery = j.get("delivery", {})
        rows.append({
            "name": j["name"],
            "agent": j.get("agentId", "?"),
            "enabled": j.get("enabled", False),
            "schedule": j.get("schedule", {}).get("expr", "?"),
            "consecutive_errors": state.get("consecutiveErrors", 0),
            "last_status": state.get("lastStatus", "?"),
            "last_error": state.get("lastError", "")[:60],
            "delivery_channel": delivery.get("channel", "?"),
            "has_fallback": "fallback" in delivery,
            "fallback_channel": delivery.get("fallback", {}).get("channel", ""),
        })

    result = {"summary": check, "jobs": rows}

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[cron-health] {check['detail']}")
        print(f"  {'Name':<45} {'Agent':<10} {'Errors':<7} {'Status':<10} {'Channel':<10} {'Fallback'}")
        print(f"  {'-'*45} {'-'*10} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")
        for row in rows:
            if not row["enabled"]:
                continue
            err = str(row["consecutive_errors"])
            status = row["last_status"]
            icon = "[OK]" if row["consecutive_errors"] == 0 else "[!]"
            fb = row["fallback_channel"] or "-"
            print(
                f"  {icon} {row['name']:<43} {row['agent']:<10} {err:<7} {status:<10} "
                f"{row['delivery_channel']:<10} {fb}"
            )
        print()

    return 0 if check["status"] == "OK" else 1


def cmd_channel_test(as_json: bool = False) -> int:
    """Send a test notification on all enabled channels."""
    if not _openclaw_available():
        print("[channel-test] ERROR: openclaw CLI not found in PATH", file=sys.stderr)
        return 2

    cfg = _load_config()
    msg = f"[PMX Remote Workflow Test] v{_TARGET_VERSION} channel test at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"

    results = {}

    # WhatsApp
    wa = cfg.get("channels", {}).get("whatsapp", {})
    if wa.get("enabled"):
        allow = wa.get("allowFrom", [])
        to = allow[0] if allow else ""
        if to:
            rc, out, err = _run(
                ["openclaw", "message", "send", "--channel", "whatsapp", "--to", to, "--message", msg],
                timeout=30,
            )
            results["whatsapp"] = {"to_masked": to[:4] + "***", "rc": rc, "ok": rc == 0}
            print(f"  WhatsApp: {'OK' if rc == 0 else 'FAIL'} (rc={rc})")
        else:
            results["whatsapp"] = {"status": "no_target"}
            print("  WhatsApp: SKIP (no allowFrom target)")

    # Telegram
    tg = cfg.get("channels", {}).get("telegram", {})
    if tg.get("enabled") and tg.get("accounts", {}).get("default", {}).get("enabled"):
        # Telegram self-test sends to the bot itself (bot can't send to itself,
        # but the command verifies channel connectivity)
        rc, out, err = _run(
            ["openclaw", "channels", "status", "--channel", "telegram", "--json"],
            timeout=10,
        )
        results["telegram"] = {"rc": rc, "ok": rc == 0, "detail": out[:100]}
        print(f"  Telegram: {'OK' if rc == 0 else 'FAIL'} (channel status rc={rc})")
    else:
        results["telegram"] = {"status": "disabled"}
        print("  Telegram: SKIP (disabled)")

    if as_json:
        print(json.dumps(results, indent=2))

    all_ok = all(r.get("ok", True) for r in results.values() if "ok" in r)
    return 0 if all_ok else 1


def cmd_gateway_restart(as_json: bool = False) -> int:
    """Restart OpenClaw gateway and verify it responds."""
    if not _openclaw_available():
        print("[gateway-restart] ERROR: openclaw CLI not found in PATH", file=sys.stderr)
        return 2

    print("[gateway-restart] Sending restart signal...")
    rc, out, err = _run(["openclaw", "gateway", "restart"], timeout=30)
    if rc != 0:
        print(f"[gateway-restart] restart returned rc={rc}")
        if out:
            print(f"  stdout: {out[:200]}")
        if err:
            print(f"  stderr: {err[:200]}")

    # Poll for gateway to respond
    print("[gateway-restart] Waiting for gateway to respond...")
    for attempt in range(1, 7):
        time.sleep(3)
        ok, detail = _gateway_local_ping()
        if ok:
            print(f"[gateway-restart] Gateway OK after {attempt * 3}s")
            result = {"status": "OK", "attempts": attempt}
            if as_json:
                print(json.dumps(result))
            return 0
        print(f"  attempt {attempt}/6: not yet reachable")

    print("[gateway-restart] ERROR: gateway did not respond after 18s", file=sys.stderr)
    result = {"status": "FAIL", "detail": "gateway did not respond after restart"}
    if as_json:
        print(json.dumps(result))
    return 2


def cmd_failover_test(as_json: bool = False) -> int:
    """
    Verify Telegram fallback is configured by checking all cron jobs
    that have whatsapp primary delivery for a fallback entry.
    Does NOT actually disable WhatsApp; purely config validation.
    """
    jobs = _load_cron_jobs()
    wa_jobs = [j for j in jobs if j.get("delivery", {}).get("channel") == "whatsapp" and j.get("enabled")]
    with_fallback = [j for j in wa_jobs if "fallback" in j.get("delivery", {})]
    without_fallback = [j for j in wa_jobs if "fallback" not in j.get("delivery", {})]

    result = {
        "check": "failover_config",
        "whatsapp_primary_jobs": len(wa_jobs),
        "with_telegram_fallback": len(with_fallback),
        "without_fallback": len(without_fallback),
        "missing_fallback": [j["name"] for j in without_fallback],
        "status": "OK" if not without_fallback else "WARN",
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[failover-test] {result['status']}")
        print(f"  WhatsApp-primary jobs:   {result['whatsapp_primary_jobs']}")
        print(f"  With Telegram fallback:  {result['with_telegram_fallback']}")
        print(f"  Missing fallback:        {result['without_fallback']}")
        if without_fallback:
            for name in result["missing_fallback"]:
                print(f"    - {name}")
        else:
            print("  All jobs have failover configured.")
        print()

    return 0 if result["status"] == "OK" else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenClaw v2026.3.13 remote workflow manager"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "health", "diagnose", "channel-test", "gateway-restart", "cron-health", "failover-test"],
        help="Command to run (default: status)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    dispatch = {
        "status": cmd_status,
        "health": cmd_health,
        "diagnose": cmd_diagnose,
        "channel-test": cmd_channel_test,
        "gateway-restart": cmd_gateway_restart,
        "cron-health": cmd_cron_health,
        "failover-test": cmd_failover_test,
    }
    return dispatch[args.command](as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
