#!/usr/bin/env python3
"""
Interactive OpenClaw WhatsApp relink helper.

Why this exists:
- Force a fresh pairing token without manual JSON editing.
- Preserve old auth sessions by rotating to a new authDir.
- Auto-apply a local OpenClaw runtime hotfix for known WA 405 handshake failures.
- Run interactive login, then verify linked/connected state.

Examples:
    python scripts/openclaw_whatsapp_relink.py --fresh-auth
    python scripts/openclaw_whatsapp_relink.py --fresh-auth --json
    python scripts/openclaw_whatsapp_relink.py --fresh-auth --force-wa-version-hotfix
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OPENCLAW_JSON = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_PROBE_URLS = (
    "http://127.0.0.1:18789/channels",
    "https://openclaw.ai/",
)
WA_SESSION_FILE_GLOBS = ("session-*.js", "plugin-sdk/session-*.js")
WA_HOTFIX_MARKER = "fetchLatestWaWebVersion failed"
WA_IMPORT_PATTERN = re.compile(
    r"fetchLatestBaileysVersion,\s*makeCacheableSignalKeyStore",
    flags=re.MULTILINE,
)
WA_VERSION_PARTIAL_PATTERN = re.compile(
    r"const latestWa = await fetchLatestWaWebVersion\(\);\s*const \{ version \} = latestWa\?\.isLatest \? latestWa : await fetchLatestBaileysVersion\(\);",
    flags=re.MULTILINE,
)
WA_VERSION_OLD_PATTERN = re.compile(
    r"const \{ version \} = await fetchLatestBaileysVersion\(\);",
    flags=re.MULTILINE,
)
WA_VERSION_HOTFIX_BLOCK = (
    "\tlet version;\n"
    "\ttry {\n"
    "\t\tconst latestWa = await fetchLatestWaWebVersion();\n"
    "\t\tconst waVersion = latestWa?.version;\n"
    "\t\tif (Array.isArray(waVersion) && waVersion.length >= 2) version = waVersion;\n"
    "\t} catch (err) {\n"
    "\t\tsessionLogger.warn({ error: String(err) }, \"fetchLatestWaWebVersion failed\");\n"
    "\t}\n"
    "\tif (!Array.isArray(version) || version.length < 2) {\n"
    "\t\ttry {\n"
    "\t\t\tconst latest = await fetchLatestBaileysVersion();\n"
    "\t\t\tconst baVersion = latest?.version;\n"
    "\t\t\tif (Array.isArray(baVersion) && baVersion.length >= 2) version = baVersion;\n"
    "\t\t} catch (err) {\n"
    "\t\t\tsessionLogger.warn({ error: String(err) }, \"fetchLatestBaileysVersion failed\");\n"
    "\t\t}\n"
    "\t}\n"
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _http_probe(url: str, *, timeout_seconds: float = 8.0) -> dict[str, Any]:
    target = str(url or "").strip()
    out: dict[str, Any] = {
        "url": target,
        "reachable": False,
        "status": None,
        "error": "",
    }
    if not target:
        out["error"] = "empty_url"
        return out
    req = urllib.request.Request(target, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_seconds))) as response:
            status = int(getattr(response, "status", 0) or 0)
            out["status"] = status
            out["reachable"] = True
    except urllib.error.HTTPError as exc:
        out["status"] = int(getattr(exc, "code", 0) or 0)
        # HTTP errors still prove the URL is reachable over TCP/TLS.
        out["reachable"] = True
        out["error"] = str(exc)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _split_command(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return ["openclaw"]
    try:
        parts = shlex.split(raw, posix=False)
    except Exception:
        parts = [raw]
    return [p for p in parts if str(p).strip()]


def _run_capture(*, oc_base: list[str], args: list[str], timeout_seconds: float = 20.0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")
    raw_cmd = [*oc_base, *args]
    if os.name == "nt":
        cmd = ["cmd", "/d", "/s", "/c", subprocess.list2cmdline(raw_cmd)]
    else:
        cmd = raw_cmd
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout_seconds)),
        env=env,
    )


def _run_interactive(*, oc_base: list[str], args: list[str]) -> int:
    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")
    raw_cmd = [*oc_base, *args]
    if os.name == "nt":
        cmd = ["cmd", "/d", "/s", "/c", subprocess.list2cmdline(raw_cmd)]
    else:
        cmd = raw_cmd
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def _extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        maybe = raw[start : end + 1]
        try:
            return json.loads(maybe)
        except Exception:
            return None
    return None


def _load_openclaw_cfg() -> dict[str, Any]:
    if not OPENCLAW_JSON.exists():
        return {}
    try:
        return json.loads(OPENCLAW_JSON.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _current_auth_dir(cfg: dict[str, Any], account_id: str) -> str:
    wa = cfg.get("channels", {}).get("whatsapp", {})
    accounts = wa.get("accounts", {}) if isinstance(wa, dict) else {}
    acct = accounts.get(account_id, {}) if isinstance(accounts, dict) else {}
    if isinstance(acct, dict):
        auth_dir = str(acct.get("authDir", "") or "").strip()
        if auth_dir:
            return auth_dir
    return str(Path.home() / ".openclaw" / "credentials" / "whatsapp" / account_id)


def _write_auth_dir_fallback(*, account_id: str, auth_dir: str) -> tuple[bool, str]:
    cfg = _load_openclaw_cfg()
    if not isinstance(cfg, dict) or not cfg:
        return False, "openclaw_config_unavailable"
    try:
        channels = cfg.setdefault("channels", {})
        if not isinstance(channels, dict):
            return False, "channels_not_object"
        whatsapp = channels.setdefault("whatsapp", {})
        if not isinstance(whatsapp, dict):
            return False, "whatsapp_not_object"
        accounts = whatsapp.setdefault("accounts", {})
        if not isinstance(accounts, dict):
            return False, "accounts_not_object"
        account = accounts.setdefault(account_id, {})
        if not isinstance(account, dict):
            return False, "account_not_object"
        account["authDir"] = str(auth_dir)
        OPENCLAW_JSON.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _status_snapshot(payload: Any, account_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    channel = payload.get("channels", {}).get("whatsapp", {})
    accounts = payload.get("channelAccounts", {}).get("whatsapp", [])
    acct_obj = {}
    if isinstance(accounts, list):
        for row in accounts:
            if isinstance(row, dict) and str(row.get("accountId", "")).strip() == account_id:
                acct_obj = row
                break
    if not acct_obj and isinstance(accounts, list) and accounts and isinstance(accounts[0], dict):
        acct_obj = accounts[0]
    return {
        "configured": bool(acct_obj.get("configured", channel.get("configured", False))),
        "linked": bool(acct_obj.get("linked", channel.get("linked", False))),
        "running": bool(acct_obj.get("running", channel.get("running", False))),
        "connected": bool(acct_obj.get("connected", channel.get("connected", False))),
        "lastError": str(acct_obj.get("lastError", channel.get("lastError", "")) or ""),
        "reconnectAttempts": int(acct_obj.get("reconnectAttempts", channel.get("reconnectAttempts", 0)) or 0),
    }


def _latest_openclaw_log() -> Path | None:
    local_app = Path(os.getenv("LOCALAPPDATA", "")).expanduser()
    if not str(local_app):
        return None
    log_dir = local_app / "Temp" / "openclaw"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("openclaw-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _tail_lines(path: Path, *, max_lines: int = 400) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-max(1, int(max_lines)) :]


def _resolve_openclaw_dist_dir() -> Path | None:
    candidates: list[Path] = []
    app_data = str(os.getenv("APPDATA", "")).strip()
    if app_data:
        candidates.append(Path(app_data) / "npm" / "node_modules" / "openclaw" / "dist")
    local_app = str(os.getenv("LOCALAPPDATA", "")).strip()
    if local_app:
        candidates.append(Path(local_app) / "Programs" / "OpenClaw" / "resources" / "app" / "dist")
        candidates.append(Path(local_app) / "Programs" / "openclaw" / "resources" / "app" / "dist")
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_session_runtime_files(dist_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in WA_SESSION_FILE_GLOBS:
        files.extend(sorted(dist_dir.glob(pattern)))
    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path)] = path
    return list(unique.values())


def _patch_session_runtime_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "status": "noop",
        "patched": False,
        "backup": "",
    }
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        result["status"] = "read_failed"
        result["error"] = str(exc)
        return result

    if "fetchLatestBaileysVersion" not in original and "fetchLatestWaWebVersion" not in original:
        result["status"] = "not_whatsapp_session_file"
        return result

    if WA_HOTFIX_MARKER in original:
        result["status"] = "already_patched"
        return result

    updated = original
    import_changes = 0
    if "fetchLatestWaWebVersion" not in updated:
        updated, import_changes = WA_IMPORT_PATTERN.subn(
            "fetchLatestBaileysVersion, fetchLatestWaWebVersion, makeCacheableSignalKeyStore",
            updated,
            count=1,
        )
        if import_changes == 0:
            result["status"] = "import_pattern_not_found"
            return result

    updated, version_changes = WA_VERSION_PARTIAL_PATTERN.subn(
        WA_VERSION_HOTFIX_BLOCK,
        updated,
        count=1,
    )
    if version_changes == 0:
        updated, version_changes = WA_VERSION_OLD_PATTERN.subn(
            WA_VERSION_HOTFIX_BLOCK,
            updated,
            count=1,
        )
    if version_changes == 0:
        result["status"] = "version_pattern_not_found"
        return result

    if updated == original:
        result["status"] = "noop"
        return result

    backup = path.with_name(f"{path.name}.pmxbak")
    try:
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
    except Exception as exc:
        result["status"] = "write_failed"
        result["error"] = str(exc)
        return result

    result["status"] = "patched"
    result["patched"] = True
    result["backup"] = str(backup)
    return result


def _apply_wa_version_hotfix() -> dict[str, Any]:
    report: dict[str, Any] = {
        "applied": False,
        "dist_dir": "",
        "scanned_files": 0,
        "patched_files": 0,
        "already_patched_files": 0,
        "errors": 0,
        "details": [],
    }
    dist_dir = _resolve_openclaw_dist_dir()
    if dist_dir is None:
        report["reason"] = "openclaw_dist_not_found"
        return report
    report["dist_dir"] = str(dist_dir)

    files = _resolve_session_runtime_files(dist_dir)
    report["scanned_files"] = len(files)
    if not files:
        report["reason"] = "session_runtime_files_not_found"
        return report

    details: list[dict[str, Any]] = []
    for file_path in files:
        row = _patch_session_runtime_file(file_path)
        details.append(row)
        status = str(row.get("status", ""))
        if status == "patched":
            report["patched_files"] = int(report["patched_files"]) + 1
        elif status == "already_patched":
            report["already_patched_files"] = int(report["already_patched_files"]) + 1
        elif status.endswith("_failed"):
            report["errors"] = int(report["errors"]) + 1

    report["details"] = details
    report["applied"] = bool(int(report["patched_files"]) > 0)
    return report


def _detect_log_markers(lines: list[str]) -> list[str]:
    markers: list[str] = []
    joined = "\n".join(lines).lower()
    if "status=405 method not allowed" in joined:
        markers.append("wa_handshake_405")
    if "hit 3 restarts/hour limit" in joined:
        markers.append("wa_restart_rate_limit")
    if "connection errored" in joined and "baileys" in joined:
        markers.append("baileys_connection_error")
    return markers


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default=os.getenv("OPENCLAW_COMMAND", "openclaw"), help="OpenClaw CLI command.")
    parser.add_argument("--account", default="default", help="WhatsApp account id (default: default).")
    parser.add_argument("--fresh-auth", action="store_true", help="Rotate to a new authDir before login.")
    parser.add_argument(
        "--probe-url",
        action="append",
        dest="probe_urls",
        help="URL to probe before relink. Can be repeated (default probes include local gateway + openclaw.ai).",
    )
    parser.add_argument("--skip-url-probe", action="store_true", help="Skip pre-login URL probes.")
    parser.add_argument(
        "--auto-wa-version-hotfix",
        dest="auto_wa_version_hotfix",
        action="store_true",
        default=True,
        help="Auto-apply runtime WhatsApp version hotfix if needed (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-wa-version-hotfix",
        dest="auto_wa_version_hotfix",
        action="store_false",
        help="Disable automatic runtime WhatsApp version hotfix.",
    )
    parser.add_argument(
        "--force-wa-version-hotfix",
        action="store_true",
        help="Force runtime WhatsApp version hotfix before login.",
    )
    parser.add_argument(
        "--auth-root",
        default="",
        help="Optional root directory for rotated authDir (defaults to current authDir parent).",
    )
    parser.add_argument("--no-verbose-login", action="store_true", help="Disable --verbose on channels login.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    args = parser.parse_args(argv)

    oc_base = _split_command(str(args.command))
    account_id = str(args.account or "default").strip() or "default"

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": oc_base,
        "account": account_id,
        "fresh_auth": bool(args.fresh_auth),
        "auto_wa_version_hotfix": bool(args.auto_wa_version_hotfix),
        "force_wa_version_hotfix": bool(args.force_wa_version_hotfix),
        "steps": [],
    }

    version_res = _run_capture(oc_base=oc_base, args=["--version"], timeout_seconds=10.0)
    report["openclaw_version"] = str(version_res.stdout or version_res.stderr or "").strip()

    gateway_res = _run_capture(oc_base=oc_base, args=["gateway", "status", "--json"], timeout_seconds=20.0)
    report["gateway_status_rc"] = int(gateway_res.returncode)
    report["gateway_status"] = _extract_json(gateway_res.stdout or "")

    probe_urls = [str(x).strip() for x in (args.probe_urls or []) if str(x).strip()]
    if not probe_urls:
        probe_urls = list(DEFAULT_PROBE_URLS)
    if not bool(args.skip_url_probe):
        report["url_probes"] = [_http_probe(url) for url in probe_urls]
    else:
        report["url_probes"] = []

    pre_log_path = _latest_openclaw_log()
    pre_markers: list[str] = []
    if pre_log_path is not None:
        pre_markers = _detect_log_markers(_tail_lines(pre_log_path, max_lines=500))
    report["pre_diagnostic_markers"] = pre_markers

    hotfix_requested = bool(args.force_wa_version_hotfix or args.auto_wa_version_hotfix)
    hotfix_report: dict[str, Any] | None = None
    if hotfix_requested:
        hotfix_report = _apply_wa_version_hotfix()
        report["hotfix"] = hotfix_report
        report["steps"].append(
            {
                "step": "wa_version_hotfix",
                "patched_files": int(hotfix_report.get("patched_files", 0)),
                "already_patched_files": int(hotfix_report.get("already_patched_files", 0)),
                "errors": int(hotfix_report.get("errors", 0)),
            }
        )
        if int(hotfix_report.get("patched_files", 0)) > 0 and not bool(args.fresh_auth):
            restart_after_hotfix = _run_capture(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
            report["steps"].append({"step": "gateway_restart_after_hotfix", "rc": int(restart_after_hotfix.returncode)})

    pre_status_res = _run_capture(oc_base=oc_base, args=["channels", "status", "--json"], timeout_seconds=20.0)
    pre_payload = _extract_json(pre_status_res.stdout or "")
    pre_snapshot = _status_snapshot(pre_payload, account_id=account_id)
    report["before"] = pre_snapshot

    rotated_auth_dir = ""
    if bool(args.fresh_auth):
        cfg = _load_openclaw_cfg()
        old_auth_dir = _current_auth_dir(cfg, account_id=account_id)
        old_auth_path = Path(old_auth_dir).expanduser()
        auth_root = Path(str(args.auth_root).strip()).expanduser() if str(args.auth_root).strip() else old_auth_path.parent
        rotated_auth_path = auth_root / f"{account_id}-{_utc_stamp()}"
        rotated_auth_path.mkdir(parents=True, exist_ok=True)
        rotated_auth_dir = str(rotated_auth_path)

        logout_res = _run_capture(
            oc_base=oc_base,
            args=["channels", "logout", "--channel", "whatsapp", "--account", account_id],
            timeout_seconds=20.0,
        )
        report["steps"].append({"step": "logout", "rc": int(logout_res.returncode)})

        set_res = _run_capture(
            oc_base=oc_base,
            args=[
                "config",
                "set",
                f"channels.whatsapp.accounts.{account_id}.authDir",
                json.dumps(rotated_auth_dir, ensure_ascii=True),
                "--json",
            ],
            timeout_seconds=25.0,
        )
        set_step: dict[str, Any] = {"step": "set_auth_dir", "rc": int(set_res.returncode), "authDir": rotated_auth_dir}
        if int(set_res.returncode) != 0:
            ok_fallback, fallback_error = _write_auth_dir_fallback(account_id=account_id, auth_dir=rotated_auth_dir)
            set_step["fallback_file_write"] = bool(ok_fallback)
            if not ok_fallback and fallback_error:
                set_step["fallback_error"] = str(fallback_error)
            if ok_fallback:
                set_step["rc"] = 0
        report["steps"].append(set_step)

        restart_res = _run_capture(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
        report["steps"].append({"step": "gateway_restart", "rc": int(restart_res.returncode)})

    login_args = ["channels", "login", "--channel", "whatsapp", "--account", account_id]
    if not bool(args.no_verbose_login):
        login_args.append("--verbose")
    login_rc = _run_interactive(oc_base=oc_base, args=login_args)
    report["steps"].append({"step": "interactive_login", "rc": int(login_rc)})

    post_status_res = _run_capture(oc_base=oc_base, args=["channels", "status", "--probe", "--json"], timeout_seconds=25.0)
    post_payload = _extract_json(post_status_res.stdout or "")
    post_snapshot = _status_snapshot(post_payload, account_id=account_id)
    report["after"] = post_snapshot
    report["post_status_rc"] = int(post_status_res.returncode)

    log_path = _latest_openclaw_log()
    if log_path is not None:
        markers = _detect_log_markers(_tail_lines(log_path, max_lines=500))
        report["log_file"] = str(log_path)
        report["diagnostic_markers"] = markers
    else:
        report["diagnostic_markers"] = []

    linked = bool(post_snapshot.get("linked"))
    connected = bool(post_snapshot.get("connected"))
    report["success"] = bool(linked and connected)
    if not report["success"]:
        hints: list[str] = []
        markers = report.get("diagnostic_markers", [])
        if "wa_handshake_405" in markers:
            hints.append(
                "Detected websocket 405 during WhatsApp handshake. Re-run with --fresh-auth --force-wa-version-hotfix."
            )
        if "wa_restart_rate_limit" in markers:
            hints.append("Health monitor restart rate limit was hit recently; pause a few minutes before retrying.")
        url_probes = report.get("url_probes", [])
        local_gateway_probe_failed = False
        openclaw_site_probe_failed = False
        for probe in url_probes if isinstance(url_probes, list) else []:
            if not isinstance(probe, dict):
                continue
            probe_url = str(probe.get("url", "")).lower()
            reachable = bool(probe.get("reachable"))
            if "127.0.0.1:18789/channels" in probe_url and not reachable:
                local_gateway_probe_failed = True
            if "openclaw.ai" in probe_url and not reachable:
                openclaw_site_probe_failed = True
        if local_gateway_probe_failed:
            hints.append("Local gateway URL probe failed. Run `openclaw gateway restart` and retry login.")
        if openclaw_site_probe_failed:
            hints.append("openclaw.ai URL probe failed. Check outbound TLS/network policies before retrying.")
        if isinstance(hotfix_report, dict) and int(hotfix_report.get("errors", 0)) > 0:
            hints.append("Runtime WA hotfix had write/read errors. Verify OpenClaw installation and permissions.")
        if not hints:
            hints.append("Rerun with --fresh-auth and verify outbound websocket access to web.whatsapp.com:443.")
        report["hints"] = hints

    if bool(args.json):
        print(_json_dumps(report))
    else:
        print("[openclaw_whatsapp_relink] OpenClaw version:", report.get("openclaw_version", ""))
        print("[openclaw_whatsapp_relink] Before:", _json_dumps(pre_snapshot))
        url_probes = report.get("url_probes", [])
        if isinstance(url_probes, list) and url_probes:
            for probe in url_probes:
                if not isinstance(probe, dict):
                    continue
                print(
                    "[openclaw_whatsapp_relink] URL probe:",
                    probe.get("url"),
                    "reachable=" + str(bool(probe.get("reachable"))),
                    "status=" + str(probe.get("status")),
                )
        if isinstance(hotfix_report, dict):
            print(
                "[openclaw_whatsapp_relink] WA hotfix:",
                f"patched={int(hotfix_report.get('patched_files', 0))}",
                f"already={int(hotfix_report.get('already_patched_files', 0))}",
                f"errors={int(hotfix_report.get('errors', 0))}",
            )
        if rotated_auth_dir:
            print("[openclaw_whatsapp_relink] Rotated authDir:", rotated_auth_dir)
        print("[openclaw_whatsapp_relink] After:", _json_dumps(post_snapshot))
        markers = report.get("diagnostic_markers", [])
        if markers:
            print("[openclaw_whatsapp_relink] Markers:", ", ".join(str(x) for x in markers))
        if report["success"]:
            print("[openclaw_whatsapp_relink] SUCCESS: WhatsApp account is linked and connected.")
        else:
            print("[openclaw_whatsapp_relink] FAILED: WhatsApp account is not fully linked/connected.")
            for hint in report.get("hints", []):
                print("[openclaw_whatsapp_relink] Hint:", hint)

    return 0 if bool(report["success"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
