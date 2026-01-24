#!/usr/bin/env bash
#
# Common helpers for bash/* scripts.
# Intended to be sourced: `source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"`.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: bash/lib/common.sh must be sourced, not executed." >&2
  exit 1
fi

if [[ "${PMX_COMMON_SH_INCLUDED:-0}" == "1" ]]; then
  return 0
fi
PMX_COMMON_SH_INCLUDED=1

pmx_repo_root() {
  local root=""
  root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "${root}" ]]; then
    echo "${root}"
    return 0
  fi

  local lib_dir=""
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  echo "$(cd "${lib_dir}/.." && pwd)"
}

pmx_timestamp() {
  date -Iseconds 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"
}

pmx_log() {
  local level="${1:-INFO}"
  shift || true
  echo "[${level}] $(pmx_timestamp) $*"
}

pmx_die() {
  pmx_log "ERROR" "$*"
  exit 1
}

pmx_unset_diagnostics() {
  unset DIAGNOSTIC_MODE TS_DIAGNOSTIC_MODE EXECUTION_DIAGNOSTIC_MODE LLM_FORCE_FALLBACK 2>/dev/null || true
}

pmx_is_wsl() {
  # Prefer /proc/version markers; fall back to WSL_DISTRO_NAME when available.
  if [[ -f /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version; then
    return 0
  fi
  [[ -n "${WSL_DISTRO_NAME:-}" ]]
}

pmx_require_wsl() {
  if ! pmx_is_wsl; then
    pmx_die "Unsupported runtime: this repo is validated only under WSL. See Documentation/RUNTIME_GUARDRAILS.md"
  fi
}

pmx_activate_simpletrader_env() {
  pmx_require_wsl
  local root=""
  root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  local activate_path="${root}/simpleTrader_env/bin/activate"
  [[ -f "${activate_path}" ]] || pmx_die "Missing WSL virtualenv: ${activate_path} (see Documentation/RUNTIME_GUARDRAILS.md)"

  # shellcheck source=/dev/null
  source "${activate_path}"
}

pmx_print_runtime_fingerprint() {
  pmx_require_wsl
  local py_path=""
  py_path="$(command -v python 2>/dev/null || true)"
  pmx_log "INFO" "which python: ${py_path:-<missing>}"
  python -V 2>&1 | sed 's/^/[INFO] /' || true

  python - <<'PY' 2>&1 | sed 's/^/[INFO] /'
import json

try:
    import torch  # type: ignore
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")

payload = {
    "torch": getattr(torch, "__version__", None),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
}
print(json.dumps(payload, indent=2))
PY
}

pmx_require_wsl_simpletrader_runtime() {
  local root=""
  root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  pmx_activate_simpletrader_env "${root}"

  local py_path=""
  py_path="$(command -v python 2>/dev/null || true)"
  if [[ -z "${py_path}" ]] || [[ "${py_path}" != *"/simpleTrader_env/bin/python"* ]]; then
    pmx_die "Wrong interpreter selected (${py_path:-<missing>}); expected .../simpleTrader_env/bin/python. See Documentation/RUNTIME_GUARDRAILS.md"
  fi

  pmx_print_runtime_fingerprint
  PMX_RUNTIME_VERIFIED=1
}

pmx_detect_python() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  local candidate=""

  pmx_require_wsl

  candidate="${root}/simpleTrader_env/bin/python"
  if [[ -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  candidate="${root}/simpleTrader_env/bin/python3"
  if [[ -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  echo ""
}

pmx_detect_venv_python() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  local candidate=""
  pmx_require_wsl
  candidate="${root}/simpleTrader_env/bin/python"
  if [[ -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  candidate="${root}/simpleTrader_env/bin/python3"
  if [[ -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  echo ""
}

pmx_require_venv_python() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  local py=""
  py="$(pmx_detect_venv_python "${root}")"
  if [[ -z "${py}" ]]; then
    pmx_die "Python interpreter not found under ${root}/simpleTrader_env (WSL only). See Documentation/RUNTIME_GUARDRAILS.md"
  fi
  echo "${py}"
}

pmx_resolve_python() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  if [[ "${PMX_RUNTIME_VERIFIED:-0}" != "1" ]]; then
    pmx_require_wsl_simpletrader_runtime "${root}"
  fi

  pmx_require_venv_python "${root}"
}

pmx_require_file() {
  local path="${1:-}"
  [[ -n "${path}" ]] || pmx_die "pmx_require_file called without a path"
  [[ -f "${path}" ]] || pmx_die "Missing file: ${path}"
}

pmx_require_executable() {
  local path="${1:-}"
  [[ -n "${path}" ]] || pmx_die "pmx_require_executable called without a path"
  [[ -x "${path}" ]] || pmx_die "Missing executable: ${path}"
}

pmx_run_tee() {
  local log_file="${1:-}"
  shift || true
  [[ -n "${log_file}" ]] || pmx_die "pmx_run_tee requires a log file path"

  pmx_log "INFO" "Command: $*"
  pmx_log "INFO" "Log: ${log_file}"

  set +e
  "$@" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  set -e

  return "${exit_code}"
}

pmx_pid_is_running() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

pmx_pidfile_read() {
  local pidfile="${1:-}"
  [[ -n "${pidfile}" ]] || return 1
  [[ -f "${pidfile}" ]] || return 1
  local pid=""
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  if pmx_pid_is_running "${pid}"; then
    echo "${pid}"
    return 0
  fi
  rm -f "${pidfile}" >/dev/null 2>&1 || true
  return 1
}

pmx_pidfile_write() {
  local pidfile="${1:-}"
  local pid="${2:-}"
  [[ -n "${pidfile}" ]] || return 1
  [[ -n "${pid}" ]] || return 1
  mkdir -p "$(dirname "${pidfile}")" >/dev/null 2>&1 || true
  echo "${pid}" >"${pidfile}"
}

pmx_kill_pidfile() {
  local pidfile="${1:-}"
  local pid=""
  pid="$(pmx_pidfile_read "${pidfile}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    rm -f "${pidfile}" >/dev/null 2>&1 || true
  fi
}

pmx_dashboard_url() {
  local port="${1:-8000}"
  echo "http://127.0.0.1:${port}/visualizations/live_dashboard.html"
}

pmx_ensure_dashboard() {
  local root="${1:-}"
  local python_bin="${2:-}"
  local port="${3:-8000}"
  local persist="${4:-1}"
  local keep_alive="${5:-1}"
  local db_path="${6:-}"

  [[ -n "${root}" ]] || root="$(pmx_repo_root)"
  [[ -n "${python_bin}" ]] || python_bin="$(pmx_require_venv_python "${root}")"

  local dash_html="${root}/visualizations/live_dashboard.html"
  local dash_json="${root}/visualizations/dashboard_data.json"
  local bridge="${root}/scripts/dashboard_db_bridge.py"

  local server_pidfile="${root}/logs/dashboard_http_${port}.pid"
  local bridge_pidfile="${root}/logs/dashboard_bridge.pid"

  # Start/keep DB->JSON bridge (writes dashboard_data.json and optionally dashboard_audit.db).
  local bridge_pid=""
  bridge_pid="$(pmx_pidfile_read "${bridge_pidfile}" 2>/dev/null || true)"
  if [[ -z "${bridge_pid}" ]]; then
    if [[ "${persist}" == "1" ]]; then
      "${python_bin}" "${bridge}" --interval-seconds 5 --persist-snapshot ${db_path:+--db-path "${db_path}"} >/dev/null 2>&1 &
    else
      "${python_bin}" "${bridge}" --interval-seconds 5 ${db_path:+--db-path "${db_path}"} >/dev/null 2>&1 &
    fi
    bridge_pid=$!
    pmx_pidfile_write "${bridge_pidfile}" "${bridge_pid}" || true
  fi

  # Start/keep local http.server for static dashboard.
  local serve_pid=""
  serve_pid="$(pmx_pidfile_read "${server_pidfile}" 2>/dev/null || true)"
  if [[ -z "${serve_pid}" ]]; then
    "${python_bin}" -m http.server "${port}" --bind 127.0.0.1 --directory "${root}" >/dev/null 2>&1 &
    serve_pid=$!
    if pmx_pid_is_running "${serve_pid}"; then
      pmx_pidfile_write "${server_pidfile}" "${serve_pid}" || true
    else
      serve_pid=""
    fi
  fi

  echo "Dashboard URL: $(pmx_dashboard_url "${port}")"
  echo "Dashboard HTML: ${dash_html}"
  echo "Dashboard JSON: ${dash_json}"

  if [[ "${keep_alive}" != "1" ]]; then
    # Best-effort cleanup instructions (caller may also trap).
    PMX_DASHBOARD_SERVER_PIDFILE="${server_pidfile}"
    PMX_DASHBOARD_BRIDGE_PIDFILE="${bridge_pidfile}"
    export PMX_DASHBOARD_SERVER_PIDFILE PMX_DASHBOARD_BRIDGE_PIDFILE
  fi
}

pmx_dashboard_cleanup() {
  pmx_kill_pidfile "${PMX_DASHBOARD_SERVER_PIDFILE:-}" || true
  pmx_kill_pidfile "${PMX_DASHBOARD_BRIDGE_PIDFILE:-}" || true
}

pmx_sanitize_logs() {
  local root="${1:-}"
  [[ -n "${root}" ]] || root="$(pmx_repo_root)"
  : "${PMX_SANITIZE_LOGS:=1}"
  : "${PMX_LOG_RETENTION:=5}"
  if [[ "${PMX_SANITIZE_LOGS}" != "1" ]]; then
    return 0
  fi
  export PMX_ROOT="${root}"
  command -v python3 >/dev/null 2>&1 || {
    pmx_log "WARN" "python3 not available; skipping log sanitization."
    return 0
  }
  python3 - <<'PY'
import os
import re
import shutil
import gzip
from datetime import datetime
from pathlib import Path

root = Path(os.environ.get("PMX_ROOT", "")).expanduser()
if not root or not root.exists():
    root = Path(".").resolve()
logs_dir = root / "logs"
if not logs_dir.exists():
    raise SystemExit(0)

retention = int(os.environ.get("PMX_LOG_RETENTION", "5"))
archive_dir = logs_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
archive_dir.mkdir(parents=True, exist_ok=True)

text_exts = {".log", ".txt", ".jsonl", ".json"}
patterns = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\\s*[:=]\\s*([^\\s,;]+)")
]

def redact_text(path: Path) -> None:
    try:
        data = path.read_text(errors="ignore")
    except Exception:
        return
    original = data
    for pat in patterns:
        data = pat.sub(lambda m: f"{m.group(1)}=REDACTED", data)
    if data != original:
        path.write_text(data)

for p in logs_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in text_exts:
        redact_text(p)

for dirpath, _, filenames in os.walk(logs_dir):
    dir_path = Path(dirpath)
    if dir_path == archive_dir or archive_dir in dir_path.parents:
        continue
    files = [dir_path / f for f in filenames if (dir_path / f).is_file()]
    files = [f for f in files if archive_dir not in f.parents]
    if not files:
        continue
    files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files_sorted[retention:]:
        rel = f.relative_to(logs_dir)
        dest = archive_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

for p in archive_dir.rglob("*"):
    if p.is_file() and p.suffix != ".gz":
        gz_path = p.with_suffix(p.suffix + ".gz")
        with open(p, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        p.unlink()
PY
}
