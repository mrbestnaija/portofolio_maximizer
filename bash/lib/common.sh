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
