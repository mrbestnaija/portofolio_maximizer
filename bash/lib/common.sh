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

pmx_detect_python() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  local candidate=""

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidate="${PYTHON_BIN}"
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  if [[ -n "${PMX_PYTHON_BIN:-}" ]]; then
    candidate="${PMX_PYTHON_BIN}"
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

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

  candidate="${root}/simpleTrader_env/Scripts/python.exe"
  if [[ -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
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

  candidate="${root}/simpleTrader_env/Scripts/python.exe"
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
    pmx_die "Python interpreter not found under ${root}/simpleTrader_env. Create it (python3 -m venv simpleTrader_env) and install requirements."
  fi
  echo "${py}"
}

pmx_resolve_python() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(pmx_repo_root)"
  fi

  local candidate=""

  if [[ -n "${PMX_PYTHON_BIN:-}" ]]; then
    candidate="${PMX_PYTHON_BIN}"
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidate="${PYTHON_BIN}"
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
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
