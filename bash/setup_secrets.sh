#!/usr/bin/env bash
# Setup Docker secrets directory (WSL + repo-local filesystem).
#
# Creates `secrets/` and placeholder secret files (never commits real keys).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/bash/lib/common.sh"
pmx_require_wsl

SECRETS_DIR="${SECRETS_DIR:-secrets}"

echo "[INFO] Setting up secrets directory: ${SECRETS_DIR}"

mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}" 2>/dev/null || true

create_placeholder() {
  local path="$1"
  local header="$2"
  if [[ -f "${path}" ]]; then
    echo "[INFO] Exists: ${path}"
    return 0
  fi

  printf '%s\n' "${header}" > "${path}"
  chmod 600 "${path}" 2>/dev/null || true
  echo "[INFO] Created placeholder: ${path}"
}

create_placeholder "${SECRETS_DIR}/alpha_vantage_api_key.txt" "# Placeholder - replace with your Alpha Vantage API key (do not commit)"
create_placeholder "${SECRETS_DIR}/finnhub_api_key.txt" "# Placeholder - replace with your Finnhub API key (do not commit)"

echo "[INFO] Done."
echo "[INFO] Next:"
echo "  1) Edit: ${SECRETS_DIR}/alpha_vantage_api_key.txt"
echo "  2) Edit: ${SECRETS_DIR}/finnhub_api_key.txt"
echo "  3) Confirm not tracked: git status --ignored | grep -i secrets"
