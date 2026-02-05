#!/usr/bin/env bash
# Deploy Monitoring Systems (WSL + simpleTrader_env only).
#
# Safe-by-default:
# - Creates required directories under the repo
# - Installs Python deps into `simpleTrader_env`
# - Runs quick health checks (no systemd/cron writes)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/bash/lib/common.sh"
pmx_require_wsl_simpletrader_runtime "${ROOT_DIR}"
PYTHON_BIN="$(pmx_require_venv_python "${ROOT_DIR}")"

echo "[INFO] Creating monitoring directories..."
mkdir -p \
  logs/alerts \
  logs/archive/errors \
  logs/archive/cache \
  config/monitoring

echo "[INFO] Ensuring monitoring scripts are executable (best-effort)..."
for target in \
  scripts/error_monitor.py \
  scripts/cache_manager.py \
  scripts/sanitize_cache_and_logs.py
do
  if [[ -f "${target}" ]]; then
    chmod +x "${target}" 2>/dev/null || true
  fi
done

echo "[INFO] Installing Python dependencies into simpleTrader_env..."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

if [[ -f scripts/cache_manager.py ]]; then
  echo "[INFO] Running cache_manager.py (one-shot)..."
  "${PYTHON_BIN}" scripts/cache_manager.py || true
fi

if [[ -f scripts/error_monitor.py ]]; then
  echo "[INFO] Running error_monitor.py --check-only..."
  "${PYTHON_BIN}" scripts/error_monitor.py --check-only || true
fi

if [[ -f tests/etl/test_method_signature_validation.py ]]; then
  echo "[INFO] Running method signature validation tests..."
  "${PYTHON_BIN}" -m pytest tests/etl/test_method_signature_validation.py -q
fi

echo "[INFO] Monitoring deployment complete."
echo "[INFO] Next: configure scheduled tasks per Documentation/SYSTEM_ERROR_MONITORING_GUIDE.md"
