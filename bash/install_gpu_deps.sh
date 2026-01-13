#!/usr/bin/env bash
#
# Install / verify GPU-capable dependencies for Portfolio Maximizer inside the ONLY
# supported runtime: WSL + `simpleTrader_env` (Linux venv).
#
# This script never creates a new environment; it installs into `simpleTrader_env/`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Runtime guardrails (prints python/torch/cuda fingerprint).
source "${ROOT_DIR}/bash/lib/common.sh"
pmx_require_wsl_simpletrader_runtime "${ROOT_DIR}"
PYTHON_BIN="$(pmx_require_venv_python "${ROOT_DIR}")"

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
# Optional: set this if you need a specific CUDA wheel index (example: https://download.pytorch.org/whl/cu128)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

echo "=== GPU dependency install (simpleTrader_env) ==="
echo "Python: ${PYTHON_BIN}"
echo "Torch : ${TORCH_VERSION}"
echo "Index : ${TORCH_INDEX_URL:-<default>}"

"${PYTHON_BIN}" -m pip install --upgrade pip

if [[ -n "${TORCH_INDEX_URL}" ]]; then
  "${PYTHON_BIN}" -m pip install --extra-index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
  "${PYTHON_BIN}" -m pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt
else
  "${PYTHON_BIN}" -m pip install "torch==${TORCH_VERSION}"
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

"${PYTHON_BIN}" - <<'PY'
import json
import sys

import torch

payload = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
}
print(json.dumps(payload, indent=2))

if not payload["cuda_available"]:
    raise SystemExit("[ERROR] torch.cuda.is_available() == False; GPU runtime not configured.")
PY

echo "=== Done ==="
