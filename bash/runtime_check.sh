#!/usr/bin/env bash
#
# Runtime guard for Portfolio Maximizer:
# - WSL only
# - simpleTrader_env Linux venv only (simpleTrader_env/bin/python)
# - Prints Python/Torch/CUDA fingerprint for reproducibility

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f /proc/version ]] || ! grep -qiE 'microsoft|wsl' /proc/version; then
  echo "[ERROR] Unsupported runtime: this repo is validated only under WSL." >&2
  echo "        See Documentation/RUNTIME_GUARDRAILS.md" >&2
  exit 2
fi

if [[ ! -f "${ROOT_DIR}/simpleTrader_env/bin/activate" ]]; then
  echo "[ERROR] Missing WSL virtualenv: ${ROOT_DIR}/simpleTrader_env/bin/activate" >&2
  echo "        See Documentation/RUNTIME_GUARDRAILS.md" >&2
  exit 3
fi

# shellcheck source=/dev/null
source "${ROOT_DIR}/simpleTrader_env/bin/activate"

PY_PATH="$(command -v python || true)"
if [[ -z "${PY_PATH}" ]]; then
  echo "[ERROR] python not found after activating simpleTrader_env." >&2
  exit 4
fi

case "${PY_PATH}" in
  *"/simpleTrader_env/bin/python"*) ;;
  *)
    echo "[ERROR] Wrong interpreter selected: ${PY_PATH}" >&2
    echo "        Expected .../simpleTrader_env/bin/python" >&2
    exit 5
    ;;
esac

echo "[INFO] which python: ${PY_PATH}"
python -V

python - <<'PY'
import json

payload = {
    "torch": None,
    "cuda_available": False,
    "device_count": 0,
    "device": None,
}

try:
    import torch  # type: ignore
except Exception as exc:
    raise SystemExit(f"[ERROR] torch import failed: {exc}")

payload["torch"] = getattr(torch, "__version__", None)
payload["cuda_available"] = bool(torch.cuda.is_available())
payload["device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
payload["device"] = torch.cuda.get_device_name(0) if payload["device_count"] else None

print(json.dumps(payload, indent=2))
PY

echo "[INFO] Runtime fingerprint OK."
