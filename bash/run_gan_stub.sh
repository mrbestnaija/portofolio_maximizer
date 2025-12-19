#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

DATASET_PATH="${1:-data/synthetic/latest.json}"

echo "[gan_stub] using dataset pointer: ${DATASET_PATH}"
PIPELINE_DEVICE="${PIPELINE_DEVICE:-}" \
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
"${PYTHON_BIN}" scripts/train_gan_stub.py \
  --dataset-path "${DATASET_PATH}" \
  --prefer-gpu \
  --train-epochs "${TRAIN_EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --seq-len "${SEQ_LEN:-32}" \
  --latent-dim "${LATENT_DIM:-16}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --output-dir "${OUTPUT_DIR:-models/synthetic/gan_stub}" \
  ${TICKERS:+--tickers "${TICKERS}"} \
  --verbose

echo "[gan_stub] completed"
