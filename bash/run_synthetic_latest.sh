#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python3 scripts/generate_synthetic_dataset.py --config config/synthetic_data_config.yml
PYTHONPATH=. SYNTHETIC_DATASET_ID=latest python3 scripts/run_etl_pipeline.py --execution-mode synthetic --data-source synthetic "$@"
