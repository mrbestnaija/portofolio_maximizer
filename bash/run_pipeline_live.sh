#!/usr/bin/env bash
# Live/auto orchestration for Portfolio Maximizer ETL pipeline.
# Executes with network data first, falling back per execution mode settings.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/simpleTrader_env/bin/python"
PIPELINE_SCRIPT="$ROOT_DIR/scripts/run_etl_pipeline.py"
LOG_DIR="$ROOT_DIR/logs/live_runs"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "Pipeline runner missing at $PIPELINE_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/pipeline_live_${RUN_STAMP}.log"

TICKERS="${TICKERS:-AAPL,MSFT}"
START_DATE="${START_DATE:-2015-01-01}"
END_DATE="${END_DATE:-2024-01-01}"
DATA_SOURCE="${DATA_SOURCE:-}"
USE_CV="${USE_CV:-0}"
ENABLE_LLM="${ENABLE_LLM:-1}"
LLM_MODEL="${LLM_MODEL:-}"
EXECUTION_MODE="${EXECUTION_MODE:-auto}"

CMD=("$PYTHON_BIN" "$PIPELINE_SCRIPT"
  --tickers "$TICKERS"
  --start "$START_DATE"
  --end "$END_DATE"
  --execution-mode "$EXECUTION_MODE"
)

if [[ -n "$DATA_SOURCE" ]]; then
  CMD+=(--data-source "$DATA_SOURCE")
fi

if [[ "$USE_CV" == "1" ]]; then
  CMD+=(--use-cv)
fi

if [[ "$ENABLE_LLM" == "1" ]]; then
  CMD+=(--enable-llm)
fi

if [[ -n "$LLM_MODEL" ]]; then
  CMD+=(--llm-model "$LLM_MODEL")
fi

# Allow callers to append additional overrides.
CMD+=("$@")

echo "Running pipeline (${EXECUTION_MODE}) @ ${RUN_STAMP}"
echo "Command: ${CMD[*]}"
echo "Streaming output to $LOG_FILE"

set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?
set -e

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "Pipeline run failed (exit code $EXIT_CODE). See $LOG_FILE for details." >&2
  exit "$EXIT_CODE"
fi

PIPELINE_ID="$(grep -oE 'pipeline_[0-9]{8}_[0-9]{6}' "$LOG_FILE" | tail -n 1 || true)"

if [[ -z "$PIPELINE_ID" ]]; then
  echo "Unable to determine pipeline ID from $LOG_FILE" >&2
  exit 2
fi

echo ""
echo "Pipeline ID: $PIPELINE_ID"
echo "Stage timing summary:"

ROOT_DIR="$ROOT_DIR" PIPELINE_ID="$PIPELINE_ID" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
pipeline_id = os.environ["PIPELINE_ID"]
events_path = root / "logs" / "events" / "events.log"

stage_rows = []
with events_path.open() as fh:
    for line in fh:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("pipeline_id") != pipeline_id:
            continue
        if record.get("event_type") == "stage_complete":
            stage = record.get("stage")
            duration = record.get("metadata", {}).get("duration_seconds")
            if stage and duration is not None:
                stage_rows.append((stage, duration))

if not stage_rows:
    print("  (no stage metrics found)")
else:
    for stage, duration in stage_rows:
        print(f"  - {stage}: {duration:0.2f}s")

print("\nLatest dataset artifacts:")
data_root = root / "data"
for split in ("training", "validation", "testing"):
    split_dir = data_root / split
    if not split_dir.exists():
        continue
    candidates = sorted(split_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0] if candidates else None
    if latest:
        size_kb = latest.stat().st_size / 1024
        print(f"  - {split}: {latest.name} ({size_kb:0.1f} KiB)")
print("")
PY

echo "Log captured at: $LOG_FILE"
echo "Bestman's Portfolio Maximizer v45 Live pipeline run complete."
