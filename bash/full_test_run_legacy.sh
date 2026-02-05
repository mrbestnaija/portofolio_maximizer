#!/usr/bin/env bash
#
# Legacy wrapper for historical references to ".bash/full_test_run.sh".
# Canonical location is now: bash/full_test_run.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/bash/full_test_run.sh" "$@"
