# Docker Setup with simpleTrader_env

## Overview
Portfolio Maximizer v4.5 Docker configuration uses the existing `simpleTrader_env` virtual environment for faster builds and consistent dependencies.

## Architecture

### Production Image
- Copies `simpleTrader_env` into `/opt/venv` at build time
- Standalone, no host dependencies at runtime
- Image size: ~1-2GB (includes full venv)

### Development Image  
- Mounts `simpleTrader_env` from host as read-only volume
- Live updates without rebuilding
- Ideal for rapid development

## Quick Start

### Build Image
```bash
docker build -t portfolio-maximizer:v4.5 .
```

### Run Production Container
```bash
docker compose up portfolio-maximizer
```

### Run Development Container
```bash
docker compose run --rm portfolio-dev /bin/bash
```

## Key Changes from Standard Setup

1. **No pip install during build** - Uses existing venv
2. **Faster builds** - Only copies files, no package downloads
3. **Dev/Prod parity** - Same packages in both environments

## Files Modified

- `Dockerfile` - Builder stage simplified to copy venv
- `.dockerignore` - Allows simpleTrader_env in build context
- `docker-compose.yml` - Dev service mounts host venv

## Requirements

- Docker Desktop running
- `simpleTrader_env` with all packages from requirements.txt
- Python 3.12 venv structure

## Testing

Run prerequisite check:
```bash
scripts/docker_build_test.bat prereq
```

Full build and test:
```bash
scripts/docker_build_test.bat all
```

## Troubleshooting

**Build fails**: Ensure Docker daemon is running
**Missing packages**: Verify simpleTrader_env/lib/python3.12/site-packages exists
**Permission issues**: Check venv directory permissions

## Live Path Readiness

**Current verdict:** Docker is suitable for staging, CI, and reproducible test runs, but it is **not yet suitable as the canonical unattended live path** for this repo.

The reason is simple: the repo still has a WSL-first validated runtime posture, while the current Docker entrypoint is not launching the real live funnel by default.

### Go / No-Go Checklist

Treat Docker as a live path only when all of the following are true:

- The production container starts the real live loop by default, not a help command or ETL-only dry run.
- `data/`, `logs/`, `cache/`, and audit/evidence artifacts survive container restart through durable volumes.
- Secrets arrive through Docker secrets or `_FILE` env vars, never through baked image layers or tracked config.
- There is an explicit CPU profile and an explicit GPU profile; GPU usage is opt-in and reproducible.
- Runtime guardrails recognize Docker as a supported production lane instead of rejecting everything except WSL.
- A one-cycle smoke test proves the container writes live audit output, PnL evidence, and calibration history.
- The live path can restart cleanly and resume state without losing portfolio continuity.
- Documentation no longer claims Docker is production-complete while runtime guardrails still say WSL-only.

### Current Blockers

- `Dockerfile` still defaults to `python scripts/run_etl_pipeline.py --help`.
- `bash/runtime_check.sh` still rejects non-WSL runtime outright.
- `README.md` still documents daily/intraday operation around WSL/Linux or Windows Task Scheduler, not Docker as the canonical live route.
- No explicit GPU live profile is wired as the default operational path.
- End-to-end live-state persistence inside the container has not yet been proven in this repo snapshot.

---
Last Updated: 2026-04-17
Phase: 4.5 - Docker containerization present, but live-path readiness still gated

