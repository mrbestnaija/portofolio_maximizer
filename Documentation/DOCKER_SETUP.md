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

---
Last Updated: 2025-10-12
Phase: 4.5 - Docker Containerization Complete

