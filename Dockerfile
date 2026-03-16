# Portfolio Maximizer v4.5 - Multi-Stage Docker Build (reproducible)
# Installs dependencies inside the image (no host venv) for portability.

# =============================================================================
# Stage 1: Base Python Environment with System Dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libatlas-base-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Builder - create venv and install Python deps from requirements
# =============================================================================
FROM base AS builder

WORKDIR /app
COPY requirements.txt .
COPY requirements-ml.txt .

ARG INSTALL_ML_EXTRAS=0

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_ML_EXTRAS" = "1" ]; then /opt/venv/bin/pip install --no-cache-dir -r requirements-ml.txt; fi

# =============================================================================
# Stage 3: Final Production Image
# =============================================================================
FROM python:3.12-slim-bookworm AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORTFOLIO_HOME=/app \
    PORTFOLIO_DATA=/app/data \
    PORTFOLIO_LOGS=/app/logs \
    PORTFOLIO_CONFIG=/app/config

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libblas3 \
    liblapack3 \
    libhdf5-103 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash portfolio && \
    mkdir -p /app /app/data /app/logs /app/config /app/cache && \
    chown -R portfolio:portfolio /app

# Copy virtual environment from builder
COPY --from=builder --chown=portfolio:portfolio /opt/venv /opt/venv

WORKDIR /app
COPY --chown=portfolio:portfolio . .

# Create required directories
RUN mkdir -p \
    data/raw \
    data/processed \
    data/training \
    data/validation \
    data/testing \
    data/checkpoints \
    logs/stages \
    cache && \
    chown -R portfolio:portfolio /app

USER portfolio

# Validate key imports
RUN python - <<'PY'\nimport importlib\nfor mod in [\"numpy\",\"pandas\",\"scipy\",\"sklearn\",\"matplotlib\",\"yfinance\"]:\n    importlib.import_module(mod)\nprint(\"Dependencies validated\")\nPY

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python - <<'PY'\nimport importlib, sys\ntry:\n    importlib.import_module('etl.yfinance_extractor')\n    sys.exit(0)\nexcept Exception:\n    sys.exit(1)\nPY

CMD ["python", "scripts/run_etl_pipeline.py", "--help"]

LABEL maintainer="Bestman Ezekwu Enock <mrbestnaija@example.com>" \
      version="4.5" \
      description="Portfolio Management System with Quantitative Trading Strategies" \
      org.opencontainers.image.source="https://github.com/mrbestnaija/portofolio_maximizer" \
      org.opencontainers.image.documentation="https://github.com/mrbestnaija/portofolio_maximizer/blob/master/README.md" \
      org.opencontainers.image.authors="Bestman Ezekwu Enock" \
      org.opencontainers.image.url="https://github.com/mrbestnaija/portofolio_maximizer" \
      org.opencontainers.image.vendor="Portfolio Maximizer Team" \
      org.opencontainers.image.licenses="MIT"

VOLUME ["/app/data", "/app/logs", "/app/config"]
