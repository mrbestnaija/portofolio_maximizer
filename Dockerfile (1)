# Portfolio Maximizer v4.5 - Multi-Stage Docker Build
# Optimized for reproducibility, security, and minimal image size

# =============================================================================
# Stage 1: Base Python Environment with System Dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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
# Stage 2: Copy Existing Virtual Environment
# =============================================================================
FROM base AS builder

# Copy existing virtual environment from host
COPY simpleTrader_env /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# =============================================================================
# Stage 3: Final Production Image
# =============================================================================
FROM python:3.12-slim-bookworm AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORTFOLIO_HOME=/app \
    PORTFOLIO_DATA=/app/data \
    PORTFOLIO_LOGS=/app/logs \
    PORTFOLIO_CONFIG=/app/config

# Install minimal runtime dependencies
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

# Set working directory
WORKDIR /app

# Copy project files
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

# Switch to non-root user
USER portfolio

# Validate installation
RUN python -c "import numpy, pandas, scipy, sklearn, matplotlib, yfinance; print('Dependencies validated')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command: show help
CMD ["python", "scripts/run_etl_pipeline.py", "--help"]

# =============================================================================
# Metadata
# =============================================================================
LABEL maintainer="Bestman Ezekwu Enock <mrbestnaija@example.com>" \
      version="4.5" \
      description="Portfolio Management System with Quantitative Trading Strategies" \
      org.opencontainers.image.source="https://github.com/mrbestnaija/portofolio_maximizer" \
      org.opencontainers.image.documentation="https://github.com/mrbestnaija/portofolio_maximizer/blob/master/README.md" \
      org.opencontainers.image.authors="Bestman Ezekwu Enock" \
      org.opencontainers.image.url="https://github.com/mrbestnaija/portofolio_maximizer" \
      org.opencontainers.image.vendor="Portfolio Maximizer Team" \
      org.opencontainers.image.licenses="MIT"

# Exposed volumes for data persistence
VOLUME ["/app/data", "/app/logs", "/app/config"]

# =============================================================================
# Usage Examples:
# =============================================================================
# Build: docker build -t portfolio-maximizer:v4.5 .
# Run ETL: docker run --rm -v $(pwd)/data:/app/data portfolio-maximizer:v4.5 python scripts/run_etl_pipeline.py
# Interactive: docker run -it --rm portfolio-maximizer:v4.5 /bin/bash
# With CV: docker run --rm -v $(pwd)/data:/app/data portfolio-maximizer:v4.5 python scripts/run_etl_pipeline.py --use-cv
# =============================================================================
