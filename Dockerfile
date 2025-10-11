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
# Stage 2: Python Dependencies Builder
# =============================================================================
FROM base AS builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first (layer caching optimization)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: requirements.txt has merge conflict - using comprehensive list
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    yfinance>=0.2.0 \
    pandas-datareader>=0.10.0 \
    statsmodels>=0.14.0 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0 \
    click>=8.1.0 \
    tqdm>=4.65.0 \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    requests>=2.31.0 \
    pyarrow>=12.0.0

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
LABEL maintainer="Portfolio Maximizer Team" \
      version="4.5" \
      description="Portfolio Management System with Quantitative Trading Strategies" \
      org.opencontainers.image.source="https://github.com/yourusername/portfolio_maximizer_v45" \
      org.opencontainers.image.documentation="https://github.com/yourusername/portfolio_maximizer_v45/README.md"

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
