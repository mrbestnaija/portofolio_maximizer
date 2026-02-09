#!/bin/bash
# Portfolio Maximizer v45 - Environment Setup Script (Linux/WSL/Git Bash)
# Run this AFTER Python 3.10-3.12 is installed
# See RECOVERY_PLAN.md for pre-requisites

set -e  # Exit on error

echo "========================================"
echo "Portfolio Maximizer v45 - Setup"
echo "========================================"
echo

# Check Python
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.10-3.12 first."
    echo "See RECOVERY_PLAN.md section 1 for installation instructions."
    exit 1
fi

echo "[OK] Python found"
python --version
echo

# Verify we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found."
    echo "Please run this script from: portfolio_maximizer_v45/portfolio_maximizer_v45/"
    exit 1
fi

# Check if venv exists
if [ -d "venv" ]; then
    echo "[WARNING] venv/ already exists."
    read -p "Delete and create fresh? (y/N): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "Removing old venv..."
        rm -rf venv
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[1/5] Using existing virtual environment"
fi
echo

# Activate venv and upgrade pip
echo "[2/5] Upgrading pip..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
python -m pip install --upgrade pip --quiet
echo "[OK] pip upgraded"
echo

# Install core dependencies
echo "[3/5] Installing core dependencies from requirements.txt..."
echo "This may take several minutes depending on internet speed..."
pip install -r requirements.txt --no-cache-dir
echo "[OK] Core dependencies installed"
echo

# Verify .env file
echo "[4/5] Checking .env file..."
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found"
    if [ -f ".env.template" ]; then
        echo "Creating .env from template..."
        cp .env.template .env
        echo "[ACTION REQUIRED] Edit .env and add your API keys"
    else
        echo "[ERROR] No .env or .env.template found"
    fi
else
    echo "[OK] .env file exists"
fi
echo

# Test imports
echo "[5/5] Testing core imports..."
if python -c "import pandas; import numpy; import torch; print('[OK] Core libraries imported successfully')" 2>/dev/null; then
    echo "[OK] Core imports successful"
else
    echo "[WARNING] Some core imports failed - verify installation"
fi
echo

# Summary
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Virtual environment: venv/"
echo "Activate with: source venv/bin/activate  (Linux/WSL/Git Bash)"
echo "            or: venv\\Scripts\\activate.bat  (CMD)"
echo "            or: venv\\Scripts\\Activate.ps1  (PowerShell)"
echo
echo "Next steps:"
echo "1. If WSL not installed, see RECOVERY_PLAN.md section 2"
echo "2. Verify .env file has valid API keys"
echo "3. Run tests: pytest tests/ -v"
echo "4. Check RECOVERY_PLAN.md for model artifact recovery"
echo
echo "Environment ready for development!"
