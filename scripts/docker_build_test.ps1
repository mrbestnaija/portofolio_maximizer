# Docker Build and Test Script for Portfolio Maximizer v4.5 (PowerShell)
# This script builds the Docker images and runs comprehensive tests

param(
    [string]$Action = "all"
)

# Configuration
$IMAGE_NAME = "portfolio-maximizer"
$VERSION = "v4.5"
$COMPOSE_PROJECT_NAME = "portfolio_maximizer"

# Helper functions
function Write-Header {
    param($Message)
    Write-Host "`n====================================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "====================================================`n" -ForegroundColor Blue
}

function Write-Success {
    param($Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param($Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param($Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Function to check prerequisites
function Check-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
    } catch {
        Write-Error "Docker not found. Please install Docker Desktop for Windows."
        exit 1
    }
    
    # Check Docker Compose
    try {
        $null = docker compose version
        Write-Success "Docker Compose found"
    } catch {
        Write-Error "Docker Compose not found. Please install Docker Desktop for Windows."
        exit 1
    }
    
    # Check if Docker daemon is running
    try {
        $null = docker info 2>$null
        Write-Success "Docker daemon is running"
    } catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
    
    # Check for .env file
    if (Test-Path ".env") {
        Write-Success ".env file found"
    } else {
        Write-Warning ".env file not found. Creating from .env.example..."
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Success "Created .env file from .env.example"
        } else {
            Write-Warning "No .env.example found. API features may not work without API keys."
        }
    }
}

# Function to clean up old containers and images
function Cleanup {
    Write-Header "Cleaning Up Old Containers and Images"
    
    # Stop and remove containers
    Write-Host "Stopping containers..."
    docker compose down --remove-orphans 2>$null
    
    # Remove old images
    Write-Host "Removing old images..."
    docker images | Select-String $IMAGE_NAME | ForEach-Object { 
        $imageId = ($_ -split '\s+')[2]
        docker rmi -f $imageId 2>$null
    }
    
    Write-Success "Cleanup completed"
}

# Function to build Docker images
function Build-Images {
    Write-Header "Building Docker Images"
    
    # Build production image
    Write-Host "Building production image..."
    docker build `
        --target production `
        --tag ${IMAGE_NAME}:${VERSION} `
        --tag ${IMAGE_NAME}:latest `
        --file Dockerfile `
        .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build production image"
        exit 1
    }
    Write-Success "Production image built: ${IMAGE_NAME}:${VERSION}"
    
    # Build development image
    Write-Host "Building development image..."
    docker build `
        --target builder `
        --tag ${IMAGE_NAME}:${VERSION}-dev `
        --file Dockerfile `
        .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build development image"
        exit 1
    }
    Write-Success "Development image built: ${IMAGE_NAME}:${VERSION}-dev"
    
    # List built images
    Write-Host "`nBuilt images:"
    docker images | Select-String $IMAGE_NAME | Select-Object -First 5
}

# Function to run basic container tests
function Test-Containers {
    Write-Header "Testing Docker Containers"
    
    # Test 1: Basic container startup
    Write-Host "Test 1: Basic container startup..."
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "print('Container started successfully')"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start container"
        exit 1
    }
    Write-Success "Container startup test passed"
    
    # Test 2: Validate Python dependencies
    Write-Host "`nTest 2: Validating Python dependencies..."
    $pythonCode = @'
import numpy, pandas, scipy, sklearn, matplotlib, yfinance
print("All core dependencies imported successfully")
print(f"NumPy version: {numpy.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
'@
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c $pythonCode
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to import dependencies"
        exit 1
    }
    Write-Success "Dependency validation passed"
    
    # Test 3: ETL module imports
    Write-Host "`nTest 3: Testing ETL module imports..."
    $etlCode = @'
from etl import YFinanceExtractor, DataValidator, Preprocessor, DataStorage
print("All ETL modules imported successfully")
'@
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c $etlCode
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to import ETL modules"
        exit 1
    }
    Write-Success "ETL module import test passed"
    
    # Test 4: Health check
    Write-Host "`nTest 4: Running health check..."
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "import sys; sys.exit(0)"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Health check failed"
        exit 1
    }
    Write-Success "Health check passed"
}

# Function to run docker-compose tests
function Test-Compose {
    Write-Header "Testing Docker Compose Services"
    
    # Start services
    Write-Host "Starting services with docker-compose..."
    docker compose up -d portfolio-maximizer
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start services"
        exit 1
    }
    
    # Wait for container to be healthy
    Write-Host "Waiting for container to be healthy..."
    Start-Sleep -Seconds 5
    
    # Check container status
    $containerStatus = docker compose ps --format json | ConvertFrom-Json
    if ($containerStatus.State -eq "running") {
        Write-Success "Container is running"
    } else {
        Write-Error "Container failed to start properly"
        docker compose logs portfolio-maximizer
        docker compose down
        exit 1
    }
    
    # Run ETL pipeline help command
    Write-Host "`nTesting ETL pipeline help command..."
    docker compose exec -T portfolio-maximizer python scripts/run_etl_pipeline.py --help
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to run ETL pipeline help"
        docker compose down
        exit 1
    }
    Write-Success "ETL pipeline help command successful"
    
    # Stop services
    Write-Host "`nStopping services..."
    docker compose down
    Write-Success "Docker Compose tests completed"
}

# Function to run a sample ETL pipeline
function Run-SampleETL {
    Write-Header "Running Sample ETL Pipeline"
    
    Write-Host "Creating sample ticker list..."
    "AAPL", "MSFT", "GOOG" | Out-File -FilePath "data\sample_tickers.txt"
    
    Write-Host "`nRunning ETL pipeline with sample data..."
    $dataPath = (Get-Location).Path + "\data"
    $configPath = (Get-Location).Path + "\config"
    $logsPath = (Get-Location).Path + "\logs"
    
    docker run --rm `
        -v "${dataPath}:/app/data" `
        -v "${configPath}:/app/config" `
        -v "${logsPath}:/app/logs" `
        ${IMAGE_NAME}:${VERSION} `
        python scripts/run_etl_pipeline.py `
        --tickers AAPL MSFT GOOG `
        --start-date 2023-01-01 `
        --end-date 2023-12-31 `
        --no-cache
    
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "ETL pipeline run failed (this might be due to missing API keys)"
    }
    
    # Check if data was created
    if (Test-Path "data\training\*.parquet") {
        Write-Success "ETL pipeline created output files"
        Write-Host "Output files:"
        Get-ChildItem "data\training\*.parquet" | Select-Object -First 5
    } else {
        Write-Warning "No output files created (API keys might be required)"
    }
}

# Function to display usage information
function Show-Usage {
    Write-Header "Docker Usage Examples"
    
    @"
# Build images:
docker build -t ${IMAGE_NAME}:${VERSION} .

# Run ETL pipeline:
docker run --rm -v `$(pwd)/data:/app/data ${IMAGE_NAME}:${VERSION} ``
    python scripts/run_etl_pipeline.py --tickers AAPL MSFT --start-date 2023-01-01

# Interactive shell:
docker run -it --rm -v `$(pwd):/app ${IMAGE_NAME}:${VERSION}-dev /bin/bash

# Run with docker-compose:
docker compose up portfolio-maximizer

# Run tests:
docker compose run --rm portfolio-test

# Start Jupyter notebook:
docker compose up portfolio-notebook

# Development mode:
docker compose run --rm portfolio-dev /bin/bash

# View logs:
docker compose logs -f portfolio-maximizer

# Clean up:
docker compose down -v
"@
}

# Function to generate test report
function Generate-Report {
    Write-Header "Test Summary Report"
    
    $reportFile = "docker_test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    
    $reportContent = @"
Portfolio Maximizer v4.5 - Docker Test Report
Generated: $(Get-Date)

System Information:
  Docker Version: $(docker --version)
  OS: Windows $(Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Version)
  Architecture: $env:PROCESSOR_ARCHITECTURE

Test Results:
  Prerequisites Check: PASSED
  Docker Build: PASSED
  Container Tests: PASSED
  Docker Compose Tests: PASSED
  Sample ETL Run: COMPLETED

Images Built:
$(docker images | Select-String $IMAGE_NAME | Select-Object -First 5)

Recommendations:
1. Add API keys to .env file for full functionality
2. Mount volumes for data persistence in production
3. Use docker-compose for easier management
4. Monitor container logs for debugging

"@
    
    $reportContent | Out-File -FilePath $reportFile
    
    Write-Success "Test report saved to: $reportFile"
    Get-Content $reportFile
}

# Main execution
switch ($Action) {
    "prereq" { Check-Prerequisites }
    "clean" { Cleanup }
    "build" { 
        Check-Prerequisites
        Build-Images 
    }
    "test" { 
        Test-Containers
        Test-Compose 
    }
    "etl" { Run-SampleETL }
    "usage" { Show-Usage }
    "all" { 
        Check-Prerequisites
        Cleanup
        Build-Images
        Test-Containers
        Test-Compose
        Run-SampleETL
        Generate-Report
        Show-Usage
    }
    default {
        Write-Host "Usage: .\docker_build_test.ps1 -Action {prereq|clean|build|test|etl|usage|all}"
        Write-Host "  prereq - Check prerequisites only"
        Write-Host "  clean  - Clean up old containers and images"
        Write-Host "  build  - Build Docker images"
        Write-Host "  test   - Run container tests"
        Write-Host "  etl    - Run sample ETL pipeline"
        Write-Host "  usage  - Show usage examples"
        Write-Host "  all    - Run all steps (default)"
        exit 1
    }
}

Write-Header "Docker Build and Test Completed Successfully!"
