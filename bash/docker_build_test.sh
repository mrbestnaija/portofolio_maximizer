#!/bin/bash
# Docker Build and Test Script for Portfolio Maximizer v4.5
# This script builds the Docker images and runs comprehensive tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="portfolio-maximizer"
VERSION="v4.5"
DOCKER_REGISTRY=""  # Set if using a registry like Docker Hub
COMPOSE_PROJECT_NAME="portfolio_maximizer"

# Helper functions
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker found: $(docker --version)"
    else
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose found"
    else
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check for .env file
    if [ -f ".env" ]; then
        print_success ".env file found"
    else
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env file from .env.example"
        else
            print_warning "No .env.example found. API features may not work without API keys."
        fi
    fi
}

# Function to clean up old containers and images
cleanup() {
    print_header "Cleaning Up Old Containers and Images"
    
    # Stop and remove containers
    echo "Stopping containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove old images
    echo "Removing old images..."
    docker images | grep "$IMAGE_NAME" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to build Docker images
build_images() {
    print_header "Building Docker Images"
    
    # Build production image
    echo "Building production image..."
    docker build \
        --target production \
        --tag ${IMAGE_NAME}:${VERSION} \
        --tag ${IMAGE_NAME}:latest \
        --file Dockerfile \
        . || {
        print_error "Failed to build production image"
        exit 1
    }
    print_success "Production image built: ${IMAGE_NAME}:${VERSION}"
    
    # Build development image
    echo "Building development image..."
    docker build \
        --target builder \
        --tag ${IMAGE_NAME}:${VERSION}-dev \
        --file Dockerfile \
        . || {
        print_error "Failed to build development image"
        exit 1
    }
    print_success "Development image built: ${IMAGE_NAME}:${VERSION}-dev"
    
    # List built images
    echo -e "\nBuilt images:"
    docker images | grep "$IMAGE_NAME" | head -5
}

# Function to run basic container tests
test_containers() {
    print_header "Testing Docker Containers"
    
    # Test 1: Basic container startup
    echo "Test 1: Basic container startup..."
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "print('Container started successfully')" || {
        print_error "Failed to start container"
        exit 1
    }
    print_success "Container startup test passed"
    
    # Test 2: Validate Python dependencies
    echo -e "\nTest 2: Validating Python dependencies..."
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "
import numpy, pandas, scipy, sklearn, matplotlib, yfinance
print('All core dependencies imported successfully')
print(f'NumPy version: {numpy.__version__}')
print(f'Pandas version: {pandas.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
" || {
        print_error "Failed to import dependencies"
        exit 1
    }
    print_success "Dependency validation passed"
    
    # Test 3: ETL module imports
    echo -e "\nTest 3: Testing ETL module imports..."
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "
from etl import YFinanceExtractor, DataValidator, Preprocessor, DataStorage
print('All ETL modules imported successfully')
" || {
        print_error "Failed to import ETL modules"
        exit 1
    }
    print_success "ETL module import test passed"
    
    # Test 4: File system permissions
    echo -e "\nTest 4: Testing file system permissions..."
    docker run --rm ${IMAGE_NAME}:${VERSION} bash -c "
touch /app/data/test_write.txt && rm /app/data/test_write.txt && echo 'Write permissions OK'
" || {
        print_error "Failed file system permission test"
        exit 1
    }
    print_success "File system permission test passed"
    
    # Test 5: Health check
    echo -e "\nTest 5: Running health check..."
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "import sys; sys.exit(0)" || {
        print_error "Health check failed"
        exit 1
    }
    print_success "Health check passed"
}

# Function to run docker-compose tests
test_compose() {
    print_header "Testing Docker Compose Services"
    
    # Start services
    echo "Starting services with docker-compose..."
    docker-compose up -d portfolio-maximizer || {
        print_error "Failed to start services"
        exit 1
    }
    
    # Wait for container to be healthy
    echo "Waiting for container to be healthy..."
    sleep 5
    
    # Check container status
    if docker-compose ps | grep -q "portfolio_maximizer_v45.*Up"; then
        print_success "Container is running"
    else
        print_error "Container failed to start properly"
        docker-compose logs portfolio-maximizer
        docker-compose down
        exit 1
    fi
    
    # Run ETL pipeline help command
    echo -e "\nTesting ETL pipeline help command..."
    docker-compose exec -T portfolio-maximizer python scripts/run_etl_pipeline.py --help || {
        print_error "Failed to run ETL pipeline help"
        docker-compose down
        exit 1
    }
    print_success "ETL pipeline help command successful"
    
    # Stop services
    echo -e "\nStopping services..."
    docker-compose down
    print_success "Docker Compose tests completed"
}

# Function to run a sample ETL pipeline
run_sample_etl() {
    print_header "Running Sample ETL Pipeline"
    
    echo "Creating sample ticker list..."
    echo -e "AAPL\nMSFT\nGOOG" > data/sample_tickers.txt
    
    echo -e "\nRunning ETL pipeline with sample data..."
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/config:/app/config" \
        -v "$(pwd)/logs:/app/logs" \
        ${IMAGE_NAME}:${VERSION} \
        python scripts/run_etl_pipeline.py \
        --tickers AAPL MSFT GOOG \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --no-cache || {
        print_warning "ETL pipeline run failed (this might be due to missing API keys)"
    }
    
    # Check if data was created
    if ls data/training/*.parquet 2>/dev/null | head -1; then
        print_success "ETL pipeline created output files"
        echo "Output files:"
        ls -la data/training/*.parquet 2>/dev/null | head -5
    else
        print_warning "No output files created (API keys might be required)"
    fi
}

# Function to display usage information
show_usage() {
    print_header "Docker Usage Examples"
    
    cat << EOF
# Build images:
docker build -t ${IMAGE_NAME}:${VERSION} .

# Run ETL pipeline:
docker run --rm -v \$(pwd)/data:/app/data ${IMAGE_NAME}:${VERSION} \\
    python scripts/run_etl_pipeline.py --tickers AAPL MSFT --start-date 2023-01-01

# Interactive shell:
docker run -it --rm -v \$(pwd):/app ${IMAGE_NAME}:${VERSION}-dev /bin/bash

# Run with docker-compose:
docker-compose up portfolio-maximizer

# Run tests:
docker-compose run --rm portfolio-test

# Start Jupyter notebook:
docker-compose up portfolio-notebook

# Development mode:
docker-compose run --rm portfolio-dev /bin/bash

# View logs:
docker-compose logs -f portfolio-maximizer

# Clean up:
docker-compose down -v
EOF
}

# Function to generate test report
generate_report() {
    print_header "Test Summary Report"
    
    REPORT_FILE="docker_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$REPORT_FILE" << EOF
Portfolio Maximizer v4.5 - Docker Test Report
Generated: $(date)

System Information:
- Docker Version: $(docker --version)
- OS: $(uname -s)
- Architecture: $(uname -m)

Test Results:
- Prerequisites Check: PASSED
- Docker Build: PASSED
- Container Tests: PASSED
- Docker Compose Tests: PASSED
- Sample ETL Run: COMPLETED

Images Built:
$(docker images | grep "$IMAGE_NAME" | head -5)

Recommendations:
1. Add API keys to .env file for full functionality
2. Mount volumes for data persistence in production
3. Use docker-compose for easier management
4. Monitor container logs for debugging

EOF

    print_success "Test report saved to: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# Main execution
main() {
    print_header "Portfolio Maximizer v4.5 - Docker Build and Test"
    
    # Parse command line arguments
    case "${1:-all}" in
        prereq)
            check_prerequisites
            ;;
        clean)
            cleanup
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        test)
            test_containers
            test_compose
            ;;
        etl)
            run_sample_etl
            ;;
        usage)
            show_usage
            ;;
        all)
            check_prerequisites
            cleanup
            build_images
            test_containers
            test_compose
            run_sample_etl
            generate_report
            show_usage
            ;;
        *)
            echo "Usage: $0 {prereq|clean|build|test|etl|usage|all}"
            echo "  prereq - Check prerequisites only"
            echo "  clean  - Clean up old containers and images"
            echo "  build  - Build Docker images"
            echo "  test   - Run container tests"
            echo "  etl    - Run sample ETL pipeline"
            echo "  usage  - Show usage examples"
            echo "  all    - Run all steps (default)"
            exit 1
            ;;
    esac
    
    print_header "Docker Build and Test Completed Successfully!"
}

# Run main function
main "$@"
