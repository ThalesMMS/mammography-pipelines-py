#!/bin/bash
# Full Test Suite Execution Script for Vision Transformer Implementation
# Subtask 5-1: Run full test suite to ensure no regressions
#
# Requirements:
#   - Python 3.11+ (as per pyproject.toml)
#   - Virtual environment with dependencies installed
#
# Usage:
#   ./run_full_test_suite.sh [options]
#
# Options:
#   --quick    : Run only fast tests (exclude slow tests)
#   --verbose  : Run with verbose output
#   --coverage : Run with coverage reporting
#   --help     : Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
QUICK_MODE=false
VERBOSE_MODE=false
COVERAGE_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --coverage)
            COVERAGE_MODE=true
            shift
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# //g'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Vision Transformer Test Suite"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

echo "Checking Python version..."
echo "Current: Python $PYTHON_VERSION"
echo "Required: Python >=$REQUIRED_VERSION"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${RED}ERROR: Python 3.11+ is required${NC}"
    echo "Please activate a virtual environment with Python 3.11+"
    exit 1
fi

echo -e "${GREEN}✓ Python version check passed${NC}"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
REQUIRED_PACKAGES="pytest torch torchvision timm numpy"
MISSING_PACKAGES=""

for package in $REQUIRED_PACKAGES; do
    if ! python -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo -e "${RED}ERROR: Missing required packages:$MISSING_PACKAGES${NC}"
    echo ""
    echo "Install dependencies with:"
    echo "  pip install -e ."
    echo "  # or"
    echo "  pip install pytest torch torchvision timm numpy"
    exit 1
fi

echo -e "${GREEN}✓ All required dependencies installed${NC}"
echo ""

# Build pytest command
PYTEST_CMD="python -m pytest tests/"
PYTEST_ARGS="-v --tb=short"

if [ "$QUICK_MODE" = true ]; then
    echo "Running in QUICK mode (excluding slow tests)..."
    PYTEST_ARGS="$PYTEST_ARGS -k 'not slow'"
fi

if [ "$VERBOSE_MODE" = true ]; then
    echo "Running in VERBOSE mode..."
    PYTEST_ARGS="$PYTEST_ARGS -vv"
fi

if [ "$COVERAGE_MODE" = true ]; then
    echo "Running with COVERAGE reporting..."
    PYTEST_CMD="python -m pytest --cov=src/mammography --cov-report=html --cov-report=term tests/"
fi

echo "========================================"
echo "Test Execution"
echo "========================================"
echo ""
echo "Command: $PYTEST_CMD $PYTEST_ARGS"
echo ""

# Run the tests
if $PYTEST_CMD $PYTEST_ARGS; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo "========================================"
    echo ""
    echo "Test suite completed successfully!"
    echo ""
    echo "Key validations:"
    echo "  ✓ No regressions in existing functionality"
    echo "  ✓ Vision Transformer models working correctly"
    echo "  ✓ ViT/DeiT embedding extraction functional"
    echo "  ✓ Integration with existing pipeline verified"
    echo ""
    exit 0
else
    echo ""
    echo "========================================"
    echo -e "${RED}✗ TESTS FAILED${NC}"
    echo "========================================"
    echo ""
    echo "Some tests failed. Please review the output above."
    echo ""
    echo "Common issues:"
    echo "  - Import errors: Ensure all dependencies are installed"
    echo "  - Model loading errors: Check internet connection for pretrained weights"
    echo "  - CUDA errors: Ensure PyTorch is configured correctly for your hardware"
    echo ""
    exit 1
fi
