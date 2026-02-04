#!/bin/bash
#
# Integration test for mammography tune command
# Tests the complete workflow with minimal configuration
#

set -e

echo "====================================================================="
echo "Integration Test: mammography tune --dry-run"
echo "====================================================================="

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
if ! python -c "import mammography" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -e . --quiet
fi

# Run the integration test with dry-run
echo ""
echo "Running: mammography tune --dataset patches_completo --n-trials 2 --subset 50 --epochs 1 --dry-run"
echo ""

PYTHONPATH=./src python -m mammography.cli tune \
    --dataset patches_completo \
    --n-trials 2 \
    --subset 50 \
    --epochs 1 \
    --dry-run

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "====================================================================="
    echo "Integration test PASSED"
    echo "====================================================================="
    exit 0
else
    echo ""
    echo "====================================================================="
    echo "Integration test FAILED"
    echo "====================================================================="
    exit 1
fi
