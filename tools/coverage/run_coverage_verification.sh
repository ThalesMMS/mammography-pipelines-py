#!/usr/bin/env bash
# Run coverage verification for training and tracking modules
# Subtask 5-1: Verify 80%+ coverage and all tests pass

set -e

# Find Python - try Python313 first to avoid PyTorch docstring bug in Python312
PYTHON_CMD="/c/Users/user/AppData/Local/Programs/Python/Python313/python.exe"

if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="/c/Users/user/AppData/Local/Programs/Python/Python312/python.exe"
fi

if [ ! -f "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Running coverage verification for training and tracking modules..."
echo ""

# Run the tests with coverage
"$PYTHON_CMD" -m pytest tests/unit/ -m "not slow and not gpu" \
    --cov=mammography.training --cov=mammography.tracking \
    --cov-report=term-missing --cov-fail-under=80 -v

exit $?
