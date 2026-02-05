#!/usr/bin/env bash
# Run comprehensive training tests

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
echo "Running comprehensive training tests..."
echo ""

# Run the tests without coverage (pytest-cov may not be installed)
"$PYTHON_CMD" -m pytest tests/unit/test_training_comprehensive.py -v -o addopts=""

exit $?
