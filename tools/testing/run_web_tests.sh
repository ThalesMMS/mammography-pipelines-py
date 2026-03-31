#!/usr/bin/env bash
# Test runner script for test_commands_web.py
# Finds Python interpreter and runs pytest

set -e

# Try to find Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    # Try Windows Python launcher
    if command -v py &> /dev/null; then
        PYTHON=py
    else
        echo "Error: Python not found in PATH"
        exit 1
    fi
fi

echo "Using Python: $PYTHON"

# Run the tests
$PYTHON -m pytest tests/unit/test_commands_web.py -v
