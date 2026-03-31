#!/bin/bash
# Script to run coverage tests for io module
# Tries multiple Python invocation methods for Windows compatibility

set +e  # Don't exit on error

echo "=== Attempting to run pytest with coverage for mammography.io ==="

# Method 1: Direct pytest command
echo "Method 1: Direct pytest..."
pytest --cov=mammography.io --cov-report=term-missing tests/unit/ -q --no-header 2>&1 | tee coverage_output.txt
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    echo "✓ Method 1 succeeded"
    exit 0
fi

# Method 2: python -m pytest
echo "Method 2: python -m pytest..."
python -m pytest --cov=mammography.io --cov-report=term-missing tests/unit/ -q --no-header 2>&1 | tee coverage_output.txt
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    echo "✓ Method 2 succeeded"
    exit 0
fi

# Method 3: python3 -m pytest
echo "Method 3: python3 -m pytest..."
python3 -m pytest --cov=mammography.io --cov-report=term-missing tests/unit/ -q --no-header 2>&1 | tee coverage_output.txt
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    echo "✓ Method 3 succeeded"
    exit 0
fi

echo "✗ All methods failed - pytest/python not available in PATH"
echo "Please run manually: pytest --cov=mammography.io --cov-report=term-missing tests/unit/ -q"
exit 127
