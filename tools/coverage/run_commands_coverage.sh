#!/bin/bash
# Script to run coverage tests for commands module
# Tries multiple Python invocation methods for Windows compatibility

set +e  # Don't exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Attempting to run pytest with coverage for mammography.commands ==="

# Method 1: Direct pytest command
echo "Method 1: Direct pytest..."
if command -v pytest &> /dev/null; then
    pytest tests/unit/test_commands_*.py tests/unit/test_wizard.py --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80 2>&1 | tee commands_coverage_output.txt
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Method 1 succeeded - Coverage check PASSED"
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "✓ Method 1 ran but tests failed or coverage not met"
        cat commands_coverage_output.txt
        exit 1
    fi
fi
echo "Method 1 skipped - pytest not found"

# Method 2: python -m pytest
echo "Method 2: python -m pytest..."
if command -v python &> /dev/null; then
    python -m pytest tests/unit/test_commands_*.py tests/unit/test_wizard.py --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80 2>&1 | tee commands_coverage_output.txt
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Method 2 succeeded - Coverage check PASSED"
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "✓ Method 2 ran but tests failed or coverage not met"
        cat commands_coverage_output.txt
        exit 1
    fi
fi
echo "Method 2 skipped - python not found"

# Method 3: python3 -m pytest
echo "Method 3: python3 -m pytest..."
if command -v python3 &> /dev/null; then
    python3 -m pytest tests/unit/test_commands_*.py tests/unit/test_wizard.py --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80 2>&1 | tee commands_coverage_output.txt
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Method 3 succeeded - Coverage check PASSED"
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "✓ Method 3 ran but tests failed or coverage not met"
        cat commands_coverage_output.txt
        exit 1
    fi
fi
echo "Method 3 skipped - python3 not found"

echo "✗ All methods failed - pytest/python not available in PATH"
echo "Please run manually: pytest tests/unit/test_commands_*.py tests/unit/test_wizard.py --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80"
exit 127
