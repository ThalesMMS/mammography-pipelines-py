#!/bin/bash
# Script to run comprehensive coverage tests for commands module
# Includes all test files that test commands, not just test_commands_* pattern

set +e  # Don't exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Running comprehensive commands module coverage ==="
echo ""
echo "Test files included:"
echo "  - tests/unit/test_commands_*.py (7 files)"
echo "  - tests/unit/test_wizard.py"
echo "  - tests/unit/test_embeddings_baselines.py"
echo "  - tests/unit/test_eval_export.py"
echo "  - tests/unit/test_inference_command.py"
echo "  - tests/unit/test_preprocess_command.py"
echo "  - tests/unit/test_train.py"
echo ""

TEST_FILES="tests/unit/test_commands_*.py tests/unit/test_wizard.py tests/unit/test_embeddings_baselines.py tests/unit/test_eval_export.py tests/unit/test_inference_command.py tests/unit/test_preprocess_command.py tests/unit/test_train.py"

# Method 1: Direct pytest command
echo "Method 1: Direct pytest..."
if command -v pytest &> /dev/null; then
    pytest $TEST_FILES --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80 2>&1 | tee full_commands_coverage.txt
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Coverage check PASSED - 80%+ achieved"
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo ""
        echo "Tests ran but coverage requirement not met or tests failed"
        exit 1
    fi
fi
echo "Method 1 skipped - pytest not found"

# Method 2: python -m pytest
echo "Method 2: python -m pytest..."
if command -v python &> /dev/null; then
    python -m pytest $TEST_FILES --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80 2>&1 | tee full_commands_coverage.txt
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Coverage check PASSED - 80%+ achieved"
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo ""
        echo "Tests ran but coverage requirement not met or tests failed"
        exit 1
    fi
fi
echo "Method 2 skipped - python not found"

# Method 3: python3 -m pytest
echo "Method 3: python3 -m pytest..."
if command -v python3 &> /dev/null; then
    python3 -m pytest $TEST_FILES --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80 2>&1 | tee full_commands_coverage.txt
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Coverage check PASSED - 80%+ achieved"
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo ""
        echo "Tests ran but coverage requirement not met or tests failed"
        exit 1
    fi
fi
echo "Method 3 skipped - python3 not found"

echo ""
echo "✗ All methods failed - pytest/python not available in PATH"
echo ""
echo "To run manually:"
echo "  pytest $TEST_FILES --cov=mammography.commands --cov=mammography.wizard --cov-report=term-missing --cov-fail-under=80"
exit 127
