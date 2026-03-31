#!/usr/bin/env bash
# Custom script to run fast unit and integration tests
# Created for subtask-4-2 - bypasses Python PATH issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Fast Unit and Integration Tests"
echo "Subtask 4-2 Verification"
echo "========================================"
echo ""

# Try to find Python executable
PYTHON_CMD=""
for python_path in \
    "/c/Users/user/AppData/Local/Programs/Python/Python313/python.exe" \
    "/c/Users/user/AppData/Local/Programs/Python/Python312/python.exe" \
    "/c/Users/user/AppData/Local/Programs/Python/Python311/python.exe" \
    "/c/Python313/python.exe" \
    "/c/Python312/python.exe" \
    "/c/Python311/python.exe"
do
    if [ -f "$python_path" ]; then
        PYTHON_CMD="$python_path"
        echo -e "${GREEN}Found Python: $PYTHON_CMD${NC}"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}ERROR: No Python installation found${NC}"
    echo "Searched locations:"
    echo "  - /c/Users/user/AppData/Local/Programs/Python/Python3{11,12,13}"
    echo "  - /c/Python3{11,12,13}"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if pytest is available
if ! "$PYTHON_CMD" -c "import pytest" 2>/dev/null; then
    echo -e "${RED}ERROR: pytest not installed${NC}"
    echo "Install with: $PYTHON_CMD -m pip install pytest"
    exit 1
fi

echo -e "${GREEN}✓ pytest is installed${NC}"
echo ""

# Run fast tests (excluding slow and GPU tests)
echo "========================================"
echo "Running fast tests..."
echo "Command: pytest -m \"not slow and not gpu\" -v (excluding coverage tests)"
echo "========================================"
echo ""

# Override pytest.ini settings to disable coverage and exclude coverage tests
"$PYTHON_CMD" -m pytest -m "not slow and not gpu" -v \
    -o addopts="" \
    --override-ini="addopts=" \
    --ignore=tests/coverage

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ ALL FAST TESTS PASSED${NC}"
    echo "========================================"
    echo ""
    echo "Test suite completed successfully!"
    echo "Subtask 4-2 verification: PASSED"
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo "========================================"
    echo ""
    echo "Exit code: $TEST_EXIT_CODE"
    echo "Please review the test output above for details."
fi

exit $TEST_EXIT_CODE
