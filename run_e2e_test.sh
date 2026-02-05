#!/usr/bin/env bash
# End-to-end workflow test for subtask-4-4
# Tests embedding extraction with minimal test data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "End-to-End Workflow Test"
echo "Subtask 4-4 Verification"
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

# Check test_data directory exists
if [ ! -d "test_data" ]; then
    echo -e "${RED}ERROR: test_data directory not found${NC}"
    echo "Expected location: ./test_data"
    exit 1
fi

echo -e "${GREEN}✓ test_data directory found${NC}"
echo ""

# Check test_data contents
echo "Test data contents:"
ls -lh test_data/
echo ""
if [ -d "test_data/images" ]; then
    echo "Test images:"
    ls -lh test_data/images/
    echo ""
fi

# Run the embedding extraction workflow
echo "========================================"
echo "Running: mammography embed --dataset test_data --subset 5"
echo "========================================"
echo ""

"$PYTHON_CMD" -m mammography.cli embed --dataset test_data --subset 5 2>&1 | tee e2e_test_output.txt

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ END-TO-END WORKFLOW SUCCEEDED${NC}"
    echo "========================================"
    echo ""
    echo "The embed command completed successfully with minimal test data."
    echo "Subtask 4-4 verification: PASSED"
    echo ""
    echo "Output saved to: e2e_test_output.txt"
else
    echo -e "${RED}✗ END-TO-END WORKFLOW FAILED${NC}"
    echo "========================================"
    echo ""
    echo "Exit code: $TEST_EXIT_CODE"
    echo "Please review the test output above for details."
    echo ""
    echo "Full output saved to: e2e_test_output.txt"
fi

exit $TEST_EXIT_CODE
