#!/usr/bin/env bash
# Script to run linting (ruff) and type checking (mypy)
# Created for subtask-4-3 - bypasses Python PATH issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "Linting and Type Checking"
echo "Subtask 4-3 Verification"
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

# Check if ruff is available
echo -e "${BLUE}Checking ruff installation...${NC}"
if ! "$PYTHON_CMD" -c "import ruff" 2>/dev/null; then
    echo -e "${YELLOW}⚠ ruff not installed, installing...${NC}"
    "$PYTHON_CMD" -m pip install ruff --quiet || {
        echo -e "${RED}ERROR: Failed to install ruff${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ ruff installed${NC}"
else
    echo -e "${GREEN}✓ ruff is installed${NC}"
fi

# Check if mypy is available
echo -e "${BLUE}Checking mypy installation...${NC}"
if ! "$PYTHON_CMD" -c "import mypy" 2>/dev/null; then
    echo -e "${YELLOW}⚠ mypy not installed, installing...${NC}"
    "$PYTHON_CMD" -m pip install mypy --quiet || {
        echo -e "${RED}ERROR: Failed to install mypy${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ mypy installed${NC}"
else
    echo -e "${GREEN}✓ mypy is installed${NC}"
fi

echo ""

# Run ruff check
echo "========================================"
echo "Running ruff linter..."
echo "Command: ruff check src tests"
echo "========================================"
echo ""

RUFF_EXIT_CODE=0
"$PYTHON_CMD" -m ruff check src tests || RUFF_EXIT_CODE=$?

echo ""
if [ $RUFF_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Ruff linting passed (no errors)${NC}"
else
    echo -e "${YELLOW}⚠ Ruff found some issues (exit code: $RUFF_EXIT_CODE)${NC}"
fi

# Run mypy type checking
echo ""
echo "========================================"
echo "Running mypy type checker..."
echo "Command: mypy src"
echo "========================================"
echo ""

MYPY_EXIT_CODE=0
"$PYTHON_CMD" -m mypy src || MYPY_EXIT_CODE=$?

echo ""
if [ $MYPY_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Mypy type checking passed (no errors)${NC}"
else
    echo -e "${YELLOW}⚠ Mypy found some issues (exit code: $MYPY_EXIT_CODE)${NC}"
fi

# Overall summary
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""

if [ $RUFF_EXIT_CODE -eq 0 ] && [ $MYPY_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓✓ ALL CHECKS PASSED ✓✓${NC}"
    echo "  ✓ Ruff linting: PASSED"
    echo "  ✓ Mypy type checking: PASSED"
    echo ""
    echo "Subtask 4-3 verification: PASSED"
    exit 0
else
    echo -e "${YELLOW}Code quality checks completed with issues:${NC}"
    if [ $RUFF_EXIT_CODE -ne 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Ruff linting: FAILED (exit code: $RUFF_EXIT_CODE)"
    else
        echo -e "  ${GREEN}✓${NC} Ruff linting: PASSED"
    fi

    if [ $MYPY_EXIT_CODE -ne 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Mypy type checking: FAILED (exit code: $MYPY_EXIT_CODE)"
    else
        echo -e "  ${GREEN}✓${NC} Mypy type checking: PASSED"
    fi

    echo ""
    echo "Note: Some issues may be pre-existing. Check if errors are NEW."
    echo "Subtask 4-3 verification: Review required"

    # Exit with error if there are failures
    exit 1
fi
