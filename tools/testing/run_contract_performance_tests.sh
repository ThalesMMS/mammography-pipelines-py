#!/usr/bin/env bash
# Test runner for contract, performance, and reproducibility tests
# This script finds Python and runs the specified tests

set -e

# Find Python executable
PYTHON_CMD=""

# Try specific paths first (Windows)
for python_path in \
    "/c/Users/$USER/.lmstudio/extensions/backends/vendor/_amphibian/cpython3.11-win-x86@2/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python313/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python312/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python311/python.exe" \
    "/c/Python313/python.exe" \
    "/c/Python312/python.exe" \
    "/c/Python311/python.exe" \
    "/usr/bin/python3" \
    "/usr/local/bin/python3"
do
    if [ -f "${python_path}" ]; then
        PYTHON_CMD="${python_path}"
        break
    fi
done

# Try commands in PATH if not found
if [ -z "$PYTHON_CMD" ]; then
    for cmd in python python3 python3.11 python3.12; do
        if command -v $cmd &> /dev/null; then
            PYTHON_CMD=$cmd
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found"
    echo "Please ensure Python 3.11+ is installed and in PATH"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Run pytest
echo ""
echo "Running contract, performance, and reproducibility tests..."
echo "=========================================="
$PYTHON_CMD -m pytest tests/contract/ tests/performance/ tests/reproducibility/ -v --tb=short "$@"
