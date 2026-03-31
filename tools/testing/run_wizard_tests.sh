#!/usr/bin/env bash
# Test runner for wizard command tests

set -e

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
        echo "Found Python: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No Python installation found"
    exit 1
fi

# Run wizard command tests
echo "Running wizard command tests..."
"$PYTHON_CMD" -m pytest tests/unit/test_wizard.py -v
