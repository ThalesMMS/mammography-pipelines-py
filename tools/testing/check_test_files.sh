#!/usr/bin/env bash
# Check ruff lint on specific test files for subtask-5-3

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

# Run ruff check on the three test files
echo "Running ruff check on test files..."
"$PYTHON_CMD" -m ruff check \
    tests/unit/test_io_mammography_image.py \
    tests/unit/test_io_dicom_reader.py \
    tests/unit/test_io_cache_modes.py 2>&1

exit $?
