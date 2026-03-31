#!/usr/bin/env bash
# Check ruff lint on specific test files for subtask-5-3 (verbose version)

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
    echo "Checking: $python_path"
    if [ -f "$python_path" ]; then
        PYTHON_CMD="$python_path"
        echo "Found Python: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No Python installation found"

    # Try alternative methods
    echo "Trying 'python3'..."
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "Using python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo "Using python"
    else
        echo "No python command found at all"
        exit 1
    fi
fi

echo "Python command: $PYTHON_CMD"
echo "Python version:"
"$PYTHON_CMD" --version

echo ""
echo "Checking if ruff is installed..."
"$PYTHON_CMD" -c "import ruff; print('ruff is installed')" 2>&1 || {
    echo "ruff not found, trying to import anyway..."
}

echo ""
echo "Running ruff check on test files..."
"$PYTHON_CMD" -m ruff check \
    tests/unit/test_io_mammography_image.py \
    tests/unit/test_io_dicom_reader.py \
    tests/unit/test_io_cache_modes.py 2>&1

exit_code=$?
echo ""
echo "Exit code: $exit_code"
exit $exit_code
