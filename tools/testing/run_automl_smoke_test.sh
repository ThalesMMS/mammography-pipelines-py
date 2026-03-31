#!/usr/bin/env bash
# End-to-end smoke test for AutoML command
# This script runs the automl command with minimal settings to verify basic functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Find Python executable
PYTHON_EXE=""
for python_cmd in \
    "/c/Users/user/AppData/Local/Programs/Python/Python313/python.exe" \
    "/c/Users/user/AppData/Local/Programs/Python/Python312/python.exe" \
    "/c/Python313/python.exe" \
    "/c/Python312/python.exe" \
    "/c/Python311/python.exe"
do
    if [ -f "$python_cmd" ]; then
        PYTHON_EXE="$python_cmd"
        break
    fi
done

if [ -z "$PYTHON_EXE" ]; then
    echo "ERROR: Python executable not found"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Python version:"
"$PYTHON_EXE" --version

# Find pip executable
PIP_EXE=""
PYTHON_DIR=$(dirname "$PYTHON_EXE")
if [ -f "$PYTHON_DIR/Scripts/pip.exe" ]; then
    PIP_EXE="$PYTHON_DIR/Scripts/pip.exe"
fi

if [ -z "$PIP_EXE" ]; then
    echo "WARNING: pip executable not found, trying python -m pip"
    PIP_EXE="$PYTHON_EXE -m pip"
fi

echo "Using pip: $PIP_EXE"

# Install package in editable mode if not already installed
echo ""
echo "========================================="
echo "Installing mammography package"
echo "========================================="
echo ""

$PIP_EXE install -e . --quiet || {
    echo "WARNING: Package installation failed, attempting to continue anyway"
}

echo ""
echo "========================================="
echo "Running AutoML E2E Smoke Test"
echo "========================================="
echo ""
echo "Command: python -m mammography.cli automl --dataset mamografias --subset 32 --epochs 2 --n-trials 3"
echo ""

# Run the automl command with minimal settings
"$PYTHON_EXE" -m mammography.cli automl \
    --dataset mamografias \
    --subset 32 \
    --epochs 2 \
    --n-trials 3

echo ""
echo "========================================="
echo "Verification Checks"
echo "========================================="
echo ""

# Check for expected outputs
if [ -f "outputs/automl/best_params.json" ]; then
    echo "✓ best_params.json created"
    echo ""
    echo "Contents:"
    cat outputs/automl/best_params.json
    echo ""

    # Verify key parameters exist in the file
    if grep -q "arch" outputs/automl/best_params.json; then
        echo "✓ Architecture parameter found in best_params.json"
    else
        echo "✗ Architecture parameter NOT found in best_params.json"
        exit 1
    fi

    if grep -q "lr" outputs/automl/best_params.json; then
        echo "✓ Learning rate parameter found in best_params.json"
    else
        echo "✗ Learning rate parameter NOT found in best_params.json"
        exit 1
    fi

    if grep -q "augment" outputs/automl/best_params.json; then
        echo "✓ Augmentation parameter found in best_params.json"
    else
        echo "✗ Augmentation parameter NOT found in best_params.json"
        exit 1
    fi
else
    echo "✗ best_params.json NOT created"
    exit 1
fi

echo ""
echo "========================================="
echo "E2E Smoke Test PASSED"
echo "========================================="
