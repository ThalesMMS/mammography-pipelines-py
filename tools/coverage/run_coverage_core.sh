#!/usr/bin/env bash
# Run coverage verification on core tests (without pre-existing failures)
set -euo pipefail

# Find Python executable
PYTHON_CMD=""
for cmd in python3 python py; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No Python found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Run coverage report on core test files (the ones we modified)
echo "Running coverage verification on core tests..."
EXIT_CODE=0
$PYTHON_CMD -m pytest \
    tests/unit/test_models.py \
    tests/unit/test_cancer_models.py \
    tests/unit/test_vit_models.py \
    --cov=mammography.models \
    --cov-report=term-missing \
    --cov-fail-under=80 \
    --no-header \
    -v \
    || EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Coverage verification PASSED - 80%+ threshold met"
else
    echo "❌ Coverage verification FAILED - Below 80% threshold"
fi

exit $EXIT_CODE
