#!/usr/bin/env bash
# Install dev dependencies and run coverage verification

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

# Install pytest-cov if not already installed
echo "Installing pytest-cov..."
$PYTHON_CMD -m pip install pytest-cov -q

# Run coverage report
echo "Running coverage verification..."
$PYTHON_CMD -m pytest \
    tests/unit/test_models.py \
    tests/unit/test_cancer_models.py \
    tests/unit/test_vit_models.py \
    tests/unit/test_efficientnet_feature_extraction.py \
    tests/unit/test_resnet_feature_extraction.py \
    tests/unit/test_vit_feature_extraction.py \
    --cov=mammography.models \
    --cov-report=term-missing \
    --cov-fail-under=80 \
    --no-header \
    -q

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
