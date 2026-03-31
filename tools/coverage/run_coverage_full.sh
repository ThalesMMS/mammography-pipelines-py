#!/usr/bin/env bash
# Run coverage verification without maxfail restriction

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

# Run coverage report with --maxfail=999 to get full coverage report
echo "Running full coverage verification..."
TMPFILE=$(mktemp)
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
    --maxfail=999 \
    --no-header \
    -q > "$TMPFILE" 2>&1
EXIT_CODE=$?

echo ""
echo "Coverage output (last 100 lines):"
tail -100 "$TMPFILE"
rm -f "$TMPFILE"

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "✅ Full coverage verification PASSED - 80%+ threshold met"
else
    echo "❌ Full coverage verification FAILED"
fi

exit $EXIT_CODE
