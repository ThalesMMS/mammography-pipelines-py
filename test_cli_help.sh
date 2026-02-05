#!/usr/bin/env bash
# Test all CLI subcommands with --help flag

# Add src to PYTHONPATH
export PYTHONPATH="./src:$PYTHONPATH"

# List of all subcommands
commands=(
    "embed"
    "train-density"
    "embeddings-baselines"
    "inference"
    "explain"
    "visualize"
    "tune"
    "eda-cancer"
    "wizard"
    "augment"
    "label-density"
    "label-patches"
    "data-audit"
    "eval-export"
    "report-pack"
)

echo "Testing --help for all subcommands..."
echo "======================================"
echo ""

failed=0
passed=0

for cmd in "${commands[@]}"; do
    echo -n "Testing: $cmd --help ... "
    # Try to run the command
    if /usr/bin/env python3 -m mammography.cli "$cmd" --help > /dev/null 2>&1; then
        echo "✓ OK"
        ((passed++))
    else
        echo "✗ FAILED"
        ((failed++))
    fi
done

echo ""
echo "======================================"
echo "Results: $passed passed, $failed failed"

if [ $failed -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
