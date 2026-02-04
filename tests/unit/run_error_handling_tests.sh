#!/bin/bash
#
# Helper script to run error handling tests
# Usage: ./run_error_handling_tests.sh
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#

echo "Running error handling tests..."
echo "================================"
echo ""

# Try to find Python
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
else
    echo "Error: Python not found in PATH"
    echo "Please ensure Python is installed and accessible"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Run tests
$PYTHON_CMD -m pytest tests/unit/test_error_handling.py -v --tb=short

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All error handling tests passed!"
else
    echo ""
    echo "✗ Some tests failed. See output above for details."
    exit 1
fi
