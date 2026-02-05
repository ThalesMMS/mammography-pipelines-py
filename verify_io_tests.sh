#!/bin/bash
# Verification script for I/O module unit tests
# This can be run manually to verify the fixes

echo "Running I/O module unit tests..."
python -m pytest tests/unit/test_dicom.py tests/unit/test_dicom_cache.py tests/unit/test_lazy_dicom.py -v

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✓ All I/O module unit tests passed!"
else
    echo "✗ Some tests failed (exit code: $exit_code)"
fi

exit $exit_code
