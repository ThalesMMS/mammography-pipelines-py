#!/usr/bin/env python
"""Run data module tests with coverage measurement."""
import subprocess
import sys

def main():
    """Run pytest with coverage for data module tests."""
    test_files = [
        "tests/unit/test_data.py",
        "tests/unit/test_format_detection.py",
        "tests/unit/test_splits.py",
        "tests/unit/test_cancer_dataset.py",
    ]

    cmd = [
        sys.executable, "-m", "pytest"
    ] + test_files + [
        "--cov=mammography.data",
        "--cov-report=term-missing",
        "-v"
    ]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
