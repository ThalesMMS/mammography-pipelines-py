#!/usr/bin/env python
"""Test runner script for contract, performance, and reproducibility tests."""
import subprocess
import sys

def main():
    """Run the tests using pytest."""
    test_dirs = [
        "tests/contract/",
        "tests/performance/",
        "tests/reproducibility/",
    ]

    cmd = [sys.executable, "-m", "pytest"] + test_dirs + ["-v", "--tb=short"]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
