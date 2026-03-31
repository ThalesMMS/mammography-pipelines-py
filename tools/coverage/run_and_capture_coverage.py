#!/usr/bin/env python
"""Run data module tests with coverage and save output to file."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    # Run and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )

    # Save to file
    output_file = REPO_ROOT / "tools" / "coverage" / "coverage_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write("=" * 80 + "\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write("=" * 80 + "\n")
        f.write(result.stderr)
        f.write(f"\n\nReturn code: {result.returncode}\n")

    print(f"Output saved to {output_file}")
    print(f"Return code: {result.returncode}")

    # Also print to console
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
