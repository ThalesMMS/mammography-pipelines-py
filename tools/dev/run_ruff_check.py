#!/usr/bin/env python3
"""Run ruff check on specific test files."""

import subprocess
import sys

def main():
    """Run ruff check on the three test files."""
    files = [
        "tests/unit/test_io_mammography_image.py",
        "tests/unit/test_io_dicom_reader.py",
        "tests/unit/test_io_cache_modes.py",
    ]

    print("Running ruff check on test files...")
    print(f"Files: {', '.join(files)}")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check"] + files,
            capture_output=True,
            text=True,
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Print result
        print()
        if result.returncode == 0:
            print("✓ Ruff check PASSED - No lint errors found")
        else:
            print(f"⚠ Ruff check found issues (exit code: {result.returncode})")

        return result.returncode

    except FileNotFoundError:
        print("ERROR: ruff module not found")
        print("Try installing with: pip install ruff")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
