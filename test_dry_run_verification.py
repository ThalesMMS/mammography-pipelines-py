#!/usr/bin/env python3
"""
Verification script for subtask-2-2: Test --dry-run mode for main commands
This script tests that the --dry-run flag works correctly for CLI commands.
"""

import sys
from pathlib import Path
from io import StringIO
import logging

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography import cli


def test_dry_run_embed():
    """Test that embed command with --dry-run displays expected message"""
    # Capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger("projeto")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        # Run CLI with dry-run flag
        exit_code = cli.main(["--dry-run", "embed"])

        # Get captured output
        log_output = log_capture.getvalue()

        # Verify exit code
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        # Verify the dry-run message appears
        assert "Dry-run habilitado" in log_output, \
            f"Expected 'Dry-run habilitado' in output, got: {log_output}"

        print("✓ Test passed: embed --dry-run works correctly")
        print(f"  Output contains: 'Dry-run habilitado'")
        return True

    finally:
        logger.removeHandler(handler)


def test_dry_run_train_density():
    """Test that train-density command with --dry-run displays expected message"""
    # Capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger("projeto")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        # Run CLI with dry-run flag
        exit_code = cli.main(["--dry-run", "train-density"])

        # Get captured output
        log_output = log_capture.getvalue()

        # Verify exit code
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        # Verify the dry-run message appears
        assert "Dry-run habilitado" in log_output, \
            f"Expected 'Dry-run habilitado' in output, got: {log_output}"

        print("✓ Test passed: train-density --dry-run works correctly")
        print(f"  Output contains: 'Dry-run habilitado'")
        return True

    finally:
        logger.removeHandler(handler)


def main():
    """Run all dry-run verification tests"""
    print("=" * 60)
    print("Dry-Run Mode Verification (subtask-2-2)")
    print("=" * 60)
    print()

    all_passed = True

    try:
        print("Test 1: embed --dry-run")
        if not test_dry_run_embed():
            all_passed = False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        all_passed = False

    print()

    try:
        print("Test 2: train-density --dry-run")
        if not test_dry_run_train_density():
            all_passed = False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("✓ All dry-run tests PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ Some dry-run tests FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
