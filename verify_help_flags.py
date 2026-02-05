#!/usr/bin/env python3
"""
Comprehensive test script to verify --help works for all CLI subcommands.

This script can be run once Python environment is properly configured.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# All subcommands from cli.py
SUBCOMMANDS = [
    "embed",
    "train-density",
    "embeddings-baselines",
    "inference",
    "explain",
    "visualize",
    "tune",
    "eda-cancer",
    "wizard",
    "augment",
    "label-density",
    "label-patches",
    "data-audit",
    "eval-export",
    "report-pack",
]


def test_help_direct() -> Tuple[int, int, List[str]]:
    """Test --help by directly importing and calling the CLI."""
    print("Method 1: Direct import test")
    print("=" * 60)

    try:
        from mammography.cli import _build_parser

        parser = _build_parser()
        passed = 0
        failed = 0
        failures = []

        # Test main help
        print("Testing: mammography --help ... ", end="", flush=True)
        try:
            parser.parse_args(["--help"])
            print("✓ OK (would exit)")
            passed += 1
        except SystemExit as e:
            if e.code == 0:
                print("✓ OK")
                passed += 1
            else:
                print(f"✗ FAILED (exit code {e.code})")
                failed += 1
                failures.append("main")

        # Test each subcommand
        for cmd in SUBCOMMANDS:
            print(f"Testing: {cmd} --help ... ", end="", flush=True)
            try:
                parser.parse_args([cmd, "--help"])
                print("✓ OK (would exit)")
                passed += 1
            except SystemExit as e:
                if e.code == 0:
                    print("✓ OK")
                    passed += 1
                else:
                    print(f"✗ FAILED (exit code {e.code})")
                    failed += 1
                    failures.append(cmd)
            except Exception as e:
                print(f"✗ FAILED ({type(e).__name__}: {e})")
                failed += 1
                failures.append(cmd)

        return passed, failed, failures

    except ImportError as e:
        print(f"✗ Cannot import mammography.cli: {e}")
        print("Make sure PYTHONPATH includes ./src")
        return 0, len(SUBCOMMANDS) + 1, ["import-error"]


def test_help_subprocess() -> Tuple[int, int, List[str]]:
    """Test --help by running as subprocess (python -m mammography.cli)."""
    print("\nMethod 2: Subprocess test")
    print("=" * 60)

    passed = 0
    failed = 0
    failures = []

    # Test main help
    print("Testing: mammography --help ... ", end="", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mammography.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PYTHONPATH": str(Path(__file__).parent / "src")}
        )
        if result.returncode == 0 and "mammography" in result.stdout:
            print("✓ OK")
            passed += 1
        else:
            print(f"✗ FAILED (exit {result.returncode})")
            failed += 1
            failures.append("main")
    except Exception as e:
        print(f"✗ FAILED ({type(e).__name__})")
        failed += 1
        failures.append("main")

    # Test each subcommand
    for cmd in SUBCOMMANDS:
        print(f"Testing: {cmd} --help ... ", end="", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mammography.cli", cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=5,
                env={"PYTHONPATH": str(Path(__file__).parent / "src")}
            )
            if result.returncode == 0:
                print("✓ OK")
                passed += 1
            else:
                print(f"✗ FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")
                failed += 1
                failures.append(cmd)
        except subprocess.TimeoutExpired:
            print("✗ FAILED (timeout)")
            failed += 1
            failures.append(cmd)
        except Exception as e:
            print(f"✗ FAILED ({type(e).__name__})")
            failed += 1
            failures.append(cmd)

    return passed, failed, failures


def main() -> int:
    """Run all verification tests."""
    print("CLI --help Flags Verification")
    print("=" * 60)
    print(f"Testing {len(SUBCOMMANDS)} subcommands + main command\n")

    # Try direct import first
    passed1, failed1, failures1 = test_help_direct()

    # Try subprocess method
    passed2, failed2, failures2 = test_help_subprocess()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Method 1 (Direct):     {passed1} passed, {failed1} failed")
    print(f"Method 2 (Subprocess): {passed2} passed, {failed2} failed")

    if failed1 == 0:
        print("\n✅ All tests PASSED (Method 1)")
        return 0
    elif failed2 == 0:
        print("\n✅ All tests PASSED (Method 2)")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        if failures1:
            print(f"  Method 1 failures: {', '.join(failures1)}")
        if failures2:
            print(f"  Method 2 failures: {', '.join(failures2)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
