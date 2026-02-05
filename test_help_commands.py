#!/usr/bin/env python3
"""Test script to verify --help works for all subcommands."""

import subprocess
import sys
from pathlib import Path

# All subcommands to test
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

def test_help(subcommand: str) -> tuple[bool, str, str]:
    """Test if --help works for a subcommand.

    Returns:
        (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mammography.cli", subcommand, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # --help should exit with 0
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout expired"
    except Exception as e:
        return False, "", str(e)

def main():
    """Test all subcommands."""
    print(f"Testing --help for {len(SUBCOMMANDS)} subcommands...\n")

    results = {}
    failed = []

    for cmd in SUBCOMMANDS:
        print(f"Testing: {cmd} --help ... ", end="", flush=True)
        success, stdout, stderr = test_help(cmd)
        results[cmd] = (success, stdout, stderr)

        if success:
            print("✓ OK")
        else:
            print("✗ FAILED")
            failed.append(cmd)
            if stderr:
                print(f"  Error: {stderr[:200]}")

    print("\n" + "="*60)
    print(f"Results: {len(SUBCOMMANDS) - len(failed)}/{len(SUBCOMMANDS)} passed")

    if failed:
        print(f"\nFailed commands: {', '.join(failed)}")
        print("\nDetails of failures:")
        for cmd in failed:
            _, stdout, stderr = results[cmd]
            print(f"\n{cmd}:")
            if stderr:
                print(f"  stderr: {stderr[:500]}")
            if stdout:
                print(f"  stdout: {stdout[:500]}")
        return 1
    else:
        print("\n✓ All subcommands passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
