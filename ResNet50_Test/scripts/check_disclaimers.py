#!/usr/bin/env python3
"""Check whether project files contain the mandatory research disclaimer."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Mandatory disclaimer text used in CLI epilogues and reports
RESEARCH_DISCLAIMER = (
    """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
)


def check_file_for_disclaimer(file_path: Path) -> Tuple[bool, List[str]]:
    """Return whether the given file contains a research disclaimer."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - we echo the error
        print(f"Error reading {file_path}: {exc}")
        return False, []

    keywords = [
        "research purposes only",
        "educational research",
        "not for clinical use",
        "for research use",
        "disclaimer",
    ]

    found_lines: List[str] = []
    for index, line in enumerate(content.splitlines(), start=1):
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in keywords):
            found_lines.append(f"Line {index}: {line.strip()}")

    return bool(found_lines), found_lines


def check_directory_for_disclaimers(
    directory: Path, extensions: List[str]
) -> Dict[str, List[str]]:
    """Check all files under *directory* for the disclaimer."""
    results: Dict[str, List[str]] = {}

    for extension in extensions:
        for file_path in directory.rglob(f"**/*{extension}"):
            if not file_path.is_file():
                continue
            has_disclaimer, lines = check_file_for_disclaimer(file_path)
            if not has_disclaimer:
                results[str(file_path)] = []
            elif lines:
                results[str(file_path)] = lines

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether files include the mandatory research disclaimer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RESEARCH_DISCLAIMER,
    )

    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/", "scripts/", "tests/", "docs/"],
        help="Files or directories to scan",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".py", ".md", ".yaml", ".yml", ".json"],
        help="File extensions to include in the scan",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any file is missing the disclaimer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show lines where the disclaimer keywords were detected",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Research Disclaimer Verification")
    print("=" * 80)
    print(RESEARCH_DISCLAIMER.strip())
    print("=" * 80)

    all_results: Dict[str, List[str]] = {}
    files_without_disclaimer = 0

    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file():
            has_disclaimer, lines = check_file_for_disclaimer(path)
            if not has_disclaimer:
                files_without_disclaimer += 1
                all_results[str(path)] = []
            elif lines:
                all_results[str(path)] = lines
        elif path.is_dir():
            results = check_directory_for_disclaimers(path, args.extensions)
            all_results.update(results)
            files_without_disclaimer += sum(1 for lines in results.values() if not lines)
        else:
            print(f"⚠️ Path not found: {path}")

    missing = {file: lines for file, lines in all_results.items() if not lines}

    print("
📊 Verification Report")
    if missing:
        print("Files missing the disclaimer:")
        for file_path in sorted(missing):
            print(f"  - {file_path}")
    else:
        print("All scanned files contain disclaimer keywords.")

    if args.verbose:
        print("
Keyword occurrences:")
        for file_path, lines in sorted(all_results.items()):
            if lines:
                print(f"
{file_path}")
                for line in lines:
                    print(f"  {line}")

    if args.strict and missing:
        print("
❌ Strict mode: missing disclaimers detected.")
        return 1

    if missing:
        print("
⚠️ Completed with missing disclaimers.")
        return 1

    print("
✅ Completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
