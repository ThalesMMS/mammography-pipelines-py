#!/usr/bin/env python3
"""Verify that datasets are split at the patient level to prevent leakage."""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

RESEARCH_DISCLAIMER = (
    """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
)


def extract_patient_id_from_path(file_path: str) -> str:
    """Extract a patient identifier from a file path using heuristics."""
    patterns = [
        rr"patient_(\d+)",
        rr"p(\d+)",
        rr"(\d{6,8})",
        rr"([A-Z]{2,4}\d{4,6})",
    ]

    for pattern in patterns:
        match = re.search(pattern, file_path, re.IGNORECASE)
        if match:
            return match.group(1)

    for part in Path(file_path).parts:
        lower = part.lower()
        if "patient" in lower or lower.startswith("p"):
            return part

    return "unknown"


def check_patient_splitting_in_code(file_path: Path) -> Tuple[bool, List[str]]:
    """Scan Python files for keywords related to patient-level splitting."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return False, []

    patterns = [
        r"patient_level",
        r"patient-level",
        r"split.*patient",
        r"groupby.*patient",
        r"patient_id",
        r"stratify.*patient",
        r"train_test_split.*patient",
        r"by.*patient",
        r"per.*patient",
    ]

    found_lines: List[str] = []
    for index, line in enumerate(content.splitlines(), start=1):
        lower_line = line.lower()
        if any(re.search(pattern, lower_line) for pattern in patterns):
            found_lines.append(f"Line {index}: {line.strip()}")

    return bool(found_lines), found_lines


def check_data_leakage_prevention(file_path: Path) -> Tuple[bool, List[str]]:
    """Search for leakage-prevention keywords inside a Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return False, []

    patterns = [
        r"data.*leakage",
        r"leakage.*prevention",
        r"prevent.*leakage",
        r"avoid.*leakage",
        r"stratify.*patient",
        r"patient.*split",
        r"patient.*level",
    ]

    found_lines: List[str] = []
    for index, line in enumerate(content.splitlines(), start=1):
        lower_line = line.lower()
        if any(re.search(pattern, lower_line) for pattern in patterns):
            found_lines.append(f"Line {index}: {line.strip()}")

    return bool(found_lines), found_lines


def check_archive_structure() -> Tuple[bool, Dict[str, List[str]]]:
    """Ensure ``archive/`` stores one directory per patient."""
    archive_dir = Path("archive")
    if not archive_dir.exists():
        return False, {}

    patient_files: Dict[str, List[str]] = {}

    for item in archive_dir.iterdir():
        if item.is_dir():
            patient_files[item.name] = [str(f) for f in item.iterdir() if f.is_file()]

    loose_files = [str(path) for path in archive_dir.iterdir() if path.is_file()]
    if loose_files:
        patient_files["_LOOSE_FILES"] = loose_files

    structure_ok = bool(patient_files) and "_LOOSE_FILES" not in patient_files
    return structure_ok, patient_files


def check_configuration_files() -> Tuple[bool, List[str]]:
    """Check configuration files for patient-level split settings."""
    config_files = [
        str(path)
        for pattern in ("configs/*.yaml", "configs/*.yml", "configs/*.json")
        for path in Path(".").glob(pattern)
    ]

    configured = False
    for config_path in config_files:
        try:
            content = Path(config_path).read_text(encoding="utf-8")
        except Exception:
            continue
        lower_content = content.lower()
        if "patient" in lower_content and "split" in lower_content:
            configured = True
            break

    return configured, config_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify patient-level data splitting and leakage prevention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RESEARCH_DISCLAIMER,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/", "tests/"],
        help="Directories or files to analyse for patient-split logic",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if issues are detected",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matching lines for diagnostic purposes",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Patient-Level Splitting Check")
    print("=" * 80)
    print(RESEARCH_DISCLAIMER.strip())
    print("=" * 80)

    patient_keywords: Dict[str, List[str]] = {}
    leakage_keywords: Dict[str, List[str]] = {}

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".py":
            has_patient, patient_lines = check_patient_splitting_in_code(path)
            has_leakage, leakage_lines = check_data_leakage_prevention(path)
            if has_patient:
                patient_keywords[str(path)] = patient_lines
            if has_leakage:
                leakage_keywords[str(path)] = leakage_lines
        elif path.is_dir():
            for file_path in path.rglob("*.py"):
                has_patient, patient_lines = check_patient_splitting_in_code(file_path)
                has_leakage, leakage_lines = check_data_leakage_prevention(file_path)
                if has_patient:
                    patient_keywords[str(file_path)] = patient_lines
                if has_leakage:
                    leakage_keywords[str(file_path)] = leakage_lines
        else:
            print(f"⚠️ Path not found or unsupported: {path}")

    structure_ok, patient_files = check_archive_structure()
    configured, config_files = check_configuration_files()

    print("
📁 ARCHIVE STRUCTURE")
    print(f"Patient folders only: {'✅' if structure_ok else '❌'}")
    if args.verbose and patient_files:
        for patient_id, files in patient_files.items():
            print(f"  {patient_id}: {len(files)} files")

    print("
⚙️ CONFIGURATION FILES")
    print(f"Patient-level split configured: {'✅' if configured else '❌'}")
    if args.verbose and config_files:
        for config_path in config_files:
            print(f"  - {config_path}")

    print("
🧪 CODE REVIEW")
    print(f"Files mentioning patient-level splits: {len(patient_keywords)}")
    print(f"Files referencing leakage prevention: {len(leakage_keywords)}")

    if args.verbose:
        for header, data in (
            ("Patient split references", patient_keywords),
            ("Leakage prevention references", leakage_keywords),
        ):
            if not data:
                continue
            print(f"
{header}:")
            for file_path, lines in data.items():
                print(f"  {file_path}")
                for line in lines:
                    print(f"    {line}")

    issues_found = not structure_ok or not configured or not patient_keywords

    if issues_found:
        print("
❌ Patient-level safeguards missing or incomplete.")
        if args.strict:
            return 1
    else:
        print("
✅ Patient-level safeguards detected. Continue monitoring for leakage.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
