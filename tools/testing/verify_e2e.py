#!/usr/bin/env python3
"""
End-to-end verification script for Enhanced Streamlit Web UI.

This script verifies that all new features are properly implemented and can be imported
without errors. It checks:
1. All new components can be imported
2. All modified pages have the expected features
3. File structure is correct
4. Integration points are present
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(msg: str) -> None:
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{msg}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"{RED}❌ {msg}{RESET}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"{YELLOW}⚠️  {msg}{RESET}")


def print_info(msg: str) -> None:
    """Print an info message."""
    print(f"   {msg}")


def verify_component_imports() -> Tuple[bool, List[str]]:
    """Verify that all new components can be imported."""
    print_header("Verifying Component Imports")

    errors = []

    try:
        from mammography.apps.web_ui.components import DatasetViewer
        print_success("DatasetViewer component imports successfully")
    except Exception as e:
        errors.append(f"DatasetViewer import failed: {e}")
        print_error(f"DatasetViewer import failed: {e}")

    try:
        from mammography.apps.web_ui.components import MetricsMonitor
        print_success("MetricsMonitor component imports successfully")
    except Exception as e:
        errors.append(f"MetricsMonitor import failed: {e}")
        print_error(f"MetricsMonitor import failed: {e}")

    try:
        from mammography.apps.web_ui.components import (
            plot_confusion_matrix,
            plot_roc_curves,
            render_confusion_matrix,
            render_roc_curves,
            render_results_summary,
        )
        print_success("Results visualizer functions import successfully")
    except Exception as e:
        errors.append(f"Results visualizer import failed: {e}")
        print_error(f"Results visualizer import failed: {e}")

    try:
        from mammography.apps.web_ui.components import ReportExporter
        print_success("ReportExporter component imports successfully")
    except Exception as e:
        errors.append(f"ReportExporter import failed: {e}")
        print_error(f"ReportExporter import failed: {e}")

    return len(errors) == 0, errors


def verify_page_structure() -> Tuple[bool, List[str]]:
    """Verify that all pages exist and have expected structure."""
    print_header("Verifying Page Structure")

    errors = []
    base_path = Path("src/mammography/apps/web_ui/pages")

    expected_pages = {
        "0_📁_Dataset_Browser.py": [
            "DatasetViewer",
            "load_dataset_dataframe",
            "st.set_page_config",
        ],
        "3_📈_Experiments.py": [
            "_render_confusion_matrix",
            "_render_roc_curve",
            "ReportExporter",
        ],
        "4_⚙️_Training.py": [
            "training_metrics",
            "MLflow",
            "poll_metrics",
        ],
    }

    for page_name, expected_content in expected_pages.items():
        page_path = base_path / page_name

        if not page_path.exists():
            errors.append(f"Page {page_name} not found")
            print_error(f"Page {page_name} not found")
            continue

        print_info(f"Checking {page_name}...")
        content = page_path.read_text(encoding="utf-8")

        missing = []
        for item in expected_content:
            if item not in content:
                missing.append(item)

        if missing:
            errors.append(f"Page {page_name} missing: {', '.join(missing)}")
            print_error(f"Missing in {page_name}: {', '.join(missing)}")
        else:
            print_success(f"{page_name} has all expected features")

    return len(errors) == 0, errors


def verify_component_files() -> Tuple[bool, List[str]]:
    """Verify that all component files exist and have expected exports."""
    print_header("Verifying Component Files")

    errors = []
    components_path = Path("src/mammography/apps/web_ui/components")

    expected_files = {
        "__init__.py": [
            "DatasetViewer",
            "MetricsMonitor",
            "ReportExporter",
            "plot_confusion_matrix",
            "plot_roc_curves",
        ],
        "dataset_viewer.py": [
            "class DatasetViewer",
            "render_grid",
            "render_table",
        ],
        "metrics_monitor.py": [
            "class MetricsMonitor",
            "render_current_metrics",
            "render_metrics_history",
        ],
        "results_visualizer.py": [
            "def plot_confusion_matrix",
            "def plot_roc_curves",
            "render_confusion_matrix",
        ],
        "report_exporter.py": [
            "class ReportExporter",
            "export_from_mlflow",
            "create_zip_archive",
        ],
    }

    for file_name, expected_content in expected_files.items():
        file_path = components_path / file_name

        if not file_path.exists():
            errors.append(f"Component file {file_name} not found")
            print_error(f"Component file {file_name} not found")
            continue

        print_info(f"Checking {file_name}...")
        content = file_path.read_text(encoding="utf-8")

        missing = []
        for item in expected_content:
            if item not in content:
                missing.append(item)

        if missing:
            errors.append(f"File {file_name} missing: {', '.join(missing)}")
            print_error(f"Missing in {file_name}: {', '.join(missing)}")
        else:
            print_success(f"{file_name} has all expected content")

    return len(errors) == 0, errors


def verify_main_page_updated() -> Tuple[bool, List[str]]:
    """Verify that the main page documentation includes Dataset Browser."""
    print_header("Verifying Main Page Updates")

    errors = []
    main_page = Path("src/mammography/apps/web_ui/streamlit_app.py")

    if not main_page.exists():
        errors.append("Main page (streamlit_app.py) not found")
        print_error("Main page (streamlit_app.py) not found")
        return False, errors

    content = main_page.read_text(encoding="utf-8")

    expected_mentions = [
        "📁 Dataset Browser",
        "Explore mammography datasets",
    ]

    missing = []
    for item in expected_mentions:
        if item not in content:
            missing.append(item)

    if missing:
        errors.append(f"Main page missing: {', '.join(missing)}")
        print_error(f"Main page missing: {', '.join(missing)}")
    else:
        print_success("Main page includes Dataset Browser documentation")

    return len(errors) == 0, errors


def verify_integration_points() -> Tuple[bool, List[str]]:
    """Verify that all integration points are correctly set up."""
    print_header("Verifying Integration Points")

    errors = []

    # Check Dataset Browser integration
    dataset_browser = Path("src/mammography/apps/web_ui/pages/0_📁_Dataset_Browser.py")
    if dataset_browser.exists():
        content = dataset_browser.read_text(encoding="utf-8")
        if "DatasetViewer" in content and "load_dataset_dataframe" in content:
            print_success("Dataset Browser integrates with DatasetViewer component")
        else:
            errors.append("Dataset Browser missing component integration")
            print_error("Dataset Browser missing component integration")

    # Check Training page integration
    training_page = Path("src/mammography/apps/web_ui/pages/4_⚙️_Training.py")
    if training_page.exists():
        content = training_page.read_text(encoding="utf-8")
        if "training_metrics" in content and "MLflow" in content:
            print_success("Training page integrates with MLflow metrics polling")
        else:
            errors.append("Training page missing MLflow integration")
            print_error("Training page missing MLflow integration")

    # Check Experiments page integration
    experiments_page = Path("src/mammography/apps/web_ui/pages/3_📈_Experiments.py")
    if experiments_page.exists():
        content = experiments_page.read_text(encoding="utf-8")
        checks = {
            "confusion matrix": "_render_confusion_matrix" in content,
            "ROC curves": "_render_roc_curve" in content,
            "export functionality": "ReportExporter" in content,
        }

        for feature, present in checks.items():
            if present:
                print_success(f"Experiments page includes {feature}")
            else:
                errors.append(f"Experiments page missing {feature}")
                print_error(f"Experiments page missing {feature}")

    return len(errors) == 0, errors


def print_summary(all_results: List[Tuple[str, bool, List[str]]]) -> bool:
    """Print verification summary."""
    print_header("Verification Summary")

    total_checks = len(all_results)
    passed_checks = sum(1 for _, success, _ in all_results if success)

    for check_name, success, errors in all_results:
        if success:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            for error in errors:
                print_info(f"  - {error}")

    print()
    print(f"Total: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print_success("All verification checks passed! ✨")
        return True
    else:
        print_error(f"{total_checks - passed_checks} check(s) failed")
        return False


def main() -> int:
    """Run all verification checks."""
    print_header("Enhanced Streamlit Web UI - End-to-End Verification")

    # Run all verification checks
    results = [
        ("Component Imports", *verify_component_imports()),
        ("Component Files", *verify_component_files()),
        ("Page Structure", *verify_page_structure()),
        ("Main Page Updates", *verify_main_page_updated()),
        ("Integration Points", *verify_integration_points()),
    ]

    # Print summary
    all_passed = print_summary(results)

    if all_passed:
        print()
        print_info("Next steps for manual verification:")
        print_info("1. Launch the Streamlit app:")
        print_info("   py -m streamlit run src/mammography/apps/web_ui/streamlit_app.py")
        print_info("2. Navigate to Dataset Browser and verify images display")
        print_info("3. Navigate to Training page and verify live metrics section")
        print_info("4. Navigate to Experiments page and verify visualizations")
        print_info("5. Test export functionality")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
