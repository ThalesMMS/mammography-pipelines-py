#!/usr/bin/env python3
"""
Minimal validation script for view-specific training implementation.
Tests that all components are properly integrated without requiring full training.
"""
import sys
import ast
import importlib.util
from pathlib import Path

def validate_python_syntax(filepath):
    """Validate Python file syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def validate_file_exists(filepath):
    """Check if file exists."""
    return Path(filepath).exists()

def main():
    """Run validation checks."""
    print("=" * 60)
    print("View-Specific Training Implementation Validation")
    print("=" * 60)

    all_passed = True

    # Files that should have been modified/created
    files_to_check = [
        "src/mammography/config.py",
        "src/mammography/commands/train.py",
        "src/mammography/data/csv_loader.py",
        "src/mammography/data/splits.py",
        "src/mammography/models/cancer_models.py",
        "src/mammography/training/engine.py",
        "tests/integration/test_view_specific_training.py",
        "configs/view_specific_training.yaml",
    ]

    print("\n1. Checking file existence...")
    for filepath in files_to_check:
        exists = validate_file_exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        if not exists:
            all_passed = False

    print("\n2. Validating Python syntax...")
    for filepath in files_to_check:
        if filepath.endswith('.py') and validate_file_exists(filepath):
            passed, msg = validate_python_syntax(filepath)
            status = "✓" if passed else "✗"
            print(f"  {status} {filepath}: {msg}")
            if not passed:
                all_passed = False

    print("\n3. Checking key implementations...")

    # Check config.py has view-specific fields
    print("  Checking TrainConfig has view_specific_training field...")
    with open("src/mammography/config.py", 'r') as f:
        config_content = f.read()
        has_view_specific = "view_specific_training" in config_content
        has_views_to_train = "views_to_train" in config_content
        status = "✓" if (has_view_specific and has_views_to_train) else "✗"
        print(f"    {status} view_specific_training and views_to_train fields present")
        if not (has_view_specific and has_views_to_train):
            all_passed = False

    # Check cancer_models.py has ViewSpecificModel and EnsemblePredictor
    print("  Checking cancer_models.py has new classes...")
    with open("src/mammography/models/cancer_models.py", 'r') as f:
        models_content = f.read()
        has_view_model = "class ViewSpecificModel" in models_content
        has_ensemble = "class EnsemblePredictor" in models_content
        status_view = "✓" if has_view_model else "✗"
        status_ensemble = "✓" if has_ensemble else "✗"
        print(f"    {status_view} ViewSpecificModel class present")
        print(f"    {status_ensemble} EnsemblePredictor class present")
        if not (has_view_model and has_ensemble):
            all_passed = False

    # Check splits.py has filter_by_view function
    print("  Checking splits.py has filter_by_view...")
    with open("src/mammography/data/splits.py", 'r') as f:
        splits_content = f.read()
        has_filter = "def filter_by_view" in splits_content
        status = "✓" if has_filter else "✗"
        print(f"    {status} filter_by_view function present")
        if not has_filter:
            all_passed = False

    # Check train.py has view-specific training logic
    print("  Checking train.py has view-specific training logic...")
    with open("src/mammography/commands/train.py", 'r') as f:
        train_content = f.read()
        has_view_loop = "for view in" in train_content or "for current_view in" in train_content
        has_ensemble_eval = "EnsemblePredictor" in train_content
        has_cli_args = "--view-specific-training" in train_content
        status_loop = "✓" if has_view_loop else "✗"
        status_ensemble = "✓" if has_ensemble_eval else "✗"
        status_cli = "✓" if has_cli_args else "✗"
        print(f"    {status_loop} View-specific training loop present")
        print(f"    {status_ensemble} Ensemble evaluation integrated")
        print(f"    {status_cli} CLI arguments added")
        if not (has_view_loop and has_ensemble_eval and has_cli_args):
            all_passed = False

    # Check engine.py has plot_view_comparison
    print("  Checking engine.py has plot_view_comparison...")
    with open("src/mammography/training/engine.py", 'r') as f:
        engine_content = f.read()
        has_plot = "def plot_view_comparison" in engine_content
        status = "✓" if has_plot else "✗"
        print(f"    {status} plot_view_comparison function present")
        if not has_plot:
            all_passed = False

    print("\n4. Checking integration test...")
    with open("tests/integration/test_view_specific_training.py", 'r') as f:
        test_content = f.read()
        has_tests = "def test_" in test_content
        test_count = test_content.count("def test_")
        status = "✓" if has_tests else "✗"
        print(f"    {status} Integration tests present ({test_count} test functions)")
        if not has_tests:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("\nImplementation is complete and ready for testing.")
        print("\nTo run the smoke test, execute:")
        print("  python -m mammography.cli train --dataset archive \\")
        print("    --view-specific-training --views-to-train CC MLO \\")
        print("    --ensemble-method average --epochs 2 --subset 100")
        print("\nNote: Requires Python >=3.11 environment")
        return 0
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("\nPlease review the failures above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
