#!/usr/bin/env python3
"""
Minimal Smoke Test for Vision Transformer Implementation
Subtask 5-1: Validation without full dependency installation

This script performs basic validation that can run even in limited environments:
- Import tests
- Code structure validation
- Configuration validation
- Basic instantiation tests (if dependencies available)

Usage:
    python3 smoke_test_vit.py
"""

import sys
import os
from pathlib import Path

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{BLUE}{text}{NC}")
    print(f"{'='*60}\n")

def print_success(text):
    """Print a success message."""
    print(f"{GREEN}✓ {text}{NC}")

def print_failure(text):
    """Print a failure message."""
    print(f"{RED}✗ {text}{NC}")

def print_warning(text):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {text}{NC}")

def test_file_existence():
    """Test that all required files exist."""
    print_header("File Existence Tests")

    files = [
        "src/mammography/models/nets.py",
        "src/mammography/models/embeddings/vit_extractor.py",
        "src/mammography/models/embeddings/__init__.py",
        "tests/unit/test_vit_models.py",
        "tests/unit/test_vit_feature_extraction.py",
        "pyproject.toml",
    ]

    all_exist = True
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            print_success(f"Found: {file_path}")
        else:
            print_failure(f"Missing: {file_path}")
            all_exist = False

    return all_exist

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print_header("Python Syntax Validation")

    files = [
        "src/mammography/models/nets.py",
        "src/mammography/models/embeddings/vit_extractor.py",
        "tests/unit/test_vit_models.py",
        "tests/unit/test_vit_feature_extraction.py",
    ]

    all_valid = True
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print_success(f"Valid syntax: {file_path}")
        except SyntaxError as e:
            print_failure(f"Syntax error in {file_path}: {e}")
            all_valid = False
        except Exception as e:
            print_failure(f"Error reading {file_path}: {e}")
            all_valid = False

    return all_valid

def test_code_structure():
    """Test that required classes and functions exist in the code."""
    print_header("Code Structure Validation")

    all_valid = True

    # Check nets.py for ViT model implementations
    print("\nChecking src/mammography/models/nets.py...")
    with open("src/mammography/models/nets.py", 'r') as f:
        nets_content = f.read()

    # ViT models are implemented inline in build_model() with elif branches
    vit_models = [
        'elif arch == "vit_b_16":',
        'elif arch == "vit_b_32":',
        'elif arch == "vit_l_16":',
        'elif arch == "deit_small":',
        'elif arch == "deit_base":',
    ]

    for model_check in vit_models:
        if model_check in nets_content:
            print_success(f"Found: {model_check}")
        else:
            print_failure(f"Missing: {model_check}")
            all_valid = False

    # Check that build_model references ViT models
    if ('"vit_b_16"' in nets_content or "'vit_b_16'" in nets_content) and \
       ('"deit_base"' in nets_content or "'deit_base'" in nets_content):
        print_success("build_model() includes ViT model references")
    else:
        print_failure("build_model() missing ViT model references")
        all_valid = False

    # Check vit_extractor.py for required class
    print("\nChecking src/mammography/models/embeddings/vit_extractor.py...")
    with open("src/mammography/models/embeddings/vit_extractor.py", 'r') as f:
        extractor_content = f.read()

    required_in_extractor = [
        "class ViTExtractor:",
        "def extract_embedding(",
        "def extract_embeddings_batch(",
        "def create_vit_extractor(",
        "def _validate_config(",
        "def _initialize_model(",
    ]

    for item in required_in_extractor:
        if item in extractor_content:
            print_success(f"Found: {item}")
        else:
            print_failure(f"Missing: {item}")
            all_valid = False

    # Check __init__.py exports
    print("\nChecking src/mammography/models/embeddings/__init__.py...")
    with open("src/mammography/models/embeddings/__init__.py", 'r') as f:
        init_content = f.read()

    if "ViTExtractor" in init_content or "vit_extractor" in init_content:
        print_success("ViTExtractor exported from __init__.py")
    else:
        print_failure("ViTExtractor not exported from __init__.py")
        all_valid = False

    return all_valid

def test_imports():
    """Test that modules can be imported (if dependencies available)."""
    print_header("Import Tests")

    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))

    # Test basic imports
    tests = []

    # Try importing pathlib, os, etc. (always available)
    try:
        import pathlib
        print_success("Standard library imports work")
        tests.append(True)
    except Exception as e:
        print_failure(f"Standard library import failed: {e}")
        tests.append(False)

    # Try importing torch (may not be available)
    try:
        import torch
        print_success("PyTorch available")
        torch_available = True
        tests.append(True)
    except ImportError:
        print_warning("PyTorch not available (expected in limited environments)")
        torch_available = False

    # Try importing torchvision (may not be available)
    try:
        import torchvision
        print_success("torchvision available")
        tests.append(True)
    except ImportError:
        print_warning("torchvision not available (expected in limited environments)")

    # Try importing timm (may not be available)
    try:
        import timm
        print_success("timm available")
        tests.append(True)
    except ImportError:
        print_warning("timm not available (expected in limited environments)")

    # Try importing our modules (only if torch available)
    if torch_available:
        try:
            from mammography.models import nets
            print_success("mammography.models.nets imports successfully")
            tests.append(True)
        except Exception as e:
            print_failure(f"Failed to import mammography.models.nets: {e}")
            tests.append(False)

        try:
            from mammography.models.embeddings import vit_extractor
            print_success("mammography.models.embeddings.vit_extractor imports successfully")
            tests.append(True)
        except Exception as e:
            print_failure(f"Failed to import vit_extractor: {e}")
            tests.append(False)
    else:
        print_warning("Skipping mammography module imports (PyTorch not available)")

    return all(tests) if tests else True

def test_model_registry():
    """Test that ViT models are properly registered in build_model."""
    print_header("Model Registry Validation")

    with open("src/mammography/models/nets.py", 'r') as f:
        content = f.read()

    # Extract the architectures dictionary or switch cases
    vit_architectures = [
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "deit_small",
        "deit_base",
    ]

    all_found = True
    for arch in vit_architectures:
        # Check if architecture name appears in quotes (indicating it's registered)
        if f'"{arch}"' in content or f"'{arch}'" in content:
            print_success(f"Architecture registered: {arch}")
        else:
            print_failure(f"Architecture not registered: {arch}")
            all_found = False

    return all_found

def test_embedding_dimensions():
    """Validate that expected embedding dimensions are documented."""
    print_header("Embedding Dimension Validation")

    with open("src/mammography/models/embeddings/vit_extractor.py", 'r') as f:
        content = f.read()

    expected_dims = {
        "vit_b_16": "768",
        "vit_b_32": "768",
        "vit_l_16": "1024",
        "deit_small": "384",
        "deit_base": "768",
    }

    all_valid = True
    for model, dim in expected_dims.items():
        # Check if dimension is mentioned in comments or code
        if dim in content:
            print_success(f"{model}: {dim} dimensions (found in code)")
        else:
            print_warning(f"{model}: {dim} dimensions (not explicitly found, may be inferred)")
            # Don't mark as failure since dimensions may be inferred from model

    # Check that get_model_info is implemented (provides embedding dimension)
    if "def get_model_info(" in content:
        print_success("get_model_info() method implemented")
    else:
        print_failure("get_model_info() method missing")
        all_valid = False

    # Check that EXPECTED_EMBEDDING_DIM is defined
    if "EXPECTED_EMBEDDING_DIM" in content:
        print_success("EXPECTED_EMBEDDING_DIM constant defined")
    else:
        print_failure("EXPECTED_EMBEDDING_DIM constant missing")
        all_valid = False

    return all_valid

def main():
    """Run all smoke tests."""
    print(f"\n{BLUE}{'='*60}")
    print(f"Vision Transformer Implementation - Smoke Test")
    print(f"Subtask 5-1: Full Test Suite Validation")
    print(f"{'='*60}{NC}\n")

    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}\n")

    tests = {
        "File Existence": test_file_existence,
        "Python Syntax": test_python_syntax,
        "Code Structure": test_code_structure,
        "Model Registry": test_model_registry,
        "Embedding Dimensions": test_embedding_dimensions,
        "Imports": test_imports,
    }

    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_failure(f"Test '{test_name}' raised exception: {e}")
            results[test_name] = False

    # Print summary
    print_header("Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_failure(f"{test_name}: FAILED")

    print(f"\n{passed}/{total} test categories passed\n")

    if passed == total:
        print(f"{GREEN}{'='*60}")
        print(f"✓ ALL SMOKE TESTS PASSED")
        print(f"{'='*60}{NC}\n")
        print("The Vision Transformer implementation is structurally sound.")
        print("Run full test suite in Python 3.11+ environment to verify functionality.\n")
        print("Next steps:")
        print("  1. Ensure Python 3.11+ is available")
        print("  2. Install dependencies: pip install -e .")
        print("  3. Run: ./run_full_test_suite.sh --quick")
        return 0
    else:
        print(f"{RED}{'='*60}")
        print(f"✗ SOME SMOKE TESTS FAILED")
        print(f"{'='*60}{NC}\n")
        print("Please review the failures above and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
