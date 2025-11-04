#!/usr/bin/env python3
"""Project status check for the ResNet50_Test research repository."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml

RESEARCH_DISCLAIMER = (
    """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""
)


def print_header() -> None:
    """Display the script header."""
    print("=" * 80)
    print("ResNet50_Test – Project Status Check")
    print("=" * 80)
    print(RESEARCH_DISCLAIMER.strip())
    print("=" * 80)


def check_python_version() -> bool:
    """Verify that Python 3.11 or newer is running."""
    print("
🐍 Checking Python version…")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True

    print(
        f"❌ Python {version.major}.{version.minor}.{version.micro} – Python 3.11+ required"
    )
    return False


def check_dependencies() -> Dict[str, bool]:
    """Attempt to import the main project dependencies."""
    print("
📦 Checking dependencies…")
    dependencies = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "sklearn": "Scikit-learn",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "pydicom": "PyDICOM",
        "PIL": "Pillow",
        "mlflow": "MLflow",
        "pytest": "Pytest",
        "black": "Black",
        "ruff": "Ruff",
        "mypy": "MyPy",
    }

    results: Dict[str, bool] = {}
    for module, name in dependencies.items():
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"❌ {name} (missing)")
            results[module] = False
        else:
            print(f"✅ {name}")
            results[module] = True
    return results


def check_gpu() -> bool:
    """Report GPU availability (CUDA or Apple Silicon MPS)."""
    print("
🖥️ Checking GPU availability…")
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed")
        return False

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA available: {gpu_name}")
        print(f"   Devices: {gpu_count}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        return True

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) available")
        print("   Device: Apple Silicon integrated GPU")
        return True

    print("⚠️ GPU not available – falling back to CPU")
    return False


def check_project_structure() -> Dict[str, bool]:
    """Ensure required directories and files are present."""
    print("
📁 Checking project structure…")

    required_dirs = [
        "archive",
        "data",
        "src",
        "tests",
        "configs",
        "results",
        "reports",
        "docs",
        ".specify",
    ]
    required_files = [
        "README.md",
        "QUICKSTART.md",
        "requirements.txt",
        ".gitignore",
        "configs/base.yaml",
    ]

    results: Dict[str, bool] = {}

    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✅ {directory}/")
            results[directory] = True
        else:
            print(f"❌ {directory}/ (missing)")
            results[directory] = False

    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name}")
            results[file_name] = True
        else:
            print(f"❌ {file_name} (missing)")
            results[file_name] = False

    return results


def check_data_structure() -> bool:
    """Validate the expected archive layout for patient-level data."""
    print("
📊 Checking data layout…")
    archive_dir = Path("archive")
    if not archive_dir.exists():
        print("❌ 'archive' directory not found")
        print("   Place DICOM data under ./archive/")
        return False

    patient_dirs = [path for path in archive_dir.iterdir() if path.is_dir()]
    if not patient_dirs:
        print("❌ No patient directories detected inside ./archive/")
        print("   Expected structure: archive/patient_001/image_001.dcm")
        return False

    dicom_files = list(archive_dir.rglob("*.dcm")) + list(archive_dir.rglob("*.dicom"))
    if not dicom_files:
        print("❌ No DICOM files found")
        print("   Add .dcm or .dicom files inside patient folders")
        return False

    print(f"✅ {len(patient_dirs)} patient directories")
    print(f"✅ {len(dicom_files)} DICOM files detected")
    return True


def check_configurations() -> bool:
    """Inspect configuration files for required sections."""
    print("
⚙️ Checking configuration files…")
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        print("❌ configs/base.yaml missing")
        return False

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"❌ Unable to parse configs/base.yaml: {exc}")
        return False

    required_sections = ["project", "data", "hardware", "reproducibility"]
    missing = [section for section in required_sections if section not in config]

    if missing:
        print(f"❌ Missing sections in base.yaml: {', '.join(missing)}")
        return False

    print("✅ Configuration file parsed successfully")
    return True


def check_governance() -> bool:
    """Verify the presence of governance artefacts in .specify/memory."""
    print("
🏛️ Checking governance artefacts…")
    memory_dir = Path(".specify/memory")
    required_files = [
        "constitution.md",
        "clarifications.md",
        "risks.md",
        "go-nogo-summary.md",
    ]

    missing = [name for name in required_files if not (memory_dir / name).exists()]
    if missing:
        print(f"❌ Missing governance files: {', '.join(missing)}")
        return False

    print("✅ Governance artefacts found")
    return True


def check_tests() -> bool:
    """Run ``pytest --collect-only`` as a lightweight sanity check."""
    print("
🧪 Collecting tests…")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("❌ pytest not available")
        return False

    if result.returncode == 0:
        print("✅ Tests collected successfully")
        return True

    print("❌ pytest collection failed")
    if result.stderr:
        print(result.stderr.strip())
    return False


def main() -> int:
    print_header()

    python_ok = check_python_version()
    deps_ok = all(check_dependencies().values())
    gpu_ok = check_gpu()
    structure_ok = all(check_project_structure().values())
    data_ok = check_data_structure()
    config_ok = check_configurations()
    governance_ok = check_governance()
    tests_ok = check_tests()

    print("
📌 SUMMARY")
    checks = {
        "Python version": python_ok,
        "Dependencies": deps_ok,
        "GPU detection": gpu_ok,
        "Project structure": structure_ok,
        "Data layout": data_ok,
        "Configuration": config_ok,
        "Governance": governance_ok,
        "Test collection": tests_ok,
    }
    for label, ok in checks.items():
        print(f"{label:>20}: {'✅' if ok else '❌'}")

    if all(checks.values()):
        print("
✅ All checks passed. Proceed with experimentation.")
        return 0

    print("
⚠️ Some checks failed. Review the output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
