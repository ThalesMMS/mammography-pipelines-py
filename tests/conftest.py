"""
Shared pytest fixtures for the mammography-pipelines test suite.

This module provides common fixtures used across unit, integration, and
contract tests. Fixtures include:
- Path management and directory setup
- Sample image and DICOM dataset creation
- Temporary file management
- Model and tensor fixtures
- CSV data fixtures

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Path setup
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Conditional imports using pytest.importorskip for graceful handling
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
pydicom = pytest.importorskip("pydicom")


# ==================== Path Fixtures ====================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return ROOT


@pytest.fixture(scope="session")
def src_root() -> Path:
    """Return the src directory path."""
    return SRC_ROOT


# ==================== Image Fixtures ====================


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple RGB test image (16x16 pixels)."""
    img = Image.new("RGB", (16, 16), color=(120, 30, 60))
    return img


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a simple grayscale test image (16x16 pixels)."""
    img = Image.new("L", (16, 16), color=128)
    return img


def _write_sample_image(path: Path, size: tuple = (16, 16), color: tuple = (120, 30, 60)) -> None:
    """Helper function to write a sample RGB image to disk."""
    img = Image.new("RGB", size, color=color)
    img.save(path)


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create and return path to a temporary sample image file."""
    image_path = tmp_path / "sample.png"
    _write_sample_image(image_path)
    return image_path


# ==================== DICOM Fixtures ====================


def _save_dicom(dataset, path: str) -> None:
    """Helper to save DICOM dataset to file."""
    try:
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(path, write_like_original=False)


@pytest.fixture
def valid_dicom_dataset():
    """Create a valid DICOM dataset for testing."""
    dataset = pydicom.Dataset()

    # Required DICOM fields
    dataset.PatientID = "TEST_PATIENT_001"
    dataset.StudyInstanceUID = "1.2.840.12345.123456789"
    dataset.SeriesInstanceUID = "1.2.840.12345.987654321"
    dataset.SOPInstanceUID = "1.2.840.12345.456789123"
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.file_meta = file_meta

    # Image attributes
    dataset.Manufacturer = "SIEMENS"
    dataset.PixelSpacing = [0.1, 0.1]
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = 128
    dataset.Columns = 128

    # Mammography-specific fields
    dataset.ViewPosition = "CC"
    dataset.ImageLaterality = "L"

    # Create pixel data with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    dataset.PixelData = rng.randint(
        0, 4095, (128, 128), dtype=np.uint16
    ).tobytes()

    return dataset


@pytest.fixture
def mono1_dicom_dataset(valid_dicom_dataset):
    """Create MONOCHROME1 DICOM dataset."""
    dataset = copy.deepcopy(valid_dicom_dataset)
    dataset.PhotometricInterpretation = "MONOCHROME1"
    return dataset


@pytest.fixture
def dicom_with_rescale(valid_dicom_dataset):
    """Create DICOM dataset with RescaleSlope and RescaleIntercept."""
    dataset = copy.deepcopy(valid_dicom_dataset)
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1024.0
    return dataset


@pytest.fixture
def dicom_with_windowing(valid_dicom_dataset):
    """Create DICOM dataset with WindowCenter and WindowWidth."""
    dataset = copy.deepcopy(valid_dicom_dataset)
    dataset.WindowCenter = 2048
    dataset.WindowWidth = 4096
    return dataset


@pytest.fixture
def dicom_file_path(valid_dicom_dataset, tmp_path: Path) -> Path:
    """Create and return path to a temporary DICOM file."""
    dcm_path = tmp_path / "test.dcm"
    _save_dicom(valid_dicom_dataset, str(dcm_path))
    return dcm_path


# ==================== Tensor Fixtures ====================


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample 3-channel tensor (3, 224, 224)."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def sample_batch_tensor() -> torch.Tensor:
    """Create a sample batch tensor (2, 3, 224, 224)."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_embedding_tensor() -> torch.Tensor:
    """Create a sample embedding tensor (2048,)."""
    return torch.randn(2048)


@pytest.fixture
def sample_embeddings_batch() -> torch.Tensor:
    """Create a sample batch of embeddings (4, 2048)."""
    return torch.randn(4, 2048)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Create sample classification labels (4 classes)."""
    return torch.tensor([0, 1, 2, 3], dtype=torch.long)


# ==================== CSV Data Fixtures ====================


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a sample CSV file with image paths and labels."""
    csv_path = tmp_path / "data.csv"

    # Create sample images
    for i in range(3):
        img_path = tmp_path / f"sample_{i}.png"
        _write_sample_image(img_path)

    # Write CSV with image paths
    csv_content = "image_path,professional_label,accession\n"
    for i in range(3):
        img_path = tmp_path / f"sample_{i}.png"
        csv_content += f"{img_path},{i % 4},ACC{i:03d}\n"

    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture
def sample_features_dir(tmp_path: Path) -> Path:
    """Create a sample directory with featureS.txt structure."""
    folder = tmp_path / "case_001"
    folder.mkdir()

    # Create sample image
    img_path = folder / "img_001.png"
    _write_sample_image(img_path)

    # Create featureS.txt
    (folder / "featureS.txt").write_text("img_001\n1\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def sample_classification_csv(tmp_path: Path) -> Path:
    """Create a sample classificacao.csv file."""
    csv_path = tmp_path / "classificacao.csv"
    csv_content = "AccessionNumber,Classification\n"
    csv_content += "ACC001,A\n"
    csv_content += "ACC002,B\n"
    csv_content += "ACC003,C\n"
    csv_content += "ACC004,D\n"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


# ==================== Model Fixtures ====================


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2048, 4)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def efficientnet_model():
    """Create an EfficientNet-B0 model (non-pretrained) for testing."""
    from mammography.models.nets import build_model
    return build_model("efficientnet_b0", num_classes=4, pretrained=False)


@pytest.fixture
def resnet50_model():
    """Create a ResNet50 model (non-pretrained) for testing."""
    from mammography.models.nets import build_model
    return build_model("resnet50", num_classes=4, pretrained=False)


# ==================== Device Fixtures ====================


@pytest.fixture
def cpu_device() -> str:
    """Return CPU device string."""
    return "cpu"


@pytest.fixture
def optimal_device() -> str:
    """Return optimal available device (cpu/cuda/mps)."""
    from mammography.utils.device_detection import get_optimal_device
    return get_optimal_device()


# ==================== Config Fixtures ====================


@pytest.fixture
def sample_train_config_dict() -> Dict[str, Any]:
    """Return a valid TrainConfig dictionary."""
    return {
        "dataset": "mamografias",
        "epochs": 10,
        "batch_size": 32,
        "lr": 1e-3,
        "arch": "efficientnet_b0",
        "device": "cpu",
        "seed": 42,
    }


@pytest.fixture
def sample_extract_config_dict() -> Dict[str, Any]:
    """Return a valid ExtractConfig dictionary."""
    return {
        "dataset": "mamografias",
        "arch": "resnet50",
        "device": "cpu",
        "batch_size": 16,
        "num_workers": 0,
    }


# ==================== Array Fixtures ====================


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Create a sample NumPy array (128, 128) with fixed seed."""
    rng = np.random.RandomState(42)
    return rng.rand(128, 128).astype(np.float32)


@pytest.fixture
def sample_embeddings_array() -> np.ndarray:
    """Create a sample embeddings array (10, 2048) with fixed seed."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 2048).astype(np.float32)


@pytest.fixture
def sample_labels_array() -> np.ndarray:
    """Create a sample labels array."""
    return np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int64)


# ==================== Markers ====================
# Note: Markers are defined in pyproject.toml [tool.pytest.ini_options] section
# to avoid duplicate registration. All markers should be defined there.
