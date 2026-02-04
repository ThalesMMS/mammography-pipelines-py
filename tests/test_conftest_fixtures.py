"""
Simple test to verify conftest.py fixtures are working correctly.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import pytest


def test_project_root_fixture(project_root):
    """Test that project_root fixture returns a valid Path."""
    assert project_root.exists()
    assert project_root.is_dir()


def test_sample_image_fixture(sample_image):
    """Test that sample_image fixture creates a valid PIL Image."""
    assert sample_image.mode == "RGB"
    assert sample_image.size == (16, 16)


def test_sample_image_path_fixture(sample_image_path):
    """Test that sample_image_path fixture creates a valid file."""
    assert sample_image_path.exists()
    assert sample_image_path.suffix == ".png"


def test_valid_dicom_dataset_fixture(valid_dicom_dataset):
    """Test that valid_dicom_dataset fixture creates valid DICOM data."""
    assert valid_dicom_dataset.PatientID == "TEST_PATIENT_001"
    assert valid_dicom_dataset.Rows == 128
    assert valid_dicom_dataset.Columns == 128
    assert valid_dicom_dataset.PhotometricInterpretation == "MONOCHROME2"


def test_mono1_dicom_dataset_fixture(mono1_dicom_dataset):
    """Test that mono1_dicom_dataset fixture creates MONOCHROME1 DICOM."""
    assert mono1_dicom_dataset.PhotometricInterpretation == "MONOCHROME1"


def test_dicom_with_rescale_fixture(dicom_with_rescale):
    """Test that dicom_with_rescale fixture has rescale parameters."""
    assert hasattr(dicom_with_rescale, "RescaleSlope")
    assert hasattr(dicom_with_rescale, "RescaleIntercept")
    assert dicom_with_rescale.RescaleSlope == 2.0
    assert dicom_with_rescale.RescaleIntercept == -1024.0


def test_sample_tensor_fixture(sample_tensor):
    """Test that sample_tensor fixture creates valid tensor."""
    assert sample_tensor.shape == (3, 224, 224)
    assert sample_tensor.dtype in [torch.float32, torch.float64]


def test_sample_batch_tensor_fixture(sample_batch_tensor):
    """Test that sample_batch_tensor fixture creates valid batch."""
    assert sample_batch_tensor.shape == (2, 3, 224, 224)


def test_sample_csv_path_fixture(sample_csv_path):
    """Test that sample_csv_path fixture creates valid CSV."""
    assert sample_csv_path.exists()
    assert sample_csv_path.suffix == ".csv"
    content = sample_csv_path.read_text(encoding="utf-8")
    assert "image_path" in content
    assert "professional_label" in content


def test_sample_features_dir_fixture(sample_features_dir):
    """Test that sample_features_dir fixture creates valid directory."""
    assert sample_features_dir.exists()
    case_dir = sample_features_dir / "case_001"
    assert case_dir.exists()
    assert (case_dir / "featureS.txt").exists()


def test_cpu_device_fixture(cpu_device):
    """Test that cpu_device fixture returns 'cpu'."""
    assert cpu_device == "cpu"


def test_sample_numpy_array_fixture(sample_numpy_array):
    """Test that sample_numpy_array fixture creates valid array."""
    import numpy as np
    assert isinstance(sample_numpy_array, np.ndarray)
    assert sample_numpy_array.shape == (128, 128)
    assert sample_numpy_array.dtype == np.float32


def test_sample_embeddings_array_fixture(sample_embeddings_array):
    """Test that sample_embeddings_array fixture creates valid embeddings."""
    import numpy as np
    assert isinstance(sample_embeddings_array, np.ndarray)
    assert sample_embeddings_array.shape == (10, 2048)
    assert sample_embeddings_array.dtype == np.float32


def test_sample_labels_array_fixture(sample_labels_array):
    """Test that sample_labels_array fixture creates valid labels."""
    import numpy as np
    assert isinstance(sample_labels_array, np.ndarray)
    assert sample_labels_array.shape == (10,)
    assert sample_labels_array.dtype == np.int64


# Import torch after pytest setup to handle importorskip
torch = pytest.importorskip("torch")
