"""
Unit tests for comprehensive utility module coverage.

Tests cover:
- Common utilities (seeding, device resolution, logging, path handling, parsing)
- Normalization utilities (stats computation, validation, z-score normalization)
- NumPy warnings suppression
- Patient data management
- Reproducibility utilities (RNG state management)

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import os
import json
import random
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from mammography.utils.common import (
    seed_everything,
    resolve_device,
    configure_runtime,
    setup_logging,
    increment_path,
    parse_float_list,
    get_reproducibility_info,
)
from mammography.utils.normalization import (
    NormalizationStats,
    compute_normalization_stats,
    validate_normalization,
    z_score_normalize,
)
from mammography.utils.numpy_warnings import (
    resolve_pca_svd_solver,
    suppress_numpy_matmul_warnings,
)
from mammography.utils.patient import (
    Patient,
    create_patient_from_images,
    validate_patient_splits,
)
from mammography.utils.reproducibility import (
    fix_seeds,
    get_random_state,
    restore_random_state,
)


# ============================================================================
# Test Common Utilities
# ============================================================================


class TestSeedEverything:
    """Test seeding functionality for reproducibility."""

    def test_seed_everything_basic(self):
        """Test that seeding sets all RNG states."""
        seed_everything(42)

        # Get initial random values
        py_rand1 = random.random()
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()

        # Re-seed and check we get same values
        seed_everything(42)
        py_rand2 = random.random()
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()

        assert py_rand1 == py_rand2
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2

    def test_seed_everything_different_seeds(self):
        """Test that different seeds produce different results."""
        seed_everything(42)
        val1 = torch.rand(1).item()

        seed_everything(123)
        val2 = torch.rand(1).item()

        assert val1 != val2

    def test_seed_everything_deterministic_mode(self):
        """Test deterministic mode configuration."""
        seed_everything(42, deterministic=True)

        # Check that deterministic settings were applied
        if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False

    def test_seed_everything_non_deterministic_mode(self):
        """Test non-deterministic mode configuration."""
        seed_everything(42, deterministic=False)

        # Check that benchmark mode is enabled for performance
        if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
            assert torch.backends.cudnn.benchmark is True


class TestResolveDevice:
    """Test device resolution functionality."""

    def test_resolve_device_auto(self):
        """Test automatic device selection."""
        device = resolve_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_resolve_device_cpu(self):
        """Test CPU device selection."""
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_resolve_device_cuda(self):
        """Test CUDA device selection (falls back to CPU if unavailable)."""
        device = resolve_device("cuda")
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

    def test_resolve_device_mps(self):
        """Test MPS device selection (falls back to CPU if unavailable)."""
        device = resolve_device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_resolve_device_priority(self):
        """Test device priority: CUDA > MPS > CPU."""
        device = resolve_device("auto")

        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"


class TestConfigureRuntime:
    """Test runtime configuration."""

    def test_configure_runtime_cpu(self):
        """Test CPU runtime configuration."""
        device = torch.device("cpu")
        # Should not raise errors
        configure_runtime(device, deterministic=False, allow_tf32=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_configure_runtime_cuda(self):
        """Test CUDA runtime configuration."""
        device = torch.device("cuda:0")
        configure_runtime(device, deterministic=False, allow_tf32=True)

        # Check TF32 settings
        assert torch.backends.cuda.matmul.allow_tf32 is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_configure_runtime_cuda_deterministic(self):
        """Test CUDA deterministic runtime configuration."""
        device = torch.device("cuda:0")
        configure_runtime(device, deterministic=True, allow_tf32=False)

        # Check deterministic settings
        assert torch.backends.cuda.matmul.allow_tf32 is False


class TestSetupLogging:
    """Test logging configuration."""

    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir, "INFO")
            assert logger is not None
            assert logger.name == "mammography"

    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir, "INFO")
            log_file = os.path.join(tmpdir, "run.log")
            assert os.path.exists(log_file)

    def test_setup_logging_writes_to_file(self):
        """Test that logging writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir, "INFO")
            test_message = "Test log message"
            logger.info(test_message)

            log_file = os.path.join(tmpdir, "run.log")
            with open(log_file, "r") as f:
                log_content = f.read()

            assert test_message in log_content

    def test_setup_logging_respects_level(self):
        """Test that logging respects log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir, "WARNING")

            # Console handler should be at WARNING level
            console_handler = logger.handlers[1]  # Second handler is console
            assert console_handler.level >= 30  # WARNING level


class TestIncrementPath:
    """Test path incrementing functionality."""

    def test_increment_path_nonexistent(self):
        """Test increment_path with non-existent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_dir")
            result = increment_path(path)
            assert result == path

    def test_increment_path_existing(self):
        """Test increment_path with existing path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_dir")
            os.makedirs(path)

            result = increment_path(path)
            assert result == f"{path}_1"

    def test_increment_path_multiple_existing(self):
        """Test increment_path with multiple existing paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_dir")
            os.makedirs(path)
            os.makedirs(f"{path}_1")
            os.makedirs(f"{path}_2")

            result = increment_path(path)
            assert result == f"{path}_3"

    def test_increment_path_strips_trailing_slash(self):
        """Test that increment_path strips trailing slashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_dir") + "/"
            result = increment_path(path)
            assert not result.endswith("/")


class TestParseFloatList:
    """Test float list parsing functionality."""

    def test_parse_float_list_comma_separated(self):
        """Test parsing comma-separated floats."""
        result = parse_float_list("1.0,2.0,3.0")
        assert result == [1.0, 2.0, 3.0]

    def test_parse_float_list_space_separated(self):
        """Test parsing space-separated floats."""
        result = parse_float_list("1.0 2.0 3.0")
        assert result == [1.0, 2.0, 3.0]

    def test_parse_float_list_mixed_separators(self):
        """Test parsing mixed separators."""
        result = parse_float_list("1.0, 2.0, 3.0")
        assert result == [1.0, 2.0, 3.0]

    def test_parse_float_list_with_brackets(self):
        """Test parsing floats with square brackets."""
        result = parse_float_list("[1.0, 2.0, 3.0]")
        assert result == [1.0, 2.0, 3.0]

    def test_parse_float_list_none(self):
        """Test parsing None returns None."""
        result = parse_float_list(None)
        assert result is None

    def test_parse_float_list_empty(self):
        """Test parsing empty string returns None."""
        result = parse_float_list("")
        assert result is None

    def test_parse_float_list_expected_length(self):
        """Test parsing with expected length validation."""
        result = parse_float_list("1.0,2.0,3.0", expected_len=3)
        assert len(result) == 3

    def test_parse_float_list_wrong_length_raises(self):
        """Test that wrong length raises ValueError."""
        with pytest.raises(ValueError, match="deve ter 3 valores"):
            parse_float_list("1.0,2.0", expected_len=3)

    def test_parse_float_list_invalid_values_raises(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match="deve conter apenas numeros"):
            parse_float_list("1.0,invalid,3.0")


class TestGetReproducibilityInfo:
    """Test reproducibility info capture."""

    def test_get_reproducibility_info_structure(self):
        """Test that get_reproducibility_info returns expected structure."""
        info = get_reproducibility_info()

        assert isinstance(info, dict)
        assert "git_commit" in info
        assert "git_dirty" in info
        assert "command_line" in info
        assert "cwd" in info
        assert "timestamp" in info

    def test_get_reproducibility_info_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        info = get_reproducibility_info()

        # Should be able to parse the timestamp
        datetime.fromisoformat(info["timestamp"])

    def test_get_reproducibility_info_cwd(self):
        """Test that current working directory is captured."""
        info = get_reproducibility_info()
        assert info["cwd"] == os.getcwd()


# ============================================================================
# Test Normalization Utilities
# ============================================================================


class TestNormalizationStats:
    """Test NormalizationStats dataclass."""

    def test_normalization_stats_creation(self):
        """Test creating NormalizationStats."""
        stats = NormalizationStats(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            method="imagenet",
        )
        assert len(stats.mean) == 3
        assert len(stats.std) == 3
        assert stats.method == "imagenet"

    def test_normalization_stats_validation_length_mismatch(self):
        """Test that mismatched mean/std lengths raise error."""
        with pytest.raises(ValueError, match="must have same length"):
            NormalizationStats(
                mean=[0.5, 0.5],
                std=[0.5],
            )

    def test_normalization_stats_validation_invalid_length(self):
        """Test that invalid lengths raise error."""
        with pytest.raises(ValueError, match="must have 1 or 3 values"):
            NormalizationStats(
                mean=[0.5, 0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5, 0.5],
            )

    def test_normalization_stats_validation_negative_std(self):
        """Test that negative std values raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            NormalizationStats(
                mean=[0.5],
                std=[-0.5],
            )

    def test_normalization_stats_validation_zero_std(self):
        """Test that zero std values raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            NormalizationStats(
                mean=[0.5],
                std=[0.0],
            )

    def test_normalization_stats_to_dict(self):
        """Test converting to dictionary."""
        stats = NormalizationStats(
            mean=[0.5],
            std=[0.5],
            method="custom",
            sample_size=1000,
        )
        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["mean"] == [0.5]
        assert stats_dict["std"] == [0.5]
        assert stats_dict["method"] == "custom"
        assert stats_dict["sample_size"] == 1000

    def test_normalization_stats_from_dict(self):
        """Test creating from dictionary."""
        stats_dict = {
            "mean": [0.5],
            "std": [0.5],
            "method": "custom",
            "sample_size": 1000,
        }
        stats = NormalizationStats.from_dict(stats_dict)

        assert stats.mean == [0.5]
        assert stats.std == [0.5]
        assert stats.method == "custom"
        assert stats.sample_size == 1000

    def test_normalization_stats_imagenet_defaults(self):
        """Test ImageNet defaults."""
        stats = NormalizationStats.imagenet_defaults()

        assert len(stats.mean) == 3
        assert len(stats.std) == 3
        assert stats.method == "imagenet"
        assert stats.sample_size is None


class TestComputeNormalizationStats:
    """Test normalization statistics computation."""

    class SimpleDataset(Dataset):
        """Simple dataset for testing."""
        def __init__(self, size=100, channels=3, img_size=32):
            self.size = size
            self.channels = channels
            self.img_size = img_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Generate deterministic test images
            torch.manual_seed(idx)
            img = torch.randn(self.channels, self.img_size, self.img_size)
            return img, 0  # Return (image, label) tuple

    def test_compute_normalization_stats_basic(self):
        """Test basic normalization stats computation."""
        dataset = self.SimpleDataset(size=50, channels=3)
        stats = compute_normalization_stats(dataset, num_samples=50, batch_size=10)

        assert len(stats.mean) == 3
        assert len(stats.std) == 3
        assert stats.method == "auto"
        assert stats.sample_size == 50

    def test_compute_normalization_stats_single_channel(self):
        """Test with single-channel images."""
        dataset = self.SimpleDataset(size=50, channels=1)
        stats = compute_normalization_stats(dataset, num_samples=50, batch_size=10)

        assert len(stats.mean) == 1
        assert len(stats.std) == 1

    def test_compute_normalization_stats_limited_samples(self):
        """Test computing stats with limited samples."""
        dataset = self.SimpleDataset(size=100, channels=3)
        stats = compute_normalization_stats(dataset, num_samples=30, batch_size=10)

        assert stats.sample_size == 30

    def test_compute_normalization_stats_from_dataloader(self):
        """Test computing stats from DataLoader."""
        dataset = self.SimpleDataset(size=50, channels=3)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

        stats = compute_normalization_stats(dataloader)

        assert len(stats.mean) == 3
        assert len(stats.std) == 3

    def test_compute_normalization_stats_empty_dataset_raises(self):
        """Test that empty dataset raises ValueError."""
        dataset = self.SimpleDataset(size=0)

        with pytest.raises(ValueError, match="Dataset is empty"):
            compute_normalization_stats(dataset)


class TestValidateNormalization:
    """Test normalization validation."""

    def test_validate_normalization_normalized_data(self):
        """Test validation with properly normalized data."""
        # Create normalized data (mean~0, std~1)
        torch.manual_seed(42)
        data = torch.randn(32, 3, 224, 224)

        result = validate_normalization(
            data,
            expected_mean=[0.0, 0.0, 0.0],
            expected_std=[1.0, 1.0, 1.0],
            tolerance=0.2,
        )

        assert result["is_normalized"] is True
        assert len(result["actual_mean"]) == 3
        assert len(result["actual_std"]) == 3
        assert len(result["warnings"]) == 0

    def test_validate_normalization_unnormalized_data(self):
        """Test validation with unnormalized data."""
        # Create unnormalized data (mean~100, std~50)
        torch.manual_seed(42)
        data = torch.randn(32, 3, 224, 224) * 50 + 100

        result = validate_normalization(
            data,
            expected_mean=[0.0, 0.0, 0.0],
            expected_std=[1.0, 1.0, 1.0],
            tolerance=0.2,
        )

        assert result["is_normalized"] is False
        assert len(result["warnings"]) > 0

    def test_validate_normalization_numpy_input(self):
        """Test validation with numpy array input."""
        np.random.seed(42)
        data = np.random.randn(32, 3, 224, 224)

        result = validate_normalization(
            data,
            expected_mean=[0.0, 0.0, 0.0],
            expected_std=[1.0, 1.0, 1.0],
            tolerance=0.2,
        )

        assert result["is_normalized"] is True

    def test_validate_normalization_3d_input(self):
        """Test validation with 3D input (adds batch dimension)."""
        torch.manual_seed(42)
        data = torch.randn(3, 224, 224)

        result = validate_normalization(
            data,
            expected_mean=[0.0, 0.0, 0.0],
            expected_std=[1.0, 1.0, 1.0],
            tolerance=0.2,
        )

        assert result["is_normalized"] is True

    def test_validate_normalization_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        data = torch.randn(224, 224)  # 2D tensor

        with pytest.raises(ValueError, match="Expected 4D tensor"):
            validate_normalization(data)


class TestZScoreNormalize:
    """Test z-score normalization."""

    def test_z_score_normalize_basic(self):
        """Test basic z-score normalization."""
        torch.manual_seed(42)
        data = torch.randn(100, 3, 32, 32) * 50 + 100

        normalized = z_score_normalize(data)

        # Check normalization worked
        assert abs(normalized.mean().item()) < 0.1
        assert abs(normalized.std().item() - 1.0) < 0.1

    def test_z_score_normalize_with_custom_stats(self):
        """Test normalization with custom mean and std."""
        torch.manual_seed(42)
        data = torch.randn(100, 3, 32, 32) * 50 + 100

        normalized = z_score_normalize(data, mean=100.0, std=50.0)

        # Should normalize using provided stats
        assert abs(normalized.mean().item()) < 0.5

    def test_z_score_normalize_numpy_input(self):
        """Test normalization with numpy input."""
        np.random.seed(42)
        data = np.random.randn(100, 3, 32, 32) * 50 + 100

        normalized = z_score_normalize(data)

        assert isinstance(normalized, np.ndarray)
        assert abs(normalized.mean()) < 0.1

    def test_z_score_normalize_per_channel(self):
        """Test per-channel normalization."""
        torch.manual_seed(42)
        data = torch.randn(100, 3, 32, 32)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalized = z_score_normalize(data, mean=mean, std=std)

        assert normalized.shape == data.shape


# ============================================================================
# Test NumPy Warnings Utilities
# ============================================================================


class TestResolvePcaSvdSolver:
    """Test PCA SVD solver resolution."""

    def test_resolve_pca_svd_solver_auto_full(self):
        """Test auto selection chooses full for small matrices."""
        solver = resolve_pca_svd_solver(
            n_samples=100,
            n_features=50,
            n_components=50,
            requested="auto",
        )
        assert solver == "full"

    def test_resolve_pca_svd_solver_auto_randomized(self):
        """Test auto selection chooses randomized for large matrices."""
        solver = resolve_pca_svd_solver(
            n_samples=2000,
            n_features=2000,
            n_components=100,
            requested="auto",
        )
        assert solver == "randomized"

    def test_resolve_pca_svd_solver_explicit(self):
        """Test explicit solver selection."""
        solver = resolve_pca_svd_solver(
            n_samples=100,
            n_features=50,
            n_components=10,
            requested="full",
        )
        assert solver == "full"

    def test_resolve_pca_svd_solver_none_defaults_to_auto(self):
        """Test None defaults to auto."""
        solver = resolve_pca_svd_solver(
            n_samples=100,
            n_features=50,
            n_components=10,
            requested=None,
        )
        assert solver in ["full", "randomized"]


class TestSuppressNumpyMatmulWarnings:
    """Test NumPy warning suppression."""

    def test_suppress_numpy_matmul_warnings(self):
        """Test that matmul warnings are suppressed."""
        with suppress_numpy_matmul_warnings():
            # This context should suppress warnings
            # In practice, we can't easily trigger these warnings in tests
            # so we just verify the context manager doesn't raise
            pass

    def test_suppress_numpy_matmul_warnings_allows_other_warnings(self):
        """Test that other warnings are still raised."""
        with suppress_numpy_matmul_warnings():
            # Other warnings should still be raised
            with pytest.warns(UserWarning):
                warnings.warn("Test warning", UserWarning)


# ============================================================================
# Test Patient Utilities
# ============================================================================


class TestPatient:
    """Test Patient class."""

    def test_patient_creation(self):
        """Test creating a Patient instance."""
        patient = Patient(
            patient_id="PAT001",
            image_count=4,
            projections=["CC", "MLO"],
            laterality=["L", "R"],
            split_assignment="train",
        )

        assert patient.patient_id == "PAT001"
        assert patient.image_count == 4
        assert set(patient.projections) == {"CC", "MLO"}
        assert set(patient.laterality) == {"L", "R"}
        assert patient.split_assignment == "train"

    def test_patient_validation_invalid_id(self):
        """Test that invalid patient ID raises error."""
        with pytest.raises((ValueError, TypeError)):
            Patient(patient_id="")

    def test_patient_validation_invalid_split(self):
        """Test that invalid split raises error."""
        with pytest.raises(ValueError):
            Patient(patient_id="PAT001", split_assignment="invalid")

    def test_patient_add_image(self):
        """Test adding image to patient."""
        patient = Patient(patient_id="PAT001")
        success = patient.add_image("IMG001", "CC", "L")

        assert success is True
        assert "IMG001" in patient.image_ids
        assert "CC" in patient.projections
        assert "L" in patient.laterality
        assert patient.image_count == 1

    def test_patient_remove_image(self):
        """Test removing image from patient."""
        patient = Patient(patient_id="PAT001")
        patient.add_image("IMG001", "CC", "L")

        success = patient.remove_image("IMG001")

        assert success is True
        assert "IMG001" not in patient.image_ids
        assert patient.image_count == 0

    def test_patient_change_split_assignment(self):
        """Test changing split assignment."""
        patient = Patient(patient_id="PAT001", split_assignment="train")
        success = patient.change_split_assignment("validation")

        assert success is True
        assert patient.split_assignment == "validation"

    def test_patient_get_summary(self):
        """Test getting patient summary."""
        patient = Patient(
            patient_id="PAT001",
            image_count=2,
            projections=["CC"],
            laterality=["L"],
        )
        patient.add_image("IMG001", "CC", "L")

        summary = patient.get_patient_summary()

        assert isinstance(summary, dict)
        assert summary["patient_id"] == "PAT001"
        assert "image_count" in summary
        assert "split_assignment" in summary


class TestCreatePatientFromImages:
    """Test patient creation from images."""

    def test_create_patient_from_images(self):
        """Test creating patient from image data."""
        image_data = [
            {"image_id": "IMG001", "projection_type": "CC", "laterality": "L"},
            {"image_id": "IMG002", "projection_type": "MLO", "laterality": "L"},
            {"image_id": "IMG003", "projection_type": "CC", "laterality": "R"},
        ]

        patient = create_patient_from_images(
            patient_id="PAT001",
            image_data=image_data,
            split_assignment="train",
        )

        assert patient.patient_id == "PAT001"
        assert patient.image_count == 3
        assert patient.split_assignment == "train"
        assert "CC" in patient.projections
        assert "MLO" in patient.projections


class TestValidatePatientSplits:
    """Test patient split validation."""

    def test_validate_patient_splits_valid(self):
        """Test validation with valid splits."""
        patients = [
            Patient(patient_id="PAT001", split_assignment="train", image_count=4),
            Patient(patient_id="PAT002", split_assignment="validation", image_count=4),
            Patient(patient_id="PAT003", split_assignment="test", image_count=4),
        ]

        result = validate_patient_splits(patients)

        assert result["is_valid"] is True
        assert result["total_patients"] == 3
        assert result["split_counts"]["train"] == 1
        assert result["split_counts"]["validation"] == 1
        assert result["split_counts"]["test"] == 1

    def test_validate_patient_splits_distribution(self):
        """Test validation checks split distribution."""
        # Create imbalanced splits
        patients = [
            Patient(patient_id=f"PAT{i:03d}", split_assignment="train", image_count=4)
            for i in range(100)
        ]

        result = validate_patient_splits(patients)

        # Should warn about validation/test splits being too small
        assert len(result["errors"]) > 0


# ============================================================================
# Test Reproducibility Utilities
# ============================================================================


class TestFixSeeds:
    """Test seed fixing functionality."""

    def test_fix_seeds_basic(self):
        """Test basic seed fixing."""
        fix_seeds(42)
        val1 = torch.rand(1).item()

        fix_seeds(42)
        val2 = torch.rand(1).item()

        assert val1 == val2

    def test_fix_seeds_deterministic(self):
        """Test deterministic seed fixing."""
        fix_seeds(42, deterministic=True)

        # Verify seeds were set
        val = torch.rand(1).item()
        assert isinstance(val, float)


class TestRandomState:
    """Test random state capture and restore."""

    def test_get_random_state_structure(self):
        """Test that get_random_state returns expected structure."""
        state = get_random_state()

        assert isinstance(state, dict)
        assert "python" in state
        assert "numpy" in state
        assert "torch" in state
        assert "torch_cuda" in state

    def test_restore_random_state(self):
        """Test restoring random state."""
        fix_seeds(42)

        # Capture state
        state = get_random_state()
        val1 = torch.rand(1).item()

        # Do some random operations
        torch.rand(100)

        # Restore state
        restore_random_state(state)
        val2 = torch.rand(1).item()

        # Should get same value as after capture
        assert val1 == val2

    def test_get_and_restore_all_rngs(self):
        """Test that all RNGs are captured and restored."""
        fix_seeds(42)

        state = get_random_state()

        # Get values from all RNGs
        py_val1 = random.random()
        np_val1 = np.random.rand()
        torch_val1 = torch.rand(1).item()

        # Do some operations
        random.random()
        np.random.rand()
        torch.rand(1)

        # Restore and get values again
        restore_random_state(state)
        py_val2 = random.random()
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        assert py_val1 == py_val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2


# ============================================================================
# Integration Tests
# ============================================================================


class TestUtilsIntegration:
    """Integration tests for utility modules working together."""

    def test_reproducible_normalization_computation(self):
        """Test that normalization computation is reproducible."""
        class SimpleDataset(Dataset):
            def __init__(self, size=50):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                torch.manual_seed(idx)
                return torch.randn(3, 32, 32), 0

        # Compute stats twice with same seed
        fix_seeds(42)
        dataset1 = SimpleDataset(size=50)
        stats1 = compute_normalization_stats(dataset1, num_samples=50, batch_size=10)

        fix_seeds(42)
        dataset2 = SimpleDataset(size=50)
        stats2 = compute_normalization_stats(dataset2, num_samples=50, batch_size=10)

        # Stats should be identical
        assert stats1.mean == stats2.mean
        assert stats1.std == stats2.std

    def test_device_and_seeding_together(self):
        """Test device resolution and seeding work together."""
        device = resolve_device("auto")
        fix_seeds(42)

        # Create tensors on device
        tensor = torch.randn(10, device=device)

        assert tensor.device.type == device.type
        assert tensor.shape == (10,)

    def test_logging_and_patient_creation(self):
        """Test logging and patient creation integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir, "INFO")

            # Create patient (should log)
            patient = Patient(patient_id="PAT001")
            patient.add_image("IMG001", "CC", "L")

            # Check log file exists and contains patient info
            log_file = os.path.join(tmpdir, "run.log")
            assert os.path.exists(log_file)
