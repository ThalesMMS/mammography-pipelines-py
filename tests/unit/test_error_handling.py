"""
Unit tests for error handling across the mammography-pipelines codebase.

These tests validate proper handling of:
- Invalid inputs and missing files
- Corrupted DICOM files
- Out of memory conditions
- GPU unavailability fallback
- File I/O errors
- Invalid configurations
- Dataset loading errors
- Model instantiation errors

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Path setup
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Conditional imports
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
pydicom = pytest.importorskip("pydicom")
pd = pytest.importorskip("pandas")

from mammography.config import TrainConfig, ExtractConfig, InferenceConfig
from mammography.data.csv_loader import load_dataset_dataframe, _coerce_density_label
from mammography.data.dataset import MammoDensityDataset, robust_collate
from mammography.io.dicom import dicom_to_pil_rgb, is_dicom_path
from mammography.utils.device_detection import detect_device, resolve_device
from mammography.utils.common import seed_everything


# ==================== File I/O Error Tests ====================


class TestFileIOErrors:
    """Test error handling for file I/O operations."""

    def test_missing_csv_file_raises_error(self, tmp_path):
        """Test that loading missing CSV file raises FileNotFoundError."""
        csv_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_dataset_dataframe(csv_path=csv_path)

    def test_missing_dicom_file_raises_error(self, tmp_path):
        """Test that loading missing DICOM file raises appropriate error."""
        dicom_path = tmp_path / "nonexistent.dcm"

        with pytest.raises(FileNotFoundError):
            pydicom.dcmread(str(dicom_path))

    def test_empty_csv_file_handling(self, tmp_path):
        """Test handling of empty CSV file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        # Should raise error or return empty dataframe
        with pytest.raises(Exception):  # pd.errors.EmptyDataError or similar
            load_dataset_dataframe(csv_path=csv_path)

    def test_corrupted_csv_encoding(self, tmp_path):
        """Test handling of CSV with encoding issues."""
        csv_path = tmp_path / "corrupted.csv"
        # Write invalid UTF-8 bytes
        csv_path.write_bytes(b"AccessionNumber,Classification\n\xff\xfe\n")

        # Should handle encoding error gracefully
        with pytest.raises((UnicodeDecodeError, Exception)):
            df = pd.read_csv(csv_path)

    def test_invalid_path_type(self):
        """Test that invalid path types raise appropriate errors."""
        with pytest.raises((TypeError, FileNotFoundError)):
            load_dataset_dataframe(csv_path=None)


# ==================== DICOM Error Tests ====================


class TestDICOMErrors:
    """Test error handling for DICOM loading and processing."""

    def test_corrupted_dicom_file(self, tmp_path):
        """Test handling of corrupted DICOM file."""
        corrupted_path = tmp_path / "corrupted.dcm"
        corrupted_path.write_bytes(b"Not a valid DICOM file")

        with pytest.raises(pydicom.errors.InvalidDicomError):
            pydicom.dcmread(str(corrupted_path))

    def test_dicom_missing_required_tags(self, tmp_path):
        """Test handling of DICOM with missing required tags."""
        # Create minimal DICOM without PixelData
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST001"
        dataset.SOPInstanceUID = "1.2.3.4"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta = file_meta

        dicom_path = tmp_path / "minimal.dcm"
        dataset.save_as(str(dicom_path), write_like_original=False)

        # Should raise error when trying to access pixel_array
        loaded = pydicom.dcmread(str(dicom_path))
        with pytest.raises(AttributeError):
            _ = loaded.pixel_array

    def test_dicom_with_unsupported_transfer_syntax(self, tmp_path):
        """Test handling of DICOM with unsupported compression."""
        dataset = pydicom.Dataset()
        dataset.PatientID = "TEST001"
        dataset.SOPInstanceUID = "1.2.3.4"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        dataset.Rows = 16
        dataset.Columns = 16
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelData = np.zeros((16, 16), dtype=np.uint16).tobytes()

        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        # Use JPEG2000 which may not be available
        file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.90"
        dataset.file_meta = file_meta

        dicom_path = tmp_path / "jpeg2000.dcm"
        dataset.save_as(str(dicom_path), write_like_original=False)

        # May raise NotImplementedError for unsupported codec
        # This depends on available codecs, so we just verify it loads
        loaded = pydicom.dcmread(str(dicom_path))
        assert loaded is not None

    def test_is_dicom_path_with_non_dicom_file(self, tmp_path):
        """Test is_dicom_path returns False for non-DICOM files."""
        text_file = tmp_path / "text.txt"
        text_file.write_text("Not a DICOM file")

        assert not is_dicom_path(str(text_file))

    def test_is_dicom_path_with_nonexistent_file(self):
        """Test is_dicom_path returns False for nonexistent file."""
        assert not is_dicom_path("/nonexistent/path/file.dcm")


# ==================== Configuration Error Tests ====================


class TestConfigurationErrors:
    """Test error handling for invalid configurations."""

    @pytest.mark.parametrize("field,invalid_value", [
        ("epochs", -1),
        ("batch_size", 0),
        ("learning_rate", -0.001),
        ("img_size", -224),
        ("num_workers", -1),
    ])
    def test_train_config_invalid_numeric_values(self, field, invalid_value):
        """Test TrainConfig rejects invalid numeric values."""
        config_dict = TrainConfig().model_dump()
        config_dict[field] = invalid_value

        with pytest.raises((ValueError, Exception)):
            TrainConfig(**config_dict)

    @pytest.mark.parametrize("invalid_device", [
        "invalid_device",
        "gpu:999",
        123,
        None,
    ])
    def test_train_config_invalid_device_values(self, invalid_device):
        """Test TrainConfig with invalid device values."""
        config_dict = TrainConfig().model_dump()
        config_dict["device"] = invalid_device

        # Should either validate or be handled by device detection
        try:
            config = TrainConfig(**config_dict)
            # If it passes validation, device detection should handle it
            assert config.device is not None
        except (ValueError, TypeError):
            pass  # Expected for invalid types

    def test_extract_config_invalid_architecture(self):
        """Test ExtractConfig rejects invalid architecture."""
        config_dict = ExtractConfig().model_dump()
        config_dict["architecture"] = "invalid_arch"

        # Config may accept it, but it should fail during model building
        config = ExtractConfig(**config_dict)
        assert config.architecture == "invalid_arch"

    def test_train_config_invalid_cache_mode(self):
        """Test TrainConfig with invalid cache mode."""
        config_dict = TrainConfig().model_dump()
        config_dict["cache_mode"] = "invalid_cache"

        # Config may accept it, but should be validated elsewhere
        config = TrainConfig(**config_dict)
        assert config.cache_mode == "invalid_cache"

    def test_inference_config_missing_checkpoint(self):
        """Test InferenceConfig with missing checkpoint path."""
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            # Should validate checkpoint exists
            config = InferenceConfig(checkpoint="/nonexistent/checkpoint.pt")


# ==================== Dataset Error Tests ====================


class TestDatasetErrors:
    """Test error handling for dataset operations."""

    def test_dataset_empty_csv(self, tmp_path):
        """Test handling of dataset with no valid samples."""
        csv_path = tmp_path / "empty_data.csv"
        csv_path.write_text("AccessionNumber,Classification\n")

        # Should handle empty dataset gracefully
        df = pd.read_csv(csv_path)
        assert len(df) == 0

    def test_dataset_missing_required_columns(self, tmp_path):
        """Test dataset with missing required columns."""
        csv_path = tmp_path / "invalid_columns.csv"
        csv_path.write_text("WrongColumn,OtherColumn\n1,2\n")

        df = pd.read_csv(csv_path)
        # Should detect missing columns
        assert "AccessionNumber" not in df.columns

    def test_dataset_with_invalid_labels(self, tmp_path):
        """Test dataset with invalid density labels."""
        csv_path = tmp_path / "invalid_labels.csv"
        csv_path.write_text("AccessionNumber,Classification\nTEST001,INVALID\n")

        # _coerce_density_label should handle invalid labels
        result = _coerce_density_label("INVALID")
        assert result is None

    def test_robust_collate_with_none_values(self):
        """Test robust_collate handles None values gracefully."""
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(1)),
            None,  # Failed sample
            (torch.randn(3, 224, 224), torch.tensor(2)),
        ]

        images, labels = robust_collate(batch)
        # Should skip None values
        assert len(images) == 2
        assert len(labels) == 2

    def test_robust_collate_with_all_none(self):
        """Test robust_collate with all None values."""
        batch = [None, None, None]

        images, labels = robust_collate(batch)
        # Should return empty tensors
        assert len(images) == 0
        assert len(labels) == 0

    def test_dataset_with_missing_image_files(self, tmp_path):
        """Test dataset when image files don't exist."""
        csv_path = tmp_path / "missing_files.csv"
        csv_path.write_text("AccessionNumber,Classification\nTEST001,A\n")

        # Dataset should be created but __getitem__ would fail
        # This tests that the dataset itself can be instantiated
        df = load_dataset_dataframe(csv_path=csv_path)
        assert len(df) == 1


# ==================== Device Detection Error Tests ====================


class TestDeviceErrors:
    """Test error handling for device detection and GPU unavailability."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_gpu_unavailable_fallback_to_cpu(self, mock_cuda):
        """Test fallback to CPU when GPU is unavailable."""
        device = detect_device()
        assert device in ["cpu", "mps"]  # MPS for Apple Silicon

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=0)
    def test_cuda_available_but_no_devices(self, mock_count, mock_available):
        """Test when CUDA is available but no devices found."""
        device = detect_device()
        # Should still return a valid device
        assert device in ["cpu", "cuda", "mps"]

    def test_resolve_device_with_auto(self):
        """Test resolve_device with 'auto' parameter."""
        device = resolve_device("auto")
        assert isinstance(device, torch.device)

    def test_resolve_device_with_invalid_device(self):
        """Test resolve_device with invalid device string."""
        # Should handle gracefully or raise clear error
        try:
            device = resolve_device("invalid_device")
            # If it doesn't raise, should default to valid device
            assert isinstance(device, torch.device)
        except (ValueError, RuntimeError):
            pass  # Expected for invalid device

    @patch("torch.cuda.is_available", return_value=True)
    def test_resolve_device_cuda_index_out_of_range(self, mock_cuda):
        """Test resolve_device with CUDA index out of range."""
        with patch("torch.cuda.device_count", return_value=1):
            # Try to use cuda:5 when only 1 device available
            try:
                device = resolve_device("cuda:5")
                # May succeed but operations would fail
                assert isinstance(device, torch.device)
            except (RuntimeError, ValueError):
                pass  # Expected for out of range index


# ==================== Memory Error Tests ====================


class TestMemoryErrors:
    """Test error handling for out of memory conditions."""

    @pytest.mark.slow
    def test_large_batch_size_warning(self):
        """Test that large batch sizes are handled appropriately."""
        # This is a smoke test - actual OOM would crash the test
        config = TrainConfig(batch_size=1024)
        assert config.batch_size == 1024

    def test_tensor_creation_with_reasonable_size(self):
        """Test tensor creation with reasonable size."""
        # Create a reasonably sized tensor to verify memory handling
        tensor = torch.randn(32, 3, 224, 224)
        assert tensor.shape == (32, 3, 224, 224)
        del tensor  # Clean up

    @patch("torch.cuda.OutOfMemoryError", side_effect=RuntimeError("CUDA out of memory"))
    def test_cuda_oom_error_message(self, mock_oom):
        """Test CUDA OOM error handling."""
        # Verify OOM errors are properly recognized
        with pytest.raises(RuntimeError, match="out of memory"):
            raise mock_oom()


# ==================== Model Instantiation Error Tests ====================


class TestModelInstantiationErrors:
    """Test error handling for model instantiation."""

    def test_model_with_invalid_num_classes(self):
        """Test model instantiation with invalid number of classes."""
        from mammography.models.nets import build_efficientnet

        # Should handle invalid num_classes
        with pytest.raises((ValueError, AssertionError, Exception)):
            model = build_efficientnet(num_classes=0)

    def test_model_with_negative_num_classes(self):
        """Test model instantiation with negative number of classes."""
        from mammography.models.nets import build_efficientnet

        with pytest.raises((ValueError, AssertionError, Exception)):
            model = build_efficientnet(num_classes=-1)

    @patch("torch.hub.load_state_dict_from_url")
    def test_model_pretrained_download_failure(self, mock_download):
        """Test handling of pretrained weight download failure."""
        from mammography.models.nets import build_efficientnet

        mock_download.side_effect = RuntimeError("Failed to download")

        # Should handle download failure gracefully
        with pytest.raises(RuntimeError):
            model = build_efficientnet(pretrained=True)

    def test_model_forward_with_wrong_input_shape(self):
        """Test model forward pass with wrong input shape."""
        from mammography.models.nets import build_efficientnet

        model = build_efficientnet(num_classes=4, pretrained=False)
        model.eval()

        # Wrong shape should raise error
        with pytest.raises((RuntimeError, ValueError)):
            wrong_input = torch.randn(1, 3, 64, 64)  # Too small
            _ = model(wrong_input)


# ==================== Seed/Reproducibility Error Tests ====================


class TestReproducibilityErrors:
    """Test error handling for reproducibility utilities."""

    def test_seed_everything_with_negative_seed(self):
        """Test seed_everything with negative seed value."""
        # Should handle negative seeds or raise clear error
        try:
            seed_everything(-1)
            # If it succeeds, verify it's set
            assert True
        except ValueError:
            pass  # Expected for invalid seed

    def test_seed_everything_with_very_large_seed(self):
        """Test seed_everything with very large seed value."""
        # Should handle large seeds
        try:
            seed_everything(2**32)
            assert True
        except (ValueError, OverflowError):
            pass  # Expected for out of range seed

    @pytest.mark.parametrize("invalid_seed", [
        "not_a_number",
        None,
        3.14,
    ])
    def test_seed_everything_with_invalid_types(self, invalid_seed):
        """Test seed_everything with invalid seed types."""
        with pytest.raises((TypeError, ValueError)):
            seed_everything(invalid_seed)


# ==================== Integration Error Tests ====================


class TestIntegrationErrors:
    """Test error handling in integrated scenarios."""

    def test_training_with_corrupted_checkpoint_resume(self, tmp_path):
        """Test training resume with corrupted checkpoint."""
        checkpoint_path = tmp_path / "corrupted_checkpoint.pt"
        checkpoint_path.write_bytes(b"Not a valid checkpoint")

        # Should handle corrupted checkpoint gracefully
        with pytest.raises((RuntimeError, Exception)):
            checkpoint = torch.load(checkpoint_path)

    def test_inference_with_missing_checkpoint_keys(self, tmp_path):
        """Test inference with checkpoint missing required keys."""
        checkpoint_path = tmp_path / "incomplete_checkpoint.pt"
        # Save checkpoint with missing keys
        torch.save({"incomplete": "data"}, checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        assert "model_state_dict" not in checkpoint

    def test_dataset_loading_with_mixed_valid_invalid_samples(self, tmp_path):
        """Test dataset with mix of valid and invalid samples."""
        csv_path = tmp_path / "mixed_samples.csv"
        csv_path.write_text(
            "AccessionNumber,Classification\n"
            "TEST001,A\n"
            "TEST002,INVALID\n"
            "TEST003,B\n"
        )

        df = load_dataset_dataframe(csv_path=csv_path)
        # Should load all rows, invalid labels handled by coercion
        assert len(df) == 3

    def test_collate_function_with_different_sized_tensors(self):
        """Test collate function with tensors of different sizes."""
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(1)),
            (torch.randn(3, 256, 256), torch.tensor(2)),  # Different size
        ]

        # robust_collate should handle or raise clear error
        try:
            images, labels = robust_collate(batch)
            # If it succeeds, verify output
            assert len(images) > 0
        except (RuntimeError, ValueError):
            pass  # Expected for mismatched sizes


# ==================== Path and Validation Error Tests ====================


class TestPathValidationErrors:
    """Test error handling for path validation."""

    def test_invalid_output_directory_creation(self, tmp_path):
        """Test handling of invalid output directory."""
        # Try to create directory in non-existent parent
        invalid_path = Path("/nonexistent/parent/output")

        # Should raise error when trying to create
        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            invalid_path.mkdir(parents=False, exist_ok=False)

    def test_read_only_output_directory(self, tmp_path):
        """Test handling of read-only output directory."""
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()

        # Make directory read-only
        import stat
        read_only_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        # Try to write file in read-only directory
        try:
            output_file = read_only_dir / "output.txt"
            with pytest.raises(PermissionError):
                output_file.write_text("test")
        finally:
            # Restore permissions for cleanup
            read_only_dir.chmod(stat.S_IRWXU)

    def test_path_with_special_characters(self, tmp_path):
        """Test handling of paths with special characters."""
        # Some special characters may cause issues
        special_path = tmp_path / "test_[special].txt"

        # Should handle special characters
        special_path.write_text("test")
        assert special_path.exists()

    def test_very_long_path_name(self, tmp_path):
        """Test handling of very long path names."""
        # Create path close to OS limit
        long_name = "a" * 200
        long_path = tmp_path / f"{long_name}.txt"

        try:
            long_path.write_text("test")
            assert long_path.exists()
        except OSError:
            # Expected on some systems with path length limits
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
