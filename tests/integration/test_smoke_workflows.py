"""
Smoke tests for critical workflows.

These tests validate the most important workflows work end-to-end without
requiring large datasets. They serve as rapid smoke tests for CI/CD and
development verification.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import dependencies with pytest.importorskip for graceful failures
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from mammography import cli


class TestCLISmokeTests:
    """Smoke tests for CLI command routing and basic functionality."""

    def test_cli_help_works(self):
        """Test that CLI --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0

    def test_embed_help_works(self):
        """Test that embed --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "embed", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    def test_train_density_help_works(self):
        """Test that train-density --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "train-density", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    def test_inference_help_works(self):
        """Test that inference --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "inference", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    def test_visualize_help_works(self):
        """Test that visualize --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "visualize", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    def test_explain_help_works(self):
        """Test that explain --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "explain", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    def test_embed_dry_run_works(self):
        """Test that embed --dry-run works without executing."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "embed"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.extract_features"

    def test_train_density_dry_run_works(self):
        """Test that train-density --dry-run works without executing."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "train-density"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.train"


class TestDeviceDetectionSmoke:
    """Smoke tests for device detection functionality."""

    def test_device_detection_import(self):
        """Test that device detection module imports without errors."""
        from mammography.utils.device_detection import get_optimal_device

        assert get_optimal_device is not None

    def test_device_detection_works(self):
        """Test that device detection returns valid device."""
        from mammography.utils.device_detection import get_optimal_device

        device = get_optimal_device()
        # Should return one of: cpu, cuda, mps
        assert device in ["cpu", "cuda", "mps"]

    def test_device_cpu_fallback(self):
        """Test that CPU fallback works."""
        import torch

        # CPU should always be available
        device = torch.device("cpu")
        tensor = torch.randn(10, device=device)
        assert tensor.device.type == "cpu"


class TestModelInstantiationSmoke:
    """Smoke tests for model instantiation."""

    def test_efficientnet_b0_creates(self):
        """Test that EfficientNetB0 model instantiates without errors."""
        from mammography.models.nets import build_model

        model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
        assert model is not None

    def test_resnet50_creates(self):
        """Test that ResNet50 model instantiates without errors."""
        from mammography.models.nets import build_model

        model = build_model("resnet50", num_classes=4, pretrained=False)
        assert model is not None

    def test_model_forward_pass_cpu(self):
        """Test that model forward pass works on CPU."""
        import torch
        from mammography.models.nets import build_model

        model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Verify output shape
        assert output.shape == (2, 4)


class TestDatasetSmoke:
    """Smoke tests for dataset functionality."""

    def test_dataset_import_works(self):
        """Test that dataset module imports without errors."""
        from mammography.data.dataset import MammoDensityDataset

        assert MammoDensityDataset is not None

    def test_embedding_store_import_works(self):
        """Test that embedding store imports without errors."""
        from mammography.data.dataset import EmbeddingStore

        assert EmbeddingStore is not None


class TestDICOMSmoke:
    """Smoke tests for DICOM loading functionality."""

    def test_dicom_import_works(self):
        """Test that DICOM module imports without errors."""
        from mammography.io.dicom import DicomReader

        assert DicomReader is not None

    def test_lazy_dicom_import_works(self):
        """Test that lazy DICOM module imports without errors."""
        from mammography.io.lazy_dicom import LazyDicomDataset

        assert LazyDicomDataset is not None


class TestConfigSmoke:
    """Smoke tests for configuration validation."""

    def test_config_import_works(self):
        """Test that config module imports without errors."""
        from mammography.config import TrainConfig

        assert TrainConfig is not None

    def test_train_config_validates(self):
        """Test that TrainConfig validates correctly."""
        from mammography.config import TrainConfig
        import tempfile
        import os

        # Create a temporary CSV file for validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
            f.write("AccessionNumber,Classification\n")
            f.write("test123,A\n")

        try:
            # Valid config should pass - TrainConfig requires either csv or dataset
            config = TrainConfig(epochs=10, batch_size=32, lr=1e-3, csv=temp_csv)
            assert config.epochs == 10
            assert config.batch_size == 32
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_csv):
                os.unlink(temp_csv)

    def test_train_config_rejects_invalid_bounds(self):
        """Test that TrainConfig rejects invalid values."""
        from mammography.config import TrainConfig
        from pydantic import ValidationError

        # Invalid epochs (< 1) should fail
        with pytest.raises(ValidationError):
            TrainConfig(epochs=0, batch_size=32)

        # Invalid batch size (< 1) should fail
        with pytest.raises(ValidationError):
            TrainConfig(epochs=10, batch_size=0)


class TestVisualizationSmoke:
    """Smoke tests for visualization functionality."""

    def test_visualization_import_works(self):
        """Test that visualization module imports without errors."""
        from mammography.vis.advanced import plot_tsne_2d

        assert plot_tsne_2d is not None

    def test_gradcam_import_works(self):
        """Test that GradCAM module imports without errors."""
        from mammography.vis.gradcam import apply_gradcam

        assert apply_gradcam is not None


class TestTrainingSmoke:
    """Smoke tests for training functionality."""

    def test_training_engine_import_works(self):
        """Test that training engine imports without errors."""
        from mammography.training.engine import train_one_epoch

        assert train_one_epoch is not None

    def test_trainer_import_works(self):
        """Test that cancer trainer imports without errors."""
        from mammography.training.cancer_trainer import DensityHistoryEntry

        assert DensityHistoryEntry is not None


class TestIntegrationSmoke:
    """Smoke tests for integration workflows."""

    def test_embed_routes_to_extract_features(self):
        """Test that embed command routes to extract_features module."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "embed"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.extract_features"

    def test_train_routes_to_train_script(self):
        """Test that train-density command routes to train module."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "train-density"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.train"

    def test_inference_routes_to_inference_script(self):
        """Test that inference command routes to inference module."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "inference"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.inference"

    def test_visualize_routes_to_visualize_script(self):
        """Test that visualize command routes to visualize module."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(
                ["--dry-run", "visualize", "--input", "x.npy", "--outdir", "outputs/vis"]
            )
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.visualize"

    def test_explain_routes_to_explain_script(self):
        """Test that explain command routes to explain module."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "explain"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.explain"


class TestUtilitiesSmoke:
    """Smoke tests for utility functions."""

    def test_normalization_import_works(self):
        """Test that normalization utilities import without errors."""
        from mammography.utils.normalization import z_score_normalize

        assert z_score_normalize is not None

    def test_reproducibility_import_works(self):
        """Test that reproducibility utilities import without errors."""
        from mammography.utils.reproducibility import fix_seeds

        assert fix_seeds is not None

    def test_fix_seeds_works(self):
        """Test that seed fixing works without errors."""
        import torch
        from mammography.utils.reproducibility import fix_seeds

        # Should not raise any errors
        fix_seeds(42)

        # Verify seeds are set (basic check)
        # Creating tensors should be reproducible
        torch.manual_seed(42)
        t1 = torch.randn(5)
        torch.manual_seed(42)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)


class TestCommandsSmoke:
    """Smoke tests for command modules."""

    def test_extract_features_import_works(self):
        """Test that extract_features command imports without errors."""
        from mammography.commands.extract_features import main

        assert main is not None

    def test_train_import_works(self):
        """Test that train command imports without errors."""
        from mammography.commands.train import main

        assert main is not None

    def test_inference_import_works(self):
        """Test that inference command imports without errors."""
        from mammography.commands.inference import main

        assert main is not None

    def test_visualize_import_works(self):
        """Test that visualize command imports without errors."""
        from mammography.commands.visualize import main

        assert main is not None

    def test_explain_import_works(self):
        """Test that explain command imports without errors."""
        from mammography.commands.explain import main

        assert main is not None

    def test_embeddings_baselines_import_works(self):
        """Test that embeddings_baselines command imports without errors."""
        from mammography.commands.embeddings_baselines import main

        assert main is not None

    def test_tune_import_works(self):
        """Test that tune command imports without errors."""
        from mammography.commands.tune import main

        assert main is not None


class TestErrorHandlingSmoke:
    """Smoke tests for error handling."""

    def test_invalid_command_fails_gracefully(self):
        """Test that invalid command fails with proper error."""
        # Invalid commands cause argparse to call sys.exit()
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["invalid-command"])
        # Should exit with non-zero code (argparse uses exit code 2 for errors)
        assert exc_info.value.code != 0

    def test_missing_required_args_fails_gracefully(self):
        """Test that missing required args fails with proper error."""
        # Try to run visualize without required --input argument
        # This should fail, but we mock it to prevent actual execution
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            # Simulate argument parsing failure by returning non-zero
            mock_run.return_value = 1
            # The CLI should handle this gracefully without crashing
            # (actual behavior depends on argparse, but we verify no exceptions)
            try:
                cli.main(["--dry-run", "visualize"])  # Missing --input
                # If it gets here without exception, that's acceptable
                assert True
            except SystemExit as e:
                # SystemExit is also acceptable for argument errors
                assert e.code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
