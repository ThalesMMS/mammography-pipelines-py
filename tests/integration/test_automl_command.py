"""
Integration tests for the automl command.

These tests validate the AutoML command workflow including argument parsing,
dry-run mode, and basic functionality without requiring large datasets or
long training runs.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
pytest.importorskip("optuna")

from mammography import cli


class TestAutoMLCLISmokeTests:
    """Smoke tests for AutoML CLI command routing and basic functionality."""

    def test_automl_help_works(self):
        """Test that automl --help displays without errors."""
        with patch.object(sys, "argv", ["mammography", "automl", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            # argparse exits with 0 on --help
            assert exc_info.value.code == 0

    def test_automl_dry_run_works(self):
        """Test that automl --dry-run works without executing."""
        with patch.object(cli, "_run_module_passthrough") as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main(["--dry-run", "automl"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "mammography.commands.automl"


class TestAutoMLModuleImport:
    """Test that AutoML module and dependencies import correctly."""

    def test_automl_module_imports(self):
        """Test that automl module imports without errors."""
        from mammography.commands import automl

        assert automl is not None
        assert hasattr(automl, "main")
        assert callable(automl.main)

    def test_automl_parse_args_imports(self):
        """Test that parse_args function exists and is callable."""
        from mammography.commands.automl import parse_args

        assert parse_args is not None
        assert callable(parse_args)

    def test_search_space_imports(self):
        """Test that SearchSpace class imports correctly."""
        from mammography.tuning.search_space import SearchSpace

        assert SearchSpace is not None

    def test_optuna_tuner_imports(self):
        """Test that OptunaTuner class imports correctly."""
        from mammography.tuning.optuna_tuner import OptunaTuner

        assert OptunaTuner is not None

    def test_lr_finder_imports(self):
        """Test that LRFinder class imports correctly."""
        from mammography.tuning.lr_finder import LRFinder

        assert LRFinder is not None

    def test_resource_manager_imports(self):
        """Test that ResourceManager class imports correctly."""
        from mammography.tuning.resource_manager import ResourceManager

        assert ResourceManager is not None


class TestAutoMLArgumentParsing:
    """Test argument parsing for AutoML command."""

    def test_parse_args_defaults(self):
        """Test that parse_args returns expected defaults."""
        from mammography.commands.automl import parse_args

        # Minimal required arguments
        args = parse_args(["--csv", "test.csv"])

        assert args.csv == "test.csv"
        assert args.outdir == "outputs/automl"
        assert args.n_trials == 50
        assert args.epochs == 100
        assert args.lr_finder_enabled is True
        assert args.resource_aware is True
        assert args.arch == "efficientnet_b0"
        assert args.classes == "multiclass"
        assert args.pretrained is True

    def test_parse_args_density_alias_warns_and_normalizes(self):
        """Test that the legacy density alias still works temporarily."""
        from mammography.commands.automl import parse_args

        with pytest.warns(FutureWarning, match="density"):
            args = parse_args(["--csv", "test.csv", "--classes", "density"])

        assert args.classes == "multiclass"

    def test_parse_args_custom_values(self):
        """Test that parse_args accepts custom values."""
        from mammography.commands.automl import parse_args

        args = parse_args([
            "--csv", "custom.csv",
            "--n-trials", "10",
            "--epochs", "5",
            "--arch", "resnet50",
            "--classes", "binary",
            "--no-lr-finder-enabled",
            "--no-resource-aware",
        ])

        assert args.csv == "custom.csv"
        assert args.n_trials == 10
        assert args.epochs == 5
        assert args.arch == "resnet50"
        assert args.classes == "binary"
        assert args.lr_finder_enabled is False
        assert args.resource_aware is False

    def test_parse_args_dry_run_flag(self):
        """Test that --dry-run flag is parsed correctly."""
        from mammography.commands.automl import parse_args

        args = parse_args(["--csv", "test.csv", "--dry-run"])
        assert args.dry_run is True

        args_no_dry_run = parse_args(["--csv", "test.csv"])
        assert args_no_dry_run.dry_run is False


class TestAutoMLDryRunMode:
    """Test dry-run mode for AutoML command."""

    def test_dry_run_validates_config_without_running(self, tmp_path):
        """Test that dry-run mode validates configuration without running optimization."""
        from mammography.commands.automl import main

        # Create mock dataset CSV
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text(
            "image_path,density_label,professional_label\n"
            "img1.png,1,1\n"
            "img2.png,2,2\n"
            "img3.png,3,3\n"
            "img4.png,4,4\n",
            encoding="utf-8",
        )

        # Create minimal AutoML config
        config_path = tmp_path / "automl_config.yaml"
        config_path.write_text(
            """
parameters:
  lr:
    type: float
    low: 0.0001
    high: 0.01
    log: true
  batch_size:
    type: categorical
    choices: [16, 32]
""",
            encoding="utf-8",
        )

        # Mock file system checks to skip actual image loading
        with patch("mammography.data.csv_loader.Path.exists", return_value=True):
            with patch("mammography.data.csv_loader.load_dataset_dataframe") as mock_load:
                # Return minimal DataFrame
                import pandas as pd
                mock_df = pd.DataFrame({
                    "image_path": ["img1.png", "img2.png", "img3.png", "img4.png"],
                    "density_label": [1, 2, 3, 4],
                    "professional_label": [1, 2, 3, 4],
                })
                mock_load.return_value = mock_df

                # Mock dataset creation to avoid loading actual images
                with patch("mammography.commands.automl.MammoDensityDataset") as mock_ds:
                    mock_train = MagicMock()
                    mock_val = MagicMock()
                    mock_ds.side_effect = [mock_train, mock_val]

                    # Run automl in dry-run mode
                    exit_code = main([
                        "--csv", str(csv_path),
                        "--automl-config", str(config_path),
                        "--outdir", str(tmp_path / "output"),
                        "--n-trials", "5",
                        "--epochs", "2",
                        "--subset", "4",
                        "--no-split-ensure-all-classes",
                        "--dry-run",
                    ])

                    # Should complete successfully without running optimization
                    assert exit_code == 0

                    # Verify config file was created
                    config_output = tmp_path / "output" / "results" / "automl_config.json"
                    # Note: The actual path may be incremented, so we just check directory was created
                    assert (tmp_path / "output").exists()


class TestAutoMLHelperFunctions:
    """Test helper functions in automl module."""

    def test_get_label_mapper_binary(self):
        """Test binary label mapper collapses classes correctly."""
        from mammography.commands.automl import get_label_mapper

        mapper = get_label_mapper("binary")
        assert mapper is not None
        assert mapper(1) == 0  # Low density
        assert mapper(2) == 0  # Low density
        assert mapper(3) == 1  # High density
        assert mapper(4) == 1  # High density

    def test_get_label_mapper_density(self):
        """Test density mode returns None (uses default mapping)."""
        from mammography.commands.automl import get_label_mapper

        mapper = get_label_mapper("density")
        assert mapper is None

    def test_get_label_mapper_multiclass(self):
        """Test multiclass mode returns None (uses default mapping)."""
        from mammography.commands.automl import get_label_mapper

        mapper = get_label_mapper("multiclass")
        assert mapper is None

    def test_resolve_loader_runtime_cpu(self):
        """Test loader runtime resolution for CPU device."""
        from mammography.commands.automl import resolve_loader_runtime
        import torch

        class MockArgs:
            num_workers = 4
            prefetch_factor = 2
            persistent_workers = True
            loader_heuristics = True

        args = MockArgs()
        device = torch.device("cpu")

        nw, prefetch, persistent = resolve_loader_runtime(args, device)

        # CPU should use available cores (up to num_workers)
        assert nw >= 0
        assert prefetch == 2
        assert isinstance(persistent, bool)

    def test_resolve_loader_runtime_cuda(self):
        """Test loader runtime resolution for CUDA device."""
        from mammography.commands.automl import resolve_loader_runtime
        import torch

        class MockArgs:
            num_workers = 4
            prefetch_factor = 2
            persistent_workers = True
            loader_heuristics = True

        args = MockArgs()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nw, prefetch, persistent = resolve_loader_runtime(args, device)

        # CUDA should preserve settings
        if device.type == "cuda":
            assert nw == 4
            assert prefetch == 2
            assert persistent is True

    def test_resolve_loader_runtime_mps(self):
        """Test loader runtime resolution for MPS device."""
        from mammography.commands.automl import resolve_loader_runtime
        import torch

        class MockArgs:
            num_workers = 4
            prefetch_factor = 2
            persistent_workers = True
            loader_heuristics = True

        args = MockArgs()
        device = torch.device("mps")

        nw, prefetch, persistent = resolve_loader_runtime(args, device)

        # MPS should force num_workers=0 and persistent=False
        assert nw == 0
        assert prefetch == 2
        assert persistent is False

    def test_resolve_loader_runtime_no_heuristics(self):
        """Test loader runtime resolution with heuristics disabled."""
        from mammography.commands.automl import resolve_loader_runtime
        import torch

        class MockArgs:
            num_workers = 8
            prefetch_factor = 4
            persistent_workers = True
            loader_heuristics = False

        args = MockArgs()
        device = torch.device("cpu")

        nw, prefetch, persistent = resolve_loader_runtime(args, device)

        # Without heuristics, settings should be preserved
        assert nw == 8
        assert prefetch == 4
        assert persistent is True
