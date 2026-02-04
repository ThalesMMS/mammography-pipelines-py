from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography import cli


def test_cli_explain_routes_to_explain_script() -> None:
    """Test that 'explain' command routes to mammography.commands.explain."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main(["--dry-run", "explain"])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.explain"


def test_cli_explain_with_model_and_images() -> None:
    """Test that 'explain' command forwards model and images arguments."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--model-path", "model.pth",
            "--images-dir", "data/test/",
            "--output-dir", "outputs/explanations",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.explain"

    # Verify forwarded arguments contain the required parameters
    forwarded_args = mock_run.call_args[0][2]
    assert "--model-path" in forwarded_args
    assert "model.pth" in forwarded_args
    assert "--images-dir" in forwarded_args
    assert "data/test/" in forwarded_args


def test_cli_explain_with_gradcam_method() -> None:
    """Test that 'explain' command forwards GradCAM method option."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--model-path", "model.pth",
            "--images-dir", "imgs/",
            "--method", "gradcam",
            "--model-type", "resnet50",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()

    # Verify method and model type are forwarded
    forwarded_args = mock_run.call_args[0][2]
    assert "--method" in forwarded_args
    assert "gradcam" in forwarded_args
    assert "--model-type" in forwarded_args
    assert "resnet50" in forwarded_args


def test_cli_explain_with_attention_method() -> None:
    """Test that 'explain' command forwards attention method for ViT models."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--model-path", "vit.pth",
            "--images-dir", "imgs/",
            "--method", "attention",
            "--model-type", "vit",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()

    forwarded_args = mock_run.call_args[0][2]
    assert "--method" in forwarded_args
    assert "attention" in forwarded_args
    assert "--model-type" in forwarded_args
    assert "vit" in forwarded_args


def test_cli_explain_with_batch_size() -> None:
    """Test that 'explain' command forwards batch size parameter."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--model-path", "model.pth",
            "--images-dir", "imgs/",
            "--batch-size", "16",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()

    forwarded_args = mock_run.call_args[0][2]
    assert "--batch-size" in forwarded_args
    assert "16" in forwarded_args


def test_cli_explain_with_visualization_options() -> None:
    """Test that 'explain' command forwards visualization customization options."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--model-path", "model.pth",
            "--images-dir", "imgs/",
            "--alpha", "0.5",
            "--colormap", "viridis",
            "--target-layer", "layer4",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()

    forwarded_args = mock_run.call_args[0][2]
    assert "--alpha" in forwarded_args
    assert "0.5" in forwarded_args
    assert "--colormap" in forwarded_args
    assert "viridis" in forwarded_args
    assert "--target-layer" in forwarded_args
    assert "layer4" in forwarded_args


def test_cli_explain_with_report_generation() -> None:
    """Test that 'explain' command forwards report generation flag."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--model-path", "model.pth",
            "--images-dir", "imgs/",
            "--generate-report",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()

    forwarded_args = mock_run.call_args[0][2]
    assert "--generate-report" in forwarded_args


def test_cli_explain_with_config_file() -> None:
    """Test that 'explain' command supports config file loading."""
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        # Note: config file doesn't need to exist in dry-run mode
        exit_code = cli.main([
            "--dry-run",
            "explain",
            "--config", "configs/explain.yaml",
        ])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.explain"
