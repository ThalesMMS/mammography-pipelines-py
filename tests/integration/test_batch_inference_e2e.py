"""
End-to-end integration tests for batch-inference command.

These tests verify the complete batch inference pipeline including:
- Progress tracking with tqdm
- Multiple output formats (CSV, JSON, JSONL)
- Checkpoint creation and saving
- Resume from checkpoint functionality
- Parallel data loading
- Memory-efficient processing

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import dependencies with pytest.importorskip for graceful failures
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pandas")
pytest.importorskip("tqdm")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal dummy checkpoint file for testing."""
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 4)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DummyModel()
    checkpoint_path = tmp_path / "dummy_model.pt"

    # Save checkpoint in expected format
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': 1,
        'arch': 'efficientnet_b0',  # Metadata
    }, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def sample_test_images(tmp_path: Path) -> Path:
    """Create a directory with 10 sample PNG images for testing."""
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()

    # Create 10 sample images with different colors
    for i in range(10):
        img = Image.new("RGB", (224, 224), color=(i * 25, 120, 60))
        img_path = test_dir / f"image_{i:03d}.png"
        img.save(img_path)

    return test_dir


@pytest.mark.integration
@pytest.mark.cpu
class TestBatchInferenceE2E:
    """End-to-end integration tests for batch-inference pipeline."""

    def test_batch_inference_csv_output(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test batch-inference with CSV output format."""
        from mammography.commands.batch_inference import main

        output_file = tmp_path / "results.csv"

        # Prepare arguments
        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--output-format", "csv",
            "--batch-size", "4",
            "--device", "cpu",
            "--num-workers", "0",  # Avoid multiprocessing issues
        ]

        # Mock sys.argv for argparse
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        # Verify successful execution
        assert exit_code == 0

        # Verify output file created
        assert output_file.exists(), "CSV output file not created"

        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(output_file)

        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
        assert "file" in df.columns
        assert "pred_class" in df.columns
        assert "prob_0" in df.columns
        assert "prob_1" in df.columns
        assert "prob_2" in df.columns
        assert "prob_3" in df.columns

        # Verify predictions are valid
        assert df["pred_class"].isin([0, 1, 2, 3]).all()

        # Verify probabilities sum to ~1.0
        prob_cols = ["prob_0", "prob_1", "prob_2", "prob_3"]
        prob_sums = df[prob_cols].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=0.01)

    def test_batch_inference_json_output(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test batch-inference with JSON output format."""
        from mammography.commands.batch_inference import main

        output_file = tmp_path / "results.json"

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--output-format", "json",
            "--batch-size", "4",
            "--device", "cpu",
            "--num-workers", "0",
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0
        assert output_file.exists(), "JSON output file not created"

        # Verify JSON content
        with open(output_file, 'r') as f:
            results = json.load(f)

        assert isinstance(results, list)
        assert len(results) == 10

        # Verify structure of first result
        first = results[0]
        assert "file" in first
        assert "pred_class" in first
        assert "prob_0" in first
        assert "prob_1" in first
        assert "prob_2" in first
        assert "prob_3" in first

    def test_batch_inference_jsonl_output(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test batch-inference with JSONL output format."""
        from mammography.commands.batch_inference import main

        output_file = tmp_path / "results.jsonl"

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--output-format", "jsonl",
            "--batch-size", "4",
            "--device", "cpu",
            "--num-workers", "0",
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0
        assert output_file.exists(), "JSONL output file not created"

        # Verify JSONL content (one JSON object per line)
        with open(output_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 10

        # Verify each line is valid JSON
        for line in lines:
            result = json.loads(line.strip())
            assert "file" in result
            assert "pred_class" in result

    def test_batch_inference_checkpoint_creation(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test that checkpoint files are created during processing."""
        from mammography.commands.batch_inference import main

        output_file = tmp_path / "results.csv"
        checkpoint_file = tmp_path / "batch_checkpoint.json"

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--output-format", "csv",
            "--batch-size", "2",
            "--device", "cpu",
            "--num-workers", "0",
            "--checkpoint-interval", "2",  # Save checkpoint every 2 batches
            "--checkpoint-file", str(checkpoint_file),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0

        # Verify checkpoint file created
        assert checkpoint_file.exists(), "Checkpoint file not created"

        # Verify checkpoint structure
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        assert "processed_files" in checkpoint
        assert "results" in checkpoint
        assert "timestamp" in checkpoint
        assert "num_processed" in checkpoint

        assert len(checkpoint["processed_files"]) == 10
        assert len(checkpoint["results"]) == 10
        assert checkpoint["num_processed"] == 10

    def test_batch_inference_resume_from_checkpoint(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test resume functionality from checkpoint."""
        from mammography.commands.batch_inference import main, save_checkpoint

        output_file = tmp_path / "results.csv"
        checkpoint_file = tmp_path / "batch_checkpoint.json"

        # Create a partial checkpoint (first 5 files processed)
        test_files = sorted(sample_test_images.glob("*.png"))
        processed_files = [str(f) for f in test_files[:5]]

        # Create dummy results for first 5 files
        partial_results = []
        for f in processed_files:
            partial_results.append({
                "file": str(f),
                "pred_class": 0,
                "prob_0": 0.7,
                "prob_1": 0.1,
                "prob_2": 0.1,
                "prob_3": 0.1,
            })

        save_checkpoint(str(checkpoint_file), processed_files, partial_results)

        # Run batch inference with --resume
        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--output-format", "csv",
            "--batch-size", "4",
            "--device", "cpu",
            "--num-workers", "0",
            "--resume",
            "--checkpoint-file", str(checkpoint_file),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0

        # Verify output file has all 10 results
        import pandas as pd
        df = pd.read_csv(output_file)
        assert len(df) == 10, f"Expected 10 rows after resume, got {len(df)}"

        # Verify checkpoint updated
        with open(checkpoint_file, 'r') as f:
            final_checkpoint = json.load(f)

        assert len(final_checkpoint["processed_files"]) == 10
        assert len(final_checkpoint["results"]) == 10

    def test_batch_inference_with_amp(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test batch-inference with AMP (Automatic Mixed Precision)."""
        from mammography.commands.batch_inference import main

        output_file = tmp_path / "results.csv"

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--output-format", "csv",
            "--batch-size", "4",
            "--device", "cpu",
            "--num-workers", "0",
            "--amp",  # Enable AMP
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0
        assert output_file.exists()

        import pandas as pd
        df = pd.read_csv(output_file)
        assert len(df) == 10

    def test_batch_inference_single_image(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test batch-inference on a single image file."""
        from mammography.commands.batch_inference import main

        output_file = tmp_path / "results.csv"

        # Select a single image
        test_files = sorted(sample_test_images.glob("*.png"))
        single_image = test_files[0]

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(single_image),
            "--output", str(output_file),
            "--output-format", "csv",
            "--device", "cpu",
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0
        assert output_file.exists()

        import pandas as pd
        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert single_image.name in str(df["file"].iloc[0])


@pytest.mark.unit
class TestBatchInferenceHelpers:
    """Unit tests for batch-inference helper functions."""

    def test_iter_inputs_directory(self, sample_test_images):
        """Test _iter_inputs with directory of images."""
        from mammography.commands.batch_inference import _iter_inputs

        files = _iter_inputs(str(sample_test_images))
        assert len(files) == 10
        assert all(f.endswith(".png") for f in files)
        # Verify sorted order
        assert files == sorted(files)

    def test_iter_inputs_single_file(self, sample_test_images):
        """Test _iter_inputs with a single file."""
        from mammography.commands.batch_inference import _iter_inputs

        test_files = sorted(sample_test_images.glob("*.png"))
        single_file = test_files[0]

        files = _iter_inputs(str(single_file))
        assert len(files) == 1
        assert files[0] == str(single_file)

    def test_strip_module_prefix(self):
        """Test _strip_module_prefix helper function."""
        from mammography.commands.batch_inference import _strip_module_prefix

        # Test with module prefix
        state_dict = {
            "module.conv1.weight": torch.randn(3, 3),
            "module.conv1.bias": torch.randn(3),
            "module.fc.weight": torch.randn(10, 10),
        }

        stripped = _strip_module_prefix(state_dict)
        assert "conv1.weight" in stripped
        assert "conv1.bias" in stripped
        assert "fc.weight" in stripped
        assert "module.conv1.weight" not in stripped

        # Test without module prefix
        state_dict_no_prefix = {
            "conv1.weight": torch.randn(3, 3),
            "fc.weight": torch.randn(10, 10),
        }

        not_stripped = _strip_module_prefix(state_dict_no_prefix)
        assert not_stripped == state_dict_no_prefix

    def test_save_checkpoint(self, tmp_path):
        """Test save_checkpoint helper function."""
        from mammography.commands.batch_inference import save_checkpoint

        checkpoint_file = tmp_path / "test_checkpoint.json"
        processed_files = ["file1.png", "file2.png", "file3.png"]
        results = [
            {"file": "file1.png", "pred_class": 0},
            {"file": "file2.png", "pred_class": 1},
            {"file": "file3.png", "pred_class": 2},
        ]

        save_checkpoint(str(checkpoint_file), processed_files, results)

        # Verify checkpoint created
        assert checkpoint_file.exists()

        # Verify content
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        assert checkpoint["processed_files"] == processed_files
        assert checkpoint["results"] == results
        assert checkpoint["num_processed"] == 3
        assert "timestamp" in checkpoint


@pytest.mark.unit
class TestBatchInferenceEdgeCases:
    """Edge case and boundary condition tests for batch-inference."""

    def test_parse_args_with_custom_normalization(self):
        """Test parse_args with custom mean and std values."""
        from mammography.commands.batch_inference import parse_args

        args = parse_args([
            "--checkpoint", "model.pt",
            "--input", "images/",
            "--output", "results.csv",
            "--mean", "0.5,0.5,0.5",
            "--std", "0.2,0.2,0.2",
        ])

        assert args.checkpoint == "model.pt"
        assert args.mean == "0.5,0.5,0.5"
        assert args.std == "0.2,0.2,0.2"

    def test_parse_args_all_output_formats(self):
        """Test parse_args with all supported output formats."""
        from mammography.commands.batch_inference import parse_args

        for fmt in ["csv", "json", "jsonl"]:
            args = parse_args([
                "--checkpoint", "model.pt",
                "--input", "images/",
                "--output", f"results.{fmt}",
                "--output-format", fmt,
            ])
            assert args.output_format == fmt

    def test_parse_args_checkpoint_interval_zero_disables(self):
        """Test that checkpoint_interval=0 disables checkpointing."""
        from mammography.commands.batch_inference import parse_args

        args = parse_args([
            "--checkpoint", "model.pt",
            "--input", "images/",
            "--output", "results.csv",
            "--checkpoint-interval", "0",
        ])

        assert args.checkpoint_interval == 0

    def test_resolve_loader_runtime_mps_device(self):
        """Test resolve_loader_runtime adjustments for MPS device."""
        from mammography.commands.batch_inference import resolve_loader_runtime
        from argparse import Namespace

        args = Namespace(
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )

        # MPS doesn't support multiprocessing well
        device = torch.device("mps")
        num_workers, _prefetch, persistent, _pin = resolve_loader_runtime(args, device)

        assert num_workers == 0
        assert persistent is False

    def test_resolve_loader_runtime_cpu_device(self):
        """Test resolve_loader_runtime adjustments for CPU device."""
        from mammography.commands.batch_inference import resolve_loader_runtime
        from argparse import Namespace
        import os

        args = Namespace(
            num_workers=100,  # Unreasonably high
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )

        device = torch.device("cpu")
        num_workers, _prefetch, _persistent, _pin = resolve_loader_runtime(args, device)

        # Should cap at CPU count
        assert num_workers <= (os.cpu_count() or 0)

    def test_iter_inputs_with_mixed_extensions(self, tmp_path: Path):
        """Test _iter_inputs with mixed image extensions."""
        from mammography.commands.batch_inference import _iter_inputs

        test_dir = tmp_path / "images"
        test_dir.mkdir()

        # Create files with different extensions
        (test_dir / "image1.png").write_text("fake")
        (test_dir / "image2.jpg").write_text("fake")
        (test_dir / "image3.jpeg").write_text("fake")
        (test_dir / "image4.dcm").write_text("fake")
        (test_dir / "image5.dicom").write_text("fake")
        (test_dir / "not_image.txt").write_text("fake")  # Should be ignored

        files = _iter_inputs(str(test_dir))

        assert len(files) == 5
        assert all(
            f.endswith((".png", ".jpg", ".jpeg", ".dcm", ".dicom"))
            for f in files
        )

    def test_iter_inputs_nested_directories(self, tmp_path: Path):
        """Test _iter_inputs recursively searches nested directories."""
        from mammography.commands.batch_inference import _iter_inputs

        # Create nested structure
        nested_dir = tmp_path / "root" / "sub1" / "sub2"
        nested_dir.mkdir(parents=True)

        (tmp_path / "root" / "image1.png").write_text("fake")
        (tmp_path / "root" / "sub1" / "image2.png").write_text("fake")
        (nested_dir / "image3.png").write_text("fake")

        files = _iter_inputs(str(tmp_path / "root"))

        assert len(files) == 3
        assert files == sorted(files)  # Should be sorted

    def test_strip_module_prefix_mixed_keys(self):
        """Test _strip_module_prefix with mixed prefix and non-prefix keys."""
        from mammography.commands.batch_inference import _strip_module_prefix

        # Mixed keys (some with prefix, some without)
        state_dict = {
            "conv1.weight": torch.randn(3, 3),
            "module.conv2.weight": torch.randn(3, 3),
        }

        # Should return unchanged when not all keys have prefix
        result = _strip_module_prefix(state_dict)
        assert result == state_dict

    def test_strip_module_prefix_empty_dict(self):
        """Test _strip_module_prefix with empty dictionary."""
        from mammography.commands.batch_inference import _strip_module_prefix

        result = _strip_module_prefix({})
        assert result == {}

    def test_save_checkpoint_atomic_write(self, tmp_path: Path):
        """Test that save_checkpoint uses atomic write (via temp file)."""
        from mammography.commands.batch_inference import save_checkpoint

        checkpoint_file = tmp_path / "checkpoint.json"
        processed = ["file1.png", "file2.png"]
        results = [{"file": "file1.png", "pred": 0}]

        save_checkpoint(str(checkpoint_file), processed, results)

        # Verify checkpoint exists and is valid JSON
        assert checkpoint_file.exists()
        with open(checkpoint_file) as f:
            data = json.load(f)
        assert data["processed_files"] == processed

        # Verify no .tmp file left behind
        tmp_file = Path(str(checkpoint_file) + ".tmp")
        assert not tmp_file.exists()

    def test_batch_inference_empty_directory(self, dummy_checkpoint, tmp_path):
        """Test batch-inference on empty directory raises error."""
        from mammography.commands.batch_inference import main

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        output_file = tmp_path / "results.csv"

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(empty_dir),
            "--output", str(output_file),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            with pytest.raises(SystemExit, match="No image files found"):
                main()

    def test_batch_inference_invalid_config_raises(self, tmp_path):
        """Test batch-inference with invalid config raises ValidationError."""
        from mammography.commands.batch_inference import main

        # Checkpoint doesn't exist
        output_file = tmp_path / "results.csv"

        args = [
            "--checkpoint", "/nonexistent/model.pt",
            "--input", str(tmp_path),
            "--output", str(output_file),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            with pytest.raises(SystemExit, match="Invalid configuration"):
                main()

    def test_batch_inference_creates_output_directory(self, dummy_checkpoint, sample_test_images, tmp_path):
        """Test that batch-inference creates output directory if needed."""
        from mammography.commands.batch_inference import main

        # Output in non-existent nested directory
        output_file = tmp_path / "nested" / "dir" / "results.csv"

        args = [
            "--checkpoint", str(dummy_checkpoint),
            "--input", str(sample_test_images),
            "--output", str(output_file),
            "--batch-size", "4",
            "--device", "cpu",
            "--num-workers", "0",
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["batch_inference", *args])
            exit_code = main()

        assert exit_code == 0
        assert output_file.exists()
        assert output_file.parent.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])