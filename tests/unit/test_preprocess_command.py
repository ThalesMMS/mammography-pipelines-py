"""
Unit tests for preprocess command module.

These tests validate the preprocessing command functionality including
argument parsing, image processing, normalization, and reporting.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from mammography.commands import preprocess


class TestParseArgs:
    """Test argument parsing for preprocess command."""

    def test_parse_args_required_arguments(self):
        """Test that required arguments are validated."""
        # Missing required arguments should fail
        with pytest.raises(SystemExit):
            preprocess.parse_args([])

        # With required arguments should succeed
        args = preprocess.parse_args(["--input", "input_dir", "--output", "output_dir"])
        assert args.input == "input_dir"
        assert args.output == "output_dir"

    def test_parse_args_defaults(self):
        """Test default argument values."""
        args = preprocess.parse_args(["--input", "in", "--output", "out"])

        assert args.normalize == "per-image"
        assert args.img_size == 512
        assert args.resize is True
        assert args.crop is False
        assert args.format == "png"
        assert args.preview is False
        assert args.preview_n == 8
        assert args.report is True
        assert args.border_removal is False

    def test_parse_args_normalization_choices(self):
        """Test normalization method choices."""
        # Valid choices
        for method in ["per-image", "per-dataset", "none"]:
            args = preprocess.parse_args(
                ["--input", "in", "--output", "out", "--normalize", method]
            )
            assert args.normalize == method

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            preprocess.parse_args(
                ["--input", "in", "--output", "out", "--normalize", "invalid"]
            )

    def test_parse_args_format_choices(self):
        """Test output format choices."""
        # Valid choices
        for fmt in ["png", "jpg", "keep"]:
            args = preprocess.parse_args(
                ["--input", "in", "--output", "out", "--format", fmt]
            )
            assert args.format == fmt

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            preprocess.parse_args(
                ["--input", "in", "--output", "out", "--format", "bmp"]
            )

    def test_parse_args_resize_flags(self):
        """Test resize enable/disable flags."""
        # Default is True
        args = preprocess.parse_args(["--input", "in", "--output", "out"])
        assert args.resize is True

        # Explicit enable
        args = preprocess.parse_args(["--input", "in", "--output", "out", "--resize"])
        assert args.resize is True

        # Disable
        args = preprocess.parse_args(
            ["--input", "in", "--output", "out", "--no-resize"]
        )
        assert args.resize is False

    def test_parse_args_report_flags(self):
        """Test report enable/disable flags."""
        # Default is True
        args = preprocess.parse_args(["--input", "in", "--output", "out"])
        assert args.report is True

        # Explicit enable
        args = preprocess.parse_args(["--input", "in", "--output", "out", "--report"])
        assert args.report is True

        # Disable
        args = preprocess.parse_args(
            ["--input", "in", "--output", "out", "--no-report"]
        )
        assert args.report is False

    def test_parse_args_numeric_values(self):
        """Test numeric argument values."""
        args = preprocess.parse_args(
            [
                "--input",
                "in",
                "--output",
                "out",
                "--img-size",
                "1024",
                "--preview-n",
                "16",
            ]
        )
        assert args.img_size == 1024
        assert args.preview_n == 16


class TestIterImages:
    """Test image file discovery."""

    def test_iter_images_empty_directory(self, tmp_path):
        """Test with empty directory."""
        images = preprocess._iter_images(tmp_path)
        assert images == []

    def test_iter_images_no_images(self, tmp_path):
        """Test with directory containing non-image files."""
        (tmp_path / "file.txt").write_text("test")
        (tmp_path / "data.json").write_text("{}")

        images = preprocess._iter_images(tmp_path)
        assert images == []

    def test_iter_images_finds_valid_formats(self, tmp_path):
        """Test that valid image formats are found."""
        # Create files with valid extensions
        valid_exts = [".png", ".jpg", ".jpeg", ".dcm", ".dicom"]
        for i, ext in enumerate(valid_exts):
            (tmp_path / f"image{i}{ext}").write_text("fake")

        images = preprocess._iter_images(tmp_path)
        assert len(images) == len(valid_exts)

    def test_iter_images_ignores_invalid_formats(self, tmp_path):
        """Test that invalid formats are ignored."""
        (tmp_path / "image.png").write_text("fake")
        (tmp_path / "image.txt").write_text("fake")
        (tmp_path / "image.bmp").write_text("fake")

        images = preprocess._iter_images(tmp_path)
        assert len(images) == 1
        assert images[0].suffix == ".png"

    def test_iter_images_recursive(self, tmp_path):
        """Test recursive directory traversal."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "image1.png").write_text("fake")
        (subdir / "image2.jpg").write_text("fake")

        images = preprocess._iter_images(tmp_path)
        assert len(images) == 2

    def test_iter_images_sorted(self, tmp_path):
        """Test that results are sorted."""
        # Create files in non-alphabetical order
        for name in ["c.png", "a.png", "b.png"]:
            (tmp_path / name).write_text("fake")

        images = preprocess._iter_images(tmp_path)
        names = [img.name for img in images]
        assert names == ["a.png", "b.png", "c.png"]


class TestLoadImage:
    """Test image loading functionality."""

    @pytest.fixture
    def sample_png(self, tmp_path):
        """Create a sample PNG image."""
        img = Image.new("L", (100, 100), color=128)
        path = tmp_path / "test.png"
        img.save(path)
        return path

    def test_load_image_png(self, sample_png):
        """Test loading PNG image."""
        img = preprocess._load_image(sample_png)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_image_converts_to_rgb(self, tmp_path):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        gray_img = Image.new("L", (100, 100), color=128)
        path = tmp_path / "gray.png"
        gray_img.save(path)

        # Load and verify RGB conversion
        img = preprocess._load_image(path)
        assert img.mode == "RGB"

    def test_load_image_nonexistent_file(self, tmp_path):
        """Test error handling for nonexistent files."""
        path = tmp_path / "nonexistent.png"
        with pytest.raises(RuntimeError, match="Failed to load image"):
            preprocess._load_image(path)

    @patch("mammography.commands.preprocess.is_dicom_path")
    @patch("mammography.commands.preprocess.dicom_to_pil_rgb")
    def test_load_image_dicom(self, mock_dicom_to_pil, mock_is_dicom, tmp_path):
        """Test DICOM image loading."""
        # Setup mocks
        mock_is_dicom.return_value = True
        mock_dicom_to_pil.return_value = Image.new("RGB", (100, 100))

        path = tmp_path / "test.dcm"
        path.write_text("fake dicom")

        # Load image
        img = preprocess._load_image(path)

        # Verify DICOM loading was called
        mock_is_dicom.assert_called_once()
        mock_dicom_to_pil.assert_called_once()
        assert isinstance(img, Image.Image)


class TestNormalizePil:
    """Test PIL image normalization."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for normalization."""
        # Create image with known values
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_normalize_none(self, sample_image):
        """Test that 'none' normalization returns image unchanged."""
        result = preprocess._normalize_pil(sample_image, "none")
        assert result is sample_image

    def test_normalize_per_image(self, sample_image):
        """Test per-image normalization."""
        result = preprocess._normalize_pil(sample_image, "per-image")

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

        # Check that result is in valid range
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255
        assert arr.dtype == np.uint8

    def test_normalize_per_image_constant_image(self):
        """Test per-image normalization with constant image."""
        # Create constant image
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)

        result = preprocess._normalize_pil(img, "per-image")

        # Should handle constant image gracefully
        assert isinstance(result, Image.Image)
        result_arr = np.array(result)
        assert result_arr.dtype == np.uint8

    def test_normalize_per_dataset_requires_stats(self, sample_image):
        """Test that per-dataset normalization requires statistics."""
        with pytest.raises(ValueError, match="Dataset statistics required"):
            preprocess._normalize_pil(sample_image, "per-dataset", stats=None)

    def test_normalize_per_dataset_with_stats(self, sample_image):
        """Test per-dataset normalization with statistics."""
        stats = {"mean": 127.5, "std": 50.0}
        result = preprocess._normalize_pil(sample_image, "per-dataset", stats=stats)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

        # Check valid range
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255
        assert arr.dtype == np.uint8

    def test_normalize_per_dataset_zero_std(self, sample_image):
        """Test per-dataset normalization with zero std."""
        stats = {"mean": 127.5, "std": 0.0}
        result = preprocess._normalize_pil(sample_image, "per-dataset", stats=stats)

        # Should handle zero std gracefully
        assert isinstance(result, Image.Image)
        result_arr = np.array(result)
        assert result_arr.dtype == np.uint8


class TestComputeDatasetStats:
    """Test dataset statistics computation."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample images for statistics."""
        paths = []
        for i in range(3):
            img = Image.new("L", (100, 100), color=i * 50)
            path = tmp_path / f"img{i}.png"
            img.save(path)
            paths.append(path)
        return paths

    def test_compute_dataset_stats_valid_images(self, sample_images):
        """Test statistics computation with valid images."""
        stats = preprocess._compute_dataset_stats(sample_images)

        assert "mean" in stats
        assert "std" in stats
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["std"], float)
        assert stats["mean"] >= 0
        assert stats["std"] >= 0

    def test_compute_dataset_stats_empty_list(self):
        """Test statistics computation with empty list."""
        stats = preprocess._compute_dataset_stats([])

        # Should return defaults
        assert stats["mean"] == 127.5
        assert stats["std"] == 1.0

    @patch("mammography.commands.preprocess._load_image")
    def test_compute_dataset_stats_handles_errors(self, mock_load, tmp_path):
        """Test that errors are handled gracefully."""
        # Create paths
        paths = [tmp_path / f"img{i}.png" for i in range(3)]

        # Mock to raise error for some images
        def side_effect(path):
            if "img1" in str(path):
                raise RuntimeError("Load failed")
            return Image.new("L", (100, 100), color=128)

        mock_load.side_effect = side_effect

        # Should not raise, but log warnings
        stats = preprocess._compute_dataset_stats(paths)
        assert "mean" in stats
        assert "std" in stats


class TestResizePil:
    """Test PIL image resizing."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for resizing."""
        return Image.new("RGB", (800, 600))

    def test_resize_without_crop_preserves_aspect_ratio(self, sample_image):
        """Test that resizing without crop preserves aspect ratio."""
        target_size = 512

        result = preprocess._resize_pil(sample_image, target_size, crop=False)

        # Longest side should be target_size
        assert max(result.size) == target_size

        # Aspect ratio should be preserved
        original_ratio = sample_image.width / sample_image.height
        result_ratio = result.width / result.height
        assert abs(original_ratio - result_ratio) < 1e-6

    def test_resize_with_crop_creates_square(self, sample_image):
        """Test that resizing with crop creates square image."""
        target_size = 512

        result = preprocess._resize_pil(sample_image, target_size, crop=True)

        # Should be exactly square
        assert result.width == target_size
        assert result.height == target_size

    def test_resize_landscape_image(self):
        """Test resizing landscape image."""
        img = Image.new("RGB", (1000, 500))
        target_size = 512

        result = preprocess._resize_pil(img, target_size, crop=False)

        # Width should be target_size (longest side)
        assert result.width == target_size
        assert result.height == 256  # Maintains 2:1 ratio

    def test_resize_portrait_image(self):
        """Test resizing portrait image."""
        img = Image.new("RGB", (500, 1000))
        target_size = 512

        result = preprocess._resize_pil(img, target_size, crop=False)

        # Height should be target_size (longest side)
        assert result.height == target_size
        assert result.width == 256  # Maintains 1:2 ratio

    def test_resize_square_image(self):
        """Test resizing square image."""
        img = Image.new("RGB", (1000, 1000))
        target_size = 512

        result = preprocess._resize_pil(img, target_size, crop=False)

        # Should remain square
        assert result.width == target_size
        assert result.height == target_size


class TestSavePreview:
    """Test preview grid generation."""

    def test_save_preview_empty_list(self, tmp_path):
        """Test with empty image list."""
        # Should not raise, just log warning
        preprocess._save_preview([], tmp_path, 8)

        # No preview file should be created
        assert not (tmp_path / "preview_grid.png").exists()

    def test_save_preview_creates_file(self, tmp_path):
        """Test that preview file is created."""
        images = [Image.new("RGB", (100, 100)) for _ in range(4)]

        preprocess._save_preview(images, tmp_path, 8)

        preview_path = tmp_path / "preview_grid.png"
        assert preview_path.exists()

        # Verify it's a valid image
        preview = Image.open(preview_path)
        assert preview.mode == "RGB"

    def test_save_preview_respects_max_n(self, tmp_path):
        """Test that preview respects maximum number of images."""
        images = [Image.new("RGB", (100, 100)) for _ in range(10)]

        preprocess._save_preview(images, tmp_path, n=4)

        # Should create preview with at most 4 images
        preview_path = tmp_path / "preview_grid.png"
        assert preview_path.exists()

    def test_save_preview_grid_dimensions(self, tmp_path):
        """Test that grid dimensions are correct."""
        images = [Image.new("RGB", (100, 100)) for _ in range(9)]

        preprocess._save_preview(images, tmp_path, 9)

        preview_path = tmp_path / "preview_grid.png"
        preview = Image.open(preview_path)

        # 9 images should create 3x3 grid
        expected_size = 3 * 100
        assert preview.width == expected_size
        assert preview.height == expected_size


class TestWriteReport:
    """Test report generation."""

    @pytest.fixture
    def sample_results(self):
        """Create sample processing results."""
        return [
            {
                "input_path": "img1.png",
                "output_path": "out/img1.png",
                "output_size": (512, 512),
                "status": "success",
            },
            {
                "input_path": "img2.png",
                "output_path": "out/img2.png",
                "output_size": (512, 512),
                "status": "success",
            },
            {
                "input_path": "img3.png",
                "status": "failed",
                "error": "Failed to load",
            },
        ]

    def test_write_report_creates_files(self, tmp_path, sample_results):
        """Test that report files are created."""
        stats = {"mean": 127.5, "std": 50.0}

        preprocess._write_report(sample_results, tmp_path, stats)

        # Check JSON report exists
        json_path = tmp_path / "preprocess_report.json"
        assert json_path.exists()

        # Check text report exists
        txt_path = tmp_path / "preprocess_report.txt"
        assert txt_path.exists()

    def test_write_report_json_content(self, tmp_path, sample_results):
        """Test JSON report content."""
        stats = {"mean": 127.5, "std": 50.0}

        preprocess._write_report(sample_results, tmp_path, stats)

        json_path = tmp_path / "preprocess_report.json"
        with open(json_path) as f:
            report = json.load(f)

        assert report["total_images"] == 3
        assert report["successful"] == 2
        assert report["failed"] == 1
        assert report["dataset_stats"] == stats
        assert len(report["results"]) == 3

    def test_write_report_text_content(self, tmp_path, sample_results):
        """Test text report content."""
        stats = {"mean": 127.5, "std": 50.0}

        preprocess._write_report(sample_results, tmp_path, stats)

        txt_path = tmp_path / "preprocess_report.txt"
        content = txt_path.read_text()

        assert "Total images: 3" in content
        assert "Successful: 2" in content
        assert "Failed: 1" in content
        assert "Mean: 127.50" in content
        assert "Std: 50.00" in content
        assert "Failed Images:" in content
        assert "img3.png" in content

    def test_write_report_without_stats(self, tmp_path, sample_results):
        """Test report generation without dataset statistics."""
        preprocess._write_report(sample_results, tmp_path, stats=None)

        json_path = tmp_path / "preprocess_report.json"
        with open(json_path) as f:
            report = json.load(f)

        assert report["dataset_stats"] is None


class TestMainFunction:
    """Test main entry point."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset for end-to-end testing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create sample images
        for i in range(3):
            img = Image.new("L", (800, 600), color=i * 80)
            img.save(input_dir / f"img{i}.png")

        return input_dir

    def test_main_with_valid_input(self, sample_dataset, tmp_path):
        """Test main function with valid input."""
        output_dir = tmp_path / "output"

        args = [
            "--input",
            str(sample_dataset),
            "--output",
            str(output_dir),
            "--normalize",
            "per-image",
            "--img-size",
            "256",
        ]

        # Should not raise
        preprocess.main(args)

        # Check output exists
        assert output_dir.exists()

        # Check that images were processed
        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) == 3

    def test_main_with_nonexistent_input(self, tmp_path):
        """Test main function with nonexistent input directory."""
        args = [
            "--input",
            str(tmp_path / "nonexistent"),
            "--output",
            str(tmp_path / "output"),
        ]

        with pytest.raises(SystemExit):
            preprocess.main(args)

    def test_main_with_empty_input(self, tmp_path):
        """Test main function with empty input directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        args = [
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path / "output"),
        ]

        with pytest.raises(SystemExit, match="No images found"):
            preprocess.main(args)

    def test_main_with_preview(self, sample_dataset, tmp_path):
        """Test main function with preview generation."""
        output_dir = tmp_path / "output"

        args = [
            "--input",
            str(sample_dataset),
            "--output",
            str(output_dir),
            "--preview",
            "--preview-n",
            "4",
        ]

        preprocess.main(args)

        # Check preview was created
        preview_path = output_dir / "preview_grid.png"
        assert preview_path.exists()

    def test_main_with_report(self, sample_dataset, tmp_path):
        """Test main function with report generation."""
        output_dir = tmp_path / "output"

        args = [
            "--input",
            str(sample_dataset),
            "--output",
            str(output_dir),
            "--report",
        ]

        preprocess.main(args)

        # Check reports were created
        assert (output_dir / "preprocess_report.json").exists()
        assert (output_dir / "preprocess_report.txt").exists()

    def test_main_with_per_dataset_normalization(self, sample_dataset, tmp_path):
        """Test main function with per-dataset normalization."""
        output_dir = tmp_path / "output"

        args = [
            "--input",
            str(sample_dataset),
            "--output",
            str(output_dir),
            "--normalize",
            "per-dataset",
        ]

        # Should compute stats and process
        preprocess.main(args)

        # Verify output
        assert output_dir.exists()
        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) == 3

    def test_main_with_crop(self, sample_dataset, tmp_path):
        """Test main function with center cropping."""
        output_dir = tmp_path / "output"

        args = [
            "--input",
            str(sample_dataset),
            "--output",
            str(output_dir),
            "--crop",
            "--img-size",
            "512",
        ]

        preprocess.main(args)

        # Verify all output images are square
        for img_path in output_dir.glob("*.png"):
            img = Image.open(img_path)
            assert img.width == img.height == 512

    def test_main_with_different_output_format(self, sample_dataset, tmp_path):
        """Test main function with JPEG output format."""
        output_dir = tmp_path / "output"

        args = [
            "--input",
            str(sample_dataset),
            "--output",
            str(output_dir),
            "--format",
            "jpg",
        ]

        preprocess.main(args)

        # Check JPEG files were created
        output_images = list(output_dir.glob("*.jpg"))
        assert len(output_images) == 3
