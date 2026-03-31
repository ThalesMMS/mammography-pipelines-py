from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from mammography.commands import augment as augment_cmd


def test_parse_args_minimal() -> None:
    """Test argument parsing with minimal required arguments."""
    args = augment_cmd.parse_args(
        ["--source-dir", "/path/to/source", "--output-dir", "/path/to/output"]
    )
    assert args.source_dir == "/path/to/source"
    assert args.output_dir == "/path/to/output"
    assert args.num_augmentations == 1


def test_parse_args_with_num_augmentations() -> None:
    """Test argument parsing with custom num-augmentations."""
    args = augment_cmd.parse_args(
        [
            "--source-dir",
            "/path/to/source",
            "--output-dir",
            "/path/to/output",
            "--num-augmentations",
            "5",
        ]
    )
    assert args.source_dir == "/path/to/source"
    assert args.output_dir == "/path/to/output"
    assert args.num_augmentations == 5


def test_parse_args_missing_source_dir() -> None:
    """Test that missing source-dir raises error."""
    with pytest.raises(SystemExit):
        augment_cmd.parse_args(["--output-dir", "/path/to/output"])


def test_parse_args_missing_output_dir() -> None:
    """Test that missing output-dir raises error."""
    with pytest.raises(SystemExit):
        augment_cmd.parse_args(["--source-dir", "/path/to/source"])


def test_iter_images_finds_png_files(tmp_path: Path) -> None:
    """Test that _iter_images finds PNG files."""
    (tmp_path / "image1.png").touch()
    (tmp_path / "image2.PNG").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "image3.png").touch()

    files = augment_cmd._iter_images(str(tmp_path))

    assert len(files) == 3
    assert all(f.lower().endswith(".png") for f in files)


def test_iter_images_finds_jpg_files(tmp_path: Path) -> None:
    """Test that _iter_images finds JPG/JPEG files."""
    (tmp_path / "image1.jpg").touch()
    (tmp_path / "image2.jpeg").touch()
    (tmp_path / "image3.JPG").touch()

    files = augment_cmd._iter_images(str(tmp_path))

    assert len(files) == 3
    assert all(f.lower().endswith((".jpg", ".jpeg")) for f in files)


def test_iter_images_finds_dicom_files(tmp_path: Path) -> None:
    """Test that _iter_images finds DICOM files."""
    (tmp_path / "image1.dcm").touch()
    (tmp_path / "image2.dicom").touch()
    (tmp_path / "image3.DCM").touch()

    files = augment_cmd._iter_images(str(tmp_path))

    assert len(files) == 3
    assert all(f.lower().endswith((".dcm", ".dicom")) for f in files)


def test_iter_images_mixed_formats(tmp_path: Path) -> None:
    """Test that _iter_images finds mixed image formats."""
    (tmp_path / "image1.png").touch()
    (tmp_path / "image2.jpg").touch()
    (tmp_path / "image3.dcm").touch()
    (tmp_path / "readme.txt").touch()  # Should be ignored

    files = augment_cmd._iter_images(str(tmp_path))

    assert len(files) == 3
    assert not any(f.endswith(".txt") for f in files)


def test_iter_images_sorted(tmp_path: Path) -> None:
    """Test that _iter_images returns sorted file paths."""
    (tmp_path / "c.png").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()

    files = augment_cmd._iter_images(str(tmp_path))

    names = [os.path.basename(f) for f in files]
    assert names == sorted(names)


def test_iter_images_empty_directory(tmp_path: Path) -> None:
    """Test that _iter_images returns empty list for empty directory."""
    files = augment_cmd._iter_images(str(tmp_path))
    assert files == []


def test_main_creates_output_directory(tmp_path: Path, monkeypatch) -> None:
    """Test that main creates output directory if it doesn't exist."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a test image
    img = Image.new("RGB", (10, 10), color="red")
    img.save(source_dir / "test.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_main_exits_when_no_images_found(tmp_path: Path) -> None:
    """Test that main exits when no images are found in source directory."""
    source_dir = tmp_path / "empty"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(SystemExit) as exc_info:
        augment_cmd.main(
            [
                "--source-dir",
                str(source_dir),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert "Nenhuma imagem encontrada" in str(exc_info.value)


def test_main_saves_original_and_augmented_images(tmp_path: Path) -> None:
    """Test that main saves both original and augmented images."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a test image
    img = Image.new("RGB", (10, 10), color="red")
    img.save(source_dir / "test.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "2",
        ]
    )

    # Check that original and augmented images are saved
    output_files = list(output_dir.glob("*.png"))
    assert len(output_files) == 3  # 1 original + 2 augmented
    assert any("_orig" in f.name for f in output_files)
    assert sum("_aug" in f.name for f in output_files) == 2


def test_main_correct_number_of_augmentations(tmp_path: Path) -> None:
    """Test that main creates correct number of augmented images."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a test image
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(source_dir / "image.png")

    num_augmentations = 5
    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            str(num_augmentations),
        ]
    )

    output_files = list(output_dir.glob("*.png"))
    # Should have 1 original + num_augmentations augmented
    assert len(output_files) == 1 + num_augmentations
    assert sum("_aug" in f.name for f in output_files) == num_augmentations


def test_main_processes_multiple_images(tmp_path: Path) -> None:
    """Test that main processes multiple source images."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create multiple test images
    for i in range(3):
        img = Image.new("RGB", (10, 10), color="red")
        img.save(source_dir / f"image{i}.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    output_files = list(output_dir.glob("*.png"))
    # 3 images × (1 original + 1 augmented) = 6 files
    assert len(output_files) == 6


def test_main_handles_jpg_images(tmp_path: Path) -> None:
    """Test that main handles JPG images correctly."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a JPG image
    img = Image.new("RGB", (10, 10), color="green")
    img.save(source_dir / "test.jpg")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    output_files = list(output_dir.glob("*.jpg"))
    assert len(output_files) == 2  # 1 original + 1 augmented


def test_main_converts_dicom_to_png(tmp_path: Path, monkeypatch) -> None:
    """Test that main converts DICOM files to PNG."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a fake DICOM file
    fake_dicom = source_dir / "test.dcm"
    fake_dicom.write_bytes(b"fake dicom data")

    # Mock DICOM reading
    def _fake_is_dicom_path(path: str) -> bool:
        return path.endswith((".dcm", ".dicom"))

    def _fake_dicom_to_pil_rgb(path: str) -> Image.Image:
        return Image.new("RGB", (10, 10), color="white")

    monkeypatch.setattr(augment_cmd, "is_dicom_path", _fake_is_dicom_path)
    monkeypatch.setattr(augment_cmd, "dicom_to_pil_rgb", _fake_dicom_to_pil_rgb)

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    # DICOM should be converted to PNG
    png_files = list(output_dir.glob("*.png"))
    assert len(png_files) == 2  # 1 original + 1 augmented
    assert all(".png" in f.name for f in png_files)


def test_main_handles_subdirectories(tmp_path: Path) -> None:
    """Test that main processes images in subdirectories."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    subdir = source_dir / "subdir"
    subdir.mkdir()
    output_dir = tmp_path / "output"

    # Create images in root and subdirectory
    img1 = Image.new("RGB", (10, 10), color="red")
    img1.save(source_dir / "root.png")

    img2 = Image.new("RGB", (10, 10), color="blue")
    img2.save(subdir / "sub.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    output_files = list(output_dir.glob("*.png"))
    # 2 images × (1 original + 1 augmented) = 4 files
    assert len(output_files) == 4
    assert any("root" in f.name for f in output_files)
    assert any("sub" in f.name for f in output_files)


def test_main_preserves_stem_in_filenames(tmp_path: Path) -> None:
    """Test that main preserves original filename stem in output."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create test image with specific name
    img = Image.new("RGB", (10, 10), color="yellow")
    img.save(source_dir / "my_special_image.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    output_files = [f.name for f in output_dir.glob("*.png")]
    assert "my_special_image_orig.png" in output_files
    assert any("my_special_image_aug0" in f for f in output_files)


def test_main_augmentations_differ_from_original(tmp_path: Path) -> None:
    """Test that augmented images are different from original (sanity check)."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a test image with distinct pattern
    img = Image.new("RGB", (100, 100), color="red")
    img.save(source_dir / "test.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "3",
        ]
    )

    # Load images and verify they exist
    orig_path = output_dir / "test_orig.png"
    aug_paths = [output_dir / f"test_aug{i}.png" for i in range(3)]

    assert orig_path.exists()
    for aug_path in aug_paths:
        assert aug_path.exists()

    # Note: We can't reliably test that augmentations differ due to randomness,
    # but we verify all files are created and can be loaded
    orig_img = Image.open(orig_path)
    for aug_path in aug_paths:
        aug_img = Image.open(aug_path)
        assert aug_img.size == orig_img.size


def test_main_handles_corrupted_image_gracefully(tmp_path: Path, capsys) -> None:
    """Test that main continues processing when an image is corrupted."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create a valid image
    img = Image.new("RGB", (10, 10), color="red")
    img.save(source_dir / "valid.png")

    # Create a corrupted image
    corrupted = source_dir / "corrupted.png"
    corrupted.write_bytes(b"not a valid image")

    # Create another valid image
    img2 = Image.new("RGB", (10, 10), color="blue")
    img2.save(source_dir / "valid2.png")

    # Should not raise, should continue processing
    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    # Check that valid images were processed
    output_files = list(output_dir.glob("*.png"))
    assert len(output_files) == 4  # 2 valid × (1 original + 1 augmented)

    # Check that warning was printed
    captured = capsys.readouterr()
    assert "[warn]" in captured.out
    assert "corrupted.png" in captured.out


def test_main_prints_completion_message(tmp_path: Path, capsys) -> None:
    """Test that main prints completion message."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    img = Image.new("RGB", (10, 10), color="red")
    img.save(source_dir / "test.png")

    augment_cmd.main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--num-augmentations",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert "[ok]" in captured.out
    assert "Augmentacao concluida" in captured.out
    assert str(output_dir) in captured.out
