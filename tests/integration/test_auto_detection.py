"""
Integration tests for dataset format auto-detection using real dataset patterns.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.data.csv_loader import load_dataset_dataframe
from mammography.data.format_detection import (
    detect_dataset_format,
    suggest_preprocessing,
    validate_format,
)
from tests.utils.dataset_sampling import sample_dataframe


def _dataset_available(dataset_name: str) -> bool:
    """Check if a dataset is available for testing."""
    if dataset_name == "archive":
        return Path("classificacao.csv").exists() and Path("archive").exists()
    return Path(dataset_name).exists()


@pytest.mark.parametrize(
    ("dataset_name", "expected_type", "expected_image_format"),
    [
        ("archive", "archive", "dicom"),
        ("mamografias", "mamografias", "png"),
        ("patches_completo", "patches_completo", "png"),
    ],
)
def test_auto_detect_dataset_format(
    dataset_name: str, expected_type: str, expected_image_format: str
) -> None:
    """Test that auto-detection correctly identifies dataset type and image format."""
    if not _dataset_available(dataset_name):
        pytest.skip(f"Dataset '{dataset_name}' not available for auto-detection test")

    # Detect format
    fmt = detect_dataset_format(dataset_name)

    # Verify dataset type
    assert fmt.dataset_type == expected_type, (
        f"Expected dataset type '{expected_type}', "
        f"got '{fmt.dataset_type}'"
    )

    # Verify image format
    assert fmt.image_format == expected_image_format, (
        f"Expected image format '{expected_image_format}', "
        f"got '{fmt.image_format}'"
    )

    # Verify image count is positive
    assert fmt.image_count > 0, (
        f"Expected positive image count, got {fmt.image_count}"
    )

    # Verify metadata detection
    if expected_type in ("mamografias", "patches_completo"):
        assert fmt.has_features_txt, (
            f"Expected featureS.txt detection for {expected_type}"
        )
        assert fmt.csv_path is not None, (
            f"Expected csv_path to be set for {expected_type}"
        )
    elif expected_type == "archive":
        assert fmt.dicom_root is not None, (
            f"Expected dicom_root to be set for {expected_type}"
        )


@pytest.mark.parametrize(
    "dataset_name",
    [
        "archive",
        "mamografias",
        "patches_completo",
    ],
)
def test_auto_detection_with_load_dataset(dataset_name: str) -> None:
    """Test that auto-detection works through load_dataset_dataframe()."""
    if not _dataset_available(dataset_name):
        pytest.skip(f"Dataset '{dataset_name}' not available for integration test")

    # Load dataset with auto-detection enabled
    df = load_dataset_dataframe(
        csv_path=dataset_name,
        dicom_root=None,
        dataset=None,
        auto_detect=True,
    )

    # Verify DataFrame was loaded
    assert df is not None
    assert len(df) > 0

    # Verify required columns
    assert "image_path" in df.columns
    assert "professional_label" in df.columns
    assert "accession" in df.columns
    assert "view" in df.columns

    # Sample and verify paths exist
    sample = sample_dataframe(df)
    assert len(sample) > 0

    for path in sample["image_path"]:
        img_path = Path(path)
        assert img_path.exists(), f"Image path does not exist: {path}"


@pytest.mark.parametrize(
    "dataset_name",
    [
        "archive",
        "mamografias",
        "patches_completo",
    ],
)
def test_validation_warnings(dataset_name: str) -> None:
    """Test that validation warnings are generated appropriately."""
    if not _dataset_available(dataset_name):
        pytest.skip(f"Dataset '{dataset_name}' not available for validation test")

    # Detect format
    fmt = detect_dataset_format(dataset_name)

    # Get validation warnings
    warnings = validate_format(fmt)

    # Warnings should be a list
    assert isinstance(warnings, list)

    # For known datasets with proper structure, warnings should be minimal
    # We expect custom dataset warnings to be absent for known presets
    custom_warning_found = any("Custom dataset" in w for w in warnings)
    if fmt.dataset_type in ("archive", "mamografias", "patches_completo"):
        assert not custom_warning_found, (
            f"Known preset {dataset_name} should not trigger custom dataset warning"
        )


@pytest.mark.parametrize(
    "dataset_name",
    [
        "archive",
        "mamografias",
        "patches_completo",
    ],
)
def test_preprocessing_suggestions(dataset_name: str) -> None:
    """Test that preprocessing suggestions are generated based on format."""
    if not _dataset_available(dataset_name):
        pytest.skip(f"Dataset '{dataset_name}' not available for suggestions test")

    # Detect format
    fmt = detect_dataset_format(dataset_name)

    # Get preprocessing suggestions
    suggestions = suggest_preprocessing(fmt)

    # Suggestions should be a list
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0, "Expected at least some preprocessing suggestions"

    # Verify format-specific suggestions
    if fmt.image_format == "dicom":
        assert any("DICOM" in s for s in suggestions), (
            "Expected DICOM-specific suggestions for DICOM format"
        )
    elif fmt.image_format in ("png", "jpg"):
        assert any(
            fmt.image_format.upper() in s or "PNG" in s or "JPG" in s
            for s in suggestions
        ), f"Expected {fmt.image_format.upper()}-specific suggestions"


def test_auto_detect_vs_explicit_params() -> None:
    """Test that explicit parameters override auto-detection."""
    dataset_name = "mamografias"
    if not _dataset_available(dataset_name):
        pytest.skip(f"Dataset '{dataset_name}' not available for comparison test")

    # Load with auto-detection
    df_auto = load_dataset_dataframe(
        csv_path=dataset_name,
        dicom_root=None,
        dataset=None,
        auto_detect=True,
    )

    # Load with explicit preset (traditional method)
    df_preset = load_dataset_dataframe(
        csv_path=None,
        dicom_root=None,
        dataset=dataset_name,
        auto_detect=False,
    )

    # Both should produce similar results
    assert len(df_auto) == len(df_preset), (
        "Auto-detection and preset should produce same row count"
    )

    # Check that image paths are similar (may not be identical order)
    auto_paths = set(df_auto["image_path"])
    preset_paths = set(df_preset["image_path"])
    assert auto_paths == preset_paths, (
        "Auto-detection and preset should produce same image paths"
    )


def test_auto_detect_nonexistent_directory() -> None:
    """Test that auto-detection handles nonexistent directories gracefully."""
    nonexistent_path = "nonexistent_dataset_xyz123"

    # Should raise ValueError for nonexistent path
    with pytest.raises(ValueError, match="Path does not exist"):
        detect_dataset_format(nonexistent_path)


def test_auto_detect_empty_directory(tmp_path: Path) -> None:
    """Test that auto-detection handles empty directories."""
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()

    # Detect format on empty directory
    fmt = detect_dataset_format(str(empty_dir))

    # Should detect as custom with warnings
    assert fmt.dataset_type == "custom"
    assert fmt.image_count == 0

    # Should have validation warnings
    warnings = validate_format(fmt)
    assert len(warnings) > 0
    assert any("empty" in w.lower() for w in warnings)


def test_backward_compatibility_with_presets() -> None:
    """Test that existing preset-based loading still works (backward compatibility)."""
    # Test that all known presets can still be loaded without auto-detection
    for dataset_name in ("archive", "mamografias", "patches_completo"):
        if not _dataset_available(dataset_name):
            continue

        # Load using traditional preset method (auto_detect=False)
        df = load_dataset_dataframe(
            csv_path=None,
            dicom_root=None,
            dataset=dataset_name,
            auto_detect=False,
        )

        # Should still work
        assert df is not None
        assert len(df) > 0
        assert "image_path" in df.columns
