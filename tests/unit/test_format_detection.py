from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

Image = pytest.importorskip("PIL.Image")
pd = pytest.importorskip("pandas")

from mammography.data.format_detection import (
    CSVSchemaInfo,
    DatasetFormat,
    detect_dataset_format,
    detect_image_format,
    infer_csv_schema,
    suggest_preprocessing,
    validate_format,
)


def _write_sample_image(path: Path, image_format: str = "png") -> None:
    """Helper to write sample images in various formats."""
    img = Image.new("RGB", (16, 16), color=(120, 30, 60))
    if image_format == "png":
        img.save(path)
    elif image_format == "jpg":
        img.save(path, format="JPEG")


def _write_fake_dicom(path: Path) -> None:
    """Helper to write fake DICOM file with signature."""
    # DICOM signature is "DICM" at offset 128
    with open(path, "wb") as f:
        f.write(b"\x00" * 128)  # 128 bytes of padding
        f.write(b"DICM")  # DICOM signature
        f.write(b"\x00" * 100)  # Additional padding


# Tests for detect_image_format()


def test_detect_image_format_png_by_extension(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    _write_sample_image(img_path, "png")

    result = detect_image_format(str(img_path), check_signature=False)
    assert result == "png"


def test_detect_image_format_jpg_by_extension(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    _write_sample_image(img_path, "jpg")

    result = detect_image_format(str(img_path), check_signature=False)
    assert result == "jpg"


def test_detect_image_format_jpeg_by_extension(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpeg"
    _write_sample_image(img_path, "jpg")

    result = detect_image_format(str(img_path), check_signature=False)
    assert result == "jpg"


def test_detect_image_format_dicom_by_extension(tmp_path: Path) -> None:
    dcm_path = tmp_path / "sample.dcm"
    _write_fake_dicom(dcm_path)

    result = detect_image_format(str(dcm_path), check_signature=False)
    assert result == "dicom"


def test_detect_image_format_unknown_extension() -> None:
    result = detect_image_format("sample.txt", check_signature=False)
    assert result == "unknown"


def test_detect_image_format_empty_path() -> None:
    result = detect_image_format("", check_signature=False)
    assert result == "unknown"


def test_detect_image_format_png_with_signature(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    _write_sample_image(img_path, "png")

    result = detect_image_format(str(img_path), check_signature=True)
    assert result == "png"


def test_detect_image_format_jpg_with_signature(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    _write_sample_image(img_path, "jpg")

    result = detect_image_format(str(img_path), check_signature=True)
    assert result == "jpg"


def test_detect_image_format_dicom_with_signature(tmp_path: Path) -> None:
    dcm_path = tmp_path / "sample.dcm"
    _write_fake_dicom(dcm_path)

    result = detect_image_format(str(dcm_path), check_signature=True)
    assert result == "dicom"


def test_detect_image_format_nonexistent_file_with_signature() -> None:
    result = detect_image_format("/nonexistent/file.png", check_signature=True)
    assert result == "png"  # Falls back to extension-based


def test_detect_image_format_case_insensitive(tmp_path: Path) -> None:
    img_path = tmp_path / "SAMPLE.PNG"
    _write_sample_image(img_path, "png")

    result = detect_image_format(str(img_path), check_signature=False)
    assert result == "png"


# Tests for infer_csv_schema()


def test_infer_csv_schema_classification(tmp_path: Path) -> None:
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n"
        "ACC001,2\n"
        "ACC002,3\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "comma"
    assert schema.schema_type == "classification"
    assert "AccessionNumber" in schema.columns
    assert "Classification" in schema.columns
    assert schema.row_count == 2
    assert schema.has_header is True
    assert schema.encoding == "utf-8"


def test_infer_csv_schema_raw_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "image_path,professional_label\n"
        "/path/to/img1.png,2\n"
        "/path/to/img2.png,3\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "comma"
    assert schema.schema_type == "raw_path"
    assert "image_path" in schema.columns
    assert "professional_label" in schema.columns
    assert schema.row_count == 2


def test_infer_csv_schema_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "image_path,professional_label,accession,view\n"
        "/path/img1.png,2,ACC001,CC\n"
        "/path/img2.png,3,ACC002,MLO\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "comma"
    assert schema.schema_type == "dataset"
    assert len(schema.columns) == 4


def test_infer_csv_schema_tab_delimited(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.tsv"
    csv_path.write_text(
        "image_path\tprofessional_label\n"
        "/path/img1.png\t2\n"
        "/path/img2.png\t3\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "tab"
    assert schema.schema_type == "raw_path"


def test_infer_csv_schema_custom(tmp_path: Path) -> None:
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text(
        "filename,category\n"
        "img1.png,cat_a\n"
        "img2.png,cat_b\n",
        encoding="utf-8",
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "comma"
    assert schema.schema_type in ("custom", "unknown")
    assert len(schema.warnings) > 0


def test_infer_csv_schema_nonexistent_file() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        infer_csv_schema("/nonexistent/file.csv")


def test_infer_csv_schema_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        infer_csv_schema(str(tmp_path))


def test_infer_csv_schema_invalid_delimiter(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("no delimiters here\njust plain text\n", encoding="utf-8")

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "unknown"
    assert schema.schema_type == "unknown"
    assert len(schema.warnings) > 0


def test_infer_csv_schema_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "unknown"


def test_infer_csv_schema_latin1_encoding(tmp_path: Path) -> None:
    csv_path = tmp_path / "latin1.csv"
    # Write file with latin-1 specific character
    csv_path.write_bytes(
        b"image_path,professional_label\n"
        b"/path/img\xe9.png,2\n"  # é in latin-1
    )

    schema = infer_csv_schema(str(csv_path))
    assert schema.delimiter == "comma"
    assert schema.encoding in ("utf-8", "latin-1")


# Tests for detect_dataset_format()


def test_detect_dataset_format_mamografias(tmp_path: Path) -> None:
    # Create mamografias-style structure
    case_folder = tmp_path / "mamografias" / "case_001"
    case_folder.mkdir(parents=True)

    img_path = case_folder / "img_001.png"
    _write_sample_image(img_path, "png")

    (case_folder / "featureS.txt").write_text("img_001\n1\n", encoding="utf-8")

    fmt = detect_dataset_format(str(tmp_path / "mamografias"))
    assert fmt.dataset_type == "mamografias"
    assert fmt.image_format == "png"
    assert fmt.has_features_txt is True
    assert fmt.image_count == 1


def test_detect_dataset_format_patches_completo(tmp_path: Path) -> None:
    # Create patches_completo-style structure
    patches_folder = tmp_path / "patches_completo"
    patches_folder.mkdir()

    for i in range(3):
        img_path = patches_folder / f"patch_{i}.png"
        _write_sample_image(img_path, "png")

    (patches_folder / "featureS.txt").write_text(
        "patch_0\n1\npatch_1\n2\npatch_2\n3\n", encoding="utf-8"
    )

    fmt = detect_dataset_format(str(patches_folder))
    assert fmt.dataset_type == "patches_completo"
    assert fmt.image_format == "png"
    assert fmt.has_features_txt is True
    assert fmt.image_count == 3


def test_detect_dataset_format_archive(tmp_path: Path) -> None:
    # Create archive-style structure
    archive_folder = tmp_path / "archive"
    archive_folder.mkdir()

    acc_folder = archive_folder / "ACC001"
    acc_folder.mkdir()

    dcm_path = acc_folder / "image_001.dcm"
    _write_fake_dicom(dcm_path)

    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\n" "ACC001,2\n", encoding="utf-8"
    )

    fmt = detect_dataset_format(str(tmp_path))
    assert fmt.dataset_type == "archive"
    assert fmt.image_format == "dicom"
    assert fmt.has_csv is True
    assert fmt.dicom_root is not None
    assert "archive" in fmt.dicom_root


def test_detect_dataset_format_custom_csv(tmp_path: Path) -> None:
    # Create custom dataset with CSV
    for i in range(2):
        img_path = tmp_path / f"image_{i}.png"
        _write_sample_image(img_path, "png")

    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "image_path,professional_label,accession\n"
        f"{tmp_path / 'image_0.png'},2,ACC001\n"
        f"{tmp_path / 'image_1.png'},3,ACC002\n",
        encoding="utf-8",
    )

    fmt = detect_dataset_format(str(tmp_path))
    assert fmt.dataset_type == "custom"
    assert fmt.image_format == "png"
    assert fmt.has_csv is True
    assert fmt.image_count == 2


def test_detect_dataset_format_mixed_formats(tmp_path: Path) -> None:
    # Create dataset with mixed formats
    _write_sample_image(tmp_path / "img1.png", "png")
    _write_sample_image(tmp_path / "img2.png", "png")
    _write_sample_image(tmp_path / "img3.jpg", "jpg")

    fmt = detect_dataset_format(str(tmp_path))
    assert fmt.image_format in ("png", "mixed")
    assert fmt.image_count == 3
    assert ".png" in fmt.format_counts
    assert ".jpg" in fmt.format_counts


def test_detect_dataset_format_empty_directory(tmp_path: Path) -> None:
    fmt = detect_dataset_format(str(tmp_path))
    assert fmt.image_count == 0
    assert len(fmt.warnings) > 0
    assert any("No image files found" in w for w in fmt.warnings)


def test_detect_dataset_format_nonexistent_path() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        detect_dataset_format("/nonexistent/path")


def test_detect_dataset_format_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        detect_dataset_format(str(file_path))


def test_detect_dataset_format_no_metadata(tmp_path: Path) -> None:
    # Dataset with images but no metadata
    _write_sample_image(tmp_path / "img1.png", "png")
    _write_sample_image(tmp_path / "img2.png", "png")

    fmt = detect_dataset_format(str(tmp_path))
    assert fmt.csv_path is None
    assert not fmt.has_csv
    assert not fmt.has_features_txt
    assert any("No metadata file detected" in w for w in fmt.warnings)


# Tests for validate_format()


def test_validate_format_valid_dataset() -> None:
    fmt = DatasetFormat(
        dataset_type="mamografias",
        image_format="png",
        csv_path="/path/to/features",
        has_features_txt=True,
        image_count=100,
        format_counts={".png": 100},
    )

    warnings = validate_format(fmt)
    # Should have minimal or no warnings for valid dataset
    assert isinstance(warnings, list)


def test_validate_format_empty_dataset() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="unknown",
        image_count=0,
    )

    warnings = validate_format(fmt)
    assert len(warnings) > 0
    assert any("No images found" in w or "empty" in w.lower() for w in warnings)


def test_validate_format_missing_metadata() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="png",
        image_count=50,
        format_counts={".png": 50},
        csv_path=None,
        has_features_txt=False,
    )

    warnings = validate_format(fmt)
    assert any("metadata" in w.lower() for w in warnings)


def test_validate_format_mixed_formats() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="mixed",
        image_count=100,
        format_counts={".png": 60, ".jpg": 40},
        warnings=["Mixed image formats detected: .png (60.0%)"],
    )

    warnings = validate_format(fmt)
    assert any("mixed" in w.lower() for w in warnings)


def test_validate_format_unknown_format() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="unknown",
        image_count=10,
    )

    warnings = validate_format(fmt)
    assert any("unknown" in w.lower() or "could not determine" in w.lower() for w in warnings)


def test_validate_format_small_dataset() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="png",
        image_count=5,
        format_counts={".png": 5},
    )

    warnings = validate_format(fmt)
    assert any("small" in w.lower() or "5 images" in w for w in warnings)


def test_validate_format_dicom_without_root() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="dicom",
        image_count=50,
        format_counts={".dcm": 50},
        dicom_root=None,
    )

    warnings = validate_format(fmt)
    assert any("dicom" in w.lower() and "root" in w.lower() for w in warnings)


def test_validate_format_custom_dataset_type() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="png",
        image_count=100,
        format_counts={".png": 100},
        csv_path="/path/to/data.csv",
        has_csv=True,
    )

    warnings = validate_format(fmt)
    assert any("custom" in w.lower() for w in warnings)


# Tests for suggest_preprocessing()


def test_suggest_preprocessing_dicom() -> None:
    fmt = DatasetFormat(
        dataset_type="archive",
        image_format="dicom",
        image_count=100,
        format_counts={".dcm": 100},
        dicom_root="/path/to/archive",
    )

    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
    assert any("DICOM" in s for s in suggestions)
    assert any("window" in s.lower() or "normalization" in s.lower() for s in suggestions)


def test_suggest_preprocessing_png() -> None:
    fmt = DatasetFormat(
        dataset_type="mamografias",
        image_format="png",
        image_count=500,
        format_counts={".png": 500},
        has_features_txt=True,
    )

    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
    assert any("PNG" in s for s in suggestions)


def test_suggest_preprocessing_jpg() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="jpg",
        image_count=200,
        format_counts={".jpg": 200},
    )

    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
    assert any("JPG" in s or "JPEG" in s for s in suggestions)


def test_suggest_preprocessing_mixed_formats() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="mixed",
        image_count=100,
        format_counts={".png": 60, ".jpg": 40},
    )

    suggestions = suggest_preprocessing(fmt)
    assert any("mixed" in s.lower() for s in suggestions)


def test_suggest_preprocessing_small_dataset() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="png",
        image_count=50,
        format_counts={".png": 50},
    )

    suggestions = suggest_preprocessing(fmt)
    assert any("small" in s.lower() or "augmentation" in s.lower() for s in suggestions)


def test_suggest_preprocessing_large_dataset() -> None:
    fmt = DatasetFormat(
        dataset_type="archive",
        image_format="dicom",
        image_count=10000,
        format_counts={".dcm": 10000},
    )

    suggestions = suggest_preprocessing(fmt)
    assert any("large" in s.lower() or "caching" in s.lower() for s in suggestions)


def test_suggest_preprocessing_no_metadata() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="png",
        image_count=100,
        format_counts={".png": 100},
        csv_path=None,
    )

    suggestions = suggest_preprocessing(fmt)
    assert any("metadata" in s.lower() or "csv" in s.lower() for s in suggestions)


def test_suggest_preprocessing_unknown_format() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="unknown",
        image_count=0,
    )

    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
    assert any("unknown" in s.lower() or "verify" in s.lower() for s in suggestions)


def test_suggest_preprocessing_features_txt() -> None:
    fmt = DatasetFormat(
        dataset_type="mamografias",
        image_format="png",
        image_count=200,
        format_counts={".png": 200},
        csv_path="/path/to/features",
        has_features_txt=True,
    )

    suggestions = suggest_preprocessing(fmt)
    assert any("featureS.txt" in s for s in suggestions)


def test_suggest_preprocessing_always_includes_validation() -> None:
    fmt = DatasetFormat(
        dataset_type="custom",
        image_format="png",
        image_count=100,
        format_counts={".png": 100},
    )

    suggestions = suggest_preprocessing(fmt)
    # Should always suggest validation
    assert any("validation" in s.lower() or "validate" in s.lower() for s in suggestions)


# Integration test: full workflow


def test_full_workflow_integration(tmp_path: Path) -> None:
    """Test full workflow: detect, validate, and suggest preprocessing."""
    # Create a realistic dataset structure
    case_folder = tmp_path / "test_dataset" / "case_001"
    case_folder.mkdir(parents=True)

    # Add images
    for i in range(5):
        img_path = case_folder / f"img_{i:03d}.png"
        _write_sample_image(img_path, "png")

    # Add featureS.txt
    features_content = "\n".join(
        [f"img_{i:03d}\n{i % 4 + 1}" for i in range(5)]
    )
    (case_folder / "featureS.txt").write_text(features_content, encoding="utf-8")

    # Detect format
    fmt = detect_dataset_format(str(tmp_path / "test_dataset"))
    assert fmt.image_count == 5
    assert fmt.image_format == "png"

    # Validate
    warnings = validate_format(fmt)
    assert isinstance(warnings, list)

    # Get suggestions
    suggestions = suggest_preprocessing(fmt)
    assert len(suggestions) > 0
