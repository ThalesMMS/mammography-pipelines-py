from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mammography.apps.density_classifier import streamlit_app


def test_load_classification_df_new_file(tmp_path: Path) -> None:
    """Test loading classification CSV when file doesn't exist."""
    csv_path = tmp_path / "classification.csv"
    df = streamlit_app._load_classification_df(csv_path)

    assert df.empty
    assert df.index.name == "AccessionNumber"
    assert "Classification" in df.columns
    assert "ClassificationDate" in df.columns


def test_load_classification_df_existing_file(tmp_path: Path) -> None:
    """Test loading classification CSV from existing file."""
    csv_path = tmp_path / "classification.csv"
    test_data = pd.DataFrame(
        {
            "AccessionNumber": ["001", "002", "003"],
            "Classification": [1, 2, 3],
            "ClassificationDate": [
                "2025-01-01 10:00:00",
                "2025-01-02 11:00:00",
                "2025-01-03 12:00:00",
            ],
        }
    )
    test_data.to_csv(csv_path, index=False)

    df = streamlit_app._load_classification_df(csv_path)

    assert not df.empty
    assert df.index.name == "AccessionNumber"
    assert len(df) == 3
    assert "001" in df.index
    assert df.loc["001", "Classification"] == 1


def test_save_classification(tmp_path: Path) -> None:
    """Test saving classification label to CSV."""
    csv_path = tmp_path / "classification.csv"
    df = pd.DataFrame(columns=["Classification", "ClassificationDate"])
    df.index.name = "AccessionNumber"

    streamlit_app._save_classification(df, csv_path, "001", 2)

    assert csv_path.exists()
    loaded_df = pd.read_csv(csv_path, dtype={"AccessionNumber": str})
    loaded_df.set_index("AccessionNumber", inplace=True)

    assert "001" in loaded_df.index
    assert loaded_df.loc["001", "Classification"] == 2
    assert "ClassificationDate" in loaded_df.columns


def test_save_classification_updates_existing(tmp_path: Path) -> None:
    """Test updating an existing classification label."""
    csv_path = tmp_path / "classification.csv"
    df = pd.DataFrame(
        {"Classification": [1], "ClassificationDate": ["2025-01-01 10:00:00"]},
        index=pd.Index(["001"], name="AccessionNumber"),
    )
    df.to_csv(csv_path)

    df = streamlit_app._load_classification_df(csv_path)
    streamlit_app._save_classification(df, csv_path, "001", 4)

    loaded_df = pd.read_csv(csv_path, dtype={"AccessionNumber": str})
    loaded_df.set_index("AccessionNumber", inplace=True)

    assert loaded_df.loc["001", "Classification"] == 4


def test_resolve_classification_path_primary(tmp_path: Path) -> None:
    """Test path resolution prefers classification.csv over legacy."""
    primary = tmp_path / "classification.csv"
    primary.write_text("AccessionNumber,Classification\n", encoding="utf-8")

    result = streamlit_app._resolve_classification_path(tmp_path)

    assert result == primary


def test_resolve_classification_path_legacy(tmp_path: Path) -> None:
    """Test path resolution falls back to classificacao.csv."""
    legacy = tmp_path / "classificacao.csv"
    legacy.write_text("AccessionNumber,Classification\n", encoding="utf-8")

    result = streamlit_app._resolve_classification_path(tmp_path)

    assert result == legacy


def test_resolve_classification_path_none_exist(tmp_path: Path) -> None:
    """Test path resolution returns primary when neither exists."""
    result = streamlit_app._resolve_classification_path(tmp_path)

    assert result == tmp_path / "classification.csv"


def test_load_train_accessions_valid(tmp_path: Path) -> None:
    """Test loading accession numbers from train.csv."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    train_csv = archive_dir / "train.csv"
    train_data = pd.DataFrame({"AccessionNumber": ["001", "002", "003"]})
    train_data.to_csv(train_csv, index=False)

    for acc in ["001", "002", "003"]:
        (archive_dir / acc).mkdir()

    accessions = streamlit_app._load_train_accessions(archive_dir)

    assert accessions == ["001", "002", "003"]


def test_load_train_accessions_missing_file(tmp_path: Path) -> None:
    """Test error handling when train.csv is missing."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="train.csv not found"):
        streamlit_app._load_train_accessions(archive_dir)


def test_load_train_accessions_missing_column(tmp_path: Path) -> None:
    """Test error handling when train.csv lacks AccessionNumber column."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    train_csv = archive_dir / "train.csv"
    train_data = pd.DataFrame({"PatientID": ["P001", "P002"]})
    train_data.to_csv(train_csv, index=False)

    with pytest.raises(ValueError, match="missing AccessionNumber column"):
        streamlit_app._load_train_accessions(archive_dir)


def test_load_train_accessions_filters_directories(tmp_path: Path) -> None:
    """Test that only directories in train.csv are returned."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    train_csv = archive_dir / "train.csv"
    train_data = pd.DataFrame({"AccessionNumber": ["001", "002", "003", "004"]})
    train_data.to_csv(train_csv, index=False)

    (archive_dir / "001").mkdir()
    (archive_dir / "002").mkdir()
    (archive_dir / "005").mkdir()

    accessions = streamlit_app._load_train_accessions(archive_dir)

    assert accessions == ["001", "002"]
    assert "003" not in accessions
    assert "005" not in accessions


def test_get_dicom_files(tmp_path: Path) -> None:
    """Test retrieving DICOM files from accession directory."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    accession_dir = archive_dir / "001"
    accession_dir.mkdir()

    (accession_dir / "image1.dcm").write_text("", encoding="utf-8")
    (accession_dir / "image2.DCM").write_text("", encoding="utf-8")
    (accession_dir / "readme.txt").write_text("", encoding="utf-8")

    files = streamlit_app._get_dicom_files(archive_dir, "001")

    assert len(files) == 2
    assert "image1.dcm" in files
    assert "image2.DCM" in files
    assert "readme.txt" not in files


def test_get_dicom_files_missing_directory(tmp_path: Path) -> None:
    """Test handling of missing accession directory."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    files = streamlit_app._get_dicom_files(archive_dir, "nonexistent")

    assert files == []


def test_guess_project_root_with_archive(tmp_path: Path, monkeypatch) -> None:
    """Test project root detection when archive exists."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    root = streamlit_app._guess_project_root()

    assert root == tmp_path


def test_guess_project_root_no_archive(tmp_path: Path, monkeypatch) -> None:
    """Test project root detection falls back to cwd when no archive."""
    # Create a fake module file inside tmp_path to avoid finding real archive
    fake_module = tmp_path / "apps" / "density_classifier"
    fake_module.mkdir(parents=True)
    fake_file = fake_module / "streamlit_app.py"
    fake_file.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(streamlit_app, "__file__", str(fake_file))

    root = streamlit_app._guess_project_root()

    assert root == tmp_path


def test_require_streamlit_raises_when_missing(monkeypatch) -> None:
    """Test that _require_streamlit raises ImportError when streamlit unavailable."""
    monkeypatch.setattr(streamlit_app, "st", None)
    monkeypatch.setattr(
        streamlit_app, "_STREAMLIT_IMPORT_ERROR", ImportError("streamlit not found")
    )

    with pytest.raises(ImportError, match="Streamlit is required"):
        streamlit_app._require_streamlit()


def test_run_raises_when_streamlit_missing(monkeypatch) -> None:
    """Test that run() raises ImportError when streamlit unavailable."""
    monkeypatch.setattr(streamlit_app, "st", None)
    monkeypatch.setattr(
        streamlit_app, "_STREAMLIT_IMPORT_ERROR", ImportError("streamlit not found")
    )

    with pytest.raises(ImportError, match="Streamlit is required"):
        streamlit_app.run([])


def test_labels_mapping() -> None:
    """Test that density labels are correctly defined."""
    assert streamlit_app.LABELS[1] == "1 - Fatty"
    assert streamlit_app.LABELS[2] == "2 - Mostly Fatty"
    assert streamlit_app.LABELS[3] == "3 - Mostly Dense"
    assert streamlit_app.LABELS[4] == "4 - Dense"
    assert streamlit_app.LABELS[5] == "5 - Issue/Skip"
    assert len(streamlit_app.LABELS) == 5
