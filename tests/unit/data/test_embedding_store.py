# ruff: noqa
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

Image = pytest.importorskip("PIL.Image")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")

from mammography.data.csv_loader import (
    _coerce_density_label,
    _extract_view_from_dicom,
    _normalize_accession,
    load_dataset_dataframe,
    load_multiple_csvs,
    resolve_dataset_cache_mode,
    resolve_paths_from_preset,
    validate_split_overlap,
    _read_csv_with_encoding,
)
from mammography.data.splits import (
    create_kfold_splits,
    create_splits,
    create_three_way_split,
    filter_by_view,
    load_splits_from_csvs,
)
from mammography.data.format_detection import (
    detect_dataset_format,
    detect_image_format,
    infer_csv_schema,
    suggest_preprocessing,
    validate_format,
)
from mammography.data.dataset import (
    EmbeddingStore,
    load_embedding_store,
    MammoDensityDataset,
    robust_collate,
)

def _write_sample_image(path: Path) -> None:
    img = Image.new("RGB", (16, 16), color=(120, 30, 60))
    img.save(path)

def test_embedding_store_lookup() -> None:
    """Test embedding store lookup functionality."""
    embeddings_by_accession = {
        "ACC001": torch.randn(2048),
        "ACC002": torch.randn(2048),
    }
    embeddings_by_path = {
        "/path/to/img1.png": torch.randn(2048),
    }

    store = EmbeddingStore(
        embeddings_by_accession=embeddings_by_accession,
        embeddings_by_path=embeddings_by_path,
        feature_dim=2048,
    )

    # Test accession lookup
    row = {"accession": "ACC001", "image_path": "/other/path.png"}
    emb = store.lookup(row)
    assert emb is not None
    assert emb.shape == (2048,)

    # Test path lookup
    row = {"accession": "UNKNOWN", "image_path": "/path/to/img1.png"}
    emb = store.lookup(row)
    assert emb is not None

    # Test missing lookup
    row = {"accession": "UNKNOWN", "image_path": "/missing.png"}
    emb = store.lookup(row)
    assert emb is None

def test_load_embedding_store_valid(tmp_path: Path) -> None:
    """Test loading embedding store from valid directory."""
    # Create valid features.npy and metadata.csv
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(3, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002", "ACC003"],
        "path": ["img1.png", "img2.png", "img3.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    # Load embedding store
    store = load_embedding_store(str(embeddings_dir))

    assert store.feature_dim == 2048
    assert len(store.embeddings_by_accession) == 3
    assert len(store.embeddings_by_path) >= 3  # May have normalized paths too
    assert "ACC001" in store.embeddings_by_accession
    assert store.embeddings_by_accession["ACC001"].shape == (2048,)

def test_load_embedding_store_1d_features(tmp_path: Path) -> None:
    """Test loading embedding store with 1D features (should be reshaped to 2D)."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # 1D features should be reshaped to (1, 2048)
    features = np.random.randn(2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001"],
        "path": ["img1.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    # Load embedding store
    store = load_embedding_store(str(embeddings_dir))

    assert store.feature_dim == 2048
    assert len(store.embeddings_by_accession) == 1
    assert "ACC001" in store.embeddings_by_accession

def test_load_embedding_store_missing_features(tmp_path: Path) -> None:
    """Test error when features.npy is missing."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # Only create metadata.csv
    metadata = pd.DataFrame({
        "accession": ["ACC001"],
        "path": ["img1.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(FileNotFoundError, match="features.npy nao encontrado"):
        load_embedding_store(str(embeddings_dir))

def test_load_embedding_store_missing_metadata(tmp_path: Path) -> None:
    """Test error when metadata.csv is missing."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # Only create features.npy
    features = np.random.randn(3, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    with pytest.raises(FileNotFoundError, match="metadata.csv nao encontrado"):
        load_embedding_store(str(embeddings_dir))

def test_load_embedding_store_invalid_features_shape(tmp_path: Path) -> None:
    """Test error when features.npy has invalid shape (3D+)."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # 3D features are invalid
    features = np.random.randn(3, 2048, 5).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002", "ACC003"],
        "path": ["img1.png", "img2.png", "img3.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="deve ter 2 dimensoes"):
        load_embedding_store(str(embeddings_dir))

def test_load_embedding_store_shape_mismatch(tmp_path: Path) -> None:
    """Test error when metadata rows don't match features rows."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    # 3 rows in features
    features = np.random.randn(3, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    # But only 2 rows in metadata
    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002"],
        "path": ["img1.png", "img2.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="nao bate com features.npy"):
        load_embedding_store(str(embeddings_dir))

def test_load_embedding_store_no_valid_columns(tmp_path: Path) -> None:
    """Test error when metadata has no valid accession/path columns."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(2, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    # Metadata with neither accession nor path/image_path columns
    metadata = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 1],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="nao possui colunas 'accession'/'path' validas"):
        load_embedding_store(str(embeddings_dir))

def test_load_embedding_store_with_image_path_column(tmp_path: Path) -> None:
    """Test loading with 'image_path' column instead of 'path'."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(2, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", "ACC002"],
        "image_path": ["img1.png", "img2.png"],  # Use image_path instead of path
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    store = load_embedding_store(str(embeddings_dir))

    assert store.feature_dim == 2048
    assert len(store.embeddings_by_accession) == 2
    assert "img1.png" in store.embeddings_by_path

def test_load_embedding_store_nan_accessions(tmp_path: Path) -> None:
    """Test loading with NaN/empty accessions (should be skipped)."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    features = np.random.randn(4, 2048).astype(np.float32)
    np.save(embeddings_dir / "features.npy", features)

    metadata = pd.DataFrame({
        "accession": ["ACC001", None, np.nan, ""],
        "path": ["img1.png", "img2.png", "img3.png", "img4.png"],
    })
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)

    store = load_embedding_store(str(embeddings_dir))

    # Only ACC001 should be in accession index (others are invalid)
    assert len(store.embeddings_by_accession) == 1
    assert "ACC001" in store.embeddings_by_accession
    # But all 4 paths should be indexed
    assert len(store.embeddings_by_path) >= 4
