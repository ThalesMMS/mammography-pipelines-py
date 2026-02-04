from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pl = pytest.importorskip("polars")
Image = pytest.importorskip("PIL.Image")
pydicom = pytest.importorskip("pydicom")

from mammography.data.cancer_dataset import (
    MammoDicomDataset,
    MammographyDataset,
    SampleInfo,
    dataset_summary,
    make_dataloader,
    split_dataset,
)


def _create_mock_dicom(path: Path, rows: int = 256, cols: int = 256) -> None:
    """Create a minimal DICOM file for testing."""
    arr = np.random.randint(0, 4096, size=(rows, cols), dtype=np.uint16)
    ds = pydicom.Dataset()
    ds.Rows = rows
    ds.Columns = cols
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(path, write_like_original=False)


def _write_sample_png(path: Path) -> None:
    """Create a sample RGB PNG for testing."""
    img = Image.new("RGB", (16, 16), color=(120, 30, 60))
    img.save(path)


def test_sample_info_creation() -> None:
    """Test SampleInfo dataclass initialization."""
    info = SampleInfo(
        accession="ACC001",
        classification=2,
        path="/path/to/image.dcm",
        idx=0,
    )
    assert info.accession == "ACC001"
    assert info.classification == 2
    assert info.path == "/path/to/image.dcm"
    assert info.idx == 0


def test_sample_info_with_none_classification() -> None:
    """Test SampleInfo with None classification for unlabeled data."""
    info = SampleInfo(
        accession="ACC002",
        classification=None,
        path="/path/to/unlabeled.dcm",
        idx=1,
    )
    assert info.classification is None


def test_mammo_dicom_dataset_builds_index(tmp_path: Path) -> None:
    """Test MammoDicomDataset correctly builds sample index."""
    case_dir = tmp_path / "ACC001"
    case_dir.mkdir()
    dcm_path = case_dir / "image.dcm"
    _create_mock_dicom(dcm_path)

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        exclude_class_5=True,
        include_unlabeled=False,
    )

    assert len(dataset) == 1
    assert dataset.samples[0].accession == "ACC001"
    assert dataset.samples[0].classification == 2
    assert dataset.samples[0].path == str(dcm_path)


def test_mammo_dicom_dataset_excludes_class_5(tmp_path: Path) -> None:
    """Test MammoDicomDataset excludes class 5 samples when configured."""
    case1 = tmp_path / "ACC001"
    case1.mkdir()
    _create_mock_dicom(case1 / "image.dcm")

    case2 = tmp_path / "ACC002"
    case2.mkdir()
    _create_mock_dicom(case2 / "image.dcm")

    labels = {"ACC001": 2, "ACC002": 5}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        exclude_class_5=True,
    )

    assert len(dataset) == 1
    assert dataset.samples[0].accession == "ACC001"


def test_mammo_dicom_dataset_includes_class_5_when_allowed(tmp_path: Path) -> None:
    """Test MammoDicomDataset includes class 5 samples when not excluded."""
    case1 = tmp_path / "ACC001"
    case1.mkdir()
    _create_mock_dicom(case1 / "image.dcm")

    case2 = tmp_path / "ACC002"
    case2.mkdir()
    _create_mock_dicom(case2 / "image.dcm")

    labels = {"ACC001": 2, "ACC002": 5}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        exclude_class_5=False,
    )

    assert len(dataset) == 2


def test_mammo_dicom_dataset_excludes_unlabeled_by_default(tmp_path: Path) -> None:
    """Test MammoDicomDataset excludes unlabeled samples by default."""
    case1 = tmp_path / "ACC001"
    case1.mkdir()
    _create_mock_dicom(case1 / "image.dcm")

    case2 = tmp_path / "ACC002"
    case2.mkdir()
    _create_mock_dicom(case2 / "image.dcm")

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        include_unlabeled=False,
    )

    assert len(dataset) == 1
    assert dataset.samples[0].accession == "ACC001"


def test_mammo_dicom_dataset_includes_unlabeled_when_allowed(tmp_path: Path) -> None:
    """Test MammoDicomDataset includes unlabeled samples when configured."""
    case1 = tmp_path / "ACC001"
    case1.mkdir()
    _create_mock_dicom(case1 / "image.dcm")

    case2 = tmp_path / "ACC002"
    case2.mkdir()
    _create_mock_dicom(case2 / "image.dcm")

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        include_unlabeled=True,
    )

    assert len(dataset) == 2


def test_mammo_dicom_dataset_getitem_returns_correct_data(tmp_path: Path) -> None:
    """Test MammoDicomDataset __getitem__ returns expected tuple."""
    case_dir = tmp_path / "ACC001"
    case_dir.mkdir()
    dcm_path = case_dir / "image.dcm"
    _create_mock_dicom(dcm_path)

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    img, label, accession, path, idx = dataset[0]
    assert isinstance(img, Image.Image)
    assert label == 2
    assert accession == "ACC001"
    assert path == str(dcm_path)
    assert idx == 0


def test_mammo_dicom_dataset_getitem_with_unlabeled(tmp_path: Path) -> None:
    """Test MammoDicomDataset __getitem__ returns -1 for unlabeled samples."""
    case_dir = tmp_path / "ACC001"
    case_dir.mkdir()
    _create_mock_dicom(case_dir / "image.dcm")

    labels = {}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        include_unlabeled=True,
    )

    img, label, accession, path, idx = dataset[0]
    assert label == -1


def test_mammo_dicom_dataset_applies_transform(tmp_path: Path) -> None:
    """Test MammoDicomDataset applies transform to images."""
    case_dir = tmp_path / "ACC001"
    case_dir.mkdir()
    _create_mock_dicom(case_dir / "image.dcm")

    labels = {"ACC001": 2}

    class DummyTransform:
        def __call__(self, img):
            return "transformed"

    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        transform=DummyTransform(),
    )

    img, _, _, _, _ = dataset[0]
    assert img == "transformed"


def test_mammo_dicom_dataset_raises_on_missing_dir() -> None:
    """Test MammoDicomDataset raises FileNotFoundError for missing directory."""
    labels = {}
    with pytest.raises(FileNotFoundError):
        MammoDicomDataset(
            data_dir="/nonexistent/path",
            labels_by_accession=labels,
        )


def test_mammo_dicom_dataset_warns_on_empty_samples(tmp_path: Path) -> None:
    """Test MammoDicomDataset warns when no samples are found."""
    labels = {"ACC001": 2}
    with pytest.warns(UserWarning, match="Nenhuma amostra encontrada"):
        dataset = MammoDicomDataset(
            data_dir=str(tmp_path),
            labels_by_accession=labels,
        )
        assert len(dataset) == 0


def test_mammo_dicom_dataset_handles_nested_dicoms(tmp_path: Path) -> None:
    """Test MammoDicomDataset finds DICOMs in nested subdirectories."""
    case_dir = tmp_path / "ACC001"
    nested_dir = case_dir / "subdir" / "deeper"
    nested_dir.mkdir(parents=True)
    dcm_path = nested_dir / "image.dcm"
    _create_mock_dicom(dcm_path)

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    assert len(dataset) == 1
    assert dataset.samples[0].path == str(dcm_path)


def test_mammography_dataset_with_polars_dataframe(tmp_path: Path) -> None:
    """Test MammographyDataset loads data from Polars DataFrame."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "sample.png"
    _write_sample_png(img_path)

    df = pl.DataFrame({
        "fname": ["sample"],
        "cancer": [1],
    })

    dataset = MammographyDataset(meta_df=df, img_dir=str(img_dir))
    assert len(dataset) == 1


def test_mammography_dataset_getitem_returns_image_and_label(tmp_path: Path) -> None:
    """Test MammographyDataset __getitem__ returns correct image and label."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "sample.png"
    _write_sample_png(img_path)

    df = pl.DataFrame({
        "fname": ["sample"],
        "cancer": [1],
    })

    dataset = MammographyDataset(meta_df=df, img_dir=str(img_dir))
    img, label = dataset[0]

    assert isinstance(img, Image.Image)
    assert isinstance(label, torch.Tensor)
    assert label.item() == 1


def test_mammography_dataset_applies_transform(tmp_path: Path) -> None:
    """Test MammographyDataset applies transform to images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "sample.png"
    _write_sample_png(img_path)

    df = pl.DataFrame({
        "fname": ["sample"],
        "cancer": [0],
    })

    class DummyTransform:
        def __call__(self, img):
            return "transformed"

    dataset = MammographyDataset(
        meta_df=df,
        img_dir=str(img_dir),
        transform=DummyTransform(),
    )

    img, label = dataset[0]
    assert img == "transformed"
    assert label.item() == 0


def test_dataset_summary_counts_classes(tmp_path: Path) -> None:
    """Test dataset_summary correctly counts samples per class."""
    case1 = tmp_path / "ACC001"
    case1.mkdir()
    _create_mock_dicom(case1 / "image.dcm")

    case2 = tmp_path / "ACC002"
    case2.mkdir()
    _create_mock_dicom(case2 / "image.dcm")

    case3 = tmp_path / "ACC003"
    case3.mkdir()
    _create_mock_dicom(case3 / "image.dcm")

    labels = {"ACC001": 1, "ACC002": 2, "ACC003": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    summary = dataset_summary(dataset)
    assert summary[1] == 1
    assert summary[2] == 2
    assert summary[3] == 0
    assert summary[4] == 0


def test_split_dataset_returns_train_and_val(tmp_path: Path) -> None:
    """Test split_dataset correctly splits into train and validation."""
    for i in range(10):
        case_dir = tmp_path / f"ACC{i:03d}"
        case_dir.mkdir()
        _create_mock_dicom(case_dir / "image.dcm")

    labels = {f"ACC{i:03d}": 2 for i in range(10)}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    train_subset, val_subset = split_dataset(dataset, val_fraction=0.2, seed=42)

    assert len(train_subset) == 8
    assert len(val_subset) == 2
    assert len(train_subset) + len(val_subset) == len(dataset)


def test_split_dataset_returns_none_when_val_fraction_zero(tmp_path: Path) -> None:
    """Test split_dataset returns None for validation when val_fraction is 0."""
    case_dir = tmp_path / "ACC001"
    case_dir.mkdir()
    _create_mock_dicom(case_dir / "image.dcm")

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    train_subset, val_subset = split_dataset(dataset, val_fraction=0.0, seed=42)

    assert train_subset is not None
    assert val_subset is None


def test_split_dataset_is_deterministic(tmp_path: Path) -> None:
    """Test split_dataset produces consistent splits with same seed."""
    for i in range(10):
        case_dir = tmp_path / f"ACC{i:03d}"
        case_dir.mkdir()
        _create_mock_dicom(case_dir / "image.dcm")

    labels = {f"ACC{i:03d}": 2 for i in range(10)}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    train1, val1 = split_dataset(dataset, val_fraction=0.2, seed=42)
    train2, val2 = split_dataset(dataset, val_fraction=0.2, seed=42)

    assert len(train1) == len(train2)
    assert len(val1) == len(val2)


def test_split_dataset_raises_on_empty_dataset(tmp_path: Path) -> None:
    """Test split_dataset raises RuntimeError for empty dataset."""
    labels = {}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
        include_unlabeled=False,
    )

    with pytest.raises(RuntimeError, match="Dataset vazio"):
        split_dataset(dataset, val_fraction=0.2, seed=42)


def test_make_dataloader_creates_dataloader(tmp_path: Path) -> None:
    """Test make_dataloader creates a DataLoader with correct settings."""
    case_dir = tmp_path / "ACC001"
    case_dir.mkdir()
    _create_mock_dicom(case_dir / "image.dcm")

    labels = {"ACC001": 2}
    dataset = MammoDicomDataset(
        data_dir=str(tmp_path),
        labels_by_accession=labels,
    )

    device = torch.device("cpu")
    loader = make_dataloader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        device=device,
    )

    assert loader.batch_size == 1
    assert loader.num_workers == 0


def test_make_dataloader_pins_memory_for_cuda() -> None:
    """Test make_dataloader sets pin_memory=True for CUDA device."""
    from torch.utils.data import TensorDataset

    dummy_dataset = TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 2, (10,)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_dataloader(
        dummy_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        device=device,
    )

    if torch.cuda.is_available():
        assert loader.pin_memory is True
    else:
        assert loader.pin_memory is False
