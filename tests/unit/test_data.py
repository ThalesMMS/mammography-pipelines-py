from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("pandas")

from mammography.data.csv_loader import load_dataset_dataframe, resolve_dataset_cache_mode


def _write_sample_image(path: Path) -> None:
    img = Image.new("RGB", (16, 16), color=(120, 30, 60))
    img.save(path)


def test_load_dataset_dataframe_from_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_sample_image(image_path)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "image_path,professional_label,accession\n"
        f"{image_path},2,ACC001\n",
        encoding="utf-8",
    )

    df = load_dataset_dataframe(str(csv_path), dicom_root=None)
    assert list(df.columns) == ["image_path", "professional_label", "accession"]
    assert len(df) == 1
    assert df.iloc[0]["professional_label"] == 2

    cache_mode = resolve_dataset_cache_mode("auto", df)
    assert cache_mode in {"memory", "none"}


def test_load_dataset_dataframe_from_features_dir(tmp_path: Path) -> None:
    folder = tmp_path / "case_001"
    folder.mkdir()

    img_path = folder / "img_001.png"
    _write_sample_image(img_path)

    (folder / "featureS.txt").write_text("img_001\n1\n", encoding="utf-8")

    df = load_dataset_dataframe(str(tmp_path), dicom_root=None)
    assert len(df) == 1
    assert df.iloc[0]["image_path"].endswith("img_001.png")
    assert df.iloc[0]["professional_label"] == 2
