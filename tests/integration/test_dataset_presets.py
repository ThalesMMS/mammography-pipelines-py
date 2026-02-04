"""
Integration tests for dataset presets using sampled real data.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.data.csv_loader import load_dataset_dataframe
from tests.utils.dataset_sampling import sample_dataframe


def _dataset_available(dataset_name: str) -> bool:
    if dataset_name == "archive":
        return Path("classificacao.csv").exists() and Path("archive").exists()
    return Path(dataset_name).exists()


@pytest.mark.parametrize(
    ("dataset_name", "expected_exts"),
    [
        ("archive", {".dcm", ".dicom"}),
        ("mamografias", {".png"}),
        ("patches_completo", {".png"}),
    ],
)
def test_dataset_preset_sampled(dataset_name: str, expected_exts: set[str]) -> None:
    if not _dataset_available(dataset_name):
        pytest.skip(f"Dataset '{dataset_name}' not available for sampling")

    df = load_dataset_dataframe(csv_path=None, dicom_root=None, dataset=dataset_name)
    sample = sample_dataframe(df)

    assert len(sample) > 0

    for path in sample["image_path"]:
        img_path = Path(path)
        assert img_path.exists()
        assert img_path.suffix.lower() in expected_exts

    labels = sample["professional_label"].dropna()
    if len(labels) > 0:
        assert labels.between(1, 4).all()

    if ".png" in expected_exts:
        to_open = sample["image_path"].head(min(5, len(sample))).tolist()
        for img_path in to_open:
            with Image.open(img_path) as img:
                img.verify()
