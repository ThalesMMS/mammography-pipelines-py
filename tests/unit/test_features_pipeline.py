from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

Image = pytest.importorskip("PIL.Image")

from mammography.tools import report_pack


def _write_dummy_image(path: Path) -> None:
    img = Image.new("RGB", (32, 32), color=(30, 120, 200))
    img.save(path)


def test_build_gradcam_grid(tmp_path: Path) -> None:
    img1 = tmp_path / "gradcam_1.png"
    img2 = tmp_path / "gradcam_2.png"
    _write_dummy_image(img1)
    _write_dummy_image(img2)

    dest = tmp_path / "grid.png"
    name = report_pack._build_gradcam_grid([img1, img2], dest, max_tiles=4)
    assert name == dest.name
    assert dest.exists()
