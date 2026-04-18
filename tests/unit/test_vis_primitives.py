from __future__ import annotations

from mammography.vis.primitives import color_palette, ensure_dir


def test_ensure_dir_creates_parent(tmp_path):
    out_path = tmp_path / "nested" / "plot.png"

    resolved = ensure_dir(out_path)

    assert resolved == out_path
    assert out_path.parent.exists()


def test_color_palette_returns_requested_count():
    colors = color_palette("viridis", 3)

    assert len(colors) == 3
