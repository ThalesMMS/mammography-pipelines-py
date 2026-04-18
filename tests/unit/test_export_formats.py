from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mammography.utils.export_formats import (
    parse_export_formats,
    save_metrics_figure_format,
)


def test_parse_export_formats_normalizes_csv() -> None:
    assert parse_export_formats("PNG, pdf,svg") == ["png", "pdf", "svg"]


def test_parse_export_formats_rejects_unknown_format() -> None:
    with pytest.raises(SystemExit):
        parse_export_formats("png,jpg")


def test_save_metrics_figure_format_creates_parent_and_delegates(tmp_path: Path) -> None:
    out_path = tmp_path / "figures" / "metrics.pdf"

    with patch("mammography.training.engine.save_metrics_figure") as save_metrics:
        save_metrics_figure_format({"acc": 1.0}, str(out_path))

    assert out_path.parent.exists()
    save_metrics.assert_called_once_with({"acc": 1.0}, str(out_path))
