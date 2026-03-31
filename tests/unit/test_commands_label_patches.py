from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mammography.apps.patch_marking import streamlit_app


def test_guess_project_root_with_archive(tmp_path: Path, monkeypatch) -> None:
    """Test project root detection when archive exists."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    root = streamlit_app._guess_project_root()

    assert root == tmp_path


def test_guess_project_root_no_archive(tmp_path: Path, monkeypatch) -> None:
    """Test project root detection falls back to cwd when no archive."""
    fake_module = tmp_path / "apps" / "patch_marking"
    fake_module.mkdir(parents=True)
    fake_file = fake_module / "streamlit_app.py"
    fake_file.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(streamlit_app, "__file__", str(fake_file))

    root = streamlit_app._guess_project_root()

    assert root == tmp_path


def test_ensure_annotations_csv_creates_file(tmp_path: Path) -> None:
    """Test that _ensure_annotations_csv creates CSV with correct headers."""
    csv_path = streamlit_app._ensure_annotations_csv(tmp_path)

    assert csv_path == tmp_path / "annotations.csv"
    assert csv_path.exists()

    content = csv_path.read_text(encoding="utf-8")
    header = content.strip().split("\n")[0]
    expected_header = (
        "AccessionNumber,DCM_Filename,Adjusted_ROI_Center_X,"
        "Adjusted_ROI_Center_Y,ROI_Size,Saved_PNG_Filename"
    )
    assert header == expected_header


def test_ensure_annotations_csv_preserves_existing(tmp_path: Path) -> None:
    """Test that _ensure_annotations_csv doesn't overwrite existing file."""
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text("AccessionNumber,DCM_Filename\n001,img.dcm\n", encoding="utf-8")

    result = streamlit_app._ensure_annotations_csv(tmp_path)

    assert result == csv_path
    content = csv_path.read_text(encoding="utf-8")
    assert "001,img.dcm" in content


def test_draw_roi_overlay_with_bounds() -> None:
    """Test ROI overlay drawing with valid bounds."""
    image = np.ones((100, 100), dtype=np.uint8) * 128
    bounds = (10, 20, 50, 60)

    result = streamlit_app._draw_roi_overlay(image, bounds)

    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_draw_roi_overlay_without_bounds() -> None:
    """Test ROI overlay returns original image when bounds is None."""
    image = np.ones((100, 100), dtype=np.uint8) * 128

    result = streamlit_app._draw_roi_overlay(image, None)

    assert np.array_equal(result, image)


def test_save_roi_without_defined_roi(tmp_path: Path) -> None:
    """Test that _save_roi returns error when ROI is not defined."""
    folder_path = tmp_path / "001"
    folder_path.mkdir()

    class MockRoiSelector:
        roi_size = 448

        def is_defined(self):
            return False

    roi_selector = MockRoiSelector()
    pixel_data = np.zeros((100, 100), dtype=np.float32)
    view_params = {"wc": 0.5, "ww": 1.0, "photometric": "MONOCHROME2"}

    success, message = streamlit_app._save_roi(
        tmp_path,
        folder_path,
        "test.dcm",
        pixel_data,
        view_params,
        roi_selector,
    )

    assert not success
    assert message == "ROI is not defined."


def test_save_roi_successful(tmp_path: Path, monkeypatch) -> None:
    """Test successful ROI save with valid data."""
    folder_path = tmp_path / "001"
    folder_path.mkdir()

    class MockRoiSelector:
        roi_size = 448
        current_roi_bounds = (10, 20, 50, 60)

        def is_defined(self):
            return True

        def extract_roi_from_image(self, _image):
            return np.ones((48, 48), dtype=np.float32) * 0.5

    roi_selector = MockRoiSelector()
    pixel_data = np.zeros((100, 100), dtype=np.float32)
    view_params = {"wc": 0.5, "ww": 1.0, "photometric": "MONOCHROME2"}

    success, message = streamlit_app._save_roi(
        tmp_path,
        folder_path,
        "test.dcm",
        pixel_data,
        view_params,
        roi_selector,
    )

    assert success
    assert "Saved test.png" in message
    assert (folder_path / "test.png").exists()
    assert (tmp_path / "annotations.csv").exists()


def test_save_roi_increments_filename(tmp_path: Path, monkeypatch) -> None:
    """Test that _save_roi increments filename when file exists."""
    folder_path = tmp_path / "001"
    folder_path.mkdir()
    (folder_path / "test.png").write_bytes(b"existing")

    class MockRoiSelector:
        roi_size = 448
        current_roi_bounds = (10, 20, 50, 60)

        def is_defined(self):
            return True

        def extract_roi_from_image(self, _image):
            return np.ones((48, 48), dtype=np.float32) * 0.5

    roi_selector = MockRoiSelector()
    pixel_data = np.zeros((100, 100), dtype=np.float32)
    view_params = {"wc": 0.5, "ww": 1.0, "photometric": "MONOCHROME2"}

    success, message = streamlit_app._save_roi(
        tmp_path,
        folder_path,
        "test.dcm",
        pixel_data,
        view_params,
        roi_selector,
    )

    assert success
    assert "Saved test_1.png" in message
    assert (folder_path / "test_1.png").exists()


def test_save_roi_extraction_failure(tmp_path: Path) -> None:
    """Test that _save_roi handles extraction failure."""
    folder_path = tmp_path / "001"
    folder_path.mkdir()

    class MockRoiSelector:
        roi_size = 448
        current_roi_bounds = None

        def is_defined(self):
            return True

        def extract_roi_from_image(self, _image):
            return None

    roi_selector = MockRoiSelector()
    pixel_data = np.zeros((100, 100), dtype=np.float32)
    view_params = {"wc": 0.5, "ww": 1.0, "photometric": "MONOCHROME2"}

    success, message = streamlit_app._save_roi(
        tmp_path,
        folder_path,
        "test.dcm",
        pixel_data,
        view_params,
        roi_selector,
    )

    assert not success
    assert message == "Failed to extract ROI."


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


def test_run_with_streamlit_cli(tmp_path: Path, monkeypatch) -> None:
    """Test run() launches streamlit using CLI method."""
    script_path_called = []
    args_called = []

    class MockStCli:
        @staticmethod
        def main():
            script_path_called.append(monkeypatch.context["sys.argv"][2])
            args_called.append(monkeypatch.context["sys.argv"][3:])
            raise SystemExit(0)

    class MockStWeb:
        cli = MockStCli

    monkeypatch.setattr(streamlit_app, "st", object())
    monkeypatch.setattr(streamlit_app, "_STREAMLIT_IMPORT_ERROR", None)
    monkeypatch.context = {"sys.argv": []}
    import sys

    original_argv = sys.argv[:]

    def mock_setattr(obj, name, value):
        if obj is sys and name == "argv":
            monkeypatch.context["sys.argv"] = value
            sys.argv = value

    monkeypatch.setattr("builtins.setattr", mock_setattr)
    monkeypatch.setattr("streamlit.web", MockStWeb)

    sys.modules["streamlit.web.cli"] = MockStCli

    try:
        exit_code = streamlit_app.run(["--server.port", "8501"])
    finally:
        sys.argv = original_argv
        if "streamlit.web.cli" in sys.modules:
            del sys.modules["streamlit.web.cli"]

    assert exit_code == 0


def test_run_with_streamlit_bootstrap(tmp_path: Path, monkeypatch) -> None:
    """Test run() launches streamlit using bootstrap method."""
    bootstrap_called = []

    class MockBootstrap:
        @staticmethod
        def run(script_path, flag, args, kwargs):
            bootstrap_called.append((script_path, args))
            raise SystemExit(0)

    class MockStWeb:
        bootstrap = MockBootstrap

    monkeypatch.setattr(streamlit_app, "st", object())
    monkeypatch.setattr(streamlit_app, "_STREAMLIT_IMPORT_ERROR", None)

    import sys

    if "streamlit.web.cli" in sys.modules:
        monkeypatch.delattr(sys.modules["streamlit.web"], "cli", raising=False)

    sys.modules["streamlit.web"] = MockStWeb
    sys.modules["streamlit.web.bootstrap"] = MockBootstrap

    try:
        exit_code = streamlit_app.run(["--server.port", "8501"])
    finally:
        if "streamlit.web" in sys.modules:
            del sys.modules["streamlit.web"]
        if "streamlit.web.bootstrap" in sys.modules:
            del sys.modules["streamlit.web.bootstrap"]

    assert exit_code == 0
    assert len(bootstrap_called) == 1


def test_main_calls_run_streamlit(monkeypatch) -> None:
    """Test that main() calls run_streamlit with arguments."""
    from mammography.commands import label_patches

    run_called = []

    def mock_run(argv):
        run_called.append(argv)
        return 0

    monkeypatch.setattr(label_patches, "run_streamlit", mock_run)

    result = label_patches.main(["--test-arg"])

    assert result == 0
    assert run_called == [["--test-arg"]]
