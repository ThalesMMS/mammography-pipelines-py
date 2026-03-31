from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mammography.commands import web as web_cmd


def test_main_calls_run(monkeypatch) -> None:
    """Test that main() properly delegates to run_streamlit()."""
    captured_args = []

    def _fake_run(argv):
        captured_args.append(argv)
        return 0

    monkeypatch.setattr(web_cmd, "run_streamlit", _fake_run)

    result = web_cmd.main(["--arg1", "--arg2"])

    assert result == 0
    assert len(captured_args) == 1
    assert captured_args[0] == ["--arg1", "--arg2"]


def test_main_with_no_args(monkeypatch) -> None:
    """Test that main() works with None argv."""
    captured_args = []

    def _fake_run(argv):
        captured_args.append(argv)
        return 0

    monkeypatch.setattr(web_cmd, "run_streamlit", _fake_run)

    result = web_cmd.main(None)

    assert result == 0
    assert len(captured_args) == 1
    assert captured_args[0] is None


def test_run_raises_when_streamlit_missing(monkeypatch) -> None:
    """Test that run() raises ImportError when streamlit unavailable."""
    from mammography.apps.web_ui import streamlit_app

    monkeypatch.setattr(streamlit_app, "st", None)
    monkeypatch.setattr(
        streamlit_app, "_STREAMLIT_IMPORT_ERROR", ImportError("streamlit not found")
    )

    with pytest.raises(ImportError, match="Streamlit is required"):
        web_cmd.run_streamlit([])


def test_run_launches_streamlit_via_cli(monkeypatch) -> None:
    """Test that run() launches streamlit via stcli.main() when available."""
    from mammography.apps.web_ui import streamlit_app

    # Mock streamlit being available
    class _FakeStreamlit:
        pass

    monkeypatch.setattr(streamlit_app, "st", _FakeStreamlit())
    monkeypatch.setattr(streamlit_app, "_STREAMLIT_IMPORT_ERROR", None)

    # Mock streamlit.web.cli
    class _FakeCLI:
        @staticmethod
        def main():
            raise SystemExit(0)

    class _FakeSTCLI:
        cli = _FakeCLI

    # Track sys.argv changes
    original_argv = sys.argv[:]
    captured_argv = []

    def _capture_main():
        captured_argv.append(sys.argv[:])
        raise SystemExit(0)

    monkeypatch.setattr("streamlit.web.cli.main", _capture_main, raising=False)

    # Make streamlit.web.cli importable
    import types
    streamlit_web = types.ModuleType("streamlit.web")
    streamlit_web.cli = types.ModuleType("streamlit.web.cli")
    streamlit_web.cli.main = _capture_main
    sys.modules["streamlit.web"] = streamlit_web
    sys.modules["streamlit.web.cli"] = streamlit_web.cli

    try:
        result = web_cmd.run_streamlit(["--server.port", "8501"])

        assert result == 0
        assert len(captured_argv) == 1
        # Verify sys.argv was set correctly
        assert captured_argv[0][0] == "streamlit"
        assert captured_argv[0][1] == "run"
        assert "--server.port" in captured_argv[0]
        assert "8501" in captured_argv[0]
        # Verify sys.argv was restored
        assert sys.argv == original_argv
    finally:
        # Cleanup
        if "streamlit.web.cli" in sys.modules:
            del sys.modules["streamlit.web.cli"]
        if "streamlit.web" in sys.modules:
            del sys.modules["streamlit.web"]


def test_run_cli_returns_zero_on_success(monkeypatch) -> None:
    """Test that run() returns 0 when streamlit cli exits with code 0."""
    from mammography.apps.web_ui import streamlit_app

    # Mock streamlit being available
    class _FakeStreamlit:
        pass

    monkeypatch.setattr(streamlit_app, "st", _FakeStreamlit())
    monkeypatch.setattr(streamlit_app, "_STREAMLIT_IMPORT_ERROR", None)

    # Mock streamlit.web.cli that exits with 0
    def _capture_main():
        raise SystemExit(0)

    import types
    streamlit_web = types.ModuleType("streamlit.web")
    streamlit_web.cli = types.ModuleType("streamlit.web.cli")
    streamlit_web.cli.main = _capture_main
    sys.modules["streamlit.web"] = streamlit_web
    sys.modules["streamlit.web.cli"] = streamlit_web.cli

    try:
        result = web_cmd.run_streamlit([])
        assert result == 0
    finally:
        # Cleanup
        if "streamlit.web.cli" in sys.modules:
            del sys.modules["streamlit.web.cli"]
        if "streamlit.web" in sys.modules:
            del sys.modules["streamlit.web"]


def test_run_with_empty_args(monkeypatch) -> None:
    """Test that run() works with empty arguments list."""
    from mammography.apps.web_ui import streamlit_app

    # Mock streamlit being available
    class _FakeStreamlit:
        pass

    monkeypatch.setattr(streamlit_app, "st", _FakeStreamlit())
    monkeypatch.setattr(streamlit_app, "_STREAMLIT_IMPORT_ERROR", None)

    # Mock streamlit.web.cli
    captured_argv = []

    def _capture_main():
        captured_argv.append(sys.argv[:])
        raise SystemExit(0)

    import types
    streamlit_web = types.ModuleType("streamlit.web")
    streamlit_web.cli = types.ModuleType("streamlit.web.cli")
    streamlit_web.cli.main = _capture_main
    sys.modules["streamlit.web"] = streamlit_web
    sys.modules["streamlit.web.cli"] = streamlit_web.cli

    try:
        result = web_cmd.run_streamlit([])

        assert result == 0
        assert len(captured_argv) == 1
        # Should have streamlit, run, and script path
        assert captured_argv[0][0] == "streamlit"
        assert captured_argv[0][1] == "run"
        assert len(captured_argv[0]) == 3  # streamlit run <script>
    finally:
        # Cleanup
        if "streamlit.web.cli" in sys.modules:
            del sys.modules["streamlit.web.cli"]
        if "streamlit.web" in sys.modules:
            del sys.modules["streamlit.web"]
