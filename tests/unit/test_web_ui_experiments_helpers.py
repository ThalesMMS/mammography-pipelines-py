from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mammography.apps.web_ui.components import export_buttons
from mammography.apps.web_ui.components.results_visualizer import plot_confusion_matrix
from mammography.apps.web_ui.pages.experiments import (
    formatters,
    mlflow_client,
    overview,
    state,
)
from tests.helpers import SessionState, _Context


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.messages = []
        self.downloads = []

    def header(self, text):
        self.messages.append(("header", text))

    def subheader(self, text):
        self.messages.append(("subheader", text))

    def markdown(self, text, **kwargs):
        self.messages.append(("markdown", text))

    def info(self, text):
        self.messages.append(("info", text))

    def error(self, text):
        self.messages.append(("error", text))

    def warning(self, text):
        self.messages.append(("warning", text))

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context(self) for _ in range(count)]

    def dataframe(self, *args, **kwargs):
        self.messages.append(("dataframe", args, kwargs))

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index] if options else None

    def number_input(self, label, value=0, **kwargs):
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        return default or []

    def download_button(self, **kwargs):
        self.downloads.append(kwargs)


class FakeFigure:
    def to_image(self, fmt=None, **kwargs):
        fmt = fmt or kwargs["format"]
        return f"{fmt}-bytes".encode()


def test_experiment_session_defaults() -> None:
    session = SessionState()

    state.ensure_session_defaults(session)

    assert session.mlflow_client is None
    assert session.mlflow_tracking_uri is None
    assert session.selected_experiment is None
    assert session.selected_run is None


def test_formatters() -> None:
    assert formatters.format_timestamp(None) == "N/A"
    assert formatters.format_duration(None, None) == "N/A"
    assert formatters.format_duration(0, 65_000) == "1m 5s"
    assert formatters.format_duration(0, 3_661_000) == "1h 1m 1s"


def test_mlflow_list_helpers() -> None:
    active = SimpleNamespace(lifecycle_stage="active", experiment_id="1")
    deleted = SimpleNamespace(lifecycle_stage="deleted", experiment_id="2")

    class FakeClient:
        def search_experiments(self):
            return [active, deleted]

        def search_runs(self, **kwargs):
            self.kwargs = kwargs
            return ["run"]

    client = FakeClient()

    assert mlflow_client.list_experiments(client) == [active]
    assert mlflow_client.list_runs(client, "1", max_results=5) == ["run"]
    assert client.kwargs["experiment_ids"] == ["1"]
    assert client.kwargs["max_results"] == 5


def test_export_plot_buttons_uses_shared_streamlit_component(monkeypatch) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(export_buttons, "st", fake_st)

    export_buttons.export_plot_buttons(FakeFigure(), "metrics")

    assert [item["file_name"] for item in fake_st.downloads] == [
        "metrics.png",
        "metrics.pdf",
        "metrics.svg",
    ]


def test_overview_section_handles_empty_experiment_list(monkeypatch) -> None:
    fake_st = FakeStreamlit()

    class FakeClient:
        def search_experiments(self):
            return []

    monkeypatch.setattr(overview, "st", fake_st)

    overview.display_experiment_overview(FakeClient())

    assert any("No experiments found" in text for kind, text in fake_st.messages if kind == "info")


def test_shared_confusion_matrix_accepts_string_normalization() -> None:
    fig = plot_confusion_matrix(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        normalize="pred",
    )

    assert fig.axes


# ---------------------------------------------------------------------------
# Additional tests for new/changed code in this PR
# ---------------------------------------------------------------------------


def test_experiment_session_defaults_idempotent() -> None:
    """Calling ensure_session_defaults twice should not overwrite existing values."""
    session = SessionState()
    state.ensure_session_defaults(session)

    # Set a non-default value
    session.mlflow_client = "fake_client"
    session.selected_experiment = "exp_123"

    # Second call must not reset existing keys
    state.ensure_session_defaults(session)

    assert session.mlflow_client == "fake_client"
    assert session.selected_experiment == "exp_123"
    # Keys added on first call but not yet set should still be None
    assert session.mlflow_tracking_uri is None
    assert session.selected_run is None


def test_format_timestamp_returns_formatted_string() -> None:
    """format_timestamp with a valid ms timestamp should return a date string."""
    # 2000-01-01 00:00:00 UTC in milliseconds
    ts_ms = 946684800_000
    result = formatters.format_timestamp(ts_ms)
    # The exact time depends on local timezone, but format should be correct
    assert result != "N/A"
    assert result != "Invalid timestamp"
    # Should match the pattern YYYY-MM-DD HH:MM:SS
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result), (
        f"Unexpected timestamp format: {result!r}"
    )


def test_format_duration_only_seconds() -> None:
    """Durations under a minute should return only seconds."""
    assert formatters.format_duration(0, 30_000) == "30s"


def test_format_duration_zero_seconds() -> None:
    """Start equals end → 0s."""
    assert formatters.format_duration(1000, 1000) == "0s"


def test_format_duration_uses_now_when_end_is_none() -> None:
    """When end_ms is None, duration is calculated from now — must not raise."""
    import time
    start_ms = int(time.time() * 1000) - 5_000  # 5 seconds ago
    result = formatters.format_duration(start_ms, None)
    assert result != "N/A"
    # Should be at least "5s"
    assert "s" in result


def test_mlflow_client_caching_reuses_existing_client() -> None:
    """get_mlflow_client should return the cached client when URI unchanged."""
    fake_client = object()
    session = SessionState()
    session.mlflow_client = fake_client
    session.mlflow_tracking_uri = "http://localhost:5000"

    # Patch mlflow so it is "available" but should NOT be called
    import mammography.apps.web_ui.pages.experiments.mlflow_client as mlflow_client_mod

    class FakeMlflow:
        @staticmethod
        def set_tracking_uri(uri):
            raise AssertionError("set_tracking_uri should not be called when client is cached")

    original_mlflow = mlflow_client_mod.mlflow
    original_MlflowClient = mlflow_client_mod.MlflowClient

    mlflow_client_mod.mlflow = FakeMlflow()
    mlflow_client_mod.MlflowClient = object  # not callable in the expected path

    try:
        result = mlflow_client.get_mlflow_client("http://localhost:5000", session)
        assert result is fake_client
    finally:
        mlflow_client_mod.mlflow = original_mlflow
        mlflow_client_mod.MlflowClient = original_MlflowClient


def test_mlflow_client_creation_does_not_mutate_global_uri() -> None:
    """get_mlflow_client should rely on scoped MlflowClient instances."""
    import mammography.apps.web_ui.pages.experiments.mlflow_client as mlflow_client_mod

    class FakeMlflow:
        @staticmethod
        def set_tracking_uri(uri):
            raise AssertionError("set_tracking_uri should not be called")

    class FakeMlflowClient:
        def __init__(self, tracking_uri):
            self.tracking_uri = tracking_uri

    original_mlflow = mlflow_client_mod.mlflow
    original_MlflowClient = mlflow_client_mod.MlflowClient

    mlflow_client_mod.mlflow = FakeMlflow()
    mlflow_client_mod.MlflowClient = FakeMlflowClient

    session = SessionState()
    session.mlflow_client = None
    session.mlflow_tracking_uri = None

    try:
        result = mlflow_client.get_mlflow_client("http://localhost:5000", session)
        assert isinstance(result, FakeMlflowClient)
        assert result.tracking_uri == "http://localhost:5000"
        assert session.mlflow_client is result
        assert session.mlflow_tracking_uri == "http://localhost:5000"
    finally:
        mlflow_client_mod.mlflow = original_mlflow
        mlflow_client_mod.MlflowClient = original_MlflowClient


def test_mlflow_client_raises_when_mlflow_unavailable() -> None:
    """get_mlflow_client should raise ImportError when mlflow is not installed."""
    import pytest
    import mammography.apps.web_ui.pages.experiments.mlflow_client as mlflow_client_mod

    original_mlflow = mlflow_client_mod.mlflow
    original_MlflowClient = mlflow_client_mod.MlflowClient
    original_error = mlflow_client_mod._MLFLOW_IMPORT_ERROR

    mlflow_client_mod.mlflow = None
    mlflow_client_mod.MlflowClient = None
    mlflow_client_mod._MLFLOW_IMPORT_ERROR = ImportError("no mlflow")

    session = SessionState()
    session.mlflow_client = None
    session.mlflow_tracking_uri = None

    try:
        with pytest.raises(ImportError, match="MLflow is required"):
            mlflow_client.get_mlflow_client("http://localhost:5000", session)
    finally:
        mlflow_client_mod.mlflow = original_mlflow
        mlflow_client_mod.MlflowClient = original_MlflowClient
        mlflow_client_mod._MLFLOW_IMPORT_ERROR = original_error


def test_mlflow_list_experiments_all_deleted_returns_empty() -> None:
    """list_experiments returns empty list when all experiments are deleted."""
    deleted1 = SimpleNamespace(lifecycle_stage="deleted", experiment_id="1")
    deleted2 = SimpleNamespace(lifecycle_stage="deleted", experiment_id="2")

    class FakeClient:
        def search_experiments(self):
            return [deleted1, deleted2]

    assert mlflow_client.list_experiments(FakeClient()) == []


def test_mlflow_list_runs_default_max_results() -> None:
    """list_runs should use default max_results=100 when not specified."""
    class FakeClient:
        def search_runs(self, **kwargs):
            self.kwargs = kwargs
            return []

    client = FakeClient()
    mlflow_client.list_runs(client, "exp1")

    assert client.kwargs["max_results"] == 100
    assert client.kwargs["order_by"] == ["start_time DESC"]


def test_export_plot_buttons_warns_on_kaleido_failure(monkeypatch) -> None:
    """export_plot_buttons should show warning when to_image raises."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(export_buttons, "st", fake_st)

    class BrokenFigure:
        def to_image(self, fmt=None, **kwargs):
            raise RuntimeError("kaleido not installed")

    export_buttons.export_plot_buttons(BrokenFigure(), "plot")

    warning_messages = [text for kind, text in fake_st.messages if kind == "warning"]
    assert len(warning_messages) == 3  # one per format
    assert all("kaleido" in msg for msg in warning_messages)


def test_export_buttons_require_streamlit_raises_when_none(monkeypatch) -> None:
    """_require_streamlit should raise ImportError when st is None."""
    import pytest
    monkeypatch.setattr(export_buttons, "st", None)
    monkeypatch.setattr(
        export_buttons, "_STREAMLIT_IMPORT_ERROR", ImportError("no streamlit")
    )

    with pytest.raises(ImportError, match="Streamlit is required"):
        export_buttons._require_streamlit()


def test_confusion_matrix_normalize_false() -> None:
    """plot_confusion_matrix with normalize=False should return integer counts."""
    fig = plot_confusion_matrix(
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 1, 1, 2, 2, 2]),
        normalize=False,
    )
    assert fig.axes


def test_confusion_matrix_normalize_true() -> None:
    """plot_confusion_matrix with normalize=True (default) should work."""
    fig = plot_confusion_matrix(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 0]),
        normalize=True,
    )
    assert fig.axes


def test_confusion_matrix_normalize_all() -> None:
    """plot_confusion_matrix with normalize='all' should work."""
    fig = plot_confusion_matrix(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([0, 1, 2, 1, 2, 0]),
        normalize="all",
    )
    assert fig.axes


def test_confusion_matrix_with_class_names() -> None:
    """plot_confusion_matrix should accept custom class names."""
    fig = plot_confusion_matrix(
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        class_names=["A", "B", "C", "D"],
    )
    assert fig.axes


def test_overview_with_experiments_renders_dataframe(monkeypatch) -> None:
    """display_experiment_overview should render a dataframe when experiments exist."""
    fake_st = FakeStreamlit()

    exp1 = SimpleNamespace(
        lifecycle_stage="active",
        experiment_id="1",
        name="my-experiment",
        artifact_location="./mlruns/1",
        creation_time=0,
    )

    class FakeClient:
        def search_experiments(self):
            return [exp1]

        def search_runs(self, **kwargs):
            return []

    monkeypatch.setattr(overview, "st", fake_st)

    overview.display_experiment_overview(FakeClient())

    dataframe_calls = [item for item in fake_st.messages if item[0] == "dataframe"]
    assert len(dataframe_calls) >= 1
