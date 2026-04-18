from __future__ import annotations

import sys

import pytest

from mammography.apps.web_ui.pages.training import (
    config_sections,
    data_loading,
    live_metrics,
    process,
    state,
    tracking,
)
from tests.helpers import SessionState, _Context


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.messages = []

    def subheader(self, text):
        self.messages.append(("subheader", text))

    def markdown(self, text, **kwargs):
        self.messages.append(("markdown", text))

    def info(self, text):
        self.messages.append(("info", text))

    def warning(self, text):
        self.messages.append(("warning", text))

    def caption(self, text):
        self.messages.append(("caption", text))

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context(self) for _ in range(count)]

    def expander(self, *args, **kwargs):
        return _Context(self)

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index]

    def text_input(self, label, value="", **kwargs):
        return value

    def number_input(self, label, value=0, **kwargs):
        return value

    def checkbox(self, label, value=False, **kwargs):
        return value

    def metric(self, *args, **kwargs):
        self.messages.append(("metric", args, kwargs))

    def plotly_chart(self, *args, **kwargs):
        self.messages.append(("plotly_chart", args, kwargs))


def test_training_session_defaults_and_shared_sync() -> None:
    session = SessionState()

    state.ensure_session_defaults(session)

    assert session.training_status == "idle"
    assert session.training_metrics["epoch"] == 0
    assert session.training_output == []

    session._training_shared = {
        "training_status": "completed",
        "active_run_id": "a" * 32,
    }
    state.sync_shared_state(session)

    assert session.training_status == "completed"
    assert session.active_run_id == "a" * 32


def test_config_sections_renderers_require_streamlit(monkeypatch) -> None:
    monkeypatch.setattr(config_sections, "st", None)
    monkeypatch.setattr(
        config_sections,
        "_STREAMLIT_IMPORT_ERROR",
        ImportError("streamlit not found"),
    )

    for renderer in (
        config_sections.render_data_section,
        config_sections.render_model_section,
        config_sections.render_training_section,
        config_sections.render_optimization_section,
    ):
        with pytest.raises(ImportError, match="Streamlit is required"):
            renderer()


def test_build_command_line_and_normalize_command() -> None:
    config = {
        "dataset": "mamografias",
        "csv": "classificacao.csv",
        "dicom_root": "archive",
        "outdir": "outputs/custom",
        "cache_mode": "auto",
        "cache_dir": "outputs/cache",
        "arch": "efficientnet_b0",
        "classes": "multiclass",
        "pretrained": True,
        "view_specific_training": False,
        "epochs": 5,
        "batch_size": 8,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "weight_decay": 1e-4,
        "img_size": 512,
        "seed": 42,
        "device": "auto",
        "val_frac": 0.20,
        "train_backbone": False,
        "unfreeze_last_block": True,
        "warmup_epochs": 0,
        "deterministic": False,
        "allow_tf32": True,
        "fused_optim": False,
        "torch_compile": False,
        "amp": False,
        "scheduler": "auto",
        "lr_reduce_patience": 0,
        "lr_reduce_factor": 0.5,
        "lr_reduce_min_lr": 1e-7,
        "lr_reduce_cooldown": 0,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "num_workers": 4,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "loader_heuristics": True,
        "class_weights": "none",
        "class_weights_alpha": 1.0,
        "sampler_weighted": False,
        "sampler_alpha": 1.0,
        "augment": True,
        "augment_vertical": False,
        "augment_color": False,
        "augment_rotation_deg": 5.0,
        "gradcam": False,
        "save_val_preds": False,
        "export_val_embeddings": False,
        "profile": False,
        "profile_dir": "outputs/profiler",
        "tracker": "mlflow",
        "tracker_uri": "./mlruns",
        "tracker_project": "",
        "tracker_run_name": "",
        "subset": 0,
        "log_level": "info",
    }
    command = process.build_command_line(config)

    assert command[:2] == ["mammography", "train-density"]
    assert command[command.index("--dataset") + 1] == "mamografias"
    assert command[command.index("--epochs") + 1] == "5"
    assert command[command.index("--tracker-uri") + 1] == "./mlruns"

    normalized = process.normalize_training_command(command, executable="python")
    assert normalized.startswith("python -m mammography.cli train-density")

    argv = process.build_training_argv(command, executable="python")
    assert argv[:4] == ["python", "-m", "mammography.cli", "train-density"]
    assert argv[4:6] == ["--dataset", "mamografias"]
    assert "--tracker-uri" in argv
    assert "./mlruns" in argv


def test_training_section_modules_render_with_mocked_streamlit(monkeypatch) -> None:
    fake_st = FakeStreamlit()

    for module in (config_sections, data_loading, tracking):
        monkeypatch.setattr(module, "st", fake_st)

    data_config = config_sections.render_data_section()
    model_config = config_sections.render_model_section()
    loading_config = data_loading.render_data_loading_section()
    tracking_config = tracking.render_tracking_section()

    assert data_config["csv"] == "classificacao.csv"
    assert model_config["arch"] == "efficientnet_b0"
    assert loading_config["num_workers"] == 4
    assert tracking_config["tracker"] == "none"


def test_live_metrics_idle_section_renders_without_plotly(monkeypatch) -> None:
    fake_st = FakeStreamlit()
    fake_st.session_state.training_status = "idle"

    monkeypatch.setattr(live_metrics, "st", fake_st)
    monkeypatch.setattr(live_metrics, "go", None)
    monkeypatch.setattr(live_metrics, "make_subplots", None)

    live_metrics.render_live_metrics_section()

    assert any(
        item[0] == "warning" and "Install plotly" in item[1]
        for item in fake_st.messages
    )


# ---------------------------------------------------------------------------
# Additional tests for new/changed code in this PR
# ---------------------------------------------------------------------------


def test_training_session_defaults_all_keys_present() -> None:
    """ensure_session_defaults must set all expected session state keys."""
    session = SessionState()
    state.ensure_session_defaults(session)

    required_keys = [
        "training_config",
        "training_process",
        "training_output",
        "training_status",
        "training_metrics",
        "training_metrics_history",
        "mlflow_client",
        "mlflow_tracking_uri",
        "active_run_id",
        "last_metrics_poll",
    ]
    for key in required_keys:
        assert key in session, f"Key {key!r} missing from session state"


def test_training_session_defaults_idempotent() -> None:
    """ensure_session_defaults called twice must not overwrite existing values."""
    session = SessionState()
    state.ensure_session_defaults(session)

    session.training_status = "running"
    session.active_run_id = "abc123"

    state.ensure_session_defaults(session)

    assert session.training_status == "running"
    assert session.active_run_id == "abc123"


def test_training_default_metrics_are_complete() -> None:
    """DEFAULT_TRAINING_METRICS should contain all metric keys used by the UI."""
    expected_keys = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate", "samples_processed"}
    assert set(state.DEFAULT_TRAINING_METRICS.keys()) == expected_keys


def test_sync_shared_state_noop_when_no_shared() -> None:
    """sync_shared_state should do nothing when _training_shared is absent."""
    session = SessionState()
    state.ensure_session_defaults(session)

    original_status = session.training_status
    state.sync_shared_state(session)  # no _training_shared present

    assert session.training_status == original_status


def test_sync_shared_state_does_not_set_run_id_when_none() -> None:
    """sync_shared_state should not update active_run_id when shared value is None."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.active_run_id = "existing_run"

    session._training_shared = {
        "training_status": "completed",
        "active_run_id": None,
    }
    state.sync_shared_state(session)

    # Status should update but run_id should remain unchanged
    assert session.training_status == "completed"
    assert session.active_run_id == "existing_run"


def test_mlflow_polling_metric_mappings_coverage() -> None:
    """METRIC_MAPPINGS must cover the canonical train/val metric keys."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    mappings = mlflow_polling.METRIC_MAPPINGS

    assert "train/loss" in mappings
    assert "val/loss" in mappings
    assert "train/acc" in mappings
    assert "val/acc" in mappings
    assert mappings["train/loss"] == "train_loss"
    assert mappings["val/acc"] == "val_acc"
    assert mappings["lr"] == "learning_rate"


def test_mlflow_polling_get_client_returns_none_when_mlflow_unavailable(monkeypatch) -> None:
    """get_mlflow_client (training version) must return None when mlflow is not installed."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    monkeypatch.setattr(mlflow_polling, "mlflow", None)
    monkeypatch.setattr(mlflow_polling, "MlflowClient", None)

    session = SessionState()
    session.mlflow_client = None
    session.mlflow_tracking_uri = None

    result = mlflow_polling.get_mlflow_client("http://localhost:5000", session)
    assert result is None


def test_mlflow_polling_get_client_returns_none_when_mlflow_unavailable_even_with_cache(monkeypatch) -> None:
    """get_mlflow_client returns None when MLflow is unavailable even with a cached client."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    fake_client = object()
    session = SessionState()
    session.mlflow_client = fake_client
    session.mlflow_tracking_uri = "http://localhost:5000"

    monkeypatch.setattr(mlflow_polling, "mlflow", None)
    monkeypatch.setattr(mlflow_polling, "MlflowClient", None)

    result = mlflow_polling.get_mlflow_client("http://localhost:5000", session)
    assert result is None


def test_mlflow_polling_get_client_reuses_cached_when_mlflow_available(monkeypatch) -> None:
    """get_mlflow_client reuses the existing client if tracking URI matches."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    fake_client = object()
    session = SessionState()
    session.mlflow_client = fake_client
    session.mlflow_tracking_uri = "http://localhost:5000"

    class FakeMlflow:
        def set_tracking_uri(self, tracking_uri):
            raise AssertionError("cached client path should not set tracking URI")

    def fake_client_factory(*args, **kwargs):
        raise AssertionError("cached client path should not create a new client")

    monkeypatch.setattr(mlflow_polling, "mlflow", FakeMlflow())
    monkeypatch.setattr(mlflow_polling, "MlflowClient", fake_client_factory)

    result = mlflow_polling.get_mlflow_client("http://localhost:5000", session)
    assert result is fake_client


def test_poll_mlflow_metrics_returns_false_when_no_client() -> None:
    """poll_mlflow_metrics returns False when mlflow_client is None."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    session = SessionState()
    state.ensure_session_defaults(session)
    # mlflow_client defaults to None

    result = mlflow_polling.poll_mlflow_metrics(session)
    assert result is False


def test_poll_mlflow_metrics_returns_false_when_no_run_id() -> None:
    """poll_mlflow_metrics returns False when active_run_id is None."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    session = SessionState()
    state.ensure_session_defaults(session)
    session.mlflow_client = object()  # non-None client
    session.active_run_id = None  # no run

    result = mlflow_polling.poll_mlflow_metrics(session)
    assert result is False


def test_poll_mlflow_metrics_respects_throttle_interval() -> None:
    """poll_mlflow_metrics returns False if called too soon after last poll."""
    from mammography.apps.web_ui.pages.training import mlflow_polling
    import time

    session = SessionState()
    state.ensure_session_defaults(session)
    session.mlflow_client = object()
    session.active_run_id = "a" * 32
    # Set last poll to right now (throttle should kick in)
    session.last_metrics_poll = time.time()

    result = mlflow_polling.poll_mlflow_metrics(session, min_interval=10.0)
    assert result is False


def test_poll_mlflow_metrics_updates_session_on_success() -> None:
    """poll_mlflow_metrics should update training_metrics and return True on success."""
    from mammography.apps.web_ui.pages.training import mlflow_polling

    original_mlflow = mlflow_polling.mlflow

    class FakeMlflow:
        @staticmethod
        def set_tracking_uri(uri):
            pass

    class FakeRun:
        def __init__(self):
            from types import SimpleNamespace

            self.data = SimpleNamespace(
                metrics={
                    "train/loss": 0.5,
                    "val/acc": 0.8,
                }
            )

    class FakeClient:
        def get_run(self, run_id):
            return FakeRun()

        def get_metric_history(self, run_id, metric_key):
            return []

    mlflow_polling.mlflow = FakeMlflow()

    session = SessionState()
    state.ensure_session_defaults(session)
    session.mlflow_client = FakeClient()
    session.active_run_id = "a" * 32
    session.last_metrics_poll = 0.0  # ancient poll time

    try:
        result = mlflow_polling.poll_mlflow_metrics(session, min_interval=0.0)
        assert result is True
        assert session.training_metrics["train_loss"] == 0.5
        assert session.training_metrics["val_acc"] == 0.8
    finally:
        mlflow_polling.mlflow = original_mlflow


def test_build_command_line_with_optional_flags() -> None:
    """build_command_line should include flags for non-default optional settings."""
    config = {
        "dataset": "",
        "csv": "classificacao.csv",
        "dicom_root": "archive",
        "outdir": "outputs/run",
        "cache_mode": "auto",
        "cache_dir": "outputs/cache",
        "embeddings_dir": "",
        "arch": "efficientnet_b0",
        "classes": "multiclass",
        "pretrained": False,  # non-default → --no-pretrained
        "view_specific_training": False,
        "epochs": 100,
        "batch_size": 16,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "weight_decay": 1e-4,
        "img_size": 512,
        "seed": 42,
        "device": "auto",
        "val_frac": 0.20,
        "train_backbone": True,  # non-default → --train-backbone
        "unfreeze_last_block": False,  # non-default → --no-unfreeze-last-block
        "warmup_epochs": 5,  # non-default → --warmup-epochs 5
        "deterministic": True,  # non-default → --deterministic
        "allow_tf32": False,  # non-default → --no-allow-tf32
        "fused_optim": True,  # non-default → --fused-optim
        "torch_compile": True,  # non-default → --torch-compile
        "amp": True,  # non-default → --amp
        "scheduler": "cosine",  # non-default → --scheduler cosine
        "lr_reduce_patience": 0,
        "lr_reduce_factor": 0.5,
        "lr_reduce_min_lr": 1e-7,
        "lr_reduce_cooldown": 0,
        "early_stop_patience": 5,  # non-default → --early-stop-patience 5
        "early_stop_min_delta": 0.001,  # non-default
        "num_workers": 4,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "loader_heuristics": True,
        "class_weights": "auto",  # non-default → --class-weights auto
        "class_weights_alpha": 1.0,
        "sampler_weighted": True,  # non-default → --sampler-weighted
        "sampler_alpha": 1.0,
        "augment": False,  # non-default → --no-augment
        "augment_vertical": False,
        "augment_color": False,
        "augment_rotation_deg": 5.0,
        "gradcam": True,  # non-default → --gradcam
        "gradcam_limit": 4,
        "save_val_preds": True,  # non-default → --save-val-preds
        "export_val_embeddings": True,  # non-default → --export-val-embeddings
        "profile": False,
        "profile_dir": "outputs/profiler",
        "tracker": "none",
        "tracker_project": "",
        "tracker_run_name": "",
        "tracker_uri": "",
        "subset": 100,  # non-default → --subset 100
        "log_level": "debug",  # non-default → --log-level debug
    }
    command = process.build_command_line(config)

    assert "--no-pretrained" in command
    assert "--train-backbone" in command
    assert "--no-unfreeze-last-block" in command
    assert command[command.index("--warmup-epochs") + 1] == "5"
    assert "--deterministic" in command
    assert "--no-allow-tf32" in command
    assert "--fused-optim" in command
    assert "--torch-compile" in command
    assert "--amp" in command
    assert command[command.index("--scheduler") + 1] == "cosine"
    assert command[command.index("--early-stop-patience") + 1] == "5"
    assert command[command.index("--class-weights") + 1] == "auto"
    assert "--sampler-weighted" in command
    assert "--no-augment" in command
    assert "--gradcam" in command
    assert "--save-val-preds" in command
    assert "--export-val-embeddings" in command
    assert command[command.index("--subset") + 1] == "100"
    assert command[command.index("--log-level") + 1] == "debug"


def test_build_command_line_minimal_defaults() -> None:
    """build_command_line with all default values should produce minimal command."""
    config = {
        "dataset": "",
        "csv": "classificacao.csv",
        "dicom_root": "archive",
        "outdir": "outputs/run",
        "cache_mode": "auto",
        "cache_dir": "outputs/cache",
        "embeddings_dir": "",
        "arch": "efficientnet_b0",
        "classes": "multiclass",
        "pretrained": True,
        "view_specific_training": False,
        "epochs": 100,
        "batch_size": 16,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "weight_decay": 1e-4,
        "img_size": 512,
        "seed": 42,
        "device": "auto",
        "val_frac": 0.20,
        "train_backbone": False,
        "unfreeze_last_block": True,
        "warmup_epochs": 0,
        "deterministic": False,
        "allow_tf32": True,
        "fused_optim": False,
        "torch_compile": False,
        "amp": False,
        "scheduler": "auto",
        "lr_reduce_patience": 0,
        "lr_reduce_factor": 0.5,
        "lr_reduce_min_lr": 1e-7,
        "lr_reduce_cooldown": 0,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "num_workers": 4,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "loader_heuristics": True,
        "class_weights": "none",
        "class_weights_alpha": 1.0,
        "sampler_weighted": False,
        "sampler_alpha": 1.0,
        "augment": True,
        "augment_vertical": False,
        "augment_color": False,
        "augment_rotation_deg": 5.0,
        "gradcam": False,
        "gradcam_limit": 4,
        "save_val_preds": False,
        "export_val_embeddings": False,
        "profile": False,
        "profile_dir": "outputs/profiler",
        "tracker": "none",
        "tracker_project": "",
        "tracker_run_name": "",
        "tracker_uri": "",
        "subset": 0,
        "log_level": "info",
    }
    command = process.build_command_line(config)
    assert command == ["mammography", "train-density"]


def test_normalize_training_command_default_executable() -> None:
    """normalize_training_command with default executable should use sys.executable."""
    import sys as _sys

    command = "mammography train-density --epochs 10"
    normalized = process.normalize_training_command(command)

    executable = _sys.executable
    expected_prefix = f'"{executable}" -m mammography.cli train-density'
    assert normalized.startswith(expected_prefix)


def test_build_training_argv_rejects_extra_tokens() -> None:
    """build_training_argv should reject tokens outside known train options."""
    with pytest.raises(ValueError, match="Unsupported training command argument"):
        process.build_training_argv("mammography train-density --dataset safe && echo bad")


def test_launch_training_handles_invalid_training_command() -> None:
    """launch_training should fail cleanly when argv validation rejects the command."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_config = {"tracker": "none"}

    process.launch_training(
        "mammography train-density --dataset safe && echo bad",
        session_state=session,
        get_mlflow_client=lambda uri: None,
        popen=lambda cmd, **kwargs: pytest.fail("popen should not be called"),
    )

    assert session.training_status == "failed"
    assert session.active_run_id is None
    assert session.training_process is None
    assert any(
        "Unsupported training command argument" in line
        for line in session.training_output
    )


def test_stop_training_with_running_process() -> None:
    """stop_training should terminate the process and update session state."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_status = "running"

    terminate_calls = []

    class FakeProcess:
        def terminate(self):
            terminate_calls.append(True)

    session.training_process = FakeProcess()

    process.stop_training(session)

    assert terminate_calls == [True]
    assert session.training_process is None
    assert session.training_status == "failed"
    assert any("stopped by user" in line for line in session.training_output)


def test_stop_training_when_no_process() -> None:
    """stop_training should be safe to call when no process is running."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_process = None
    session.training_status = "idle"

    # Should not raise
    process.stop_training(session)

    # Status unchanged since training_process was None
    assert session.training_status == "idle"


def test_stop_training_clears_finished_process_without_terminating() -> None:
    """stop_training should not overwrite status after the worker marks completion."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_status = "completed"
    session._training_shared = {
        "training_status": "completed",
        "active_run_id": None,
        "training_output": [],
        "finished": True,
    }

    class FakeProcess:
        def terminate(self):
            raise AssertionError("finished process should not be terminated")

    session.training_process = FakeProcess()

    process.stop_training(session)

    assert session.training_process is None
    assert session.training_status == "completed"


def test_launch_training_updates_status_to_running(monkeypatch) -> None:
    """launch_training should set training_status to 'running' immediately."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_config = {"tracker": "none"}

    launched_commands = []

    class FakeProcess:
        returncode = 0

        class stdout:
            @staticmethod
            def __iter__():
                return iter([])

        def wait(self):
            pass

    def fake_popen(cmd, **kwargs):
        assert isinstance(cmd, list)
        assert kwargs["shell"] is False
        launched_commands.append(cmd)
        return FakeProcess()

    class DeferredThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

    monkeypatch.setattr(process.threading, "Thread", DeferredThread)

    process.launch_training(
        "mammography train-density",
        session_state=session,
        get_mlflow_client=lambda uri: None,
        popen=fake_popen,
    )

    assert session.training_status == "running"
    assert len(launched_commands) == 1
    assert launched_commands[0][:4] == [
        sys.executable,
        "-m",
        "mammography.cli",
        "train-density",
    ]


def test_launch_training_syncs_background_updates_to_session(monkeypatch) -> None:
    """Background output updates should flow through the shared training state."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_config = {"tracker": "none"}
    run_id = "a" * 32

    class FakeProcess:
        returncode = 0

        def __init__(self):
            self.stdout = iter([f"MLFLOW_RUN_ID:{run_id}\n"])

        def wait(self):
            pass

    class ImmediateThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    monkeypatch.setattr(process.threading, "Thread", ImmediateThread)

    process.launch_training(
        "mammography train-density",
        session_state=session,
        get_mlflow_client=lambda uri: None,
        popen=lambda cmd, **kwargs: FakeProcess(),
    )

    assert session._training_shared["active_run_id"] == run_id
    assert session._training_shared["training_status"] == "completed"
    assert session._training_shared["finished"] is True
    assert session._training_shared["training_output"][0] == f"MLFLOW_RUN_ID:{run_id}"
    assert session.active_run_id is None
    assert session.training_status == "running"
    assert session.training_output == []
    assert session.training_process is not None

    state.sync_shared_state(session)

    assert session.active_run_id == run_id
    assert session.training_status == "completed"
    assert session.training_output[0] == f"MLFLOW_RUN_ID:{run_id}"
    assert session.training_process is None


def test_launch_training_resets_output() -> None:
    """launch_training should clear previous training_output before starting."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_config = {"tracker": "none"}
    session.training_output = ["old line 1", "old line 2"]

    class FakeProcess:
        returncode = 0

        class stdout:
            @staticmethod
            def __iter__():
                return iter([])

        def wait(self):
            pass

    def fake_popen(cmd, **kwargs):
        return FakeProcess()

    process.launch_training(
        "mammography train-density",
        session_state=session,
        get_mlflow_client=lambda uri: None,
        popen=fake_popen,
    )

    # Output is reset at the start of launch_training
    # (background thread may later add lines, but old_output should be gone)
    assert "old line 1" not in session.training_output
    assert "old line 2" not in session.training_output


def test_launch_training_handles_popen_failure() -> None:
    """launch_training should handle failures when Popen raises an exception."""
    session = SessionState()
    state.ensure_session_defaults(session)
    session.training_config = {"tracker": "none"}

    def failing_popen(cmd, **kwargs):
        raise OSError("Command not found")

    process.launch_training(
        "mammography train-density",
        session_state=session,
        get_mlflow_client=lambda uri: None,
        popen=failing_popen,
    )

    assert session.training_status == "failed"
    assert any("Failed to start" in line for line in session.training_output)


def test_data_loading_augmentation_disabled(monkeypatch) -> None:
    """render_augmentation_section with augment=False should set augment_* to False/0."""
    fake_st = FakeStreamlit()
    # Override checkbox to return False for "Enable Augmentation"
    call_count = [0]

    def fake_checkbox(label, value=False, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return False  # Enable Augmentation = False
        return value

    fake_st.checkbox = fake_checkbox
    monkeypatch.setattr(data_loading, "st", fake_st)

    config = data_loading.render_augmentation_section()

    assert config["augment"] is False
    assert config["augment_vertical"] is False
    assert config["augment_color"] is False
    assert config["augment_rotation_deg"] == 0.0


def test_data_loading_class_balancing_defaults(monkeypatch) -> None:
    """render_class_balancing_section should return expected defaults."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(data_loading, "st", fake_st)

    config = data_loading.render_class_balancing_section()

    assert config["class_weights"] == "none"
    assert config["class_weights_alpha"] == 1.0
    assert config["sampler_weighted"] is False
    assert config["sampler_alpha"] == 1.0


def test_tracking_render_analysis_defaults(monkeypatch) -> None:
    """render_analysis_section should return all expected keys with defaults."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(tracking, "st", fake_st)

    config = tracking.render_analysis_section()

    assert "gradcam" in config
    assert "gradcam_limit" in config
    assert "save_val_preds" in config
    assert "export_val_embeddings" in config
    assert "profile" in config
    assert "profile_dir" in config
    assert config["gradcam"] is False
    assert config["save_val_preds"] is False
    assert config["export_val_embeddings"] is False
    assert config["profile_dir"] == "outputs/profiler"


def test_config_sections_render_optimization_defaults(monkeypatch) -> None:
    """render_optimization_section should return expected keys and defaults."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(config_sections, "st", fake_st)

    config = config_sections.render_optimization_section()

    assert "train_backbone" in config
    assert "unfreeze_last_block" in config
    assert "warmup_epochs" in config
    assert "deterministic" in config
    assert "allow_tf32" in config
    assert "fused_optim" in config
    assert "torch_compile" in config
    assert "scheduler" in config
    assert config["train_backbone"] is False
    assert config["allow_tf32"] is True
    assert config["scheduler"] == "auto"


def test_config_sections_render_training_defaults(monkeypatch) -> None:
    """render_training_section should return expected hyperparameter defaults."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(config_sections, "st", fake_st)

    config = config_sections.render_training_section()

    assert config["epochs"] == 100
    assert config["batch_size"] == 16
    assert config["lr"] == 1e-4
    assert config["device"] == "auto"
    assert config["amp"] is False
