"""Unit tests for LocalTracker core experiment tracking API."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.tracking import LocalTracker


class TestLocalTrackerInitialization:
    """Test LocalTracker initialization and database setup."""

    def test_basic_initialization(self, tmp_path):
        """Test basic LocalTracker initialization."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        assert tracker.db_path == db_path
        assert tracker.experiment_name == "test_experiment"
        assert tracker.run_name == "test_run"
        assert tracker.experiment_id is not None
        assert tracker.run_id is not None
        assert db_path.exists()

    def test_initialization_without_run_name(self, tmp_path):
        """Test initialization without explicit run name."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        assert tracker.run_name is not None
        assert tracker.run_name.startswith("run_")
        assert tracker.run_id is not None

    def test_initialization_with_params(self, tmp_path):
        """Test initialization with hyperparameters."""
        db_path = tmp_path / "test_experiments.db"
        params = {"lr": 0.001, "batch_size": 32, "epochs": 10}
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
            params=params,
        )

        run_info = tracker.get_run_info()
        assert run_info is not None
        assert run_info["params"] == params

    def test_database_creation(self, tmp_path):
        """Test that database file is created with correct schema."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        assert db_path.exists()
        # Verify tables exist by accessing the database
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "experiments", "runs", "metrics", "artifacts",
            "studies", "trials", "trial_intermediate_values"
        }
        assert expected_tables.issubset(tables)

    def test_nested_db_path_creation(self, tmp_path):
        """Test that nested database paths are created."""
        db_path = tmp_path / "nested" / "deep" / "path" / "test.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        assert db_path.exists()
        assert db_path.parent.exists()

    def test_default_experiment_name(self, tmp_path):
        """Test initialization with default experiment name."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(db_path=db_path)

        assert tracker.experiment_name == "default"


class TestLocalTrackerExperiments:
    """Test experiment management functionality."""

    def test_create_experiment(self, tmp_path):
        """Test creating a new experiment."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="experiment1",
        )

        assert tracker.experiment_id is not None
        assert tracker.experiment_name == "experiment1"

    def test_experiment_reuse(self, tmp_path):
        """Test that existing experiments are reused."""
        db_path = tmp_path / "test_experiments.db"

        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="shared_experiment",
        )
        exp_id1 = tracker1.experiment_id

        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="shared_experiment",
        )
        exp_id2 = tracker2.experiment_id

        # Should reuse same experiment
        assert exp_id1 == exp_id2

    def test_multiple_experiments(self, tmp_path):
        """Test creating multiple experiments in same database."""
        db_path = tmp_path / "test_experiments.db"

        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="experiment1",
        )
        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="experiment2",
        )

        assert tracker1.experiment_id != tracker2.experiment_id
        assert tracker1.experiment_name == "experiment1"
        assert tracker2.experiment_name == "experiment2"


class TestLocalTrackerMetrics:
    """Test metrics logging and retrieval."""

    def test_log_metrics_basic(self, tmp_path):
        """Test basic metric logging."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        metrics = {"loss": 0.5, "accuracy": 0.95}
        tracker.log_metrics(metrics, step=0)

        logged_metrics = tracker.get_metrics()
        assert len(logged_metrics) == 2

        # Verify metric values
        metric_dict = {m["key"]: m["value"] for m in logged_metrics}
        assert metric_dict["loss"] == 0.5
        assert metric_dict["accuracy"] == 0.95

    def test_log_metrics_multiple_steps(self, tmp_path):
        """Test logging metrics across multiple steps."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        # Log metrics for multiple steps
        for step in range(5):
            metrics = {"loss": 1.0 / (step + 1), "accuracy": 0.5 + (step * 0.1)}
            tracker.log_metrics(metrics, step=step)

        logged_metrics = tracker.get_metrics()
        assert len(logged_metrics) == 10  # 2 metrics × 5 steps

        # Verify steps are in order
        steps = [m["step"] for m in logged_metrics]
        assert sorted(steps) == steps

    def test_log_metrics_empty_dict(self, tmp_path):
        """Test logging empty metrics dictionary."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        tracker.log_metrics({}, step=0)
        logged_metrics = tracker.get_metrics()
        assert len(logged_metrics) == 0

    def test_log_metrics_numeric_conversion(self, tmp_path):
        """Test that numeric values are properly converted to float."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        metrics = {"int_metric": 42, "float_metric": 3.14, "bool_metric": True}
        tracker.log_metrics(metrics, step=0)

        logged_metrics = tracker.get_metrics()
        metric_dict = {m["key"]: m["value"] for m in logged_metrics}

        assert metric_dict["int_metric"] == 42.0
        assert metric_dict["float_metric"] == 3.14
        assert metric_dict["bool_metric"] == 1.0

    def test_get_metrics_for_specific_run(self, tmp_path):
        """Test retrieving metrics for a specific run."""
        db_path = tmp_path / "test_experiments.db"

        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run1",
        )
        tracker1.log_metrics({"loss": 0.5}, step=0)
        run1_id = tracker1.run_id

        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run2",
        )
        tracker2.log_metrics({"loss": 0.3}, step=0)

        # Get metrics for specific run
        run1_metrics = tracker1.get_metrics(run_id=run1_id)
        assert len(run1_metrics) == 1
        assert run1_metrics[0]["value"] == 0.5

    def test_get_metrics_ordering(self, tmp_path):
        """Test that metrics are returned in correct order."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        # Log metrics in reverse order
        for step in [2, 1, 0]:
            tracker.log_metrics({"loss": float(step)}, step=step)

        logged_metrics = tracker.get_metrics()
        steps = [m["step"] for m in logged_metrics]

        # Should be sorted by step
        assert steps == [0, 1, 2]


class TestLocalTrackerArtifacts:
    """Test artifact logging functionality."""

    def test_log_artifact_basic(self, tmp_path):
        """Test basic artifact logging."""
        db_path = tmp_path / "test_experiments.db"
        artifact_file = tmp_path / "test_model.pt"
        artifact_file.write_text("fake model data")

        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        tracker.log_artifact(artifact_file)

        # Verify artifact was logged
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, path, artifact_type FROM artifacts WHERE run_id = ?",
                (tracker.run_id,)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "test_model.pt"
        assert row[1] == str(artifact_file.absolute())
        assert row[2] == "model"

    def test_log_artifact_with_custom_name(self, tmp_path):
        """Test logging artifact with custom name."""
        db_path = tmp_path / "test_experiments.db"
        artifact_file = tmp_path / "checkpoint.pth"
        artifact_file.write_text("checkpoint data")

        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        tracker.log_artifact(artifact_file, name="best_model")

        # Verify custom name
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM artifacts WHERE run_id = ?",
                (tracker.run_id,)
            )
            row = cursor.fetchone()

        assert row[0] == "best_model"

    def test_log_artifact_type_detection(self, tmp_path):
        """Test artifact type detection based on file extension."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        # Create different artifact types
        model_file = tmp_path / "model.pt"
        model_file.write_text("model")
        checkpoint_file = tmp_path / "checkpoint.pth"
        checkpoint_file.write_text("checkpoint")
        other_file = tmp_path / "config.json"
        other_file.write_text("{}")

        tracker.log_artifact(model_file)
        tracker.log_artifact(checkpoint_file)
        tracker.log_artifact(other_file)

        # Verify artifact types
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, artifact_type FROM artifacts WHERE run_id = ? ORDER BY name",
                (tracker.run_id,)
            )
            artifacts = cursor.fetchall()

        artifact_types = {name: atype for name, atype in artifacts}
        assert artifact_types["checkpoint.pth"] == "model"
        assert artifact_types["config.json"] == "file"
        assert artifact_types["model.pt"] == "model"

    def test_log_artifact_nonexistent_file(self, tmp_path):
        """Test logging artifact that doesn't exist."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        nonexistent_file = tmp_path / "does_not_exist.pt"
        tracker.log_artifact(nonexistent_file)  # Should not raise

        # Verify no artifact was logged
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM artifacts WHERE run_id = ?",
                (tracker.run_id,)
            )
            count = cursor.fetchone()[0]

        assert count == 0


class TestLocalTrackerRunManagement:
    """Test run management functionality."""

    def test_finish_run(self, tmp_path):
        """Test marking a run as finished."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        tracker.finish()

        run_info = tracker.get_run_info()
        assert run_info is not None
        assert run_info["status"] == "FINISHED"
        assert run_info["finished_at"] is not None

    def test_get_run_info_basic(self, tmp_path):
        """Test getting basic run information."""
        db_path = tmp_path / "test_experiments.db"
        params = {"lr": 0.001, "epochs": 10}
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
            params=params,
        )

        run_info = tracker.get_run_info()
        assert run_info is not None
        assert run_info["name"] == "test_run"
        assert run_info["experiment_name"] == "test_experiment"
        assert run_info["params"] == params
        assert run_info["status"] == "RUNNING"
        assert run_info["started_at"] is not None
        assert run_info["run_id"] == tracker.run_id

    def test_get_run_info_for_specific_run(self, tmp_path):
        """Test getting info for a specific run ID."""
        db_path = tmp_path / "test_experiments.db"

        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run1",
        )
        run1_id = tracker1.run_id

        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run2",
        )

        # Get info for run1 from tracker2
        run1_info = tracker2.get_run_info(run_id=run1_id)
        assert run1_info is not None
        assert run1_info["name"] == "run1"
        assert run1_info["run_id"] == run1_id

    def test_list_runs_basic(self, tmp_path):
        """Test listing runs for an experiment."""
        db_path = tmp_path / "test_experiments.db"

        # Create multiple runs
        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run1",
        )
        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run2",
        )
        tracker3 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run3",
        )

        runs = tracker1.list_runs()
        assert len(runs) == 3

        run_names = {run["name"] for run in runs}
        assert run_names == {"run1", "run2", "run3"}

    def test_list_runs_for_specific_experiment(self, tmp_path):
        """Test listing runs for a specific experiment."""
        db_path = tmp_path / "test_experiments.db"

        # Create runs in different experiments
        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="experiment1",
            run_name="run1",
        )
        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="experiment2",
            run_name="run2",
        )

        # List runs for experiment1
        runs = tracker1.list_runs(experiment_name="experiment1")
        assert len(runs) == 1
        assert runs[0]["name"] == "run1"

        # List runs for experiment2
        runs = tracker2.list_runs(experiment_name="experiment2")
        assert len(runs) == 1
        assert runs[0]["name"] == "run2"

    def test_list_runs_ordering(self, tmp_path):
        """Test that runs are listed in reverse chronological order."""
        db_path = tmp_path / "test_experiments.db"

        # Create runs in sequence
        import time
        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run1",
        )
        time.sleep(0.01)  # Ensure different timestamps
        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run2",
        )
        time.sleep(0.01)
        tracker3 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run3",
        )

        runs = tracker1.list_runs()
        run_names = [run["name"] for run in runs]

        # Should be in reverse order (newest first)
        assert run_names == ["run3", "run2", "run1"]


class TestLocalTrackerEdgeCases:
    """Test edge cases and error handling."""

    def test_log_metrics_with_none_run_id(self, tmp_path):
        """Test logging metrics when run_id is None."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        # Manually set run_id to None
        tracker.run_id = None

        # Should not raise error
        tracker.log_metrics({"loss": 0.5}, step=0)

        # Restore run_id
        tracker.run_id = tracker._create_run("recovery_run", {})
        logged_metrics = tracker.get_metrics()
        assert len(logged_metrics) == 0

    def test_log_artifact_with_none_run_id(self, tmp_path):
        """Test logging artifact when run_id is None."""
        db_path = tmp_path / "test_experiments.db"
        artifact_file = tmp_path / "test.pt"
        artifact_file.write_text("data")

        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        # Manually set run_id to None
        tracker.run_id = None

        # Should not raise error
        tracker.log_artifact(artifact_file)

    def test_finish_with_none_run_id(self, tmp_path):
        """Test finishing run when run_id is None."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        # Manually set run_id to None
        tracker.run_id = None

        # Should not raise error
        tracker.finish()

    def test_get_metrics_with_none_run_id(self, tmp_path):
        """Test getting metrics when run_id is None."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        # Manually set run_id to None
        tracker.run_id = None

        metrics = tracker.get_metrics()
        assert metrics == []

    def test_get_run_info_with_none_run_id(self, tmp_path):
        """Test getting run info when run_id is None."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        # Manually set run_id to None
        tracker.run_id = None

        run_info = tracker.get_run_info()
        assert run_info is None

    def test_get_run_info_nonexistent_run(self, tmp_path):
        """Test getting info for nonexistent run."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
        )

        run_info = tracker.get_run_info(run_id=99999)
        assert run_info is None

    def test_concurrent_tracker_instances(self, tmp_path):
        """Test multiple tracker instances accessing same database."""
        db_path = tmp_path / "test_experiments.db"

        tracker1 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run1",
        )
        tracker2 = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="run2",
        )

        # Log metrics from both trackers
        tracker1.log_metrics({"loss": 0.5}, step=0)
        tracker2.log_metrics({"loss": 0.3}, step=0)

        # Each should see only their own metrics
        metrics1 = tracker1.get_metrics()
        metrics2 = tracker2.get_metrics()

        assert len(metrics1) == 1
        assert len(metrics2) == 1
        assert metrics1[0]["value"] == 0.5
        assert metrics2[0]["value"] == 0.3

    def test_log_metrics_with_non_numeric_values(self, tmp_path):
        """Test logging metrics with non-numeric values."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        # Mix numeric and non-numeric values
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "invalid_string": "not_a_number",
            "invalid_none": None,
        }
        tracker.log_metrics(metrics, step=0)

        # Should only log numeric values
        logged_metrics = tracker.get_metrics()
        metric_keys = {m["key"] for m in logged_metrics}

        assert "loss" in metric_keys
        assert "accuracy" in metric_keys
        assert "invalid_string" not in metric_keys
        assert "invalid_none" not in metric_keys

    def test_log_metrics_skips_non_finite_values(self, tmp_path):
        """Test logging metrics with NaN/Inf values."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
        )

        metrics = {
            "loss": 0.5,
            "val_loss": float("nan"),
            "val_auc": float("inf"),
            "val_f1": float("-inf"),
        }
        tracker.log_metrics(metrics, step=0)

        logged_metrics = tracker.get_metrics()
        metric_keys = {m["key"] for m in logged_metrics}

        assert "loss" in metric_keys
        assert "val_loss" not in metric_keys
        assert "val_auc" not in metric_keys
        assert "val_f1" not in metric_keys

    def test_params_with_complex_types(self, tmp_path):
        """Test params with complex data types."""
        db_path = tmp_path / "test_experiments.db"
        params = {
            "lr": 0.001,
            "batch_size": 32,
            "architecture": "resnet50",
            "layers": [64, 128, 256],
            "config": {"nested": "value"},
        }
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_experiment",
            run_name="test_run",
            params=params,
        )

        run_info = tracker.get_run_info()
        assert run_info is not None
        # Complex types should be preserved through JSON serialization
        assert run_info["params"]["lr"] == 0.001
        assert run_info["params"]["architecture"] == "resnet50"
        assert run_info["params"]["layers"] == [64, 128, 256]
        assert run_info["params"]["config"] == {"nested": "value"}
