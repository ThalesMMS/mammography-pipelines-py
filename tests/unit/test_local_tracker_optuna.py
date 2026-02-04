"""Unit tests for LocalTracker Optuna integration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.tracking import LocalTracker


class TestLocalTrackerOptunaIntegration:
    """Test LocalTracker integration with Optuna study storage."""

    def test_save_study(self, tmp_path):
        """Test saving an Optuna study."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Save a study
        study_id = tracker.save_study("test_study", direction="maximize")
        assert study_id > 0

        # Verify study was created
        study_info = tracker.get_study_info("test_study")
        assert study_info is not None
        assert study_info["study_name"] == "test_study"
        assert study_info["direction"] == "maximize"

    def test_save_trial(self, tmp_path):
        """Test saving an Optuna trial."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Create study
        tracker.save_study("test_study", direction="maximize")

        # Save a trial
        trial_params = {"lr": 0.001, "batch_size": 32}
        trial_id = tracker.save_trial(
            study_name="test_study",
            trial_number=0,
            state="COMPLETE",
            params=trial_params,
            value=0.95,
            duration=120.5,
        )
        assert trial_id > 0

        # Verify trial was saved
        study_info = tracker.get_study_info("test_study")
        assert len(study_info["trials"]) == 1
        trial = study_info["trials"][0]
        assert trial["number"] == 0
        assert trial["state"] == "COMPLETE"
        assert trial["value"] == 0.95
        assert trial["params"] == trial_params
        assert trial["duration"] == 120.5

    def test_save_trial_intermediate_values(self, tmp_path):
        """Test saving intermediate values for pruning."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Create study and trial
        tracker.save_study("test_study", direction="maximize")
        tracker.save_trial(
            study_name="test_study",
            trial_number=0,
            state="RUNNING",
            params={"lr": 0.001},
        )

        # Save intermediate values
        tracker.save_trial_intermediate_value("test_study", 0, step=0, value=0.7)
        tracker.save_trial_intermediate_value("test_study", 0, step=1, value=0.8)
        tracker.save_trial_intermediate_value("test_study", 0, step=2, value=0.85)

        # Note: We don't have a method to retrieve intermediate values separately yet,
        # but we can verify they were saved without errors

    def test_get_study_best_trial(self, tmp_path):
        """Test retrieving best trial from study."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Create study with multiple trials
        tracker.save_study("test_study", direction="maximize")
        tracker.save_trial("test_study", 0, "COMPLETE", {"lr": 0.001}, value=0.85)
        tracker.save_trial("test_study", 1, "COMPLETE", {"lr": 0.002}, value=0.90)
        tracker.save_trial("test_study", 2, "COMPLETE", {"lr": 0.003}, value=0.88)
        tracker.save_trial("test_study", 3, "PRUNED", {"lr": 0.004}, value=None)

        # Get study info
        study_info = tracker.get_study_info("test_study")
        assert study_info["best_trial"] == 1  # Trial 1 has highest value
        assert study_info["best_value"] == 0.90
        assert study_info["n_trials"] == 4

    def test_get_study_best_trial_minimize(self, tmp_path):
        """Test retrieving best trial for minimization."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Create study with minimize direction
        tracker.save_study("test_study", direction="minimize")
        tracker.save_trial("test_study", 0, "COMPLETE", {"lr": 0.001}, value=0.85)
        tracker.save_trial("test_study", 1, "COMPLETE", {"lr": 0.002}, value=0.90)
        tracker.save_trial("test_study", 2, "COMPLETE", {"lr": 0.003}, value=0.80)

        # Get study info
        study_info = tracker.get_study_info("test_study")
        assert study_info["best_trial"] == 2  # Trial 2 has lowest value
        assert study_info["best_value"] == 0.80

    def test_list_studies(self, tmp_path):
        """Test listing all studies."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Create multiple studies
        tracker.save_study("study1", direction="maximize")
        tracker.save_study("study2", direction="minimize")
        tracker.save_trial("study1", 0, "COMPLETE", {"lr": 0.001}, value=0.85)
        tracker.save_trial("study2", 0, "COMPLETE", {"lr": 0.002}, value=0.90)
        tracker.save_trial("study2", 1, "PRUNED", {"lr": 0.003}, value=None)

        # List studies
        studies = tracker.list_studies()
        assert len(studies) == 2

        # Verify study details
        study_names = {s["study_name"] for s in studies}
        assert study_names == {"study1", "study2"}

        # Find study2 and check trial count
        study2 = next(s for s in studies if s["study_name"] == "study2")
        assert study2["n_trials"] == 2
        assert study2["direction"] == "minimize"

    def test_study_not_found(self, tmp_path):
        """Test error handling for non-existent study."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Try to get info for non-existent study
        study_info = tracker.get_study_info("nonexistent")
        assert study_info is None

    def test_trial_without_study(self, tmp_path):
        """Test error handling when saving trial without study."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Try to save trial without creating study first
        with pytest.raises(ValueError, match="Study not found"):
            tracker.save_trial(
                study_name="nonexistent",
                trial_number=0,
                state="COMPLETE",
                params={"lr": 0.001},
                value=0.85,
            )

    def test_study_idempotency(self, tmp_path):
        """Test that saving same study twice doesn't create duplicates."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Save study twice
        study_id1 = tracker.save_study("test_study", direction="maximize")
        study_id2 = tracker.save_study("test_study", direction="maximize")

        # Should return same study ID
        assert study_id1 == study_id2

        # Should only have one study
        studies = tracker.list_studies()
        assert len(studies) == 1

    def test_trial_update(self, tmp_path):
        """Test updating an existing trial."""
        db_path = tmp_path / "test_experiments.db"
        tracker = LocalTracker(
            db_path=db_path,
            experiment_name="test_optuna",
            run_name="test_run",
        )

        # Create study and trial
        tracker.save_study("test_study", direction="maximize")
        trial_id1 = tracker.save_trial(
            study_name="test_study",
            trial_number=0,
            state="RUNNING",
            params={"lr": 0.001},
        )

        # Update the same trial
        trial_id2 = tracker.save_trial(
            study_name="test_study",
            trial_number=0,
            state="COMPLETE",
            params={"lr": 0.001},
            value=0.95,
            duration=120.0,
        )

        # Should be same trial ID
        assert trial_id1 == trial_id2

        # Verify updated state
        study_info = tracker.get_study_info("test_study")
        assert len(study_info["trials"]) == 1
        trial = study_info["trials"][0]
        assert trial["state"] == "COMPLETE"
        assert trial["value"] == 0.95
