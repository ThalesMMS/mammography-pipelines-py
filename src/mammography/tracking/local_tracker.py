#!/usr/bin/env python3
#
# local_tracker.py
# mammography-pipelines
#
# SQLite-based experiment tracker for offline/air-gapped environments.
# Implements the same interface as MLflow/W&B trackers.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""LocalTracker: SQLite-based experiment tracking for offline use."""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LocalTracker:
    """
    Local experiment tracker using SQLite.

    Provides offline experiment tracking compatible with the ExperimentTracker interface.
    Stores experiments, runs, metrics, and artifacts in a local SQLite database.

    Args:
        db_path: Path to SQLite database file
        experiment_name: Name of the experiment (project)
        run_name: Optional name for the run
        params: Optional dictionary of hyperparameters to log
    """

    def __init__(
        self,
        db_path: Path,
        experiment_name: str = "default",
        run_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.run_id: Optional[int] = None
        self.experiment_id: Optional[int] = None

        # Initialize database
        self._init_db()

        # Create experiment and run
        self.experiment_id = self._create_experiment(experiment_name)
        self.run_id = self._create_run(self.run_name, params or {})

        logger.info(
            f"LocalTracker initialized: db={self.db_path}, "
            f"experiment={experiment_name}, run={self.run_name}"
        )

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    params TEXT,
                    status TEXT DEFAULT 'RUNNING',
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            # Create index for faster metric queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_run_key_step
                ON metrics(run_id, key, step)
            """)

            # Artifacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    artifact_type TEXT DEFAULT 'file',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            # Optuna studies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_name TEXT UNIQUE NOT NULL,
                    direction TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Optuna trials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id INTEGER NOT NULL,
                    trial_number INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    value REAL,
                    params TEXT NOT NULL,
                    duration REAL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id),
                    UNIQUE(study_id, trial_number)
                )
            """)

            # Optuna trial intermediate values (for pruning)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trial_intermediate_values (
                    intermediate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    value REAL NOT NULL,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
                    UNIQUE(trial_id, step)
                )
            """)

            conn.commit()

    def _create_experiment(self, name: str) -> int:
        """Create or get existing experiment."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Try to get existing experiment
            cursor.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            # Create new experiment
            cursor.execute(
                "INSERT INTO experiments (name) VALUES (?)",
                (name,)
            )
            conn.commit()
            return cursor.lastrowid

    def _create_run(self, name: str, params: dict[str, Any]) -> int:
        """Create a new run."""
        params_json = json.dumps(params, default=str)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (experiment_id, name, params, status)
                VALUES (?, ?, ?, 'RUNNING')
                """,
                (self.experiment_id, name, params_json)
            )
            conn.commit()
            return cursor.lastrowid

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names to values
            step: Step/epoch number
        """
        if not metrics:
            return

        if self.run_id is None:
            logger.warning("Cannot log metrics: run_id is None")
            return

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            for key, value in metrics.items():
                try:
                    # Ensure value is float
                    float_value = float(value)

                    cursor.execute(
                        """
                        INSERT INTO metrics (run_id, key, value, step)
                        VALUES (?, ?, ?, ?)
                        """,
                        (self.run_id, key, float_value, step)
                    )
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        f"Skipping non-numeric metric {key}={value}: {exc}"
                    )

            conn.commit()

    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """
        Log an artifact (file) for the current run.

        Args:
            path: Path to the artifact file
            name: Optional custom name for the artifact
        """
        if self.run_id is None:
            logger.warning("Cannot log artifact: run_id is None")
            return

        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        artifact_name = name or path.name
        artifact_type = "model" if path.suffix in [".pt", ".pth", ".ckpt"] else "file"

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO artifacts (run_id, name, path, artifact_type)
                VALUES (?, ?, ?, ?)
                """,
                (self.run_id, artifact_name, str(path.absolute()), artifact_type)
            )
            conn.commit()

        logger.debug(f"Logged artifact: {artifact_name} -> {path}")

    def finish(self) -> None:
        """Mark the current run as finished."""
        if self.run_id is None:
            logger.warning("Cannot finish: run_id is None")
            return

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs
                SET status = 'FINISHED', finished_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (self.run_id,)
            )
            conn.commit()

        logger.info(f"Run {self.run_name} finished")

    def get_metrics(self, run_id: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Retrieve metrics for a run.

        Args:
            run_id: Run ID (uses current run if None)

        Returns:
            List of metric dictionaries
        """
        target_run_id = run_id or self.run_id
        if target_run_id is None:
            return []

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value, step, timestamp
                FROM metrics
                WHERE run_id = ?
                ORDER BY step, key
                """,
                (target_run_id,)
            )

            results = []
            for row in cursor.fetchall():
                results.append({
                    "key": row[0],
                    "value": row[1],
                    "step": row[2],
                    "timestamp": row[3],
                })

            return results

    def get_run_info(self, run_id: Optional[int] = None) -> Optional[dict[str, Any]]:
        """
        Get information about a run.

        Args:
            run_id: Run ID (uses current run if None)

        Returns:
            Dictionary with run information or None
        """
        target_run_id = run_id or self.run_id
        if target_run_id is None:
            return None

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.run_id, r.name, r.params, r.status,
                       r.started_at, r.finished_at, e.name
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                WHERE r.run_id = ?
                """,
                (target_run_id,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            return {
                "run_id": row[0],
                "name": row[1],
                "params": json.loads(row[2]) if row[2] else {},
                "status": row[3],
                "started_at": row[4],
                "finished_at": row[5],
                "experiment_name": row[6],
            }

    def list_runs(self, experiment_name: Optional[str] = None) -> list[dict[str, Any]]:
        """
        List all runs for an experiment.

        Args:
            experiment_name: Experiment name (uses current experiment if None)

        Returns:
            List of run dictionaries
        """
        target_experiment = experiment_name or self.experiment_name

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.run_id, r.name, r.params, r.status,
                       r.started_at, r.finished_at
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                WHERE e.name = ?
                ORDER BY r.started_at DESC
                """,
                (target_experiment,)
            )

            runs = []
            for row in cursor.fetchall():
                runs.append({
                    "run_id": row[0],
                    "name": row[1],
                    "params": json.loads(row[2]) if row[2] else {},
                    "status": row[3],
                    "started_at": row[4],
                    "finished_at": row[5],
                })

            return runs

    def save_study(
        self,
        study_name: str,
        direction: str = "maximize",
    ) -> int:
        """
        Create or get existing Optuna study.

        Args:
            study_name: Name of the study
            direction: Optimization direction ("maximize" or "minimize")

        Returns:
            Study ID
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Try to get existing study
            cursor.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,)
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Found existing study: {study_name}")
                return row[0]

            # Create new study
            cursor.execute(
                "INSERT INTO studies (study_name, direction) VALUES (?, ?)",
                (study_name, direction)
            )
            conn.commit()
            study_id = cursor.lastrowid
            logger.info(f"Created new study: {study_name} (id={study_id})")
            return study_id

    def save_trial(
        self,
        study_name: str,
        trial_number: int,
        state: str,
        params: dict[str, Any],
        value: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> int:
        """
        Save an Optuna trial result.

        Args:
            study_name: Name of the study
            trial_number: Trial number
            state: Trial state (COMPLETE, PRUNED, FAIL, etc.)
            params: Trial hyperparameters
            value: Final objective value (if completed)
            duration: Trial duration in seconds

        Returns:
            Trial ID
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Get study_id
            cursor.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Study not found: {study_name}")
            study_id = row[0]

            params_json = json.dumps(params, default=str)

            # Check if trial already exists
            cursor.execute(
                "SELECT trial_id FROM trials WHERE study_id = ? AND trial_number = ?",
                (study_id, trial_number)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing trial
                cursor.execute(
                    """
                    UPDATE trials
                    SET state = ?, value = ?, params = ?, duration = ?, finished_at = CURRENT_TIMESTAMP
                    WHERE trial_id = ?
                    """,
                    (state, value, params_json, duration, existing[0])
                )
                trial_id = existing[0]
                logger.debug(f"Updated trial {trial_number} for study {study_name}")
            else:
                # Insert new trial
                cursor.execute(
                    """
                    INSERT INTO trials (study_id, trial_number, state, value, params, duration, finished_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (study_id, trial_number, state, value, params_json, duration)
                )
                trial_id = cursor.lastrowid
                logger.debug(f"Saved trial {trial_number} for study {study_name}")

            conn.commit()
            return trial_id

    def save_trial_intermediate_value(
        self,
        study_name: str,
        trial_number: int,
        step: int,
        value: float,
    ) -> None:
        """
        Save an intermediate value for trial pruning.

        Args:
            study_name: Name of the study
            trial_number: Trial number
            step: Step/epoch number
            value: Intermediate metric value
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Get trial_id
            cursor.execute(
                """
                SELECT t.trial_id
                FROM trials t
                JOIN studies s ON t.study_id = s.study_id
                WHERE s.study_name = ? AND t.trial_number = ?
                """,
                (study_name, trial_number)
            )
            row = cursor.fetchone()
            if not row:
                logger.warning(
                    f"Trial not found: study={study_name}, trial={trial_number}"
                )
                return
            trial_id = row[0]

            # Insert or replace intermediate value
            cursor.execute(
                """
                INSERT OR REPLACE INTO trial_intermediate_values (trial_id, step, value)
                VALUES (?, ?, ?)
                """,
                (trial_id, step, value)
            )
            conn.commit()

    def get_study_info(self, study_name: str) -> Optional[dict[str, Any]]:
        """
        Get information about an Optuna study.

        Args:
            study_name: Name of the study

        Returns:
            Dictionary with study information or None
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Get study info
            cursor.execute(
                """
                SELECT study_id, study_name, direction, created_at
                FROM studies
                WHERE study_name = ?
                """,
                (study_name,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            study_id, name, direction, created_at = row

            # Get trials
            cursor.execute(
                """
                SELECT trial_number, state, value, params, duration
                FROM trials
                WHERE study_id = ?
                ORDER BY trial_number
                """,
                (study_id,)
            )
            trials = []
            best_value = None
            best_trial = None

            for trial_row in cursor.fetchall():
                trial_number, state, value, params_json, duration = trial_row
                params = json.loads(params_json) if params_json else {}

                trial_info = {
                    "number": trial_number,
                    "state": state,
                    "value": value,
                    "params": params,
                    "duration": duration,
                }
                trials.append(trial_info)

                # Track best trial
                if state == "COMPLETE" and value is not None:
                    if best_value is None:
                        best_value = value
                        best_trial = trial_number
                    elif (direction == "maximize" and value > best_value) or \
                         (direction == "minimize" and value < best_value):
                        best_value = value
                        best_trial = trial_number

            return {
                "study_id": study_id,
                "study_name": name,
                "direction": direction,
                "created_at": created_at,
                "n_trials": len(trials),
                "trials": trials,
                "best_trial": best_trial,
                "best_value": best_value,
            }

    def list_studies(self) -> list[dict[str, Any]]:
        """
        List all Optuna studies in the database.

        Returns:
            List of study dictionaries
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.study_id, s.study_name, s.direction, s.created_at,
                       COUNT(t.trial_id) as n_trials
                FROM studies s
                LEFT JOIN trials t ON s.study_id = t.study_id
                GROUP BY s.study_id
                ORDER BY s.created_at DESC
                """
            )

            studies = []
            for row in cursor.fetchall():
                studies.append({
                    "study_id": row[0],
                    "study_name": row[1],
                    "direction": row[2],
                    "created_at": row[3],
                    "n_trials": row[4],
                })

            return studies
