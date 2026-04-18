#!/usr/bin/env python3
"""Experiment tracking utilities for training workflows."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional


class ExperimentTracker:
    def __init__(self, kind: str, module: Any, run: Any):
        """
        Initialize an ExperimentTracker with the backend type and its associated module and run objects.

        Parameters:
            kind (str): Backend identifier (e.g., "wandb", "mlflow", "local", or "none").
            module (Any): The backend library/module instance (or `None` for local/no-op backends).
            run (Any): The backend run/session object used to record metrics and artifacts.
        """
        self.kind = kind
        self.module = module
        self.run = run

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """
        Log a set of scalar metrics to the configured tracking backend.

        Parameters:
            metrics (dict[str, float]): Mapping from metric names to scalar values to be logged.
            step (int): Integer step or epoch index associated with these metrics.
        """
        if not metrics:
            return
        if self.kind == "wandb":
            self.module.log(metrics, step=step)
        elif self.kind == "mlflow":
            self.module.log_metrics(metrics, step=step)
        elif self.kind == "local":
            self.run.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """
        Upload or record a file as an experiment artifact using the configured tracking backend.

        Parameters:
            path (Path | str): Path to the file to add as an artifact.
            name (Optional[str]): Optional artifact name. For the `wandb` backend, if omitted the name defaults to
                `<run.id>-<path.stem>`. For `mlflow`, the file is logged under the `models` artifact path. For `local`,
                the call is delegated to the local run's `log_artifact` with this `name`.
        """
        path = Path(path)
        if self.kind == "wandb":
            artifact_name = name or f"{self.run.id}-{path.stem}"
            artifact = self.module.Artifact(artifact_name, type="model")
            artifact.add_file(str(path))
            self.module.log_artifact(artifact)
        elif self.kind == "mlflow":
            self.module.log_artifact(str(path), artifact_path="models")
        elif self.kind == "local":
            self.run.log_artifact(path, name=name)

    def finish(self) -> None:
        """
        End the current experiment run for the configured tracking backend.

        For `"wandb"` and `"local"` this calls the run's `finish()` method; for `"mlflow"` this calls `mlflow.end_run()`.
        """
        if self.kind == "wandb":
            self.run.finish()
        elif self.kind == "mlflow":
            self.module.end_run()
        elif self.kind == "local":
            self.run.finish()


def _sanitize_tracking_params(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a tracking configuration payload by removing None values and converting non-scalar types to backend-friendly formats.

    Parameters:
        payload (dict[str, Any]): Mapping of parameter names to values to be sanitized.

    Returns:
        dict[str, Any]: A cleaned copy of `payload` where:
            - entries with value `None` are omitted,
            - `Path` values are converted to their string paths,
            - `dict`, `list`, and `tuple` values are JSON-encoded strings,
            - all other values are copied unchanged.
    """
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Path):
            cleaned[key] = str(value)
            continue
        if isinstance(value, (dict, list, tuple)):
            cleaned[key] = json.dumps(value)
        else:
            cleaned[key] = value
    return cleaned


def _init_tracker(
    args: argparse.Namespace,
    summary_payload: dict[str, Any],
    outdir_path: Path,
    logger: logging.Logger,
) -> Optional["ExperimentTracker"]:
    """
    Initialize an experiment tracking client based on CLI arguments.

    Selects and initializes one of the supported tracking backends ("wandb", "mlflow", or "local") using values from `args` and logs/returns a corresponding ExperimentTracker. Prints the MLflow run id to stdout when MLflow is used. If `args.tracker` is "none", returns None.

    Parameters:
        args (argparse.Namespace): Parsed CLI arguments; expected attributes include
            `tracker`, `tracker_project`, `tracker_run_name`, and `tracker_uri`.
        summary_payload (dict[str, Any]): Parameter payload to be sanitized and passed to the tracker as configuration.
        outdir_path (Path): Output directory path used for tracker workspace or local DB location.
        logger (logging.Logger): Logger used to emit informational messages.

    Returns:
        ExperimentTracker | None: An ExperimentTracker for the selected backend, or `None` when tracking is disabled.

    Raises:
        SystemExit: If a requested backend is invalid or its required library (wandb or mlflow) is unavailable.
    """
    tracker = (args.tracker or "none").lower()
    if tracker == "none":
        return None

    params = _sanitize_tracking_params(summary_payload)

    if tracker == "wandb":
        try:
            import wandb  # type: ignore
        except Exception as exc:
            raise SystemExit(f"wandb nao disponivel: {exc}") from exc
        project = args.tracker_project or "mammography"
        run = wandb.init(
            project=project,
            name=args.tracker_run_name,
            dir=str(outdir_path),
            config=params,
        )
        logger.info("Tracker wandb ativo (project=%s).", project)
        return ExperimentTracker("wandb", wandb, run)

    if tracker == "mlflow":
        try:
            import mlflow  # type: ignore
        except Exception as exc:
            raise SystemExit(f"mlflow nao disponivel: {exc}") from exc
        if args.tracker_uri:
            mlflow.set_tracking_uri(args.tracker_uri)
        if args.tracker_project:
            mlflow.set_experiment(args.tracker_project)
        run = mlflow.start_run(run_name=args.tracker_run_name)
        print(f"MLFLOW_RUN_ID:{run.info.run_id}", flush=True)
        mlflow.log_params(params)
        logger.info("Tracker mlflow ativo.")
        return ExperimentTracker("mlflow", mlflow, run)

    if tracker == "local":
        from mammography.tracking import LocalTracker

        project = args.tracker_project or "mammography"
        db_path = outdir_path / "experiments.db"
        run = LocalTracker(
            db_path=db_path,
            experiment_name=project,
            run_name=args.tracker_run_name,
            params=params,
        )
        logger.info("Tracker local ativo (db=%s, project=%s).", db_path, project)
        return ExperimentTracker("local", None, run)

    raise SystemExit(f"Tracker invalido: {tracker}")


def _maybe_register_training_run(
    *,
    args: argparse.Namespace,
    outdir_root: Path,
    command: str,
    logger: logging.Logger,
) -> str | None:
    """
    Optionally register the training run in the local registry and return the registry-assigned run id.

    Attempts to register the run only when registry usage is enabled and a tracker run name is provided. Early exits and returns None when any of the following apply: args.no_registry is true, args.view_specific_training is true, or args.tracker_run_name is empty. On success returns the run id produced by the registry; on import or registration failure returns None.

    Parameters:
        args (argparse.Namespace): CLI arguments; this function reads these fields:
            - no_registry (bool): if true, skip registration.
            - view_specific_training (bool): if true, skip registration for view-specific training.
            - tracker_run_name (str): name used to register the run; required for registration.
            - dataset (str | None): dataset identifier recorded in the registry (defaults to empty string).
            - registry_csv (Path | None): CSV path/value passed to the registry helper.
            - registry_md (Path | None): markdown path/value passed to the registry helper.
            - tracker_uri (str | None): tracking URI forwarded to the registry helper.
            - tracker_project (str | None): experiment name forwarded to the registry helper.
        outdir_root (Path): Output directory path passed to the registry as the run output location.
        command (str): Command string (invocation) recorded with the registry entry.
        logger (logging.Logger): Logger used for informational and warning messages.

    Returns:
        str | None: The registry-assigned run id on successful registration, or `None` if registration was skipped or failed.
    """
    if getattr(args, "no_registry", False):
        logger.info("Registry local desativado (--no-registry).")
        return None
    if getattr(args, "view_specific_training", False):
        logger.info("Registry local ignorado para treino view-specific.")
        return None
    run_name = getattr(args, "tracker_run_name", None) or ""
    if not run_name:
        logger.info("Registry local ignorado: --tracker-run-name nao informado.")
        return None
    try:
        from mammography.tools import train_registry
    except Exception as exc:
        logger.warning("Falha ao importar train_registry: %s", exc)
        return None
    try:
        run_id = train_registry.register_training_run(
            outdir=outdir_root,
            dataset=args.dataset or "",
            workflow="train-density",
            run_name=run_name,
            command=command,
            registry_csv=args.registry_csv,
            registry_md=args.registry_md,
            tracking_uri=args.tracker_uri or None,
            experiment=args.tracker_project or None,
        )
    except Exception as exc:
        logger.warning("Falha ao registrar no registry local: %s", exc)
        return None
    logger.info("Registry atualizado (run_name=%s, run_id=%s).", run_name, run_id)
    return run_id
