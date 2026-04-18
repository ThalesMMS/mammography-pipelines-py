# ruff: noqa
"""Setup helpers for the density training command."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import ValidationError


@dataclass(frozen=True)
class TrainCommandSetup:
    args: object
    argv_list: list[str]
    csv_path: str
    dicom_root: str | None
    outdir_root: Path
    outdir_path: Path
    metrics_dir: Path
    export_formats: list[str]
    logger: object
    device: object
    killer: object


def _apply_environment_overrides(args, facade) -> None:
    override_subset = os.environ.get("MAMMO_SUBSET")
    if override_subset and not args.subset:
        try:
            args.subset = max(0, int(override_subset))
        except ValueError:
            logging.getLogger("mammography").warning(
                "MAMMO_SUBSET invalido (%s). Usando valor de linha de comando.",
                override_subset,
            )

    override_epochs = os.environ.get("MAMMO_EPOCHS")
    if override_epochs and not getattr(args, "_epochs_explicit", False):
        try:
            args.epochs = max(1, int(override_epochs))
        except ValueError:
            logging.getLogger("mammography").warning(
                "MAMMO_EPOCHS invalido (%s). Usando valor de linha de comando.",
                override_epochs,
            )


def _resolve_auto_resume(args, facade) -> Path | None:
    checkpoint_path = facade._find_resume_checkpoint(args.outdir)
    if checkpoint_path and not args.resume_from:
        print(
            f"[INFO] Checkpoint encontrado em {checkpoint_path}. Retomando automaticamente."
        )
        args.resume_from = str(checkpoint_path)
        return checkpoint_path
    return None


def _resolve_training_config(args, facade) -> tuple[str | None, str | None]:
    csv_path, dicom_root = facade.resolve_paths_from_preset(
        args.csv, args.dataset, args.dicom_root
    )
    try:
        cfg = facade.TrainConfig.from_args(args, csv=csv_path, dicom_root=dicom_root)
    except ValidationError as exc:
        raise SystemExit(f"Config invalida: {exc}") from exc
    args.classes = getattr(cfg, "classes", args.classes)
    csv_path = str(cfg.csv) if cfg.csv else None
    dicom_root = str(cfg.dicom_root) if cfg.dicom_root else None
    args.csv = csv_path
    args.dicom_root = dicom_root
    return csv_path, dicom_root


def _parse_export_formats(args, facade) -> list[str]:
    return facade._parse_export_formats(args.export_figures)


def _outdir_from_auto_resume(auto_resume_path: Path) -> Path:
    outdir_path = auto_resume_path.parent
    if auto_resume_path.name.startswith("checkpoint_"):
        view = auto_resume_path.stem.replace("checkpoint_", "", 1).lower()
        suffix = f"_{view}"
        if view and outdir_path.name.lower().endswith(suffix):
            outdir_path = outdir_path.parent / outdir_path.name[: -len(suffix)]
    return outdir_path


def _create_output_paths(
    args,
    config_input: bool,
    auto_resume_path: Path | None,
    export_formats: list[str],
    facade,
) -> tuple[Path, Path, Path]:
    if auto_resume_path is not None:
        outdir_path = _outdir_from_auto_resume(auto_resume_path)
        outdir_root = (
            outdir_path.parent
            if outdir_path.name.startswith("results")
            else Path(args.outdir)
        )
    elif config_input:
        outdir_root = Path(args.outdir)
        outdir_path = outdir_root
    else:
        outdir_root = Path(facade.increment_path(args.outdir))
        outdir_path = Path(facade.increment_path(str(outdir_root / "results")))
    outdir_root.mkdir(parents=True, exist_ok=True)
    outdir_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = outdir_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if export_formats:
        (outdir_path / "figures").mkdir(parents=True, exist_ok=True)
    return outdir_root, outdir_path, metrics_dir


def build_train_command(argv_list: Sequence[str]) -> str:
    import shlex

    command = "mammography train-density"
    if argv_list:
        command = f"{command} {shlex.join(list(argv_list))}"
    return command


def prepare_train_command_setup(argv, facade) -> TrainCommandSetup:
    config_input = isinstance(argv, facade.TrainConfig)
    argv_list = (
        []
        if config_input
        else (list(argv) if argv is not None else facade.sys.argv[1:])
    )
    args = facade.parse_args(argv)

    _apply_environment_overrides(args, facade)
    auto_resume_path = _resolve_auto_resume(args, facade)
    csv_path, dicom_root = _resolve_training_config(args, facade)
    if not csv_path:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    facade.seed_everything(args.seed, deterministic=args.deterministic)
    export_formats = _parse_export_formats(args, facade)
    outdir_root, outdir_path, metrics_dir = _create_output_paths(
        args,
        config_input,
        auto_resume_path,
        export_formats,
        facade,
    )

    outdir = str(outdir_path)
    logger = facade.setup_logging(outdir, args.log_level)
    logger.info("Args: %s", args)
    logger.info("Resultados serao gravados em: %s", outdir_path)
    if export_formats:
        logger.info(
            "Figuras serao exportadas em formatos: %s", ", ".join(export_formats)
        )

    device = facade.resolve_device(args.device)
    facade.configure_runtime(device, args.deterministic, args.allow_tf32)
    killer = facade.GracefulKiller(register_signals=True)
    return TrainCommandSetup(
        args=args,
        argv_list=argv_list,
        csv_path=csv_path,
        dicom_root=dicom_root,
        outdir_root=outdir_root,
        outdir_path=outdir_path,
        metrics_dir=metrics_dir,
        export_formats=export_formats,
        logger=logger,
        device=device,
        killer=killer,
    )
