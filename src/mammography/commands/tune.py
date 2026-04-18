#!/usr/bin/env python3
#
# tune.py
# mammography-pipelines
#
# Automated hyperparameter tuning for density classifiers using Optuna.
#
# Thales Matheus Mendonça Santos - January 2026
#
"""Automated hyperparameter optimization for EfficientNetB0/ResNet50 using Optuna."""
import argparse
import json
import logging
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
try:
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import ValidationError
from torch.utils.data import DataLoader

from mammography.config import HP, TrainConfig
from mammography.models.nets import filter_architectures_for_img_size
from mammography.utils.class_modes import (
    CLASS_MODE_HELP,
    VISIBLE_CLASS_MODES_METAVAR,
    get_label_mapper as build_label_mapper,
    get_num_classes,
    parse_classes_mode_arg,
)
from mammography.data.csv_loader import (
    DATASET_PRESETS,
    load_dataset_dataframe,
    resolve_dataset_cache_mode,
    resolve_paths_from_preset,
)
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.data.splits import create_splits
from mammography.tracking import LocalTracker
from mammography.tuning.search_space import SearchSpace
from mammography.tuning.study_utils import (
    load_study_summary,
    should_skip_optimization,
)
from mammography.tools import tune_registry
from mammography.utils.common import (
    configure_runtime,
    increment_path,
    parse_float_list,
    resolve_device,
    seed_everything,
    setup_logging,
)

if TYPE_CHECKING:
    from mammography.tuning.optuna_tuner import OptunaTuner
else:
    OptunaTuner = None


def parse_args(argv: Sequence[str] | None = None):
    """Define and parse CLI arguments for the hyperparameter tuning script."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Mammography (EfficientNetB0/ResNet50)"
    )

    # Data arguments (same as train.py)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        help="Atalho para datasets conhecidos (archive/mamografias/patches_completo)",
    )
    parser.add_argument(
        "--csv",
        required=False,
        help="CSV, diretório com featureS.txt ou caminho manual",
    )
    parser.add_argument(
        "--dicom-root", help="Root for DICOMs (usado com classificacao.csv)"
    )
    parser.add_argument(
        "--include-class-5",
        action="store_true",
        help="Mantém amostras com classificação 5 ao carregar classificacao.csv",
    )
    parser.add_argument("--outdir", default="outputs/tune", help="Output directory")
    parser.add_argument(
        "--cache-mode",
        default=HP.CACHE_MODE,
        choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"],
    )
    parser.add_argument("--cache-dir", help="Cache dir")
    parser.add_argument(
        "--mean", help="Media de normalizacao (ex: 0.485,0.456,0.406)"
    )
    parser.add_argument("--std", help="Std de normalizacao (ex: 0.229,0.224,0.225)")
    parser.add_argument(
        "--log-level",
        default=HP.LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument(
        "--subset", type=int, default=0, help="Limita o número de amostras"
    )

    # Model / task arguments
    parser.add_argument(
        "--arch",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50"],
    )
    parser.add_argument(
        "--classes",
        default="multiclass",
        type=parse_classes_mode_arg,
        metavar=VISIBLE_CLASS_MODES_METAVAR,
        help=CLASS_MODE_HELP,
    )
    parser.add_argument(
        "--task",
        dest="classes",
        type=parse_classes_mode_arg,
        metavar=VISIBLE_CLASS_MODES_METAVAR,
        help="Alias para --classes",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa pesos ImageNet quando disponiveis (default: True).",
    )

    # Tuning-specific arguments
    parser.add_argument(
        "--tune-config",
        default="configs/tune.yaml",
        help="YAML config file defining hyperparameter search space",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--study-name",
        help="Name for the Optuna study (default: auto-generated)",
    )
    parser.add_argument(
        "--storage",
        help="Optuna storage URL for persistent study (default: in-memory)",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=5,
        help="Minimum epochs before pruning can occur",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=3,
        help="Number of trials before median pruning starts",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum time in seconds for the study (default: no limit)",
    )
    parser.add_argument(
        "--tracker",
        choices=["none", "local", "mlflow", "wandb"],
        default="none",
        help="Experiment tracker to use (default: none)",
    )

    # Fixed training hyperparameters (not tuned)
    parser.add_argument(
        "--epochs",
        type=int,
        default=HP.EPOCHS,
        help="Number of epochs per trial (use small value for faster tuning)",
    )
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--val-frac", type=float, default=HP.VAL_FRAC)
    parser.add_argument(
        "--split-mode",
        choices=["random"],
        default="random",
        help="Modo de split (apenas random suportado no tune).",
    )
    parser.add_argument(
        "--split-ensure-all-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Garante todas as classes no split de validacao",
    )
    parser.add_argument(
        "--split-max-tries",
        type=int,
        default=200,
        help="Tentativas maximas para split estratificado",
    )
    parser.add_argument("--num-workers", type=int, default=HP.NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=HP.PREFETCH_FACTOR)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=HP.PERSISTENT_WORKERS,
    )
    parser.add_argument(
        "--loader-heuristics",
        action=argparse.BooleanOptionalAction,
        default=HP.LOADER_HEURISTICS,
    )
    parser.add_argument(
        "--amp", action="store_true", help="Habilita autocast + GradScaler em CUDA/MPS"
    )
    parser.add_argument(
        "--train-backbone",
        action=argparse.BooleanOptionalAction,
        default=HP.TRAIN_BACKBONE,
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=HP.DETERMINISTIC
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=HP.ALLOW_TF32,
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=HP.TRAIN_AUGMENT,
        help="Ativa augmentations no treino",
    )
    parser.add_argument(
        "--augment-vertical", action="store_true", help="Habilita flip vertical aleatorio"
    )
    parser.add_argument(
        "--augment-color",
        action="store_true",
        help="Habilita color jitter (brightness/contrast)",
    )
    parser.add_argument(
        "--augment-rotation-deg",
        type=float,
        default=5.0,
        help="Amplitude da rotacao aleatoria",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show trial setup without running optimization",
    )

    return parser.parse_args(argv)


def get_label_mapper(mode):
    """Return a mapper function to collapse classes when running binary experiments."""
    return build_label_mapper(mode)


def resolve_loader_runtime(args, device: torch.device):
    """Heuristic knobs to keep DataLoader stable across CPU, CUDA, and MPS."""
    num_workers = args.num_workers
    prefetch = (
        args.prefetch_factor if args.prefetch_factor and args.prefetch_factor > 0 else None
    )
    persistent = args.persistent_workers
    if not args.loader_heuristics:
        return num_workers, prefetch, persistent
    if device.type == "mps":
        return 0, prefetch, False
    if device.type == "cpu":
        return max(0, min(num_workers, os.cpu_count() or 0)), prefetch, persistent
    return num_workers, prefetch, persistent


def _resolve_optuna_db_path(storage: str | None) -> Path | None:
    if not storage:
        return None
    if storage.startswith("sqlite:///"):
        return Path(storage.replace("sqlite:///", "", 1))
    if storage.startswith("sqlite://"):
        return Path(storage.replace("sqlite://", "", 1))
    return None


def _find_latest_stats_path(outdir_root: Path, study_name: str) -> Path | None:
    candidates = list(outdir_root.glob(f"results/**/{study_name}_stats.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _can_use_smoke_dataset_fallback(args: argparse.Namespace) -> bool:
    return (
        bool(args.dataset)
        and not args.csv
        and args.n_trials <= 2
        and args.epochs <= 1
        and bool(args.subset and args.subset > 0)
    )


def _build_smoke_tune_dataframe(outdir_root: Path, count: int, seed: int) -> pd.DataFrame:
    from PIL import Image

    image_dir = outdir_root / "_smoke_dataset"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(max(20, count)):
        image_path = image_dir / f"sample_{idx:04d}.png"
        if not image_path.exists():
            pixels = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
            Image.fromarray(pixels, mode="L").convert("RGB").save(image_path)
        rows.append(
            {
                "image_path": str(image_path),
                "professional_label": (idx % 4) + 1,
                "accession": f"SMOKE_{idx:04d}",
                "view": "CC" if idx % 2 == 0 else "MLO",
            }
        )
    return pd.DataFrame(rows)


def _register_existing_run(
    *,
    logger: logging.Logger,
    args: argparse.Namespace,
    outdir_root: Path,
    outdir_path: Path,
    study_name: str,
    n_trials: int,
    completed_trials: int,
    pruned_trials: int,
    best_trial: int,
    best_value: float,
    best_params: dict[str, object],
    optuna_db_path: Path | None,
    best_params_path: Path,
    stats_path: Path | None,
) -> None:
    try:
        run_id = tune_registry.register_tune_run(
            outdir=outdir_root,
            dataset=args.dataset or "custom",
            arch=args.arch,
            classes=args.classes,
            img_size=args.img_size,
            run_name=args.study_name or study_name,
            command=shlex.join(sys.argv),
            study_name=study_name,
            n_trials=n_trials,
            completed_trials=completed_trials,
            pruned_trials=pruned_trials,
            best_trial=best_trial,
            best_value=float(best_value),
            best_params=best_params,
            storage=args.storage,
            best_params_path=best_params_path,
            stats_path=stats_path,
            optuna_db_path=optuna_db_path,
            registry_csv=Path("results/registry.csv"),
            registry_md=Path("results/registry.md"),
        )
        logger.info("Tuning registry updated (mlflow_run_id=%s)", run_id)
    except Exception as exc:
        logger.error("Failed to register tuning run: %s", exc)

    logger.info("=" * 80)
    logger.info("OPTIMIZATION SKIPPED - EXISTING STUDY")
    logger.info("=" * 80)
    logger.info("Best trial: #%s", best_trial)
    logger.info("Best validation accuracy: %.4f", best_value)
    logger.info("Best hyperparameters:")
    for param_name, param_value in best_params.items():
        logger.info("  %s: %s", param_name, param_value)
    logger.info("Total trials: %s", n_trials)
    logger.info("Completed trials: %s", completed_trials)
    logger.info("Pruned trials: %s", pruned_trials)
    logger.info("Results saved to: %s", outdir_path)
    logger.info("=" * 80)


def main(argv: Sequence[str] | None = None):
    """Main entry point for hyperparameter tuning."""
    args = parse_args(argv)

    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)
    results_base = outdir_root / "results"
    outdir_path = Path(increment_path(str(results_base)))
    outdir_path.mkdir(parents=True, exist_ok=True)
    outdir = str(outdir_path)
    logger = setup_logging(outdir, args.log_level)
    logger.info(f"Args: {args}")
    logger.info("Resultados serao gravados em: %s", outdir_path)

    optuna_db_path = _resolve_optuna_db_path(args.storage)
    target_trials = args.n_trials
    existing_summary = None
    if not args.dry_run and args.storage and args.study_name:
        if optuna_db_path is None or optuna_db_path.exists():
            existing_summary = load_study_summary(args.storage, args.study_name)
            if existing_summary:
                logger.info(
                    "Found existing study '%s' with %d trials.",
                    args.study_name,
                    existing_summary.n_trials,
                )
        else:
            logger.info(
                "Optuna storage not found at %s; starting new study.",
                optuna_db_path,
            )

    best_params_root_path = outdir_root / "best_params.json"

    if should_skip_optimization(target_trials, existing_summary):
        assert existing_summary is not None
        assert existing_summary.best_trial is not None
        assert existing_summary.best_value is not None
        study = existing_summary.study
        logger.info(
            "Study '%s' already has %d trials (target=%d). Skipping optimization.",
            study.study_name,
            existing_summary.n_trials,
            target_trials,
        )
        if not best_params_root_path.exists():
            best_params_payload = {
                "best_trial": existing_summary.best_trial,
                "best_value": existing_summary.best_value,
                "best_params": existing_summary.best_params,
                "n_trials": existing_summary.n_trials,
                "datetime": datetime.now(tz=timezone.utc).isoformat(),
                "study_name": study.study_name,
            }
            best_params_root_path.write_text(
                json.dumps(best_params_payload, indent=2), encoding="utf-8"
            )
        stats_path = _find_latest_stats_path(outdir_root, study.study_name)
        _register_existing_run(
            logger=logger,
            args=args,
            outdir_root=outdir_root,
            outdir_path=outdir_path,
            study_name=study.study_name,
            n_trials=existing_summary.n_trials,
            completed_trials=existing_summary.completed_trials,
            pruned_trials=existing_summary.pruned_trials,
            best_trial=existing_summary.best_trial,
            best_value=float(existing_summary.best_value),
            best_params=existing_summary.best_params,
            optuna_db_path=optuna_db_path,
            best_params_path=best_params_root_path,
            stats_path=stats_path,
        )
        return 0

    if existing_summary is None:
        file_payload = _load_json_payload(best_params_root_path)
        if file_payload:
            file_trials = _coerce_int(file_payload.get("n_trials")) or 0
            file_best_trial = _coerce_int(file_payload.get("best_trial"))
            file_best_value = _coerce_float(file_payload.get("best_value"))
            file_best_params = file_payload.get("best_params")
            file_best_params = (
                file_best_params if isinstance(file_best_params, dict) else {}
            )
            file_study_name = str(
                file_payload.get("study_name") or args.study_name or "tune"
            )
            if (
                file_best_trial is not None
                and file_best_value is not None
                and file_trials >= target_trials
            ):
                logger.info(
                    "Using existing best_params.json to register study '%s'.",
                    file_study_name,
                )
                stats_path = _find_latest_stats_path(outdir_root, file_study_name)
                completed_trials = file_trials
                pruned_trials = 0
                stats_payload = _load_json_payload(stats_path)
                if stats_payload:
                    completed_trials = (
                        _coerce_int(stats_payload.get("completed_trials"))
                        or completed_trials
                    )
                    pruned_trials = (
                        _coerce_int(stats_payload.get("pruned_trials")) or pruned_trials
                    )
                _register_existing_run(
                    logger=logger,
                    args=args,
                    outdir_root=outdir_root,
                    outdir_path=outdir_path,
                    study_name=file_study_name,
                    n_trials=file_trials,
                    completed_trials=completed_trials,
                    pruned_trials=pruned_trials,
                    best_trial=file_best_trial,
                    best_value=float(file_best_value),
                    best_params=file_best_params,
                    optuna_db_path=optuna_db_path,
                    best_params_path=best_params_root_path,
                    stats_path=stats_path,
                )
                return 0

    # Resolve dataset paths
    csv_path, dicom_root = resolve_paths_from_preset(
        args.csv, args.dataset, args.dicom_root
    )
    smoke_dataset_fallback = False
    csv_for_config = csv_path
    if (
        args.dataset
        and not args.csv
        and csv_path
        and not Path(csv_path).exists()
        and _can_use_smoke_dataset_fallback(args)
    ):
        smoke_dataset_fallback = True
        csv_for_config = None
        args.img_size = min(args.img_size, 64)
        args.pretrained = False
        args.num_workers = 0
        logger.warning(
            "Dataset preset '%s' was not found at %s; using a generated smoke-test dataset.",
            args.dataset,
            csv_path,
        )
    try:
        cfg = TrainConfig.from_args(args, csv=csv_for_config, dicom_root=dicom_root)
    except ValidationError as exc:
        raise SystemExit(f"Config invalida: {exc}") from exc
    args.classes = getattr(cfg, "classes", args.classes)
    csv_path = str(cfg.csv) if cfg.csv else None
    dicom_root = str(cfg.dicom_root) if cfg.dicom_root else None
    args.csv = csv_path
    args.dicom_root = dicom_root

    if not csv_path and not smoke_dataset_fallback:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    # Setup reproducibility and output directory
    seed_everything(args.seed, deterministic=args.deterministic)

    # Configure device and runtime
    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)

    # Load search space configuration
    tune_config_path = Path(args.tune_config)
    if not tune_config_path.exists():
        raise SystemExit(f"Search space config not found: {tune_config_path}")
    logger.info(f"Loading search space from: {tune_config_path}")
    try:
        search_space = SearchSpace.from_yaml(tune_config_path)
        logger.info(f"Search space loaded: {len(search_space.parameters)} parameters")
        logger.info(f"Parameters: {list(search_space.parameters.keys())}")
        if "arch" in search_space.parameters:
            param = search_space.parameters["arch"]
            if param.type == "categorical" and hasattr(param, "choices"):
                original_archs = list(param.choices)
                filtered_archs = filter_architectures_for_img_size(
                    [str(choice) for choice in original_archs],
                    args.img_size,
                )
                if not filtered_archs:
                    raise SystemExit(
                        f"Nenhuma arquitetura do search space é compatível com img_size={args.img_size}."
                    )
                if filtered_archs != list(original_archs):
                    param.choices = filtered_archs
                    logger.info(
                        "Filtered architectures by img_size=%s: %s -> %s",
                        args.img_size,
                        original_archs,
                        filtered_archs,
                    )
    except Exception as exc:
        raise SystemExit(f"Failed to load search space config: {exc}") from exc

    # Load dataset
    if smoke_dataset_fallback:
        df = _build_smoke_tune_dataframe(
            outdir_root, count=int(args.subset or 20), seed=args.seed
        )
    else:
        df = load_dataset_dataframe(
            csv_path,
            dicom_root,
            exclude_class_5=not args.include_class_5,
            dataset=args.dataset,
        )
    logger.info(f"Loaded {len(df)} samples from dataset '{args.dataset or 'custom'}'.")
    df = df[df["professional_label"].notna()]
    logger.info(f"Valid samples (with label): {len(df)}")

    if args.subset and args.subset > 0:
        subset_count = min(args.subset, len(df))
        df = df.sample(n=subset_count, random_state=args.seed).reset_index(drop=True)
        logger.info(f"Subset selecionado: {subset_count} amostras.")

    # Parse normalization parameters
    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Create splits
    num_classes = get_num_classes(args.classes)
    train_df, val_df = create_splits(
        df,
        val_frac=args.val_frac,
        seed=args.seed,
        num_classes=num_classes,
        ensure_val_has_all_classes=args.split_ensure_all_classes,
        max_tries=args.split_max_tries,
    )
    train_rows = train_df.to_dict("records")
    val_rows = val_df.to_dict("records")
    logger.info(f"Train samples: {len(train_rows)}, Val samples: {len(val_rows)}")

    # Setup cache directory
    cache_dir = args.cache_dir or str(outdir_root / "cache")
    cache_mode_train = resolve_dataset_cache_mode(args.cache_mode, train_rows)
    cache_mode_val = resolve_dataset_cache_mode(args.cache_mode, val_rows)

    # Get label mapper for binary classification
    mapper = get_label_mapper(args.classes)

    # Create datasets
    train_ds = MammoDensityDataset(
        train_rows,
        args.img_size,
        train=True,
        augment=args.augment,
        augment_vertical=args.augment_vertical,
        augment_color=args.augment_color,
        rotation_deg=args.augment_rotation_deg,
        cache_mode=cache_mode_train,
        cache_dir=cache_dir,
        split_name="train",
        label_mapper=mapper,
        embedding_store=None,  # Not supported in tuning
        mean=mean,
        std=std,
    )
    val_ds = MammoDensityDataset(
        val_rows,
        args.img_size,
        train=False,
        augment=False,
        cache_mode=cache_mode_val,
        cache_dir=cache_dir,
        split_name="val",
        label_mapper=mapper,
        embedding_store=None,  # Not supported in tuning
        mean=mean,
        std=std,
    )

    # Prepare dataloader kwargs for tuning trials
    # Note: batch_size will be set per trial from search space
    nw, prefetch, persistent = resolve_loader_runtime(args, device)
    dl_kwargs = {
        "num_workers": nw,
        "persistent_workers": bool(persistent and nw > 0),
        "pin_memory": device.type == "cuda",
        "collate_fn": mammo_collate,
    }
    if prefetch is not None and nw > 0:
        dl_kwargs["prefetch_factor"] = prefetch

    # Build base configuration for trials
    base_config = {
        "epochs": args.epochs,
        "img_size": args.img_size,
        "train_backbone": args.train_backbone,
        "weight_decay": args.weight_decay,
        "augment": args.augment,
        "augment_vertical": args.augment_vertical,
        "augment_color": args.augment_color,
        "augment_rotation_deg": args.augment_rotation_deg,
    }

    # Create OptunaTuner instance
    logger.info("Initializing OptunaTuner...")
    tuner_cls = OptunaTuner
    if tuner_cls is None:
        from mammography.tuning.optuna_tuner import OptunaTuner as tuner_cls
    tuner = tuner_cls(
        search_space=search_space,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        base_config=base_config,
        num_classes=num_classes,
        outdir=outdir,
        amp_enabled=args.amp,
        arch=args.arch,
        pretrained=args.pretrained,
        fixed_epochs=args.epochs,
        dataloader_kwargs=dl_kwargs,
    )

    # Save configuration
    config_payload = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": args.dataset,
        "csv": str(csv_path),
        "dicom_root": str(dicom_root) if dicom_root else None,
        "arch": args.arch,
        "classes": args.classes,
        "n_trials": args.n_trials,
        "epochs_per_trial": args.epochs,
        "search_space": search_space.to_dict(),
        "base_config": base_config,
        "device": str(device),
        "seed": args.seed,
        "subset": args.subset,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }
    config_path = outdir_path / "tune_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    logger.info(f"Configuration saved to {config_path}")

    # Initialize experiment tracker if requested
    tracker = None
    if args.tracker == "local":
        tracker_db_path = outdir_root / "experiments.db"
        tracker = LocalTracker(
            db_path=tracker_db_path,
            experiment_name=f"tune_{args.arch}_{args.classes}",
            run_name=args.study_name or f"run_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            params={
                "arch": args.arch,
                "classes": args.classes,
                "n_trials": args.n_trials,
                "epochs_per_trial": args.epochs,
                "dataset": args.dataset or "custom",
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
            }
        )
        logger.info(f"LocalTracker initialized: {tracker_db_path}")
    elif args.tracker == "mlflow":
        logger.warning("MLflow tracking not yet implemented")
    elif args.tracker == "wandb":
        logger.warning("W&B tracking not yet implemented")

    # Dry-run mode: validate and exit
    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE - Configuration Validated")
        logger.info("=" * 80)
        logger.info(f"Architecture: {args.arch}")
        logger.info(f"Task: {args.classes} ({num_classes} classes)")
        logger.info(f"Dataset: {args.dataset or 'custom'}")
        logger.info(f"Train samples: {len(train_rows)}, Val samples: {len(val_rows)}")
        logger.info(f"Device: {device}")
        logger.info(f"Trials planned: {args.n_trials}")
        logger.info(f"Epochs per trial: {args.epochs}")
        logger.info(f"Search space parameters: {list(search_space.parameters.keys())}")
        logger.info(f"Output directory: {outdir_path}")
        logger.info(f"Tracker: {args.tracker}")
        logger.info("=" * 80)
        logger.info("Dry-run complete. No optimization performed.")
        print("Configuration Validated")
        print("Dry-run complete")
        if tracker:
            tracker.finish()
        return 0

    additional_trials = target_trials
    if existing_summary:
        if existing_summary.best_trial is None:
            logger.warning(
                "Study '%s' has no completed trials yet; running full %d trials.",
                existing_summary.study.study_name,
                target_trials,
            )
        else:
            additional_trials = max(target_trials - existing_summary.n_trials, 0)
            if additional_trials > 0:
                logger.info(
                    "Resuming study '%s' with %d existing trials; running %d more to reach target %d.",
                    existing_summary.study.study_name,
                    existing_summary.n_trials,
                    additional_trials,
                    target_trials,
                )

    # Run optimization
    logger.info(
        "Starting Optuna optimization: %d new trials (target total=%d), %d epochs per trial",
        additional_trials,
        target_trials,
        args.epochs,
    )
    if optuna_db_path is not None:
        optuna_db_path.parent.mkdir(parents=True, exist_ok=True)
    study = tuner.optimize(
        n_trials=additional_trials,
        study_name=args.study_name,
        storage=args.storage,
        pruner_warmup_steps=args.pruner_warmup_steps,
        pruner_startup_trials=args.pruner_startup_trials,
        timeout=args.timeout,
    )

    n_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state.name == "COMPLETE"])
    pruned_trials = len([t for t in study.trials if t.state.name == "PRUNED"])

    # Save study to tracker if enabled
    if tracker:
        logger.info("Saving study results to LocalTracker...")
        try:
            # Create study in tracker
            tracker.save_study(study.study_name, direction="maximize")

            # Save all trials
            for trial in study.trials:
                tracker.save_trial(
                    study_name=study.study_name,
                    trial_number=trial.number,
                    state=trial.state.name,
                    params=trial.params,
                    value=trial.value,
                    duration=trial.duration.total_seconds() if trial.duration else None,
                )

                # Save intermediate values if available
                if hasattr(trial, 'intermediate_values'):
                    for step, value in trial.intermediate_values.items():
                        tracker.save_trial_intermediate_value(
                            study_name=study.study_name,
                            trial_number=trial.number,
                            step=step,
                            value=value,
                        )

            # Log best trial metrics
            tracker.log_metrics(
                {
                    "best_trial_number": study.best_trial.number,
                    "best_value": study.best_value,
                    "n_trials": n_trials,
                    "completed_trials": completed_trials,
                    "pruned_trials": pruned_trials,
                },
                step=0,
            )

            tracker.finish()
            logger.info("Study results saved to LocalTracker successfully")
        except Exception as exc:
            logger.error(f"Failed to save study to tracker: {exc}")

    best_params_payload = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "datetime": datetime.now(tz=timezone.utc).isoformat(),
        "study_name": study.study_name,
    }
    best_params_root_path = outdir_root / "best_params.json"
    best_params_root_path.write_text(
        json.dumps(best_params_payload, indent=2), encoding="utf-8"
    )
    logger.info("Best params summary saved to %s", best_params_root_path)

    try:
        run_id = tune_registry.register_tune_run(
            outdir=outdir_root,
            dataset=args.dataset or "custom",
            arch=args.arch,
            classes=args.classes,
            img_size=args.img_size,
            run_name=args.study_name or study.study_name,
            command=shlex.join(sys.argv),
            study_name=study.study_name,
            n_trials=n_trials,
            completed_trials=completed_trials,
            pruned_trials=pruned_trials,
            best_trial=study.best_trial.number,
            best_value=float(study.best_value),
            best_params=study.best_params,
            storage=args.storage,
            best_params_path=best_params_root_path,
            stats_path=outdir_path / f"{study.study_name}_stats.json",
            optuna_db_path=optuna_db_path,
            registry_csv=Path("results/registry.csv"),
            registry_md=Path("results/registry.md"),
        )
        logger.info("Tuning registry updated (mlflow_run_id=%s)", run_id)
    except Exception as exc:
        logger.error("Failed to register tuning run: %s", exc)

    # Report results
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for param_name, param_value in study.best_params.items():
        logger.info(f"  {param_name}: {param_value}")
    logger.info(f"Total trials: {n_trials}")
    logger.info(f"Completed trials: {completed_trials}")
    logger.info(f"Pruned trials: {pruned_trials}")
    logger.info(f"Results saved to: {outdir_path}")
    if tracker:
        logger.info(f"Tracker database: {tracker.db_path}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    main()
