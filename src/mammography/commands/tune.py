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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from mammography.config import HP, TrainConfig
from mammography.data.csv_loader import (
    DATASET_PRESETS,
    load_dataset_dataframe,
    resolve_dataset_cache_mode,
    resolve_paths_from_preset,
)
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.data.splits import create_splits
from mammography.tracking import LocalTracker
from mammography.tuning.optuna_tuner import OptunaTuner
from mammography.tuning.search_space import SearchSpace
from mammography.utils.common import (
    configure_runtime,
    increment_path,
    parse_float_list,
    resolve_device,
    seed_everything,
    setup_logging,
)


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
        default="density",
        choices=["density", "binary", "multiclass"],
        help="density/multiclass = BI-RADS 1..4, binary = A/B vs C/D",
    )
    parser.add_argument(
        "--task",
        dest="classes",
        choices=["density", "binary", "multiclass"],
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
    mode = (mode or "density").lower()
    if mode == "binary":
        # 1,2 -> 0 (Low); 3,4 -> 1 (High)
        def mapper(y):
            if y in [1, 2]:
                return 0
            if y in [3, 4]:
                return 1
            return y - 1  # Fallback

        return mapper
    return None  # Default 1..4 -> 0..3


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


def main(argv: Sequence[str] | None = None):
    """Main entry point for hyperparameter tuning."""
    args = parse_args(argv)

    # Resolve dataset paths
    csv_path, dicom_root = resolve_paths_from_preset(
        args.csv, args.dataset, args.dicom_root
    )
    try:
        cfg = TrainConfig.from_args(args, csv=csv_path, dicom_root=dicom_root)
    except ValidationError as exc:
        raise SystemExit(f"Config invalida: {exc}") from exc
    csv_path = str(cfg.csv) if cfg.csv else None
    dicom_root = str(cfg.dicom_root) if cfg.dicom_root else None
    args.csv = csv_path
    args.dicom_root = dicom_root

    if not csv_path:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    # Setup reproducibility and output directory
    seed_everything(args.seed, deterministic=args.deterministic)

    outdir_root = Path(increment_path(args.outdir))
    outdir_root.mkdir(parents=True, exist_ok=True)
    results_base = outdir_root / "results"
    outdir_path = Path(increment_path(str(results_base)))
    outdir_path.mkdir(parents=True, exist_ok=True)
    outdir = str(outdir_path)
    logger = setup_logging(outdir, args.log_level)
    logger.info(f"Args: {args}")
    logger.info("Resultados serao gravados em: %s", outdir_path)

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
    except Exception as exc:
        raise SystemExit(f"Failed to load search space config: {exc}") from exc

    # Load dataset
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
    num_classes = 2 if args.classes == "binary" else 4
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
    tuner = OptunaTuner(
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
        if tracker:
            tracker.finish()
        return 0

    # Run optimization
    logger.info(
        f"Starting Optuna optimization: {args.n_trials} trials, {args.epochs} epochs per trial"
    )
    study = tuner.optimize(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        pruner_warmup_steps=args.pruner_warmup_steps,
        pruner_startup_trials=args.pruner_startup_trials,
        timeout=args.timeout,
    )

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
            tracker.log_metrics({
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "completed_trials": len([t for t in study.trials if t.state.name == "COMPLETE"]),
                "pruned_trials": len([t for t in study.trials if t.state.name == "PRUNED"]),
            }, step=0)

            tracker.finish()
            logger.info("Study results saved to LocalTracker successfully")
        except Exception as exc:
            logger.error(f"Failed to save study to tracker: {exc}")

    # Report results
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for param_name, param_value in study.best_params.items():
        logger.info(f"  {param_name}: {param_value}")
    logger.info(f"Total trials: {len(study.trials)}")
    logger.info(
        f"Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}"
    )
    logger.info(
        f"Pruned trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}"
    )
    logger.info(f"Results saved to: {outdir_path}")
    if tracker:
        logger.info(f"Tracker database: {tracker.db_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
