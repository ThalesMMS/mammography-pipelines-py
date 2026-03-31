#!/usr/bin/env python3
#
# automl.py
# mammography-pipelines
#
# Automated ML pipeline with learning rate finder, architecture search, and augmentation optimization.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Automated ML pipeline for breast density classification with AutoML features."""
import argparse
import json
import logging
import os
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
    from mammography.tuning.lr_finder import LRFinder
    from mammography.tuning.resource_manager import ResourceManager


def parse_args(argv: Sequence[str] | None = None):
    """Define and parse CLI arguments for the AutoML pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated ML pipeline for Mammography (architecture search, LR finder, augmentation optimization)"
    )

    # Data arguments (same as tune.py)
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
    parser.add_argument("--outdir", default="outputs/automl", help="Output directory")
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
    parser.add_argument(
        "--auto-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detecta automaticamente formato do dataset a partir da estrutura de diretórios (default: True)",
    )

    # Model / task arguments
    parser.add_argument(
        "--arch",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50", "vit_b_16", "vit_b_32", "vit_l_16"],
        help="Default architecture (can be overridden by search space)",
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

    # AutoML-specific arguments
    parser.add_argument(
        "--automl-config",
        default="configs/automl.yaml",
        help="YAML config file defining AutoML search space (default: configs/automl.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50)",
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
        help="Minimum epochs before pruning can occur (default: 5)",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=3,
        help="Number of trials before median pruning starts (default: 3)",
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

    # Learning rate finder arguments
    parser.add_argument(
        "--lr-finder-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learning rate finder before optimization (default: True)",
    )
    parser.add_argument(
        "--lr-finder-epochs",
        type=int,
        default=2,
        help="Number of epochs for LR finder (default: 2)",
    )
    parser.add_argument(
        "--lr-finder-start-lr",
        type=float,
        default=1e-7,
        help="Starting learning rate for LR finder (default: 1e-7)",
    )
    parser.add_argument(
        "--lr-finder-end-lr",
        type=float,
        default=10.0,
        help="Ending learning rate for LR finder (default: 10.0)",
    )

    # Resource management arguments
    parser.add_argument(
        "--resource-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable resource-aware search space adjustment (default: True)",
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
        help="Modo de split (apenas random suportado no automl).",
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=HP.NUM_WORKERS,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=HP.PREFETCH_FACTOR,
        help="Prefetch fator do DataLoader",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=HP.PERSISTENT_WORKERS,
        help="Workers persistentes no DataLoader",
    )
    parser.add_argument(
        "--loader-heuristics",
        action=argparse.BooleanOptionalAction,
        default=HP.LOADER_HEURISTICS,
        help="Habilita heuristicas auto para num_workers/prefetch_factor",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=HP.PIN_MEMORY,
        help="Pin memory no DataLoader (default: auto via LOADER_HEURISTICS)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compila modelo via torch.compile (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Habilita autocast + GradScaler em CUDA/MPS",
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
        "--augment-vertical",
        action="store_true",
        help="Habilita flip vertical (padrao: apenas horizontal)",
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

    # Dry-run mode for testing
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: validate configuration without running optimization",
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


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main entry point for the AutoML pipeline.

    Orchestrates:
    1. Dataset loading and splitting
    2. Resource detection (GPU memory, CPU cores)
    3. Learning rate finder (optional)
    4. Hyperparameter optimization with architecture search and augmentation tuning
    5. Export of best configuration

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args(argv)

    # Setup output directory and logging
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)
    results_base = outdir_root / "results"
    outdir_path = Path(increment_path(str(results_base)))
    outdir_path.mkdir(parents=True, exist_ok=True)
    outdir = str(outdir_path)
    logger = setup_logging(outdir, args.log_level)

    logger.info("=== AutoML Pipeline for Breast Density Classification ===")
    logger.info(f"Args: {args}")
    logger.info(f"Output directory: {outdir_path}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"LR finder enabled: {args.lr_finder_enabled}")
    logger.info(f"Resource-aware mode: {args.resource_aware}")

    # Resolve dataset paths
    csv_path, dicom_root = resolve_paths_from_preset(
        args.csv, args.dataset, args.dicom_root
    )
    try:
        cfg = TrainConfig.from_args(args, csv=csv_path, dicom_root=dicom_root)
    except ValidationError as exc:
        raise SystemExit(f"Config invalida: {exc}") from exc
    args.classes = getattr(cfg, "classes", args.classes)
    csv_path = str(cfg.csv) if cfg.csv else None
    dicom_root = str(cfg.dicom_root) if cfg.dicom_root else None
    args.csv = csv_path
    args.dicom_root = dicom_root

    if not csv_path:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    # Setup reproducibility
    seed_everything(args.seed, deterministic=args.deterministic)

    # Configure device and runtime
    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)
    logger.info(f"Device: {device}")

    # Detect resources if resource-aware mode is enabled
    resource_constraints = None
    if args.resource_aware:
        logger.info("Detecting available resources...")
        from mammography.tuning.resource_manager import ResourceManager
        resource_manager = ResourceManager(device=device)
        resource_constraints = resource_manager.get_resource_summary()
        logger.info(f"GPU memory available: {resource_constraints.get('gpu_memory_gb', 0):.2f} GB")
        logger.info(f"CPU cores: {resource_constraints.get('cpu_cores', 0)}")
        if 'ram_gb' in resource_constraints:
            logger.info(f"System RAM: {resource_constraints.get('ram_gb', 0):.2f} GB")

    # Load search space configuration
    automl_config_path = Path(args.automl_config)
    if not automl_config_path.exists():
        raise SystemExit(f"AutoML config not found: {automl_config_path}")
    logger.info(f"Loading search space from: {automl_config_path}")
    try:
        search_space = SearchSpace.from_yaml(automl_config_path)
        logger.info(f"Search space loaded: {len(search_space.parameters)} parameters")
        logger.info(f"Parameters: {list(search_space.parameters.keys())}")

        # Adjust search space based on resource constraints
        if args.resource_aware and resource_constraints and resource_manager:
            logger.info("Adjusting search space based on detected resources...")
            # Filter architectures based on available resources
            if 'arch' in search_space.parameters:
                param = search_space.parameters['arch']
                if param.type == 'categorical' and hasattr(param, 'choices'):
                    original_archs = param.choices
                    filtered_archs = resource_manager.filter_architectures(original_archs)
                    if filtered_archs != original_archs:
                        param.choices = filtered_archs
                        logger.info(f"Filtered architectures from {original_archs} to {filtered_archs}")
            logger.info("Search space adjusted successfully")
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
        embedding_store=None,
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
        embedding_store=None,
        mean=mean,
        std=std,
    )

    # Prepare dataloader kwargs
    nw, prefetch, persistent = resolve_loader_runtime(args, device)
    dl_kwargs = {
        "num_workers": nw,
        "persistent_workers": bool(persistent and nw > 0),
        "pin_memory": device.type == "cuda",
        "collate_fn": mammo_collate,
    }
    if prefetch is not None and nw > 0:
        dl_kwargs["prefetch_factor"] = prefetch

    # Run learning rate finder if enabled
    suggested_lr = None
    if args.lr_finder_enabled:
        logger.info("=" * 80)
        logger.info("Running Learning Rate Finder...")
        logger.info("=" * 80)
        try:
            from mammography.tuning.lr_finder import LRFinder
            from mammography.models.nets import build_model
            import torch.nn as nn
            import torch.optim as optim

            # Create temporary model and optimizer for LR finder
            temp_model = build_model(args.arch, num_classes, pretrained=args.pretrained)
            temp_model = temp_model.to(device)
            temp_optimizer = optim.SGD(temp_model.parameters(), lr=args.lr_finder_start_lr)
            temp_criterion = nn.CrossEntropyLoss()

            # AutoML does not expose a fixed batch size on the outer CLI; use
            # the training default for the LR finder bootstrap.
            lr_finder_batch_size = int(getattr(args, "batch_size", HP.BATCH_SIZE))
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=lr_finder_batch_size,
                shuffle=True,
                **dl_kwargs,
            )

            # Initialize LR finder with model components
            lr_finder = LRFinder(
                model=temp_model,
                optimizer=temp_optimizer,
                criterion=temp_criterion,
                device=device,
                amp_enabled=args.amp,
            )
            suggested_lr = lr_finder.range_test(
                train_loader=train_loader,
                start_lr=args.lr_finder_start_lr,
                end_lr=args.lr_finder_end_lr,
                num_iter=max(len(train_loader) * args.lr_finder_epochs, 10),
            )
            logger.info(f"Suggested learning rate: {suggested_lr:.2e}")

            # Save plot if available
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            lr_finder.plot(save_path=str(outdir_path / "lr_finder_plot.png"))
            logger.info("LR finder plot saved to: %s", outdir_path / "lr_finder_plot.png")

            # Clean up temporary objects
            del temp_model, temp_optimizer, temp_criterion, train_loader

            # Update search space with suggested LR if available
            if suggested_lr and "lr" in search_space.parameters:
                logger.info("Updating search space with suggested learning rate bounds")
                # Use suggested_lr as midpoint, with ±1 order of magnitude range
                lr_min = max(suggested_lr / 10, 1e-6)
                lr_max = min(suggested_lr * 10, 1.0)
                search_space.parameters["lr"].low = lr_min
                search_space.parameters["lr"].high = lr_max
                logger.info(f"LR search range adjusted: [{lr_min:.2e}, {lr_max:.2e}]")
        except Exception as exc:
            logger.error(f"LR finder failed: {exc}")
            logger.warning("Continuing with default search space...")

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

    # Save configuration before optimization
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
        "lr_finder_enabled": args.lr_finder_enabled,
        "suggested_lr": suggested_lr,
        "resource_aware": args.resource_aware,
        "resource_constraints": resource_constraints,
    }
    config_path = outdir_path / "automl_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    logger.info(f"Configuration saved to {config_path}")

    # Initialize experiment tracker if requested
    tracker = None
    if args.tracker == "local":
        tracker_db_path = outdir_root / "experiments.db"
        tracker = LocalTracker(
            db_path=tracker_db_path,
            experiment_name=f"automl_{args.arch}_{args.classes}",
            run_name=args.study_name or f"run_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            params={
                "arch": args.arch,
                "classes": args.classes,
                "n_trials": args.n_trials,
                "epochs_per_trial": args.epochs,
                "dataset": args.dataset or "custom",
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "lr_finder_enabled": args.lr_finder_enabled,
                "suggested_lr": suggested_lr,
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
        logger.info(f"LR finder enabled: {args.lr_finder_enabled}")
        logger.info(f"Suggested LR: {suggested_lr:.2e}" if suggested_lr else "LR finder disabled")
        logger.info(f"Resource-aware: {args.resource_aware}")
        logger.info(f"Output directory: {outdir_path}")
        logger.info(f"Tracker: {args.tracker}")
        logger.info("=" * 80)
        logger.info("Dry-run complete. No optimization performed.")
        if tracker:
            tracker.finish()
        return 0

    # Create OptunaTuner instance
    logger.info("Initializing OptunaTuner for AutoML...")
    from mammography.tuning.optuna_tuner import OptunaTuner
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

    # Run optimization
    logger.info("=" * 80)
    logger.info(
        "Starting AutoML optimization: %d trials, %d epochs per trial",
        args.n_trials,
        args.epochs,
    )
    logger.info("=" * 80)

    # Ensure storage directory exists if using persistent storage
    if args.storage and args.storage.startswith("sqlite:"):
        storage_path = args.storage.replace("sqlite:///", "").replace("sqlite://", "")
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    study = tuner.optimize(
        n_trials=args.n_trials,
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

    # Save best parameters
    best_params_payload = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "datetime": datetime.now(tz=timezone.utc).isoformat(),
        "study_name": study.study_name,
        "lr_finder_suggested_lr": suggested_lr,
        "resource_constraints": resource_constraints,
    }
    best_params_path = outdir_root / "best_params.json"
    best_params_path.write_text(
        json.dumps(best_params_payload, indent=2), encoding="utf-8"
    )
    logger.info("Best params summary saved to %s", best_params_path)

    # Export best configuration as training command
    export_path = outdir_path / "best_train_command.sh"
    best_params = study.best_params
    train_command_parts = [
        "py -m mammography.cli train-density",
        f"--dataset {args.dataset or 'custom'}",
        f"--csv {csv_path}" if csv_path else "",
        f"--dicom-root {dicom_root}" if dicom_root else "",
        f"--arch {best_params.get('arch', args.arch)}",
        f"--classes {args.classes}",
        f"--epochs 50",  # Use full epochs for final training
        f"--batch-size {best_params.get('batch_size', 32)}",
        f"--lr {best_params.get('lr', 1e-3):.2e}",
        f"--weight-decay {best_params.get('weight_decay', args.weight_decay)}",
        f"--img-size {args.img_size}",
        f"--seed {args.seed}",
    ]
    # Add augmentation flags if enabled in best params
    if best_params.get("augment", args.augment):
        train_command_parts.append("--augment")
    if best_params.get("augment_vertical", args.augment_vertical):
        train_command_parts.append("--augment-vertical")
    if best_params.get("augment_color", args.augment_color):
        train_command_parts.append("--augment-color")

    train_command = " \\\n    ".join([p for p in train_command_parts if p])
    export_path.write_text(train_command, encoding="utf-8")
    logger.info("Best training command exported to %s", export_path)

    # Report results
    logger.info("=" * 80)
    logger.info("AUTOML OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for param_name, param_value in study.best_params.items():
        logger.info(f"  {param_name}: {param_value}")
    logger.info(f"Total trials: {n_trials}")
    logger.info(f"Completed trials: {completed_trials}")
    logger.info(f"Pruned trials: {pruned_trials}")
    if suggested_lr:
        logger.info(f"LR finder suggested: {suggested_lr:.2e}")
    logger.info(f"Results saved to: {outdir_path}")
    logger.info(f"Best configuration: {best_params_path}")
    logger.info(f"Training command: {export_path}")
    if tracker:
        logger.info(f"Tracker database: {tracker.db_path}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
