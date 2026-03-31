#!/usr/bin/env python3
#
# cross_validate.py
# mammography-pipelines
#
# K-fold cross-validation for breast density classification with automated metrics aggregation.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""K-fold cross-validation for breast density classification."""
import os
import sys
import argparse
import logging
from typing import Optional, Sequence
from pathlib import Path

try:
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import ValidationError

from mammography.config import HP, TrainConfig
from mammography.utils.class_modes import (
    CLASS_MODE_HELP,
    VISIBLE_CLASS_MODES_METAVAR,
    parse_classes_mode_arg,
)
from mammography.utils.common import (
    seed_everything,
    resolve_device,
    configure_runtime,
    setup_logging,
    increment_path,
    parse_float_list,
)
from mammography.data.csv_loader import (
    load_dataset_dataframe,
    DATASET_PRESETS,
    resolve_paths_from_preset,
)
from mammography.training.cv_engine import CrossValidationEngine


_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


def _parse_bool_literal(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    return None


def _normalize_bool_flags(argv: Optional[Sequence[str]]) -> list[str] | None:
    """Normalize '--flag true/false' tokens to argparse-compatible bool flags."""
    if argv is None:
        return None
    normalized: list[str] = []
    bool_flags = {
        "--sampler-weighted": ("--sampler-weighted", "--no-sampler-weighted"),
        "--unfreeze-last-block": ("--unfreeze-last-block", "--no-unfreeze-last-block"),
        "--augment": ("--augment", "--no-augment"),
        "--save-all-folds": ("--save-all-folds", "--no-save-all-folds"),
    }
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        mapping = bool_flags.get(token)
        if mapping and idx + 1 < len(argv):
            literal = _parse_bool_literal(argv[idx + 1])
            if literal is not None:
                normalized.append(mapping[0] if literal else mapping[1])
                idx += 2
                continue
        normalized.append(token)
        idx += 1
    return normalized


def parse_args(argv: Optional[Sequence[str]] = None):
    """Define and parse CLI arguments for cross-validation."""
    argv = _normalize_bool_flags(argv)
    parser = argparse.ArgumentParser(
        description="Cross-validation for breast density classification (EfficientNetB0/ResNet50)"
    )

    # Cross-validation specific arguments
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=42,
        help="Random seed for fold splitting (default: 42)",
    )
    parser.add_argument(
        "--save-all-folds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save checkpoints for all folds (default: False, only best fold)",
    )

    # Data
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        help="Atalho para datasets conhecidos (archive/mamografias/patches_completo)",
    )
    parser.add_argument("--csv", required=False, help="CSV, diretório com featureS.txt ou caminho manual")
    parser.add_argument("--dicom-root", help="Root for DICOMs (usado com classificacao.csv)")
    parser.add_argument(
        "--include-class-5",
        action="store_true",
        help="Mantém amostras com classificação 5 ao carregar classificacao.csv",
    )
    parser.add_argument(
        "--auto-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detecta automaticamente formato do dataset a partir da estrutura de diretórios (default: True)",
    )
    parser.add_argument("--outdir", default="outputs/cv_run", help="Output directory")
    parser.add_argument(
        "--cache-mode",
        default=HP.CACHE_MODE,
        choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"],
    )
    parser.add_argument("--cache-dir", help="Cache dir")
    parser.add_argument("--embeddings-dir", help="Diretorio com features.npy + metadata.csv (embeddings)")
    parser.add_argument("--mean", help="Media de normalizacao (ex: 0.485,0.456,0.406)")
    parser.add_argument("--std", help="Std de normalizacao (ex: 0.229,0.224,0.225)")
    parser.add_argument(
        "--log-level",
        default=HP.LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument("--subset", type=int, default=0, help="Limita o número de amostras")

    # Model / task
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
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

    # HP overrides
    parser.add_argument("--epochs", type=int, default=HP.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=HP.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=HP.LR)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=HP.BACKBONE_LR,
        help="Learning rate para o backbone (cabeça usa --lr)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
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
    parser.add_argument("--amp", action="store_true", help="Habilita autocast + GradScaler em CUDA/MPS")
    parser.add_argument(
        "--class-weights",
        default=HP.CLASS_WEIGHTS,
        help="auto/none ou lista (ex: 1.0,0.8,1.2,1.0)",
    )
    parser.add_argument("--class-weights-alpha", type=float, default=1.0, help="Expoente para class_weights auto")
    parser.add_argument(
        "--sampler-weighted",
        action=argparse.BooleanOptionalAction,
        default=HP.SAMPLER_WEIGHTED,
    )
    parser.add_argument("--sampler-alpha", type=float, default=1.0, help="Expoente para sampler ponderado")
    parser.add_argument(
        "--train-backbone",
        action=argparse.BooleanOptionalAction,
        default=HP.TRAIN_BACKBONE,
    )
    parser.add_argument(
        "--unfreeze-last-block",
        action=argparse.BooleanOptionalAction,
        default=HP.UNFREEZE_LAST_BLOCK,
    )
    parser.add_argument("--warmup-epochs", type=int, default=HP.WARMUP_EPOCHS)
    parser.add_argument("--deterministic", action="store_true", default=HP.DETERMINISTIC)
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=HP.ALLOW_TF32,
    )
    parser.add_argument(
        "--fused-optim",
        action="store_true",
        default=HP.FUSED_OPTIM,
        help="Ativa AdamW(fused=True) em CUDA quando disponível",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=HP.TORCH_COMPILE,
        help="Otimiza o modelo com torch.compile quando suportado",
    )
    parser.add_argument("--lr-reduce-patience", type=int, default=HP.LR_REDUCE_PATIENCE)
    parser.add_argument("--lr-reduce-factor", type=float, default=HP.LR_REDUCE_FACTOR)
    parser.add_argument("--lr-reduce-min-lr", type=float, default=HP.LR_REDUCE_MIN_LR)
    parser.add_argument("--lr-reduce-cooldown", type=int, default=HP.LR_REDUCE_COOLDOWN)
    parser.add_argument(
        "--scheduler",
        choices=["auto", "none", "plateau", "cosine", "step"],
        default="auto",
    )
    parser.add_argument("--scheduler-min-lr", type=float, default=HP.LR_REDUCE_MIN_LR)
    parser.add_argument("--scheduler-step-size", type=int, default=5)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=HP.EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=HP.EARLY_STOP_MIN_DELTA)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=HP.TRAIN_AUGMENT,
        help="Ativa augmentations no treino",
    )
    parser.add_argument("--augment-vertical", action="store_true", help="Habilita flip vertical aleatorio")
    parser.add_argument("--augment-color", action="store_true", help="Habilita color jitter (brightness/contrast)")
    parser.add_argument("--augment-rotation-deg", type=float, default=5.0, help="Amplitude da rotacao aleatoria")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    """Main entry point for cross-validation command."""
    args = parse_args(argv)

    # Handle environment variable overrides
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
    if override_epochs:
        try:
            args.epochs = max(1, int(override_epochs))
        except ValueError:
            logging.getLogger("mammography").warning(
                "MAMMO_EPOCHS invalido (%s). Usando valor de linha de comando.",
                override_epochs,
            )

    # Resolve dataset paths
    csv_path, dicom_root = resolve_paths_from_preset(args.csv, args.dataset, args.dicom_root)
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

    # Seed everything for reproducibility
    seed_everything(args.seed, deterministic=args.deterministic)

    # Create output directory
    outdir_root = Path(increment_path(args.outdir))
    outdir_root.mkdir(parents=True, exist_ok=True)
    outdir = str(outdir_root)

    # Setup logging
    logger = setup_logging(outdir, args.log_level)
    logger.info(f"Args: {args}")
    logger.info("Resultados de cross-validation serao gravados em: %s", outdir_root)
    logger.info("Cross-validation: n_folds=%d, cv_seed=%d", args.n_folds, args.cv_seed)

    # Configure device and runtime
    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)

    # Parse normalization parameters
    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Update config with parsed mean/std
    if mean is not None:
        cfg.mean = mean
    if std is not None:
        cfg.std = std

    # Update config output directory
    cfg.outdir = outdir

    # Create and run cross-validation engine
    logger.info("Iniciando cross-validation com %d folds...", args.n_folds)
    cv_engine = CrossValidationEngine(
        config=cfg,
        n_folds=args.n_folds,
        cv_seed=args.cv_seed,
        save_all_folds=args.save_all_folds,
    )

    try:
        results = cv_engine.run()
        logger.info("Cross-validation concluida com sucesso!")
        logger.info("Resultados agregados:")
        logger.info("  Accuracy: %.3f ± %.3f", results["mean_val_acc"], results["std_val_acc"])
        logger.info("  Kappa: %.3f ± %.3f", results["mean_val_kappa"], results["std_val_kappa"])
        logger.info("  Macro F1: %.3f ± %.3f", results["mean_val_macro_f1"], results["std_val_macro_f1"])
        if results.get("mean_val_auc") is not None:
            logger.info("  AUC: %.3f ± %.3f", results["mean_val_auc"], results["std_val_auc"])
        logger.info("Detalhes salvos em: %s/cv_summary.json", outdir)
        return 0
    except KeyboardInterrupt:
        logger.warning("Cross-validation interrompida pelo usuario.")
        return 1
    except Exception as exc:
        logger.exception("Erro durante cross-validation: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
