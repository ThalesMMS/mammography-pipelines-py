#!/usr/bin/env python3
#
# train_args.py
# mammography-pipelines
#
# Argument parsing helpers for density training commands.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Argument parsing helpers for mammography density training."""

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Sequence, Union

from mammography.config import HP, TrainConfig
from mammography.data.csv_loader import (
    DATASET_PRESETS,
)
from mammography.utils import bool_flags as bool_flag_utils
from mammography.utils.class_modes import (
    CLASS_MODE_HELP,
    VISIBLE_CLASS_MODES_METAVAR,
    parse_classes_mode_arg,
)


def _parse_bool_literal(value: str | None) -> bool | None:
    """
    Parse a boolean literal string into its boolean value.

    Parameters:
        value (str | None): A textual boolean literal (e.g., "true", "false", case-insensitive) or None.

    Returns:
        bool | None: `True` if `value` represents true, `False` if it represents false, or `None` if `value` is `None` or not a boolean literal.
    """
    return bool_flag_utils.parse_bool_literal(value)


def _normalize_bool_flags(argv: Optional[Sequence[str]]) -> list[str] | None:
    """
    Normalize boolean flag spellings in an argv sequence for consistent argument parsing.

    Parameters:
        argv (Optional[Sequence[str]]): Sequence of command-line arguments (e.g., sys.argv[1:]) or None.

    Returns:
        list[str] | None: A normalized argument list with boolean flags standardized, or `None` if `argv` is `None`.
    """
    return bool_flag_utils.normalize_bool_flags(argv)


def _argv_has_option(argv: Sequence[str], option: str) -> bool:
    """Return True when `argv` contains `option` or an `option=value` token."""
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def parse_args(argv: Optional[Union[Sequence[str], TrainConfig]] = None):
    """
    Builds the command-line argument parser and returns parsed training options.

    When `argv` is a sequence of strings it is treated like sys.argv[1:] (boolean flags are normalized before parsing).
    When `argv` is a `TrainConfig` instance, parser defaults are loaded and any matching `TrainConfig` fields overwrite those defaults;
    if `views_to_train` is present on the config it is converted to a comma-separated `views` string.

    Parameters:
        argv (Optional[Sequence[str] | TrainConfig]): CLI argv to parse or a `TrainConfig` object to map into parser defaults.
            If `None`, sys.argv[1:] is used.

    Returns:
        argparse.Namespace: Namespace with all parsed CLI options (fields correspond to the registered arguments).
    """
    parser = argparse.ArgumentParser(
        description="Treinamento Mammography (EfficientNetB0/ResNet50/ViT)"
    )

    # Data
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
    parser.add_argument(
        "--auto-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detecta automaticamente formato do dataset a partir da estrutura de diretórios (default: True)",
    )
    parser.add_argument("--outdir", default="outputs/run", help="Output directory")
    parser.add_argument(
        "--cache-mode",
        default=HP.CACHE_MODE,
        choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"],
    )
    parser.add_argument("--cache-dir", help="Cache dir")
    parser.add_argument(
        "--embeddings-dir",
        help="Diretorio com features.npy + metadata.csv (embeddings)",
    )
    parser.add_argument("--mean", help="Media de normalizacao (ex: 0.485,0.456,0.406)")
    parser.add_argument("--std", help="Std de normalizacao (ex: 0.229,0.224,0.225)")
    parser.add_argument(
        "--log-level",
        default=HP.LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument(
        "--subset", type=int, default=0, help="Limita o número de amostras"
    )

    # View-specific training
    parser.add_argument(
        "--view-specific-training",
        action="store_true",
        help="Treina modelos separados para cada view (CC/MLO)",
    )
    parser.add_argument(
        "--view-column",
        default="view",
        help="Nome da coluna que contem a view no CSV (default: view)",
    )
    parser.add_argument(
        "--views",
        default="CC,MLO",
        help="Lista de views separadas por virgula para treinar (default: CC,MLO)",
    )
    parser.add_argument(
        "--ensemble-method",
        default="none",
        choices=["none", "average", "weighted", "max"],
        help="Metodo de ensemble para combinar predicoes de views (none/average/weighted/max)",
    )

    # Model / task
    parser.add_argument(
        "--arch",
        default="efficientnet_b0",
        choices=[
            "efficientnet_b0",
            "resnet50",
            "vit_b_16",
            "vit_b_32",
            "vit_l_16",
            "deit_small",
            "deit_base",
        ],
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
    parser.add_argument("--val-frac", type=float, default=HP.VAL_FRAC)
    parser.add_argument(
        "--split-mode",
        choices=["random", "patient", "preset"],
        default="random",
        help="Modo de split: random (aleatorio), patient (por paciente), preset (CSVs pre-definidos)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        help="Fracao para split de teste (0.0 = sem teste)",
    )
    parser.add_argument(
        "--train-csv",
        help="CSV pre-definido para treino (usado com --split-mode=preset)",
    )
    parser.add_argument(
        "--val-csv",
        help="CSV pre-definido para validacao (usado com --split-mode=preset)",
    )
    parser.add_argument(
        "--test-csv", help="CSV pre-definido para teste (usado com --split-mode=preset)"
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
        "--class-weights",
        default=HP.CLASS_WEIGHTS,
        help="auto/none ou lista (ex: 1.0,0.8,1.2,1.0)",
    )
    parser.add_argument(
        "--class-weights-alpha",
        type=float,
        default=1.0,
        help="Expoente para class_weights auto",
    )
    parser.add_argument(
        "--sampler-weighted",
        action=argparse.BooleanOptionalAction,
        default=HP.SAMPLER_WEIGHTED,
    )
    parser.add_argument(
        "--sampler-alpha",
        type=float,
        default=1.0,
        help="Expoente para sampler ponderado",
    )
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
    parser.add_argument(
        "--deterministic", action="store_true", default=HP.DETERMINISTIC
    )
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=HP.ALLOW_TF32
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Habilita torch.profiler no primeiro epoch",
    )
    parser.add_argument(
        "--profile-dir",
        default=os.path.join("outputs", "profiler"),
        help="Destino dos traces do profiler",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=HP.EARLY_STOP_PATIENCE
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=HP.EARLY_STOP_MIN_DELTA
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
        help="Habilita flip vertical aleatorio",
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
        "--resume-from", help="Checkpoint para retomar treino (ex: checkpoints/last.pt)"
    )
    parser.add_argument(
        "--tracker",
        choices=["none", "wandb", "mlflow", "local"],
        default="none",
        help="Backend de tracking (none/wandb/mlflow/local)",
    )
    parser.add_argument("--tracker-project", help="Projeto/experimento para o tracker")
    parser.add_argument("--tracker-run-name", help="Nome opcional do run no tracker")
    parser.add_argument("--tracker-uri", help="Tracking URI (apenas MLflow)")

    # Outputs/analysis
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--gradcam-limit", type=int, default=4)
    parser.add_argument("--save-val-preds", action="store_true")
    parser.add_argument("--export-val-embeddings", action="store_true")
    parser.add_argument(
        "--export-figures",
        help="Formatos de exportacao para figuras publicacao (ex: png,pdf,svg)",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local apos o treino",
    )

    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, TrainConfig):
        args = parser.parse_args([])
        args._epochs_explicit = "epochs" in getattr(argv, "model_fields_set", set())
        config_values = argv.model_dump(mode="python")
        ignored_keys = {"views_to_train"}
        namespace_only_keys = {
            "auto_normalize",
            "auto_normalize_samples",
            "n_folds",
            "cv_fold",
            "cv_stratified",
        }
        explicitly_set = getattr(argv, "model_fields_set", set())
        unmapped_keys = sorted(
            key
            for key in config_values
            if key not in ignored_keys
            and key not in namespace_only_keys
            and not hasattr(args, key)
            and key in explicitly_set
        )
        if unmapped_keys:
            raise ValueError(
                "TrainConfig contains fields that are not mapped to the "
                "training argparse Namespace: "
                f"{', '.join(unmapped_keys)}. Add matching CLI options or "
                "remove those TrainConfig fields before calling parse_args()."
            )
        for key, value in config_values.items():
            if hasattr(args, key) or key in namespace_only_keys:
                setattr(args, key, value)
        if argv.views_to_train is not None:
            args.views = ",".join(argv.views_to_train)
        return args

    original_argv = list(argv)
    argv = _normalize_bool_flags(argv)
    args = parser.parse_args(argv)
    args._epochs_explicit = _argv_has_option(original_argv, "--epochs")
    return args
