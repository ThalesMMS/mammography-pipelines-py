# ruff: noqa
#
# wizard.py
# mammography-pipelines
#
# Interactive wizard for guided mammography pipeline workflows.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mammography.config import HP
from mammography.data.csv_loader import DATASET_PRESETS
from mammography.data.format_detection import (
    detect_dataset_format,
    validate_format,
    suggest_preprocessing,
)
from mammography.utils.smart_defaults import SmartDefaults
from mammography.utils.wizard_helpers import (
    ask_choice_with_help,
    ask_float_with_help,
    ask_int_with_help,
    ask_optional_with_help,
    ask_with_help,
    ask_yes_no_with_help,
    print_help,
)

from mammography.wizard_core import (
    WizardCommand,
    _ask_choice,
    _ask_config_args,
    _ask_extra_args,
    _ask_float,
    _ask_int,
    _ask_optional,
    _ask_string,
    _ask_yes_no,
    _build_cli_command,
    _dataset_prompt,
    _print_progress,
    _validate_dataset_path,
)

def _wizard_train() -> WizardCommand:
    # Determine total steps: 1=dataset, 2=basic config, 3=advanced (optional)
    total_steps = 3
    current_step = 1

    _print_progress(current_step, total_steps, "Selecao de Dataset e Config")
    args = _ask_config_args()
    dataset_args, _, _ = _dataset_prompt()
    args.extend(dataset_args)

    # Initialize SmartDefaults for hardware-aware recommendations
    smart_defaults = SmartDefaults()
    print(f"\n→ Hardware detectado: {smart_defaults.device_type.upper()}")
    print(f"  (Recomendacoes baseadas no hardware serao sugeridas)")

    current_step += 1
    _print_progress(current_step, total_steps, "Configuracao Basica")
    arch_idx = ask_choice_with_help("Arquitetura:", ["efficientnet_b0", "resnet50"], param_name="arch", default=0)
    arch = ["efficientnet_b0", "resnet50"][arch_idx]
    class_idx = ask_choice_with_help("Classes:", ["multiclass (A-D)", "binary (AB vs CD)"], param_name="classes", default=0)
    classes = ["multiclass", "binary"][class_idx]

    outdir = ask_with_help("Outdir", param_name="outdir", default="outputs/mammo_efficientnetb0_density")
    epochs = ask_int_with_help("Epocas", param_name="epochs", default=HP.EPOCHS)

    # Get smart defaults based on hardware and task
    img_size = ask_int_with_help("Img size", param_name="img_size", default=HP.IMG_SIZE)
    smart_batch_size = smart_defaults.get_batch_size(task="train", image_size=img_size)
    batch_size = ask_int_with_help("Batch size", param_name="batch_size", default=smart_batch_size)

    device = ask_with_help("Device (auto/cuda/mps/cpu)", param_name="device", default=HP.DEVICE)
    smart_cache_mode = smart_defaults.get_cache_mode()
    cache_mode = ask_with_help("Cache mode", param_name="cache_mode", default=smart_cache_mode)
    pretrained = ask_yes_no_with_help("Usar pesos pretrained?", param_name="pretrained", default=True)

    args.extend(
        [
            "--arch",
            arch,
            "--classes",
            classes,
            "--outdir",
            outdir,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--img-size",
            str(img_size),
            "--device",
            device,
            "--cache-mode",
            cache_mode,
        ]
    )
    if not pretrained:
        args.append("--no-pretrained")

    if _ask_yes_no("Configurar opcoes avancadas?", False):
        current_step += 1
        _print_progress(current_step, total_steps, "Opcoes Avancadas")
        include_class_5 = ask_yes_no_with_help("Incluir classe 5?", param_name="include_class_5", default=False)
        if include_class_5:
            args.append("--include-class-5")

        cache_dir = ask_optional_with_help("Cache dir", param_name="cache_dir")
        if cache_dir:
            args.extend(["--cache-dir", cache_dir])

        embeddings_dir = ask_optional_with_help("Embeddings (diretorio)", param_name="embeddings_dir")
        if embeddings_dir:
            args.extend(["--embeddings-dir", embeddings_dir])

        seed = ask_int_with_help("Seed", param_name="seed", default=HP.SEED)
        args.extend(["--seed", str(seed)])
        val_frac = ask_float_with_help("Val frac", param_name="val_frac", default=HP.VAL_FRAC)
        args.extend(["--val-frac", str(val_frac)])

        if not ask_yes_no_with_help("Garantir todas as classes no val?", param_name="split_ensure_all_classes", default=True):
            args.append("--no-split-ensure-all-classes")
        split_max_tries = ask_int_with_help("Split max tries", param_name="split_max_tries", default=200)
        args.extend(["--split-max-tries", str(split_max_tries)])

        if not ask_yes_no_with_help("Ativar augmentations?", param_name="augment", default=True):
            args.append("--no-augment")
        else:
            if ask_yes_no_with_help("Flip vertical?", param_name="augment_vertical", default=False):
                args.append("--augment-vertical")
            if ask_yes_no_with_help("Color jitter?", param_name="augment_color", default=False):
                args.append("--augment-color")
            rotation_deg = ask_float_with_help("Rotacao (graus)", param_name="augment_rotation_deg", default=5.0)
            if rotation_deg != 5.0:
                args.extend(["--augment-rotation-deg", str(rotation_deg)])

        if _ask_yes_no("Normalizacao customizada?", False):
            mean = ask_with_help("Mean (ex: 0.485,0.456,0.406)", param_name="mean", default="0.485,0.456,0.406")
            std = ask_with_help("Std (ex: 0.229,0.224,0.225)", param_name="std", default="0.229,0.224,0.225")
            args.extend(["--mean", mean, "--std", std])

        class_choice = ask_choice_with_help("Class weights:", ["auto", "none", "manual"], param_name="class_weights", default=0)
        if class_choice == 2:
            weights = _ask_string("Pesos (ex: 1.0,0.8,1.2,1.0)")
            args.extend(["--class-weights", weights])
        else:
            args.extend(["--class-weights", ["auto", "none"][class_choice]])
        if class_choice == 0:
            alpha = ask_float_with_help("Class weights alpha", param_name="class_weights_alpha", default=1.0)
            if alpha != 1.0:
                args.extend(["--class-weights-alpha", str(alpha)])

        if ask_yes_no_with_help("Usar sampler ponderado?", param_name="sampler_weighted", default=True):
            args.append("--sampler-weighted")
            sampler_alpha = ask_float_with_help("Sampler alpha", param_name="sampler_alpha", default=1.0)
            if sampler_alpha != 1.0:
                args.extend(["--sampler-alpha", str(sampler_alpha)])

        if ask_yes_no_with_help("Treinar todo o backbone?", param_name="train_backbone", default=False):
            args.append("--train-backbone")
        else:
            if ask_yes_no_with_help("Descongelar ultimo bloco?", param_name="unfreeze_last_block", default=True):
                args.append("--unfreeze-last-block")
            else:
                args.append("--no-unfreeze-last-block")

        lr = ask_float_with_help("Learning rate", param_name="lr", default=HP.LR)
        backbone_lr = ask_float_with_help("Backbone LR", param_name="backbone_lr", default=HP.BACKBONE_LR)
        weight_decay = ask_float_with_help("Weight decay", param_name="weight_decay", default=1e-4)
        args.extend(["--lr", str(lr), "--backbone-lr", str(backbone_lr), "--weight-decay", str(weight_decay)])

        warmup_epochs = ask_int_with_help("Warmup epochs", param_name="warmup_epochs", default=HP.WARMUP_EPOCHS)
        args.extend(["--warmup-epochs", str(warmup_epochs)])

        early_stop_patience = ask_int_with_help("Early stop patience (0 = off)", param_name="early_stop_patience", default=HP.EARLY_STOP_PATIENCE)
        args.extend(["--early-stop-patience", str(early_stop_patience)])
        early_stop_min_delta = ask_float_with_help("Early stop min delta", param_name="early_stop_min_delta", default=HP.EARLY_STOP_MIN_DELTA)
        args.extend(["--early-stop-min-delta", str(early_stop_min_delta)])

        scheduler_idx = ask_choice_with_help("Scheduler:", ["auto", "none", "plateau", "cosine", "step"], param_name="scheduler", default=0)
        scheduler = ["auto", "none", "plateau", "cosine", "step"][scheduler_idx]
        if scheduler != "auto":
            args.extend(["--scheduler", scheduler])
        if scheduler == "plateau":
            lr_patience = ask_int_with_help("LR reduce patience", param_name="lr_reduce_patience", default=HP.LR_REDUCE_PATIENCE)
            lr_factor = ask_float_with_help("LR reduce factor", param_name="lr_reduce_factor", default=HP.LR_REDUCE_FACTOR)
            lr_min = ask_float_with_help("LR reduce min lr", param_name="lr_reduce_min_lr", default=HP.LR_REDUCE_MIN_LR)
            lr_cooldown = ask_int_with_help("LR reduce cooldown", param_name="lr_reduce_cooldown", default=HP.LR_REDUCE_COOLDOWN)
            args.extend(
                [
                    "--lr-reduce-patience",
                    str(lr_patience),
                    "--lr-reduce-factor",
                    str(lr_factor),
                    "--lr-reduce-min-lr",
                    str(lr_min),
                    "--lr-reduce-cooldown",
                    str(lr_cooldown),
                ]
            )
        elif scheduler == "cosine":
            min_lr = ask_float_with_help("Scheduler min lr", param_name="scheduler_min_lr", default=HP.LR_REDUCE_MIN_LR)
            args.extend(["--scheduler-min-lr", str(min_lr)])
        elif scheduler == "step":
            step_size = ask_int_with_help("Scheduler step size", param_name="scheduler_step_size", default=5)
            gamma = ask_float_with_help("Scheduler gamma", param_name="scheduler_gamma", default=0.5)
            args.extend(["--scheduler-step-size", str(step_size), "--scheduler-gamma", str(gamma)])

        smart_num_workers = smart_defaults.get_num_workers(task="train")
        num_workers = ask_int_with_help("Num workers", param_name="num_workers", default=smart_num_workers)
        args.extend(["--num-workers", str(num_workers)])
        prefetch_factor = ask_int_with_help("Prefetch factor (0 = off)", param_name="prefetch_factor", default=HP.PREFETCH_FACTOR)
        args.extend(["--prefetch-factor", str(prefetch_factor)])
        if not ask_yes_no_with_help("Persistent workers?", param_name="persistent_workers", default=HP.PERSISTENT_WORKERS):
            args.append("--no-persistent-workers")
        if not ask_yes_no_with_help("Loader heuristics?", param_name="loader_heuristics", default=HP.LOADER_HEURISTICS):
            args.append("--no-loader-heuristics")

        if ask_yes_no_with_help("Habilitar AMP?", param_name="amp", default=False):
            args.append("--amp")
        if ask_yes_no_with_help("Ativar torch.compile?", param_name="torch_compile", default=False):
            args.append("--torch-compile")
        if ask_yes_no_with_help("Ativar fused optimizer?", param_name="fused_optim", default=False):
            args.append("--fused-optim")
        if ask_yes_no_with_help("Salvar predicoes de validacao?", param_name="save_val_preds", default=False):
            args.append("--save-val-preds")

        if ask_yes_no_with_help("Salvar Grad-CAM?", param_name="gradcam", default=False):
            args.append("--gradcam")
            gradcam_limit = ask_int_with_help("Grad-CAM limit", param_name="gradcam_limit", default=4)
            args.extend(["--gradcam-limit", str(gradcam_limit)])
        if ask_yes_no_with_help("Exportar embeddings de validacao?", param_name="export_val_embeddings", default=False):
            args.append("--export-val-embeddings")

        subset = ask_int_with_help("Subset (0 = desativado)", param_name="subset", default=0)
        if subset > 0:
            args.extend(["--subset", str(subset)])

        if ask_yes_no_with_help("Habilitar profiler?", param_name="profile", default=False):
            args.append("--profile")
            profile_dir = ask_with_help("Profile dir", param_name="profile_dir", default="outputs/profiler")
            args.extend(["--profile-dir", profile_dir])

        if ask_yes_no_with_help("Deterministic?", param_name="deterministic", default=HP.DETERMINISTIC):
            args.append("--deterministic")
        if not ask_yes_no_with_help("Permitir TF32?", param_name="allow_tf32", default=HP.ALLOW_TF32):
            args.append("--no-allow-tf32")

        log_level = ask_with_help("Log level", param_name="log_level", default=HP.LOG_LEVEL)
        args.extend(["--log-level", log_level])

    args.extend(_ask_extra_args())
    return WizardCommand("Treinamento de densidade", _build_cli_command("train-density", args))

def _wizard_quick_train() -> WizardCommand:
    args = _ask_config_args()
    dataset_args, _, _ = _dataset_prompt()
    args.extend(dataset_args)

    arch_idx = _ask_choice("Arquitetura:", ["efficientnet_b0", "resnet50"], default=0)
    arch = ["efficientnet_b0", "resnet50"][arch_idx]
    outdir = _ask_string("Outdir", "outputs/quick_train")
    args.extend(
        [
            "--arch",
            arch,
            "--classes",
            "binary",
            "--epochs",
            "5",
            "--batch-size",
            "16",
            "--outdir",
            outdir,
            "--cache-mode",
            "auto",
            "--amp",
            "--save-val-preds",
            "--export-val-embeddings",
        ]
    )
    return WizardCommand("Treino rapido", _build_cli_command("train-density", args))

def _wizard_embed() -> WizardCommand:
    # Determine total steps: 1=dataset, 2=basic config, 3=advanced (optional)
    total_steps = 3
    current_step = 1

    _print_progress(current_step, total_steps, "Selecao de Dataset e Config")
    args = _ask_config_args()
    dataset_args, _, _ = _dataset_prompt()
    args.extend(dataset_args)

    # Initialize SmartDefaults for hardware-aware recommendations
    smart_defaults = SmartDefaults()
    print(f"\n→ Hardware detectado: {smart_defaults.device_type.upper()}")
    print(f"  (Recomendacoes baseadas no hardware serao sugeridas)")

    current_step += 1
    _print_progress(current_step, total_steps, "Configuracao Basica")
    arch_idx = ask_choice_with_help(
        "Arquitetura:", ["resnet50", "efficientnet_b0"], param_name="arch", default=0
    )
    arch = ["resnet50", "efficientnet_b0"][arch_idx]
    class_idx = ask_choice_with_help(
        "Classes:",
        ["multiclass (A-D)", "binary (AB vs CD)"],
        param_name="classes",
        default=0,
    )
    classes = ["multiclass", "binary"][class_idx]

    outdir = ask_with_help("Outdir", param_name="outdir", default="outputs/features")

    # Get smart defaults based on hardware and task
    img_size = ask_int_with_help("Img size", param_name="img_size", default=HP.IMG_SIZE)
    smart_batch_size = smart_defaults.get_batch_size(task="embed", image_size=img_size)
    batch_size = ask_int_with_help("Batch size", param_name="batch_size", default=smart_batch_size)

    device = ask_with_help("Device (auto/cuda/mps/cpu)", param_name="device", default=HP.DEVICE)
    smart_cache_mode = smart_defaults.get_cache_mode()
    cache_mode = ask_with_help("Cache mode", param_name="cache_mode", default=smart_cache_mode)
    pretrained = ask_yes_no_with_help("Usar pesos pretrained?", param_name="pretrained", default=True)

    args.extend(
        [
            "--arch",
            arch,
            "--classes",
            classes,
            "--outdir",
            outdir,
            "--batch-size",
            str(batch_size),
            "--img-size",
            str(img_size),
            "--device",
            device,
            "--cache-mode",
            cache_mode,
        ]
    )
    if not pretrained:
        args.append("--no-pretrained")

    if _ask_yes_no("Configurar opcoes avancadas?", False):
        current_step += 1
        _print_progress(current_step, total_steps, "Opcoes Avancadas")
        include_class_5 = ask_yes_no_with_help(
            "Incluir classe 5?", param_name="include_class_5", default=False
        )
        if include_class_5:
            args.append("--include-class-5")

        seed = ask_int_with_help("Seed", param_name="seed", default=HP.SEED)
        args.extend(["--seed", str(seed)])
        if ask_yes_no_with_help(
            "Deterministic?", param_name="deterministic", default=HP.DETERMINISTIC
        ):
            args.append("--deterministic")
        if not ask_yes_no_with_help(
            "Permitir TF32?", param_name="allow_tf32", default=HP.ALLOW_TF32
        ):
            args.append("--no-allow-tf32")

        smart_num_workers = smart_defaults.get_num_workers(task="embed")
        num_workers = ask_int_with_help(
            "Num workers", param_name="num_workers", default=smart_num_workers
        )
        args.extend(["--num-workers", str(num_workers)])
        prefetch_factor = ask_int_with_help(
            "Prefetch factor (0 = off)", param_name="prefetch_factor", default=HP.PREFETCH_FACTOR
        )
        args.extend(["--prefetch-factor", str(prefetch_factor)])
        if not ask_yes_no_with_help(
            "Persistent workers?", param_name="persistent_workers", default=HP.PERSISTENT_WORKERS
        ):
            args.append("--no-persistent-workers")
        if not ask_yes_no_with_help(
            "Loader heuristics?", param_name="loader_heuristics", default=HP.LOADER_HEURISTICS
        ):
            args.append("--no-loader-heuristics")

        if _ask_yes_no("Normalizacao customizada?", False):
            mean = ask_with_help(
                "Mean (ex: 0.485,0.456,0.406)", param_name="mean", default="0.485,0.456,0.406"
            )
            std = ask_with_help(
                "Std (ex: 0.229,0.224,0.225)", param_name="std", default="0.229,0.224,0.225"
            )
            args.extend(["--mean", mean, "--std", std])

        layer_name = ask_with_help("Layer name (avgpool)", param_name="layer_name", default="avgpool")
        if layer_name and layer_name != "avgpool":
            args.extend(["--layer-name", layer_name])

        if ask_yes_no_with_help("Usar AMP?", param_name="amp", default=False):
            args.append("--amp")
        if _ask_yes_no("Salvar joined.csv?", False):
            args.append("--save-csv")

        wants_pca = False
        if _ask_yes_no("Atalho de reducao (PCA + t-SNE + UMAP)?", False):
            args.append("--run-reduction")
            wants_pca = True
        else:
            if _ask_yes_no("Executar PCA?", True):
                args.append("--pca")
                wants_pca = True
            if _ask_yes_no("Executar t-SNE?", True):
                args.append("--tsne")
            if _ask_yes_no("Executar UMAP?", True):
                args.append("--umap")

        if wants_pca:
            solver_idx = ask_choice_with_help(
                "Solver do PCA:",
                ["auto", "full", "randomized", "arpack"],
                param_name="pca_svd_solver",
                default=0,
            )
            args.extend(
                ["--pca-svd-solver", ["auto", "full", "randomized", "arpack"][solver_idx]]
            )

        if _ask_yes_no("Atalho de clustering (k-means auto)?", False):
            args.append("--run-clustering")
            cluster_k = ask_int_with_help("K fixo (0 = auto)", param_name="cluster_k", default=0)
            if cluster_k >= 2:
                args.extend(["--cluster-k", str(cluster_k)])
        elif _ask_yes_no("Executar clustering?", False):
            args.append("--cluster")
            cluster_k = ask_int_with_help("K fixo (0 = auto)", param_name="cluster_k", default=0)
            if cluster_k >= 2:
                args.extend(["--cluster-k", str(cluster_k)])
        sample_grid = ask_int_with_help("Sample grid", param_name="sample_grid", default=16)
        args.extend(["--sample-grid", str(sample_grid)])

        log_level = ask_with_help("Log level", param_name="log_level", default=HP.LOG_LEVEL)
        args.extend(["--log-level", log_level])

    args.extend(_ask_extra_args())
    return WizardCommand("Extracao de embeddings", _build_cli_command("embed", args))

def _wizard_visualize() -> WizardCommand:
    args = _ask_config_args()
    input_path = _ask_string("Caminho de entrada (features.npy ou run dir)")
    outdir = _ask_string("Outdir", "outputs/visualizations")
    from_run = _ask_yes_no("Tratar input como run dir?", False)
    args.extend(["--input", input_path, "--output", outdir])
    if from_run:
        args.append("--from-run")

    labels_path = _ask_optional("CSV de labels (opcional)")
    if labels_path:
        args.extend(["--labels", labels_path])
        label_col = _ask_string("Coluna de labels", "raw_label")
        if label_col:
            args.extend(["--label-col", label_col])

    predictions_path = _ask_optional("CSV de predictions (opcional)")
    if predictions_path:
        args.extend(["--predictions", predictions_path])

    history_path = _ask_optional("Historico de treino (CSV/JSON opcional)")
    if history_path:
        args.extend(["--history", history_path])

    prefix = _ask_optional("Prefixo de arquivos (opcional)")
    if prefix:
        args.extend(["--prefix", prefix])

    report = _ask_yes_no("Gerar report completo?", True)
    wants_pca = report
    wants_tsne = report
    if report:
        args.append("--report")
    else:
        if _ask_yes_no("Gerar PCA?", True):
            args.append("--pca")
            wants_pca = True
        if _ask_yes_no("Gerar t-SNE?", True):
            args.append("--tsne")
            wants_tsne = True
        if _ask_yes_no("Gerar t-SNE 3D?", False):
            args.append("--tsne-3d")
            wants_tsne = True
        if _ask_yes_no("Gerar UMAP?", False):
            args.append("--umap")
        if _ask_yes_no("Comparar embeddings?", False):
            args.append("--compare-embeddings")
            wants_pca = True
            wants_tsne = True
        if _ask_yes_no("Gerar heatmap?", False):
            args.append("--heatmap")
        if _ask_yes_no("Gerar heatmap de features?", False):
            args.append("--feature-heatmap")
        if _ask_yes_no("Gerar confusion matrix?", False):
            args.append("--confusion-matrix")
        if _ask_yes_no("Gerar scatter matrix?", False):
            args.append("--scatter-matrix")
        if _ask_yes_no("Gerar distribuicoes?", False):
            args.append("--distribution")
            wants_pca = True
        if _ask_yes_no("Gerar separacao de classes?", False):
            args.append("--class-separation")
        if _ask_yes_no("Gerar curvas de aprendizado?", False):
            args.append("--learning-curves")

    if _ask_yes_no("Rotulos binarios (low/high)?", False):
        args.append("--binary")

    if wants_tsne and _ask_yes_no("Configurar parametros do t-SNE?", False):
        perplexity = _ask_float("Perplexity", 30.0)
        tsne_iter = _ask_int("Iteracoes t-SNE", 1000)
        args.extend(["--perplexity", str(perplexity), "--tsne-iter", str(tsne_iter)])

    if wants_pca:
        solver_idx = _ask_choice(
            "Solver do PCA:",
            ["auto", "full", "randomized", "arpack"],
            default=0,
        )
        args.extend(
            ["--pca-svd-solver", ["auto", "full", "randomized", "arpack"][solver_idx]]
        )

    if _ask_yes_no("Configurar parametros gerais?", False):
        seed = _ask_int("Seed", 42)
        log_level = _ask_string("Log level", "INFO")
        args.extend(["--seed", str(seed), "--log-level", log_level])

    args.extend(_ask_extra_args())
    return WizardCommand("Visualizacao", _build_cli_command("visualize", args))

def _wizard_embeddings_baselines() -> WizardCommand:
    args = _ask_config_args()
    embeddings_dir = _ask_string("Embeddings dir", "outputs/embeddings_resnet50")
    outdir = _ask_string("Outdir", "outputs/embeddings_baselines")
    cache_classic = _ask_string("Cache classic (opcional)", "")
    img_size = _ask_int("Img size (resize classico)", 224)
    solver_idx = _ask_choice(
        "Solver do PCA:",
        ["auto", "full", "randomized", "arpack"],
        default=0,
    )
    args.extend(["--embeddings-dir", embeddings_dir, "--outdir", outdir, "--img-size", str(img_size)])
    if cache_classic:
        args.extend(["--cache-classic", cache_classic])
    args.extend(["--pca-svd-solver", ["auto", "full", "randomized", "arpack"][solver_idx]])
    args.extend(_ask_extra_args())
    return WizardCommand("Baselines (embeddings)", _build_cli_command("embeddings-baselines", args))

def _wizard_inference() -> WizardCommand:
    """
    Interactively collect inference configuration from the user and construct a CLI command for running model inference.
    
    Prompts for checkpoint path, input file/directory, architecture, class mode, image size, batch size, device, optional output CSV, optional custom normalization (mean/std), and optional AMP; includes any supplied config and extra CLI arguments.
    
    Returns:
        WizardCommand: A command labeled "Inferencia" whose argv is the full CLI invocation for the "inference" subcommand with the collected options.
    """
    args = _ask_config_args()
    checkpoint = _ask_string("Checkpoint (.pt)")
    input_path = _ask_string("Imagem ou diretorio de entrada")
    arch_idx = _ask_choice("Arquitetura:", ["resnet50", "efficientnet_b0"], default=0)
    arch = ["resnet50", "efficientnet_b0"][arch_idx]
    class_idx = _ask_choice("Classes:", ["multiclass (A-D)", "binary (AB vs CD)"], default=0)
    classes = ["multiclass", "binary"][class_idx]
    img_size = _ask_int("Img size", HP.IMG_SIZE)
    batch_size = _ask_int("Batch size", 16)
    device = _ask_string("Device (auto/cuda/mps/cpu)", HP.DEVICE)
    output_csv = _ask_string("CSV de saida (opcional)", "")

    args.extend(
        [
        "--checkpoint",
        checkpoint,
        "--input",
        input_path,
        "--arch",
        arch,
        "--classes",
        classes,
        "--img-size",
        str(img_size),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        ]
    )
    if output_csv:
        args.extend(["--output", output_csv])
    if _ask_yes_no("Normalizacao customizada?", False):
        mean = _ask_string("Mean (ex: 0.485,0.456,0.406)", "0.485,0.456,0.406")
        std = _ask_string("Std (ex: 0.229,0.224,0.225)", "0.229,0.224,0.225")
        args.extend(["--mean", mean, "--std", std])
    if _ask_yes_no("Usar AMP?", False):
        args.append("--amp")

    args.extend(_ask_extra_args())
    return WizardCommand("Inferencia", _build_cli_command("inference", args))

def _wizard_batch_inference() -> WizardCommand:
    """
    Builds a batch inference wizard command by interactively collecting configuration and dataset parameters.
    
    Prompts the user for checkpoint, input/output paths, model architecture, class mode, image size, batch size, device, output format, optional resume/checkpoint settings, and optional advanced runtime options (custom normalization, AMP, workers, prefetch, persistent workers). Appends any extra CLI arguments entered by the user.
    
    Returns:
        WizardCommand: A command labeled "Inferencia em lote (batch)" whose argv is the full CLI invocation for the "batch-inference" subcommand.
    """
    args = _ask_config_args()
    checkpoint = _ask_string("Checkpoint (.pt)")
    input_path = _ask_string("Diretorio de entrada (imagens/DICOMs)")
    output_path = _ask_string("Arquivo de saida")
    arch_idx = _ask_choice("Arquitetura:", ["resnet50", "efficientnet_b0"], default=0)
    arch = ["resnet50", "efficientnet_b0"][arch_idx]
    class_idx = _ask_choice("Classes:", ["multiclass (A-D)", "binary (AB vs CD)"], default=0)
    classes = ["multiclass", "binary"][class_idx]
    img_size = _ask_int("Img size", HP.IMG_SIZE)
    batch_size = _ask_int("Batch size", 16)
    device = _ask_string("Device (auto/cuda/mps/cpu)", HP.DEVICE)

    args.extend(
        [
            "--checkpoint",
            checkpoint,
            "--input",
            input_path,
            "--output",
            output_path,
            "--arch",
            arch,
            "--classes",
            classes,
            "--img-size",
            str(img_size),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
        ]
    )

    # Output format
    format_idx = _ask_choice("Formato de saida:", ["csv", "json", "jsonl"], default=0)
    output_format = ["csv", "json", "jsonl"][format_idx]
    if output_format != "csv":
        args.extend(["--output-format", output_format])

    # Resume capability
    if _ask_yes_no("Habilitar resume/checkpoint?", False):
        args.append("--resume")
        checkpoint_file = _ask_string("Checkpoint file", "batch_inference_checkpoint.json")
        checkpoint_interval = _ask_int("Checkpoint interval (batches)", 100)
        args.extend(["--checkpoint-file", checkpoint_file, "--checkpoint-interval", str(checkpoint_interval)])

    # Advanced options
    if _ask_yes_no("Configurar opcoes avancadas?", False):
        if _ask_yes_no("Normalizacao customizada?", False):
            mean = _ask_string("Mean (ex: 0.485,0.456,0.406)", "0.485,0.456,0.406")
            std = _ask_string("Std (ex: 0.229,0.224,0.225)", "0.229,0.224,0.225")
            args.extend(["--mean", mean, "--std", std])

        if _ask_yes_no("Usar AMP?", False):
            args.append("--amp")

        num_workers = _ask_int("Num workers", HP.NUM_WORKERS)
        if num_workers != HP.NUM_WORKERS:
            args.extend(["--num-workers", str(num_workers)])

        prefetch_factor = _ask_int("Prefetch factor", HP.PREFETCH_FACTOR)
        if prefetch_factor != HP.PREFETCH_FACTOR:
            args.extend(["--prefetch-factor", str(prefetch_factor)])

        if _ask_yes_no("Persistent workers?", HP.PERSISTENT_WORKERS):
            args.append("--persistent-workers")
        else:
            args.append("--no-persistent-workers")

    args.extend(_ask_extra_args())
    return WizardCommand("Inferencia em lote (batch)", _build_cli_command("batch-inference", args))

def _wizard_augment() -> WizardCommand:
    """
    Prompt the user for augmentation inputs and produce a configured augmentation CLI command.
    
    Collects source directory, output directory, number of augmentations, optional config and extra CLI arguments, and returns a WizardCommand that runs the "augment" subcommand with the collected options.
    
    Returns:
        WizardCommand: A command labeled "Augmentacao de dados" configured to invoke the "augment" CLI subcommand with the prompted arguments.
    """
    args = _ask_config_args()
    source_dir = _ask_string("Diretorio de origem")
    output_dir = _ask_string("Diretorio de saida", f"{source_dir}_aug")
    num_aug = _ask_int("Numero de augmentations", 1)
    args.extend(
        [
        "--source-dir",
        source_dir,
        "--output-dir",
        output_dir,
        "--num-augmentations",
        str(num_aug),
        ]
    )
    args.extend(_ask_extra_args())
    return WizardCommand("Augmentacao de dados", _build_cli_command("augment", args))

def _wizard_label_density() -> WizardCommand:
    return WizardCommand("Rotulagem de densidade", _build_cli_command("label-density", []))

def _wizard_label_patches() -> WizardCommand:
    return WizardCommand("Rotulagem de patches", _build_cli_command("label-patches", []))

def _wizard_eda() -> WizardCommand:
    args: list[str] = []
    csv_dir = _ask_optional("CSV dir (train.csv/test.csv, opcional)")
    if csv_dir:
        args.extend(["--csv-dir", csv_dir])
    dicom_dir = _ask_optional("DICOM dir (train_images, opcional)")
    if dicom_dir:
        args.extend(["--dicom-dir", dicom_dir])
    png_dir = _ask_optional("PNG dir (256x256, opcional)")
    if png_dir:
        args.extend(["--png-dir", png_dir])
    output_dir = _ask_optional("Output dir (opcional)")
    if output_dir:
        args.extend(["--output-dir", output_dir])
    args.extend(_ask_extra_args())
    return WizardCommand("EDA cancer", _build_cli_command("eda-cancer", args))

def _wizard_eval_export() -> WizardCommand:
    args = _ask_config_args()
    run_list = _ask_optional("Runs (separados por virgula, opcional)")
    if run_list:
        runs = [r.strip() for r in run_list.split(",") if r.strip()]
        for run in runs:
            args.extend(["--run", run])
    args.extend(_ask_extra_args())
    return WizardCommand("Eval-export", _build_cli_command("eval-export", args))

def _wizard_data_audit() -> WizardCommand:
    args = _ask_config_args()
    archive = _ask_string("Diretorio archive", "archive")
    csv_path = _ask_string("CSV de rotulos", "classificacao.csv")
    manifest = _ask_string("Manifest JSON", "data_manifest.json")
    audit_csv = _ask_string(
        "CSV de auditoria",
        "outputs/embeddings_resnet50/data_audit.csv",
    )
    log_path = _ask_string("Log do artigo", "Article/assets/data_qc.log")
    args.extend(
        [
            "--archive",
            archive,
            "--csv",
            csv_path,
            "--manifest",
            manifest,
            "--audit-csv",
            audit_csv,
            "--log",
            log_path,
        ]
    )
    args.extend(_ask_extra_args())
    return WizardCommand("Auditoria de dados", _build_cli_command("data-audit", args))

def _wizard_report_pack() -> WizardCommand:
    run_list = _ask_string("Runs (separados por virgula, ex: outputs/run/results_1)")
    runs = [r.strip() for r in run_list.split(",") if r.strip()]
    assets_dir = _ask_string("Assets dir", "Article/assets")
    tex_path = _ask_string("Arquivo LaTeX (opcional)", "")
    gradcam_limit = _ask_int("Grad-CAM limit", 4)

    args: list[str] = []
    for run in runs:
        args.extend(["--run", run])
    args.extend(["--assets-dir", assets_dir, "--gradcam-limit", str(gradcam_limit)])
    if tex_path:
        args.extend(["--tex", tex_path])
    args.extend(_ask_extra_args())
    return WizardCommand("Report-pack", _build_cli_command("report-pack", args))
