from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

from mammography.config import HP
from mammography.data.csv_loader import DATASET_PRESETS


@dataclass
class WizardCommand:
    label: str
    argv: list[str]


def _ask_choice(title: str, options: Sequence[str], default: int = 0) -> int:
    print(f"\n{title}")
    for i, option in enumerate(options):
        print(f"  [{i}] {option}")
    while True:
        raw = input(f"Escolha [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit():
            value = int(raw)
            if 0 <= value < len(options):
                return value
        print(f"Opcao invalida. Informe um numero entre 0 e {len(options) - 1}.")


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes", "s", "sim"}:
            return True
        if raw in {"n", "no", "nao"}:
            return False
        print("Resposta invalida. Use y/n.")


def _ask_string(prompt: str, default: str | None = None) -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit():
            return int(raw)
        print("Informe um numero inteiro valido.")


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("Informe um numero valido.")


def _ask_optional(prompt: str) -> str | None:
    raw = input(f"{prompt} (enter para pular): ").strip()
    return raw or None


def _ask_config_args() -> list[str]:
    path = _ask_optional("Config YAML/JSON")
    if path:
        return ["--config", path]
    return []


def _ask_extra_args() -> list[str]:
    raw = input("Args extras (opcional, ex: --foo 1 --bar): ").strip()
    return shlex.split(raw) if raw else []


def _run_command(cmd: WizardCommand, dry_run: bool) -> int:
    print("\nResumo do comando:")
    print(f"  {cmd.label}")
    print("  " + " ".join(shlex.quote(part) for part in cmd.argv))
    if not _ask_yes_no("Executar agora?"):
        print("Comando cancelado.")
        return 0
    if dry_run:
        print("Dry-run habilitado; comando nao sera executado.")
        return 0
    completed = subprocess.run(cmd.argv, check=False)
    return completed.returncode


def _dataset_prompt() -> tuple[list[str], str | None, str | None]:
    presets = list(DATASET_PRESETS.keys())
    options = presets + ["custom"]
    idx = _ask_choice("Formato do dataset:", options, default=0)
    selection = options[idx]
    args: list[str] = []
    csv_path = None
    dicom_root = None
    if selection != "custom":
        args.extend(["--dataset", selection])
    if selection == "archive":
        csv_path = _ask_string("CSV (classificacao.csv)", "classificacao.csv")
        dicom_root = _ask_string("Diretorio DICOM", "archive")
    elif selection in {"mamografias", "patches_completo"}:
        csv_path = _ask_string("Diretorio com featureS.txt", selection)
    else:
        csv_path = _ask_string("CSV ou diretorio com featureS.txt")
        dicom_root = _ask_string("Diretorio DICOM (opcional)", "")
        if not dicom_root:
            dicom_root = None

    if csv_path:
        args.extend(["--csv", csv_path])
    if dicom_root:
        args.extend(["--dicom-root", dicom_root])
    return args, csv_path, dicom_root


def _build_cli_command(subcommand: str, args: Iterable[str]) -> list[str]:
    return [sys.executable, "-m", "mammography.cli", subcommand, *args]


def _wizard_train() -> WizardCommand:
    args = _ask_config_args()
    dataset_args, _, _ = _dataset_prompt()
    args.extend(dataset_args)

    arch_idx = _ask_choice("Arquitetura:", ["efficientnet_b0", "resnet50"], default=0)
    arch = ["efficientnet_b0", "resnet50"][arch_idx]
    class_idx = _ask_choice("Classes:", ["density (A-D)", "binary (AB vs CD)", "multiclass (A-D)"], default=0)
    classes = ["density", "binary", "multiclass"][class_idx]

    outdir = _ask_string("Outdir", "outputs/mammo_efficientnetb0_density")
    epochs = _ask_int("Epocas", HP.EPOCHS)
    batch_size = _ask_int("Batch size", HP.BATCH_SIZE)
    img_size = _ask_int("Img size", HP.IMG_SIZE)
    device = _ask_string("Device (auto/cuda/mps/cpu)", HP.DEVICE)
    cache_mode = _ask_string("Cache mode", HP.CACHE_MODE)
    pretrained = _ask_yes_no("Usar pesos pretrained?", True)

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
        include_class_5 = _ask_yes_no("Incluir classe 5?", False)
        if include_class_5:
            args.append("--include-class-5")

        cache_dir = _ask_optional("Cache dir")
        if cache_dir:
            args.extend(["--cache-dir", cache_dir])

        embeddings_dir = _ask_optional("Embeddings Stage 1 (diretorio)")
        if embeddings_dir:
            args.extend(["--embeddings-dir", embeddings_dir])

        seed = _ask_int("Seed", HP.SEED)
        args.extend(["--seed", str(seed)])
        val_frac = _ask_float("Val frac", HP.VAL_FRAC)
        args.extend(["--val-frac", str(val_frac)])

        if not _ask_yes_no("Garantir todas as classes no val?", True):
            args.append("--no-split-ensure-all-classes")
        split_max_tries = _ask_int("Split max tries", 200)
        args.extend(["--split-max-tries", str(split_max_tries)])

        if not _ask_yes_no("Ativar augmentations?", True):
            args.append("--no-augment")
        else:
            if _ask_yes_no("Flip vertical?", False):
                args.append("--augment-vertical")
            if _ask_yes_no("Color jitter?", False):
                args.append("--augment-color")
            rotation_deg = _ask_float("Rotacao (graus)", 5.0)
            if rotation_deg != 5.0:
                args.extend(["--augment-rotation-deg", str(rotation_deg)])

        if _ask_yes_no("Normalizacao customizada?", False):
            mean = _ask_string("Mean (ex: 0.485,0.456,0.406)", "0.485,0.456,0.406")
            std = _ask_string("Std (ex: 0.229,0.224,0.225)", "0.229,0.224,0.225")
            args.extend(["--mean", mean, "--std", std])

        class_choice = _ask_choice("Class weights:", ["auto", "none", "manual"], default=0)
        if class_choice == 2:
            weights = _ask_string("Pesos (ex: 1.0,0.8,1.2,1.0)")
            args.extend(["--class-weights", weights])
        else:
            args.extend(["--class-weights", ["auto", "none"][class_choice]])
        if class_choice == 0:
            alpha = _ask_float("Class weights alpha", 1.0)
            if alpha != 1.0:
                args.extend(["--class-weights-alpha", str(alpha)])

        if _ask_yes_no("Usar sampler ponderado?", True):
            args.append("--sampler-weighted")
            sampler_alpha = _ask_float("Sampler alpha", 1.0)
            if sampler_alpha != 1.0:
                args.extend(["--sampler-alpha", str(sampler_alpha)])

        if _ask_yes_no("Treinar todo o backbone?", False):
            args.append("--train-backbone")
        else:
            if _ask_yes_no("Descongelar ultimo bloco?", True):
                args.append("--unfreeze-last-block")
            else:
                args.append("--no-unfreeze-last-block")

        lr = _ask_float("Learning rate", HP.LR)
        backbone_lr = _ask_float("Backbone LR", HP.BACKBONE_LR)
        weight_decay = _ask_float("Weight decay", 1e-4)
        args.extend(["--lr", str(lr), "--backbone-lr", str(backbone_lr), "--weight-decay", str(weight_decay)])

        warmup_epochs = _ask_int("Warmup epochs", HP.WARMUP_EPOCHS)
        args.extend(["--warmup-epochs", str(warmup_epochs)])

        early_stop_patience = _ask_int("Early stop patience (0 = off)", HP.EARLY_STOP_PATIENCE)
        args.extend(["--early-stop-patience", str(early_stop_patience)])
        early_stop_min_delta = _ask_float("Early stop min delta", HP.EARLY_STOP_MIN_DELTA)
        args.extend(["--early-stop-min-delta", str(early_stop_min_delta)])

        scheduler_idx = _ask_choice("Scheduler:", ["auto", "none", "plateau", "cosine", "step"], default=0)
        scheduler = ["auto", "none", "plateau", "cosine", "step"][scheduler_idx]
        if scheduler != "auto":
            args.extend(["--scheduler", scheduler])
        if scheduler == "plateau":
            lr_patience = _ask_int("LR reduce patience", HP.LR_REDUCE_PATIENCE)
            lr_factor = _ask_float("LR reduce factor", HP.LR_REDUCE_FACTOR)
            lr_min = _ask_float("LR reduce min lr", HP.LR_REDUCE_MIN_LR)
            lr_cooldown = _ask_int("LR reduce cooldown", HP.LR_REDUCE_COOLDOWN)
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
            min_lr = _ask_float("Scheduler min lr", HP.LR_REDUCE_MIN_LR)
            args.extend(["--scheduler-min-lr", str(min_lr)])
        elif scheduler == "step":
            step_size = _ask_int("Scheduler step size", 5)
            gamma = _ask_float("Scheduler gamma", 0.5)
            args.extend(["--scheduler-step-size", str(step_size), "--scheduler-gamma", str(gamma)])

        num_workers = _ask_int("Num workers", HP.NUM_WORKERS)
        args.extend(["--num-workers", str(num_workers)])
        prefetch_factor = _ask_int("Prefetch factor (0 = off)", HP.PREFETCH_FACTOR)
        args.extend(["--prefetch-factor", str(prefetch_factor)])
        if not _ask_yes_no("Persistent workers?", HP.PERSISTENT_WORKERS):
            args.append("--no-persistent-workers")
        if not _ask_yes_no("Loader heuristics?", HP.LOADER_HEURISTICS):
            args.append("--no-loader-heuristics")

        if _ask_yes_no("Habilitar AMP?", False):
            args.append("--amp")
        if _ask_yes_no("Ativar torch.compile?", False):
            args.append("--torch-compile")
        if _ask_yes_no("Ativar fused optimizer?", False):
            args.append("--fused-optim")
        if _ask_yes_no("Salvar predicoes de validacao?", False):
            args.append("--save-val-preds")

        if _ask_yes_no("Salvar Grad-CAM?", False):
            args.append("--gradcam")
            gradcam_limit = _ask_int("Grad-CAM limit", 4)
            args.extend(["--gradcam-limit", str(gradcam_limit)])
        if _ask_yes_no("Exportar embeddings de validacao?", False):
            args.append("--export-val-embeddings")

        subset = _ask_int("Subset (0 = desativado)", 0)
        if subset > 0:
            args.extend(["--subset", str(subset)])

        if _ask_yes_no("Habilitar profiler?", False):
            args.append("--profile")
            profile_dir = _ask_string("Profile dir", "outputs/profiler")
            args.extend(["--profile-dir", profile_dir])

        if _ask_yes_no("Deterministic?", HP.DETERMINISTIC):
            args.append("--deterministic")
        if not _ask_yes_no("Permitir TF32?", HP.ALLOW_TF32):
            args.append("--no-allow-tf32")

        log_level = _ask_string("Log level", HP.LOG_LEVEL)
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
    args = _ask_config_args()
    dataset_args, _, _ = _dataset_prompt()
    args.extend(dataset_args)

    arch_idx = _ask_choice("Arquitetura:", ["resnet50", "efficientnet_b0"], default=0)
    arch = ["resnet50", "efficientnet_b0"][arch_idx]
    class_idx = _ask_choice("Classes:", ["multiclass (A-D)", "binary (AB vs CD)", "density (A-D)"], default=0)
    classes = ["multiclass", "binary", "density"][class_idx]

    outdir = _ask_string("Outdir", "outputs/features")
    batch_size = _ask_int("Batch size", 32)
    img_size = _ask_int("Img size", HP.IMG_SIZE)
    device = _ask_string("Device (auto/cuda/mps/cpu)", HP.DEVICE)
    cache_mode = _ask_string("Cache mode", HP.CACHE_MODE)
    pretrained = _ask_yes_no("Usar pesos pretrained?", True)

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
        include_class_5 = _ask_yes_no("Incluir classe 5?", False)
        if include_class_5:
            args.append("--include-class-5")

        seed = _ask_int("Seed", HP.SEED)
        args.extend(["--seed", str(seed)])
        if _ask_yes_no("Deterministic?", HP.DETERMINISTIC):
            args.append("--deterministic")
        if not _ask_yes_no("Permitir TF32?", HP.ALLOW_TF32):
            args.append("--no-allow-tf32")

        num_workers = _ask_int("Num workers", HP.NUM_WORKERS)
        args.extend(["--num-workers", str(num_workers)])
        prefetch_factor = _ask_int("Prefetch factor (0 = off)", HP.PREFETCH_FACTOR)
        args.extend(["--prefetch-factor", str(prefetch_factor)])
        if not _ask_yes_no("Persistent workers?", HP.PERSISTENT_WORKERS):
            args.append("--no-persistent-workers")
        if not _ask_yes_no("Loader heuristics?", HP.LOADER_HEURISTICS):
            args.append("--no-loader-heuristics")

        if _ask_yes_no("Normalizacao customizada?", False):
            mean = _ask_string("Mean (ex: 0.485,0.456,0.406)", "0.485,0.456,0.406")
            std = _ask_string("Std (ex: 0.229,0.224,0.225)", "0.229,0.224,0.225")
            args.extend(["--mean", mean, "--std", std])

        layer_name = _ask_string("Layer name (avgpool)", "avgpool")
        if layer_name and layer_name != "avgpool":
            args.extend(["--layer-name", layer_name])

        if _ask_yes_no("Usar AMP?", False):
            args.append("--amp")
        if _ask_yes_no("Salvar joined.csv?", False):
            args.append("--save-csv")

        if _ask_yes_no("Executar PCA?", True):
            args.append("--pca")
        if _ask_yes_no("Executar t-SNE?", True):
            args.append("--tsne")
        if _ask_yes_no("Executar UMAP?", True):
            args.append("--umap")
        if _ask_yes_no("Executar clustering?", False):
            args.append("--cluster")
            cluster_k = _ask_int("K fixo (0 = auto)", 0)
            if cluster_k >= 2:
                args.extend(["--cluster-k", str(cluster_k)])
        sample_grid = _ask_int("Sample grid", 16)
        args.extend(["--sample-grid", str(sample_grid)])

        log_level = _ask_string("Log level", HP.LOG_LEVEL)
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

    if _ask_yes_no("Gerar report completo?", True):
        args.append("--report")
    else:
        if _ask_yes_no("Gerar PCA?", True):
            args.append("--pca")
        if _ask_yes_no("Gerar t-SNE?", True):
            args.append("--tsne")
        if _ask_yes_no("Gerar t-SNE 3D?", False):
            args.append("--tsne-3d")
        if _ask_yes_no("Gerar UMAP?", False):
            args.append("--umap")
        if _ask_yes_no("Comparar embeddings?", False):
            args.append("--compare-embeddings")
        if _ask_yes_no("Gerar heatmap?", False):
            args.append("--heatmap")
        if _ask_yes_no("Gerar confusion matrix?", False):
            args.append("--confusion-matrix")
        if _ask_yes_no("Gerar scatter matrix?", False):
            args.append("--scatter-matrix")
        if _ask_yes_no("Gerar distribuicoes?", False):
            args.append("--distribution")
        if _ask_yes_no("Gerar separacao de classes?", False):
            args.append("--class-separation")
        if _ask_yes_no("Gerar curvas de aprendizado?", False):
            args.append("--learning-curves")
        if _ask_yes_no("Rotulos binarios (low/high)?", False):
            args.append("--binary")

    args.extend(_ask_extra_args())
    return WizardCommand("Visualizacao", _build_cli_command("visualize", args))


def _wizard_inference() -> WizardCommand:
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


def _wizard_augment() -> WizardCommand:
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
    return WizardCommand("EDA cancer", _build_cli_command("eda-cancer", []))


def _wizard_eval_export() -> WizardCommand:
    args = _ask_config_args()
    run_list = _ask_optional("Runs (separados por virgula, opcional)")
    if run_list:
        runs = [r.strip() for r in run_list.split(",") if r.strip()]
        for run in runs:
            args.extend(["--run", run])
    args.extend(_ask_extra_args())
    return WizardCommand("Eval-export", _build_cli_command("eval-export", args))


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


def run_wizard(dry_run: bool = False) -> int:
    print("\n=== MAMMOGRAPHY WIZARD ===")
    options = [
        "Treinamento de densidade (Stage 2)",
        "Treino rapido (Stage 2)",
        "Extracao de embeddings (Stage 1)",
        "Visualizacao de embeddings",
        "Inferencia",
        "Augmentacao de dados",
        "Rotular densidade (GUI)",
        "Rotular patches (GUI)",
        "EDA cancer",
        "Eval-export (checklist)",
        "Report-pack (Article)",
        "Sair",
    ]
    choice = _ask_choice("Selecione a tarefa:", options, default=0)
    if choice == 0:
        cmd = _wizard_train()
    elif choice == 1:
        cmd = _wizard_quick_train()
    elif choice == 2:
        cmd = _wizard_embed()
    elif choice == 3:
        cmd = _wizard_visualize()
    elif choice == 4:
        cmd = _wizard_inference()
    elif choice == 5:
        cmd = _wizard_augment()
    elif choice == 6:
        cmd = _wizard_label_density()
    elif choice == 7:
        cmd = _wizard_label_patches()
    elif choice == 8:
        cmd = _wizard_eda()
    elif choice == 9:
        cmd = _wizard_eval_export()
    elif choice == 10:
        cmd = _wizard_report_pack()
    else:
        print("Saindo do wizard.")
        return 0

    return _run_command(cmd, dry_run)
