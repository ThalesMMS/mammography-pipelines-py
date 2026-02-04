#!/usr/bin/env python3
#
# train.py
# mammography-pipelines
#
# Trains EfficientNetB0/ResNet50 density classifiers with optional caching, AMP, Grad-CAM, and evaluation exports.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Train EfficientNetB0/ResNet50 for breast density with optional caches and AMP."""
import os
import signal
import sys
import hashlib
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Sequence, Optional, Tuple, List, Dict
import torch
from torch import profiler
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


from pydantic import ValidationError

from mammography.config import HP, TrainConfig
from mammography.utils.common import (
    seed_everything,
    resolve_device,
    configure_runtime,
    setup_logging,
    increment_path,
    parse_float_list,
    get_reproducibility_info,
)
from mammography.data.csv_loader import (
    load_dataset_dataframe,
    resolve_dataset_cache_mode,
    DATASET_PRESETS,
    resolve_paths_from_preset,
)
from mammography.data.dataset import MammoDensityDataset, mammo_collate, load_embedding_store
from mammography.data.splits import create_splits, create_three_way_split, load_splits_from_csvs
from mammography.io.dicom import is_dicom_path
from mammography.models.nets import build_model
from mammography.training.engine import (
    train_one_epoch,
    validate,
    save_metrics_figure,
    extract_embeddings,
    plot_history,
    save_predictions,
    save_atomic,
)

def parse_args(argv: Optional[Sequence[str]] = None):
    """Define and parse CLI arguments for the density training script."""
    parser = argparse.ArgumentParser(description="Treinamento Mammography (EfficientNetB0/ResNet50)")

    # Data
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), help="Atalho para datasets conhecidos (archive/mamografias/patches_completo)")
    parser.add_argument("--csv", required=False, help="CSV, diretório com featureS.txt ou caminho manual")
    parser.add_argument("--dicom-root", help="Root for DICOMs (usado com classificacao.csv)")
    parser.add_argument("--include-class-5", action="store_true", help="Mantém amostras com classificação 5 ao carregar classificacao.csv")
    parser.add_argument("--auto-detect", action=argparse.BooleanOptionalAction, default=True, help="Detecta automaticamente formato do dataset a partir da estrutura de diretórios (default: True)")
    parser.add_argument("--outdir", default="outputs/run", help="Output directory")
    parser.add_argument("--cache-mode", default=HP.CACHE_MODE, choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"])
    parser.add_argument("--cache-dir", help="Cache dir")
    parser.add_argument("--embeddings-dir", help="Diretorio com features.npy + metadata.csv (embeddings)")
    parser.add_argument("--mean", help="Media de normalizacao (ex: 0.485,0.456,0.406)")
    parser.add_argument("--std", help="Std de normalizacao (ex: 0.229,0.224,0.225)")
    parser.add_argument("--log-level", default=HP.LOG_LEVEL, choices=["critical","error","warning","info","debug"])
    parser.add_argument("--subset", type=int, default=0, help="Limita o número de amostras")

    # View-specific training
    parser.add_argument("--view-specific-training", action="store_true", help="Treina modelos separados para cada view (CC/MLO)")
    parser.add_argument("--view-column", default="view", help="Nome da coluna que contem a view no CSV (default: view)")
    parser.add_argument("--views", default="CC,MLO", help="Lista de views separadas por virgula para treinar (default: CC,MLO)")
    parser.add_argument("--ensemble-method", default="none", choices=["none", "average", "weighted", "max"], help="Metodo de ensemble para combinar predicoes de views (none/average/weighted/max)")

    # Model / task
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--classes", default="density", choices=["density", "binary", "multiclass"], help="density/multiclass = BI-RADS 1..4, binary = A/B vs C/D")
    parser.add_argument("--task", dest="classes", choices=["density", "binary", "multiclass"], help="Alias para --classes")
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
    parser.add_argument("--backbone-lr", type=float, default=HP.BACKBONE_LR, help="Learning rate para o backbone (cabeça usa --lr)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--val-frac", type=float, default=HP.VAL_FRAC)
    parser.add_argument("--split-mode", choices=["random", "patient", "preset"], default="random", help="Modo de split: random (aleatorio), patient (por paciente), preset (CSVs pre-definidos)")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Fracao para split de teste (0.0 = sem teste)")
    parser.add_argument("--train-csv", help="CSV pre-definido para treino (usado com --split-mode=preset)")
    parser.add_argument("--val-csv", help="CSV pre-definido para validacao (usado com --split-mode=preset)")
    parser.add_argument("--test-csv", help="CSV pre-definido para teste (usado com --split-mode=preset)")
    parser.add_argument("--split-ensure-all-classes", action=argparse.BooleanOptionalAction, default=True, help="Garante todas as classes no split de validacao")
    parser.add_argument("--split-max-tries", type=int, default=200, help="Tentativas maximas para split estratificado")
    parser.add_argument("--num-workers", type=int, default=HP.NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=HP.PREFETCH_FACTOR)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=HP.PERSISTENT_WORKERS)
    parser.add_argument("--loader-heuristics", action=argparse.BooleanOptionalAction, default=HP.LOADER_HEURISTICS)
    parser.add_argument("--amp", action="store_true", help="Habilita autocast + GradScaler em CUDA/MPS")
    parser.add_argument("--class-weights", default=HP.CLASS_WEIGHTS, help="auto/none ou lista (ex: 1.0,0.8,1.2,1.0)")
    parser.add_argument("--class-weights-alpha", type=float, default=1.0, help="Expoente para class_weights auto")
    parser.add_argument("--sampler-weighted", action="store_true", default=HP.SAMPLER_WEIGHTED)
    parser.add_argument("--sampler-alpha", type=float, default=1.0, help="Expoente para sampler ponderado")
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=HP.TRAIN_BACKBONE)
    parser.add_argument("--unfreeze-last-block", action=argparse.BooleanOptionalAction, default=HP.UNFREEZE_LAST_BLOCK)
    parser.add_argument("--warmup-epochs", type=int, default=HP.WARMUP_EPOCHS)
    parser.add_argument("--deterministic", action="store_true", default=HP.DETERMINISTIC)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=HP.ALLOW_TF32)
    parser.add_argument("--fused-optim", action="store_true", default=HP.FUSED_OPTIM, help="Ativa AdamW(fused=True) em CUDA quando disponível")
    parser.add_argument("--torch-compile", action="store_true", default=HP.TORCH_COMPILE, help="Otimiza o modelo com torch.compile quando suportado")
    parser.add_argument("--lr-reduce-patience", type=int, default=HP.LR_REDUCE_PATIENCE)
    parser.add_argument("--lr-reduce-factor", type=float, default=HP.LR_REDUCE_FACTOR)
    parser.add_argument("--lr-reduce-min-lr", type=float, default=HP.LR_REDUCE_MIN_LR)
    parser.add_argument("--lr-reduce-cooldown", type=int, default=HP.LR_REDUCE_COOLDOWN)
    parser.add_argument("--scheduler", choices=["auto", "none", "plateau", "cosine", "step"], default="auto")
    parser.add_argument("--scheduler-min-lr", type=float, default=HP.LR_REDUCE_MIN_LR)
    parser.add_argument("--scheduler-step-size", type=int, default=5)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--profile", action="store_true", help="Habilita torch.profiler no primeiro epoch")
    parser.add_argument("--profile-dir", default=os.path.join("outputs", "profiler"), help="Destino dos traces do profiler")
    parser.add_argument("--early-stop-patience", type=int, default=HP.EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=HP.EARLY_STOP_MIN_DELTA)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=HP.TRAIN_AUGMENT, help="Ativa augmentations no treino")
    parser.add_argument("--augment-vertical", action="store_true", help="Habilita flip vertical aleatorio")
    parser.add_argument("--augment-color", action="store_true", help="Habilita color jitter (brightness/contrast)")
    parser.add_argument("--augment-rotation-deg", type=float, default=5.0, help="Amplitude da rotacao aleatoria")
    parser.add_argument("--resume-from", help="Checkpoint para retomar treino (ex: checkpoints/last.pt)")
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
    parser.add_argument("--export-figures", help="Formatos de exportacao para figuras publicacao (ex: png,pdf,svg)")

    return parser.parse_args(argv)

def get_label_mapper(mode):
    """Return a mapper function to collapse classes when running binary experiments."""
    mode = (mode or "density").lower()
    if mode == "binary":
        # 1,2 -> 0 (Low); 3,4 -> 1 (High)
        def mapper(y):
            if y in [1, 2]: return 0
            if y in [3, 4]: return 1
            return y - 1 # Fallback
        return mapper
    return None # Default 1..4 -> 0..3


def get_file_hash(path: str) -> str:
    if not path or not os.path.exists(path):
        return "unknown"
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except Exception:
        return "unknown"


class GracefulKiller:
    def __init__(self) -> None:
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame) -> None:
        print("\n[INFO] Sinal de interrupcao recebido! Terminando epoch atual antes de sair...")
        self.kill_now = True


def _find_resume_checkpoint(outdir: str) -> Optional[Path]:
    base = Path(outdir)
    direct = base / "checkpoint.pt"
    if direct.exists():
        return direct
    if base.exists():
        candidates = []
        for results_path in base.glob("results*"):
            if not results_path.is_dir():
                continue
            ckpt = results_path / "checkpoint.pt"
            if ckpt.exists():
                candidates.append(ckpt)
        if candidates:
            try:
                return max(candidates, key=lambda p: p.stat().st_mtime)
            except Exception:
                return candidates[0]
    return None


def _parse_class_weights(raw: str, num_classes: int):
    text = str(raw or "").strip()
    if not text:
        return "none"
    lower = text.lower()
    if lower in {"none", "auto"}:
        return lower
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        weights = parse_float_list(text, expected_len=num_classes, name="class_weights")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return weights or "none"


def _normalize_patient_id(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _unique_paths(paths: Sequence[str]) -> list[str]:
    uniq = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        uniq.append(path)
    return uniq


def _collect_dicom_paths(df: pd.DataFrame) -> list[str]:
    if "image_path" not in df.columns:
        return []
    paths = [str(p) for p in df["image_path"].dropna().tolist()]
    dicom_paths = [p for p in paths if is_dicom_path(p)]
    return _unique_paths(dicom_paths)


def _preflight_dicom_headers(
    dicom_paths: Sequence[str],
    seed: int,
    logger: logging.Logger,
    max_samples: int = 1000,
    error_threshold: float = 0.01,
) -> None:
    if not dicom_paths:
        logger.info("Preflight DICOM: nenhum caminho DICOM detectado; pulando.")
        return
    total = len(dicom_paths)
    sample_paths = list(dicom_paths)
    sampled = False
    if total > max_samples:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(total, size=max_samples, replace=False)
        sample_paths = [dicom_paths[i] for i in idxs]
        sampled = True
    errors = []
    for path in sample_paths:
        try:
            pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            errors.append(path)
    err_count = len(errors)
    err_rate = err_count / max(1, len(sample_paths))
    if sampled:
        logger.info(
            "Preflight DICOM: %d/%d erros (%.2f%%) na amostra %d/%d.",
            err_count,
            len(sample_paths),
            err_rate * 100.0,
            len(sample_paths),
            total,
        )
    else:
        logger.info(
            "Preflight DICOM: %d/%d erros (%.2f%%).",
            err_count,
            len(sample_paths),
            err_rate * 100.0,
        )
    if err_rate > error_threshold:
        examples = errors[:3]
        raise RuntimeError(
            "CRITICO: preflight DICOM falhou. "
            f"{err_count}/{len(sample_paths)} ({err_rate:.2%}) com erro (ex: {examples})."
        )


def _select_patient_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("patient_id", "PatientID"):
        if col in df.columns:
            series = df[col].apply(_normalize_patient_id)
            if series.notna().any():
                return col
    return None


def _missing_id_examples(df: pd.DataFrame) -> list[str]:
    for col in ("accession", "image_path"):
        if col in df.columns:
            return df[col].astype(str).head(3).tolist()
    return []


def _patient_ids_from_column(df: pd.DataFrame, column: str, split_label: str) -> set[str]:
    series = df[column].apply(_normalize_patient_id)
    missing_mask = series.isna()
    if missing_mask.any():
        examples = _missing_id_examples(df[missing_mask])
        raise RuntimeError(
            f"CRITICO: coluna '{column}' com {missing_mask.sum()} valores vazios "
            f"no split {split_label}; nao e possivel validar vazamento (ex: {examples})."
        )
    return set(series.tolist())


def _patient_ids_from_dicom(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[set[str], set[str]]:
    if "image_path" not in train_df.columns or "image_path" not in val_df.columns:
        raise RuntimeError("CRITICO: coluna image_path ausente; nao e possivel validar vazamento.")
    all_paths = _unique_paths(
        [str(p) for p in pd.concat([train_df["image_path"], val_df["image_path"]], axis=0).dropna().tolist()]
    )
    if not all_paths:
        raise RuntimeError("CRITICO: nenhum image_path valido para validar vazamento.")
    non_dicom = [p for p in all_paths if not is_dicom_path(p)]
    if non_dicom:
        examples = non_dicom[:3]
        logger.warning(
            "Amostras nao-DICOM detectadas (%d/%d); pulando verificacao de vazamento por PatientID (ex: %s).",
            len(non_dicom), len(all_paths), examples
        )
        return set(), set()
    logger.info("Lendo cabecalhos DICOM para obter PatientID (%d arquivos).", len(all_paths))
    patient_by_path: dict[str, str] = {}
    read_errors = []
    missing_ids = []
    for path in all_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as exc:
            read_errors.append((path, exc))
            continue
        patient_id = _normalize_patient_id(getattr(ds, "PatientID", None))
        if not patient_id:
            missing_ids.append(path)
            continue
        patient_by_path[path] = patient_id
    if read_errors:
        err_rate = len(read_errors) / max(1, len(all_paths))
        examples = [p for p, _ in read_errors[:3]]
        raise RuntimeError(
            "CRITICO: falha ao ler cabecalhos DICOM para obter PatientID. "
            f"{len(read_errors)}/{len(all_paths)} ({err_rate:.2%}) com erro (ex: {examples})."
        )
    if missing_ids:
        examples = missing_ids[:3]
        raise RuntimeError(
            "CRITICO: PatientID ausente em arquivos DICOM; "
            f"{len(missing_ids)} amostras sem PatientID (ex: {examples})."
        )

    def _ids_for(df: pd.DataFrame) -> set[str]:
        ids = set()
        for path in df["image_path"].tolist():
            patient_id = patient_by_path.get(str(path))
            if patient_id:
                ids.add(patient_id)
        return ids

    return _ids_for(train_df), _ids_for(val_df)


def _assert_no_patient_leakage(train_patients: set[str], val_patients: set[str]) -> None:
    intersec = train_patients.intersection(val_patients)
    if intersec:
        sample = sorted(intersec)[:3]
        raise RuntimeError(
            "CRITICO: vazamento de dados detectado! "
            f"{len(intersec)} pacientes aparecem em train e val (ex: {sample})."
        )


class ExperimentTracker:
    def __init__(self, kind: str, module: Any, run: Any):
        self.kind = kind
        self.module = module
        self.run = run

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        if self.kind == "wandb":
            self.module.log(metrics, step=step)
        elif self.kind == "mlflow":
            self.module.log_metrics(metrics, step=step)
        elif self.kind == "local":
            self.run.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
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
        if self.kind == "wandb":
            self.run.finish()
        elif self.kind == "mlflow":
            self.module.end_run()
        elif self.kind == "local":
            self.run.finish()


def _sanitize_tracking_params(payload: dict[str, Any]) -> dict[str, Any]:
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
) -> Optional['ExperimentTracker']:
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


def _parse_export_formats(raw: Optional[str]) -> list[str]:
    """Parse comma-separated export formats from CLI argument."""
    if not raw:
        return []
    formats = [fmt.strip().lower() for fmt in raw.split(",")]
    valid = {"png", "pdf", "svg"}
    invalid = [fmt for fmt in formats if fmt not in valid]
    if invalid:
        raise SystemExit(f"Formatos invalidos: {invalid}. Use: png, pdf, svg")
    return formats


def _export_figure_multi_format(
    fig_func,
    output_path: Path,
    formats: list[str],
    *args,
    **kwargs
) -> None:
    """Export a matplotlib figure to multiple formats for publication."""
    if not formats:
        # Default behavior: save as PNG
        fig_func(*args, **kwargs)
        return

    # Create figures directory
    figures_dir = output_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Get base name from original path
    base_name = output_path.stem

    # Save in each requested format
    for fmt in formats:
        export_path = figures_dir / f"{base_name}.{fmt}"
        # Call the original function with modified path
        modified_kwargs = kwargs.copy()
        if "out_path" in kwargs:
            modified_kwargs["out_path"] = str(export_path)
        else:
            # For functions that take path as last positional arg
            args_list = list(args)
            if args_list:
                args_list[-1] = str(export_path)
                args = tuple(args_list)

        # Try to save the figure
        try:
            if fig_func.__name__ == "plot_history":
                # Special handling for plot_history
                _plot_history_format(args[0], export_path.parent, export_path.name, fmt)
            elif fig_func.__name__ == "save_metrics_figure":
                # Special handling for save_metrics_figure
                _save_metrics_figure_format(args[0], str(export_path))
        except Exception as exc:
            import logging
            logger = logging.getLogger("mammography")
            logger.warning(f"Falha ao exportar figura {export_path}: {exc}")


def _plot_history_format(history: List[Dict[str, Any]], outdir: Path, base_name: str, fmt: str) -> None:
    """Save training history plot in specified format."""
    import matplotlib.pyplot as plt
    if not history:
        return
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(df["epoch"], df["train_loss"], label="train")
        ax[0].plot(df["epoch"], df["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(df["epoch"], df["train_acc"], label="train")
        ax[1].plot(df["epoch"], df["val_acc"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        fig.tight_layout()

        # Determine output path
        if base_name.endswith(f".{fmt}"):
            out_path = outdir / base_name
        else:
            out_path = outdir / f"{base_name.replace('.png', '')}.{fmt}"

        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass


def _save_metrics_figure_format(metrics: Dict[str, Any], out_path: str) -> None:
    """Save metrics figure in specified format (wrapper for save_metrics_figure)."""
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_metrics_figure(metrics, out_path)


def _sort_top_k(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda e: (e["score"], e["epoch"]), reverse=True)


def _clean_top_k(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        try:
            score = float(entry.get("score", 0.0))
            epoch = int(entry.get("epoch", 0))
            path = Path(entry.get("path", ""))
        except Exception:
            continue
        if not path or not path.exists():
            continue
        cleaned.append({"score": score, "epoch": epoch, "path": str(path)})
    return _sort_top_k(cleaned)


def _update_top_k(
    top_k: list[dict[str, Any]],
    score: float,
    epoch: int,
    model_state: dict[str, Any],
    out_dir: Path,
    k: int,
    metric_name: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if k <= 0:
        return top_k, None
    if len(top_k) >= k:
        lowest = min(top_k, key=lambda e: (e["score"], e["epoch"]))
        if (score, epoch) <= (lowest["score"], lowest["epoch"]):
            return top_k, None

    out_dir.mkdir(parents=True, exist_ok=True)
    metric_tag = metric_name.replace("/", "_")
    filename = f"model_epoch{epoch:03d}_{metric_tag}{score:.4f}.pt"
    path = out_dir / filename
    save_atomic(model_state, path)
    entry = {"score": float(score), "epoch": int(epoch), "path": str(path)}
    top_k.append(entry)

    sorted_entries = _sort_top_k(top_k)
    keep = sorted_entries[:k]
    for item in sorted_entries[k:]:
        item_path = Path(item["path"])
        if item_path.exists():
            try:
                item_path.unlink()
            except Exception:
                pass
    return keep, entry

def resolve_loader_runtime(args, device: torch.device):
    """Heuristic knobs to keep DataLoader stable across CPU, CUDA, and MPS."""
    num_workers = args.num_workers
    prefetch = args.prefetch_factor if args.prefetch_factor and args.prefetch_factor > 0 else None
    persistent = args.persistent_workers
    if not args.loader_heuristics:
        return num_workers, prefetch, persistent
    if device.type == "mps":
        return 0, prefetch, False
    if device.type == "cpu":
        return max(0, min(num_workers, os.cpu_count() or 0)), prefetch, persistent
    return num_workers, prefetch, persistent

def _resolve_head_module(model: torch.nn.Module, arch: str) -> Optional[torch.nn.Module]:
    if arch == "resnet50":
        if hasattr(model, "classifier"):
            return model.classifier
        if hasattr(model, "fc"):
            return model.fc
    if arch == "efficientnet_b0":
        if hasattr(model, "classifier"):
            return model.classifier
    return None

def _resolve_backbone_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.backbone if hasattr(model, "backbone") else model

def freeze_backbone(model: torch.nn.Module, arch: str) -> None:
    """Disable backbone gradients so only the classifier head keeps training."""
    for p in model.parameters():
        p.requires_grad = False
    head = _resolve_head_module(model, arch)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

def unfreeze_last_block(model: torch.nn.Module, arch: str) -> None:
    backbone = _resolve_backbone_module(model)
    if arch == "resnet50" and hasattr(backbone, "layer4"):
        for p in backbone.layer4.parameters():
            p.requires_grad = True
    if arch == "efficientnet_b0" and hasattr(backbone, "features"):
        features = backbone.features
        if len(features) > 0:
            for p in features[-1].parameters():
                p.requires_grad = True

def build_param_groups(model: torch.nn.Module, arch: str, lr_head: float, lr_backbone: float) -> list[dict]:
    """Create optimizer parameter groups to support differential LR for backbone vs head."""
    head = _resolve_head_module(model, arch)
    if head is None:
        if arch == "resnet50":
            head_params = [p for n, p in model.named_parameters() if n.startswith("fc") and p.requires_grad]
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc") and p.requires_grad]
        else:
            head_params = [p for n, p in model.named_parameters() if n.startswith("classifier") and p.requires_grad]
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
    else:
        head_ids = {id(p) for p in head.parameters()}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    params = []
    if head_params:
        params.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        params.append({"params": backbone_params, "lr": lr_backbone})
    return params

def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    checkpoint_path = _find_resume_checkpoint(args.outdir)
    if checkpoint_path and not args.resume_from:
        print(f"[INFO] Checkpoint encontrado em {checkpoint_path}. Retomando automaticamente.")
        args.resume_from = str(checkpoint_path)

    csv_path, dicom_root = resolve_paths_from_preset(args.csv, args.dataset, args.dicom_root)
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

    seed_everything(args.seed, deterministic=args.deterministic)

    # Parse export formats
    try:
        export_formats = _parse_export_formats(args.export_figures)
    except SystemExit as exc:
        raise exc

    outdir_root = Path(increment_path(args.outdir))
    outdir_root.mkdir(parents=True, exist_ok=True)
    results_base = outdir_root / "results"
    outdir_path = Path(increment_path(str(results_base)))
    outdir_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = outdir_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Create figures directory if export formats are specified
    if export_formats:
        figures_dir = outdir_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

    outdir = str(outdir_path)
    logger = setup_logging(outdir, args.log_level)
    logger.info(f"Args: {args}")
    logger.info("Resultados serao gravados em: %s", outdir_path)
    if export_formats:
        logger.info("Figuras serao exportadas em formatos: %s", ", ".join(export_formats))

    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)
    killer = GracefulKiller()

    # Load Data
    df = load_dataset_dataframe(
        csv_path,
        dicom_root,
        exclude_class_5=not args.include_class_5,
        dataset=args.dataset,
        auto_detect=args.auto_detect,
    )
    logger.info(f"Loaded {len(df)} samples from dataset '{args.dataset or 'custom'}'.")
    df = df[df["professional_label"].notna()]
    logger.info(f"Valid samples (with label): {len(df)}")
    if args.subset and args.subset > 0:
        subset_count = min(args.subset, len(df))
        df = df.sample(n=subset_count, random_state=args.seed).reset_index(drop=True)
        logger.info(f"Subset selecionado: {subset_count} amostras.")

    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    num_classes = 2 if args.classes == "binary" else 4

    # Route to appropriate split function based on configuration
    test_df = None
    if args.split_mode == "preset":
        # Load pre-defined splits from CSV files
        if not args.train_csv or not args.val_csv:
            raise SystemExit("Modo 'preset' requer --train-csv e --val-csv.")
        train_df, val_df, test_df = load_splits_from_csvs(
            args.train_csv,
            args.val_csv,
            args.test_csv,
        )
    elif args.test_frac > 0:
        # Create three-way split (train/val/test)
        train_df, val_df, test_df = create_three_way_split(
            df,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            num_classes=num_classes,
            ensure_all_splits_have_all_classes=args.split_ensure_all_classes,
            max_tries=args.split_max_tries,
        )
    else:
        # Default two-way split (train/val) for backward compatibility
        train_df, val_df = create_splits(
            df,
            val_frac=args.val_frac,
            seed=args.seed,
            num_classes=num_classes,
            ensure_val_has_all_classes=args.split_ensure_all_classes,
            max_tries=args.split_max_tries,
        )

    patient_col = _select_patient_id_column(df)
    if patient_col:
        logger.info("Leakage check: usando coluna '%s' para patient_id.", patient_col)
        dicom_paths = _collect_dicom_paths(df)
        _preflight_dicom_headers(dicom_paths, args.seed, logger)
        train_patients = _patient_ids_from_column(train_df, patient_col, "train")
        val_patients = _patient_ids_from_column(val_df, patient_col, "val")
        test_patients = _patient_ids_from_column(test_df, patient_col, "test") if test_df is not None else set()
    else:
        logger.info("Leakage check: usando PatientID dos cabecalhos DICOM.")
        if test_df is not None:
            # Extended DICOM patient ID extraction for 3-way split
            all_paths = _unique_paths(
                [str(p) for p in pd.concat([train_df["image_path"], val_df["image_path"], test_df["image_path"]], axis=0).dropna().tolist()]
            )
            if not all_paths:
                raise RuntimeError("CRITICO: nenhum image_path valido para validar vazamento.")
            non_dicom = [p for p in all_paths if not is_dicom_path(p)]
            if non_dicom:
                examples = non_dicom[:3]
                raise RuntimeError(
                    "CRITICO: amostras nao-DICOM sem patient_id; nao e possivel validar vazamento "
                    f"(ex: {examples})."
                )
            logger.info("Lendo cabecalhos DICOM para obter PatientID (%d arquivos).", len(all_paths))
            patient_by_path: dict[str, str] = {}
            read_errors = []
            missing_ids = []
            for path in all_paths:
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                except Exception as exc:
                    read_errors.append((path, exc))
                    continue
                patient_id = _normalize_patient_id(getattr(ds, "PatientID", None))
                if not patient_id:
                    missing_ids.append(path)
                    continue
                patient_by_path[path] = patient_id
            if read_errors:
                err_rate = len(read_errors) / max(1, len(all_paths))
                examples = [p for p, _ in read_errors[:3]]
                raise RuntimeError(
                    "CRITICO: falha ao ler cabecalhos DICOM para obter PatientID. "
                    f"{len(read_errors)}/{len(all_paths)} ({err_rate:.2%}) com erro (ex: {examples})."
                )
            if missing_ids:
                examples = missing_ids[:3]
                raise RuntimeError(
                    "CRITICO: PatientID ausente em arquivos DICOM; "
                    f"{len(missing_ids)} amostras sem PatientID (ex: {examples})."
                )
            def _ids_for(df_split: pd.DataFrame) -> set[str]:
                ids = set()
                for path in df_split["image_path"].tolist():
                    patient_id = patient_by_path.get(str(path))
                    if patient_id:
                        ids.add(patient_id)
                return ids
            train_patients = _ids_for(train_df)
            val_patients = _ids_for(val_df)
            test_patients = _ids_for(test_df)
        else:
            train_patients, val_patients = _patient_ids_from_dicom(train_df, val_df, logger)
            test_patients = set()
    _assert_no_patient_leakage(train_patients, val_patients)
    if test_patients:
        # Also check for train/test and val/test leakage
        train_test_intersec = train_patients.intersection(test_patients)
        if train_test_intersec:
            sample = sorted(train_test_intersec)[:3]
            raise RuntimeError(
                "CRITICO: vazamento de dados detectado! "
                f"{len(train_test_intersec)} pacientes aparecem em train e test (ex: {sample})."
            )
        val_test_intersec = val_patients.intersection(test_patients)
        if val_test_intersec:
            sample = sorted(val_test_intersec)[:3]
            raise RuntimeError(
                "CRITICO: vazamento de dados detectado! "
                f"{len(val_test_intersec)} pacientes aparecem em val e test (ex: {sample})."
            )

    train_rows = train_df.to_dict("records")
    val_rows = val_df.to_dict("records")
    test_rows = test_df.to_dict("records") if test_df is not None else None

    # View-specific training setup
    views_to_train = []
    if args.view_specific_training:
        if args.view_column not in train_df.columns:
            raise SystemExit(f"View column '{args.view_column}' nao encontrada no DataFrame.")
        views_to_train = [v.strip() for v in args.views.split(",") if v.strip()]
        if not views_to_train:
            raise SystemExit("Nenhuma view especificada em --views.")
        logger.info("View-specific training habilitado para views: %s", views_to_train)
    else:
        views_to_train = [None]  # Single training run without view filtering

    embedding_store = None
    if args.embeddings_dir:
        if args.arch not in ["efficientnet_b0", "resnet50"]:
            raise SystemExit("Fusao de embeddings so esta disponivel para efficientnet_b0 e resnet50.")
        embedding_store = load_embedding_store(args.embeddings_dir)
        logger.info("Embeddings carregadas de %s", args.embeddings_dir)

        def _count_missing(rows):
            return sum(1 for r in rows if embedding_store.lookup(r) is None)  # type: ignore[union-attr]

        missing_train = _count_missing(train_rows)
        missing_val = _count_missing(val_rows)
        missing_test = _count_missing(test_rows) if test_rows else 0
        if missing_train or missing_val or missing_test:
            if test_rows:
                logger.warning(
                    "Embeddings ausentes: train=%s, val=%s, test=%s (total train=%s, val=%s, test=%s).",
                    missing_train,
                    missing_val,
                    missing_test,
                    len(train_rows),
                    len(val_rows),
                    len(test_rows),
                )
            else:
                logger.warning(
                    "Embeddings ausentes: train=%s, val=%s (total train=%s, val=%s).",
                    missing_train,
                    missing_val,
                    len(train_rows),
                    len(val_rows),
                )

    cache_dir = args.cache_dir or str(outdir_root / "cache")

    mapper = get_label_mapper(args.classes)

    # Loop over views (or single iteration if view-specific training is disabled)
    for current_view in views_to_train:
        # Filter data by view if view-specific training is enabled
        if current_view is not None:
            view_train_df = train_df[train_df[args.view_column] == current_view].reset_index(drop=True)
            view_val_df = val_df[val_df[args.view_column] == current_view].reset_index(drop=True)
            view_train_rows = view_train_df.to_dict("records")
            view_val_rows = view_val_df.to_dict("records")
            logger.info(
                "Training view-specific model for view '%s': %d train samples, %d val samples",
                current_view,
                len(view_train_rows),
                len(view_val_rows),
            )
            # Create view-specific output directory
            view_outdir_path = outdir_path.parent / f"{outdir_path.name}_{current_view}"
            view_outdir_path.mkdir(parents=True, exist_ok=True)
            view_metrics_dir = view_outdir_path / "metrics"
            view_metrics_dir.mkdir(parents=True, exist_ok=True)
            # Create figures directory for view-specific training if export formats are specified
            if export_formats:
                view_figures_dir = view_outdir_path / "figures"
                view_figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use full dataset (no view filtering)
            view_train_rows = train_rows
            view_val_rows = val_rows
            view_outdir_path = outdir_path
            view_metrics_dir = metrics_dir

        cache_mode_train = resolve_dataset_cache_mode(args.cache_mode, view_train_rows)
        cache_mode_val = resolve_dataset_cache_mode(args.cache_mode, view_val_rows)

        train_ds = MammoDensityDataset(
            view_train_rows,
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
            embedding_store=embedding_store,
            mean=mean,
            std=std,
        )
        val_ds = MammoDensityDataset(
            view_val_rows,
            args.img_size,
            train=False,
            augment=False,
            cache_mode=cache_mode_val,
            cache_dir=cache_dir,
            split_name="val",
            label_mapper=mapper,
            embedding_store=embedding_store,
            mean=mean,
            std=std,
        )

        # Sampler/loader settings
        def _map_row_label(row):
            raw = row.get("professional_label")
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                return None
            val = int(raw)
            if mapper:
                return mapper(val)
            return val - 1

        sampler = None
        if args.sampler_weighted:
            mapped = [_map_row_label(r) for r in view_train_rows if _map_row_label(r) is not None]
            counts = np.bincount(np.array(mapped, dtype=int), minlength=num_classes)
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
            class_weights = torch.tensor(inv ** float(args.sampler_alpha), dtype=torch.float)
            sample_weights = torch.tensor(
                [class_weights[_map_row_label(r)] if _map_row_label(r) is not None else 0.0 for r in view_train_rows],
                dtype=torch.float,
            )
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        nw, prefetch, persistent = resolve_loader_runtime(args, device)
        dl_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": nw,
            "persistent_workers": bool(persistent and nw > 0),
            "pin_memory": device.type == "cuda",
            "collate_fn": mammo_collate,
        }
        if prefetch is not None and nw > 0:
            dl_kwargs["prefetch_factor"] = prefetch

        train_loader = DataLoader(
            train_ds,
            shuffle=sampler is None,
            sampler=sampler,
            **dl_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            **dl_kwargs,
        )

        # Model
        model = build_model(
            args.arch,
            num_classes=num_classes,
            train_backbone=args.train_backbone,
            unfreeze_last_block=args.unfreeze_last_block,
            pretrained=args.pretrained,
            extra_feature_dim=embedding_store.feature_dim if embedding_store else 0,
        ).to(device)

        if args.torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
                logger.info("torch.compile ativado.")
            except Exception as exc:
                logger.warning("torch.compile falhou; seguindo sem compile: %s", exc)

        optim_params = build_param_groups(model, args.arch, args.lr, args.backbone_lr)
        optim_kwargs = {"weight_decay": args.weight_decay}
        if args.fused_optim and device.type == "cuda":
            optim_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(optim_params if optim_params else model.parameters(), **optim_kwargs)
        weights = None
        class_weights = _parse_class_weights(args.class_weights, num_classes)
        if class_weights == "auto":
            mapped = [_map_row_label(r) for r in view_train_rows if _map_row_label(r) is not None]
            counts = np.bincount(np.array(mapped, dtype=int), minlength=num_classes)
            if np.any(counts == 0):
                logger.warning("class_weights=auto ignorado: alguma classe sem amostras.")
            else:
                total = float(np.sum(counts))
                weights_np = (total / counts) ** float(args.class_weights_alpha)
                weights_np = weights_np / np.mean(weights_np)
                weights = torch.tensor(weights_np, dtype=torch.float32).to(device)
                logger.info("Pesos de classe (auto): %s", weights_np.tolist())
        elif isinstance(class_weights, list):
            weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        scaler = GradScaler() if args.amp and device.type == "cuda" else None
        scheduler = None
        scheduler_mode = args.scheduler
        if scheduler_mode == "auto":
            scheduler_mode = "plateau" if args.lr_reduce_patience and args.lr_reduce_patience > 0 else "none"
        if scheduler_mode == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.lr_reduce_factor,
                patience=args.lr_reduce_patience,
                min_lr=args.lr_reduce_min_lr,
                cooldown=args.lr_reduce_cooldown,
            )
        elif scheduler_mode == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs),
                eta_min=args.scheduler_min_lr,
            )
        elif scheduler_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, args.scheduler_step_size),
                gamma=args.scheduler_gamma,
            )

        best_acc = -1.0
        best_metric = -1.0
        best_epoch = -1
        patience_ctr = 0
        history = []
        top_k_metric = "macro_f1"
        top_k_limit = 3
        top_k_dir = view_outdir_path / "top_k"
        top_k: list[dict[str, Any]] = []
        gradcam_dir = view_outdir_path / "gradcam" if args.gradcam else None
        summary_path = view_outdir_path / "summary.json"
        # Add view suffix to checkpoint filename for view-specific training
        checkpoint_name = f"checkpoint_{current_view.lower()}.pt" if current_view else "checkpoint.pt"
        checkpoint_path = view_outdir_path / checkpoint_name
        resume_epoch = 1
        resume_path: Optional[Path] = None
        if args.resume_from:
            resume_path = Path(args.resume_from)
            if not resume_path.exists():
                raise SystemExit(f"Checkpoint nao encontrado: {resume_path}")
            logger.info("Retomando treino de %s", resume_path)
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                if "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                else:
                    logger.warning("Checkpoint sem optimizer_state; retomando apenas pesos.")
                if scheduler is not None:
                    sched_state = checkpoint.get("scheduler_state")
                    if sched_state is not None:
                        scheduler.load_state_dict(sched_state)
                elif checkpoint.get("scheduler_state") is not None:
                    logger.warning("Checkpoint contem scheduler_state, mas nenhum scheduler ativo.")
                if scaler is not None:
                    scaler_state = checkpoint.get("scaler_state")
                    if scaler_state is not None:
                        scaler.load_state_dict(scaler_state)
                elif checkpoint.get("scaler_state") is not None:
                    logger.warning("Checkpoint contem scaler_state, mas AMP desativado.")
                best_acc = float(checkpoint.get("best_acc", best_acc))
                best_metric = float(checkpoint.get("best_metric", checkpoint.get("best_acc", best_metric)))
                best_epoch = int(checkpoint.get("best_epoch", best_epoch))
                patience_ctr = int(checkpoint.get("patience_ctr", patience_ctr))
                resume_epoch = int(checkpoint.get("epoch", 0)) + 1
                top_k = _clean_top_k(checkpoint.get("top_k", []))
            else:
                model.load_state_dict(checkpoint)
                logger.warning("Checkpoint sem metadados; retomando apenas pesos do modelo.")
            if resume_epoch < 1:
                resume_epoch = 1
        if resume_epoch > args.epochs:
            logger.warning(
                "Checkpoint epoch %s >= epochs solicitadas (%s); nenhum epoch sera executado.",
                resume_epoch - 1,
                args.epochs,
            )
        repro_info = get_reproducibility_info()
        summary_payload = {
            "run_id": view_outdir_path.name,
            "seed": args.seed,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "reproducibility": repro_info,
            "arch": args.arch,
            "classes": args.classes,
            "dataset": args.dataset,
            "csv": str(csv_path),
            "dicom_root": str(dicom_root) if dicom_root else None,
            "data_hashes": {
                "csv": get_file_hash(str(csv_path)),
            },
            "embeddings_dir": args.embeddings_dir,
            "outdir": str(view_outdir_path),
            "outdir_root": str(outdir_root),
            "subset": args.subset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "pretrained": args.pretrained,
            # Optimizer hyperparameters
            "optimizer": "AdamW",
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "fused_optim": args.fused_optim,
            # Scheduler hyperparameters
            "scheduler": scheduler_mode,
            "lr_reduce_patience": args.lr_reduce_patience,
            "lr_reduce_factor": args.lr_reduce_factor,
            "lr_reduce_min_lr": args.lr_reduce_min_lr,
            "lr_reduce_cooldown": args.lr_reduce_cooldown,
            "scheduler_min_lr": args.scheduler_min_lr,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            # Augmentation settings
            "augment": args.augment,
            "augment_vertical": args.augment_vertical,
            "augment_color": args.augment_color,
            "augment_rotation_deg": args.augment_rotation_deg,
            # Model training settings
            "train_backbone": args.train_backbone,
            "unfreeze_last_block": args.unfreeze_last_block,
            "warmup_epochs": args.warmup_epochs,
            # Runtime settings
            "amp": args.amp,
            "deterministic": args.deterministic,
            "allow_tf32": args.allow_tf32,
            "torch_compile": args.torch_compile,
            # Data loading settings
            "cache_mode": args.cache_mode,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
            "loader_heuristics": args.loader_heuristics,
            # Normalization
            "mean": mean,
            "std": std,
            # Split settings
            "split_mode": args.split_mode,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "split_ensure_all_classes": args.split_ensure_all_classes,
            "split_max_tries": args.split_max_tries,
            # Class weighting
            "class_weights": args.class_weights,
            "class_weights_alpha": args.class_weights_alpha,
            "sampler_weighted": args.sampler_weighted,
            "sampler_alpha": args.sampler_alpha,
            # Early stopping
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            # Resume settings
            "resume_from": str(resume_path) if resume_path else None,
            "resume_epoch": resume_epoch,
            # Tracker settings
            "tracker": args.tracker,
            "tracker_project": args.tracker_project,
            "tracker_run_name": args.tracker_run_name,
            "tracker_uri": args.tracker_uri,
            # Evaluation settings
            "top_k_metric": top_k_metric,
            "top_k_limit": top_k_limit,
            # View-specific training
            "view": current_view,
            "view_specific_training": args.view_specific_training,
            "view_column": args.view_column if args.view_specific_training else None,
            "views": args.views if args.view_specific_training else None,
            "ensemble_method": args.ensemble_method if args.view_specific_training else None,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        tracker = _init_tracker(args, summary_payload, view_outdir_path, logger)

        logger.info("Starting training%s...", f" for view '{current_view}'" if current_view else "")
        for epoch in range(resume_epoch, args.epochs + 1):
            if args.warmup_epochs and epoch <= args.warmup_epochs:
                freeze_backbone(model, args.arch)
            elif args.train_backbone:
                for p in model.parameters():
                    p.requires_grad = True
                if args.unfreeze_last_block:
                    unfreeze_last_block(model, args.arch)

            prof_ctx = None
            if args.profile and epoch == resume_epoch:
                Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
                activities = [profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(profiler.ProfilerActivity.CUDA)
                prof_ctx = profiler.profile(activities=activities, record_shapes=False)
                prof_ctx.__enter__()

            t_loss, t_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_fn=loss_fn,
                scaler=scaler,
                amp_enabled=args.amp and device.type in ["cuda", "mps"],
            )

            val_metrics, pred_rows = validate(
                model,
                val_loader,
                device,
                amp_enabled=args.amp and device.type in ["cuda", "mps"],
                loss_fn=loss_fn,
                collect_preds=args.save_val_preds,
                gradcam=args.gradcam,
                gradcam_dir=gradcam_dir,
                gradcam_limit=args.gradcam_limit,
            )
            v_acc = float(val_metrics.get("acc", 0.0) or 0.0)
            v_loss = float(val_metrics.get("loss", 0.0) or 0.0)
            v_macro_f1 = float(val_metrics.get("macro_f1", 0.0) or 0.0)
            if not np.isfinite(v_macro_f1):
                v_macro_f1 = 0.0

            if prof_ctx is not None:
                try:
                    prof_ctx.__exit__(None, None, None)
                    trace_path = Path(args.profile_dir) / "trace.json"
                    prof_ctx.export_chrome_trace(trace_path)
                    logger.info(f"Trace salvo em {trace_path}")
                except Exception as exc:
                    logger.warning("Falha ao salvar trace do profiler: %s", exc)

            logger.info(
                "Epoch %s/%s%s | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f Macro-F1: %.4f",
                epoch,
                args.epochs,
                f" (view={current_view})" if current_view else "",
                t_loss,
                t_acc,
                v_loss,
                v_acc,
                v_macro_f1,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": t_loss,
                    "train_acc": t_acc,
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "val_auc": val_metrics.get("auc_ovr", 0.0) or 0.0,
                    "val_kappa": val_metrics.get("kappa_quadratic", 0.0) or 0.0,
                    "val_macro_f1": val_metrics.get("macro_f1", 0.0) or 0.0,
                    "val_bal_acc": val_metrics.get("bal_acc", 0.0) or 0.0,
                    "val_bal_acc_adj": val_metrics.get("bal_acc_adj", 0.0) or 0.0,
                }
            )
            plot_history(history, view_outdir_path)

            # Export figures in publication formats if requested
            if export_formats:
                for fmt in export_formats:
                    _plot_history_format(history, view_outdir_path / "figures", "train_history", fmt)

            if tracker:
                # Comprehensive per-epoch metric logging
                metrics_payload = {
                    # Training metrics
                    "train_loss": float(t_loss),
                    "train_acc": float(t_acc),
                    # Validation metrics
                    "val_loss": float(v_loss),
                    "val_acc": float(v_acc),
                    "val_f1": float(v_macro_f1),
                    "val_kappa": float(val_metrics.get("kappa_quadratic", 0.0) or 0.0),
                    "balanced_acc": float(val_metrics.get("bal_acc", 0.0) or 0.0),
                    # Additional metrics (kept for backward compatibility)
                    "val_macro_f1": float(v_macro_f1),
                    "val_bal_acc": float(val_metrics.get("bal_acc", 0.0) or 0.0),
                    "val_bal_acc_adj": float(val_metrics.get("bal_acc_adj", 0.0) or 0.0),
                }
                # Add AUC if available
                auc_val = val_metrics.get("auc_ovr", None)
                if auc_val is not None and np.isfinite(auc_val):
                    metrics_payload["val_auc"] = float(auc_val)
                    metrics_payload["val_auc_ovr"] = float(auc_val)  # Keep for backward compatibility
                tracker.log_metrics(metrics_payload, step=epoch)

            # Persist latest metrics
            val_metrics_filename = f"val_metrics_{current_view.lower()}.json" if current_view else "val_metrics.json"
            val_metrics_fig_name = f"val_metrics_{current_view.lower()}.png" if current_view else "val_metrics.png"
            with open(metrics_dir / val_metrics_filename, "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2, default=str)
            save_metrics_figure(val_metrics, str(metrics_dir / val_metrics_fig_name))

            # Export metrics figure in publication formats if requested
            if export_formats:
                base_name = f"val_metrics_{current_view.lower()}" if current_view else "val_metrics"
                for fmt in export_formats:
                    _save_metrics_figure_format(val_metrics, str(view_outdir_path / "figures" / f"{base_name}.{fmt}"))

            if args.save_val_preds:
                save_predictions(pred_rows, view_outdir_path)

            if v_acc > best_acc:
                best_acc = v_acc

            top_k, new_entry = _update_top_k(
                top_k,
                v_macro_f1,
                epoch,
                model.state_dict(),
                top_k_dir,
                top_k_limit,
                top_k_metric,
            )
            if new_entry and tracker:
                tracker.log_artifact(Path(new_entry["path"]), name=f"topk_{top_k_metric}_epoch{epoch}")

            improved = v_macro_f1 > best_metric + args.early_stop_min_delta
            if improved:
                best_metric = v_macro_f1
                best_epoch = epoch
                # Add view suffix to best model filename for view-specific training
                best_model_name = f"best_model_{current_view.lower()}.pt" if current_view else "best_model.pt"
                best_model_path = view_outdir_path / best_model_name
                save_atomic(model.state_dict(), best_model_path)
                if tracker:
                    tracker.log_artifact(best_model_path, name=f"best_{top_k_metric}_epoch{epoch}")
                best_metrics_filename = f"best_metrics_{current_view.lower()}.json" if current_view else "best_metrics.json"
                best_metrics_fig_name = f"best_metrics_{current_view.lower()}.png" if current_view else "best_metrics.png"
                with open(metrics_dir / best_metrics_filename, "w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2, default=str)
                save_metrics_figure(val_metrics, str(metrics_dir / best_metrics_fig_name))

                # Export best metrics figure in publication formats if requested
                if export_formats:
                    base_name = f"best_metrics_{current_view.lower()}" if current_view else "best_metrics"
                    for fmt in export_formats:
                        _save_metrics_figure_format(val_metrics, str(view_outdir_path / "figures" / f"{base_name}.{fmt}"))

                patience_ctr = 0
            else:
                patience_ctr += 1

            if scheduler_mode == "plateau" and scheduler is not None:
                scheduler.step(v_macro_f1)
            elif scheduler is not None:
                scheduler.step()

            checkpoint_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_acc": float(best_acc),
                "best_metric": float(best_metric),
                "best_metric_name": top_k_metric,
                "best_epoch": int(best_epoch),
                "patience_ctr": int(patience_ctr),
                "top_k": top_k,
                "mean": mean,
                "std": std,
            }
            save_atomic(checkpoint_state, checkpoint_path)

            if killer.kill_now:
                print("[INFO] Estado salvo com sucesso. Encerrando execucao.")
                sys.exit(0)

            if args.early_stop_patience and patience_ctr >= args.early_stop_patience:
                logger.info(f"Early stopping ativado após {patience_ctr} épocas sem melhoria.")
                break

        logger.info(
            "Done%s. Best %s: %.4f (epoch %s) | Best Acc: %.4f",
            f" for view '{current_view}'" if current_view else "",
            top_k_metric,
            best_metric,
            best_epoch,
            best_acc,
        )
        summary_payload.update(
            {
                "best_acc": float(best_acc),
                "best_metric": float(best_metric),
                "best_metric_name": top_k_metric,
                "best_epoch": int(best_epoch),
                "top_k": top_k,
                "finished_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        if args.export_val_embeddings:
            logger.info("Extraindo embeddings do conjunto de validação%s...", f" (view={current_view})" if current_view else "")
            # Load best model with view suffix for view-specific training
            best_model_name = f"best_model_{current_view.lower()}.pt" if current_view else "best_model.pt"
            best_path = view_outdir_path / best_model_name
            if best_path.exists():
                state = torch.load(best_path, map_location=device)
                model.load_state_dict(state, strict=False)
            feats, metas = extract_embeddings(model, val_loader, device, amp_enabled=args.amp and device.type in ["cuda", "mps"])
            np.save(view_outdir_path / "embeddings_val.npy", feats)
            pd.DataFrame(metas).to_csv(view_outdir_path / "embeddings_val.csv", index=False)

        if tracker:
            tracker.finish()

    # Ensemble evaluation after all view-specific training completes
    if args.view_specific_training and args.ensemble_method != "none":
        logger.info("Evaluando ensemble com metodo '%s'...", args.ensemble_method)

        # Import EnsemblePredictor
        from mammography.models.cancer_models import EnsemblePredictor

        # Load best models for each view
        view_models = {}
        for view in views_to_train:
            view_outdir_path = outdir_path.parent / f"{outdir_path.name}_{view}"
            best_model_name = f"best_model_{view.lower()}.pt"
            best_model_path = view_outdir_path / best_model_name

            if not best_model_path.exists():
                logger.warning("Modelo para view '%s' nao encontrado em %s; pulando ensemble.", view, best_model_path)
                continue

            # Create a new model instance and load weights
            view_model = build_model(
                args.arch,
                num_classes=num_classes,
                train_backbone=args.train_backbone,
                unfreeze_last_block=args.unfreeze_last_block,
                pretrained=False,
                extra_feature_dim=embedding_store.feature_dim if embedding_store else 0,
            ).to(device)

            state = torch.load(best_model_path, map_location=device)
            view_model.load_state_dict(state, strict=False)
            view_model.eval()
            view_models[view] = view_model
            logger.info("Carregado modelo para view '%s' de %s", view, best_model_path)

        if len(view_models) < 2:
            logger.warning("Ensemble requer pelo menos 2 modelos treinados; encontrado %d. Pulando ensemble.", len(view_models))
        else:
            # Create ensemble predictor
            ensemble = EnsemblePredictor(view_models, method=args.ensemble_method)

            # Create validation dataset with all views (no filtering)
            full_val_ds = MammoDensityDataset(
                val_rows,
                args.img_size,
                train=False,
                augment=False,
                cache_mode=cache_mode_val,
                cache_dir=cache_dir,
                split_name="val_ensemble",
                label_mapper=mapper,
                embedding_store=embedding_store,
                mean=mean,
                std=std,
            )

            full_val_loader = DataLoader(
                full_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=nw,
                persistent_workers=bool(persistent and nw > 0),
                pin_memory=device.type == "cuda",
                collate_fn=mammo_collate,
            )
            if prefetch is not None and nw > 0:
                full_val_loader.prefetch_factor = prefetch

            # Group validation samples by patient for ensemble evaluation
            logger.info("Agrupando amostras de validacao por paciente para ensemble...")
            patient_samples = {}
            for row in val_rows:
                patient_id = row.get("patient_id") or row.get("PatientID") or row.get("accession")
                if not patient_id:
                    continue
                view = row.get(args.view_column)
                if not view or view not in view_models:
                    continue

                if patient_id not in patient_samples:
                    patient_samples[patient_id] = {}
                patient_samples[patient_id][view] = row

            # Filter to patients with all required views
            required_views = set(view_models.keys())
            complete_patients = {
                pid: samples for pid, samples in patient_samples.items()
                if set(samples.keys()) == required_views
            }

            logger.info(
                "Encontrados %d pacientes com todas as views (%s) para ensemble de %d pacientes totais.",
                len(complete_patients), ", ".join(sorted(required_views)), len(patient_samples)
            )

            if len(complete_patients) == 0:
                logger.warning("Nenhum paciente com todas as views; nao e possivel avaliar ensemble.")
            else:
                # Evaluate ensemble on patients with all views
                all_y = []
                all_p = []
                all_prob = []

                with torch.no_grad():
                    for patient_id, view_samples in tqdm(complete_patients.items(), desc="Ensemble Val", leave=False):
                        # Get predictions from each view
                        view_probs = {}
                        label = None

                        for view, row in view_samples.items():
                            # Get sample label (should be same across views for same patient)
                            if label is None:
                                raw_label = row.get("professional_label")
                                if raw_label is not None and not (isinstance(raw_label, float) and np.isnan(raw_label)):
                                    label_val = int(raw_label)
                                    label = mapper(label_val) if mapper else (label_val - 1)

                            # Create single-sample dataset to load image
                            sample_ds = MammoDensityDataset(
                                [row],
                                args.img_size,
                                train=False,
                                augment=False,
                                cache_mode=cache_mode_val,
                                cache_dir=cache_dir,
                                split_name="val_ensemble",
                                label_mapper=mapper,
                                embedding_store=embedding_store,
                                mean=mean,
                                std=std,
                            )

                            sample_data = sample_ds[0]
                            if sample_data is None:
                                continue

                            if len(sample_data) == 4:
                                x, y, meta, extra_feat = sample_data
                            else:
                                x, y, meta = sample_data
                                extra_feat = None

                            x = x.unsqueeze(0).to(device=device, memory_format=torch.channels_last)
                            extra_tensor = None
                            if extra_feat is not None:
                                extra_tensor = extra_feat.unsqueeze(0).to(device=device)

                            # Get prediction from view-specific model
                            with torch.autocast(device_type=device.type, enabled=args.amp and device.type in ["cuda", "mps"]):
                                logits = view_models[view](x, extra_tensor)

                            probs = torch.softmax(logits, dim=1)
                            view_probs[view] = probs

                        # Combine predictions using ensemble
                        if len(view_probs) == len(required_views) and label is not None:
                            ensemble_probs = ensemble.predict(view_probs)
                            ensemble_pred = torch.argmax(ensemble_probs, dim=1)

                            all_y.append(label)
                            all_p.append(ensemble_pred.cpu().item())
                            all_prob.append(ensemble_probs.cpu().numpy())

            if len(all_y) > 0:
                from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score

                acc = accuracy_score(all_y, all_p)
                kappa = cohen_kappa_score(all_y, all_p, weights="quadratic")

                try:
                    all_prob_concat = np.concatenate(all_prob, axis=0)
                    auc = roc_auc_score(all_y, all_prob_concat, multi_class="ovr", average="macro")
                except Exception:
                    auc = 0.0

                ensemble_metrics = {
                    "acc": float(acc),
                    "kappa_quadratic": float(kappa),
                    "auc_ovr": float(auc),
                    "num_samples": len(all_y),
                }

                logger.info(
                    "Ensemble Metrics | Acc: %.4f | Kappa: %.4f | AUC: %.4f (n=%d)",
                    acc, kappa, auc, len(all_y)
                )

                # Save ensemble metrics
                ensemble_metrics_dir = outdir_path / "metrics"
                ensemble_metrics_dir.mkdir(parents=True, exist_ok=True)
                with open(ensemble_metrics_dir / "ensemble_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(ensemble_metrics, f, indent=2, default=str)

                logger.info("Metricas de ensemble salvas em %s", ensemble_metrics_dir / "ensemble_metrics.json")
            else:
                logger.warning("Nenhuma predicao de ensemble gerada; pulando metricas.")

if __name__ == "__main__":
    main()
