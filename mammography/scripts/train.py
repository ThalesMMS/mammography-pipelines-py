#!/usr/bin/env python3
import sys
import os
import argparse
import json
import logging
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add src to path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.config import HP
from mammography.utils.common import seed_everything, resolve_device, configure_runtime, setup_logging, increment_path
from mammography.data.csv_loader import (
    load_dataset_dataframe,
    resolve_dataset_cache_mode,
    DATASET_PRESETS,
    resolve_paths_from_preset,
)
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.training.engine import (
    train_one_epoch,
    validate,
    save_metrics_figure,
    extract_embeddings,
    plot_history,
    save_predictions,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento Mammography (EfficientNetB0/ResNet50)")

    # Data
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), help="Atalho para datasets conhecidos (archive/mamografias/patches_completo)")
    parser.add_argument("--csv", required=False, help="CSV, diretório com featureS.txt ou caminho manual")
    parser.add_argument("--dicom-root", help="Root for DICOMs (usado com classificacao.csv)")
    parser.add_argument("--outdir", default="outputs/run", help="Output directory")
    parser.add_argument("--cache-mode", default=HP.CACHE_MODE, choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"])
    parser.add_argument("--cache-dir", help="Cache dir")
    parser.add_argument("--log-level", default=HP.LOG_LEVEL, choices=["critical","error","warning","info","debug"])

    # Model / task
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--classes", default="density", choices=["density", "binary", "multiclass"], help="density/multiclass = BI-RADS 1..4, binary = A/B vs C/D")
    parser.add_argument("--task", dest="classes", choices=["density", "binary", "multiclass"], help="Alias para --classes")

    # HP overrides
    parser.add_argument("--epochs", type=int, default=HP.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=HP.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=HP.LR)
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--val-frac", type=float, default=HP.VAL_FRAC)
    parser.add_argument("--num-workers", type=int, default=HP.NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=HP.PREFETCH_FACTOR)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=HP.PERSISTENT_WORKERS)
    parser.add_argument("--loader-heuristics", action=argparse.BooleanOptionalAction, default=HP.LOADER_HEURISTICS)
    parser.add_argument("--amp", action="store_true", help="Habilita autocast + GradScaler em CUDA/MPS")
    parser.add_argument("--class-weights", choices=["none", "auto"], default=HP.CLASS_WEIGHTS)
    parser.add_argument("--sampler-weighted", action="store_true", default=HP.SAMPLER_WEIGHTED)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=HP.TRAIN_BACKBONE)
    parser.add_argument("--unfreeze-last-block", action=argparse.BooleanOptionalAction, default=HP.UNFREEZE_LAST_BLOCK)
    parser.add_argument("--warmup-epochs", type=int, default=HP.WARMUP_EPOCHS)
    parser.add_argument("--deterministic", action="store_true", default=HP.DETERMINISTIC)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=HP.ALLOW_TF32)
    parser.add_argument("--early-stop-patience", type=int, default=HP.EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=HP.EARLY_STOP_MIN_DELTA)

    # Outputs/analysis
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--gradcam-limit", type=int, default=4)
    parser.add_argument("--save-val-preds", action="store_true")
    parser.add_argument("--export-val-embeddings", action="store_true")

    return parser.parse_args()

def get_label_mapper(mode):
    mode = (mode or "density").lower()
    if mode == "binary":
        # 1,2 -> 0 (Low); 3,4 -> 1 (High)
        def mapper(y):
            if y in [1, 2]: return 0
            if y in [3, 4]: return 1
            return y - 1 # Fallback
        return mapper
    return None # Default 1..4 -> 0..3

def resolve_loader_runtime(args, device: torch.device):
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

def main():
    args = parse_args()

    csv_path, dicom_root = resolve_paths_from_preset(args.csv, args.dataset, args.dicom_root)
    if not csv_path:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    seed_everything(args.seed, deterministic=args.deterministic)

    outdir = increment_path(args.outdir)
    logger = setup_logging(outdir, args.log_level)
    logger.info(f"Args: {args}")

    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)

    # Load Data
    df = load_dataset_dataframe(csv_path, dicom_root, exclude_class_5=True, dataset=args.dataset)
    logger.info(f"Loaded {len(df)} samples from dataset '{args.dataset or 'custom'}'.")
    df = df[df["professional_label"].notna()]
    logger.info(f"Valid samples (with label): {len(df)}")

    # Prepare Split by accession
    y_all = df["professional_label"].astype(int).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, y_all, groups=df["accession"]))
    train_rows = df.iloc[train_idx].to_dict("records")
    val_rows = df.iloc[val_idx].to_dict("records")

    cache_dir = args.cache_dir or os.path.join(outdir, "cache")
    cache_mode_train = resolve_dataset_cache_mode(args.cache_mode, train_rows)
    cache_mode_val = resolve_dataset_cache_mode(args.cache_mode, val_rows)

    num_classes = 2 if args.classes == "binary" else 4
    mapper = get_label_mapper(args.classes)

    train_ds = MammoDensityDataset(
        train_rows,
        args.img_size,
        train=True,
        cache_mode=cache_mode_train,
        cache_dir=cache_dir,
        split_name="train",
        label_mapper=mapper,
    )
    val_ds = MammoDensityDataset(
        val_rows,
        args.img_size,
        train=False,
        cache_mode=cache_mode_val,
        cache_dir=cache_dir,
        split_name="val",
        label_mapper=mapper,
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
        mapped = [_map_row_label(r) for r in train_rows if _map_row_label(r) is not None]
        counts = np.bincount(np.array(mapped, dtype=int), minlength=num_classes)
        weights = torch.tensor(len(mapped) / (counts + 1e-6), dtype=torch.float)
        sample_weights = torch.tensor([weights[_map_row_label(r)] for r in train_rows], dtype=torch.float)
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = None
    if args.class_weights == "auto":
        mapped = [_map_row_label(r) for r in train_rows if _map_row_label(r) is not None]
        counts = np.bincount(np.array(mapped, dtype=int), minlength=num_classes)
        weights = torch.tensor(len(mapped) / (num_classes * counts + 1e-6), dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler() if args.amp and device.type == "cuda" else None

    best_acc = -1.0
    best_epoch = -1
    patience_ctr = 0
    history = []
    gradcam_dir = Path(outdir) / "gradcam" if args.gradcam else None

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
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
        v_acc = val_metrics.get("acc", 0.0)
        v_loss = val_metrics.get("loss", 0.0) or 0.0

        logger.info(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}"
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
            }
        )
        plot_history(history, Path(outdir))

        # Persist latest metrics
        with open(os.path.join(outdir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2, default=str)
        save_metrics_figure(val_metrics, os.path.join(outdir, "val_metrics.png"))
        if args.save_val_preds:
            save_predictions(pred_rows, Path(outdir))

        improved = v_acc > best_acc + args.early_stop_min_delta
        if improved:
            best_acc = v_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pt"))
            with open(os.path.join(outdir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2, default=str)
            save_metrics_figure(val_metrics, os.path.join(outdir, "best_metrics.png"))
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.early_stop_patience and patience_ctr >= args.early_stop_patience:
            logger.info(f"Early stopping ativado após {patience_ctr} épocas sem melhoria.")
            break

    logger.info(f"Done. Best Acc: {best_acc:.4f} (epoch {best_epoch})")

    if args.export_val_embeddings:
        logger.info("Extraindo embeddings do conjunto de validação...")
        best_path = Path(outdir) / "best_model.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state, strict=False)
        feats, metas = extract_embeddings(model, val_loader, device, amp_enabled=args.amp and device.type in ["cuda", "mps"])
        np.save(Path(outdir) / "embeddings_val.npy", feats)
        pd.DataFrame(metas).to_csv(Path(outdir) / "embeddings_val.csv", index=False)

if __name__ == "__main__":
    main()
