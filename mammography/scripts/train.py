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
from sklearn.model_selection import StratifiedKFold

# Add src to path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.config import HP
from mammography.utils.common import seed_everything, resolve_device, configure_runtime, setup_logging, increment_path
from mammography.data.csv_loader import load_dataset_dataframe, resolve_dataset_cache_mode
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.training.engine import train_one_epoch, validate, save_metrics_figure, extract_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento Mammography")
    
    # Data
    parser.add_argument("--csv", required=True, help="Path to CSV")
    parser.add_argument("--dicom-root", help="Root for DICOMs")
    parser.add_argument("--outdir", default="outputs/run", help="Output directory")
    parser.add_argument("--cache-mode", default="auto", choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"])
    parser.add_argument("--cache-dir", help="Cache dir")
    
    # Model
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--classes", default="density", choices=["density", "binary"], help="density (4 classes) or binary (A/B vs C/D)")
    parser.add_argument("--weights", help="Path to pretrained weights")
    parser.add_argument("--resume", help="Resume from checkpoint")
    
    # HP overrides
    parser.add_argument("--epochs", type=int, default=HP.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=HP.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=HP.LR)
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--val-frac", type=float, default=HP.VAL_FRAC)
    
    return parser.parse_args()

def get_label_mapper(mode):
    if mode == "binary":
        # 1,2 -> 0 (Low); 3,4 -> 1 (High)
        def mapper(y):
            if y in [1, 2]: return 0
            if y in [3, 4]: return 1
            return y - 1 # Fallback
        return mapper
    return None # Default 1..4 -> 0..3

def main():
    args = parse_args()
    seed_everything(args.seed)
    
    outdir = increment_path(args.outdir)
    logger = setup_logging(outdir, "info")
    logger.info(f"Args: {args}")
    
    device = resolve_device(args.device)
    configure_runtime(device, False, True)
    
    # Load Data
    df = load_dataset_dataframe(args.csv, args.dicom_root)
    logger.info(f"Loaded {len(df)} samples from CSV.")
    
    # Filter
    df = df[df["professional_label"].notna()]
    logger.info(f"Valid samples (with label): {len(df)}")
    
    # Prepare Split
    y = df["professional_label"].astype(int).values
    skf = StratifiedKFold(n_splits=int(1/args.val_frac), shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(skf.split(df, y))
    
    train_rows = df.iloc[train_idx].to_dict("records")
    val_rows = df.iloc[val_idx].to_dict("records")
    
    # Cache resolution
    cache_mode = resolve_dataset_cache_mode(args.cache_mode, df)
    cache_dir = args.cache_dir or os.path.join(outdir, "cache")
    
    num_classes = 2 if args.classes == "binary" else 4
    mapper = get_label_mapper(args.classes)
    
    train_ds = MammoDensityDataset(
        train_rows, args.img_size, train=True, 
        cache_mode=cache_mode, cache_dir=cache_dir, split_name="train", label_mapper=mapper
    )
    val_ds = MammoDensityDataset(
        val_rows, args.img_size, train=False, 
        cache_mode=cache_mode, cache_dir=cache_dir, split_name="val", label_mapper=mapper
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=mammo_collate, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=mammo_collate, pin_memory=True
    )
    
    # Model
    model = build_model(args.arch, num_classes=num_classes, train_backbone=False)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_acc = 0.0
    
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, optimizer, device, 
            scaler=scaler, amp_enabled=(device.type in ['cuda', 'mps'])
        )
        
        val_metrics = validate(model, val_loader, device, amp_enabled=(device.type in ['cuda', 'mps']))
        v_acc = val_metrics["acc"]
        
        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        
        # Save Best
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
            with open(os.path.join(outdir, "best_metrics.json"), "w") as f:
                json.dump(val_metrics, f, indent=2, default=str)
            save_metrics_figure(val_metrics, os.path.join(outdir, "best_metrics.png"))
            
    logger.info(f"Done. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
