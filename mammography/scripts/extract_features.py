#!/usr/bin/env python3
import sys
import os
import argparse
import json
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.config import HP
from mammography.utils.common import seed_everything, resolve_device, setup_logging, increment_path, configure_runtime
from mammography.data.csv_loader import load_dataset_dataframe, resolve_dataset_cache_mode, DATASET_PRESETS, resolve_paths_from_preset
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.training.engine import extract_embeddings
from mammography.analysis.clustering import run_pca, run_tsne, find_optimal_k, run_kmeans, run_umap
from mammography.vis.plots import plot_scatter, plot_clustering_metrics

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings + projections (EfficientNetB0/ResNet50)")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), help="Atalho para datasets conhecidos (archive/mamografias/patches_completo)")
    parser.add_argument("--csv", required=False, help="Caminho do CSV ou diretório com featureS.txt")
    parser.add_argument("--dicom-root", required=False, help="Raiz dos DICOMs para classificacao.csv")
    parser.add_argument("--outdir", default="outputs/features", help="Output directory")
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--classes", default="multiclass", choices=["binary", "density", "multiclass"], help="Define mapeamento de labels (binário ou BI-RADS 1..4)")
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-mode", default=HP.CACHE_MODE, choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"])
    parser.add_argument("--amp", action="store_true", help="Usa autocast durante inferência")
    parser.add_argument("--save-csv", action="store_true", help="Salva joined.csv com projeções")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--tsne", action="store_true", help="Run t-SNE")
    parser.add_argument("--umap", action="store_true", help="Run UMAP")
    parser.add_argument("--cluster", dest="cluster_auto", action="store_true", help="Auto k-means (usa silhouette)")
    parser.add_argument("--cluster-k", type=int, default=0, help="Força K fixo (>=2) em k-means")
    args = parser.parse_args()
    
    csv_path, dicom_root = resolve_paths_from_preset(args.csv, args.dataset, args.dicom_root)
    if not csv_path:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    seed_everything(args.seed)
    outdir = increment_path(args.outdir)
    logger = setup_logging(outdir, "info")

    device = resolve_device(args.device)
    configure_runtime(device, False, True)

    # Load Data
    df = load_dataset_dataframe(csv_path, dicom_root, exclude_class_5=True, dataset=args.dataset)
    logger.info(f"Loaded {len(df)} samples.")

    rows = df.to_dict("records")
    mapper = None
    if args.classes == "binary":
        def _mapper(y):
            if y in [1, 2]:
                return 0
            if y in [3, 4]:
                return 1
            return y - 1
        mapper = _mapper

    cache_dir = Path(outdir) / "cache_extract"
    cache_mode = resolve_dataset_cache_mode(args.cache_mode, rows)
    ds = MammoDensityDataset(
        rows,
        img_size=args.img_size,
        train=False,
        cache_mode=cache_mode,
        cache_dir=str(cache_dir),
        split_name="extract",
        label_mapper=mapper,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": HP.NUM_WORKERS,
        "collate_fn": mammo_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": HP.PERSISTENT_WORKERS and HP.NUM_WORKERS > 0,
    }
    if HP.NUM_WORKERS > 0 and HP.PREFETCH_FACTOR:
        loader_kwargs["prefetch_factor"] = HP.PREFETCH_FACTOR
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    num_classes = 2 if args.classes == "binary" else 4
    model = build_model(args.arch, num_classes=num_classes, train_backbone=False, unfreeze_last_block=False).to(device)

    # Extract
    features, metadata = extract_embeddings(
        model,
        loader,
        device,
        amp_enabled=args.amp and device.type in ["cuda", "mps"],
    )

    if len(features) == 0:
        logger.error("No features extracted.")
        return

    logger.info(f"Features shape: {features.shape}")

    # Save raw
    np.save(os.path.join(outdir, "features.npy"), features)
    pd.DataFrame(metadata).to_csv(os.path.join(outdir, "metadata.csv"), index=False)

    # Analysis
    joined = pd.DataFrame(metadata)
    if args.pca and features.shape[0] > 1:
        logger.info("Running PCA...")
        pca_2d = run_pca(features, 2, seed=args.seed)
        joined["pca_x"] = pca_2d[:, 0]
        joined["pca_y"] = pca_2d[:, 1]
        plot_scatter(joined, "pca_x", "pca_y", hue="raw_label", title="PCA by Label", out_path=os.path.join(outdir, "pca_label.png"))

    if args.tsne and features.shape[0] > 2:
        logger.info("Running t-SNE...")
        tsne_2d = run_tsne(features, 2, seed=args.seed)
        joined["tsne_x"] = tsne_2d[:, 0]
        joined["tsne_y"] = tsne_2d[:, 1]
        plot_scatter(joined, "tsne_x", "tsne_y", hue="raw_label", title="t-SNE by Label", out_path=os.path.join(outdir, "tsne_label.png"))

    if args.umap and features.shape[0] > 5:
        logger.info("Running UMAP...")
        umap_2d = run_umap(features, 2, seed=args.seed)
        joined["umap_x"] = umap_2d[:, 0]
        joined["umap_y"] = umap_2d[:, 1]
        plot_scatter(joined, "umap_x", "umap_y", hue="raw_label", title="UMAP by Label", out_path=os.path.join(outdir, "umap_label.png"))

    if args.cluster_auto or args.cluster_k:
        k_values = [args.cluster_k] if args.cluster_k and args.cluster_k >= 2 else list(range(2, min(8, features.shape[0] + 1)))
        logger.info(f"Running k-means for k in {k_values}")
        best_k = None
        best_score = -np.inf
        best_labels = None
        history = []
        for k in k_values:
            labels, _ = run_kmeans(features, k, seed=args.seed)
            try:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(features, labels)
            except Exception:
                score = -np.inf
            history.append({"k": k, "silhouette": float(score)})
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        if best_labels is not None:
            joined["cluster"] = best_labels
            plot_clustering_metrics(history, os.path.join(outdir, "clustering_metrics.png"))
            if "pca_x" in joined.columns:
                plot_scatter(joined, "pca_x", "pca_y", hue="cluster", title=f"PCA by Cluster (K={best_k})", out_path=os.path.join(outdir, "pca_cluster.png"))
            if "tsne_x" in joined.columns:
                plot_scatter(joined, "tsne_x", "tsne_y", hue="cluster", title=f"t-SNE by Cluster (K={best_k})", out_path=os.path.join(outdir, "tsne_cluster.png"))
            with open(os.path.join(outdir, "clustering_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"best_k": best_k, "history": history}, f, indent=2)

    if args.save_csv and not joined.empty:
        joined.to_csv(os.path.join(outdir, "joined.csv"), index=False)

    session_info = {
        "device": str(device),
        "num_samples": len(ds),
        "features_shape": list(features.shape),
        "args": vars(args),
    }
    with open(Path(outdir) / "session_info.json", "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2)
    logger.info("Done.")

if __name__ == "__main__":
    main()
