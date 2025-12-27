#!/usr/bin/env python3
#
# extract_features.py
# mammography-pipelines
#
# Extracts CNN embeddings and optional projections/clustering analyses for mammography datasets.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Embedding extraction plus optional PCA/t-SNE/UMAP/clustering analysis."""
import sys
import os
import argparse
import json
import logging
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from mammography.config import HP
from mammography.utils.common import (
    seed_everything,
    resolve_device,
    setup_logging,
    increment_path,
    configure_runtime,
    parse_float_list,
)
from mammography.data.csv_loader import load_dataset_dataframe, resolve_dataset_cache_mode, DATASET_PRESETS, resolve_paths_from_preset
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.training.engine import extract_embeddings
from mammography.analysis.clustering import run_pca, run_tsne, run_kmeans, run_umap
from mammography.vis.plots import plot_scatter, plot_clustering_metrics

def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to uint8 HWC for previews."""
    arr = tensor.permute(1, 2, 0).numpy()
    arr = (arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    return np.uint8(arr * 255)

def save_first_image_preview(dataset: MammoDensityDataset, out_dir: Path) -> None:
    """Persist a single decoded example to quickly verify loader correctness."""
    if len(dataset) == 0:
        return
    img, _, meta, _ = dataset[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_tensor_to_uint8_image(img)).save(out_dir / "first_image_loaded.png")

def save_samples_grid(dataset: MammoDensityDataset, out_dir: Path, max_samples: int = 16) -> None:
    """Save a grid of sample thumbnails annotated with accession IDs."""
    if len(dataset) == 0:
        return
    idxs = list(range(min(max_samples, len(dataset))))
    imgs: list[Tuple[np.ndarray, dict]] = []
    for i in idxs:
        img, _, meta, _ = dataset[i]
        imgs.append((_tensor_to_uint8_image(img), meta))
    if not imgs:
        return
    cols = int(len(imgs) ** 0.5) or 1
    rows = int(np.ceil(len(imgs) / cols))
    h, w, _ = imgs[0][0].shape
    grid = Image.new("RGB", (w * cols, h * rows))
    for idx, (arr, meta) in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        patch = Image.fromarray(arr)
        draw = ImageDraw.Draw(patch)
        draw.text((4, 4), str(meta.get("accession", "?"))[:12], fill=(255, 0, 0))
        grid.paste(patch, (c * w, r * h))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / "samples_grid.png")

def plot_labels_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot a simple histogram of BI-RADS density labels present in the dataset."""
    if "professional_label" not in df.columns:
        return
    counts = df["professional_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("BI-RADS")
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "labels_distribution.png", dpi=150)
    plt.close(fig)

def resolve_loader_runtime(args, device: torch.device):
    """Pick DataLoader knobs that are friendlier to CPU/MPS while reusing CLI defaults."""
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
    parser = argparse.ArgumentParser(description="Extract embeddings + projections (EfficientNetB0/ResNet50)")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), help="Atalho para datasets conhecidos (archive/mamografias/patches_completo)")
    parser.add_argument("--csv", required=False, help="Caminho do CSV ou diretório com featureS.txt")
    parser.add_argument("--dicom-root", required=False, help="Raiz dos DICOMs para classificacao.csv")
    parser.add_argument("--outdir", default="outputs/features", help="Output directory")
    parser.add_argument("--seed", type=int, default=HP.SEED)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--deterministic", action="store_true", default=HP.DETERMINISTIC)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=HP.ALLOW_TF32)
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa pesos ImageNet quando disponiveis (default: True).",
    )
    parser.add_argument("--classes", default="multiclass", choices=["binary", "density", "multiclass"], help="Define mapeamento de labels (binário ou BI-RADS 1..4)")
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=HP.NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=HP.PREFETCH_FACTOR)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=HP.PERSISTENT_WORKERS)
    parser.add_argument("--loader-heuristics", action=argparse.BooleanOptionalAction, default=HP.LOADER_HEURISTICS)
    parser.add_argument("--cache-mode", default=HP.CACHE_MODE, choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"])
    parser.add_argument("--include-class-5", action="store_true", help="Mantém amostras com classificação 5 ao carregar classificacao.csv")
    parser.add_argument("--log-level", default=HP.LOG_LEVEL, choices=["critical","error","warning","info","debug"])
    parser.add_argument("--amp", action="store_true", help="Usa autocast durante inferência")
    parser.add_argument("--mean", help="Media de normalizacao (ex: 0.485,0.456,0.406)")
    parser.add_argument("--std", help="Std de normalizacao (ex: 0.229,0.224,0.225)")
    parser.add_argument("--layer-name", default="avgpool", help="Nome do layer para embeddings (named_modules)")
    parser.add_argument("--save-csv", action="store_true", help="Salva joined.csv com projeções")
    parser.add_argument("--run-reduction", action="store_true", help="Atalho para PCA + t-SNE + UMAP")
    parser.add_argument("--run-clustering", action="store_true", help="Atalho para k-means")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--tsne", action="store_true", help="Run t-SNE")
    parser.add_argument("--umap", action="store_true", help="Run UMAP")
    parser.add_argument("--cluster", dest="cluster_auto", action="store_true", help="Auto k-means (usa silhouette)")
    parser.add_argument("--cluster-k", type=int, default=0, help="Força K fixo (>=2) em k-means")
    parser.add_argument("--n-clusters", type=int, default=0, help="Alias para --cluster-k")
    parser.add_argument("--sample-grid", type=int, default=16, help="Número de exemplos na grade de pré-visualização")
    args = parser.parse_args()

    if args.n_clusters and args.cluster_k <= 0:
        args.cluster_k = args.n_clusters
    if args.run_reduction:
        args.pca = True
        args.tsne = True
        args.umap = True
    if args.run_clustering and not args.cluster_auto and args.cluster_k <= 0:
        args.cluster_auto = True
    
    csv_path, dicom_root = resolve_paths_from_preset(args.csv, args.dataset, args.dicom_root)
    if not csv_path:
        raise SystemExit("Informe --csv ou --dataset para localizar os dados.")

    seed_everything(args.seed)
    outdir = increment_path(args.outdir)
    logger = setup_logging(outdir, args.log_level)

    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)

    # Load Data
    df = load_dataset_dataframe(
        csv_path,
        dicom_root,
        exclude_class_5=not args.include_class_5,
        dataset=args.dataset,
    )
    logger.info(f"Loaded {len(df)} samples.")

    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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
        mean=mean,
        std=std,
    )
    nw, prefetch, persistent = resolve_loader_runtime(args, device)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": nw,
        "collate_fn": mammo_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(persistent and nw > 0),
    }
    if prefetch is not None and nw > 0:
        loader_kwargs["prefetch_factor"] = prefetch
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    num_classes = 2 if args.classes == "binary" else 4
    model = build_model(
        args.arch,
        num_classes=num_classes,
        train_backbone=False,
        unfreeze_last_block=False,
        pretrained=args.pretrained,
    ).to(device)

    # Extract
    features, metadata = extract_embeddings(
        model,
        loader,
        device,
        amp_enabled=args.amp and device.type in ["cuda", "mps"],
        layer_name=args.layer_name,
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
        history = []
        best_labels = None
        best_k = None
        best_score = -np.inf
        for k in k_values:
            labels, _ = run_kmeans(features, k, seed=args.seed)
            try:
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                sil = silhouette_score(features, labels)
                db = davies_bouldin_score(features, labels)
            except Exception:
                sil = -np.inf
                db = np.inf
            history.append({"k": k, "silhouette": float(sil), "davies_bouldin": float(db)})
            if sil > best_score:
                best_score = sil
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

    preview_dir = Path(outdir) / "preview"
    save_first_image_preview(ds, preview_dir)
    save_samples_grid(ds, preview_dir, max_samples=args.sample_grid)
    plot_labels_distribution(df, preview_dir)

    session_info = {
        "device": str(device),
        "num_samples": len(ds),
        "features_shape": list(features.shape),
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(outdir) / "session_info.json", "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2)
    if len(features) > 0:
        sample_embedding = {"embedding": features[0].tolist(), **metadata[0]}
        with open(Path(outdir) / "example_embedding.json", "w", encoding="utf-8") as f:
            json.dump(sample_embedding, f, indent=2)
    logger.info("Done.")

if __name__ == "__main__":
    main()
