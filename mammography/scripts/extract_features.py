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

from mammography.utils.common import seed_everything, resolve_device, setup_logging, increment_path
from mammography.data.csv_loader import load_dataset_dataframe
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.features.extractor import ResNet50FeatureExtractor
from mammography.analysis.clustering import run_pca, run_tsne, find_optimal_k, run_kmeans
from mammography.vis.plots import plot_scatter, plot_clustering_metrics

def main():
    parser = argparse.ArgumentParser(description="Extract Features (Stage 1)")
    parser.add_argument("--csv", required=True, help="Path to CSV")
    parser.add_argument("--dicom-root", required=True, help="Root for DICOMs")
    parser.add_argument("--outdir", default="outputs/features", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tsne", action="store_true", help="Run t-SNE")
    parser.add_argument("--cluster", action="store_true", help="Run Clustering")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    outdir = increment_path(args.outdir)
    logger = setup_logging(outdir, "info")
    
    device = resolve_device(args.device)
    
    # Load Data
    df = load_dataset_dataframe(args.csv, args.dicom_root, exclude_class_5=True)
    logger.info(f"Loaded {len(df)} samples.")
    
    rows = df.to_dict("records")
    # Dataset just for inference (train=False, no augment)
    ds = MammoDensityDataset(
        rows, img_size=224, train=False, cache_mode="auto", cache_dir=os.path.join(outdir, "cache"), split_name="extract"
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=mammo_collate)
    
    # Extract
    extractor = ResNet50FeatureExtractor(device)
    features, metadata = extractor.extract(loader)
    
    if len(features) == 0:
        logger.error("No features extracted.")
        return

    logger.info(f"Features shape: {features.shape}")
    
    # Save raw
    np.save(os.path.join(outdir, "features.npy"), features)
    pd.DataFrame(metadata).to_csv(os.path.join(outdir, "metadata.csv"), index=False)
    
    # Analysis
    logger.info("Running PCA...")
    pca_2d = run_pca(features, 2, seed=args.seed)
    
    # Prepare DataFrame for plotting
    plot_df = pd.DataFrame(metadata)
    plot_df["pca_x"] = pca_2d[:, 0]
    plot_df["pca_y"] = pca_2d[:, 1]
    
    plot_scatter(plot_df, "pca_x", "pca_y", hue="professional_label", title="PCA by Label", out_path=os.path.join(outdir, "pca_label.png"))
    
    if args.tsne:
        logger.info("Running t-SNE...")
        tsne_2d = run_tsne(features, 2, seed=args.seed)
        plot_df["tsne_x"] = tsne_2d[:, 0]
        plot_df["tsne_y"] = tsne_2d[:, 1]
        plot_scatter(plot_df, "tsne_x", "tsne_y", hue="professional_label", title="t-SNE by Label", out_path=os.path.join(outdir, "tsne_label.png"))

    if args.cluster:
        logger.info("Finding optimal K...")
        k_res = find_optimal_k(features, range(2, 8), seed=args.seed)
        best_k = k_res["best_k"]
        logger.info(f"Best K: {best_k}")
        
        with open(os.path.join(outdir, "clustering_metrics.json"), "w") as f:
            json.dump(k_res, f, indent=2)
            
        plot_clustering_metrics(k_res["history"], os.path.join(outdir, "clustering_metrics.png"))
        
        labels, _ = run_kmeans(features, best_k, seed=args.seed)
        plot_df["cluster"] = labels
        plot_df["cluster"] = plot_df["cluster"].astype(str) # discrete hue
        
        plot_scatter(plot_df, "pca_x", "pca_y", hue="cluster", title=f"PCA by Cluster (K={best_k})", out_path=os.path.join(outdir, "pca_cluster.png"))
        if args.tsne:
             plot_scatter(plot_df, "tsne_x", "tsne_y", hue="cluster", title=f"t-SNE by Cluster (K={best_k})", out_path=os.path.join(outdir, "tsne_cluster.png"))

    plot_df.to_csv(os.path.join(outdir, "results.csv"), index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()
