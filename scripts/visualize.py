#!/usr/bin/env python3
#
# visualize.py
# mammography-pipelines-py
#
# CLI script for generating visualizations from embeddings and training outputs.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""
Visualization CLI — Generate t-SNE, heatmaps, scatterplots and more from embeddings.

Usage:
  python visualize.py --input features.npy --labels metadata.csv --outdir vis_output
  python visualize.py --input features.npy --report  # Generate full report
  python visualize.py --input run_dir --from-run     # Visualize from training run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.vis.advanced import (
    plot_tsne_2d,
    plot_tsne_3d,
    plot_heatmap_correlation,
    plot_confusion_matrix_heatmap,
    plot_feature_heatmap,
    plot_scatter_matrix,
    plot_distribution,
    plot_embedding_comparison,
    plot_class_separation,
    plot_feature_importance,
    plot_learning_curves,
    generate_visualization_report,
)

BIRADS_NAMES = {
    0: "BI-RADS A",
    1: "BI-RADS B", 
    2: "BI-RADS C",
    3: "BI-RADS D",
}

BINARY_NAMES = {
    0: "Low Density",
    1: "High Density",
}


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the visualization script."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_features(path: str) -> np.ndarray:
    """Load features from .npy or .npz file."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        # Try common key names
        for key in ["features", "embeddings", "arr_0", "data"]:
            if key in data:
                return data[key]
        # Return first array
        return data[list(data.keys())[0]]
    return np.load(path)


def load_labels(path: str, label_col: str = "raw_label") -> Optional[np.ndarray]:
    """Load labels from CSV file."""
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if label_col in df.columns:
        return df[label_col].values.astype(int)
    # Try alternative column names
    for col in ["label", "class", "professional_label", "y", "target"]:
        if col in df.columns:
            return df[col].values.astype(int)
    return None


def load_predictions(path: str) -> Optional[pd.DataFrame]:
    """Load prediction CSV with true/pred columns."""
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    return df


def load_history(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load training history from CSV or JSON."""
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")
    return None


def discover_run_artifacts(run_dir: Path) -> Dict[str, Optional[Path]]:
    """Find relevant files in a training run directory."""
    artifacts = {
        "features": None,
        "metadata": None,
        "predictions": None,
        "history": None,
        "metrics": None,
    }
    
    # Look for feature files
    for name in ["features.npy", "embeddings_val.npy", "embeddings.npy"]:
        if (run_dir / name).exists():
            artifacts["features"] = run_dir / name
            break
    
    # Look for metadata
    for name in ["metadata.csv", "embeddings_val.csv", "joined.csv"]:
        if (run_dir / name).exists():
            artifacts["metadata"] = run_dir / name
            break
    
    # Look for predictions
    for name in ["val_predictions.csv", "predictions.csv"]:
        if (run_dir / name).exists():
            artifacts["predictions"] = run_dir / name
            break
    
    # Look for training history
    for name in ["train_history.csv", "history.csv", "train_history.json"]:
        if (run_dir / name).exists():
            artifacts["history"] = run_dir / name
            break
    
    # Look for metrics
    for name in ["val_metrics.json", "best_metrics.json", "metrics.json"]:
        if (run_dir / name).exists():
            artifacts["metrics"] = run_dir / name
            break
    if "metrics" not in artifacts:
        metrics_dir = run_dir / "metrics"
        for name in ["val_metrics.json", "best_metrics.json", "metrics.json"]:
            if (metrics_dir / name).exists():
                artifacts["metrics"] = metrics_dir / name
                break
    
    return artifacts


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from mammography pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate t-SNE plot from features
  python visualize.py --input features.npy --labels metadata.csv --tsne

  # Generate full visualization report
  python visualize.py --input features.npy --labels metadata.csv --report

  # Visualize from a training run directory
  python visualize.py --from-run outputs/run_001

  # Generate specific visualizations
  python visualize.py --input features.npy --tsne --heatmap --scatter-matrix
        """,
    )
    
    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--input", "-i",
        help="Path to features (.npy/.npz) or run directory (with --from-run)",
    )
    input_group.add_argument(
        "--labels", "-l",
        help="Path to CSV with labels (column: raw_label, label, class)",
    )
    input_group.add_argument(
        "--label-col",
        default="raw_label",
        help="Column name for labels in CSV (default: raw_label)",
    )
    input_group.add_argument(
        "--predictions", "-p",
        help="Path to predictions CSV for confusion matrix",
    )
    input_group.add_argument(
        "--history",
        help="Path to training history (CSV or JSON) for learning curves",
    )
    input_group.add_argument(
        "--from-run",
        action="store_true",
        help="Treat --input as a run directory and auto-discover artifacts",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output", "-o", "--outdir",
        dest="output",
        default="outputs/visualizations",
        help="Output directory (default: outputs/visualizations)",
    )
    output_group.add_argument(
        "--prefix",
        default="",
        help="Prefix for output filenames",
    )
    
    # Visualization types
    viz_group = parser.add_argument_group("Visualizations")
    viz_group.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive visualization report",
    )
    viz_group.add_argument(
        "--tsne",
        action="store_true",
        help="Generate t-SNE 2D plot",
    )
    viz_group.add_argument(
        "--tsne-3d",
        action="store_true",
        help="Generate t-SNE 3D plot",
    )
    viz_group.add_argument(
        "--pca",
        action="store_true",
        help="Generate PCA scatter plot",
    )
    viz_group.add_argument(
        "--umap",
        action="store_true",
        help="Generate UMAP scatter plot",
    )
    viz_group.add_argument(
        "--compare-embeddings",
        action="store_true",
        help="Compare PCA, t-SNE, and UMAP side by side",
    )
    viz_group.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate feature correlation heatmap",
    )
    viz_group.add_argument(
        "--feature-heatmap",
        action="store_true",
        help="Generate clustered feature heatmap",
    )
    viz_group.add_argument(
        "--scatter-matrix",
        action="store_true",
        help="Generate pairwise scatter plot matrix",
    )
    viz_group.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Generate confusion matrix heatmap (requires --predictions)",
    )
    viz_group.add_argument(
        "--distribution",
        action="store_true",
        help="Generate distribution plots",
    )
    viz_group.add_argument(
        "--class-separation",
        action="store_true",
        help="Generate class separation analysis",
    )
    viz_group.add_argument(
        "--learning-curves",
        action="store_true",
        help="Generate learning curves (requires --history)",
    )
    
    # t-SNE parameters
    tsne_group = parser.add_argument_group("t-SNE Parameters")
    tsne_group.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30)",
    )
    tsne_group.add_argument(
        "--tsne-iter",
        type=int,
        default=1000,
        help="t-SNE iterations (default: 1000)",
    )
    
    # General options
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    general_group.add_argument(
        "--binary",
        action="store_true",
        help="Use binary class names (Low/High Density)",
    )
    general_group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for visualization CLI."""
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    # Determine label names
    label_names = BINARY_NAMES if args.binary else BIRADS_NAMES
    
    # Handle run directory mode
    if args.from_run:
        if not args.input:
            logger.error("--from-run requires --input pointing to a run directory")
            return 1
        
        run_dir = Path(args.input)
        if not run_dir.is_dir():
            logger.error(f"Run directory not found: {run_dir}")
            return 1
        
        logger.info(f"Discovering artifacts in {run_dir}...")
        artifacts = discover_run_artifacts(run_dir)
        
        if artifacts["features"]:
            args.input = str(artifacts["features"])
            logger.info(f"  Features: {artifacts['features']}")
        else:
            logger.error("No feature file found in run directory")
            return 1
        
        if artifacts["metadata"] and not args.labels:
            args.labels = str(artifacts["metadata"])
            logger.info(f"  Labels: {artifacts['metadata']}")
        
        if artifacts["predictions"] and not args.predictions:
            args.predictions = str(artifacts["predictions"])
            logger.info(f"  Predictions: {artifacts['predictions']}")
        
        if artifacts["history"] and not args.history:
            args.history = str(artifacts["history"])
            logger.info(f"  History: {artifacts['history']}")
    
    # Validate input
    if not args.input:
        logger.error("--input is required. Use --help for usage.")
        return 1
    
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Load features
    logger.info(f"Loading features from {args.input}...")
    features = load_features(args.input)
    logger.info(f"  Shape: {features.shape}")
    
    # Load labels
    labels = None
    if args.labels:
        logger.info(f"Loading labels from {args.labels}...")
        labels = load_labels(args.labels, args.label_col)
        if labels is not None:
            unique_labels = np.unique(labels)
            logger.info(f"  Unique labels: {unique_labels}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    prefix = f"{args.prefix}_" if args.prefix else ""
    
    # Check if any visualization is requested
    any_viz = any([
        args.report,
        args.tsne,
        args.tsne_3d,
        args.pca,
        args.umap,
        args.compare_embeddings,
        args.heatmap,
        args.feature_heatmap,
        args.scatter_matrix,
        args.confusion_matrix,
        args.distribution,
        args.class_separation,
        args.learning_curves,
    ])
    
    if not any_viz:
        logger.warning("No visualization type specified. Use --report for full report or specify individual plots.")
        logger.info("Available: --tsne, --tsne-3d, --pca, --umap, --compare-embeddings, --heatmap, --feature-heatmap, --scatter-matrix, --confusion-matrix, --distribution, --class-separation, --learning-curves, --report")
        return 0
    
    # Generate visualizations
    try:
        if args.report:
            logger.info("Generating comprehensive visualization report...")
            report_paths = generate_visualization_report(
                features,
                labels,
                output_dir=output_dir / f"{prefix}report",
                seed=args.seed,
                label_names=label_names,
            )
            logger.info(f"Report generated with {len(report_paths)} visualizations")
        
        if args.tsne:
            logger.info("Generating t-SNE 2D plot...")
            out_path = output_dir / f"{prefix}tsne_2d.png"
            _, fig = plot_tsne_2d(
                features,
                labels,
                perplexity=args.perplexity,
                n_iter=args.tsne_iter,
                seed=args.seed,
                out_path=str(out_path),
                label_names=label_names,
            )
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")
        
        if args.tsne_3d:
            logger.info("Generating t-SNE 3D plot...")
            out_path = output_dir / f"{prefix}tsne_3d.png"
            _, fig = plot_tsne_3d(
                features,
                labels,
                perplexity=args.perplexity,
                seed=args.seed,
                out_path=str(out_path),
                label_names=label_names,
            )
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")
        
        if args.pca:
            logger.info("Generating PCA plot...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=args.seed)
            pca_emb = pca.fit_transform(features)
            df = pd.DataFrame({"pca_x": pca_emb[:, 0], "pca_y": pca_emb[:, 1]})
            if labels is not None:
                df["label"] = labels
            
            from mammography.vis.plots import plot_scatter
            out_path = output_dir / f"{prefix}pca.png"
            plot_scatter(
                df,
                "pca_x",
                "pca_y",
                hue="label" if labels is not None else None,
                title=f"PCA (var: {pca.explained_variance_ratio_.sum():.1%})",
                out_path=str(out_path),
            )
            logger.info(f"  Saved: {out_path}")
        
        if args.umap:
            logger.info("Generating UMAP plot...")
            try:
                from umap import UMAP
                umap_model = UMAP(n_components=2, random_state=args.seed)
                umap_emb = umap_model.fit_transform(features)
                df = pd.DataFrame({"umap_x": umap_emb[:, 0], "umap_y": umap_emb[:, 1]})
                if labels is not None:
                    df["label"] = labels
                
                from mammography.vis.plots import plot_scatter
                out_path = output_dir / f"{prefix}umap.png"
                plot_scatter(
                    df,
                    "umap_x",
                    "umap_y",
                    hue="label" if labels is not None else None,
                    title="UMAP",
                    out_path=str(out_path),
                )
                logger.info(f"  Saved: {out_path}")
            except ImportError:
                logger.warning("UMAP not installed. Skipping UMAP plot.")
        
        if args.compare_embeddings:
            logger.info("Generating embedding comparison...")
            out_path = output_dir / f"{prefix}embedding_comparison.png"
            methods = ["pca", "tsne"]
            try:
                from umap import UMAP
                methods.append("umap")
            except ImportError:
                pass
            _, fig = plot_embedding_comparison(
                features,
                labels,
                methods=methods,
                seed=args.seed,
                out_path=str(out_path),
                label_names=label_names,
            )
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")
        
        if args.heatmap:
            logger.info("Generating correlation heatmap...")
            out_path = output_dir / f"{prefix}correlation_heatmap.png"
            fig = plot_heatmap_correlation(
                features,
                out_path=str(out_path),
            )
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")
        
        if args.feature_heatmap:
            logger.info("Generating feature heatmap...")
            out_path = output_dir / f"{prefix}feature_heatmap.png"
            fig = plot_feature_heatmap(
                features,
                out_path=str(out_path),
            )
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")
        
        if args.scatter_matrix:
            logger.info("Generating scatter matrix...")
            out_path = output_dir / f"{prefix}scatter_matrix.png"
            fig = plot_scatter_matrix(
                features,
                labels,
                out_path=str(out_path),
                label_names=label_names,
            )
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")
        
        if args.confusion_matrix:
            if not args.predictions:
                logger.warning("--confusion-matrix requires --predictions. Skipping.")
            else:
                logger.info("Generating confusion matrix...")
                pred_df = load_predictions(args.predictions)
                if pred_df is not None:
                    y_true = pred_df.get("true_label", pred_df.get("y_true", pred_df.get("label")))
                    y_pred = pred_df.get("pred_label", pred_df.get("y_pred", pred_df.get("prediction")))
                    
                    if y_true is not None and y_pred is not None:
                        class_names = [label_names.get(i, str(i)) for i in sorted(y_true.unique())]
                        out_path = output_dir / f"{prefix}confusion_matrix.png"
                        fig = plot_confusion_matrix_heatmap(
                            y_true.values,
                            y_pred.values,
                            class_names=class_names,
                            out_path=str(out_path),
                        )
                        plt.close(fig)
                        logger.info(f"  Saved: {out_path}")
                    else:
                        logger.warning("Could not find true/pred columns in predictions file")
        
        if args.distribution:
            logger.info("Generating distribution plots...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1, random_state=args.seed)
            pc1 = pca.fit_transform(features).ravel()
            
            for kind in ["hist", "kde"]:
                out_path = output_dir / f"{prefix}pc1_{kind}.png"
                fig = plot_distribution(
                    pc1,
                    labels,
                    kind=kind,
                    title=f"First Principal Component ({kind.upper()})",
                    xlabel="PC1",
                    out_path=str(out_path),
                    label_names=label_names,
                )
                plt.close(fig)
                logger.info(f"  Saved: {out_path}")
        
        if args.class_separation:
            if labels is None:
                logger.warning("--class-separation requires labels. Skipping.")
            else:
                logger.info("Generating class separation analysis...")
                out_path = output_dir / f"{prefix}class_separation.png"
                fig = plot_class_separation(
                    features,
                    labels,
                    out_path=str(out_path),
                    label_names=label_names,
                )
                plt.close(fig)
                logger.info(f"  Saved: {out_path}")
        
        if args.learning_curves:
            if not args.history:
                logger.warning("--learning-curves requires --history. Skipping.")
            else:
                logger.info("Generating learning curves...")
                history = load_history(args.history)
                if history:
                    out_path = output_dir / f"{prefix}learning_curves.png"
                    fig = plot_learning_curves(
                        history,
                        out_path=str(out_path),
                    )
                    plt.close(fig)
                    logger.info(f"  Saved: {out_path}")
                else:
                    logger.warning("Could not load training history")
        
        logger.info("Visualization complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1


# Import matplotlib at module level after functions are defined
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys.exit(main())
