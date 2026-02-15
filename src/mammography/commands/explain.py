#!/usr/bin/env python3
#
# explain.py
# mammography-pipelines
#
# CLI script for generating explainability visualizations (GradCAM, attention maps) for mammography models.
#
# Thales Matheus Mendonça Santos - January 2026
#
"""
Explainability CLI — Generate GradCAM heatmaps and attention maps to explain model predictions.

Usage:
  python explain.py --model-path model.pth --images-dir imgs/ --output-dir explanations/
  python explain.py --model-path model.pth --images-dir imgs/ --method gradcam
  python explain.py --model-path vit.pth --images-dir imgs/ --method attention --model-type vit
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mammography.config import HP
from mammography.utils.common import (
    seed_everything,
    resolve_device,
    setup_logging,
    increment_path,
    configure_runtime,
)
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.io.dicom import dicom_to_pil_rgb, is_dicom_path
from mammography.models.nets import build_model
from mammography.tools import explain_registry
from mammography.vis.explainability import (
    GradCAMExplainer,
    ViTAttentionVisualizer,
    generate_explanations_batch,
    export_explanations_report,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_v2_F
from PIL import Image

DEFAULT_EXPLAIN_OUTPUT_DIR = Path("outputs/explanations")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate explainability visualizations for mammography model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GradCAM explanations for ResNet model
  python explain.py --model-path outputs/model.pth --images-dir data/test/ --model-type resnet50

  # Generate ViT attention maps
  python explain.py --model-path outputs/vit.pth --images-dir data/test/ --method attention --model-type vit

  # Generate both GradCAM and attention maps
  python explain.py --model-path outputs/model.pth --images-dir data/test/ --method both

  # Use specific output directory
  python explain.py --model-path model.pth --images-dir imgs/ --output-dir my_explanations/
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--images-dir", "-i",
        required=True,
        help="Directory containing input images or CSV with image paths",
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        default=str(DEFAULT_EXPLAIN_OUTPUT_DIR),
        help="Output directory for explanation visualizations (default: outputs/explanations)",
    )
    parser.add_argument(
        "--increment",
        action="store_true",
        help="Auto-increment output directory if it exists",
    )

    # Explanation method
    parser.add_argument(
        "--method",
        default="gradcam",
        choices=["gradcam", "attention", "both"],
        help="Explanation method to use (default: gradcam)",
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit"],
        help="Model architecture type (default: resnet50)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of output classes (default: 4 for BI-RADS)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=HP.NUM_WORKERS,
        help=f"DataLoader workers (default: {HP.NUM_WORKERS})",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=HP.IMG_SIZE,
        help=f"Image size for model input (default: {HP.IMG_SIZE})",
    )

    # Device and runtime
    parser.add_argument(
        "--device",
        default=HP.DEVICE,
        help=f"Device to use: cpu, cuda, mps, or auto (default: {HP.DEVICE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=HP.SEED,
        help=f"Random seed for reproducibility (default: {HP.SEED})",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for inference",
    )

    # Explanation options
    parser.add_argument(
        "--target-layer",
        help="Target layer name for GradCAM (default: auto-detect based on model type)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Overlay alpha for heatmap visualization (default: 0.4)",
    )
    parser.add_argument(
        "--colormap",
        default="jet",
        choices=["jet", "viridis", "plasma", "inferno", "turbo"],
        help="Colormap for heatmap visualization (default: jet)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comprehensive explanation report with grid visualizations",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default=HP.LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug"],
        help=f"Logging level (default: {HP.LOG_LEVEL})",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # Registry / MLflow
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "MLflow run name (default: derived from dataset/method, or output dir when customized)"
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        default="",
        help="MLflow tracking URI (optional)",
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="MLflow experiment name (optional)",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Local registry CSV path (default: results/registry.csv)",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Local registry Markdown path (default: results/registry.md)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip local registry updates",
    )

    return parser.parse_args(argv)


def load_model(
    model_path: str,
    model_type: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = build_model(
        arch=model_type,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def discover_images(images_path: str) -> List[str]:
    """Discover image files from directory or CSV."""
    path = Path(images_path)

    if path.is_file():
        # Assume CSV with image paths
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            # Try common column names
            for col in ["image_path", "path", "file_path", "dicom_path", "filename"]:
                if col in df.columns:
                    return df[col].tolist()
            raise ValueError(f"No image path column found in CSV: {path}")
        else:
            # Single image file
            return [str(path)]

    # Directory with images
    image_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(path.glob(f"**/*{ext}"))

    return [str(f) for f in sorted(image_files)]


def get_target_layer(model: torch.nn.Module, model_type: str, custom_layer: Optional[str] = None) -> torch.nn.Module:
    """Determine target layer for GradCAM based on model architecture."""
    if custom_layer:
        # Try to resolve custom layer name to actual module
        parts = custom_layer.split(".")
        layer = model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                raise ValueError(f"Layer '{custom_layer}' not found in model")
        return layer

    # Auto-detect based on model type
    if model_type.startswith("resnet"):
        if hasattr(model, "layer4"):
            return model.layer4[-1]  # Last block of layer4
        else:
            raise ValueError("ResNet model missing 'layer4' attribute")
    elif model_type.startswith("efficientnet"):
        if hasattr(model, "features"):
            return model.features[-1]  # Last feature layer
        else:
            raise ValueError("EfficientNet model missing 'features' attribute")
    else:
        raise ValueError(f"Unknown model type for GradCAM: {model_type}")


def load_images_as_tensors(
    image_paths: List[str],
    img_size: int,
    logger: logging.Logger,
) -> List[torch.Tensor]:
    """
    Load images from disk and convert to tensors.

    Follows the same preprocessing pipeline as MammoDensityDataset:
    - Read image (supports DICOM and regular images)
    - Convert to RGB
    - Resize and center crop
    - Convert to float32 tensor [0, 1]
    - Apply ImageNet normalization

    Args:
        image_paths: List of file paths to images
        img_size: Target image size (will be resized to img_size x img_size)
        logger: Logger for warnings

    Returns:
        List of preprocessed image tensors, each of shape (3, img_size, img_size)
    """
    # ImageNet normalization (default for pretrained models)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    tensors = []
    for path in image_paths:
        try:
            # Read image (supports both DICOM and regular images)
            if is_dicom_path(path):
                img = dicom_to_pil_rgb(path)
            else:
                img = Image.open(path).convert('RGB')

            # Convert to tensor and apply transformations
            # Following MammoDensityDataset._convert_to_tensor and _apply_transforms
            tensor = tv_v2_F.to_image(img)

            # Resize and center crop to square
            tensor = tv_v2_F.resize(tensor, [img_size], interpolation=InterpolationMode.BICUBIC, antialias=False)
            tensor = tv_v2_F.center_crop(tensor, [img_size, img_size])

            # Convert to float32 [0, 1]
            tensor = tv_v2_F.to_dtype(tensor, torch.float32, scale=True)

            # Apply normalization
            tensor = tv_v2_F.normalize(tensor, norm_mean, norm_std)

            tensors.append(tensor)

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            continue

    return tensors


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for explain command."""
    args = parse_args(argv)

    # Setup
    seed_everything(args.seed)

    if args.increment:
        output_dir = increment_path(args.output_dir)
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = "ERROR" if args.quiet else args.log_level.upper()
    logger = setup_logging(output_dir, log_level)

    device = resolve_device(args.device)
    configure_runtime(device, deterministic=False, allow_tf32=True)

    logger.info("=" * 80)
    logger.info("Explainability Visualization")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Images: {args.images_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # Validate inputs
    if not Path(args.model_path).exists():
        logger.error(f"Model path not found: {args.model_path}")
        return 1

    if not Path(args.images_dir).exists():
        logger.error(f"Images path not found: {args.images_dir}")
        return 1

    # Discover images
    try:
        image_paths = discover_images(args.images_dir)
        logger.info(f"Found {len(image_paths)} images to process")

        if len(image_paths) == 0:
            logger.error("No images found")
            return 1
    except Exception as e:
        logger.error(f"Failed to discover images: {e}")
        return 1

    # Load model
    try:
        logger.info("Loading model...")
        model = load_model(
            args.model_path,
            args.model_type,
            args.num_classes,
            device,
        )
        logger.info(f"Model loaded: {args.model_type}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    img_size = args.img_size
    if args.model_type == "vit":
        vit_model = model.backbone if hasattr(model, "backbone") else model
        vit_image_size = getattr(vit_model, "image_size", None)
        if isinstance(vit_image_size, (tuple, list)):
            vit_image_size = vit_image_size[0]
        if vit_image_size:
            vit_image_size = int(vit_image_size)
            if img_size != vit_image_size:
                logger.warning(
                    "Adjusting img_size from %s to %s to match ViT input size.",
                    img_size,
                    vit_image_size,
                )
                img_size = vit_image_size

    # Initialize explainers
    explainers: Dict[str, Any] = {}
    layer_name = ""

    try:
        if args.method in ["gradcam", "both"]:
            if args.model_type == "vit":
                logger.warning("GradCAM not supported for ViT models, skipping")
            else:
                target_layer = get_target_layer(model, args.model_type, args.target_layer)
                layer_name = (
                    target_layer.__class__.__name__
                    if hasattr(target_layer, "__class__")
                    else str(target_layer)
                )
                logger.info(f"Initializing GradCAM explainer (target layer: {layer_name})")
                explainers["gradcam"] = GradCAMExplainer(
                    model=model,
                    target_layer=target_layer,
                    device=device,
                )

        if args.method in ["attention", "both"]:
            if args.model_type != "vit":
                logger.warning("Attention maps only supported for ViT models, skipping")
            else:
                logger.info("Initializing ViT attention visualizer")
                explainers["attention"] = ViTAttentionVisualizer(
                    model=model,
                    device=device,
                )
    except Exception as e:
        logger.error(f"Failed to initialize explainers: {e}")
        return 1

    if not explainers:
        logger.error("No valid explainers initialized")
        return 1

    # Process images and generate explanations
    total_saved = 0
    total_failed = 0
    loaded_images_count = 0
    summary: Dict[str, Any] = {}
    report_images: List[torch.Tensor] | None = [] if args.generate_report else None

    try:
        logger.info("Generating explanations...")

        logged_explainers: set[str] = set()

        # Process images in batches to avoid loading everything into memory
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start:start + args.batch_size]
            if not batch_paths:
                continue

            batch_images = load_images_as_tensors(batch_paths, img_size, logger)
            if not batch_images:
                continue

            if report_images is not None:
                report_images.extend(batch_images)

            batch_offset = loaded_images_count
            loaded_images_count += len(batch_images)

            batch_images_tensor = torch.stack(batch_images)

            # Process each explainer type
            for explainer_name, explainer in explainers.items():
                if explainer_name not in logged_explainers:
                    logger.info(f"Generating {explainer_name} explanations...")
                    logged_explainers.add(explainer_name)

                # Determine explainer type
                if explainer_name == "gradcam":
                    explainer_type = "gradcam"
                elif explainer_name == "attention":
                    explainer_type = "vit_attention"
                else:
                    logger.warning(f"Unknown explainer name: {explainer_name}, skipping")
                    continue

                # Generate heatmaps using correct function signature
                heatmaps = generate_explanations_batch(
                    images=batch_images,              # List[torch.Tensor]
                    model=model,
                    explainer_type=explainer_type,
                    target_classes=None,               # Let model predict
                    device=device,
                    batch_size=args.batch_size,
                )

                # Save the heatmaps using explainer's save method
                output_subdir = output_dir / explainer_name
                output_subdir.mkdir(exist_ok=True)

                try:
                    # Replace None heatmaps with zeros
                    valid_heatmaps = [
                        hm if hm is not None else torch.zeros_like(batch_images[0][0])
                        for hm in heatmaps
                    ]
                    batch_heatmaps = torch.stack(valid_heatmaps)

                    # Save overlays
                    saved_count = explainer.save_batch_overlays(
                        batch_images_tensor,
                        batch_heatmaps,
                        output_dir=output_subdir,
                        alpha=args.alpha,
                        colormap=args.colormap,
                        index_offset=batch_offset,
                    )

                    logger.info(f"Saved {saved_count} {explainer_name} visualizations")
                    total_saved += saved_count

                    # Count failed
                    failed_count = len([hm for hm in heatmaps if hm is None])
                    total_failed += failed_count

                except Exception as exc:
                    logger.error(f"Failed to save {explainer_name} overlays: {exc}")
                    logger.exception(exc)
                    total_failed += len(batch_images)

        if loaded_images_count == 0:
            logger.error("No images loaded successfully")
            return 1

        # Save summary
        summary = {
            "total_images": len(image_paths),
            "loaded_images": loaded_images_count,
            "total_saved": total_saved,
            "total_failed": total_failed,
            "explainers": list(explainers.keys()),
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")

        logger.info(f"Generated explanations for {loaded_images_count} images")
        logger.info(f"Total saved: {total_saved}, Failed: {total_failed}")

    except Exception as e:
        logger.error(f"Failed to generate explanations: {e}")
        logger.exception(e)
        return 1

    # Generate comprehensive report if requested
    if args.generate_report:
        try:
            logger.info("Generating explanation report...")
            if report_images is None:
                logger.error("Report requested but no images were collected.")
                return 1
            report = export_explanations_report(
                images=report_images,            # Required
                model=model,                     # Required
                output_dir=output_dir / "report",
                metas=None,                      # Could extract from image_paths if needed
                target_classes=None,
                device=device,
                batch_size=args.batch_size,
                alpha=args.alpha,
            )
            logger.info(f"Report generated in: {report['output_dir']}")
            logger.info(f"Processed {report['num_images']} images")
            logger.info(f"Successful: {report['num_successful']}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            logger.exception(e)
            return 1

    if not args.no_registry:
        dataset = explain_registry.infer_dataset_name(Path(args.images_dir))
        if args.run_name:
            run_name = args.run_name
        elif output_dir != DEFAULT_EXPLAIN_OUTPUT_DIR:
            run_name = explain_registry.default_run_name_from_output(
                output_dir,
                args.method,
            )
        else:
            run_name = explain_registry.default_run_name(
                dataset,
                args.method,
            )
        if args.method in ["gradcam", "both"] and (output_dir / "gradcam").exists():
            explain_output = output_dir / "gradcam"
        elif args.method == "attention" and (output_dir / "attention").exists():
            explain_output = output_dir / "attention"
        else:
            explain_output = output_dir

        artifacts = []
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            artifacts.append(summary_path)

        artifact_dirs = []
        if explain_output.exists():
            artifact_dirs.append(explain_output)
        report_dir = output_dir / "report"
        if args.generate_report and report_dir.exists():
            artifact_dirs.append(report_dir)

        try:
            explain_registry.register_explain_run(
                input_path=Path(args.images_dir),
                output_dir=output_dir,
                output_path=explain_output,
                checkpoint_path=Path(args.model_path),
                arch=args.model_type,
                method=args.method,
                layer=layer_name,
                img_size=img_size,
                batch_size=args.batch_size,
                metrics=explain_registry.ExplainMetrics(
                    total_images=int(summary.get("total_images", len(image_paths))),
                    loaded_images=int(summary.get("loaded_images", loaded_images_count)),
                    saved_images=int(summary.get("total_saved", total_saved)),
                    failed_images=int(summary.get("total_failed", total_failed)),
                ),
                run_name=run_name,
                command=shlex.join(sys.argv),
                registry_csv=args.registry_csv,
                registry_md=args.registry_md,
                artifacts=artifacts,
                artifact_dirs=artifact_dirs,
                tracking_uri=args.tracking_uri or None,
                experiment=args.experiment or None,
                log_mlflow=not args.no_mlflow,
            )
            logger.info("Registry updated (run_name=%s).", run_name)
        except Exception as exc:
            logger.warning("Failed to register explain run: %s", exc)

    logger.info("=" * 80)
    logger.info("Explainability visualization complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
