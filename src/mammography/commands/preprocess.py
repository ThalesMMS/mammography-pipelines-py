#!/usr/bin/env python3
#
# preprocess.py
# mammography-pipelines
#
# DISCLAIMER: Educational project only - NOT for clinical use.
#
# Comprehensive dataset preprocessing command for normalization, resizing,
# cropping, and format conversion with preview and validation.
#
"""Preprocess mammography datasets with normalization, resizing, and format conversion."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from mammography.io.dicom import dicom_to_pil_rgb, is_dicom_path

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with normalization, resize, and format conversion."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory for preprocessed images",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["per-image", "per-dataset", "none"],
        default="per-image",
        help="Normalization method (default: per-image)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Target image size (default: 512)",
    )
    parser.add_argument(
        "--resize",
        dest="resize",
        action="store_true",
        default=True,
        help="Enable resizing (default: True)",
    )
    parser.add_argument(
        "--no-resize",
        dest="resize",
        action="store_false",
        help="Disable resizing",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Enable center cropping to square (default: False)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "keep"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Generate preview grid (default: False)",
    )
    parser.add_argument(
        "--preview-n",
        type=int,
        default=8,
        help="Number of images in preview grid (default: 8)",
    )
    parser.add_argument(
        "--report",
        dest="report",
        action="store_true",
        default=True,
        help="Generate validation report (default: True)",
    )
    parser.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="Disable validation report",
    )
    parser.add_argument(
        "--border-removal",
        action="store_true",
        default=False,
        help="Enable border removal (default: False)",
    )
    return parser.parse_args(argv)


def _iter_images(root: Path) -> List[Path]:
    """
    Iterate through directory and find all image files.

    Args:
        root: Root directory to search

    Returns:
        List of image file paths sorted by name
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")
    images: List[Path] = []

    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in valid_exts:
            images.append(item)

    images.sort()
    return images


def _load_image(path: Path) -> Image.Image:
    """
    Load image from DICOM or standard format.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode

    Raises:
        RuntimeError: If image cannot be loaded
    """
    try:
        if is_dicom_path(str(path)):
            return dicom_to_pil_rgb(str(path))
        else:
            return Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to load image {path}: {exc!r}") from exc


def _normalize_pil(
    img: Image.Image, method: str, stats: Optional[Dict[str, float]] = None
) -> Image.Image:
    """
    Apply normalization to PIL image.

    Args:
        img: Input PIL image
        method: Normalization method ('none', 'per-image', 'per-dataset')
        stats: Dataset statistics (required for 'per-dataset' method)

    Returns:
        Normalized PIL image
    """
    if method == "none":
        return img

    # Convert to numpy array
    arr = np.array(img, dtype=np.float32)

    if method == "per-image":
        # Per-image z-score normalization
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            arr = (arr - mean) / std
            # Rescale to [0, 255]
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr = np.zeros_like(arr)
        else:
            # Constant image - use min-max scaling
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr = np.zeros_like(arr)

    elif method == "per-dataset":
        # Per-dataset normalization using pre-computed stats
        if stats is None:
            raise ValueError("Dataset statistics required for per-dataset normalization")

        mean = stats["mean"]
        std = stats["std"]

        if std > 0:
            arr = (arr - mean) / std
            # Rescale to [0, 255] using robust percentile clipping
            p_low, p_high = np.percentile(arr, [0.5, 99.5])
            arr = np.clip(arr, p_low, p_high)
            if p_high > p_low:
                arr = (arr - p_low) / (p_high - p_low) * 255.0
            else:
                arr = np.zeros_like(arr)
        else:
            arr = np.zeros_like(arr)

    # Convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _compute_dataset_stats(paths: List[Path]) -> Dict[str, float]:
    """
    Compute mean and std over all images for per-dataset normalization.

    Args:
        paths: List of image paths

    Returns:
        Dictionary with 'mean' and 'std' keys
    """
    logger.info("Computing dataset statistics...")
    all_values: List[float] = []

    for path in tqdm(paths, desc="Computing stats"):
        try:
            img = _load_image(path)
            arr = np.array(img, dtype=np.float32)
            # Sample pixels to avoid memory issues with large datasets
            if arr.size > 1_000_000:
                # Sample 10% of pixels
                sample_size = arr.size // 10
                indices = np.random.choice(arr.size, size=sample_size, replace=False)
                values = arr.flat[indices].tolist()
            else:
                values = arr.flatten().tolist()
            all_values.extend(values)
        except Exception as exc:
            logger.warning(f"Failed to compute stats for {path}: {exc!r}")
            continue

    if not all_values:
        logger.warning("No valid images for statistics computation, using defaults")
        return {"mean": 127.5, "std": 1.0}

    mean = float(np.mean(all_values))
    std = float(np.std(all_values))

    logger.info(f"Dataset statistics: mean={mean:.2f}, std={std:.2f}")
    return {"mean": mean, "std": std}


def _resize_pil(img: Image.Image, img_size: int, crop: bool) -> Image.Image:
    """
    Resize image preserving aspect ratio, optionally center-crop to square.

    Args:
        img: Input PIL image
        img_size: Target size (longest side or square dimension)
        crop: If True, center-crop to square after resizing

    Returns:
        Resized (and optionally cropped) PIL image
    """
    if crop:
        # Resize and center-crop to square
        # First, resize so that the shorter side equals img_size
        width, height = img.size
        if width < height:
            new_width = img_size
            new_height = int(height * img_size / width)
        else:
            new_height = img_size
            new_width = int(width * img_size / height)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to square
        width, height = img.size
        left = (width - img_size) // 2
        top = (height - img_size) // 2
        right = left + img_size
        bottom = top + img_size
        img = img.crop((left, top, right, bottom))
    else:
        # Resize preserving aspect ratio (fit longest side)
        width, height = img.size
        if width > height:
            new_width = img_size
            new_height = int(height * img_size / width)
        else:
            new_height = img_size
            new_width = int(width * img_size / height)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def _save_preview(images: List[Image.Image], output_dir: Path, n: int) -> None:
    """
    Create preview grid from sample images.

    Args:
        images: List of processed images
        output_dir: Output directory
        n: Maximum number of images to include
    """
    if not images:
        logger.warning("No images available for preview")
        return

    # Take up to n images
    sample = images[:n]
    grid_size = int(np.ceil(np.sqrt(len(sample))))

    # Find max dimensions
    max_width = max(img.width for img in sample)
    max_height = max(img.height for img in sample)

    # Create grid
    grid_img = Image.new(
        "RGB",
        (grid_size * max_width, grid_size * max_height),
        color=(0, 0, 0),
    )

    for idx, img in enumerate(sample):
        row = idx // grid_size
        col = idx % grid_size
        x = col * max_width
        y = row * max_height
        grid_img.paste(img, (x, y))

    preview_path = output_dir / "preview_grid.png"
    grid_img.save(preview_path)
    logger.info(f"Preview grid saved to {preview_path}")


def _write_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    stats: Optional[Dict[str, float]],
) -> None:
    """
    Write validation report to JSON and text files.

    Args:
        results: List of processing results
        output_dir: Output directory
        stats: Dataset statistics (if computed)
    """
    # Prepare report data
    report = {
        "total_images": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "dataset_stats": stats,
        "results": results,
    }

    # Save JSON report
    json_path = output_dir / "preprocess_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to {json_path}")

    # Save text report
    txt_path = output_dir / "preprocess_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Preprocessing Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images: {report['total_images']}\n")
        f.write(f"Successful: {report['successful']}\n")
        f.write(f"Failed: {report['failed']}\n\n")

        if stats:
            f.write("Dataset Statistics:\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std: {stats['std']:.2f}\n\n")

        if report["failed"] > 0:
            f.write("Failed Images:\n")
            for r in results:
                if r["status"] == "failed":
                    f.write(f"  - {r['input_path']}: {r.get('error', 'Unknown error')}\n")

    logger.info(f"Text report saved to {txt_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """
    Main entry point for preprocessing command.

    Args:
        argv: Command-line arguments (None for sys.argv)
    """
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Convert paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    logger.info(f"Scanning {input_dir} for images...")
    image_paths = _iter_images(input_dir)

    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    logger.info(f"Found {len(image_paths)} images")

    # Compute dataset statistics if needed
    stats = None
    if args.normalize == "per-dataset":
        stats = _compute_dataset_stats(image_paths)

    # Process images
    results: List[Dict[str, Any]] = []
    preview_images: List[Image.Image] = []

    for path in tqdm(image_paths, desc="Preprocessing"):
        result: Dict[str, Any] = {
            "input_path": str(path),
            "status": "success",
        }

        try:
            # Load image
            img = _load_image(path)

            # Apply normalization
            if args.normalize != "none":
                img = _normalize_pil(img, args.normalize, stats)

            # Apply resizing
            if args.resize:
                img = _resize_pil(img, args.img_size, args.crop)

            # Determine output format
            if args.format == "keep":
                if is_dicom_path(str(path)):
                    out_ext = ".png"  # DICOM always converts to PNG
                else:
                    out_ext = path.suffix
            else:
                out_ext = f".{args.format}"

            # Determine output path (preserve relative structure)
            try:
                rel_path = path.relative_to(input_dir)
            except ValueError:
                # Path is not relative to input_dir, use just the filename
                rel_path = path.name

            out_path = output_dir / rel_path.with_suffix(out_ext)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save processed image
            if out_ext == ".jpg":
                img.save(out_path, quality=95)
            else:
                img.save(out_path)

            result["output_path"] = str(out_path)
            result["output_size"] = img.size

            # Collect for preview
            if args.preview and len(preview_images) < args.preview_n:
                preview_images.append(img.copy())

        except Exception as exc:
            logger.error(f"Failed to process {path}: {exc!r}")
            result["status"] = "failed"
            result["error"] = str(exc)

        results.append(result)

    # Save preview if requested
    if args.preview and preview_images:
        _save_preview(preview_images, output_dir, args.preview_n)

    # Write report if requested
    if args.report:
        _write_report(results, output_dir, stats)

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    logger.info(f"\nPreprocessing complete:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
