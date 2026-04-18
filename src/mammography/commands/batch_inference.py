#!/usr/bin/env python3
#
# batch_inference.py
# mammography-pipelines
#
# Optimized batch inference for processing large datasets with progress tracking,
# resumable processing, and output format options.
#
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
"""Batch inference with progress tracking and checkpoint/resume capabilities."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import IO, Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import ValidationError

from mammography.config import HP, BatchInferenceConfig
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.utils.class_modes import (
    CLASS_MODE_HELP,
    VISIBLE_CLASS_MODES_METAVAR,
    get_label_mapper,
    get_num_classes,
    parse_classes_mode_arg,
)
from mammography.utils.common import resolve_device, configure_runtime, parse_float_list
from mammography.io.dicom import is_dicom_path

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_file: str,
    processed_files: list[str],
    results: list[dict[str, Any]] | None = None,
) -> None:
    """
    Write checkpoint data to disk atomically and ensure temporary-file cleanup.

    Writes a JSON object containing `processed_files`, `timestamp`, and `num_processed`
    to a temporary file (checkpoint_file + ".tmp") and atomically replaces the target.
    An optional `results` list is stored when provided (for backward compatibility);
    streaming callers omit it since results are already on disk.

    Parameters:
        checkpoint_file (str): Path to the target checkpoint file to create or replace.
        processed_files (list[str]): List of file paths that have been processed.
        results (list[dict] | None): Optional per-file result dicts (legacy).
    """
    target = Path(checkpoint_file)
    tmp_path = Path(str(target) + ".tmp")

    checkpoint_data: dict[str, Any] = {
        "processed_files": processed_files,
        "timestamp": time.time(),
        "num_processed": len(processed_files),
    }
    if results is not None:
        checkpoint_data["results"] = results

    try:
        with open(tmp_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as exc:
                logger.debug("Failed to remove temp file %s: %s", tmp_path, exc)


def _iter_inputs(root: str) -> list[str]:
    """
    Collect image and DICOM file paths from a filesystem path.
    
    If `root` is a file, returns a single-element list containing that file.
    If `root` is a directory, recursively finds files with extensions .png, .jpg, .jpeg, .dcm, and .dicom and returns their paths sorted lexicographically.
    
    Parameters:
        root (str): File or directory path to search.
    
    Returns:
        list[str]: Sorted list of matching file paths.
    """
    if os.path.isfile(root):
        return [root]
    files: list[str] = []
    for base, _, names in os.walk(root):
        for name in names:
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".dcm", ".dicom")):
                files.append(os.path.join(base, name))
    files.sort()
    return files


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Remove the leading "module." prefix from every key in a state dictionary when all keys start with that prefix.
    
    Returns:
        dict: A new dictionary with the "module." prefix removed from each key if every key in the input started with "module."; otherwise returns the original input dictionary unchanged.
    """
    if not state_dict:
        return state_dict
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


class _ResultWriter:
    """Streams inference results to disk so memory stays flat regardless of dataset size.

    CSV and JSONL are written row-by-row; JSON must be accumulated in memory
    (the format is inherently non-streamable).
    """

    def __init__(
        self,
        path: Path,
        fmt: str,
        fieldnames: list[str],
        previous: list[dict[str, Any]],
        *,
        append: bool = False,
    ) -> None:
        """Open *path* for streaming output.

        Parameters:
            path: Destination file.
            fmt: One of ``"csv"``, ``"jsonl"``, or ``"json"``.
            fieldnames: Column names (used for CSV header).
            previous: Results carried over from a checkpoint (written first).
            append: When ``True``, open CSV/JSONL in append mode and skip the
                CSV header.  Ignored for JSON output.
        """
        self._path = path
        self._fmt = fmt
        self._file: IO[str] | None = None

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            mode = "a" if append else "w"
            self._file = open(path, mode, newline="")
            self._csv = csv.DictWriter(self._file, fieldnames=fieldnames)
            if not append:
                self._csv.writeheader()
            for row in previous:
                self._csv.writerow(row)
        elif fmt == "jsonl":
            mode = "a" if append else "w"
            self._file = open(path, mode)
            for row in previous:
                self._file.write(json.dumps(row) + "\n")
        else:  # json — must accumulate
            if append and path.exists():
                with open(path) as fh:
                    self._json_buf: list[dict[str, Any]] = json.load(fh)
                self._json_buf.extend(previous)
            else:
                self._json_buf = list(previous)

    # ------------------------------------------------------------------

    def write(self, record: dict[str, Any]) -> None:
        """Append a single result record to the output stream."""
        if self._fmt == "csv":
            assert self._file is not None
            self._csv.writerow(record)
        elif self._fmt == "jsonl":
            assert self._file is not None
            self._file.write(json.dumps(record) + "\n")
        else:
            self._json_buf.append(record)

    def flush(self) -> None:
        """Flush the underlying file handle, if open."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Finalize the output: write JSON buffer to disk or close the file handle."""
        if self._fmt == "json":
            with open(self._path, "w") as fh:
                json.dump(self._json_buf, fh, indent=2)
            self._json_buf.clear()
        elif self._file is not None:
            self._file.close()
            self._file = None


def resolve_loader_runtime(
    args: argparse.Namespace, device: torch.device
) -> tuple[int, int | None, bool, bool]:
    """
    Select DataLoader runtime parameters adjusted for the given device and CLI options.
    
    Parameters:
        args: Parsed CLI namespace exposing `num_workers`, `prefetch_factor`, `persistent_workers`, and `pin_memory`.
        device (torch.device): Target device used for inference.
    
    Returns:
        tuple: (num_workers, prefetch_factor, persistent_workers, pin_memory)
            - num_workers (int): Worker count to use for DataLoader (may be capped or set to 0 for some devices).
            - prefetch_factor (int | None): Prefetch factor to pass to DataLoader or `None` if not applicable.
            - persistent_workers (bool): Whether to keep DataLoader worker processes alive between epochs.
            - pin_memory (bool): Whether to enable pinned memory for host-to-device transfers.
    """
    num_workers = args.num_workers
    prefetch = args.prefetch_factor if args.prefetch_factor and args.prefetch_factor > 0 else None
    persistent = args.persistent_workers
    pin_memory = args.pin_memory

    # Device-specific adjustments for better performance
    if device.type == "mps":
        # MPS doesn't support multiprocessing well
        return 0, prefetch, False, pin_memory
    if device.type == "cpu":
        # Cap num_workers at CPU count
        return max(0, min(num_workers, os.cpu_count() or 0)), prefetch, persistent, pin_memory

    return num_workers, prefetch, persistent, pin_memory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for batch inference."""
    parser = argparse.ArgumentParser(
        description="Batch inference with progress tracking and checkpoint/resume."
    )

    # Model & Input
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory or file of images/DICOM to process",
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Model architecture (default: resnet50)",
    )
    parser.add_argument(
        "--classes",
        default="multiclass",
        type=parse_classes_mode_arg,
        metavar=VISIBLE_CLASS_MODES_METAVAR,
        help=CLASS_MODE_HELP,
    )

    # Processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=HP.IMG_SIZE,
        help=f"Image size for resizing (default: {HP.IMG_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=HP.DEVICE,
        help=f"Device to use (default: {HP.DEVICE})",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) on CUDA/MPS",
    )

    # DataLoader optimization
    parser.add_argument(
        "--num-workers",
        type=int,
        default=HP.NUM_WORKERS,
        help=f"DataLoader workers (default: {HP.NUM_WORKERS})",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=HP.PREFETCH_FACTOR,
        help=f"Prefetch factor (default: {HP.PREFETCH_FACTOR})",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=HP.PERSISTENT_WORKERS,
        help="Keep workers alive between batches",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=HP.PIN_MEMORY,
        help="Pin memory for faster GPU transfer (default: True)",
    )

    # Output
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for results",
    )
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json", "jsonl"],
        help="Output format (default: csv)",
    )

    # Checkpoint/Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="batch_inference_checkpoint.json",
        help="Checkpoint file for resume (default: batch_inference_checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N batches (default: 100)",
    )

    # Normalization
    parser.add_argument(
        "--mean",
        help="Normalization mean (e.g., 0.485,0.456,0.406)",
    )
    parser.add_argument(
        "--std",
        help="Normalization std (e.g., 0.229,0.224,0.225)",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for batch inference command."""
    args = parse_args(argv)

    # Validate configuration
    try:
        _config = BatchInferenceConfig.from_args(args)
    except ValidationError as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc
    args.classes = getattr(_config, "classes", args.classes)

    # Find all input files
    inputs = _iter_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No image files found in {args.input}")

    print(f"Found {len(inputs)} files to process", file=sys.stderr)

    # Parse normalization parameters
    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Determine number of classes and label mapper
    num_classes = get_num_classes(args.classes)
    mapper = get_label_mapper(args.classes)

    # Load checkpoint/resume state if requested
    processed_files: set[str] = set()
    previous_results: list[dict[str, Any]] = []

    if args.resume and os.path.exists(args.checkpoint_file):
        print(f"Resuming from checkpoint: {args.checkpoint_file}", file=sys.stderr)
        with open(args.checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
            processed_files = set(checkpoint_data.get("processed_files", []))
            previous_results = checkpoint_data.get("results", [])
        print(f"Skipping {len(processed_files)} already-processed files", file=sys.stderr)

    # Filter out already-processed files
    inputs = [path for path in inputs if path not in processed_files]

    if not inputs:
        print("All files already processed. Nothing to do.", file=sys.stderr)
        return 0

    # Prepare dataset rows
    rows = []
    for path in inputs:
        rows.append(
            {
                "image_path": path,
                "professional_label": None,
                "accession": (
                    os.path.basename(os.path.dirname(path))
                    if is_dicom_path(path)
                    else None
                ),
            }
        )

    # Setup device and runtime first to optimize DataLoader parameters
    device = resolve_device(args.device)
    configure_runtime(device, deterministic=False, allow_tf32=True)

    # Create dataset
    dataset = MammoDensityDataset(
        rows,
        img_size=args.img_size,
        train=False,
        cache_mode="none",
        split_name="batch_inference",
        label_mapper=mapper,
        mean=mean,
        std=std,
    )

    # Optimize DataLoader parameters for device type
    num_workers, prefetch_factor, persistent_workers, pin_memory = resolve_loader_runtime(args, device)

    # Create DataLoader with optimized parameters
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": mammo_collate,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers and num_workers > 0,
        "pin_memory": pin_memory and device.type == "cuda",
    }
    if num_workers > 0 and prefetch_factor:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Create DataLoader with fallback for multiprocessing errors
    try:
        loader = DataLoader(dataset, **loader_kwargs)
        # Test the loader by creating an iterator (will fail early if multiprocessing doesn't work)
        _ = iter(loader)
    except (PermissionError, RuntimeError) as exc:
        if num_workers == 0:
            raise
        print(
            f"Warning: Failed to create DataLoader with num_workers={num_workers} ({exc}). "
            f"Retrying with num_workers=0.",
            file=sys.stderr,
        )
        loader_kwargs["num_workers"] = 0
        loader_kwargs["persistent_workers"] = False
        loader_kwargs.pop("prefetch_factor", None)
        loader = DataLoader(dataset, **loader_kwargs)

    # Load model
    model = build_model(
        arch=args.arch,
        num_classes=num_classes,
        train_backbone=False,
        unfreeze_last_block=False,
        pretrained=False,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _strip_module_prefix(state)
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        logger.warning("Missing keys in checkpoint: %s", load_result.missing_keys)
    if load_result.unexpected_keys:
        logger.warning("Unexpected keys in checkpoint: %s", load_result.unexpected_keys)
    model.to(device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(f"Batch size: {args.batch_size}", file=sys.stderr)
    print(
        f"DataLoader: num_workers={loader_kwargs['num_workers']}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'N/A')}, "
        f"persistent_workers={loader_kwargs['persistent_workers']}, "
        f"pin_memory={loader_kwargs['pin_memory']}",
        file=sys.stderr,
    )
    print(f"Processing {len(inputs)} files...", file=sys.stderr)

    # Open streaming output writer (CSV/JSONL written row-by-row; JSON buffered)
    # When resuming from a new-style checkpoint (no stored results), append to the
    # existing output file so earlier results are not overwritten.
    fieldnames = ["file", "pred_class"] + [f"prob_{i}" for i in range(num_classes)]
    out_path = Path(args.output)
    append_mode = args.resume and not previous_results and out_path.exists()
    writer = _ResultWriter(
        out_path, args.output_format, fieldnames, previous_results, append=append_mode,
    )
    checkpoint_results = list(previous_results)
    del previous_results  # free memory — results are now on disk / in writer

    # Batch processing loop with tqdm progress bar
    processed_count = 0
    start_time = time.perf_counter()

    try:
        with torch.no_grad():
            pbar = tqdm(loader, desc="Batch Inference", unit="batch")
            for batch_idx, batch in enumerate(pbar, 1):
                if batch is None:
                    continue

                # Unpack batch
                if len(batch) == 4:
                    x, _y, meta_batch, _ = batch
                else:
                    x, _y, meta_batch = batch

                # Move to device with optimized memory format
                x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)

                # Forward pass with optional AMP
                with torch.autocast(device_type=device.type, enabled=args.amp):
                    logits = model(x)

                # Convert to probabilities
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

                # Stream results for this batch
                for i, meta in enumerate(meta_batch):
                    file_path = meta.get("image_path") or meta.get("path") or "unknown"
                    result = {
                        "file": file_path,
                        "pred_class": int(preds[i]),
                    }
                    # Add probability columns for each class
                    for cls_idx in range(probs.shape[1]):
                        result[f"prob_{cls_idx}"] = float(probs[i, cls_idx])

                    writer.write(result)
                    checkpoint_results.append(result)
                    processed_files.add(file_path)
                    processed_count += 1

                # Save checkpoint at specified intervals
                if args.checkpoint_interval > 0 and batch_idx % args.checkpoint_interval == 0:
                    writer.flush()
                    save_checkpoint(
                        args.checkpoint_file,
                        list(processed_files),
                        checkpoint_results,
                    )
                    print(
                        f"\nCheckpoint saved: {len(processed_files)} files processed",
                        file=sys.stderr,
                    )

                # Calculate and display throughput
                elapsed = time.perf_counter() - start_time
                throughput = processed_count / elapsed if elapsed > 0 else 0
                pbar.set_postfix(img_per_sec=f"{throughput:.1f}")
    finally:
        writer.close()

    # Final processing stats
    total_time = time.perf_counter() - start_time
    avg_throughput = processed_count / total_time if total_time > 0 else 0
    print(
        f"\nProcessed {processed_count} images in {total_time:.2f}s "
        f"({avg_throughput:.2f} images/sec)",
        file=sys.stderr,
    )

    # Save final checkpoint
    save_checkpoint(args.checkpoint_file, list(processed_files), checkpoint_results)
    print(
        f"Final checkpoint saved: {len(processed_files)} total files processed",
        file=sys.stderr,
    )

    fmt = args.output_format.upper()
    print(f"[ok] {fmt} saved to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    main()
