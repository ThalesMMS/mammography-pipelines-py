#!/usr/bin/env python3
#
# inference.py
# mammography-pipelines
#
# Run inference for trained density classifiers on images or DICOM folders.
#
"""Run inference with a trained EfficientNetB0/ResNet50 checkpoint."""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import ValidationError

from mammography.config import HP, InferenceConfig
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.utils.common import resolve_device, configure_runtime, parse_float_list
from mammography.io.dicom import is_dicom_path
from mammography.tools import inference_registry


def _iter_inputs(root: str) -> list[str]:
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


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia com checkpoint treinado.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Caminho para o checkpoint (.pt)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Arquivo ou diretorio de imagens/DICOM",
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
    )
    parser.add_argument(
        "--classes",
        default="multiclass",
        choices=["binary", "density", "multiclass"],
    )
    parser.add_argument("--img-size", type=int, default=HP.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=HP.DEVICE)
    parser.add_argument("--output", help="CSV de saida (opcional)")
    parser.add_argument("--amp", action="store_true", help="Usa autocast em CUDA/MPS")
    parser.add_argument(
        "--mean",
        help="Media de normalizacao (ex: 0.485,0.456,0.406)",
    )
    parser.add_argument(
        "--std",
        help="Std de normalizacao (ex: 0.229,0.224,0.225)",
    )
    parser.add_argument("--run-name", default="", help="Nome do run no MLflow")
    parser.add_argument("--tracking-uri", default="", help="Tracking URI para MLflow")
    parser.add_argument("--experiment", default="", help="Experimento MLflow")
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Nao registrar no MLflow",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        InferenceConfig.from_args(args)
    except ValidationError as exc:
        raise SystemExit(f"Config invalida: {exc}") from exc
    inputs = _iter_inputs(args.input)
    if not inputs:
        raise SystemExit(f"Nenhum arquivo encontrado em {args.input}.")

    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    num_classes = 2 if args.classes == "binary" else 4
    mapper = None
    if args.classes == "binary":
        def _mapper(y: int) -> int:
            if y in [1, 2]:
                return 0
            if y in [3, 4]:
                return 1
            return y - 1
        mapper = _mapper

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

    dataset = MammoDensityDataset(
        rows,
        img_size=args.img_size,
        train=False,
        cache_mode="none",
        split_name="inference",
        label_mapper=mapper,
        mean=mean,
        std=std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mammo_collate,
    )

    device = resolve_device(args.device)
    configure_runtime(device, deterministic=False, allow_tf32=True)

    model = build_model(
        arch=args.arch,
        num_classes=num_classes,
        train_backbone=False,
        unfreeze_last_block=False,
        pretrained=False,
    )
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    results: list[dict[str, object]] = []
    use_amp = args.amp and device.type in {"cuda", "mps"}

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, _, metas, _ = batch
            imgs = imgs.to(device)
            if use_amp:
                with torch.autocast(device.type, dtype=torch.float16):
                    logits = model(imgs)
            else:
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            for meta, pred, prob in zip(metas, preds, probs):
                row = {
                    "file": meta.get("path"),
                    "pred_class": int(pred),
                }
                for i, p in enumerate(prob.tolist()):
                    row[f"prob_{i}"] = float(p)
                results.append(row)
    duration_sec = time.perf_counter() - start_time

    df = pd.DataFrame(results)
    total_images = len(df)
    images_per_sec = (
        float(total_images) / duration_sec if duration_sec > 0 else 0.0
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[ok] CSV salvo em {out_path}")
        if not args.no_registry:
            dataset = inference_registry.infer_dataset_name(Path(args.input))
            run_name = args.run_name or inference_registry.default_run_name(
                dataset,
                args.arch,
            )
            try:
                inference_registry.register_inference_run(
                    input_path=Path(args.input),
                    output_path=out_path,
                    checkpoint_path=Path(args.checkpoint),
                    arch=args.arch,
                    classes=args.classes,
                    img_size=args.img_size,
                    batch_size=args.batch_size,
                    metrics=inference_registry.InferenceMetrics(
                        total_images=total_images,
                        images_per_sec=images_per_sec,
                        duration_sec=duration_sec,
                    ),
                    run_name=run_name,
                    command=shlex.join(sys.argv),
                    registry_csv=args.registry_csv,
                    registry_md=args.registry_md,
                    tracking_uri=args.tracking_uri or None,
                    experiment=args.experiment or None,
                    log_mlflow=not args.no_mlflow,
                )
                print(f"[ok] Registry atualizado (run_name={run_name}).")
            except Exception as exc:
                print(f"[warn] Falha ao registrar inferencia: {exc}")
    else:
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
