#!/usr/bin/env python3
#
# augment.py
# mammography-pipelines
#
# Simple dataset augmentation runner (DICOM + image files).
#
"""Augment images from a directory and save to a new folder."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

from mammography.io.dicom import dicom_to_pil_rgb, is_dicom_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment imagens em lote.")
    parser.add_argument("--source-dir", required=True, help="Diretorio de origem")
    parser.add_argument("--output-dir", required=True, help="Diretorio de saida")
    parser.add_argument("--num-augmentations", type=int, default=1, help="Augmentations por imagem")
    return parser.parse_args(argv)


def _iter_images(root: str) -> list[str]:
    files: list[str] = []
    for base, _, names in os.walk(root):
        for name in names:
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".dcm", ".dicom")):
                files.append(os.path.join(base, name))
    files.sort()
    return files


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_dir = args.source_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmenter = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )

    files = _iter_images(source_dir)
    if not files:
        raise SystemExit(f"Nenhuma imagem encontrada em {source_dir}.")

    for fpath in tqdm(files, desc="Augmenting"):
        try:
            ext = Path(fpath).suffix.lower()
            if is_dicom_path(fpath):
                img = dicom_to_pil_rgb(fpath)
                ext = ".png"
            else:
                img = Image.open(fpath).convert("RGB")

            stem = Path(fpath).stem
            img.save(output_dir / f"{stem}_orig{ext}")
            for idx in range(args.num_augmentations):
                aug = augmenter(img)
                aug.save(output_dir / f"{stem}_aug{idx}{ext}")
        except Exception as exc:
            print(f"[warn] Falha ao processar {fpath}: {exc}")

    print(f"[ok] Augmentacao concluida em {output_dir}")


if __name__ == "__main__":
    main()
