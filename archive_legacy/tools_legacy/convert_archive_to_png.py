#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_archive_to_png.py
----------------------------------
Converte todos os arquivos DICOM (.dcm) do diretório archive para PNG,
mantendo a mesma estrutura de pastas em archive_png.

Uso:
    python convert_archive_to_png.py
    python convert_archive_to_png.py --source-dir ./archive --output-dir ./archive_png
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
except ImportError:
    raise RuntimeError(
        "pydicom não está disponível. Instale com:\n"
        "  pip install pydicom\n"
        "Para DICOMs comprimidos, instale também:\n"
        "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
    )


def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Verifica se a imagem é MONOCHROME1 (preto-branco invertidos)."""
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    return photometric == "MONOCHROME1"


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Garante float32 para cálculos numéricos estáveis."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Aplica RescaleSlope/RescaleIntercept (quando presentes)."""
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    try:
        arr = arr * float(slope) + float(intercept)
    except Exception:
        try:
            arr = apply_modality_lut(arr, ds)
        except Exception:
            pass
    return arr


def robust_window(arr: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Windowing por percentis para padronizar contraste de mamografias."""
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


def dicom_to_pil_rgb(dcm_path: str) -> Image.Image:
    """Lê um DICOM de mamografia, aplica pré-processamento e retorna PIL Image RGB 8-bit."""
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array
    except Exception as e:
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. Se for DICOM comprimido, instale plugins:\n"
            "  pip install -q pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
            f"Erro original: {repr(e)}"
        )

    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)

    # MONOCHROME1: 0 = branco, valores altos = preto => inverte para ficar preto=0
    if _is_mono1(ds):
        arr = arr.max() - arr

    # Windowing robusto e conversão para uint8
    arr = robust_window(arr, 0.5, 99.5)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # Cria imagem em escala de cinza e replica para RGB
    pil = Image.fromarray(arr, mode="L")
    pil_rgb = Image.merge("RGB", (pil, pil, pil))
    return pil_rgb


def convert_dcm_to_png(dcm_path: str, png_path: str) -> bool:
    """Converte um arquivo DICOM para PNG."""
    try:
        pil_image = dicom_to_pil_rgb(dcm_path)
        pil_image.save(png_path, "PNG")
        return True
    except Exception as e:
        print(f"\nErro ao converter {dcm_path}: {e}", file=sys.stderr)
        return False


def find_all_dcm_files(source_dir: str) -> list[tuple[str, str]]:
    """
    Encontra todos os arquivos .dcm no diretório source_dir.
    Retorna lista de tuplas: (caminho_completo_dcm, caminho_relativo_para_estrutura)
    """
    source_path = Path(source_dir).resolve()
    dcm_files = []
    
    for dcm_file in source_path.rglob("*.dcm"):
        # Calcula o caminho relativo a partir do source_dir
        rel_path = dcm_file.relative_to(source_path)
        dcm_files.append((str(dcm_file), str(rel_path)))
    
    return sorted(dcm_files)


def main():
    parser = argparse.ArgumentParser(
        description="Converte arquivos DICOM do archive para PNG, mantendo estrutura de pastas"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="./archive",
        help="Diretório fonte com arquivos DICOM (padrão: ./archive)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./archive_png",
        help="Diretório de saída para PNGs (padrão: ./archive_png)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula arquivos PNG que já existem"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Verifica se o diretório fonte existe
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Erro: Diretório fonte '{source_dir}' não encontrado.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Procurando arquivos DICOM em: {source_dir}")
    print(f"Salvando PNGs em: {output_dir}")
    print()
    
    # Encontra todos os arquivos .dcm
    dcm_files = find_all_dcm_files(str(source_dir))
    
    if not dcm_files:
        print("Nenhum arquivo .dcm encontrado no diretório fonte.")
        sys.exit(0)
    
    print(f"Encontrados {len(dcm_files)} arquivo(s) DICOM para converter.\n")
    
    # Estatísticas
    converted = 0
    skipped = 0
    failed = 0
    
    # Processa cada arquivo
    for dcm_path, rel_path in tqdm(dcm_files, desc="Convertendo DICOMs", unit="arquivo"):
        # Calcula o caminho de saída mantendo a estrutura de pastas
        rel_path_obj = Path(rel_path)
        png_rel_path = rel_path_obj.with_suffix(".png")
        png_path = output_dir / png_rel_path
        
        # Verifica se já existe
        if args.skip_existing and png_path.exists():
            skipped += 1
            continue
        
        # Cria o diretório de destino se necessário
        png_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converte
        if convert_dcm_to_png(dcm_path, str(png_path)):
            converted += 1
        else:
            failed += 1
    
    # Resumo
    print("\n" + "=" * 50)
    print("Conversão concluída!")
    print("=" * 50)
    print(f"Total de arquivos processados: {len(dcm_files)}")
    print(f"  - Convertidos com sucesso: {converted}")
    if skipped > 0:
        print(f"  - Pulados (já existiam): {skipped}")
    if failed > 0:
        print(f"  - Falhas: {failed}")
    print(f"\nPNGs salvos em: {output_dir}")


if __name__ == "__main__":
    main()

