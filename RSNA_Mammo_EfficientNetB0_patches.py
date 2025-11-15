#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA_Mammo_EfficientNetB0_Density_ABxCD.py
----------------------------------
Treinamento/inferência com EfficientNetB0 para classificação binária de densidade mamária:
  - Classe 0: A/B (baixa densidade, BI-RADS 1-2)
  - Classe 1: C/D (alta densidade, BI-RADS 3-4)
Mapeia automaticamente labels originais (1,2,3,4) para binário (0,1) e extrai embeddings (1280-D) a partir de DICOMs.

Compatível diretamente com CSV `classificacao.csv` no formato:
  AccessionNumber,Classification,ClassificationDate
Onde `Classification` em {1,2,3,4,5}, e 5 é excluído por padrão (incidência não-padrão).
Também aceita CSV baseado em caminhos com colunas:
  image_path, density_label (A/B/C/D ou 0..3/1..4), professional_label (opcional), patient_id (opcional)

Saídas em `<outdir>/results[_k]`:
- train_history.csv
- metrics/val_metrics.json
- val_predictions.csv
- embeddings_val.csv, embeddings_val.npz

Cache reutiliza `<outdir>/cache` (compartilhado entre execuções).

Uso rápido:
    python RSNA_Mammo_EfficientNetB0_patches.py `
    --csv patches_completo `
    --epochs 20 `
    --batch-size 32 `
    --img-size 128 `
    --lr 1e-4 `
    --backbone-lr 1e-5 `
    --outdir outputs/mammo_efficientnetb0_patches `
    --device auto `
    --amp `
    --unfreeze-last-block `
    --class-weights auto `
    --cache-mode auto

Mapeamento de labels:
  - 1,2 (A,B) → 0 (baixa densidade)
  - 3,4 (C,D) → 1 (alta densidade)

"""

from __future__ import annotations
import os
import sys
import json
import argparse
import time
import warnings
import contextlib
import logging
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from tqdm.contrib.logging import TqdmLoggingHandler
except ImportError:
    TqdmLoggingHandler = logging.StreamHandler

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_v2_F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit

# Matplotlib para salvar figuras (sem janela)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("mammo_efficientnetb0_density")
LOGGER.addHandler(logging.NullHandler())

CACHE_AUTO_DISK_MAX = 6000  # acima disso, materializar todas as PILs pode ocupar muito tempo/disk
CACHE_AUTO_MEMORY_MAX = 1000  # limite simples para cache em RAM quando já temos PNGs/JPGs
DICOM_EXTS = (".dcm", ".dicom")


# ------------------------- Utilidades básicas -------------------------

# Hiperparâmetros didáticos (ponto único de ajuste)
#
# Ajuste aqui para afetar o comportamento padrão do script. Os valores também
# são usados como defaults nos argumentos de linha de comando.
class HP:
    """Hiperparâmetros centrais deste pipeline.

    - IMG_SIZE: tamanho da imagem de entrada ao modelo (após resize+crop). 224–512 são comuns.
    - WINDOW_P_LOW/HIGH: percentis do windowing robusto para DICOM (contraste). 0.5/99.5 é um bom padrão.
    - EPOCHS / BATCH_SIZE / NUM_WORKERS: treino e carregamento de dados.
    - LR / BACKBONE_LR: taxas de aprendizado (head e últimos blocos de features/backbone).
    - VAL_FRAC: fração de validação no split estratificado.
    - SEED: reprodutibilidade (split, inicializações).
    - DEVICE: 'auto' escolhe CUDA/MPS/CPU automaticamente.
    - UNFREEZE_LAST_BLOCK / TRAIN_BACKBONE: fine-tuning do backbone.
    - CLASS_WEIGHTS / SAMPLER_WEIGHTED / WARMUP_EPOCHS: anti-desbalanceamento (ajudam classes raras).
    Nota: normalização usa stats do ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    pois o backbone foi treinado assim; é o ponto de partida mais estável.
    """

    IMG_SIZE: int = 512 # tamanho da imagem de entrada ao modelo (após resize+crop). 224–512 são comuns.
    WINDOW_P_LOW: float = 0.5 # percentis do windowing robusto para DICOM (contraste)
    WINDOW_P_HIGH: float = 99.5 # percentis do windowing robusto para DICOM (contraste)
    EPOCHS: int = 100 # número de épocas de treino (colocar umas 20)
    BATCH_SIZE: int = 16 # tamanho de batch para a inferência (em 16 deu 4.5 it/s pra nw 10 pf 12)
    NUM_WORKERS: int = 4 # workers do DataLoader (0=sem multiprocess)
    LR: float = 1e-4 # taxas de aprendizado (head e últimos blocos de features/backbone)
    BACKBONE_LR: float = 1e-5 # taxas de aprendizado (head e últimos blocos de features/backbone)
    VAL_FRAC: float = 0.20 # fração de validação no split estratificado
    SEED: int = 42 # reprodutibilidade (split, inicializações)
    DEVICE: str = "auto" # 'auto' escolhe CUDA/MPS/CPU automaticamente
    UNFREEZE_LAST_BLOCK: bool = True # padrão: descongela últimos blocos de features (equivalente ao layer4); use --no-unfreeze-last-block para congelar
    TRAIN_BACKBONE: bool = False # se True, treina o backbone inteiro (não recomendado; use apenas para experimentação)
    CLASS_WEIGHTS: str = "none"  # 'none' | 'auto'
    SAMPLER_WEIGHTED: bool = False # se True, usa sampling ponderado para classes raras
    WARMUP_EPOCHS: int = 0 # número de épocas de warmup para o learning rate
    DETERMINISTIC: bool = False # modo determinístico pode reduzir performance; habilite se precisar reprodutibilidade
    ALLOW_TF32: bool = True # TF32 acelera matmuls em GPUs NVIDIA Ampere+/Ada
    PREFETCH_FACTOR: int = 4 # quantas batches antecipar por worker
    PERSISTENT_WORKERS: bool = True # mantém workers vivos entre épocas (evita overhead de spawn)
    CACHE_MODE: str = "auto"  # 'auto' | 'disk' | 'memory' | 'none'
    LOG_LEVEL: str = "info" # nível padrão de log (info/debug/warning/...)
    TRAIN_AUGMENT: bool = True # aplica flips/rotações leves no dataset de treino
    LOADER_HEURISTICS: bool = True # aplica heurísticas automáticas de num_workers/prefetch/persistência
    FUSED_OPTIM: bool = False # habilita AdamW fused (CUDA) quando disponível
    TORCH_COMPILE: bool = False # envolve o modelo em torch.compile para otimização adicional
    EARLY_STOP_PATIENCE: int = 0 # 0 desativa early stopping
    EARLY_STOP_MIN_DELTA: float = 0.0 # melhoria mínima em val_acc para resetar paciência
    LR_REDUCE_PATIENCE: int = 0 # 0 desativa ReduceLROnPlateau
    LR_REDUCE_FACTOR: float = 0.5
    LR_REDUCE_MIN_LR: float = 1e-7
    LR_REDUCE_COOLDOWN: int = 0

def seed_everything(seed: int = 42, deterministic: bool = False):
    """Deixa o treinamento mais reprodutível (mesmos splits e inicializações)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    if deterministic:
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def resolve_device(device_choice: str) -> torch.device: # escolhe o device automaticamente (CUDA > MPS > CPU), ou respeita a escolha explícita.
    """Escolhe o device automaticamente (CUDA > MPS > CPU), ou respeita a escolha explícita."""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_choice == "mps":
        return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if device_choice == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def configure_runtime(device: torch.device, deterministic: bool, allow_tf32: bool) -> None:
    """Ajusta flags de backend para equilíbrio entre performance e reprodutibilidade."""
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if not deterministic:
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
    elif device.type == "mps":
        if not deterministic:
            try:
                torch.set_float32_matmul_precision("medium")
            except AttributeError:
                pass


def setup_logging(outdir: str, level: str) -> logging.Logger:
    """Configura logging para console + arquivo."""
    log_dir = outdir
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("mammo_efficientnetb0_density")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_level = getattr(logging, level.upper(), logging.INFO)
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def _increment_path(path: str) -> str:
    """Se `path` já existir, retorna `path_1`, `path_2`, ... até encontrar livre.

    Ex.: "outputs/mammo_efficientnetb0_density" → "outputs/mammo_efficientnetb0_density_1" se a primeira existir.
    """
    base = path.rstrip("/\\")
    if not os.path.exists(base):
        return base
    i = 1
    while True:
        cand = f"{base}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Detecta fotometria MONOCHROME1 (preto=alto, branco=baixo) para inverter tons.

    Em MONOCHROME1, valores altos são escuros. Para visualização/treino
    convencionais, invertemos a escala (preto=0, branco=alto).
    """
    return getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1"


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Garante dtype float32 antes de operações aritméticas (estável e compatível com torch)."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Aplica RescaleSlope/Intercept (ou LUT) para trazer pixels à escala correta."""
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
    """Windowing por percentis: recorta extremos e normaliza para [0,1].
    Em mamografia, reduz saturação do fundo e estabiliza contraste do parênquima.
    """
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


def dicom_to_pil_rgb(dcm_path: str) -> Image.Image:
    """Lê DICOM, aplica rescale/intercept, inverte MONOCHROME1, windowing e converte para RGB 8-bit.
    Mantemos 3 canais replicados para compatibilidade com modelos pré-treinados no ImageNet (EfficientNetB0, ResNet50, etc).
    """
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
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr = robust_window(arr, HP.WINDOW_P_LOW, HP.WINDOW_P_HIGH)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    # Pillow passa a inferir o modo automaticamente; evitar 'mode' remove warning futuro
    pil = Image.fromarray(arr)
    pil_rgb = Image.merge("RGB", (pil, pil, pil))
    return pil_rgb


# ---------------------- Cache helpers ----------------------

def _is_dicom_path(path: str) -> bool:
    """Checa extensão contra `DICOM_EXTS`; ajuste aqui centraliza qualquer novo sufixo."""
    return str(path).lower().endswith(DICOM_EXTS)


def resolve_dataset_cache_mode(requested_mode: str, df: pd.DataFrame) -> str:
    """Resolve modo de cache considerando heurísticas simples quando --cache-mode=auto."""

    mode = (requested_mode or "none").lower()
    if mode != "auto":
        return mode

    if "image_path" not in df.columns:
        return "none"
    paths = [str(p) for p in df["image_path"].tolist() if pd.notna(p)]
    if not paths:
        return "none"

    total = len(paths)
    dicom_mask = [_is_dicom_path(p) for p in paths]
    if all(dicom_mask):
        # DICOMs: cache em disco reduz custo de decode, mas limite simples evita explosion
        if total > CACHE_AUTO_DISK_MAX:
            return "none"
        return "disk"

    # PNG/JPG/etc: cache em memória só para datasets pequenos
    if total <= CACHE_AUTO_MEMORY_MAX:
        return "memory"
    return "none"


# ---------------------- Leitura do CSV/paths ----------------------

def _find_first_dicom(folder: str) -> Optional[str]:
    """Varre a pasta do exame e retorna o primeiro arquivo DICOM encontrado.

    Didático: usamos um representante por exame (primeiro na ordem lexicográfica)
    para acelerar. Se quiser usar todas as incidências, adapte para retornar a
    lista completa e ajustar o Dataset para iterar sobre elas.
    """
    exts = (".dcm", ".dicom", ".DCM", ".DICOM")
    dicoms: List[str] = []
    for curr, _, files in os.walk(folder):
        for f in files:
            fp = os.path.join(curr, f)
            lower = f.lower()
            if fp.endswith(exts) or lower.endswith(".dcm") or lower.endswith(".dicom"):
                dicoms.append(fp)
    dicoms.sort()
    return dicoms[0] if dicoms else None


def _read_classificacao_csv(csv_path: str) -> pd.DataFrame:
    """Lê classificacao.csv garantindo AccessionNumber como string (preserva zeros à esquerda)."""
    df = pd.read_csv(csv_path, dtype={"AccessionNumber": str})
    # Normaliza e garante colunas
    if "Classification" not in df.columns:
        raise ValueError("CSV precisa de coluna 'Classification'.")
    df["AccessionNumber"] = df["AccessionNumber"].str.strip()
    return df


def _df_from_classificacao_with_paths(df: pd.DataFrame, dicom_root: str, exclude_class_5: bool = True) -> pd.DataFrame:
    """Mapeia AccessionNumber -> caminho do primeiro DICOM encontrado sob --dicom-root/<Accession>.
    Class 5 é excluída por padrão (incidência não-padrão).
    """
    rows = []
    for _, r in df.iterrows():
        acc = str(r["AccessionNumber"]).strip()
        lab = int(r["Classification"]) if pd.notna(r["Classification"]) else None
        if lab == 5 and exclude_class_5:
            continue
        folder = os.path.join(dicom_root, acc)
        if not os.path.isdir(folder):
            continue
        dcm = _find_first_dicom(folder)
        if dcm is None:
            continue
        rows.append({
            "accession": acc,
            "image_path": dcm,
            "professional_label": lab,
        })
    out = pd.DataFrame(rows)
    return out


def _map_to_binary_label(label_1_4: Optional[int]) -> Optional[int]:
    """Mapeia label de densidade 4-classes (1..4) para binário (0..1).
    
    - 1,2 (A,B) → 0 (baixa densidade)
    - 3,4 (C,D) → 1 (alta densidade)
    - None → None
    """
    if label_1_4 is None or (isinstance(label_1_4, float) and np.isnan(label_1_4)):
        return None
    label_1_4 = int(label_1_4)
    if label_1_4 in {1, 2}:
        return 0  # A/B (baixa densidade)
    elif label_1_4 in {3, 4}:
        return 1  # C/D (alta densidade)
    return None


def _coerce_density_label(val: Any) -> Optional[int]:
    """Converte rótulos diversos para {1,2,3,4} (A,B,C,D ou 0..3/1..4).

    Retorna None quando não reconhecido.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        s = val.strip().upper()
        if s in {"A", "B", "C", "D"}:
            return {"A": 1, "B": 2, "C": 3, "D": 4}[s]
        try:
            ival = int(s)
            if ival in {0, 1, 2, 3}:
                return ival + 1
            return ival
        except Exception:
            return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    try:
        return int(val)
    except Exception:
        return None


def _df_from_path_csv(csv_path: str) -> pd.DataFrame:
    """Aceita CSV baseado em caminhos. Tenta inferir rótulos em colunas comuns e cria 'accession'."""
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError("CSV baseado em caminhos requer coluna 'image_path'.")
    # tenta densidade em diversas colunas
    label_col_candidates = ["density_label", "label", "y", "professional_label"]
    lab_col = next((c for c in label_col_candidates if c in df.columns), None)
    if lab_col is None:
        df["professional_label"] = None
    else:
        df["professional_label"] = df[lab_col].apply(_coerce_density_label)
    # accession opcional
    if "AccessionNumber" in df.columns:
        df["accession"] = df["AccessionNumber"].astype(str)
    elif "accession" not in df.columns:
        df["accession"] = [os.path.basename(os.path.dirname(p)) for p in df["image_path"]]
    return df[["image_path", "professional_label", "accession"]]


def _df_from_patches_txt(dataset_root: str) -> pd.DataFrame:
    """Lê dataset patches_completo a partir dos arquivos .txt na raiz.
    
    O arquivo featureS.txt tem formato alternado:
    nome_patch
    classe
    nome_patch
    classe
    ...
    
    Os nomes no arquivo não têm extensão, mas os arquivos reais têm .png.
    Mapeia classes 0-3 para BI-RADS 1-4:
    - 0 → 1 (BI-RADS 1, baixa densidade)
    - 1 → 2 (BI-RADS 2, baixa densidade)
    - 2 → 3 (BI-RADS 3, alta densidade)
    - 3 → 4 (BI-RADS 4, alta densidade)
    """
    feature_path = os.path.join(dataset_root, "featureS.txt")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Arquivo featureS.txt não encontrado em {dataset_root}")
    
    rows = []
    try:
        with open(feature_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Processar linhas alternadas: nome, classe, nome, classe...
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            patch_name = lines[i]
            try:
                class_raw = int(lines[i + 1])
            except (ValueError, IndexError):
                continue
            
            # Mapear classe 0-3 → 1-4 (BI-RADS)
            if class_raw in {0, 1, 2, 3}:
                birads_label = class_raw + 1
            else:
                continue
            
            # Adicionar extensão .png se não tiver
            if not patch_name.endswith('.png'):
                patch_name = patch_name + '.png'
            
            # Construir caminho completo
            image_path = os.path.join(dataset_root, patch_name)
            if not os.path.exists(image_path):
                continue
            
            # Derivar accession do nome do patch (ex: p_d_left_cc(10) → p_d_left_cc)
            accession = patch_name.split('(')[0] if '(' in patch_name else patch_name.rsplit('.', 1)[0]
            
            rows.append({
                "accession": accession,
                "image_path": image_path,
                "professional_label": birads_label,
            })
    except Exception as exc:
        raise RuntimeError(f"Falha ao processar {feature_path}: {exc}")
    
    if not rows:
        raise RuntimeError(f"Nenhuma amostra encontrada em {dataset_root}. Verifique se o arquivo featureS.txt está no formato correto.")
    
    return pd.DataFrame(rows)


# ---------------------- Dataset/Transforms ----------------------

class MammoDensityDataset(Dataset):
    """Dataset minimalista para densidade mamária.

    Pipeline por amostra:
      1) Lê imagem (DICOM → dicom_to_pil_rgb; PNG/JPG → RGB direto)
      2) (opcional) Cacheia a PIL em disco/RAM conforme cache_mode
      3) Aplica Resize/Crop para `img_size`
      4) (treino) Augmentations leves: flip horizontal e rotação ±5°
      5) Converte para tensor e normaliza com estatísticas do ImageNet
      6) Converte rótulo 1..4 para 0..3 (CrossEntropyLoss)
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        img_size: int,
        train: bool,
        augment: bool = True,
        cache_mode: str = "none",
        cache_dir: Optional[str] = None,
        split_name: str = "train",
    ):
        """Inicializa o dataset.

        - `rows`: amostras já filtradas/normalizadas pelo pipeline de CSV.
        - `img_size`: lado final (quadrado) após resize+crop central.
        - `train`: ativa branch de augment/estatísticas para treino.
        - `augment`: ativa flips/rotações quando `train=True`.
        - `cache_mode`: controla materialização rápida da base:
            * `none`: decode on demand.
            * `memory`: guarda PILs decodificados em RAM.
            * `disk`: salva DICOMs como PNG em `cache_dir`.
            * `tensor-disk`: persiste tensores brutos `.pt` (um por amostra).
            * `tensor-memmap`: persiste tensores em `.dat` + `.json` (memory-mapped).
            Em `auto`, quem decide é `resolve_dataset_cache_mode`.
        - `cache_dir`: raiz onde artefatos de cache serão criados (necessário para modos em disco).
        - `split_name`: apenas para etiquetar logs/progress bars.

        Os modos baseados em disco materializam arquivos na inicialização; `tensor-*`
        precomputam tensores apenas uma vez, evitando decodificações repetidas.
        """
        self.rows = rows
        self.img_size = img_size
        self.cache_mode = (cache_mode or "none").lower()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name

        valid_cache_modes = {"none", "memory", "disk", "tensor-disk", "tensor-memmap"}
        if self.cache_mode not in valid_cache_modes:
            raise ValueError(f"cache_mode inválido: {self.cache_mode}")
        if self.cache_mode in {"disk", "tensor-disk", "tensor-memmap"} and self.cache_dir is None:
            raise ValueError("cache_dir é obrigatório quando cache_mode requer persistência em disco")

        self._image_cache: Optional[Dict[str, Image.Image]] = {} if self.cache_mode == "memory" else None
        self._disk_cache_index: Dict[str, str] = {}
        self._tensor_disk_index: Dict[str, str] = {}
        self._tensor_memmap_index: Dict[str, Dict[str, Any]] = {}
        if self.cache_mode in {"disk", "tensor-disk", "tensor-memmap"}:
            assert self.cache_dir is not None
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_mode == "disk":
            self._prepare_disk_cache()
        elif self.cache_mode in {"tensor-disk", "tensor-memmap"}:
            self._prepare_tensor_cache()

        self.train = train
        self.augment = bool(augment and train)
        self._norm_mean = [0.485, 0.456, 0.406]
        self._norm_std = [0.229, 0.224, 0.225]

    def __len__(self):
        # Número de amostras já incorpora filtros prévios (ex.: linhas descartadas por falta de DICOM).
        return len(self.rows)

    def _read_image(self, path: str) -> Image.Image:
        """Decodifica a imagem base a partir do caminho.

        - DICOM: aplica pipeline dicom_to_pil_rgb (rescale, invert MONOCHROME1, windowing)
        - PNG/JPG: carrega arquivo e converte para RGB
        """
        if _is_dicom_path(path):
            return dicom_to_pil_rgb(path)
        return Image.open(path).convert("RGB")

    def _cache_path_for(self, path: str) -> Path:
        """Gera nome baseado em SHA1 para evitar colisão ao achatar paths em cache PNG."""
        assert self.cache_dir is not None
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.png"

    def _tensor_cache_base_path(self, path: str) -> Path:
        """Base (sem sufixo) derivada de SHA1; garante layout estável para `.pt` ou memmap."""
        assert self.cache_dir is not None
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return self.cache_dir / h

    def _prepare_disk_cache(self) -> None:
        """Percorre DICOMs → decodifica → salva PNG cacheados; leave=False recolhe o tqdm após concluir."""
        assert self.cache_dir is not None
        iterable = self.rows
        for row in tqdm(
            iterable,
            desc=f"Cache[{self.split_name}]",
            leave=False,
            disable=len(iterable) < 16,
        ):
            path = str(row.get("image_path"))
            if not path or not _is_dicom_path(path):
                continue
            cache_path = self._cache_path_for(path)
            self._disk_cache_index[path] = str(cache_path)
            if cache_path.exists():
                continue
            try:
                img = self._read_image(path)
            except Exception as exc:
                LOGGER.warning("Falha ao materializar cache de %s: %s", path, exc)
                continue
            try:
                img.save(cache_path, format="PNG")
            except Exception as exc:
                LOGGER.warning("Falha ao salvar cache em %s: %s", cache_path, exc)

    def _prepare_tensor_cache(self) -> None:
        """Materializa tensores antecipadamente (.pt em tensor-disk; .dat/.json em tensor-memmap) e indexa os caminhos."""
        assert self.cache_dir is not None
        iterable = self.rows
        for row in tqdm(
            iterable,
            desc=f"TensorCache[{self.split_name}]",
            leave=False,
            disable=len(iterable) < 16,
        ):
            path = str(row.get("image_path"))
            if not path:
                continue
            if self.cache_mode == "tensor-disk":
                cache_path = self._tensor_cache_base_path(path).with_suffix(".pt")
                self._tensor_disk_index[path] = str(cache_path)
                if cache_path.exists():
                    continue
            elif self.cache_mode == "tensor-memmap":
                base = self._tensor_cache_base_path(path)
                data_path = base.with_suffix(".dat")
                meta_path = base.with_suffix(".json")
                self._tensor_memmap_index[path] = {
                    "data_path": str(data_path),
                    "meta_path": str(meta_path),
                }
                if data_path.exists() and meta_path.exists():
                    continue
            try:
                tensor = self._decode_to_tensor(path)
            except Exception as exc:
                LOGGER.warning("Falha ao decodificar %s para o cache de tensores: %s", path, exc)
                continue
            self._materialize_tensor_cache(path, tensor)

    def _decode_to_tensor(self, path: str) -> torch.Tensor:
        img = self._read_image(path)
        return self._convert_to_tensor(img)

    def _materialize_tensor_cache(self, path: str, tensor: torch.Tensor) -> None:
        """Persiste tensores conforme modo: `.pt` exclusivo ou par memmap; branch final vazio é apenas para clareza."""
        if self.cache_mode == "tensor-disk":
            cache_path = self._tensor_cache_base_path(path).with_suffix(".pt")
            self._tensor_disk_index[path] = str(cache_path)
            if cache_path.exists():
                return
            arr = tensor.detach().cpu().numpy()
            try:
                tensor_to_save = torch.from_numpy(np.array(arr, copy=True))
                torch.save(tensor_to_save, cache_path)
            except Exception as exc:
                LOGGER.warning("Falha ao salvar tensor cache em %s: %s", cache_path, exc)
        elif self.cache_mode == "tensor-memmap":
            base = self._tensor_cache_base_path(path)
            data_path = base.with_suffix(".dat")
            meta_path = base.with_suffix(".json")
            self._tensor_memmap_index[path] = {
                "data_path": str(data_path),
                "meta_path": str(meta_path),
            }
            if data_path.exists() and meta_path.exists():
                return
            arr = tensor.detach().cpu().numpy()
            try:
                mm = np.memmap(data_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
                mm[:] = arr
                mm.flush()
                meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                meta_path.write_text(json.dumps(meta))
                del mm
            except Exception as exc:
                LOGGER.warning("Falha ao salvar memmap em %s: %s", data_path, exc)
        else:
            # fallback para modos que não usam cache de tensor
            pass

    def _get_base_image(self, path: str) -> Image.Image:
        """Retorna uma PIL Image com cache opcional em memória para acelerar IO/decodificação.

        cache_mode='memory': armazena PIL em `_image_cache`.
        cache_mode='disk': reutiliza PNGs materializados durante a inicialização.
        """
        if self.cache_mode == "disk":
            cache_path = self._disk_cache_index.get(path)
            if cache_path and os.path.exists(cache_path):
                with Image.open(cache_path) as im:
                    return im.convert("RGB")
            img = self._read_image(path)
            if cache_path and _is_dicom_path(path):
                try:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    img.save(cache_path, format="PNG")
                except Exception as exc:
                    LOGGER.debug("Não foi possível atualizar cache em %s: %s", cache_path, exc)
            return img

        if self.cache_mode == "memory":
            if self._image_cache is None:
                self._image_cache = {}
            cached = self._image_cache.get(path)
            if cached is None:
                cached = self._read_image(path)
                self._image_cache[path] = cached
            return cached.copy()

        return self._read_image(path)

    def _get_cached_tensor(self, path: str) -> Optional[torch.Tensor]:
        """Recupera tensor cacheado (disco ou memmap) em CPU; se arquivo faltar/corromper, retorna None para regenerar."""
        if self.cache_mode == "tensor-disk":
            cache_path = self._tensor_disk_index.get(path)
            if cache_path is None and self.cache_dir is not None:
                cache_path = str(self._tensor_cache_base_path(path).with_suffix(".pt"))
                self._tensor_disk_index[path] = cache_path
            if cache_path and os.path.exists(cache_path):
                try:
                    tensor = torch.load(cache_path, map_location="cpu")
                    return tensor
                except Exception as exc:
                    LOGGER.debug("Falha ao carregar tensor cache de %s: %s", cache_path, exc)
                    return None
        elif self.cache_mode == "tensor-memmap":
            info = self._tensor_memmap_index.get(path)
            if info is None and self.cache_dir is not None:
                base = self._tensor_cache_base_path(path)
                info = {
                    "data_path": str(base.with_suffix(".dat")),
                    "meta_path": str(base.with_suffix(".json")),
                }
                self._tensor_memmap_index[path] = info
            if info:
                data_path = info.get("data_path")
                meta_path = info.get("meta_path")
                if data_path and meta_path and os.path.exists(data_path) and os.path.exists(meta_path):
                    try:
                        if "meta" not in info:
                            meta = json.loads(Path(meta_path).read_text())
                            info["meta"] = meta
                        else:
                            meta = info["meta"]
                        shape = tuple(meta["shape"])
                        dtype = np.dtype(meta["dtype"])
                        mm = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
                        arr = np.array(mm, copy=True)
                        return torch.from_numpy(arr)
                    except Exception as exc:
                        LOGGER.debug("Falha ao carregar memmap de %s: %s", data_path, exc)
                        return None
        return None

    def __getitem__(self, i: int):
        """Retorna `(tensor_normalizado, label_binário, metadados)` e materializa o cache sob demanda quando ausente.
        
        Labels são mapeados de 1..4 (A,B,C,D) para 0..1 (AB,CD):
        - 1,2 (A,B) → 0 (baixa densidade)
        - 3,4 (C,D) → 1 (alta densidade)
        """
        r = self.rows[i]
        base_tensor = self._get_cached_tensor(r["image_path"])
        if base_tensor is None:
            img = self._get_base_image(r["image_path"])
            base_tensor = self._convert_to_tensor(img)
            if self.cache_mode in {"tensor-disk", "tensor-memmap"}:
                self._materialize_tensor_cache(r["image_path"], base_tensor)
        img = self._apply_transforms(base_tensor)
        img = self._to_channels_last(img)
        y = r.get("professional_label")
        # map 1..4 -> 0..1 (binário: AB vs CD)
        y = _map_to_binary_label(y)
        return img, y, r

    @staticmethod
    def _convert_to_tensor(img: Image.Image) -> torch.Tensor:
        """Converte PIL Image para tensor no formato esperado pelos transforms v2."""
        return tv_v2_F.to_image(img)

    @staticmethod
    def _to_channels_last(tensor: torch.Tensor) -> torch.Tensor:
        """Garante layout channels_last para tensores 3D/4D."""
        if tensor.ndim == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last).squeeze(0)
        return tensor

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        # Resize ajusta a menor aresta para `img_size`, mantendo o lado mais longo >= img_size e permitindo crop central sem distorção.
        tensor = tv_v2_F.resize(
            tensor,
            [self.img_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=False,
        )
        tensor = tv_v2_F.center_crop(tensor, [self.img_size, self.img_size])
        tensor = tv_v2_F.to_dtype(tensor, torch.float32, scale=True)
        if self.train and getattr(self, "augment", True):
            # Flips e rotações ±5° mantêm plausibilidade clínica evitando inversões exageradas.
            if float(torch.rand(1, device=tensor.device)) < 0.5:
                tensor = tv_v2_F.horizontal_flip(tensor)
            angle = float(torch.empty(1, device=tensor.device).uniform_(-5.0, 5.0))
            tensor = tv_v2_F.rotate(
                tensor,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0,
            )
        tensor = tv_v2_F.normalize(tensor, self._norm_mean, self._norm_std)
        return tensor


def _collate(batch):
    """Collate simples que mantém metadados como lista de dicts (não tensores)."""
    xs = torch.stack([b[0] for b in batch], dim=0)
    xs = MammoDensityDataset._to_channels_last(xs)
    ys = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return xs, ys, metas


def make_splits(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split estratificado por rótulo binário (0=AB, 1=CD) agrupado por accession para evitar vazamento.
    
    Mapeia labels originais 1..4 para binário 0..1 antes do split estratificado.
    Agrupa por 'accession' para garantir que todas as imagens de um mesmo paciente/exame
    fiquem no mesmo split (treino ou validação), evitando vazamento de dados.
    Se faltar alguma classe binária, cai para split aleatório agrupado.
    """
    rng = np.random.RandomState(seed)
    y_orig = df["professional_label"].values
    y_orig = np.array([l for l in y_orig])
    idx = np.arange(len(df))
    # filtra apenas rótulos 1..4 e mapeia para binário
    mask = np.isin(y_orig, [1,2,3,4])
    idx_candidates = idx[mask]
    y_binary_list = []
    idx_valid = []
    groups_list = []
    for i, orig_label in zip(idx_candidates, y_orig[mask]):
        binary_label = _map_to_binary_label(int(orig_label))
        if binary_label is not None:
            y_binary_list.append(binary_label)
            idx_valid.append(i)
            # Usar accession como grupo para evitar vazamento
            groups_list.append(df.iloc[i]["accession"])
    y_labeled_binary = np.array(y_binary_list)
    idx_labeled = np.array(idx_valid)
    groups = np.array(groups_list)
    
    if len(np.unique(y_labeled_binary)) < 2:
        # fallback: split simples agrupado por accession
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        cut = int((1.0 - val_frac) * len(unique_groups))
        train_groups = set(unique_groups[:cut])
        train_idx = [i for i, g in enumerate(groups) if g in train_groups]
        val_idx = [i for i, g in enumerate(groups) if g not in train_groups]
        return [idx_labeled[i] for i in train_idx], [idx_labeled[i] for i in val_idx]
    
    # Usar GroupShuffleSplit para garantir que accessions inteiros fiquem no mesmo split
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    for tr, va in gss.split(idx_labeled, y_labeled_binary, groups=groups):
        return idx_labeled[tr].tolist(), idx_labeled[va].tolist()
    return idx_labeled.tolist(), [].tolist()


# ---------------------- Modelo/treino ----------------------

def build_model(num_classes: int = 2, train_backbone: bool = False, unfreeze_last_block: bool = True) -> nn.Module:
    """Cria EfficientNetB0 pré-treinada, troca o 'classifier' por uma cabeça de 2 saídas (binário: AB vs CD).
    
    Por padrão: congela blocos iniciais de features e descongela últimos blocos + classifier.
    - unfreeze_last_block=True (padrão): descongela últimos blocos de features (equivalente ao layer4)
    - unfreeze_last_block=False: congela últimos blocos também (apenas classifier treina)
    - train_backbone=True: descongela todo o backbone (não recomendado; apenas para experimentação)
    
    EfficientNetB0 tem 7 blocos principais em features. Por padrão, descongelamos os últimos 2 blocos
    (equivalente ao layer4 do ResNet50).
    """
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Troca o classifier (Dropout + Linear) por uma nova cabeça de classificação
    in_features = m.classifier[1].in_features  # o Linear está em classifier[1]
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    # congela todas as camadas por padrão
    for p in m.parameters():
        p.requires_grad = False
    
    # sempre treina a head nova (classifier)
    for p in m.classifier.parameters():
        p.requires_grad = True
    
    # por padrão, descongela últimos blocos de features (equivalente ao layer4)
    # EfficientNetB0 tem 7 blocos principais: descongelamos os últimos 2 (índices -2 e -1)
    if unfreeze_last_block:
        # Últimos 2 blocos do features (equivalente ao layer4 do ResNet50)
        num_blocks = len(m.features)
        for i in range(num_blocks - 2, num_blocks):  # últimos 2 blocos
            for p in m.features[i].parameters():
                p.requires_grad = True
    
    # apenas se explicitamente solicitado, descongela todo o backbone
    if train_backbone:
        for p in m.parameters():
            p.requires_grad = True
    return m


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extrai features 1280-D da 'avgpool' (penúltima camada) via forward-hook.
    Útil para visualização (UMAP/t-SNE), busca de vizinhos e modelos tabulares.
    Mantém o modo AMP alinhado ao treino, convertendo as features para float32 antes de salvar.
    O hook acumula as ativações em `buffer`; a função concatena e limpa esse buffer a cada batch.
    """
    model.eval()
    feats: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    handle = None
    buffer: List[np.ndarray] = []

    def hook_fn(module, inp, out):
        # out shape: [B, 1280, 1, 1] para EfficientNetB0
        v = out.detach().float().cpu().numpy()
        v = v.reshape(v.shape[0], -1)
        buffer.append(v)

    # hook na avgpool
    handle = model.avgpool.register_forward_hook(hook_fn)

    for batch in tqdm(loader, desc="Embeddings", leave=False):
        x, y, metas = batch
        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x)  # forward para acionar hook
        feat = np.concatenate(buffer, axis=0)
        buffer.clear()  # limpa o acumulado do hook para o próximo batch
        feats.append(feat)
        rows.extend(list(metas))

    if handle is not None:
        handle.remove()

    return np.concatenate(feats, axis=0), rows


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_fn: Optional[nn.Module] = None,
    profiler=None,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
):
    """Uma época de treino: forward -> loss -> backward -> step. Mede loss média e acc do batch.
    Suporta AMP (autocast + GradScaler em CUDA, apenas autocast em MPS) para reduzir uso de memória em GPUs.
    Se 'loss_fn' vier com pesos de classe, classes raras influenciam mais a atualização.
    """
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    if amp_enabled and device.type not in {"cuda", "mps"}:
        raise ValueError(f"AMP habilitado, mas não suportado no device {device}.")
    use_scaler = scaler is not None
    if amp_enabled and device.type == "cuda" and not use_scaler:
        raise ValueError("AMP em CUDA requer GradScaler disponível.")
    logger = logging.getLogger("mammo_efficientnetb0_density")
    log_per_iter = logger.isEnabledFor(logging.DEBUG)
    last_step_end = time.perf_counter()

    for step, (x, y, _) in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        iter_start = time.perf_counter()
        # Tempos a seguir alimentam apenas logs DEBUG, ajudando a identificar gargalos de IO, transferência ou backward.
        data_wait = iter_start - last_step_end

        to_device_start = time.perf_counter()
        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        to_device_time = time.perf_counter() - to_device_start

        zero_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        zero_time = time.perf_counter() - zero_start

        forward_start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = loss_fn(logits, y_tensor)
        forward_time = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        if use_scaler:
            scaler.scale(loss).backward()
            backward_time = time.perf_counter() - backward_start

            step_start = time.perf_counter()
            scaler.step(optimizer)
            step_time = time.perf_counter() - step_start

            update_start = time.perf_counter()
            scaler.update()
            scaler_update_time = time.perf_counter() - update_start
        else:
            loss.backward()
            backward_time = time.perf_counter() - backward_start

            step_start = time.perf_counter()
            optimizer.step()
            step_time = time.perf_counter() - step_start
            scaler_update_time = 0.0

        iter_total = time.perf_counter() - iter_start
        last_step_end = time.perf_counter()

        if log_per_iter:
            logger.debug(
                "iter=%04d | wait=%.2fms | to_device=%.2fms | zero=%.2fms | forward=%.2fms | backward=%.2fms | step=%.2fms | scaler_update=%.2fms | total=%.2fms",
                step,
                data_wait * 1e3,
                to_device_time * 1e3,
                zero_time * 1e3,
                forward_time * 1e3,
                backward_time * 1e3,
                step_time * 1e3,
                scaler_update_time * 1e3,
                iter_total * 1e3,
            )

        losses.append(loss.item())
        pred = logits.detach().float().argmax(dim=1)
        correct += (pred == y_tensor).sum().item()
        total += y_tensor.numel()
        if profiler is not None:
            profiler.step()
    acc = correct / max(1, total)
    return float(np.mean(losses) if losses else 0.0), acc


@torch.no_grad()
def validate(model, loader, device, *, amp_enabled: bool = False) -> Dict[str, Any]:
    """Validação binária: acumula predições e probabilidades, calcula métricas e retorna linhas detalhadas.
    
    Labels binários: 0=AB (baixa densidade), 1=CD (alta densidade).
    Respeita o modo AMP para consistência, recastando logits para float32 antes das métricas.
    """
    model.eval()
    all_y = []
    all_p = []
    all_prob = []
    rows = []
    for x, y, metas in tqdm(loader, desc="Val", leave=False):
        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
        logits = logits.float()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        all_prob.append(probs)
        all_p.append(pred)
        all_y.append(np.array(y))
        rows.extend(list(metas))

    y_true = np.concatenate(all_y).astype(int)
    y_pred = np.concatenate(all_p).astype(int)
    prob = np.concatenate(all_prob, axis=0)

    # métricas binárias (0=AB, 1=CD)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        # Para binário, usa apenas a classe positiva (CD)
        auc = roc_auc_score(y_true, prob[:, 1])
    except Exception:
        auc = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0, target_names=["AB", "CD"])

    # salva predições detalhadas
    out_rows = []
    for i, m in enumerate(rows):
        pr = prob[i]
        out_rows.append({
            **m,
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "y_true_label": "AB" if y_true[i] == 0 else "CD",
            "y_pred_label": "AB" if y_pred[i] == 0 else "CD",
            "p_AB": float(pr[0]),
            "p_CD": float(pr[1]),
        })

    return {
        "acc": acc,
        "kappa": kappa,
        "auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "val_rows": out_rows,
    }


def save_metrics_figure(metrics: Dict[str, Any], out_path: str) -> None:
    """Gera uma figura com matriz de confusão e barras de precisão/recall/F1 por classe (binário).
    - metrics: dicionário retornado por validate()
    - out_path: caminho do PNG de saída (ex.: .../metrics/val_metrics.png)
    """
    try:
        cm = np.array(metrics.get("confusion_matrix", [[0,0],[0,0]]), dtype=float)
        report = metrics.get("classification_report", {})
        acc = float(metrics.get("acc", 0.0))
        kappa = float(metrics.get("kappa", 0.0))
        auc = metrics.get("auc", None)

        labels = ["AB", "CD"]  # binário: baixa densidade vs alta densidade
        cls_names = ["AB", "CD"]

        def _get(rep: Dict[str, Any], cls_name: str, key: str) -> float:
            # Helper evita KeyError quando o relatório estiver incompleto
            return float(rep.get(cls_name, {}).get(key, 0.0))

        prec = [ _get(report, c, "precision") for c in cls_names ]
        rec  = [ _get(report, c, "recall")    for c in cls_names ]
        f1   = [ _get(report, c, "f1-score")  for c in cls_names ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion matrix
        ax = axes[0]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Matriz de Confusão (True x Pred)")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        ax.set_xticks(range(2), labels)
        ax.set_yticks(range(2), labels)
        # anotações
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Barras por classe
        ax = axes[1]
        x = np.arange(2)
        w = 0.25
        ax.bar(x - w, prec, width=w, label="Precisão")
        ax.bar(x,       rec,  width=w, label="Recall")
        ax.bar(x + w,   f1,   width=w, label="F1")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, labels)
        ax.set_title("Métricas por classe")
        ax.legend()

        # Título geral com métricas globais
        auc_txt = f"{auc:.3f}" if isinstance(auc, (float, np.floating)) else "NA"
        fig.suptitle(f"Acc={acc:.3f} | Kappa={kappa:.3f} | AUC={auc_txt}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        # Não quebra o fluxo se falhar; apenas registra alerta
        LOGGER.warning("Falha ao salvar figura de métricas: %s", e)


def save_history_figure(hist_rows: List[Dict[str, Any]], out_path: str) -> None:
    """Gera uma figura com curvas de treino/val (loss/acc/kappa/AUC) acumuladas até o momento.

    - hist_rows: lista de dicionários como em train_history.csv
    - out_path: caminho do PNG gerado (ex.: .../train_history.png)
    """
    try:
        if not hist_rows:
            return
        dfh = pd.DataFrame(hist_rows)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss (treino)
        ax = axes[0]
        ax.plot(dfh["epoch"], dfh["train_loss"], label="train_loss", color="#1f77b4")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_title("Curva de Loss (treino)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Acc/Kappa/AUC (val)
        ax = axes[1]
        ax.plot(dfh["epoch"], dfh["val_acc"], label="val_acc", color="#2ca02c")
        if "val_kappa" in dfh.columns:
            ax.plot(dfh["epoch"], dfh["val_kappa"], label="kappa", color="#d62728")
        elif "val_kappa_quadratic" in dfh.columns:
            ax.plot(dfh["epoch"], dfh["val_kappa_quadratic"], label="kappa", color="#d62728")
        if "val_auc" in dfh.columns:
            try:
                ax.plot(dfh["epoch"], dfh["val_auc"], label="auc", color="#9467bd")
            except Exception:
                pass
        elif "val_auc_ovr" in dfh.columns:
            try:
                ax.plot(dfh["epoch"], dfh["val_auc_ovr"], label="auc", color="#9467bd")
            except Exception:
                pass
        ax.set_xlabel("Época")
        ax.set_ylabel("Métrica")
        ax.set_title("Validação: Acc / Kappa / AUC")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        LOGGER.warning("Falha ao salvar curva de treino: %s", e)


def _resolve_loader_runtime(args: argparse.Namespace, device: torch.device) -> Tuple[Dict[str, Any], List[str]]:
    """Aplica heurísticas para configurar DataLoader (num_workers/prefetch/persistência).

    Retorna um dicionário com os parâmetros efetivos e uma lista de mensagens explicando
    os ajustes feitos. Mantém `args` como fonte da intenção original do usuário, mas garante
    que o DataLoader receba combinações válidas independentemente da plataforma.
    """

    platform_name = sys.platform.lower()
    heuristics_enabled = bool(getattr(args, "loader_heuristics", True))
    num_workers = max(0, int(getattr(args, "num_workers", 0) or 0))
    prefetch_factor = getattr(args, "prefetch_factor", None)
    if prefetch_factor is not None and prefetch_factor <= 0:
        prefetch_factor = None
    persistent_workers = bool(getattr(args, "persistent_workers", False))
    preset = getattr(args, "preset", "auto") or "auto"
    notes: List[str] = []

    def disable_workers(reason: str) -> None:
        # Normaliza num_workers/prefetch/persistent em um único lugar quando heurísticas exigem carregamento síncrono.
        nonlocal num_workers, prefetch_factor, persistent_workers
        if num_workers != 0 or persistent_workers or prefetch_factor:
            notes.append(reason)
        num_workers = 0
        prefetch_factor = None
        persistent_workers = False

    if heuristics_enabled:
        if preset == "windows":
            if num_workers > 2:
                notes.append("preset=windows → num_workers limitado a 2")
                num_workers = 2
            if num_workers > 0 and (prefetch_factor is None or prefetch_factor > 2):
                notes.append("preset=windows → prefetch_factor ajustado para 2")
                prefetch_factor = 2
            persistent_workers = num_workers > 0
        elif preset == "mps":
            disable_workers("preset=mps → carregamento síncrono (num_workers=0)")
        else:
            if platform_name.startswith("win"):
                if num_workers > 2:
                    notes.append("Plataforma Windows detectada → num_workers limitado a 2")
                    num_workers = 2
                if num_workers > 0 and (prefetch_factor is None or prefetch_factor > 2):
                    notes.append("Plataforma Windows detectada → prefetch_factor ajustado para 2")
                    prefetch_factor = 2
            if device.type == "mps":
                if getattr(args, "mps_sync_loaders", False):
                    disable_workers("flag --mps-sync-loaders ativa carregamento síncrono")
                elif num_workers > 2:
                    notes.append("Device MPS detectado → num_workers limitado a 2")
                    num_workers = 2
    else:
        notes.append("heurísticas de DataLoader desativadas (--no-loader-heuristics)")

    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    pin_memory = device.type == "cuda"
    cfg = dict(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    return cfg, notes


def run(args: argparse.Namespace):
    """Orquestra o pipeline de ponta a ponta (dados -> modelo -> métricas -> embeddings)."""
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    configure_runtime(device, args.deterministic, args.allow_tf32)
    amp_capable = device.type in {"cuda", "mps"}
    amp_requested = args.amp if getattr(args, "amp", None) is not None else amp_capable
    amp_enabled = bool(amp_requested and amp_capable)
    setattr(args, "amp_requested", bool(amp_requested))
    setattr(args, "amp_effective", amp_enabled)
    scaler = (
        GradScaler(device=device.type)
        if amp_enabled and device.type == "cuda"
        else None
    )
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)
    results_base = outdir_root / "results"
    if getattr(args, "auto_increment_outdir", False):
        results_dir = Path(_increment_path(str(results_base)))
    else:
        results_dir = results_base
    args.outdir_base = str(outdir_root)
    args.outdir = str(results_dir)
    # A raiz (`outdir_base`) guarda cache compartilhado; `outdir` aponta para o diretório incrementado da execução atual.
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "metrics"), exist_ok=True)
    logger = setup_logging(args.outdir, args.log_level)
    logger.info("Resultados serão gravados em: %s", args.outdir)
    logger.info("Iniciando execução | torch=%s | torchvision=%s", torch.__version__, torchvision.__version__)
    logger.debug("Argumentos recebidos: %s", vars(args))
    if amp_requested and not amp_enabled:
        logger.warning(
            "AMP solicitado, mas o device %s não oferece suporte; mantendo execução em precisão total.",
            device,
        )
    elif amp_enabled and getattr(args, "amp", None) is None:
        logger.info(
            "AMP habilitado automaticamente para %s; use --no-amp para desativar.",
            device,
        )
    logger.info(
        "Device selecionado: %s | deterministic=%s | allow_tf32=%s | amp=%s",
        device,
        args.deterministic,
        args.allow_tf32,
        "on" if amp_enabled else "off",
    )
    loader_cfg, loader_notes = _resolve_loader_runtime(args, device)
    setattr(args, "num_workers_effective", loader_cfg["num_workers"])
    setattr(args, "prefetch_factor_effective", loader_cfg["prefetch_factor"])
    setattr(args, "persistent_workers_effective", loader_cfg["persistent_workers"])
    logger.info(
        "DataLoader (efetivo) -> batch_size=%d | num_workers=%d | persistent_workers=%s | prefetch_factor=%s | pin_memory=%s | preset=%s",
        args.batch_size,
        loader_cfg["num_workers"],
        loader_cfg["persistent_workers"],
        loader_cfg["prefetch_factor"] if loader_cfg["prefetch_factor"] is not None else "NA",
        loader_cfg["pin_memory"],
        getattr(args, "preset", "auto"),
    )
    for note in loader_notes:
        logger.info("  ↳ %s", note)
    if not loader_notes:
        logger.info("  ↳ nenhuma heurística adicional aplicada")

    profiler = None
    if args.profile:
        try:
            from torch.profiler import profile as torch_profile, ProfilerActivity, tensorboard_trace_handler, schedule
            os.makedirs(args.profile_dir, exist_ok=True)
            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            profiler = torch_profile(
                activities=activities,
                schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=tensorboard_trace_handler(args.profile_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            logger.info("Profiler ativado; traces em %s", args.profile_dir)
        except Exception as exc:
            profiler = None
            logger.warning("Não foi possível inicializar o profiler: %s", exc)

    # 1) Carrega dados (CSV ou diretório do dataset)
    if os.path.isdir(args.csv):
        # Verificar se é dataset patches_completo (featureS.txt na raiz)
        feature_path_root = os.path.join(args.csv, "featureS.txt")
        if os.path.exists(feature_path_root):
            logger.info("Detectado dataset patches_completo (featureS.txt na raiz)")
            df = _df_from_patches_txt(args.csv)
        else:
            raise ValueError(
                f"Diretório '{args.csv}' não contém arquivo featureS.txt na raiz. "
                "Esperado: diretório patches_completo com featureS.txt na raiz."
            )
    elif {"AccessionNumber", "Classification"}.issubset(set(pd.read_csv(args.csv, nrows=0).columns)):
        df_cls = _read_classificacao_csv(args.csv)
        df = _df_from_classificacao_with_paths(df_cls, args.dicom_root, exclude_class_5=not args.include_class_5)
    else:
        df = _df_from_path_csv(args.csv)

    if df.empty:
        raise RuntimeError("Nenhuma linha válida encontrada após mapear caminhos a partir do CSV.")

    # 2) Filtra labels 1..4 e prepara para mapeamento binário
    df = df.copy()
    df = df[df["professional_label"].isin([1,2,3,4])]
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Após excluir classe 5, o CSV ficou vazio.")
    logger.info("Total de amostras utilizáveis: %d", len(df))
    # Log distribuição original e binária
    dist_orig = df["professional_label"].value_counts().sort_index().to_dict()
    df_binary_temp = df.copy()
    df_binary_temp["binary_label"] = df_binary_temp["professional_label"].apply(_map_to_binary_label)
    dist_binary = df_binary_temp["binary_label"].value_counts().sort_index().to_dict()
    logger.info("Distribuição de rótulos originais (1-4): %s", dist_orig)
    logger.info("Distribuição de rótulos binários (0=AB, 1=CD): %s", dist_binary)

    resolved_cache_mode = resolve_dataset_cache_mode(args.cache_mode, df)
    setattr(args, "cache_mode_resolved", resolved_cache_mode)
    dicom_count = int(df["image_path"].map(_is_dicom_path).sum())
    if args.cache_mode == "auto":
        logger.info(
            "Cache mode auto → %s (dicom=%d/%d)",
            resolved_cache_mode,
            dicom_count,
            len(df),
        )
    elif resolved_cache_mode != args.cache_mode:
        logger.info("Cache mode ajustado de %s para %s", args.cache_mode, resolved_cache_mode)
    else:
        logger.info("Cache mode: %s", resolved_cache_mode)
    cache_required = {"disk", "tensor-disk", "tensor-memmap"}
    cache_root = outdir_root / "cache" if resolved_cache_mode in cache_required else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        logger.info("Cache compartilhado em: %s", cache_root)

    # 3) Split treino/val estratificado
    tr_idx, va_idx = make_splits(df, val_frac=args.val_frac, seed=args.seed)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)
    logger.info("Split estratificado -> treino=%d | validação=%d", len(df_tr), len(df_va))

    # 4) Datasets/Loaders
    rows_tr = df_tr.to_dict(orient="records")
    rows_va = df_va.to_dict(orient="records")
    cache_train = str(cache_root / "train") if cache_root is not None else None
    cache_val = str(cache_root / "val") if cache_root is not None else None
    ds_tr = MammoDensityDataset(
        rows_tr,
        img_size=args.img_size,
        train=True,
        augment=args.train_augment,
        cache_mode=resolved_cache_mode,
        cache_dir=cache_train,
        split_name="train",
    )
    ds_va = MammoDensityDataset(
        rows_va,
        img_size=args.img_size,
        train=False,
        augment=False,
        cache_mode=resolved_cache_mode,
        cache_dir=cache_val,
        split_name="val",
    )
    # Sanidade: garantir que estamos usando nossa subclasse (e não a base Dataset)
    import torch.utils.data as tud
    if type(ds_tr) is tud.Dataset:
        raise RuntimeError("Instância de Dataset base detectada (sem __getitem__). Verifique a definição de MammoDensityDataset.")
    # Sampler ponderado opcional: balanceia batches favorecendo classes raras
    loader_common = dict(num_workers=loader_cfg["num_workers"], pin_memory=loader_cfg["pin_memory"], collate_fn=_collate)
    if loader_cfg["num_workers"] > 0:
        loader_common["persistent_workers"] = loader_cfg["persistent_workers"]
        if loader_cfg["prefetch_factor"] is not None:
            loader_common["prefetch_factor"] = loader_cfg["prefetch_factor"]
    dl_tr_kwargs = dict(batch_size=args.batch_size, **loader_common)
    if args.sampler_weighted:
        # constrói pesos por amostra com base na frequência de classe binária no treino (0=AB, 1=CD)
        ys = [_map_to_binary_label(r.get("professional_label")) for r in rows_tr]
        ys_binary = [y for y in ys if y is not None]
        counts = {c: max(1, ys_binary.count(c)) for c in range(2)}
        import torch.utils.data as tud
        sample_weights = [1.0 / counts[y] for y in ys_binary]
        sampler = tud.WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        dl_tr = DataLoader(ds_tr, sampler=sampler, shuffle=False, **dl_tr_kwargs)
        LOGGER.info("Treinando com WeightedRandomSampler (classes binárias balanceadas: AB vs CD).")
    else:
        dl_tr = DataLoader(ds_tr, shuffle=True, **dl_tr_kwargs)
        LOGGER.info("Treinando com shuffle aleatório convencional.")
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, **dict(loader_common))

    # 5) Modelo e otimizador
    model = build_model(num_classes=2, train_backbone=args.train_backbone, unfreeze_last_block=args.unfreeze_last_block)
    model.to(device)
    compiled = False
    if getattr(args, "torch_compile", False):
        import importlib.util

        triton_available = importlib.util.find_spec("triton") is not None
        if not triton_available:
            LOGGER.warning("torch.compile exige o pacote 'triton', mas ele não está instalado; prosseguindo sem compilação.")
        elif hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
                compiled = True
                LOGGER.info("torch.compile habilitado (modo=max-autotune).")
            except Exception as exc:
                LOGGER.warning("torch.compile falhou, prosseguindo sem compilação: %s", exc)
        else:
            LOGGER.warning("torch.compile não disponível nesta versão do PyTorch; ignorando flag.")
    setattr(args, "torch_compile_effective", compiled)
    # 5.1) Loss ponderada opcional: 'auto' usa pesos ~ inverso da frequência por classe binária no treino
    loss_fn = nn.CrossEntropyLoss()
    if args.class_weights == "auto":
        ys = [_map_to_binary_label(r.get("professional_label")) for r in rows_tr]
        ys_binary = [y for y in ys if y is not None]
        counts = np.array([max(1, ys_binary.count(c)) for c in range(2)], dtype=np.float64)
        if np.any(counts == 0):
            LOGGER.warning("Alguma classe binária ausente no treino; ignorando class weights.")
        else:
            # Regra simples: weight_c = N_total / (n_classes * count_c)
            weights = (len(ys_binary) / (2.0 * counts)).astype(np.float32)
            cw = torch.tensor(weights, dtype=torch.float32, device=device)
            loss_fn = nn.CrossEntropyLoss(weight=cw)
            LOGGER.info("Class weights (auto, binário): AB=%s, CD=%s", weights[0], weights[1])

    # 5.2) Otimizador com grupos (head vs backbone)
    # Warmup opcional: se warmup_epochs > 0, congela últimos blocos temporariamente e treina só a head nas primeiras épocas
    # Função auxiliar para obter parâmetros dos últimos blocos de features
    def get_last_blocks_params(model):
        """Retorna parâmetros dos últimos 2 blocos de features (equivalente ao layer4)."""
        num_blocks = len(model.features)
        params = []
        for i in range(num_blocks - 2, num_blocks):  # últimos 2 blocos
            params.extend(list(model.features[i].parameters()))
        return params
    
    if args.warmup_epochs > 0 and not args.train_backbone and args.unfreeze_last_block:
        last_blocks_params = get_last_blocks_params(model)
        for p in last_blocks_params:
            p.requires_grad = False
        LOGGER.info("Warmup ativo: treinar apenas a head por %d época(s); últimos blocos serão liberados após warmup", args.warmup_epochs)
    head_params = list(model.classifier.parameters())
    # backbone inicialmente congelado; se args.unfreeze_last_block ou warmup vai habilitar depois
    last_blocks_params = get_last_blocks_params(model)
    param_groups = [
        {"params": [p for p in head_params if p.requires_grad], "lr": args.lr},
    ]
    # se já veio com unfreeze_last_block/train_backbone, adiciona grupo já
    if any(p.requires_grad for p in last_blocks_params):
        param_groups.append({"params": [p for p in last_blocks_params if p.requires_grad], "lr": args.backbone_lr})
    adamw_kwargs = dict(weight_decay=args.weight_decay)
    if getattr(args, "fused_optim", False) and device.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        optim = torch.optim.AdamW(param_groups, **adamw_kwargs)
        if adamw_kwargs.get("fused"):
            LOGGER.info("Otimizador AdamW (fused=True) ativado.")
    except TypeError as exc:
        if adamw_kwargs.get("fused"):
            LOGGER.warning("AdamW fused não suportado neste PyTorch (%s); retornando ao modo padrão.", exc)
            adamw_kwargs.pop("fused", None)
        optim = torch.optim.AdamW(param_groups, **adamw_kwargs)

    # 6) Loop de treino
    hist_rows = []
    best = {"acc": -1, "state": None}
    patience = max(0, int(getattr(args, "early_stop_patience", 0) or 0))
    min_delta = float(getattr(args, "early_stop_min_delta", 0.0) or 0.0)
    if min_delta < 0:
        LOGGER.warning("early-stop-min-delta negativo (%s) ajustado para 0.0.", min_delta)
        min_delta = 0.0
    no_improve_epochs = 0
    early_stop_triggered = False
    lr_patience = max(0, int(getattr(args, "lr_reduce_patience", 0) or 0))
    scheduler = None
    if lr_patience > 0:
        factor = float(getattr(args, "lr_reduce_factor", 0.5) or 0.5)
        min_lr = float(getattr(args, "lr_reduce_min_lr", 1e-7) or 1e-7)
        cooldown = max(0, int(getattr(args, "lr_reduce_cooldown", 0) or 0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max",
            factor=factor,
            patience=lr_patience,
            threshold=min_delta,
            threshold_mode="rel",
            cooldown=cooldown,
            min_lr=min_lr,
        )
        LOGGER.info(
            "ReduceLROnPlateau ativado (factor=%.4f, patience=%d, min_lr=%.2e, cooldown=%d, threshold=%.4f).",
            factor,
            lr_patience,
            min_lr,
            cooldown,
            min_delta,
        )
    context = profiler if profiler is not None else contextlib.nullcontext()
    with context:
        for epoch in range(1, args.epochs + 1):
            # Warmup: após warmup_epochs, libera últimos blocos (se estavam congelados pelo warmup)
            if args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1 and not args.train_backbone and args.unfreeze_last_block:
                # Função auxiliar para obter parâmetros dos últimos blocos
                num_blocks = len(model.features)
                last_blocks_params = []
                for i in range(num_blocks - 2, num_blocks):
                    last_blocks_params.extend(list(model.features[i].parameters()))
                
                if not any(p.requires_grad for p in last_blocks_params):
                    for p in last_blocks_params:
                        p.requires_grad = True
                    param_groups = [
                        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": args.lr},
                        {"params": [p for p in last_blocks_params if p.requires_grad], "lr": args.backbone_lr},
                    ]
                    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
                    LOGGER.info("Warmup concluído: liberando últimos blocos de features no epoch %d com backbone_lr=%s", epoch, args.backbone_lr)
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model,
                dl_tr,
                optim,
                device,
                loss_fn=loss_fn,
                profiler=profiler if profiler is not None else None,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            val = validate(model, dl_va, device, amp_enabled=amp_enabled)
            t1 = time.time()
            hist_rows.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_acc": val["acc"],
                "val_kappa": val["kappa"],
                "val_auc": (val["auc"] if val["auc"] is not None else np.nan),
                "sec": round(t1 - t0, 2),
            })
            auc_txt = "NA" if val["auc"] is None else f"{val['auc']:.4f}"
            LOGGER.info(
                "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | val_acc=%.4f | kappa=%.4f | auc=%s | tempo=%.2fs",
                epoch,
                args.epochs,
                tr_loss,
                tr_acc,
                val["acc"],
                val["kappa"],
                auc_txt,
                t1 - t0,
            )
            if scheduler is not None:
                prev_lrs = [pg["lr"] for pg in optim.param_groups]
                scheduler.step(val["acc"])
                new_lrs = [pg["lr"] for pg in optim.param_groups]
                if any(new < old - 1e-12 for new, old in zip(new_lrs, prev_lrs)):
                    LOGGER.info(
                        "ReduceLROnPlateau reduziu LR: %s -> %s",
                        ", ".join(f"{lr:.6e}" for lr in prev_lrs),
                        ", ".join(f"{lr:.6e}" for lr in new_lrs),
                    )

            improved = val["acc"] > (best["acc"] + min_delta)
            if improved:
                best["acc"] = val["acc"]
                best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                if patience > 0:
                    no_improve_epochs += 1
                    LOGGER.info(
                        "Early stopping: val_acc=%.4f não superou %.4f (delta>%.4f) [%d/%d]",
                        val["acc"],
                        best["acc"],
                        min_delta,
                        no_improve_epochs,
                        patience,
                    )
                    if no_improve_epochs >= patience:
                        LOGGER.info(
                            "Early stopping acionado no epoch %d (melhor val_acc=%.4f).",
                            epoch,
                            best["acc"],
                        )
                        early_stop_triggered = True
                        break
            pd.DataFrame(val["val_rows"]).to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
            metrics_path = os.path.join(args.outdir, "metrics", "val_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in val.items() if k != "val_rows"}, f, indent=2)
            save_metrics_figure(val, os.path.join(args.outdir, "metrics", "val_metrics.png"))
            LOGGER.debug("Artefatos de validacao atualizados (epoch %d).", epoch)

            # Atualiza figura de historico (loss/acc/kappa/AUC) a cada epoca
            save_history_figure(hist_rows, os.path.join(args.outdir, "train_history.png"))

    history_path = os.path.join(args.outdir, "train_history.csv")
    pd.DataFrame(hist_rows).to_csv(history_path, index=False)
    LOGGER.info("Histórico de treino salvo em %s", history_path)

    # 7) Embeddings do conjunto de validação com o melhor modelo
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        best_epoch = max(hist_rows, key=lambda r: r["val_acc"]) if hist_rows else None
        LOGGER.info("Melhor val_acc observada: %.4f", best["acc"])
        if best_epoch is not None:
            LOGGER.info(
                "Resumo da melhor época: epoch=%d | val_acc=%.4f | kappa=%.4f | auc=%.4f | train_acc=%.4f | train_loss=%.4f",
                best_epoch["epoch"],
                best_epoch["val_acc"],
                best_epoch.get("val_kappa", best_epoch.get("val_kappa_quadratic", float("nan"))),
                best_epoch.get("val_auc", best_epoch.get("val_auc_ovr", float("nan"))),
                best_epoch.get("train_acc", float("nan")),
                best_epoch.get("train_loss", float("nan")),
            )
    dl_va_eval = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, **dict(loader_common))
    LOGGER.info(
        "Resumo final do split (device=%s, pin_memory=%s): train=%d | val=%d",
        device,
        loader_cfg["pin_memory"],
        len(ds_tr),
        len(ds_va),
    )
    feats, metas = extract_embeddings(model, dl_va_eval, device, amp_enabled=amp_enabled)
    emb_npz_path = os.path.join(args.outdir, "embeddings_val.npz")
    np.savez_compressed(emb_npz_path, embeddings=feats)
    # salva CSV combinando metadados + embeddings (como colunas e0..e1279 para EfficientNetB0)
    emb_cols = {f"e{i}": feats[:, i] for i in range(feats.shape[1])}
    df_emb = pd.DataFrame(metas)
    df_emb = pd.concat([df_emb, pd.DataFrame(emb_cols)], axis=1)
    emb_csv_path = os.path.join(args.outdir, "embeddings_val.csv")
    df_emb.to_csv(emb_csv_path, index=False)
    LOGGER.info("Embeddings do conjunto de validação salvos em %s e %s", emb_npz_path, emb_csv_path)

    LOGGER.info("Treino e extração de embeddings concluídos. Resultados em: %s", args.outdir)
    LOGGER.info("Log detalhado disponível em: %s", os.path.join(args.outdir, "run.log"))


def build_argparser() -> argparse.ArgumentParser:
    """Cria e documenta os argumentos de linha de comando.

    Dicas rápidas:
    - Use --csv com seu classificacao.csv e --dicom-root apontando para as pastas por AccessionNumber.
    - Se tiver um CSV com caminhos (image_path), não precisa de --dicom-root.
    - Para melhorar classes raras: --class-weights auto e/ou --sampler-weighted.
    - Para fine-tuning leve: --warmup-epochs 2 --unfreeze-last-block --backbone-lr 1e-5.
    """
    p = argparse.ArgumentParser(description="EfficientNetB0 - Classificação binária de densidade mamária (AB vs CD) com Transfer Learning")
    p.add_argument("--csv", required=True, help="Caminho para classificacao.csv, CSV baseado em caminhos, ou diretório do dataset patches_completo")
    p.add_argument("--dicom-root", dest="dicom_root", default="archive", help="Raiz das pastas por AccessionNumber (para classificacao.csv)")
    p.add_argument("--outdir", default="outputs/mammo_efficientnetb0_density_abxcd", help="Pasta raiz para results_*/metrics e cache compartilhado")
    p.add_argument("--epochs", type=int, default=HP.EPOCHS)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=HP.BATCH_SIZE)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=HP.NUM_WORKERS)
    p.add_argument(
        "--preset",
        dest="preset",
        choices=["auto", "windows", "mps"],
        default="auto",
        help="Ajuste automático de num_workers/prefetch/persistência: 'windows' limita spawn, 'mps' usa loaders síncronos",
    )
    p.add_argument("--img-size", dest="img_size", type=int, default=HP.IMG_SIZE)
    p.add_argument("--lr", type=float, default=HP.LR)
    p.add_argument("--backbone-lr", dest="backbone_lr", type=float, default=HP.BACKBONE_LR, help="LR para últimos blocos de features/backbone quando destravado")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--val-frac", dest="val_frac", type=float, default=HP.VAL_FRAC)
    p.add_argument("--seed", type=int, default=HP.SEED)
    p.add_argument("--device", default=HP.DEVICE, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--amp",
        dest="amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Habilita Automatic Mixed Precision (autocast + GradScaler). Ativado automaticamente em CUDA/MPS; use --no-amp para "
            "forçar precisão total."
        ),
    )
    p.add_argument("--train-backbone", dest="train_backbone", action="store_true", help="Descongela todo o backbone (não recomendado; apenas para experimentação)")
    p.add_argument("--unfreeze-last-block", dest="unfreeze_last_block", action="store_true", help="Descongela últimos blocos de features (padrão)")
    p.add_argument("--no-unfreeze-last-block", dest="unfreeze_last_block", action="store_false", help="Congela últimos blocos (apenas classifier treina)")
    p.add_argument("--include-class-5", dest="include_class_5", action="store_true", help="Inclui exemplos com Classification==5")
    p.add_argument("--class-weights", dest="class_weights", default=HP.CLASS_WEIGHTS, choices=["none","auto"], help="Pondera CrossEntropy por frequência de classe (auto)")
    p.add_argument("--sampler-weighted", dest="sampler_weighted", action="store_true", help="Usa WeightedRandomSampler no treino")
    p.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=HP.WARMUP_EPOCHS, help="Épocas só na head antes de liberar últimos blocos (últimos blocos precisam estar descongelados via --unfreeze-last-block)")
    p.add_argument("--prefetch-factor", dest="prefetch_factor", type=int, default=HP.PREFETCH_FACTOR, help="Batches antecipadas por worker do DataLoader (num_workers>0)")
    p.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", help="Mantém workers do DataLoader vivos entre épocas (recomendado no Windows/macOS)")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false", help="Desativa persistent_workers")
    p.add_argument(
        "--train-augment",
        dest="train_augment",
        action=argparse.BooleanOptionalAction,
        default=HP.TRAIN_AUGMENT,
        help="Ativa augmentations leves (flip/rotação ±5°) no treino; use --no-train-augment para desativar.",
    )
    p.add_argument(
        "--loader-heuristics",
        dest="loader_heuristics",
        action=argparse.BooleanOptionalAction,
        default=HP.LOADER_HEURISTICS,
        help="Aplica heurísticas automáticas para DataLoader (ajustes de num_workers/prefetch). Use --no-loader-heuristics para respeitar valores brutos.",
    )
    p.add_argument(
        "--fused-optim",
        dest="fused_optim",
        action=argparse.BooleanOptionalAction,
        default=HP.FUSED_OPTIM,
        help="Usa AdamW com fused=True em CUDA quando suportado (acelera o passo do otimizador).",
    )
    p.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action=argparse.BooleanOptionalAction,
        default=HP.TORCH_COMPILE,
        help="Envolve o modelo em torch.compile (modo=max-autotune) para otimizar o forward/backward.",
    )
    p.add_argument(
        "--early-stop-patience",
        dest="early_stop_patience",
        type=int,
        default=HP.EARLY_STOP_PATIENCE,
        help="Número de épocas sem melhoria em val_acc antes de interromper (0 desativa).",
    )
    p.add_argument(
        "--early-stop-min-delta",
        dest="early_stop_min_delta",
        type=float,
        default=HP.EARLY_STOP_MIN_DELTA,
        help="Melhoria mínima exigida em val_acc para resetar a paciência do early stopping.",
    )
    p.add_argument(
        "--lr-reduce-patience",
        dest="lr_reduce_patience",
        type=int,
        default=HP.LR_REDUCE_PATIENCE,
        help="Patience (em épocas) para ReduceLROnPlateau em val_acc; 0 desativa.",
    )
    p.add_argument(
        "--lr-reduce-factor",
        dest="lr_reduce_factor",
        type=float,
        default=HP.LR_REDUCE_FACTOR,
        help="Fator multiplicativo ao reduzir o LR (e.g., 0.5 divide o LR pela metade).",
    )
    p.add_argument(
        "--lr-reduce-min-lr",
        dest="lr_reduce_min_lr",
        type=float,
        default=HP.LR_REDUCE_MIN_LR,
        help="Limite inferior para o LR quando reduzir (min_lr).",
    )
    p.add_argument(
        "--lr-reduce-cooldown",
        dest="lr_reduce_cooldown",
        type=int,
        default=HP.LR_REDUCE_COOLDOWN,
        help="Cooldown (épocas) após uma redução antes de voltar a monitorar (ReduceLROnPlateau).",
    )
    p.add_argument(
        "--mps-sync-loaders",
        dest="mps_sync_loaders",
        action="store_true",
        help="Força carregamento síncrono (num_workers=0) ao usar device MPS, útil para depuração",
    )
    p.add_argument(
        "--no-mps-sync-loaders",
        dest="mps_sync_loaders",
        action="store_false",
        help="Mantém multiprocessamento mesmo no device MPS (se suportado)",
    )
    p.add_argument(
        "--cache-mode",
        dest="cache_mode",
        default=HP.CACHE_MODE,
        choices=["auto", "disk", "memory", "none", "tensor-disk", "tensor-memmap"],
        help=(
            "Estratégia de cache das imagens base: 'disk' materializa DICOMs como PNGs em <outdir>/cache, "
            "'memory' guarda PILs em RAM, 'tensor-disk' persiste tensores decodificados em .pt (mais espaço em disco, sem custo de decode), "
            "'tensor-memmap' usa arquivos memory-mapped para compartilhar tensores sem duplicar RAM (depende de IO de disco), 'none' desativa. "
            "'auto' escolhe heurística conforme dataset."
        ),
    )
    # Compatibilidade com versões anteriores
    p.add_argument("--cache-images", dest="cache_mode", action="store_const", const="memory", help="(legado) Equivalente a --cache-mode memory")
    p.add_argument("--no-cache-images", dest="cache_mode", action="store_const", const="none", help="(legado) Equivalente a --cache-mode none")
    p.add_argument("--deterministic", dest="deterministic", action="store_true", help="Força execução determinística (menor desempenho)")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="Desativa determinismo para máxima performance")
    p.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", help="Permite TF32 em GPUs NVIDIA (quando disponível)")
    p.add_argument("--no-tf32", dest="allow_tf32", action="store_false", help="Desativa TF32 mesmo que a GPU suporte")
    p.add_argument("--log-level", dest="log_level", default=HP.LOG_LEVEL, choices=["debug", "info", "warning", "error", "critical"], help="Nível de log para console")
    p.add_argument("--profile", dest="profile", action="store_true", help="Habilita PyTorch Profiler (gera trace para análise de gargalos)")
    p.add_argument("--profile-dir", dest="profile_dir", default=os.path.join("outputs", "profiler"), help="Diretório onde o profiler salvará os traces")
    # Auto-incremento do diretório de saída: se já existir, cria _1, _2, ...
    p.add_argument("--auto-increment-outdir", dest="auto_increment_outdir", action="store_true", help="Incrementa --outdir (_1, _2, ...) se já existir")
    p.add_argument("--no-auto-increment-outdir", dest="auto_increment_outdir", action="store_false", help="Não incrementa --outdir se já existir")
    p.set_defaults(
        deterministic=HP.DETERMINISTIC,
        allow_tf32=HP.ALLOW_TF32,
        persistent_workers=HP.PERSISTENT_WORKERS,
        cache_mode=HP.CACHE_MODE,
        train_augment=HP.TRAIN_AUGMENT,
        loader_heuristics=HP.LOADER_HEURISTICS,
        fused_optim=HP.FUSED_OPTIM,
        torch_compile=HP.TORCH_COMPILE,
        early_stop_patience=HP.EARLY_STOP_PATIENCE,
        early_stop_min_delta=HP.EARLY_STOP_MIN_DELTA,
        preset="auto",
        mps_sync_loaders=False,
        auto_increment_outdir=True,
        unfreeze_last_block=HP.UNFREEZE_LAST_BLOCK,
        train_backbone=HP.TRAIN_BACKBONE,
    )
    return p


def main(argv: Optional[List[str]] = None):
    """Faz parse dos argumentos CLI e delega a execução principal para `run` (ideal ao importar sem executar)."""
    args = build_argparser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
