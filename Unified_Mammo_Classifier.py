#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified_Mammo_Classifier.py
---------------------------
Script unificado que cobre as funcionalidades presentes em todos os .py da raiz do projeto:

- Treino/validação EfficientNetB0 e ResNet50 (binário ABxCD e multiclasse 1..4) com todos os hiperparâmetros dos scripts RSNA_*.
- Modos de extração/análise de embeddings (PCA/t-SNE/UMAP/k-means, pré-visualizações e tabelas) que substituem extract_mammo_resnet50.py.
- EDA rápida (resumo de labels, grade de exemplos, pré-processamento) inspirado no RSNA_Mammography_EDA.py.
- Heurísticas de cache (auto/memória/disk/tensor-*) e DataLoader (num_workers, prefetch, persistência) usadas nos pipelines originais.
- Geração de artefatos de saída: history.csv/png, val_predictions.csv, embeddings_val.*, best_metrics.json, Grad-CAM opcional.
- Subcomandos utilitários equivalentes ao Projeto.py: checklist de exportação e empacotamento de runs; stub de rl-refine.

Exemplos rápidos
----------------
# Treino binário EfficientNet com cache auto e sampler balanceado
python Unified_Mammo_Classifier.py \
  --mode train --task binary --model efficientnet_b0 \
  --csv classificacao.csv --dicom-root archive \
  --outdir outputs/effnet_binary --epochs 20 --batch-size 16 \
  --cache-mode auto --sampler-weighted --class-weights auto

# Extração + PCA/t-SNE/cluster, salvando figuras e tabelas
python Unified_Mammo_Classifier.py \
  --mode extract --csv archive --model resnet50 \
  --outdir outputs/embed_analysis --pca --tsne --cluster-auto --save-csv

# Checklist de exportação de runs existentes
python Unified_Mammo_Classifier.py --mode eval-export --runs outputs/run_a --runs outputs/run_b
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from tqdm.contrib.logging import TqdmLoggingHandler
except ImportError:  # pragma: no cover - fallback mínimo
    TqdmLoggingHandler = logging.StreamHandler

import torch
import torch.nn as nn
from torch import profiler
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_v2_F
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights, efficientnet_b0, resnet50

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image, ImageDraw

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

LOGGER = logging.getLogger("mammo_unified")
LOGGER.addHandler(logging.NullHandler())

DICOM_EXTS = (".dcm", ".dicom")
CACHE_AUTO_DISK_MAX = 6000
CACHE_AUTO_MEMORY_MAX = 1000


# ==============================================================================
# Utilidades e pré-processamento DICOM
# ==============================================================================

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    if deterministic:
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def configure_runtime(allow_tf32: bool = True, deterministic: bool = False) -> None:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def find_best_data_dir(pref: str) -> str:
    if os.path.isdir(pref):
        return pref
    alt = pref.replace("archieve", "archive")
    if os.path.isdir(alt):
        LOGGER.info("data_dir '%s' não encontrado; usando '%s'", pref, alt)
        return alt
    return pref


def human_readable_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"


def _is_mono1(ds: pydicom.dataset.FileDataset) -> bool:
    return getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1"


def _to_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _apply_rescale(ds: pydicom.dataset.FileDataset, arr: np.ndarray) -> np.ndarray:
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
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


def dicom_to_pil_rgb(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = _to_float32(ds.pixel_array)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr = robust_window(arr)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    return Image.merge("RGB", (pil, pil, pil))


def _is_dicom_path(path: str) -> bool:
    return str(path).lower().endswith(DICOM_EXTS)


# ==============================================================================
# Parsing de dados
# ==============================================================================

def _map_label(label: Optional[int], task: str) -> Optional[int]:
    if label not in {1, 2, 3, 4}:
        return None
    if task == "binary":
        return 0 if label in {1, 2} else 1
    if task == "multiclass":
        return label - 1
    return None


def resolve_dataset_cache_mode(choice: str, rows: Sequence[Dict[str, Any]]) -> str:
    if choice != "auto":
        return choice
    n = len(rows)
    has_dicom = any(_is_dicom_path(r.get("image_path", "")) for r in rows)
    if n <= CACHE_AUTO_MEMORY_MAX:
        return "memory"
    if n <= CACHE_AUTO_DISK_MAX and has_dicom:
        return "disk"
    if has_dicom and n > CACHE_AUTO_DISK_MAX:
        return "tensor-disk"
    return "none"


def _parse_input_source(args) -> pd.DataFrame:
    # Diretório com featureS.txt (mamografias/patches)
    if os.path.isdir(args.csv):
        rows: List[Dict[str, Any]] = []
        has_subfolders = any(
            os.path.isdir(os.path.join(args.csv, d))
            for d in os.listdir(args.csv)
            if not d.startswith(".")
        )
        search_dirs = [
            os.path.join(args.csv, d)
            for d in os.listdir(args.csv)
            if os.path.isdir(os.path.join(args.csv, d))
        ] if has_subfolders else [args.csv]

        for folder in search_dirs:
            feat_path = os.path.join(folder, "featureS.txt")
            if not os.path.exists(feat_path):
                continue
            with open(feat_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for i in range(0, len(lines), 2):
                if i + 1 >= len(lines):
                    break
                fname, cls_raw = lines[i], lines[i + 1]
                try:
                    birads = int(cls_raw) + 1
                    if "(" in fname and " (" not in fname:
                        fname = fname.replace("(", " (")
                    if not fname.endswith(".png"):
                        fname += ".png"
                    full_path = os.path.join(folder, fname)
                    if os.path.exists(full_path):
                        rows.append(
                            {
                                "image_path": full_path,
                                "professional_label": birads,
                                "accession": os.path.basename(folder),
                            }
                        )
                except Exception:
                    continue
        if not rows:
            raise ValueError("Nenhuma imagem encontrada via featureS.txt")
        return pd.DataFrame(rows)

    df_raw = pd.read_csv(args.csv)
    if {"AccessionNumber", "Classification"}.issubset(df_raw.columns):
        rows = []
        dicom_root = find_best_data_dir(args.dicom_root)
        df_raw["AccessionNumber"] = df_raw["AccessionNumber"].astype(str).str.strip()
        for _, r in df_raw.iterrows():
            lab = int(r["Classification"]) if pd.notna(r["Classification"]) else None
            if lab == 5 and not args.include_class_5:
                continue
            acc = r["AccessionNumber"]
            folder = os.path.join(dicom_root, acc)
            if not os.path.isdir(folder):
                continue
            dicoms = [f for f in os.listdir(folder) if f.lower().endswith(DICOM_EXTS)]
            if dicoms:
                rows.append(
                    {
                        "image_path": os.path.join(folder, sorted(dicoms)[0]),
                        "professional_label": lab,
                        "accession": acc,
                    }
                )
        return pd.DataFrame(rows)

    if "image_path" in df_raw.columns:
        label_col = next(
            (c for c in ["density_label", "label", "y", "professional_label"] if c in df_raw.columns),
            None,
        )
        df_raw["professional_label"] = df_raw[label_col] if label_col else None
        if "accession" not in df_raw.columns:
            df_raw["accession"] = [os.path.basename(os.path.dirname(p)) for p in df_raw["image_path"]]
        return df_raw

    raise ValueError("Formato de entrada não reconhecido.")


# ==============================================================================
# Dataset com cache avançado
# ==============================================================================


class UnifiedMammoDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        img_size: int,
        task: str,
        train: bool,
        augment: bool = True,
        cache_mode: str = "none",
        cache_dir: Optional[str] = None,
        split_name: str = "train",
    ):
        self.rows = rows
        self.img_size = img_size
        self.task = task
        self.train = train
        self.augment = bool(augment and train)
        self.cache_mode = cache_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name

        valid_modes = {"none", "memory", "disk", "tensor-disk", "tensor-memmap"}
        if self.cache_mode not in valid_modes:
            raise ValueError(f"cache_mode inválido: {self.cache_mode}")
        if self.cache_mode in {"disk", "tensor-disk", "tensor-memmap"} and self.cache_dir is None:
            raise ValueError("cache_dir é obrigatório para cache em disco")

        self._image_cache: Optional[Dict[str, Image.Image]] = {} if self.cache_mode == "memory" else None
        self._disk_cache_index: Dict[str, str] = {}
        self._tensor_disk_index: Dict[str, str] = {}
        self._tensor_memmap_index: Dict[str, Dict[str, Any]] = {}
        if self.cache_dir and self.cache_mode in {"disk", "tensor-disk", "tensor-memmap"}:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_mode == "disk":
            self._prepare_disk_cache()
        elif self.cache_mode in {"tensor-disk", "tensor-memmap"}:
            self._prepare_tensor_cache()

        self._norm_mean = [0.485, 0.456, 0.406]
        self._norm_std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.rows)

    def _read_image(self, path: str) -> Image.Image:
        if _is_dicom_path(path):
            return dicom_to_pil_rgb(path)
        return Image.open(path).convert("RGB")

    def _cache_path_for(self, path: str) -> Path:
        assert self.cache_dir is not None
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.png"

    def _tensor_cache_base_path(self, path: str) -> Path:
        assert self.cache_dir is not None
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return self.cache_dir / h

    def _prepare_disk_cache(self) -> None:
        assert self.cache_dir is not None
        for row in tqdm(self.rows, desc=f"Cache[{self.split_name}]", leave=False, disable=len(self.rows) < 16):
            path = str(row.get("image_path"))
            if not path or not _is_dicom_path(path):
                continue
            cache_path = self._cache_path_for(path)
            self._disk_cache_index[path] = str(cache_path)
            if cache_path.exists():
                continue
            try:
                img = self._read_image(path)
                img.save(cache_path, format="PNG")
            except Exception as exc:  # pragma: no cover - melhor log possível
                LOGGER.warning("Falha ao materializar cache de %s: %s", path, exc)

    def _prepare_tensor_cache(self) -> None:
        assert self.cache_dir is not None
        for row in tqdm(self.rows, desc=f"TensorCache[{self.split_name}]", leave=False, disable=len(self.rows) < 16):
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
                self._tensor_memmap_index[path] = {"data_path": str(data_path), "meta_path": str(meta_path)}
                if data_path.exists() and meta_path.exists():
                    continue
            try:
                tensor = self._decode_to_tensor(path)
                self._materialize_tensor_cache(path, tensor)
            except Exception as exc:
                LOGGER.warning("Falha ao decodificar %s para cache de tensores: %s", path, exc)

    def _decode_to_tensor(self, path: str) -> torch.Tensor:
        img = self._read_image(path)
        return tv_v2_F.to_image(img)

    def _materialize_tensor_cache(self, path: str, tensor: torch.Tensor) -> None:
        if self.cache_mode == "tensor-disk":
            cache_path = self._tensor_cache_base_path(path).with_suffix(".pt")
            self._tensor_disk_index[path] = str(cache_path)
            if cache_path.exists():
                return
            arr = tensor.detach().cpu().numpy()
            torch.save(torch.from_numpy(np.array(arr, copy=True)), cache_path)
        elif self.cache_mode == "tensor-memmap":
            base = self._tensor_cache_base_path(path)
            data_path = base.with_suffix(".dat")
            meta_path = base.with_suffix(".json")
            self._tensor_memmap_index[path] = {"data_path": str(data_path), "meta_path": str(meta_path)}
            if data_path.exists() and meta_path.exists():
                return
            arr = tensor.detach().cpu().numpy()
            mm = np.memmap(data_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
            mm[:] = arr
            mm.flush()
            meta_path.write_text(json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}))
            del mm

    def _get_base_image(self, path: str) -> Image.Image:
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
                except Exception:
                    pass
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
        if self.cache_mode == "tensor-disk":
            cache_path = self._tensor_disk_index.get(path)
            if cache_path is None and self.cache_dir is not None:
                cache_path = str(self._tensor_cache_base_path(path).with_suffix(".pt"))
                self._tensor_disk_index[path] = cache_path
            if cache_path and os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, map_location="cpu")
                except Exception:
                    return None
        if self.cache_mode == "tensor-memmap":
            info = self._tensor_memmap_index.get(path)
            if info is None and self.cache_dir is not None:
                base = self._tensor_cache_base_path(path)
                info = {"data_path": str(base.with_suffix(".dat")), "meta_path": str(base.with_suffix(".json"))}
                self._tensor_memmap_index[path] = info
            if info:
                data_path, meta_path = info.get("data_path"), info.get("meta_path")
                if data_path and meta_path and os.path.exists(data_path) and os.path.exists(meta_path):
                    meta = info.get("meta") or json.loads(Path(meta_path).read_text())
                    info["meta"] = meta
                    shape = tuple(meta["shape"])
                    dtype = np.dtype(meta["dtype"])
                    mm = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
                    arr = np.array(mm, copy=True)
                    return torch.from_numpy(arr)
        return None

    def __getitem__(self, i: int):
        row = self.rows[i]
        base_tensor = self._get_cached_tensor(row["image_path"])
        if base_tensor is None:
            img = self._get_base_image(row["image_path"])
            base_tensor = tv_v2_F.to_image(img)
            if self.cache_mode in {"tensor-disk", "tensor-memmap"}:
                self._materialize_tensor_cache(row["image_path"], base_tensor)

        tensor = tv_v2_F.resize(base_tensor, [self.img_size], interpolation=InterpolationMode.BICUBIC, antialias=False)
        tensor = tv_v2_F.center_crop(tensor, [self.img_size, self.img_size])
        tensor = tv_v2_F.to_dtype(tensor, torch.float32, scale=True)
        if self.augment:
            if float(torch.rand(1)) < 0.5:
                tensor = tv_v2_F.horizontal_flip(tensor)
            angle = float(torch.empty(1).uniform_(-5.0, 5.0))
            tensor = tv_v2_F.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0.0)
        tensor = tv_v2_F.normalize(tensor, self._norm_mean, self._norm_std)

        raw_label = row.get("professional_label")
        y = _map_label(raw_label, self.task)
        meta = {"path": row.get("image_path"), "accession": row.get("accession", "unknown"), "raw_label": raw_label}
        tensor = tensor.contiguous(memory_format=torch.channels_last)
        y_tensor = -1 if y is None else y
        return tensor, y_tensor, meta


def _collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    return xs, ys_tensor, metas


# ==============================================================================
# Modelos
# ==============================================================================


class EfficientNetWithFusion(nn.Module):
    def __init__(self, base: nn.Module, num_classes: int, extra_dim: int = 0):
        super().__init__()
        self.backbone = base
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(in_features + extra_dim, num_classes))

    def forward(self, x, extra=None):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if extra is not None:
            x = torch.cat([x, extra], dim=1)
        return self.classifier(x)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
        return base
    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Arquitetura {arch} desconhecida.")


def _unfreeze_last_block(model: nn.Module, arch: str) -> None:
    if arch == "resnet50" and hasattr(model, "layer4"):
        for p in model.layer4.parameters():
            p.requires_grad = True
    if arch == "efficientnet_b0" and hasattr(model, "features"):
        for p in model.features[-1].parameters():
            p.requires_grad = True


def _freeze_backbone(model: nn.Module, arch: str) -> None:
    for name, p in model.named_parameters():
        if arch == "resnet50" and name.startswith("fc"):
            continue
        if arch == "efficientnet_b0" and name.startswith("classifier"):
            continue
        p.requires_grad = False


def _build_param_groups(model: nn.Module, arch: str, lr_head: float, lr_backbone: float, train_backbone: bool) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    if arch == "resnet50":
        head_params = [p for n, p in model.named_parameters() if n.startswith("fc") and p.requires_grad]
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc") and p.requires_grad]
    else:
        head_params = [p for n, p in model.named_parameters() if n.startswith("classifier") and p.requires_grad]
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
    if head_params:
        params.append({"params": head_params, "lr": lr_head})
    if train_backbone and backbone_params:
        params.append({"params": backbone_params, "lr": lr_backbone})
    return params


# ==============================================================================
# Treino/validação
# ==============================================================================


def _resolve_loader_runtime(args, device: torch.device) -> Tuple[int, Optional[int], bool]:
    nw = args.num_workers
    prefetch = args.prefetch_factor if args.prefetch_factor and args.prefetch_factor > 0 else None
    persistent = args.persistent_workers
    if not args.loader_heuristics:
        return nw, prefetch, persistent
    if device.type == "mps":
        return 0, prefetch, False
    if device.type == "cpu":
        return max(0, min(nw, os.cpu_count() or 0)), prefetch, persistent
    return nw, prefetch, persistent


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler: Optional[GradScaler] = None):
    model.train()
    losses, correct, total = [], 0, 0
    for x, y, _ in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        mask = y >= 0
        if not mask.any():
            continue
        x, y = x[mask], y[mask]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            loss = loss_fn(logits, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return np.mean(losses) if losses else 0.0, correct / max(total, 1)


def validate(
    model,
    loader,
    loss_fn,
    device,
    task: str,
    collect_preds: bool = False,
    gradcam: bool = False,
    gradcam_dir: Optional[Path] = None,
    gradcam_limit: int = 4,
):
    model.eval()
    losses, all_preds, all_true, all_probs = [], [], [], []
    pred_rows: List[Dict[str, Any]] = []
    gradcam_saved = 0
    with torch.set_grad_enabled(gradcam):
        for x, y, meta in tqdm(loader, desc="Val", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            mask = y >= 0
            if not mask.any():
                continue
            x, y = x[mask], y[mask]
        with torch.autocast(device_type=device.type, enabled=device.type in {"cuda", "mps"}):
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(float(loss.item()))
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if collect_preds:
                for i in range(len(meta)):
                    pred_rows.append(
                        {
                            "path": meta[i].get("path"),
                            "accession": meta[i].get("accession"),
                            "raw_label": meta[i].get("raw_label"),
                            "target": int(y[i].item()),
                            "pred": int(preds[i].item()),
                            "probs": probs[i].detach().cpu().numpy().tolist(),
                        }
                    )
            if gradcam and gradcam_dir and gradcam_saved < gradcam_limit:
                gradcam_saved += _save_gradcam_batch(model, x, preds, meta, gradcam_dir, gradcam_saved, device)

    metrics: Dict[str, Any] = {}
    if all_true:
        metrics["loss"] = np.mean(losses)
        metrics["acc"] = accuracy_score(all_true, all_preds)
        metrics["kappa"] = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        try:
            if len(np.unique(all_true)) == 2:
                metrics["auc"] = roc_auc_score(all_true, np.array(all_probs)[:, 1])
            else:
                metrics["auc"] = roc_auc_score(all_true, all_probs, multi_class="ovr")
        except Exception:
            metrics["auc"] = 0.0
        metrics["cm"] = confusion_matrix(all_true, all_preds).tolist()
        metrics["report"] = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
    return metrics, pred_rows


def _save_gradcam_batch(model, x, preds, meta, out_dir: Path, already: int, device: torch.device) -> int:
    try:
        target_layer = None
        if hasattr(model, "layer4"):
            target_layer = model.layer4[-1]
        elif hasattr(model, "features"):
            target_layer = model.features[-1]
        if target_layer is None:
            return 0

        act: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []

        def fwd_hook(_, __, output):
            act.append(output.detach())

        def bwd_hook(_, grad_in, grad_out):
            grads.append(grad_out[0].detach())

        handle_fwd = target_layer.register_forward_hook(fwd_hook)
        handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

        out = model(x)
        class_idxs = preds.detach()
        selected = out.gather(1, class_idxs.unsqueeze(1)).sum()
        model.zero_grad()
        selected.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        if not act or not grads:
            return 0
        activations = act[0]
        gradients = grads[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = (cam - cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / (cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-6)

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i in range(min(len(meta), cam.shape[0])):
            heatmap = cam[i].detach().cpu().numpy()
            img = x[i].detach().cpu()
            img = img.permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]))
            heatmap_img = heatmap_img.convert("RGBA")
            base = Image.fromarray(np.uint8(img * 255))
            base = base.convert("RGBA")
            blended = Image.blend(base, heatmap_img, alpha=0.35)
            fname = out_dir / f"gradcam_{already + saved}_{meta[i].get('accession','sample')}.png"
            blended.save(fname)
            saved += 1
        return saved
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("Grad-CAM falhou: %s", exc)
        return 0


def _plot_history(history: List[Dict[str, Any]], outdir: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    csv_path = outdir / "train_history.csv"
    df.to_csv(csv_path, index=False)
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(df["epoch"], df["train_loss"], label="train")
        ax[0].plot(df["epoch"], df["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(df["epoch"], df["train_acc"], label="train")
        ax[1].plot(df["epoch"], df["val_acc"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(outdir / "train_history.png", dpi=150)
        plt.close(fig)
    except Exception:
        LOGGER.debug("Plot de history falhou; salvando apenas CSV.")


def _save_predictions(pred_rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not pred_rows:
        return
    df = pd.DataFrame(pred_rows)
    df.to_csv(outdir / "val_predictions.csv", index=False)


def _maybe_save_metrics(metrics: Dict[str, Any], outdir: Path, best: bool = False) -> None:
    name = "best_metrics.json" if best else "val_metrics.json"
    with open(outdir / name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def _extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    backbone = model
    if isinstance(model, torchvision.models.ResNet):
        backbone.fc = nn.Identity()
    elif isinstance(model, torchvision.models.EfficientNet):
        backbone.classifier = nn.Identity()
    feats: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    for x, y, meta in tqdm(loader, desc="Extracting", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type in {"cuda", "mps"}):
            emb = backbone(x)
        feats.append(emb.detach().cpu().numpy())
        for i, m in enumerate(meta):
            metas.append({**m, "label": int(y[i].item()) if isinstance(y, torch.Tensor) else None})
    return np.concatenate(feats, axis=0), metas


def run_training(args) -> None:
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    configure_runtime(args.allow_tf32, args.deterministic)
    LOGGER.info("Device: %s", device)

    df = _parse_input_source(args)
    df["target"] = df["professional_label"].apply(lambda x: _map_label(x, args.task))
    df = df.dropna(subset=["target"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, df["target"], groups=df["accession"]))
    rows_train = df.iloc[train_idx].to_dict("records")
    rows_val = df.iloc[val_idx].to_dict("records")

    cache_dir = Path(args.outdir) / "cache"
    cache_mode_train = resolve_dataset_cache_mode(args.cache_mode, rows_train)
    cache_mode_val = resolve_dataset_cache_mode(args.cache_mode, rows_val)

    ds_train = UnifiedMammoDataset(rows_train, args.img_size, args.task, train=True, cache_mode=cache_mode_train, cache_dir=cache_dir, split_name="train")
    ds_val = UnifiedMammoDataset(rows_val, args.img_size, args.task, train=False, cache_mode=cache_mode_val, cache_dir=cache_dir, split_name="val")

    nw, prefetch, persistent = _resolve_loader_runtime(args, device)
    sampler = None
    if args.sampler_weighted:
        targets = [r.get("professional_label") for r in rows_train]
        mapped = [_map_label(t, args.task) for t in targets]
        mapped = [m for m in mapped if m is not None]
        counts = np.bincount(np.array(mapped, dtype=int), minlength=2 if args.task == "binary" else 4)
        weights = torch.tensor(len(mapped) / (counts + 1e-6), dtype=torch.float)
        sample_weights = torch.tensor([weights[_map_label(r.get("professional_label"), args.task)] for r in rows_train], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    common_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": nw,
        "persistent_workers": bool(persistent and nw > 0),
        "pin_memory": device.type == "cuda",
        "collate_fn": _collate_fn,
    }
    if prefetch is not None:
        common_loader_kwargs["prefetch_factor"] = prefetch

    dl_train = DataLoader(
        ds_train,
        shuffle=sampler is None,
        sampler=sampler,
        **common_loader_kwargs,
    )
    dl_val = DataLoader(
        ds_val,
        shuffle=False,
        **common_loader_kwargs,
    )

    num_classes = 2 if args.task == "binary" else 4
    model = build_model(args.model, num_classes).to(device)

    if args.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            LOGGER.info("torch.compile ativado.")
        except Exception as exc:
            LOGGER.warning("torch.compile falhou; seguindo sem compile: %s", exc)

    if args.freeze_backbone:
        _freeze_backbone(model, args.model)
    if args.unfreeze_last_block:
        _unfreeze_last_block(model, args.model)

    param_groups = _build_param_groups(model, args.model, args.lr, args.backbone_lr, train_backbone=not args.freeze_backbone or args.train_backbone)
    optim_kwargs = {"weight_decay": args.weight_decay}
    if args.fused_optim and device.type == "cuda":
        try:
            optim_kwargs["fused"] = True
        except Exception:
            pass
    optimizer = torch.optim.AdamW(param_groups, **optim_kwargs)

    weights = None
    if args.class_weights == "auto":
        targets = df.iloc[train_idx]["target"].values.astype(int)
        counts = np.bincount(targets, minlength=num_classes)
        weights = torch.tensor(len(targets) / (num_classes * counts + 1e-6), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    scheduler = None
    if args.lr_reduce_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_reduce_factor,
            patience=args.lr_reduce_patience,
            min_lr=args.lr_reduce_min_lr,
            cooldown=args.lr_reduce_cooldown,
        )

    best_acc = -1.0
    best_epoch = -1
    patience_ctr = 0
    history: List[Dict[str, Any]] = []
    gradcam_dir = Path(args.outdir) / "gradcam" if args.gradcam else None

    for epoch in range(args.epochs):
        if args.warmup_epochs and epoch < args.warmup_epochs:
            _freeze_backbone(model, args.model)
        elif args.train_backbone:
            for p in model.parameters():
                p.requires_grad = True
            if args.unfreeze_last_block:
                _unfreeze_last_block(model, args.model)

        prof_ctx = None
        if args.profile and epoch == 0:
            Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
            activities = [profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(profiler.ProfilerActivity.CUDA)
            prof_ctx = profiler.profile(activities=activities, record_shapes=False)
            prof_ctx.__enter__()

        train_loss, train_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device, scaler)

        if prof_ctx is not None:
            try:
                prof_ctx.__exit__(None, None, None)
                prof_ctx.export_chrome_trace(Path(args.profile_dir) / "trace.json")
                LOGGER.info("Trace salvo em %s", Path(args.profile_dir) / "trace.json")
            except Exception as exc:
                LOGGER.warning("Falha ao salvar trace do profiler: %s", exc)
        val_metrics, pred_rows = validate(
            model,
            dl_val,
            loss_fn,
            device,
            args.task,
            collect_preds=args.save_val_preds,
            gradcam=args.gradcam,
            gradcam_dir=gradcam_dir,
            gradcam_limit=args.gradcam_limit,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics.get("loss", 0.0),
                "val_acc": val_metrics.get("acc", 0.0),
                "val_auc": val_metrics.get("auc", 0.0),
                "val_kappa": val_metrics.get("kappa", 0.0),
            }
        )
        LOGGER.info(
            "Epoch %s/%s | Train %.4f acc %.4f | Val %.4f acc %.4f kappa %.4f auc %.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_metrics.get("loss", 0.0),
            val_metrics.get("acc", 0.0),
            val_metrics.get("kappa", 0.0),
            val_metrics.get("auc", 0.0),
        )

        _plot_history(history, Path(args.outdir))
        if args.save_val_preds:
            _save_predictions(pred_rows, Path(args.outdir))
        _maybe_save_metrics(val_metrics, Path(args.outdir), best=False)

        improved = val_metrics.get("acc", 0.0) > best_acc + args.early_stop_min_delta
        if improved:
            best_acc = val_metrics.get("acc", 0.0)
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), Path(args.outdir) / "best_model.pt")
            _maybe_save_metrics(val_metrics, Path(args.outdir), best=True)
        else:
            patience_ctr += 1

        if scheduler is not None:
            scheduler.step(val_metrics.get("acc", 0.0))

        if args.early_stop_patience and patience_ctr >= args.early_stop_patience:
            LOGGER.info("Early stopping ativado (sem melhoria por %s épocas).", patience_ctr)
            break

    LOGGER.info("Melhor época: %s | melhor acc=%.4f", best_epoch + 1, best_acc)

    if args.export_val_embeddings:
        LOGGER.info("Extraindo embeddings do conjunto de validação...")
        best_path = Path(args.outdir) / "best_model.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state, strict=False)
        feats, metas = _extract_embeddings(model, dl_val, device)
        np.save(Path(args.outdir) / "embeddings_val.npy", feats)
        pd.DataFrame(metas).to_csv(Path(args.outdir) / "embeddings_val.csv", index=False)


# ==============================================================================
# Extração e análise (Stage 1)
# ==============================================================================


def _save_first_image_preview(dataset: UnifiedMammoDataset, out_dir: Path) -> None:
    if len(dataset) == 0:
        return
    img, _, meta = dataset[0]
    arr = img.permute(1, 2, 0).numpy()
    arr = (arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(arr * 255)).save(out_dir / "first_image_loaded.png")


def _save_samples_grid(dataset: UnifiedMammoDataset, out_dir: Path, max_samples: int = 16) -> None:
    if len(dataset) == 0:
        return
    idxs = list(range(min(max_samples, len(dataset))))
    imgs = []
    for i in idxs:
        img, _, meta = dataset[i]
        arr = img.permute(1, 2, 0).numpy()
        arr = (arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        imgs.append((arr, meta))
    if not imgs:
        return
    cols = int(math.sqrt(len(imgs))) or 1
    rows = math.ceil(len(imgs) / cols)
    h, w, _ = imgs[0][0].shape
    grid = Image.new("RGB", (w * cols, h * rows))
    for idx, (arr, meta) in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        patch = Image.fromarray(np.uint8(arr * 255))
        draw = ImageDraw.Draw(patch)
        draw.text((4, 4), str(meta.get("accession", "?"))[:12], fill=(255, 0, 0))
        grid.paste(patch, (c * w, r * h))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / "samples_grid.png")


def _plot_labels_distribution(df: pd.DataFrame, out_dir: Path) -> None:
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


def _run_pca_tsne_umap(features: np.ndarray, meta: List[Dict[str, Any]], args, out_dir: Path) -> pd.DataFrame:
    df_meta = pd.DataFrame(meta)
    joined = df_meta.copy()

    if args.pca and features.shape[0] > 1:
        pca = PCA(n_components=2, random_state=args.seed)
        pca_res = pca.fit_transform(features)
        joined[["pca_x", "pca_y"]] = pca_res
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(pca_res[:, 0], pca_res[:, 1], c=df_meta.get("raw_label", pd.Series([0]*len(df_meta))), cmap="viridis", s=8)
        fig.colorbar(scatter, ax=ax, label="Label")
        ax.set_title("PCA 2D")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_2d.png", dpi=150)
        plt.close(fig)

    if args.tsne and features.shape[0] > 2:
        tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto", perplexity=min(30, max(2, features.shape[0] - 1)))
        tsne_res = tsne.fit_transform(features)
        joined[["tsne_x", "tsne_y"]] = tsne_res
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(tsne_res[:, 0], tsne_res[:, 1], c=df_meta.get("raw_label", pd.Series([0]*len(df_meta))), cmap="plasma", s=8)
        fig.colorbar(scatter, ax=ax, label="Label")
        ax.set_title("t-SNE 2D")
        fig.tight_layout()
        fig.savefig(out_dir / "tsne_2d.png", dpi=150)
        plt.close(fig)

    if args.umap and features.shape[0] > 5:
        try:
            from umap import UMAP  # type: ignore

            umap_model = UMAP(n_components=2, random_state=args.seed)
            umap_res = umap_model.fit_transform(features)
            joined[["umap_x", "umap_y"]] = umap_res
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(umap_res[:, 0], umap_res[:, 1], c=df_meta.get("raw_label", pd.Series([0]*len(df_meta))), cmap="Spectral", s=8)
            fig.colorbar(scatter, ax=ax, label="Label")
            ax.set_title("UMAP 2D")
            fig.tight_layout()
            fig.savefig(out_dir / "umap_2d.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            LOGGER.warning("UMAP não disponível: %s", exc)

    if args.cluster_auto or args.cluster_k:
        k_values = [args.cluster_k] if args.cluster_k else list(range(2, min(8, features.shape[0] + 1)))
        best_k, best_score, best_labels = None, -np.inf, None
        history = []
        for k in k_values:
            if k < 2 or k >= features.shape[0]:
                continue
            km = KMeans(n_clusters=k, random_state=args.seed)
            labels = km.fit_predict(features)
            try:
                score = silhouette_score(features, labels)
            except Exception:
                score = -np.inf
            history.append({"k": k, "silhouette": float(score)})
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        if best_labels is not None:
            joined["cluster_kmeans"] = best_labels
            fig, ax = plt.subplots(figsize=(6, 4))
            ks = [h["k"] for h in history]
            ss = [h["silhouette"] for h in history]
            ax.plot(ks, ss, marker="o")
            ax.set_xlabel("k")
            ax.set_ylabel("silhouette")
            ax.set_title("k-means (auto)")
            fig.tight_layout()
            fig.savefig(out_dir / "kmeans_metrics.png", dpi=150)
            plt.close(fig)
            if args.save_csv:
                pd.DataFrame(history).to_csv(out_dir / "clustering_metrics.csv", index=False)
    return joined


def run_extraction(args) -> None:
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    configure_runtime(args.allow_tf32, args.deterministic)
    LOGGER.info("Device: %s", device)

    df = _parse_input_source(args)
    rows = df.to_dict("records")
    cache_dir = Path(args.outdir) / "cache_extract"
    cache_mode = resolve_dataset_cache_mode(args.cache_mode, rows)
    ds = UnifiedMammoDataset(rows, args.img_size, args.task, train=False, augment=False, cache_mode=cache_mode, cache_dir=cache_dir, split_name="extract")
    nw, prefetch, persistent = _resolve_loader_runtime(args, device)
    dl_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": nw,
        "persistent_workers": bool(persistent and nw > 0),
        "pin_memory": device.type == "cuda",
        "collate_fn": _collate_fn,
    }
    if prefetch is not None:
        dl_kwargs["prefetch_factor"] = prefetch
    dl = DataLoader(ds, **dl_kwargs)

    num_classes = 2 if args.task == "binary" else 4
    model = build_model(args.model, num_classes).to(device)
    features, metas = _extract_embeddings(model, dl, device)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.outdir) / "features.npy", features)
    pd.DataFrame(metas).to_csv(Path(args.outdir) / "metadata.csv", index=False)
    sample_embedding = {"embedding": features[0].tolist(), **metas[0]} if len(features) else {}
    with open(Path(args.outdir) / "example_embedding.json", "w", encoding="utf-8") as f:
        json.dump(sample_embedding, f, indent=2)

    preview_dir = Path(args.outdir) / "preview"
    _save_first_image_preview(ds, preview_dir)
    _save_samples_grid(ds, preview_dir, max_samples=args.sample_grid)
    _plot_labels_distribution(df, preview_dir)

    joined = _run_pca_tsne_umap(features, metas, args, Path(args.outdir))
    if args.save_csv and not joined.empty:
        joined.to_csv(Path(args.outdir) / "joined.csv", index=False)
    session_info = {
        "device": str(device),
        "num_samples": len(ds),
        "features_shape": list(features.shape),
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(args.outdir) / "session_info.json", "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2)


# ==============================================================================
# EDA / utilitários (Projeto.py)
# ==============================================================================


def run_eda(args) -> None:
    df = _parse_input_source(args)
    LOGGER.info("Total de amostras: %s", len(df))
    if "professional_label" in df.columns:
        LOGGER.info("Distribuição de labels:\n%s", df["professional_label"].value_counts())
    preview_dir = Path(args.outdir) / "eda"
    rows = df.to_dict("records")
    ds = UnifiedMammoDataset(rows, args.img_size, args.task, train=False, augment=False, cache_mode=resolve_dataset_cache_mode(args.cache_mode, rows), cache_dir=Path(args.outdir) / "cache_eda", split_name="eda")
    _save_first_image_preview(ds, preview_dir)
    _save_samples_grid(ds, preview_dir, max_samples=args.sample_grid)
    _plot_labels_distribution(df, preview_dir)


def run_eval_export(args) -> None:
    config_hint = "Args sugeridos: use --runs outputs/.../results_* para auditar."
    LOGGER.info(config_hint)
    checklist = [
        "Reutilize checkpoints aprovados (outputs/mammo_efficientnetb0_density/results_*).",
        "Exporte val_predictions.csv, embeddings_val.*, metrics/val_metrics.{json,png}.",
        "Gere figuras (ROC, confusion, Grad-CAM) e copie para Article/assets via report-pack.",
        "Versione os arquivos (timestamp + git SHA) na pasta Article/assets/.",
    ]
    for item in checklist:
        LOGGER.info(" • %s", item)
    run_list = getattr(args, "runs", None) or []
    for run in run_list:
        run_path = Path(run)
        LOGGER.info("Auditando artefatos em: %s", run_path)
        required = [
            "summary.json",
            "train_history.csv",
            "train_history.png",
            "val_predictions.csv",
            "best_model.pt",
            "best_metrics.json",
        ]
        missing = [rel for rel in required if not (run_path / rel).exists()]
        if missing:
            LOGGER.warning("Itens faltantes: %s", ", ".join(missing))
        summary_path = run_path / "summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                metrics = payload.get("val_metrics", {})
                LOGGER.info(
                    "Resumo: seed=%s | acc=%.3f | κ=%.3f | macro-F1=%.3f | AUC=%.3f",
                    payload.get("seed", "?"),
                    float(metrics.get("accuracy", 0.0)),
                    float(metrics.get("kappa_quadratic", 0.0)),
                    float(metrics.get("macro_f1", 0.0)),
                    float(metrics.get("auc_ovr", 0.0)),
                )
            except Exception as exc:
                LOGGER.warning("Falha ao ler summary.json: %s", exc)


def run_report_pack(args) -> None:
    if not args.runs:
        raise SystemExit("Informe pelo menos um --runs outputs/.../results_* para empacotar.")
    try:
        from tools import report_pack
    except Exception as exc:
        raise SystemExit(f"report_pack não disponível: {exc}")
    run_paths = [Path(r) for r in args.runs]
    assets_dir = Path(args.assets_dir)
    tex_path = Path(args.tex_path) if args.tex_path else None
    report_pack.package_stage2_runs(run_paths, assets_dir, tex_path=tex_path, gradcam_limit=args.gradcam_limit)


def run_rl_refine_stub(args) -> None:
    LOGGER.info("rl_refinement não disponível; stub executado. Args: %s", " ".join(args.forwarded or []))
    if args.dry_run:
        LOGGER.info("Dry-run habilitado; nenhuma ação realizada.")


# ==============================================================================
# CLI
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Mammography Pipelines")
    parser.add_argument("--mode", choices=["train", "extract", "eda", "eval-export", "report-pack", "rl-refine"], default="train")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0"], default="efficientnet_b0")
    parser.add_argument("--csv", required=True, help="CSV, diretório com featureS.txt ou raiz dos dados")
    parser.add_argument("--dicom-root", default="archive", help="Raiz dos DICOMs quando usar classificacao.csv")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--include-class-5", action="store_true")
    parser.add_argument("--cache-mode", choices=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"], default="auto")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loader-heuristics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--class-weights", choices=["none", "auto"], default="none")
    parser.add_argument("--sampler-weighted", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unfreeze-last-block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-reduce-patience", type=int, default=0)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-reduce-min-lr", type=float, default=1e-7)
    parser.add_argument("--lr-reduce-cooldown", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-optim", action="store_true", help="Usa AdamW com fused=True em CUDA quando suportado")
    parser.add_argument("--torch-compile", action="store_true", help="Otimiza o modelo com torch.compile quando disponível")
    parser.add_argument("--profile", action="store_true", help="Habilita torch.profiler no primeiro epoch e exporta trace")
    parser.add_argument("--profile-dir", default=os.path.join("outputs", "profiler"))
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--gradcam-limit", type=int, default=4)
    parser.add_argument("--save-val-preds", action="store_true")
    parser.add_argument("--export-val-embeddings", action="store_true")
    parser.add_argument("--sample-grid", type=int, default=16, help="Número de exemplos na grade de pré-visualização para extract/eda")

    # Extração/análise
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--cluster-auto", action="store_true")
    parser.add_argument("--cluster-k", type=int, default=0)

    # Projeto.py equivalentes
    parser.add_argument("--runs", action="append")
    parser.add_argument("--assets-dir", default=os.path.join("Article", "assets"))
    parser.add_argument("--tex-path", default=os.path.join("Article", "sections", "stage2_model.tex"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forwarded", nargs=argparse.REMAINDER, help="Argumentos extras para modos stub (rl-refine)")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[TqdmLoggingHandler(), logging.FileHandler(Path(args.outdir) / "run.log")],
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        if args.mode == "train":
            run_training(args)
        elif args.mode == "extract":
            run_extraction(args)
        elif args.mode == "eda":
            run_eda(args)
        elif args.mode == "eval-export":
            run_eval_export(args)
        elif args.mode == "report-pack":
            run_report_pack(args)
        elif args.mode == "rl-refine":
            run_rl_refine_stub(args)
        else:
            parser.error(f"Modo desconhecido: {args.mode}")
        return 0
    except Exception as exc:
        LOGGER.error("Falha na execução: %s", exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
