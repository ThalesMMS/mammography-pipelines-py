#
# dataset.py
# mammography-pipelines
#
# Implements the mammography density Dataset with optional tensor/disk caching and embedding lookup.
#
# Thales Matheus Mendonça Santos - November 2025
#
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_F
from torchvision.transforms.v2 import functional as tv_v2_F

from ..io.dicom import dicom_to_pil_rgb, is_dicom_path

LOGGER = logging.getLogger("mammography")

@dataclass
class EmbeddingStore:
    """Stores 2048-D embeddings indexed by accession and raw path."""

    embeddings_by_accession: Dict[str, torch.Tensor]
    embeddings_by_path: Dict[str, torch.Tensor]
    feature_dim: int

    def lookup(self, row: Dict[str, Any]) -> Optional[torch.Tensor]:
        acc = row.get("accession")
        if acc and acc in self.embeddings_by_accession:
            return self.embeddings_by_accession[acc]
        path = row.get("image_path")
        if path:
            # Try exact path first to avoid surprising matches.
            if path in self.embeddings_by_path:
                return self.embeddings_by_path[path]
            # Normalize to make Windows/Unix paths line up when comparing.
            norm = str(Path(path).expanduser().resolve()).replace("\\", "/").lower()
            if norm in self.embeddings_by_path:
                return self.embeddings_by_path[norm]
        return None


def load_embedding_store(embeddings_dir: str) -> EmbeddingStore:
    """Load embeddings saved by extract_features.py into a lookup store."""
    root = Path(embeddings_dir)
    features_path = root / "features.npy"
    metadata_path = root / "metadata.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"features.npy nao encontrado em {root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv nao encontrado em {root}")

    features = np.load(features_path)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"features.npy deve ter 2 dimensoes, recebido: {features.shape}")

    meta = pd.read_csv(metadata_path)
    if len(meta) != features.shape[0]:
        raise ValueError(
            f"metadata.csv ({len(meta)}) nao bate com features.npy ({features.shape[0]})."
        )

    features_tensor = torch.from_numpy(features).float()
    feature_dim = int(features_tensor.shape[1])
    embeddings_by_accession: Dict[str, torch.Tensor] = {}
    embeddings_by_path: Dict[str, torch.Tensor] = {}

    if "accession" in meta.columns:
        for acc, idxs in meta.groupby("accession").indices.items():
            if acc is None or (isinstance(acc, float) and np.isnan(acc)):
                continue
            acc_key = str(acc).strip()
            if not acc_key:
                continue
            acc_feats = features_tensor[torch.tensor(list(idxs), dtype=torch.long)]
            embeddings_by_accession[acc_key] = acc_feats.mean(dim=0)

    path_col = None
    if "path" in meta.columns:
        path_col = "path"
    elif "image_path" in meta.columns:
        path_col = "image_path"
    if path_col:
        for idx, raw_path in enumerate(meta[path_col].fillna("")):
            if not raw_path:
                continue
            path = str(raw_path)
            tensor = features_tensor[idx]
            embeddings_by_path[path] = tensor
            norm = str(Path(path).expanduser().resolve()).replace("\\", "/").lower()
            embeddings_by_path[norm] = tensor

    if not embeddings_by_accession and not embeddings_by_path:
        raise ValueError("metadata.csv nao possui colunas 'accession'/'path' validas.")

    return EmbeddingStore(
        embeddings_by_accession=embeddings_by_accession,
        embeddings_by_path=embeddings_by_path,
        feature_dim=feature_dim,
    )

class MammoDensityDataset(Dataset):
    """Dataset for mammography density classification."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        img_size: int,
        train: bool,
        augment: bool = True,
        augment_vertical: bool = False,
        augment_color: bool = False,
        rotation_deg: float = 5.0,
        cache_mode: str = "none",
        cache_dir: Optional[str] = None,
        split_name: str = "train",
        embedding_store: Optional[EmbeddingStore] = None,
        label_mapper: Optional[callable] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.rows = rows
        self.img_size = img_size
        self.cache_mode = (cache_mode or "none").lower()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name
        self.embedding_store = embedding_store
        self.label_mapper = label_mapper

        valid_cache_modes = {"none", "memory", "disk", "tensor-disk", "tensor-memmap"}
        if self.cache_mode not in valid_cache_modes:
            raise ValueError(f"cache_mode inválido: {self.cache_mode}")
        if self.cache_mode in {"disk", "tensor-disk", "tensor-memmap"} and self.cache_dir is None:
            raise ValueError("cache_dir é obrigatório quando cache_mode requer persistência em disco")

        # Cache indexes are populated at startup so __getitem__ can stay lightweight.
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
        self.augment_vertical = bool(augment_vertical)
        self.augment_color = bool(augment_color)
        self.rotation_deg = float(rotation_deg)
        self._norm_mean = mean or [0.485, 0.456, 0.406]
        self._norm_std = std or [0.229, 0.224, 0.225]
        if len(self._norm_mean) != len(self._norm_std):
            raise ValueError("mean e std devem ter o mesmo tamanho.")
        if len(self._norm_mean) != 3:
            raise ValueError("mean/std devem ter 3 valores para imagens RGB.")

    def __len__(self):
        return len(self.rows)

    def _read_image(self, path: str) -> Image.Image:
        if is_dicom_path(path):
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
        iterable = self.rows
        for row in tqdm(iterable, desc=f"Cache[{self.split_name}]", leave=False, disable=len(iterable) < 16):
            path = str(row.get("image_path"))
            if not path or not is_dicom_path(path):
                continue
            cache_path = self._cache_path_for(path)
            self._disk_cache_index[path] = str(cache_path)
            if cache_path.exists():
                continue
            try:
                img = self._read_image(path)
                img.save(cache_path, format="PNG")
            except Exception as exc:
                LOGGER.warning("Falha ao materializar cache de %s: %s", path, exc)

    def _prepare_tensor_cache(self) -> None:
        assert self.cache_dir is not None
        iterable = self.rows
        for row in tqdm(iterable, desc=f"TensorCache[{self.split_name}]", leave=False, disable=len(iterable) < 16):
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
                LOGGER.warning("Falha ao decodificar %s para o cache de tensores: %s", path, exc)

    def _decode_to_tensor(self, path: str) -> torch.Tensor:
        img = self._read_image(path)
        return self._convert_to_tensor(img)

    def _materialize_tensor_cache(self, path: str, tensor: torch.Tensor) -> None:
        if self.cache_mode == "tensor-disk":
            cache_path = self._tensor_cache_base_path(path).with_suffix(".pt")
            self._tensor_disk_index[path] = str(cache_path)
            if cache_path.exists():
                return
            try:
                torch.save(tensor, cache_path)
            except Exception as exc:
                LOGGER.warning("Falha ao salvar tensor cache em %s: %s", cache_path, exc)
        elif self.cache_mode == "tensor-memmap":
            base = self._tensor_cache_base_path(path)
            data_path = base.with_suffix(".dat")
            meta_path = base.with_suffix(".json")
            self._tensor_memmap_index[path] = {"data_path": str(data_path), "meta_path": str(meta_path)}
            if data_path.exists() and meta_path.exists():
                return
            arr = tensor.detach().cpu().numpy()
            try:
                mm = np.memmap(data_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
                mm[:] = arr
                mm.flush()
                meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                Path(meta_path).write_text(json.dumps(meta))
            except Exception as exc:
                LOGGER.warning("Falha ao salvar memmap em %s: %s", data_path, exc)

    def _get_base_image(self, path: str) -> Image.Image:
        if self.cache_mode == "disk":
            cache_path = self._disk_cache_index.get(path)
            if cache_path is None and self.cache_dir is not None:
                cache_path = str(self._cache_path_for(path))
                self._disk_cache_index[path] = cache_path
            if cache_path and os.path.exists(cache_path):
                try:
                    with Image.open(cache_path) as im:
                        return im.convert("RGB")
                except Exception:
                     # Tenta ler original
                     pass
            img = self._read_image(path)
            if cache_path and is_dicom_path(path):
                 try:
                    img.save(cache_path, format="PNG")
                 except Exception:
                    pass
            return img

        if self.cache_mode == "memory":
            if self._image_cache is None:
                self._image_cache = {}
            if path in self._image_cache:
                return self._image_cache[path].copy()
            img = self._read_image(path)
            self._image_cache[path] = img
            return img.copy()

        return self._read_image(path)

    def _get_cached_tensor(self, path: str) -> Optional[torch.Tensor]:
        if self.cache_mode == "tensor-disk":
            cache_path = self._tensor_disk_index.get(path)
            if cache_path is None and self.cache_dir is not None:
                cache_path = str(self._tensor_cache_base_path(path).with_suffix(".pt"))
            if cache_path and os.path.exists(cache_path):
                try:
                    # Cache files are created locally; allow full unpickling.
                    return torch.load(cache_path, map_location="cpu", weights_only=False)
                except Exception:
                    return None
        elif self.cache_mode == "tensor-memmap":
            info = self._tensor_memmap_index.get(path)
            if info is None and self.cache_dir is not None:
                base = self._tensor_cache_base_path(path)
                info = {"data_path": str(base.with_suffix(".dat")), "meta_path": str(base.with_suffix(".json"))}
            if info:
                data_path = info.get("data_path")
                meta_path = info.get("meta_path")
                if data_path and meta_path and os.path.exists(data_path) and os.path.exists(meta_path):
                    try:
                         meta = json.loads(Path(meta_path).read_text())
                         shape = tuple(meta["shape"])
                         dtype = np.dtype(meta["dtype"])
                         mm = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
                         return torch.from_numpy(np.array(mm, copy=True))
                    except Exception:
                        return None
        return None

    def __getitem__(self, i: int):
        r = self.rows[i]
        path = str(r.get("image_path", ""))
        try:
            if not path:
                raise ValueError("image_path vazio")
            base_tensor = self._get_cached_tensor(path)
            if base_tensor is None:
                img = self._get_base_image(path)
                base_tensor = self._convert_to_tensor(img)
                if self.cache_mode in {"tensor-disk", "tensor-memmap"}:
                    self._materialize_tensor_cache(path, base_tensor)

            img = self._apply_transforms(base_tensor)
            img = self._to_channels_last(img)

            y = r.get("professional_label")
            if y is not None and not pd.isna(y):
                y = int(y)
                if self.label_mapper:
                    y = self.label_mapper(y)
                else:
                    y = y - 1  # Default mapping 1..4 -> 0..3 for CrossEntropyLoss
            else:
                y = -1  # Keep alignment in the batch even when labels are missing

            embedding_tensor = None
            if self.embedding_store:
                embedding_tensor = self.embedding_store.lookup(r)

            meta = {
                "path": path,
                "accession": r.get("accession"),
                "raw_label": r.get("professional_label"),
            }
            return img, y, meta, embedding_tensor
        except Exception as exc:
            LOGGER.warning("Falha ao carregar amostra idx=%s path=%s: %s", i, path, exc)
            return None

    @staticmethod
    def _convert_to_tensor(img: Image.Image) -> torch.Tensor:
        return tv_v2_F.to_image(img)

    @staticmethod
    def _to_channels_last(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last).squeeze(0)
        return tensor

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        # Normalize to a square crop so EfficientNet receives consistent inputs.
        tensor = tv_v2_F.resize(tensor, [self.img_size], interpolation=InterpolationMode.BICUBIC, antialias=False)
        tensor = tv_v2_F.center_crop(tensor, [self.img_size, self.img_size])
        tensor = tv_v2_F.to_dtype(tensor, torch.float32, scale=True)
        if self.train and getattr(self, "augment", True):
            if float(torch.rand(1)) < 0.5:
                tensor = tv_v2_F.horizontal_flip(tensor)
            if self.augment_vertical and float(torch.rand(1)) < 0.1:
                tensor = tv_v2_F.vertical_flip(tensor)
            if self.rotation_deg > 0:
                angle = float(torch.empty(1).uniform_(-self.rotation_deg, self.rotation_deg))
                tensor = tv_v2_F.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0.0)
            if self.augment_color:
                brightness = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                contrast = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                tensor = tv_F.adjust_brightness(tensor, brightness)
                tensor = tv_F.adjust_contrast(tensor, contrast)
        tensor = tv_v2_F.normalize(tensor, self._norm_mean, self._norm_std)
        return tensor

def robust_collate(batch):
    """Collate that filters failed samples (None) before building the batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta = [b[2] for b in batch]

    emb_list = [b[3] for b in batch if b[3] is not None]
    embeddings = None
    if len(emb_list) == len(batch):
        embeddings = torch.stack(emb_list, dim=0)

    return xs, ys, meta, embeddings


def mammo_collate(batch):
    """Custom collate that keeps metadata as a list of dictionaries."""
    return robust_collate(batch)
