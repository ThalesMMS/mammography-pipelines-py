"""Anotações didáticas para o pipeline RSNA Breast Cancer Detection.

Este script reúne, em formato de caderno executável, as etapas mais comuns
em um fluxo de trabalho de *deep learning* supervisionado para mamografia:

1. **Caracterização da coorte**: leitura dos CSVs oficiais da competição
   RSNA 2022, inspeção rápida de valores ausentes e criação de colunas
   auxiliares para visualização.
2. **Exploração visual**: gráficos com a distribuição de câncer (geral e por
   invasividade) e amostras de DICOM normalizadas em tons de cinza.
3. **Preparação de dados**: balanceamento da classe minoritária, imputação e
   normalização de idade, *one-hot encoding* dos metadados categóricos e
   separação treino/validação estratificada.
4. **Dataset PyTorch**: carregamento das imagens PNG 256×256 balanceadas com
   seus respectivos rótulos binários.
5. **Modelo e treino**: ResNet50 pré-treinada (ajustada para canal único)
   empregada de forma direta para classificação binária, seguida por laço de
   treino com métricas clínicas relevantes (sensibilidade/especificidade).
6. **Avaliação gráfica**: acompanhamento da evolução de perda, acurácia e
   métricas sensibilidade/especificidade por época, utilizando Lets-Plot.

Notas práticas
--------------
Para executar o script fora do Kaggle, ajuste os caminhos de entrada/saída por
meio dos argumentos ``--csv-dir``, ``--dicom-dir``, ``--png-dir`` e
``--output-dir``. Eles aceitam pastas locais, permitindo apontar tanto para os
CSVs originais quanto para imagens DICOM/PNG pré-processadas.
"""

# ------------------------------- Imports -------------------------------
import argparse
import copy
import json
import os
import random
import warnings

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
import torch.optim as optim
import torchvision
import pydicom
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from lets_plot import (
    GGBunch,
    LetsPlot,
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    geom_bar,
    geom_imshow,
    geom_line,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    theme,
    theme_minimal,
)
from lets_plot.mapping import as_discrete
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from pydicom.pixel_data_handlers.util import apply_modality_lut

# ---------------------- Hiperparâmetros Didáticos ----------------------
# Esta seção concentra "botões" importantes do pipeline em um só lugar, com
# explicações em linguagem simples. Ajustá-los aqui propaga o efeito pelo script.

class HP:
    """Hiperparâmetros centrais e decisões de pré-processamento.

    - IMG_RESIZE: lado para redimensionar imagens antes do recorte central.
    - IMG_CROP: tamanho final enviado ao modelo (ResNet50 espera 224).
    - WINDOW_P_LOW/HIGH: percentis usados no windowing robusto do DICOM.
      Ideia: recortar extremos (mto claro/escuro) e normalizar o contraste.
    - TRAIN_SPLIT: fração de pacientes no treino (restante vira validação).
    - DATALOADER_WORKERS: número de processos auxiliares para carregar dados.
    - RANDOM_SEED: tomada de decisão pseudo-aleatória reprodutível (splits, etc.).
    """

    IMG_RESIZE: int = 256
    IMG_CROP: int = 224
    WINDOW_P_LOW: float = 0.5
    WINDOW_P_HIGH: float = 99.5
    TRAIN_SPLIT: float = 0.80
    DATALOADER_WORKERS: int = 2
    RANDOM_SEED: int = 19970507
    # Visualização (número de pacientes no grid e tamanho dos tiles)
    VIS_N_PATIENTS: int = 3
    VIS_TILE: int = 300
    VIS_GAP: int = 10
    # Balanceamento (downsampling de negativos)
    DOWNSAMPLE_NEGATIVE: bool = True
    DOWNSAMPLE_RATIO: float = 1.0  # negativos ~ ratio * positivos
    # Pré-processamento de idade
    AGE_IMPUTE: str = "mean"  # 'mean' | 'median'
    AGE_NORM: str = "minmax"   # 'minmax' | 'zscore'
    # Colunas categóricas para one-hot
    CAT_COLS = ['view', 'laterality', 'implant']

# ----------------------- Utilidades integradas -----------------------

def seed_everything(seed: int = 42):
    """Define seeds para reprodutibilidade razoável (atenção: DataLoader workers podem introduzir variação)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_best_data_dir(pref: str) -> str:
    """Tenta localizar a pasta de dados corrigindo grafias comuns."""
    if os.path.isdir(pref):
        return pref
    alt = pref.replace("archieve", "archive")
    if os.path.isdir(alt):
        print(f"[info] data_dir '{pref}' não encontrado; usando '{alt}'")
        return alt
    if os.path.isdir("/kaggle/input"):
        print("[aviso] data_dir não encontrado; verifique o caminho. Usando /kaggle/input para inspeção.")
        return "/kaggle/input"
    return pref


def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Verifica se a imagem é MONOCHROME1 (preto-branco invertidos)."""
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    return photometric == "MONOCHROME1"


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Garante float32 para cálculos numéricos estáveis e previsíveis."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Aplica RescaleSlope/RescaleIntercept (quando presentes) de forma segura."""
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
    """Windowing por percentis para padronizar contraste de mamografias (robusto a outliers)."""
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


def dicom_to_pil_rgb(dcm_path: str) -> Image.Image:
    """Lê um DICOM de mamografia, aplica pré-processamento e retorna PIL Image RGB."""
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

    pil = Image.fromarray(arr, mode="L")
    pil_rgb = Image.merge("RGB", (pil, pil, pil))
    return pil_rgb


def dicom_debug_preprocess(dcm_path: str) -> Dict[str, object]:
    """Versão detalhada do pipeline de pré-processamento para depuração/visualização."""
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array
    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr_raw = arr.copy()

    lo_raw, hi_raw = float(arr_raw.min()), float(arr_raw.max())
    eps = 1e-6 if hi_raw - lo_raw == 0 else 0.0
    arr_raw_mm = (arr_raw - lo_raw) / (hi_raw - lo_raw + eps)
    raw_uint8 = (arr_raw_mm * 255.0).clip(0, 255).astype(np.uint8)

    arr_win = robust_window(arr, HP.WINDOW_P_LOW, HP.WINDOW_P_HIGH)
    win_uint8 = (arr_win * 255.0).clip(0, 255).astype(np.uint8)

    pil_raw = Image.fromarray(raw_uint8, mode="L")
    pil_win = Image.fromarray(win_uint8, mode="L")
    pil_raw_rgb = Image.merge("RGB", (pil_raw, pil_raw, pil_raw))
    pil_win_rgb = Image.merge("RGB", (pil_win, pil_win, pil_win))

    vis_tf = transforms.Compose([
        transforms.Resize(HP.IMG_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(HP.IMG_CROP),
    ])
    pil_resized = vis_tf(pil_win_rgb)

    safe_header = {}
    safe_keys = [
        "Manufacturer",
        "ManufacturerModelName",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "BitsStored",
        "BitsAllocated",
        "HighBit",
        "PixelRepresentation",
        "RescaleIntercept",
        "RescaleSlope",
        "ViewPosition",
        "Laterality",
        "BodyPartExamined",
        "SeriesDescription",
        "SOPClassUID",
        "Modality",
    ]
    for k in safe_keys:
        if hasattr(ds, k):
            v = getattr(ds, k)
            try:
                safe_header[k] = str(v)
            except Exception:
                pass

    return {
        "raw_uint8": raw_uint8,
        "win_uint8": win_uint8,
        "pil_raw_rgb": pil_raw_rgb,
        "pil_win_rgb": pil_win_rgb,
        "pil_resized_rgb": pil_resized,
        "safe_header": safe_header,
        "shape_raw": [int(arr_raw.shape[0]), int(arr_raw.shape[1])],
    }


@dataclass
class SampleInfo:
    """Container leve com os metadados necessários para reconstruir cada amostra."""
    accession: str
    classification: Optional[int]
    path: str
    idx: int


class MammoDicomDataset(Dataset):
    """Dataset que varre subpastas de data_dir e retorna amostras pré-processadas."""

    def __init__(
        self,
        data_dir: str,
        labels_by_accession: Dict[str, int],
        exclude_class_5: bool = True,
        include_unlabeled: bool = False,
        transform: Optional[torch.nn.Module] = None,
        exts: Tuple[str, ...] = (".dcm", ".dicom", ".DCM", ".DICOM"),
    ):
        self.data_dir = data_dir
        self.labels_by_accession = labels_by_accession
        self.exclude_class_5 = exclude_class_5
        self.include_unlabeled = include_unlabeled
        self.transform = transform
        self.exts = exts

        self.samples: List[SampleInfo] = []
        self._build_index()

    def _list_dirs(self, root: str) -> List[str]:
        return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

    def _find_first_dicom(self, folder: str) -> Optional[str]:
        dicoms = []
        for curr, _, files in os.walk(folder):
            for f in files:
                fp = os.path.join(curr, f)
                name = f.lower()
                if fp.endswith(self.exts) or name.endswith(".dcm") or name.endswith(".dicom"):
                    dicoms.append(fp)
        dicoms.sort()
        return dicoms[0] if dicoms else None

    def _build_index(self):
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir '{self.data_dir}' não existe. Verifique o caminho.")

        idx = 0
        for sub in self._list_dirs(self.data_dir):
            accession = str(sub).strip()
            label = self.labels_by_accession.get(accession)

            if label == 5 and self.exclude_class_5:
                continue
            if (label is None) and (not self.include_unlabeled):
                continue

            folder = os.path.join(self.data_dir, sub)
            dcm_path = self._find_first_dicom(folder)
            if dcm_path is None:
                continue

            self.samples.append(SampleInfo(accession=accession, classification=label, path=dcm_path, idx=idx))
            idx += 1

        if len(self.samples) == 0:
            warnings.warn("Nenhuma amostra encontrada. Verifique diretórios e CSV.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        info = self.samples[i]
        img = dicom_to_pil_rgb(info.path)
        if self.transform is not None:
            img = self.transform(img)
        label = info.classification if (info.classification is not None) else -1
        return img, label, info.accession, info.path, info.idx


def _strip_module_prefixes(state_dict: dict) -> dict:
    """Remove prefixos típicos ('module.', etc.) para compatibilidade ao carregar pesos."""
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module."):]
        new_sd[normalized_key] = value
    return new_sd


def _apply_state_dict(model: nn.Module, state: dict) -> nn.Module:
    """Carrega um state_dict no modelo e imprime diferenças relevantes."""
    state_dict = state.get("state_dict", state)
    state_dict = _strip_module_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[aviso] Chaves ausentes no state_dict: {missing[:5]} ...")
    if unexpected:
        print(f"[aviso] Chaves inesperadas no state_dict: {unexpected[:5]} ...")
    return model


def _load_weights_from_path(resnet_ctor, weights_path: Optional[str]) -> Optional[nn.Module]:
    """Retorna um modelo ResNet50 carregado a partir de um caminho explícito (se existir)."""
    if not weights_path or not os.path.isfile(weights_path):
        return None
    print(f"[info] Carregando pesos locais: {weights_path}")
    model = resnet_ctor(weights=None)
    state = torch.load(weights_path, map_location="cpu")
    return _apply_state_dict(model, state)


def _list_cache_candidates(torch_home: Optional[str]) -> List[str]:
    """Retorna uma lista ordenada com os possíveis checkpoints resnet50 encontrados em cache."""
    cache_dirs = []
    if torch_home:
        cache_dirs.append(os.path.join(os.path.abspath(torch_home), "hub", "checkpoints"))
    cache_dirs.append(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
    cache_dirs.append(os.path.join(os.getcwd(), ".torch_cache", "hub", "checkpoints"))

    candidates: List[str] = []
    for directory in cache_dirs:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            lower = fname.lower()
            if lower.startswith("resnet50") and lower.endswith(".pth"):
                candidates.append(os.path.join(directory, fname))
    return candidates


def _load_weights_from_cache(resnet_ctor, torch_home: Optional[str]) -> Optional[nn.Module]:
    """Tenta carregar checkpoints já presentes em cache, imprimindo avisos em caso de falha."""
    for candidate in _list_cache_candidates(torch_home):
        try:
            print(f"[info] Carregando pesos do cache local: {candidate}")
            model = resnet_ctor(weights=None)
            state = torch.load(candidate, map_location="cpu")
            return _apply_state_dict(model, state)
        except Exception as err:
            print(f"[aviso] Falha ao carregar '{candidate}': {err}")
    return None


def _download_resnet_weights(resnet_ctor, avoid_download: bool) -> nn.Module:
    """Baixa pesos do torchvision respeitando a flag --avoid_download e aplicando fallbacks."""
    from torchvision.models import ResNet50_Weights

    if avoid_download:
        raise RuntimeError(
            "Nenhum peso local encontrado e --avoid_download foi definido.\n"
            "Forneça --weights_path ou coloque um 'resnet50-*.pth' em TORCH_HOME/hub/checkpoints."
        )

    try:
        print("[info] Baixando pesos oficiais (ResNet50_Weights.IMAGENET1K_V2)...")
        return resnet_ctor(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception as primary_error:
        print(f"[aviso] Falha ao baixar pesos (SSL/proxy?): {primary_error}")
        print("[dica] macOS: rode 'Install Certificates.command' ou use --weights_path/--avoid_download/--torch_home.")
        try:
            return resnet_ctor(weights=ResNet50_Weights.DEFAULT)
        except Exception as fallback_error:
            raise RuntimeError("Não foi possível obter pesos da ResNet50 (nem baixar, nem cache local).") from fallback_error


def resolve_device(device_choice: str) -> torch.device:
    """Analisa a opção --device e retorna o torch.device apropriado, com fallbacks amigáveis."""
    if device_choice == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_choice == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[aviso] MPS não disponível; usando CPU.")
        return torch.device("cpu")

    if device_choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device("cpu")


def load_labels_dict(csv_path: str) -> Dict[str, int]:
    """Lê o CSV de classificações garantindo que AccessionNumber permaneça como string."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV '{csv_path}' não encontrado. Ajuste --csv_path.")
    df_csv = pd.read_csv(
        csv_path,
        dtype={"AccessionNumber": str, "Classification": int},
        parse_dates=["ClassificationDate"],
        dayfirst=False,
    )
    df_csv["AccessionNumber"] = df_csv["AccessionNumber"].str.strip()
    return {row["AccessionNumber"]: int(row["Classification"]) for _, row in df_csv.iterrows()}


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Retorna transforms de modelo e de visualização coerentes com a ResNet50."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_model = transforms.Compose([
        transforms.Resize(HP.IMG_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(HP.IMG_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_vis = transforms.Compose([
        transforms.Resize(HP.IMG_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(HP.IMG_CROP),
    ])
    return transform_model, transform_vis

# ------------------- Constantes reutilizadas -------------------

NO_CANCER_LABEL = "No Cancer"
COLOR_HEALTHY = "#d8e2dc"
COLOR_PRESENT = "#f4acb7"
COLOR_INVASIVE = "#ee4266"
# Espelha os hiperparâmetros centrais para variáveis usadas adiante
RANDOM_SEED = HP.RANDOM_SEED
TRAIN_SPLIT_RATIO = HP.TRAIN_SPLIT
DATALOADER_WORKERS = HP.DATALOADER_WORKERS

# So the plots look nice
LetsPlot.setup_html()
random.seed(RANDOM_SEED)

# --------------------------- Argumentos de caminho ---------------------------

parser = argparse.ArgumentParser(
    description="RSNA Breast Cancer Detection pipeline com caminhos configuráveis.",
)
parser.add_argument(
    "--csv-dir",
    default=os.environ.get("RSNA_CSV_DIR", "/kaggle/input/rsna-breast-cancer-detection"),
    help="Diretório contendo train.csv/test.csv da competição.",
)
parser.add_argument(
    "--dicom-dir",
    default=os.environ.get(
        "RSNA_DICOM_DIR", "/kaggle/input/rsna-breast-cancer-detection/train_images"
    ),
    help="Diretório com as imagens DICOM originais para visualização.",
)
parser.add_argument(
    "--png-dir",
    default=os.environ.get(
        "RSNA_PNG_DIR", "/kaggle/input/rsna-breast-cancer-256-pngs"
    ),
    help="Diretório com as imagens PNG 256×256 utilizadas no dataset balanceado.",
)
parser.add_argument(
    "--output-dir",
    default=os.environ.get("RSNA_OUTPUT_DIR", "/kaggle/working"),
    help="Diretório raiz onde as cópias balanceadas (train/valid) serão salvas.",
)

# Em ambientes como notebooks, argumentos extras podem ser injetados pelo kernel.
ARGS, _ = parser.parse_known_args()

CSV_DIR = ARGS.csv_dir
TRAIN_CSV_PATH = os.path.join(CSV_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(CSV_DIR, "test.csv")

if not os.path.isfile(TRAIN_CSV_PATH):
    alt_csv_dir = find_best_data_dir(CSV_DIR)
    if alt_csv_dir != CSV_DIR:
        print(
            f"[info] train.csv não encontrado em '{CSV_DIR}'; usando diretório alternativo '{alt_csv_dir}'."
        )
        CSV_DIR = alt_csv_dir
        TRAIN_CSV_PATH = os.path.join(CSV_DIR, "train.csv")
        TEST_CSV_PATH = os.path.join(CSV_DIR, "test.csv")

DICOM_DATA_DIR = find_best_data_dir(ARGS.dicom_dir)
PNG_DATA_DIR = find_best_data_dir(ARGS.png_dir)
OUTPUT_DIR = ARGS.output_dir
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "train")
VALID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "valid")

# -------------------- Data and Cohort Characteristics --------------------

# Leitura dos CSVs oficiais. No Kaggle, os arquivos ficam em /kaggle/input.
# O `glimpse` (Polars) é um equivalente ao `df.info()` que exibe tipo e amostra.
# Discussão: optamos por manter a leitura direta dos CSVs originais para
# destacar o formato nativo da competição. Em cenários clínicos, essa primeira
# inspeção costuma revelar problemas de integridade (pacientes duplicados,
# outliers de idade) antes de pensar em técnicas sofisticadas.
df_train = pl.read_csv(TRAIN_CSV_PATH)
df_test = pl.read_csv(TEST_CSV_PATH)

df_train.glimpse()

# Verificação rápida de valores ausentes para entender onde é necessário
# imputar ou descartar registros. O print serve de cabeçalho amigável.
print('Missing values by column')
df_train.select(pl.all().is_null().sum())

# ----------------------- Exploratory Data Analysis ----------------------

# Conjunto auxiliar com colunas categóricas legíveis (strings) para gráficos.
# `target` não é usado diretamente, mas mantém o código alinhado à estrutura
# de outros cadernos onde poderíamos alternar rótulos.
# Discussão: preferimos derivar colunas textuais em vez de mapear legendas no
# próprio gráfico. Isso facilita replicar a mesma tabela em relatórios clínicos
# sem depender da camada de visualização.
target = 'cancer'

df_plt = df_train.with_columns(
    pl.when(pl.col('cancer') == 1).then('Cancer Present').otherwise(NO_CANCER_LABEL).alias('cancer'),
    # Coluna `invasive2`: combina status de câncer + invasividade em texto.
    # Serve para colorir gráficos com três categorias principais.
    pl.when((pl.col('cancer') == 1) & (pl.col('invasive') == 1))
        .then('Invasive')
        .when((pl.col('cancer') == 1) & (pl.col('invasive') == 0))
        .then('Non-Invasive')
        .when((pl.col('cancer') == 0) & (pl.col('invasive') == 0))
        .then(NO_CANCER_LABEL)
        .otherwise(NO_CANCER_LABEL)
        .alias('invasive2')
)

# --------------------- Gráfico da variável alvo ---------------------
# A ideia é manter a mesma linguagem visual para todos os gráficos Lets-Plot:
# tema minimalista, paleta consistente e legendas simplificadas.
# O `coord_flip` inverte eixos para ganhar espaço em labels.
var = 'cancer'
title = 'Cancer Distribution'
legend_title = ''

plt1 = \
    ggplot(df_plt)+\
    geom_bar(aes(x = as_discrete(var),
                fill = as_discrete(var)),
            color = None,
            size = 0.5)+\
    scale_fill_manual(values = [COLOR_HEALTHY, COLOR_PRESENT])+\
    theme_minimal()+\
    theme(
        plot_title = element_text(hjust = 0.5, face = 'bold'),
        legend_position = "top",
        panel_grid_major = element_blank(),
        panel_grid_minor = element_blank(),
        legend_title = element_blank(),
        axis_title_x = element_blank(),
        axis_line_y = element_line(size = 1))+\
    coord_flip()+\
    labs(y = "Count", title = title)

# Distribuição de câncer estratificada por invasividade (3 categorias).
# A paleta utiliza uma cor distinta para cada subgrupo clínico.
var = 'invasive2'
title = 'Cancer Distribution by Invasiveness'
legend_title = ''

plt2 = \
    ggplot(df_plt)+\
    geom_bar(aes(x = as_discrete(var),
                fill = as_discrete(var)),
            color = None,
            size = 0.5)+\
    scale_fill_manual(values = [COLOR_HEALTHY, COLOR_INVASIVE, COLOR_PRESENT])+\
    theme_minimal()+\
    theme(
        plot_title = element_text(hjust = 0.5, face = 'bold'),
        legend_position = "top",
        panel_grid_major = element_blank(),
        panel_grid_minor = element_blank(),
        legend_title = element_blank(),
        axis_title_x = element_blank(),
        axis_line_y = element_line(size = 1))+\
    coord_flip()+\
    labs(y = "Count", title = title)



bunch = GGBunch()
bunch.add_plot(plt1, 0, 0, 500, 250)
bunch.add_plot(plt2, 520, 0, 500, 250)
bunch.show()

# --------------------- Opcional: pipeline de densidade ---------------------
# Integração leve: defina RUN_DENSITY=1 no ambiente para rodar o pipeline de
# densidade (ResNet50) usando seu classificacao.csv e os DICOMs em `archive/`.
# Variáveis de ambiente opcionais:
#   DENSITY_CSV (default: classificacao.csv)
#   DENSITY_DICOM_ROOT (default: archive)
#   DENSITY_OUTDIR (default: outputs/mammo_resnet50_density)
#   DENSITY_EPOCHS, DENSITY_BATCH_SIZE, DENSITY_IMG_SIZE, DENSITY_DEVICE
try:
    if os.environ.get("RUN_DENSITY", "0") == "1":
        from types import SimpleNamespace
        from RSNA_Mammo_ResNet50_Density import run as run_density
        args = SimpleNamespace(
            csv=os.environ.get("DENSITY_CSV", "classificacao.csv"),
            dicom_root=os.environ.get("DENSITY_DICOM_ROOT", "archive"),
            outdir=os.environ.get("DENSITY_OUTDIR", "outputs/mammo_resnet50_density"),
            epochs=int(os.environ.get("DENSITY_EPOCHS", "8")),
            batch_size=int(os.environ.get("DENSITY_BATCH_SIZE", "16")),
            num_workers=4,
            img_size=int(os.environ.get("DENSITY_IMG_SIZE", "512")),
            lr=1e-4,
            weight_decay=1e-4,
            val_frac=0.2,
            seed=42,
            device=os.environ.get("DENSITY_DEVICE", "auto"),
            train_backbone=False,
            unfreeze_last_block=True,
            include_class_5=False,
        )
        print("[info] RUN_DENSITY=1 detectado — iniciando pipeline de densidade...")
        run_density(args)
except Exception as _density_err:
    print(f"[aviso] Pipeline de densidade não executado: {_density_err}")

# ------------------------- Visualização de DICOMs -------------------------

# Função utilitária: reproduz exatamente o pipeline de pré-processamento da
# ResNet50 (windowing robusto + conversão para RGB) e devolve figura Lets-Plot.
# Ao reutilizar o mesmo helper do script principal garantimos consistência entre
# o preview e o que entra na rede.
def get_dicom_plt(dcm_path: str, title: str):
    """Cria figura Lets-Plot replicando o pré-processamento completo da ResNet50."""
    dbg = dicom_debug_preprocess(dcm_path)
    img_rgb = np.array(dbg["pil_win_rgb"])

    plt = \
        ggplot() + \
        geom_imshow(img_rgb) + \
        theme(
            legend_position='none',
            panel_grid=element_blank(),
            axis=element_blank(),
            plot_title=element_text(hjust=0.5, face='bold')) + \
        labs(title=title)

    return plt

# Recupera subconjuntos de pacientes por categoria clínica para amostragem.
# Selecionamos apenas alguns IDs para montar um grid compacto de imagens.
invasive_patients = df_plt.filter(pl.col('invasive2') == 'Invasive').select(['patient_id', 'image_id'])
invasive_patient_ids = invasive_patients.get_column('patient_id')
invasive_img_ids = invasive_patients.get_column('image_id')

noninvasive_patients = df_plt.filter(pl.col('invasive2') == 'Non-Invasive').select(['patient_id', 'image_id'])
noninvasive_patient_ids = noninvasive_patients.get_column('patient_id')
noninvasive_img_ids = noninvasive_patients.get_column('image_id')

no_cancer_patients = df_plt.filter(pl.col('invasive2') == NO_CANCER_LABEL).select(['patient_id', 'image_id'])
no_cancer_patient_ids = no_cancer_patients.get_column('patient_id')
no_cancer_img_ids = no_cancer_patients.get_column('image_id')

# Diretório raiz das imagens DICOM originais. `npatients` define quantas
# amostras por classe serão exibidas (3 aqui para manter o grid enxuto).
# Discussão: o recorte de três pacientes por grupo é deliberado para priorizar
# velocidade em ambientes compartilhados (como notebooks públicos) sem perder
# a noção de variação intra-classe nas imagens.
dicom_dir = DICOM_DATA_DIR
npatients = HP.VIS_N_PATIENTS
tile = HP.VIS_TILE
gap = HP.VIS_GAP
bunch = GGBunch()

# Renderiza um grid 3×3 (linhas: classes clínicas, colunas: pacientes).
for i in range(npatients):

    # For invasive cancer patients
    dcm_path = os.path.join(
        dicom_dir, str(invasive_patient_ids[i]), f"{invasive_img_ids[i]}.dcm"
    )
    bunch.add_plot(get_dicom_plt(dcm_path, title = 'Invasive Cancer'), 0 + i*(tile), 0, tile, tile)

    # For noninvasive cancer patients
    dcm_path = os.path.join(
        dicom_dir, str(noninvasive_patient_ids[i]), f"{noninvasive_img_ids[i]}.dcm"
    )
    bunch.add_plot(get_dicom_plt(dcm_path, title = 'Non-Invasive Cancer'), 0 + i*(tile), tile + gap, tile, tile)

    # For those without cancer
    dcm_path = os.path.join(
        dicom_dir, str(no_cancer_patient_ids[i]), f"{no_cancer_img_ids[i]}.dcm"
    )
    bunch.add_plot(get_dicom_plt(dcm_path, title=NO_CANCER_LABEL), 0 + i*(tile), 2*(tile + gap), tile, tile)
                   
# Exibe a coletânea de figuras no notebook (Lets-Plot).
bunch.show()

# ------------------------- Data Set and Data Loader ------------------------

# Balanceamento simples: a classe "câncer" é bem minoritária. Aqui fazemos
# *downsampling* aleatório dos negativos para igualar o número de positivos.
# Em produção consideraríamos estratégias mais robustas (ex.: `class_weight`).
# Discussão: o downsampling reduz a sensibilidade estatística, mas é uma forma
# rápida de tornar o protótipo treinável sem precisar recalcular pesos por
# batch. A intenção é enfatizar a mecânica de balanceamento, não otimizar AUC.
df_target1 = df_train.filter(pl.col('cancer') == 1)
df_target0 = df_train.filter(pl.col('cancer') == 0)

# Didático: quando HP.DOWNSAMPLE_NEGATIVE=True, reduzimos o número de negativos
# para manter uma razão controlada em relação aos positivos. Isso acelera o
# treino e evita que a classe majoritária domine todas as iterações.
if HP.DOWNSAMPLE_NEGATIVE:
    n_pos = len(df_target1)
    n_neg_total = len(df_target0)
    n_neg_keep = min(n_neg_total, int(round(HP.DOWNSAMPLE_RATIO * n_pos)))
    df_target0 = (
        df_target0
        .with_row_count()
        .filter(pl.col('row_nr').is_in(random.sample(range(n_neg_total), n_neg_keep)))
        .drop('row_nr')
    )

# Une as classes (balanceadas ou não) e embaralha as linhas para quebrar a ordem.
df_keep = (
    pl.concat([df_target1, df_target0], how='vertical')
    .select(pl.all().shuffle(seed=RANDOM_SEED))
)

# --- Preparação de metadados tabulares ---
# Idade possui valores ausentes: imputamos com a média e normalizamos
# linearmente para [0, 1], útil para estabilidade numérica da rede tabular.
if HP.AGE_IMPUTE == 'median':
    age_fill = float(df_keep.get_column('age').median())
else:
    age_fill = float(round(df_keep.get_column('age').mean()))

age_min = df_keep.get_column('age').min()
age_max = df_keep.get_column('age').max()
age_mean = float(df_keep.get_column('age').mean())
age_std = float(df_keep.get_column('age').std()) or 1.0

# Imputação
df_keep = df_keep.with_columns(
    pl.when(pl.col('age') == None)
    .then(age_fill)
    .otherwise(pl.col('age'))
    .alias('age')
)

# Normalização
if HP.AGE_NORM.lower() == 'zscore':
    df_keep = df_keep.with_columns(((pl.col('age') - age_mean) / age_std).alias('age'))
else:
    df_keep = df_keep.with_columns(((pl.col('age') - age_min) / (age_max - age_min)).alias('age'))

# One-hot encoding de colunas categóricas definidas em HP
df_keep = df_keep.to_dummies(HP.CAT_COLS)

# Adiciona coluna "trainvalid" particionando por paciente para evitar que um
# mesmo indivíduo apareça nos dois lados do split. O `GroupShuffleSplit` mantém
# a proporção desejada (80/20, definida por TRAIN_SPLIT_RATIO) e reflete a
# prática clínica de avaliar cada paciente em apenas um conjunto, minimizando
# vazamento contralateral.
patients = df_keep.get_column('patient_id').to_list()
gss = GroupShuffleSplit(
    n_splits=1,
    train_size=TRAIN_SPLIT_RATIO,
    random_state=RANDOM_SEED,
)
train_idx, _ = next(gss.split(np.zeros(len(patients)), groups=patients))
train_patients = list(
    set(df_keep.get_column('patient_id').take(list(train_idx)).to_list())
)

df_keep = df_keep.with_columns(
    pl.when(pl.col('patient_id').is_in(train_patients))
    .then('train')
    .otherwise('valid')
    .alias('trainvalid')
)
    
# Construção do nome do arquivo PNG (formato `{patient_id}_{image_id}.png`).
# Essa etapa permite ligar os metadados tabulares ao arquivo de imagem.
df_keep = df_keep\
    .with_columns(pl.lit('_').alias('underscore'))\
    .with_columns(
        pl.concat_str(
            [
                pl.col('patient_id'),
                pl.col('underscore'),
                pl.col('image_id')
            ]
        ).alias('fname')
    ).drop('underscore')

df_train_meta = df_keep.filter(pl.col('trainvalid') == 'train')
df_valid_meta = df_keep.filter(pl.col('trainvalid') == 'valid')

train_fnames = df_train_meta.get_column('fname')
valid_fnames = df_valid_meta.get_column('fname')

train_labels = df_train_meta.get_column('cancer').to_numpy()
valid_labels = df_valid_meta.get_column('cancer').to_numpy()

print('Training Metadata Characteristics')
df_train_meta.glimpse()

print('Validation Metadata Characteristics')
df_valid_meta.glimpse()

# Diretorios destino onde salvaremos cópias balanceadas convertidas dos DICOMs.
# Em notebooks Kaggle, /kaggle/working é gravável, mas `--output-dir` permite
# customizar a pasta ao executar localmente.
# Discussão: salvar cópias balanceadas evita acessar diretórios originais a
# cada época, reduzindo gargalos de I/O no Kaggle. Em pipelines orquestrados,
# usaríamos *data loaders* com cache ou *datastores* dedicados.
train_dir = TRAIN_OUTPUT_DIR
valid_dir = VALID_OUTPUT_DIR

# Garantir que as pastas existam antes de salvar as imagens.
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Salva as PNGs balanceadas em pastas separadas (train/valid).
# O uso de `dicom_to_pil_rgb` garante o windowing percentílico e a replicação
# para 3 canais, alinhando-se ao pipeline do extrator com ResNet50.
for file in train_fnames:
    patient_id, image_id = file.split("_", 1)
    dcm_path = os.path.join(dicom_dir, patient_id, f"{image_id}.dcm")
    img = dicom_to_pil_rgb(dcm_path)
    img.save(os.path.join(train_dir, f"{file}.png"))

for file in valid_fnames:
    patient_id, image_id = file.split("_", 1)
    dcm_path = os.path.join(dicom_dir, patient_id, f"{image_id}.dcm")
    img = dicom_to_pil_rgb(dcm_path)
    img.save(os.path.join(valid_dir, f"{file}.png"))

# Defining the data set
class MammographyDataset(Dataset):
    """Dataset PyTorch que fornece pares (imagem RGB, rótulo) balanceados."""

    def __init__(self, meta_df: pl.DataFrame, img_dir: str, transform=None):
        """Armazena amostras balanceadas e o diretório raiz das imagens RGB.

        Parameters
        ----------
        meta_df : pl.DataFrame
            Tabela com colunas auxiliares (incluindo `fname`) para localizar as imagens.
        img_dir : str
            Diretório base (train ou valid) contendo as PNGs geradas dos DICOMs.
        transform : callable, opcional
            Pipeline de transformações do torchvision (ToTensor, aug, etc.).
        """
        self.df = meta_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Retorna quantas amostras estão disponíveis após o balanceamento."""
        return len(self.df)

    def __getitem__(self, idx):
        """Carrega imagem RGB derivada do DICOM, aplica transforms e devolve (imagem, rótulo)."""

        # Recupera o rótulo binário diretamente do DataFrame Polars.
        label = self.df.get_column('cancer')
        label = torch.tensor(label[idx], dtype=torch.float32)

        # Constrói o caminho absoluto da PNG balanceada.
        img_fname = self.df.get_column('fname')
        img_fname = img_fname[idx]

        # Abre a imagem RGB gerada a partir do DICOM e aplica as transformações.
        img_path = f'{self.img_dir}/{img_fname}.png'
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label


def preview_transformed_samples(dataset: Dataset, num_samples: int = 8, seed: Optional[int] = None) -> None:
    """Renderiza um grid com amostras do `train_dataset` após a aplicação do `transform`."""

    if len(dataset) == 0:
        print('Dataset vazio: nada a visualizar.')
        return

    rng = random.Random(seed) if seed is not None else random
    sample_indices = rng.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

    imgs, labels = [], []
    for idx in sample_indices:
        img, label = dataset[idx]

        if img.ndim == 2:
            img = img.unsqueeze(0)

        imgs.append(img)
        labels.append(int(label.item()))

    grid_tensor = make_grid(torch.stack(imgs), nrow=min(4, len(imgs)), normalize=True)
    grid_np = grid_tensor.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    if grid_np.shape[-1] == 1:
        plt.imshow(grid_np[..., 0], cmap='gray')
    else:
        plt.imshow(grid_np)
    plt.title(f'Amostras transformadas (rótulos: {labels})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Pipeline de transformação mínimo: converte PIL RGB -> tensor (C×H×W, float [0, 1]).
# Discussão: mantemos apenas `ToTensor` para isolar o efeito do balanceamento.
# Em aplicações clínicas reais, validaríamos aumentos geométricos com equipe
# médica para garantir plausibilidade anatômica.
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Instancia os datasets de treino/validação apontando para os diretórios
# balanceados montados anteriormente.
train_dataset = MammographyDataset(
    meta_df=df_train_meta,
    img_dir=train_dir,
    transform=transform,
)

valid_dataset = MammographyDataset(
    meta_df=df_valid_meta,
    img_dir=valid_dir,
    transform=transform,
)

# DataLoaders com batch size 64 e *pin_memory* ativado quando houver CUDA.
# Discussão: `num_workers=2` é conservador pensando em ambientes com restrição
# de RAM. Ajustar esse número impacta diretamente o throughput de leitura.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=DATALOADER_WORKERS, pin_memory=torch.cuda.is_available())
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=DATALOADER_WORKERS, pin_memory=torch.cuda.is_available())

# Visualização rápida para checar contrastes, canais e rótulos antes do treino.
preview_transformed_samples(train_dataset)

# ------------------------------- PyTorch Model ------------------------------

class MammographyModel(nn.Module):
    """Classificador baseado apenas em uma ResNet50 adaptada para canal único."""

    def __init__(self):
        """Inicializa a ResNet50 adaptada para imagens mamográficas em tons de cinza."""
        super().__init__()

        self.rnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rnet.fc = nn.Linear(self.rnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Processa a imagem e retorna a probabilidade estimada de câncer."""

        logits = self.rnet(img)
        out = self.sigmoid(logits)
        return out

model = MammographyModel()

# ------------------------- Função de perda e otimizador -------------------------
# Binária (BCELoss) pois a última camada aplica sigmoid. Otimizador SGD clássico
# com momentum/weight decay para ilustrar configuração tradicional.
# Discussão: BCELoss + sigmoid é didático, mas em produção preferiríamos
# `BCEWithLogitsLoss` para estabilidade numérica e possibilidade de calibrar a
# decisão clínica com *logits*.
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# -------------------------------- Training Loop -------------------------------

# Seleciona GPU se disponível; caso contrário usa CPU. O print ajuda a rastrear
# execuções em logs do Kaggle.
# Discussão: deixar o `to(device)` explícito reforça o padrão PyTorch para quem
# está migrando de notebooks CPU-only. A visibilidade do dispositivo evita
# surpresas de performance em sessões desconectadas.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Using device: {device}')

# Quantidade de épocas e buffers para armazenar métricas por época.
npochs = 25
train_loss, valid_loss = [], []
train_accuracy, valid_accuracy = [], []
train_sensitivity, valid_sensitivity = [], []
train_specificity, valid_specificity = [], []

# Helper para métricas clínicas (sensibilidade/especificidade) derivadas da
# matriz de confusão. Mantemos separado para reutilizar no treino e validação.
# Discussão: separar a função facilita substituí-la por métricas adicionais
# (ex.: `balanced_accuracy`) sem poluir o loop principal.
def get_sens_spec(y_true, y_pred):
    """Calcula sensibilidade e especificidade a partir da matriz de confusão."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity

# Laço principal de treino: `model.train()` habilita dropout/BN, e acumulamos
# perdas e predições para computar métricas ao final de cada época.
for epoch in range(npochs):

    # Training section
    model.train()
    running_loss = 0.0
    train_preds, train_labels = [], []

    for batch, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(img)

        loss = loss_fn(outputs.view(-1), label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Guarda predições binárias (threshold 0.5) para métricas agregadas.
        preds = outputs.detach().cpu().view(-1).numpy().round()
        train_preds.extend(preds)
        train_labels.extend(label.cpu().numpy())

        if batch%5 == 0:
            print(f'epoch {epoch + 1}  batch {batch + 1}  train loss: {loss.item():10.8f}')

    # Métricas de treino: sensibilidade/especificidade + perda média do epoch.
    train_sens, train_spec = get_sens_spec(train_labels, train_preds)
    train_sensitivity.append(train_sens)
    train_specificity.append(train_spec)
    avg_train_loss = running_loss/len(train_loader)
    train_loss.append(avg_train_loss)
    train_accuracy.append(accuracy_score(train_labels, train_preds))
    
    # Eval section
    model.eval()
    running_loss = 0.0
    valid_preds, valid_labels = [], []

    with torch.no_grad():
        for batch, (img, label) in enumerate(valid_loader):
            img = img.to(device)
            label = label.to(device)

            outputs = model(img)
            loss = loss_fn(outputs.view(-1), label)
            
            running_loss += loss.item()
            
            # Armazena predições para calcular métricas de validação.
            preds = outputs.detach().cpu().view(-1).numpy().round()
            valid_preds.extend(preds)
            valid_labels.extend(label.cpu().numpy())

    # Métricas de validação espelham as de treino para comparar generalização.
    # Discussão: comparar pares treino/validação por época ajuda a identificar
    # *overfitting* inicial — especialmente importante quando usamos poucos
    # negativos e o modelo tende a decorar padrões.
    valid_sens, valid_spec = get_sens_spec(valid_labels, valid_preds)
    valid_sensitivity.append(valid_sens)
    valid_specificity.append(valid_spec)
    avg_valid_loss = running_loss/len(valid_loader)
    valid_loss.append(avg_valid_loss)
    valid_accuracy.append(accuracy_score(valid_labels, valid_preds))
            
    print('---------------------------------------------------------------------------------')
    print(f'Metrics for epoch {epoch + 1}')
    print(f'Accuracy     train: {train_accuracy[epoch]}  valid: {valid_accuracy[epoch]}')
    print(f'Sensitivity  train: {train_sensitivity[epoch]}  valid: {valid_sensitivity[epoch]}')
    print(f'Specificity  train: {train_specificity[epoch]}  valid: {valid_specificity[epoch]}')
    print('---------------------------------------------------------------------------------')

# ----------------------------------- Evaluation ----------------------------------

# Constrói DataFrames com métricas por época para alimentar Lets-Plot.
# Discussão: consolidar as listas em um `DataFrame` facilita exportar as
# métricas para CSV ou dashboards externos, caso seja necessário reportar
# desempenho longitudinal para a equipe médica.
epoch = list(range(1, npochs + 1)) * 2
set_type = ['Train'] * npochs + ['Valid'] * npochs

# Loss plot
losses = train_loss + valid_loss # Lists

df_plt = pl.DataFrame({'epoch':epoch, 'set_type':set_type,'loss':losses})

plt_loss=\
    ggplot(df_plt)+\
    geom_line(aes(x='epoch',y='loss',color='set_type'),size=2)+\
    labs(x='Epoch',y='Loss', title='Loss Tracking', color = '')+\
    scale_x_continuous(breaks=list(range(1, npochs + 1)))+\
    theme(
        plot_title = element_text(hjust = 0.5, face = 'bold'),
        legend_position = 'top',
        axis_line_y = element_line(size = 1),	
        axis_line_x = element_line(size = 1),
    )

# Accuracy plot
accuracies = train_accuracy + valid_accuracy # Lists

df_plt = pl.DataFrame({'epoch':epoch, 'set_type':set_type,'accuracy':accuracies})

plt_acc=\
    ggplot(df_plt)+\
    geom_line(aes(x='epoch',y='accuracy',color='set_type'),size=2)+\
    labs(x='Epoch',y='Accuracy', title='Accuracy Tracking', color = '')+\
    scale_x_continuous(breaks=list(range(1, npochs + 1)))+\
    theme(
        plot_title = element_text(hjust = 0.5, face = 'bold'),
        legend_position = 'top',
        axis_line_y = element_line(size = 1),	
        axis_line_x = element_line(size = 1),
    )


# Sensitivity plot
sensitivities = train_sensitivity + valid_sensitivity # Lists

df_plt = pl.DataFrame({'epoch':epoch, 'set_type':set_type,'sensitivity':sensitivities})

plt_sens=\
    ggplot(df_plt)+\
    geom_line(aes(x='epoch',y='sensitivity',color='set_type'),size=2)+\
    labs(x='Epoch',y='Sensitivity', title='Sensitivity Tracking', color = '')+\
    scale_x_continuous(breaks=list(range(1, npochs + 1)))+\
    theme(
        plot_title = element_text(hjust = 0.5, face = 'bold'),
        legend_position = 'top',
        axis_line_y = element_line(size = 1),	
        axis_line_x = element_line(size = 1),
    )

# Specificity plot
specificities = train_specificity + valid_specificity # Lists

df_plt = pl.DataFrame({'epoch':epoch, 'set_type':set_type,'specificity':specificities})

plt_spec=\
    ggplot(df_plt)+\
    geom_line(aes(x='epoch',y='specificity',color='set_type'),size=2)+\
    labs(x='Epoch',y='Specificity', title='Specificity Tracking', color = '')+\
    scale_x_continuous(breaks=list(range(1, npochs + 1)))+\
    theme(
        plot_title = element_text(hjust = 0.5, face = 'bold'),
        legend_position = 'top',
        axis_line_y = element_line(size = 1),	
        axis_line_x = element_line(size = 1),
    )

# Agrupa os quatro gráficos em um *bunch* para visualização conjunta.
bunch = GGBunch()
bunch.add_plot(plt_loss, 0, 0, 800, 400)
bunch.add_plot(plt_acc, 0, 410, 800, 400)
bunch.add_plot(plt_sens, 0, 820, 800, 400)
bunch.add_plot(plt_spec, 0, 1230, 800, 400)
bunch.show()

# ============================ Density Classifier ============================

# Esta seção agrega o pipeline completo de treinamento para classificação
# de densidade mamária (categorias BI-RADS 1–4) utilizando uma ResNet50.
# A ideia é manter o projeto autocontido: os blocos abaixo podem ser
# reutilizados a partir deste mesmo arquivo, seja interativamente (import)
# ou via `run_density_classifier_cli()` com uma lista de argumentos.

# ------------------------ Modelo e configuração base ------------------------

def build_resnet50_classifier(
    device: torch.device,
    num_classes: int = 4,
    *,
    weights_path: Optional[str] = None,
    avoid_download: bool = False,
    torch_home: Optional[str] = None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Instancia uma ResNet50 ajustada para classificação multi-classe 1–4.

    A busca de pesos reutiliza a mesma lógica do extrator de embeddings:
    prioriza checkpoints fornecidos pelo usuário, depois o cache local,
    e por fim tenta o download (a menos que ``avoid_download`` seja verdadeiro).
    Sempre que necessário, a camada ``fc`` é substituída por uma ``nn.Linear``
    compatível com ``num_classes``.
    """

    from torchvision.models import resnet50

    if torch_home:
        os.environ["TORCH_HOME"] = os.path.abspath(torch_home)

    model = _load_weights_from_path(resnet50, weights_path)
    if model is None:
        model = _load_weights_from_cache(resnet50, torch_home)
    if model is None:
        model = _download_resnet_weights(resnet50, avoid_download)

    in_features = model.fc.in_features
    if getattr(model.fc, "out_features", None) != num_classes:
        model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    model.to(device)
    return model


# -------------------------- Registro de histórico --------------------------

@dataclass
class DensityHistoryEntry:
    """Snapshot por época para facilitar exportação e plotagem posterior."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float]
    val_acc: Optional[float]


# ------------------------ Treinamento e avaliação -------------------------

def _prepare_targets(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converte rótulos 1–4 para faixa 0–3 e devolve máscara de exemplos válidos."""

    if labels.ndim != 1:
        labels = labels.view(-1)
    mask = labels > 0
    return (labels[mask] - 1).long(), mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Executa uma época de treino e retorna ``(loss_médio, acc_média)``."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = device.type in {"cuda", "mps"}
    use_scaler = use_amp and device.type == "cuda"
    scaler = GradScaler(device=device.type) if use_scaler else None

    for images, labels, *_ in tqdm(loader, desc="treino densidade", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        targets, mask = _prepare_targets(labels)
        if targets.numel() == 0:
            continue

        images = images[mask]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Avalia o modelo em ``loader`` e devolve ``(loss, acc)`` médios."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = device.type in {"cuda", "mps"}

    for images, labels, *_ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        targets, mask = _prepare_targets(labels)
        if targets.numel() == 0:
            continue

        images = images[mask]

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Gera previsões detalhadas (probabilidades + classe) para cada DICOM."""

    model.eval()
    records: List[Dict[str, object]] = []

    use_amp = device.type in {"cuda", "mps"}

    for images, labels, accessions, paths, idxs in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        for i in range(images.size(0)):
            true_label = int(labels[i].item())
            if true_label <= 0:
                # Mantém compatibilidade: pula exemplos sem classificação válida.
                continue
            records.append(
                {
                    "idx": int(idxs[i]),
                    "AccessionNumber": accessions[i],
                    "dicom_path": paths[i],
                    "true_class": true_label,
                    "predicted_class": int(preds[i].item()) + 1,
                    "prob_1": float(probs[i, 0].item()),
                    "prob_2": float(probs[i, 1].item()),
                    "prob_3": float(probs[i, 2].item()),
                    "prob_4": float(probs[i, 3].item()),
                }
            )

    return records


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> Tuple[nn.Module, List[DensityHistoryEntry]]:
    """Executa o laço completo de treino, preservando o melhor estado."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    history: List[DensityHistoryEntry] = []
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = None
        val_acc = None

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            metric = val_acc
        else:
            metric = train_acc

        history.append(DensityHistoryEntry(epoch, train_loss, train_acc, val_loss, val_acc))

        print(
            f"Época {epoch:02d} | "
            f"loss treino: {train_loss:.4f} | acc treino: {train_acc:.4f}"
            + (" | loss val: {:.4f} | acc val: {:.4f}".format(val_loss, val_acc) if val_loader is not None else "")
        )

        if metric > best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


# ------------------------------- Utilidades -------------------------------

def dataset_summary(dataset: Dataset) -> Dict[int, int]:
    """Conta quantas amostras existem por classe 1–4 no dataset fornecido."""

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for sample in dataset.samples:  # type: ignore[attr-defined]
        label = sample.classification
        if label in counts:
            counts[label] += 1
    return counts


def split_dataset(
    dataset: MammoDicomDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Divide o dataset em treino/validação preservando índices originais."""

    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError("Dataset vazio após aplicar filtros. Nada a treinar.")

    if val_fraction <= 0.0:
        return dataset, None

    val_size = int(round(n_total * val_fraction))
    if val_size <= 0:
        val_size = 1
    if val_size >= n_total:
        val_size = max(1, n_total - 1)

    train_size = n_total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def make_dataloader(
    subset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Padroniza a criação de DataLoaders para treino/validação."""

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def indices_from_subset(subset: Optional[Dataset]) -> Iterable[int]:
    """Recupera índices originais mesmo quando o conjunto é um ``Subset``."""

    if subset is None:
        return []
    if isinstance(subset, Subset):
        return subset.indices
    return range(len(subset))


def history_to_dict(history: List[DensityHistoryEntry]) -> List[Dict[str, Optional[float]]]:
    """Transforma o histórico (dataclasses) em estrutura serializável JSON."""

    return [
        {
            "epoch": h.epoch,
            "train_loss": h.train_loss,
            "train_acc": h.train_acc,
            "val_loss": h.val_loss,
            "val_acc": h.val_acc,
        }
        for h in history
    ]


# ------------------------------- Interface CLI -------------------------------

def parse_density_classifier_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parser dedicado ao classificador de densidade (autônomo e reutilizável)."""

    parser = argparse.ArgumentParser(
        description="Treino de ResNet50 para classificação de densidade mamária (categorias 1–4)."
    )
    parser.add_argument("--data_dir", type=str, default="./archive", help="Diretório com subpastas por AccessionNumber.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./classificacao.csv",
        help="CSV com colunas AccessionNumber/Classification.",
    )
    parser.add_argument("--out_dir", type=str, default="./outputs_classifier", help="Onde salvar pesos e relatórios.")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho de batch para treino/validação.")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers do DataLoader.")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de treino.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizado do AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay do AdamW.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fração de validação (0 desativa).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo de execução.",
    )
    parser.add_argument("--weights_path", type=str, default=None, help="Checkpoint ResNet50 prévio para iniciar o treino.")
    parser.add_argument("--avoid_download", action="store_true", help="Não baixa pesos caso não encontre localmente.")
    parser.add_argument("--torch_home", type=str, default=None, help="Diretório TORCH_HOME alternativo.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Congela todas as camadas exceto a FC final.")
    parser.add_argument("--seed", type=int, default=42, help="Seed global para reprodutibilidade.")
    return parser.parse_args(argv)


def run_density_classifier_cli(argv: Optional[List[str]] = None) -> None:
    """Executa o pipeline completo usando os argumentos fornecidos."""

    args = parse_density_classifier_args(argv)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    print(f"[info] Usando dispositivo: {device}")

    data_dir = find_best_data_dir(args.data_dir)
    labels_map = load_labels_dict(args.csv_path)

    transform_model, _ = get_transforms()
    dataset = MammoDicomDataset(
        data_dir=data_dir,
        labels_by_accession=labels_map,
        exclude_class_5=True,
        include_unlabeled=False,
        transform=transform_model,
    )
    print(f"[info] Amostras disponíveis: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("Nenhuma amostra disponível para treino (verifique diretórios e CSV).")

    counts = dataset_summary(dataset)
    print("[info] Distribuição por classe:", counts)

    train_subset, val_subset = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = make_dataloader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = None
    if val_subset is not None:
        val_loader = make_dataloader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )

    model = build_resnet50_classifier(
        device,
        num_classes=4,
        weights_path=args.weights_path,
        avoid_download=args.avoid_download,
        torch_home=args.torch_home,
        freeze_backbone=args.freeze_backbone,
    )

    print("[info] Iniciando treino...")
    model, history = fit_classifier(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    history_path = os.path.join(args.out_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_to_dict(history), f, indent=2, ensure_ascii=False)
    print(f"[ok] Histórico salvo em {history_path}")

    model_path = os.path.join(args.out_dir, "resnet50_density_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[ok] Pesos salvos em {model_path}")

    full_loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    prediction_records = collect_predictions(model, full_loader, device)
    df_preds = pd.DataFrame(prediction_records)

    if not df_preds.empty:
        val_indices = set(indices_from_subset(val_subset))
        df_preds["split"] = np.where(df_preds["idx"].isin(val_indices), "val", "train")

        report = classification_report(
            df_preds["true_class"],
            df_preds["predicted_class"],
            labels=[1, 2, 3, 4],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(df_preds["true_class"], df_preds["predicted_class"], labels=[1, 2, 3, 4])

        metrics_path = os.path.join(args.out_dir, "classification_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2, ensure_ascii=False)
        print(f"[ok] Métricas salvas em {metrics_path}")

        preds_path = os.path.join(args.out_dir, "predictions.csv")
        df_preds.sort_values("idx").to_csv(preds_path, index=False)
        print(f"[ok] Previsões salvas em {preds_path}")
    else:
        print("[aviso] Nenhuma previsão válida gerada (todas as amostras sem rótulo?).")
