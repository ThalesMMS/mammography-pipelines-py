#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_mammo_resnet50.py
----------------------------------
Script único e didático para:
  1) Ler mamografias no formato DICOM organizadas por subpastas (cada subpasta representa um exame).
  2) Pré-processar cada imagem (windowing robusto, conversão para RGB 224×224) e extrair embeddings 2048-D com ResNet50 pré-treinada no ImageNet.
  3) Salvar embeddings/metadados prontos para reuso (NPY/CSV) e registrar um exemplo completo de vetor com identificadores.
  4) Visualizar o espaço gerado via PCA/t-SNE, com gráficos coloridos por classe e/ou cluster.
  5) Rodar k-means com seleção automática de K (silhouette), salvar métricas detalhadas e gráficos auxiliares (histórico vs K e distribuição dos clusters).
  6) **Registrar o processo inteiro**: pré-visualizações obrigatórias, contagem de classes e log de sessão para rastreabilidade.

Pressupostos do dataset:
- Diretório raiz: "./archive" (o script mantém fallback para a grafia "archieve").
- Estrutura: ./archive/<AccessionNumber>/* (podem existir subdiretórios). Pegamos a **primeira** imagem DICOM encontrada
  dentro de cada pasta <AccessionNumber> (varredura recursiva e ordenada).
- CSV de rótulos: "./classificacao.csv" no diretório atual, com colunas:
    AccessionNumber,Classification,ClassificationDate
  Onde Classification: 1=adiposa, 2=predominantemente adiposa, 3=predominantemente densa, 4=densa, 5=incidência não-padrão.
  Por padrão, **excluímos** classificação 5.

Como usar (exemplos):
---------------------
# Uso simples (Kaggle ou local):
python extract_mammo_resnet50.py \
  --data_dir ./archive \
  --csv_path ./classificacao.csv \
  --out_dir ./outputs \
  --save_csv \
  --tsne

# Se o diretório correto for ./archive (sem "e"):
python extract_mammo_resnet50.py --data_dir ./archive --csv_path ./classificacao.csv

# Evitar downloads (ex.: problema de SSL/proxy) usando pesos locais baixados via navegador/cURL:
python extract_mammo_resnet50.py \
  --data_dir ./archive \
  --csv_path ./classificacao.csv \
  --out_dir ./outputs \
  --weights_path ./resnet50-11ad3fa6.pth \
  --avoid_download \
  --torch_home ./.torch_cache

Dependências principais:
- pydicom, numpy, pandas, pillow, torch, torchvision, scikit-learn, matplotlib, tqdm
Em DICOMs comprimidos (p.ex. JPEG2000), instale também:
- pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg
  pip install -q pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg

Notas práticas (radiologia + computação):
- Mamografias são 1-canal (grayscale) e de alta resolução. Aqui fazemos:
  (a) leitura do pixel_array com pydicom e aplicação de RescaleSlope/Intercept se existirem;
  (b) inversão automática para MONOCHROME1;
  (c) *windowing* robusto por percentis (0.5–99.5) para reduzir saturações e padronizar contraste;
  (d) conversão p/ uint8 (0–255), PIL Image em 'L' e replicação para 3 canais (RGB) para interface com ResNet50;
  (e) *resize* preservando aspecto (Resize->CenterCrop) para 224×224 e normalização padrão ImageNet.
- Embeddings gerados (2048-D) servem como *features* gerais. Não são "ótimos" para detalhes microcalcificações/lesões, mas
  oferecem um espaço útil para exploração/agrupamento de padrão global de densidade/posicionamento.
- Resultados devem ser interpretados com cuidado: visualizar outliers, revisar casos com label "5" (excluídos por padrão),
  e considerar harmonização (ex.: inversão inconsistências, equipamentos diferentes).

Saídas:
- <out_dir>/features.npy / features.csv        # Embeddings (float32) e versão tabular (opcional)
- <out_dir>/metadata.csv                       # AccessionNumber, Classification, dicom_path, idx
- <out_dir>/example_embedding.json             # Amostra de embedding 2048-D com metadados
- <out_dir>/joined.csv                         # metadata + projeções 2D + cluster_k
- <out_dir>/pca_2d*.{csv,png} ; tsne_2d*.{csv,png}   # Dispersões por classe/cluster
- <out_dir>/clustering_metrics.json            # k ótimo (silhouette, Davies-Bouldin) + histórico por K
- <out_dir>/kmeans_metrics.png                 # Curvas silhouette (↑) e Davies-Bouldin (↓) vs K
- <out_dir>/cluster_distribution.png           # Barras com o tamanho de cada cluster
- <out_dir>/preview/first_image_loaded.png     # **Exemplo obrigatório de imagem pré-processada**
- <out_dir>/preview/preprocess_steps.png       # Pipeline: RAW(min-max) vs WINDOWED vs RESIZED
- <out_dir>/preview/samples_grid.png           # Grade com N exemplos do dataset (após pré-processamento)
- <out_dir>/preview/labels_distribution.png    # Gráfico da distribuição de classes do CSV
- <out_dir>/run/session_info.json              # Informações da sessão (device, versões, tempos, argumentos)

Autor: gerado sob demanda, com foco didático.
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# --- Dependências usuais ---
import numpy as np
import pandas as pd
from tqdm import tqdm

# Matplotlib apenas (evitar seaborn por simplicidade/portabilidade)
import matplotlib
matplotlib.use("Agg")  # backend não interativo para salvar PNGs em Kaggle/servidor
import matplotlib.pyplot as plt

# Torch / Torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception as e:
    _HAS_TORCHVISION = False
    raise RuntimeError("torchvision não está disponível. Instale torch/torchvision antes de executar.") from e

# DICOM
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
except Exception as e:
    raise RuntimeError(
        "pydicom não está disponível. Instale com `pip install pydicom`.\n"
        "Para DICOMs comprimidos (JPEG2000), considere também:\n"
        "`pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`"
    ) from e

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from PIL import Image
# ------------------- Configurações e utilidades -------------------

CLASS_MAP = {
    1: "adiposa",
    2: "predominantemente adiposa",
    3: "predominantemente densa",
    4: "densa",
    5: "incidência não-padrão (excluir)"
}
# Conversão numérica -> descrição em português para tornar os logs e gráficos
# mais amigáveis. Mantemos o código clínico original junto às strings.

# Define seeds globais para reduzir variação entre execuções.
def seed_everything(seed: int = 42):
    """Define seeds para reprodutibilidade razoável (atenção: DataLoader workers podem introduzir variação)."""
    # O objetivo aqui é minimizar variações aleatórias entre execuções (inicialização,
    # ordem de batches etc.), facilitando comparar resultados e depurar o pipeline.
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Resolve o diretório raiz do dataset, corrigindo grafias comuns.
def find_best_data_dir(pref: str) -> str:
    """Tenta localizar a pasta de dados. Por padrão usa o que o usuário passou.
    Se não existir, tenta trocar 'archieve' por 'archive'.
    """
    # Em alguns datasets a pasta vem com nome digitado incorretamente; esta rotina
    # fornece um fallback amigável antes de falhar explicitamente, o que ajuda quem
    # está explorando os dados pela primeira vez.
    if os.path.isdir(pref):
        return pref
    alt = pref.replace("archieve", "archive")
    if os.path.isdir(alt):
        print(f"[info] data_dir '{pref}' não encontrado; usando '{alt}'")
        return alt
    # último recurso: raiz do Kaggle input
    if os.path.isdir("/kaggle/input"):
        print("[aviso] data_dir não encontrado; verifique o caminho. Usando /kaggle/input para inspeção.")
        return "/kaggle/input"
    return pref  # deixamos falhar adiante com mensagem clara


# Converte um tamanho em bytes para formato legível (B, KB, MB...).
def human_readable_size(num_bytes: int) -> str:
    """Formata bytes em unidades legíveis."""
    # Usamos este helper para que os arquivos gerados (npy/csv/png) fiquem autoexplicativos
    # no log. Assim, o leitor consegue estimar quanto espaço em disco será consumido.
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"


# ------------------- Leitura e pré-processamento DICOM -------------------

# Detecta se o DICOM usa fotometria MONOCHROME1 para inverter tons.
def _is_mono1(ds: "pydicom.dataset.FileDataset") -> bool:
    """Verifica se a imagem é MONOCHROME1 (preto-branco invertidos)."""
    # Em mamografias MONOCHROME1, o valor 0 representa branco (fundo) e valores altos são
    # áreas escurecidas. A maioria dos algoritmos e telas assumem o inverso, então
    # precisamos detectar este caso para corrigir a tonalidade.
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    return photometric == "MONOCHROME1"


# Converte arrays para float32, evitando casts repetidos.
def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Garante float32 para cálculos numéricos estáveis e previsíveis."""
    # Trabalhar com float32 evita estouro/underflow ao aplicar LUTs ou normalizações,
    # além de se alinhar ao tipo esperado pelo PyTorch.
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


# Aplica RescaleSlope e RescaleIntercept aos pixels do DICOM.
def _apply_rescale(ds: "pydicom.dataset.FileDataset", arr: np.ndarray) -> np.ndarray:
    """Aplica RescaleSlope/RescaleIntercept (quando presentes) de forma segura."""
    # Esses parâmetros, comuns em imagens médicas, convertem o pixel bruto para unidades
    # físicas (p.ex. densidade). Aplicá-los garante que o contraste ficará correto.
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    try:
        arr = arr * float(slope) + float(intercept)
    except Exception:
        # Fallback usando utilitário pydicom (inclui LUT, se aplicável)
        try:
            arr = apply_modality_lut(arr, ds)
        except Exception:
            pass
    return arr


# Executa windowing por percentis para padronizar contraste da mamografia.
def robust_window(arr: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Windowing por percentis para padronizar contraste de mamografias (robusto a outliers).
    - Clipa no intervalo [p_low, p_high] dos valores da imagem;
    - Normaliza para [0, 1].
    """
    # Em mamografias é comum haver regiões muito claras ou muito escuras. Em vez de usar
    # mínimos/máximos globais (que seriam sensíveis a ruído), recortamos pelos percentis
    # para preservar o tecido mamário em um intervalo visualmente consistente.
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        # Em casos degenerados (imagem quase constante), evita divisão por zero
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr


# Converte um DICOM para imagem PIL RGB após pré-processamento completo.
def dicom_to_pil_rgb(dcm_path: str) -> Image.Image:
    """Lê um DICOM de mamografia, aplica pré-processamento e retorna PIL Image RGB 8-bit.
    Passos:
      - pixel_array -> float32
      - RescaleSlope/Intercept (quando houver)
      - Inversão para MONOCHROME1
      - Windowing por percentis (0.5–99.5)
      - Conversão p/ uint8 (0–255) e PIL em 'L'
      - Replicação para 3 canais (RGB) para compatibilidade com ResNet50
    """
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array  # pode exigir plugins (pylibjpeg) para DICOMs comprimidos
    except Exception as e:
        # Caso o arquivo esteja compactado em formatos específicos (JPEG2000, por exemplo),
        # o pydicom precisa de plugins extras. Se não estiverem instalados, emitimos uma
        # mensagem direta indicando como prosseguir.
        raise RuntimeError(
            f"Falha ao ler pixel data de {dcm_path}. Se for DICOM comprimido, instale plugins:\n"
            "  pip install -q pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
            f"Erro original: {repr(e)}"
        )

    # Convertendo para float32 cedo, garantimos que operações posteriores (rescale, windowing)
    # manipulem valores com precisão adequada.
    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)

    # MONOCHROME1: 0 = branco, 4095 = preto => inverte para ficar preto=0
    if _is_mono1(ds):
        arr = arr.max() - arr

    # Aplicamos nosso windowing robusto para reduzir extremos e depois escalamos
    # para 0-255 (uint8), formato que bibliotecas de visão utilizam facilmente.
    arr = robust_window(arr, 0.5, 99.5)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # Criamos primeiro uma imagem em escala de cinza ('L') e replicamos o canal
    # para obter (R,G,B). A ResNet50 original espera 3 canais porque foi treinada
    # com fotografias RGB do ImageNet.
    pil = Image.fromarray(arr, mode="L")
    pil_rgb = Image.merge("RGB", (pil, pil, pil))
    return pil_rgb


# Retorna intermediários visuais do pré-processamento para depuração.
def dicom_debug_preprocess(dcm_path: str) -> Dict[str, object]:
    """Versão detalhada para visualização do pipeline de pré-processamento.
    Retorna dicionário com arrays e PILs em estágios: raw_minmax, windowed, resized224.
    """
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array
    arr = _to_float32(arr)
    arr = _apply_rescale(ds, arr)
    if _is_mono1(ds):
        arr = arr.max() - arr
    arr_raw = arr.copy()

    # RAW min-max -> [0,1]
    # Esta etapa mostra a imagem apenas normalizada pelo min/max tradicionais.
    # Ela é útil para perceber como o contraste original pode estar comprimido.
    lo_raw, hi_raw = float(arr_raw.min()), float(arr_raw.max())
    eps = 1e-6 if hi_raw - lo_raw == 0 else 0.0
    arr_raw_mm = (arr_raw - lo_raw) / (hi_raw - lo_raw + eps)
    raw_uint8 = (arr_raw_mm * 255.0).clip(0, 255).astype(np.uint8)

    # WINDOWED [0,1]
    # Aqui aplicamos o windowing por percentis, que tende a evidenciar o tecido.
    arr_win = robust_window(arr, 0.5, 99.5)
    win_uint8 = (arr_win * 255.0).clip(0, 255).astype(np.uint8)

    pil_raw = Image.fromarray(raw_uint8, mode="L")
    pil_win = Image.fromarray(win_uint8, mode="L")
    pil_raw_rgb = Image.merge("RGB", (pil_raw, pil_raw, pil_raw))
    pil_win_rgb = Image.merge("RGB", (pil_win, pil_win, pil_win))

    # Resize->CenterCrop 224 (para visualização)
    # A rede espera entradas 224x224. Usamos Resize+CenterCrop, padrão ImageNet,
    # para adaptar as dimensões mantendo o centro da mama na janela.
    vis_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    pil_resized = vis_tf(pil_win_rgb)

    # Headers não sensíveis (evitar PII)
    safe_header = {}
    safe_keys = [
        "Manufacturer", "ManufacturerModelName", "PhotometricInterpretation",
        "Rows", "Columns", "BitsStored", "BitsAllocated", "HighBit",
        "PixelRepresentation", "RescaleIntercept", "RescaleSlope",
        "ViewPosition", "Laterality", "BodyPartExamined", "SeriesDescription",
        "SOPClassUID", "Modality"
    ]
    for k in safe_keys:
        if hasattr(ds, k):
            v = getattr(ds, k)
            try:
                safe_header[k] = str(v)
            except Exception:
                pass

    # Retornamos dicionário com as diferentes representações para que o script
    # possa montar figuras comparativas e, ao mesmo tempo, mostrar cabeçalhos sem PII.
    return {
        "raw_uint8": raw_uint8,
        "win_uint8": win_uint8,
        "pil_raw_rgb": pil_raw_rgb,
        "pil_win_rgb": pil_win_rgb,
        "pil_resized_rgb": pil_resized,
        "safe_header": safe_header,
        "shape_raw": [int(arr_raw.shape[0]), int(arr_raw.shape[1])],
    }


# ------------------- Dataset e DataLoader -------------------

@dataclass
class SampleInfo:
    """Container leve com os metadados necessários para reconstruir cada amostra."""
    accession: str
    classification: Optional[int]   # 1..5, ou None se não houver no CSV
    path: str                       # caminho do DICOM selecionado
    idx: int                        # índice sequencial


class MammoDicomDataset(Dataset):
    """Dataset que:
      - Varre subpastas de data_dir (cada uma com nome 'AccessionNumber');
      - Seleciona o primeiro DICOM (ordem lexicográfica) dentro de cada subpasta (varredura recursiva);
      - Exclui opcionalmente Classification == 5;
      - Retorna imagem pré-processada p/ ResNet50 + metadados.
    """
    def __init__(
        self,
        data_dir: str,
        labels_by_accession: Dict[str, int],
        exclude_class_5: bool = True,
        include_unlabeled: bool = False,
        transform: Optional[torch.nn.Module] = None,
        exts: Tuple[str, ...] = (".dcm", ".dicom", ".DCM", ".DICOM"),
    ):
        """Indexa as subpastas do dataset aplicando filtros de rótulo e extensões aceitas."""
        self.data_dir = data_dir
        self.labels_by_accession = labels_by_accession
        self.exclude_class_5 = exclude_class_5
        self.include_unlabeled = include_unlabeled
        self.transform = transform
        self.exts = exts

        self.samples: List[SampleInfo] = []
        # Ao inicializar já construímos a lista de amostras válidas, evitando
        # custo de varrer diretórios a cada acesso.
        self._build_index()

    def _list_dirs(self, root: str) -> List[str]:
        """Retorna subpastas ordenadas alfabeticamente, ignorando arquivos simples."""
        # Ordenar as subpastas garante que o processamento seja determinístico,
        # algo útil para auditorias e para reproduzir resultados.
        return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

    def _find_first_dicom(self, folder: str) -> Optional[str]:
        """Procura recursivamente o primeiro arquivo DICOM (ordem lexicográfica) dentro de 'folder'."""
        # Muitos exames guardam várias incidências na mesma pasta. Escolher o primeiro
        # arquivo padroniza o comportamento e evita amostras duplicadas logo de início.
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
        """Percorre data_dir e preenche self.samples com as entradas válidas."""
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir '{self.data_dir}' não existe. Verifique o caminho.")

        idx = 0
        for sub in self._list_dirs(self.data_dir):
            accession = str(sub).strip()
            label = self.labels_by_accession.get(accession)

            # Regra de exclusão: Classification == 5
            if label == 5 and self.exclude_class_5:
                continue
            if (label is None) and (not self.include_unlabeled):
                # pula se não tiver rótulo e não quisermos incluí-los
                continue

            folder = os.path.join(self.data_dir, sub)
            dcm_path = self._find_first_dicom(folder)
            if dcm_path is None:
                continue  # sem DICOM legível dentro da pasta

            # Guardamos metadados mínimos (acesso, rótulo, caminho, índice) para
            # recuperar posteriormente após a inferência do modelo.
            self.samples.append(SampleInfo(accession=accession, classification=label, path=dcm_path, idx=idx))
            idx += 1

        if len(self.samples) == 0:
            warnings.warn("Nenhuma amostra encontrada. Verifique diretórios e CSV.")

    def __len__(self) -> int:
        """Retorna o número de amostras disponíveis depois dos filtros aplicados."""
        # Permite que len(dataset) funcione como esperado, revelando quantos exames
        # serão processados depois dos filtros aplicados.
        return len(self.samples)

    def __getitem__(self, i: int):
        """Obtém a i-ésima amostra já pré-processada e seus metadados associados."""
        info = self.samples[i]
        # Convertendo o DICOM para PIL já com pré-processamento;
        # na sequência aplicamos as transformações de data augmentation/normalização do modelo.
        img = dicom_to_pil_rgb(info.path)
        if self.transform is not None:
            img = self.transform(img)
        # Retornamos -1 para itens sem rótulo, o que permite tratá-los como "desconhecidos"
        # durante métricas e visualizações.
        label = info.classification if (info.classification is not None) else -1
        return img, label, info.accession, info.path, info.idx


# ------------------- Modelo (ResNet50) -------------------

## Remove prefixos de DataParallel (ex.: 'module.') de state_dicts.
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
    # Alguns checkpoints treinados com DataParallel salvam os pesos com prefixo "module.".
    # Remover esses prefixos evita erros ao carregar os pesos em um modelo simples.
    return new_sd


## Aplica um state_dict ao modelo, reportando chaves ausentes ou inesperadas.
def _apply_state_dict(model: nn.Module, state: dict) -> nn.Module:
    """Carrega um state_dict no modelo e imprime diferenças relevantes para o usuário."""
    state_dict = state.get("state_dict", state)
    state_dict = _strip_module_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[aviso] Chaves ausentes no state_dict: {missing[:5]} ...")
    if unexpected:
        print(f"[aviso] Chaves inesperadas no state_dict: {unexpected[:5]} ...")
    return model


## Tenta carregar pesos explícitos fornecidos pelo usuário via CLI.
def _load_weights_from_path(resnet_ctor, weights_path: Optional[str]) -> Optional[nn.Module]:
    """Retorna um modelo ResNet50 carregado a partir de um caminho explícito (se existir)."""
    if not weights_path or not os.path.isfile(weights_path):
        return None
    print(f"[info] Carregando pesos locais: {weights_path}")
    model = resnet_ctor(weights=None)
    state = torch.load(weights_path, map_location="cpu")
    return _apply_state_dict(model, state)


## Lista checkpoints resnet50 disponíveis nos caches conhecidos do torchvision.
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


## Carrega pesos da cache local, caso existam e estejam legíveis.
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


## Baixa pesos oficiais do torchvision (com fallback) quando nada local está disponível.
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


# Monta a ResNet50 pré-treinada como extratora de embeddings 2048-D.
def build_resnet50_feature_extractor(
    device: torch.device,
    weights_path: Optional[str] = None,
    avoid_download: bool = False,
    torch_home: Optional[str] = None
) -> nn.Module:
    """Monta a ResNet50 pré-treinada e a adapta como extratora de embeddings 2048-D.

    A busca por pesos segue esta ordem:
    1. Caminho explícito informado via --weights_path;
    2. Checkpoints presentes nos caches locais do torchvision;
    3. Download oficial (respeitando --avoid_download).
    """
    from torchvision.models import resnet50

    if torch_home:
        os.environ["TORCH_HOME"] = os.path.abspath(torch_home)
        # Definir TORCH_HOME manualmente é útil quando não temos permissão de escrita
        # em ~/.cache (caso comum em servidores/Kaggle). Assim o download fica em pasta conhecida.

    model = _load_weights_from_path(resnet50, weights_path)
    if model is None:
        model = _load_weights_from_cache(resnet50, torch_home)
    if model is None:
        model = _download_resnet_weights(resnet50, avoid_download)

    # A ResNet50 termina com uma camada totalmente conectada (FC) de 1000 neurônios
    # usada para classificar o ImageNet. Como queremos apenas o vetor de características,
    # trocamos essa FC por uma identidade: a saída passa a ser o vetor de 2048 features
    # extraído da penúltima camada (global average pooling).
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


# Define transforms para inferência (modelo) e visualização (PNG).
def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Retorna:
      - transform_model: Resize->CenterCrop->ToTensor->Normalize (para o modelo),
      - transform_vis:   Resize->CenterCrop (PIL) para visualização.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    # Essas médias e desvios são os mesmos usados no treinamento do ImageNet.
    # Normalizar com eles garante que a ResNet50 receba imagens em uma escala
    # semelhante ao que viu durante o treinamento original.
    transform_model = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_vis = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    return transform_model, transform_vis


@torch.inference_mode()
# Percorre o DataLoader e devolve embeddings numpy + metadados.
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[np.ndarray, List[SampleInfo]]:
    """Extrai embeddings [N, 2048] do DataLoader, preservando metadados na ordem."""
    feats = []
    infos: List[SampleInfo] = []

    # autocast em CUDA apenas (não usar em MPS/CPU)
    # Quando disponível, o autocast usa precisão mista (float16/32) acelerando a inferência
    # em GPUs NVIDIA. Mantemos desabilitado em CPU/MPS para evitar degradação numérica.
    amp_enabled = use_amp and (device.type == "cuda")

    for batch in tqdm(loader, desc="Extraindo embeddings", total=len(loader)):
        imgs, labels, accessions, paths, idxs = batch
        imgs = imgs.to(device, non_blocking=True)

        if amp_enabled:
            with torch.cuda.amp.autocast():
                out = model(imgs)  # [B, 2048]
        else:
            out = model(imgs)

        out = out.detach().cpu().numpy()
        feats.append(out)

        # reconstrói SampleInfo por item do batch
        # Aqui reconstruímos a lista de SampleInfo na mesma ordem em que o DataLoader
        # forneceu as imagens, preservando os metadados para unir depois com os vetores.
        for lab, acc, pth, idx in zip(labels, accessions, paths, idxs):
            infos.append(SampleInfo(accession=acc, classification=int(lab) if int(lab) >= 0 else None, path=pth, idx=int(idx)))

    feats = np.concatenate(feats, axis=0) if len(feats) > 0 else np.empty((0, 2048), dtype=np.float32)
    feats = feats.astype(np.float32, copy=False)
    return feats, infos


# ------------------- Visualização e utilidades de saída -------------------

# Garante que o diretório informado exista antes de salvar arquivos.
def ensure_dir(path: str):
    """Cria o diretório informado caso ele ainda não exista."""
    # Garantimos a existência dos diretórios de saída antes de salvar arquivos.
    # Isso evita exceções difíceis de interpretar para quem está começando.
    os.makedirs(path, exist_ok=True)


# Salva uma figura com as etapas principais do pré-processamento.
def save_preprocess_figure(example_path: str, out_png: str):
    """Salva figura com 3 painéis: RAW(min-max), WINDOWED, WINDOWED+224 (PIL)."""
    try:
        dbg = dicom_debug_preprocess(example_path)
    except Exception as e:
        print(f"[aviso] Falhou debug de pré-processamento: {e}")
        return

    imgs = [dbg["pil_raw_rgb"], dbg["pil_win_rgb"], dbg["pil_resized_rgb"]]
    titles = ["RAW (min-max)", "WINDOWED (0.5–99.5)", "WINDOWED + Resize/Crop 224"]
    plt.figure(figsize=(12, 4))
    for i, (im, tt) in enumerate(zip(imgs, titles), start=1):
        ax = plt.subplot(1, 3, i)
        ax.imshow(im)
        ax.set_title(tt, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# Grava a primeira imagem pré-processada para inspeção rápida.
def save_first_image_preview(dataset: MammoDicomDataset, out_png: str, transform_vis):
    """Salva a primeira imagem (pré-processada + 224) como preview obrigatório."""
    if len(dataset) == 0:
        return
    _, _, _, path, _ = dataset[0]
    try:
        dbg = dicom_debug_preprocess(path)
        im = transform_vis(dbg["pil_win_rgb"])
    except Exception:
        # fallback simples
        # Caso o debug detalhado falhe (p.ex. DICOM problemático), carregamos a imagem
        # diretamente pelo pipeline padrão para não perder o preview obrigatório.
        im = transform_vis(dicom_to_pil_rgb(path))
    im.save(out_png)


# Gera uma grade de amostras representativas do dataset tratado.
def save_samples_grid(dataset: MammoDicomDataset, out_png: str, transform_vis, max_n: int = 12):
    """Salva uma grade NxM com as primeiras N imagens (após pré-processamento + 224)."""
    from math import ceil, sqrt
    n = min(max_n, len(dataset))
    if n == 0:
        return
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Prepara canvas
    # Criamos uma grade simples preenchida com cinza claro, onde cada "célula"
    # receberá uma mamografia já normalizada/recortada. Ajuda a visualizar coesão
    # do dataset logo após o pré-processamento.
    w, h = 224, 224
    grid = np.ones((rows*h, cols*w, 3), dtype=np.uint8) * 240

    for i in range(n):
        _, _, _, path, _ = dataset[i]
        try:
            dbg = dicom_debug_preprocess(path)
            im = transform_vis(dbg["pil_win_rgb"])
        except Exception:
            im = transform_vis(dicom_to_pil_rgb(path))
        im = np.array(im)  # (224,224,3)
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w, :] = im

    plt.figure(figsize=(cols*2.4, rows*2.4))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# Plota a distribuição das classes consideradas após filtros.
def plot_labels_distribution(labels: List[int], out_png: str):
    """Gráfico de barras da distribuição de classes 1..4 (+ -1)."""
    from collections import Counter
    cnt = Counter(labels)
    order = [-1, 1, 2, 3, 4]
    names = ["sem rótulo"] + [CLASS_MAP[i] for i in [1,2,3,4]]
    vals = [cnt.get(k, 0) for k in order]

    # Esta visualização ajuda a entender desbalanceamentos de classe, algo crítico
    # em problemas de densidade mamária. Classes com poucos exemplos merecem atenção
    # nas análises e em eventuais treinamentos futuros.
    plt.figure(figsize=(6,4))
    plt.bar(range(len(order)), vals)
    plt.xticks(range(len(order)), names, rotation=20, ha="right")
    plt.title("Distribuição de classes nas amostras consideradas")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# Plota a evolução de silhouette e Davies-Bouldin para cada K testado.
def plot_kmeans_history(history: List[Dict[str, Optional[float]]], out_png: str):
    """Plota as métricas (silhouette e Davies-Bouldin) avaliadas para cada K testado."""
    if not history:
        return
    import math
    ks: List[int] = []
    silhouettes: List[float] = []
    davies: List[float] = []
    for entry in history:
        k_val = entry.get("k")
        if k_val is None:
            continue
        ks.append(int(k_val))
        sil = entry.get("silhouette")
        db = entry.get("davies_bouldin")
        silhouettes.append(float(sil) if sil is not None else math.nan)
        davies.append(float(db) if db is not None else math.nan)

    if not ks:
        return

    plt.figure(figsize=(6, 6))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ks, silhouettes, marker="o", color="#1f77b4")
    ax1.set_ylabel("Silhouette (↑)")
    ax1.set_title("k-means: métricas por número de clusters (K)")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(ks, davies, marker="s", color="#ff7f0e")
    ax2.set_ylabel("Davies-Bouldin (↓)")
    ax2.set_xlabel("Número de clusters (K)")
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# Mostra o tamanho de cada cluster identificado pelo k-means.
def plot_cluster_counts(cluster_labels: List[int], out_png: str):
    """Gera um gráfico de barras com o tamanho de cada cluster encontrado."""
    if not cluster_labels:
        return
    from collections import Counter

    cnt = Counter(cluster_labels)
    clusters = sorted(cnt.keys())
    counts = [cnt[c] for c in clusters]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(clusters)), counts, color="#2ca02c")
    plt.xticks(range(len(clusters)), [f"cluster {c}" for c in clusters])
    plt.ylabel("Número de exames")
    plt.title("Distribuição de tamanhos dos clusters (k-means)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# Plota projeções 2D coloridas por classe e diferenciadas por cluster.
def plot_scatter(
    xy_coords: np.ndarray,
    labels: List[Optional[int]],
    clusters: Optional[List[int]],
    title: str,
    out_path: str,
    alpha: float = 0.85,
):
    """Gera um scatter plot 2D e salva em PNG.
    - `labels` indicam as classes (1..4) usadas para colorir (quando disponíveis);
    - `clusters` (se fornecidos) mudam o marcador para cada cluster.
    """
    plt.figure(figsize=(8, 6))
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]

    # Cores por classe (1..4); None/-1 vai para cinza
    # Usamos o colormap tab10 para oferecer cores distintas e fáceis de distinguir.
    cmap = plt.cm.get_cmap("tab10", 6)
    color_map = {1: cmap(0), 2: cmap(1), 3: cmap(2), 4: cmap(3), None: (0.6, 0.6, 0.6, 0.6), -1: (0.6, 0.6, 0.6, 0.6)}
    colors = [color_map.get(lab, (0.6, 0.6, 0.6, 0.6)) for lab in labels]

    # Marcadores por cluster (até 10 clusters)
    # Diferentes marcadores nos permitem sobrepor informações: cor = classe clínica,
    # marcador = agrupamento automático (k-means). Isso facilita inspeções visuais.
    markers_list = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
    markers = None
    if clusters is not None:
        unique_clusters = sorted(set(clusters))
        marker_map = {c: markers_list[i % len(markers_list)] for i, c in enumerate(unique_clusters)}
        markers = [marker_map[c] for c in clusters]

    if markers is None:
        plt.scatter(xs, ys, c=colors, s=20, alpha=alpha, edgecolor="none")
    else:
        for marker in sorted(set(markers)):
            idxs = [i for i, recorded in enumerate(markers) if recorded == marker]
            plt.scatter(xs[idxs], ys[idxs], c=[colors[i] for i in idxs], s=20, alpha=alpha, marker=marker, edgecolor="none", label=f"cluster '{marker}'")
        plt.legend(loc="best", fontsize=8)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# Executa k-means para um K específico e calcula métricas associadas.
def _score_kmeans(
    feats: np.ndarray,
    n_clusters: int,
    random_state: int
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Executa k-means para um valor de K retornando rótulos, silhouette e Davies-Bouldin."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(feats)
    if len(set(labels)) < 2:
        return labels, None, None
    silhouette = silhouette_score(feats, labels)
    try:
        davies = davies_bouldin_score(feats, labels)
    except Exception:
        davies = None
    return labels, float(silhouette), float(davies) if davies is not None else None


# Seleciona automaticamente o melhor K (2..8) usando silhouette e salva métricas.
def auto_kmeans(
    feats: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Optional[float]], List[Dict[str, Optional[float]]]]:
    """Procura K ótimo via silhouette em [k_min, k_max] registrando o histórico completo das métricas."""
    best_labels: Optional[np.ndarray] = None
    best_silhouette = -float("inf")
    best_k: Optional[int] = None
    best_db: Optional[float] = None
    history: List[Dict[str, Optional[float]]] = []

    for k in range(k_min, k_max + 1):
        labels, silhouette, davies = _score_kmeans(feats, k, random_state)
        history.append({"k": int(k), "silhouette": silhouette, "davies_bouldin": davies})
        if silhouette is None:
            continue
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_labels = labels
            best_k = k
            best_db = davies

    if best_labels is None:
        best_k = 2
        best_labels, silhouette, davies = _score_kmeans(feats, best_k, random_state)
        best_silhouette = silhouette if silhouette is not None else float("nan")
        best_db = davies

    if best_labels is None:
        best_labels = np.zeros((feats.shape[0],), dtype=int)

    metrics = {
        "best_k": int(best_k) if best_k is not None else None,
        "silhouette": None if (best_silhouette is None or not math.isfinite(best_silhouette)) else float(best_silhouette),
        "davies_bouldin": best_db if (best_db is None or isinstance(best_db, float)) else float(best_db),
    }
    return best_labels, metrics, history


# ------------------- Funções auxiliares para o pipeline -------------------

# Processa e valida os argumentos de linha de comando do script.
def parse_arguments() -> argparse.Namespace:
    """Constrói o parser CLI e devolve o namespace com todas as opções do usuário."""
    parser = argparse.ArgumentParser(description="Extração de embeddings com ResNet50 a partir de mamografias DICOM.")
    parser.add_argument("--data_dir", type=str, default="./archive", help="Pasta raiz com subpastas por AccessionNumber (default: ./archive).")
    parser.add_argument("--csv_path", type=str, default="./classificacao.csv", help="Caminho do arquivo CSV de classificações.")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Diretório de saída (será criado).")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamanho de batch para a inferência.")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers do DataLoader (0=sem multiprocess).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Dispositivo de execução ('mps' para Macs Apple Silicon).")
    parser.add_argument("--save_csv", action="store_true", help="Se setado, salva também os embeddings em CSV (além de NPY).")
    parser.add_argument("--include_unlabeled", action="store_true", help="Inclui subpastas sem rótulo no CSV (label = -1).")
    parser.add_argument("--no_exclude_5", action="store_true", help="Não exclui classificação 5 (por padrão exclui).")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--tsne", action="store_true", help="Se setado, calcula t-SNE (pode ser mais demorado que PCA).")
    parser.add_argument("--weights_path", type=str, default=None, help="Caminho local para um .pth da ResNet50 (opcional).")
    parser.add_argument("--avoid_download", action="store_true", help="Evita qualquer tentativa de download de pesos.")
    parser.add_argument("--torch_home", type=str, default=None, help="Define TORCH_HOME (cache do torchvision).")
    parser.add_argument("--preview_max", type=int, default=12, help="Quantidade de amostras na grade de preview.")
    return parser.parse_args()


# Determina o dispositivo de execução mais adequado considerando as opções passadas.
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


# Cria a estrutura de diretórios que receberá os artefatos gerados.
def prepare_output_dirs(out_dir: str) -> Tuple[str, str]:
    """Garante a existência de out_dir, preview/ e run/, retornando os dois últimos caminhos."""
    ensure_dir(out_dir)
    preview_dir = os.path.join(out_dir, "preview")
    run_dir = os.path.join(out_dir, "run")
    ensure_dir(preview_dir)
    ensure_dir(run_dir)
    return preview_dir, run_dir


# Carrega o CSV clínico e monta o dicionário AccessionNumber -> classe.
def load_labels_dict(csv_path: str) -> Dict[str, int]:
    """Lê o CSV de classificações garantindo que AccessionNumber permaneça como string."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV '{csv_path}' não encontrado. Ajuste --csv_path.")
    df_csv = pd.read_csv(
        csv_path,
        dtype={"AccessionNumber": str, "Classification": int},
        parse_dates=["ClassificationDate"],
        dayfirst=False
    )
    df_csv["AccessionNumber"] = df_csv["AccessionNumber"].str.strip()
    return {row["AccessionNumber"]: int(row["Classification"]) for _, row in df_csv.iterrows()}


# Instancia o dataset de mamografias e seu DataLoader com os filtros desejados.
def build_dataset_and_loader(
    data_dir: str,
    labels_map: Dict[str, int],
    args: argparse.Namespace,
    transform_model: transforms.Compose
) -> Tuple[MammoDicomDataset, DataLoader]:
    """Cria e retorna o dataset de mamografias e seu DataLoader, respeitando as flags de filtro."""
    dataset = MammoDicomDataset(
        data_dir=data_dir,
        labels_by_accession=labels_map,
        exclude_class_5=(not args.no_exclude_5),
        include_unlabeled=args.include_unlabeled,
        transform=transform_model,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
    )
    return dataset, loader


# Gera as visualizações obrigatórias do pré-processamento antes da inferência.
def generate_mandatory_previews(
    dataset: MammoDicomDataset,
    preview_dir: str,
    transform_vis: transforms.Compose,
    preview_max: int
) -> None:
    """Produz first_image_loaded, preprocess_steps, samples_grid e labels_distribution."""
    if len(dataset) == 0:
        return
    save_first_image_preview(dataset, os.path.join(preview_dir, "first_image_loaded.png"), transform_vis)
    first_path = dataset.samples[0].path
    save_preprocess_figure(first_path, os.path.join(preview_dir, "preprocess_steps.png"))
    save_samples_grid(dataset, os.path.join(preview_dir, "samples_grid.png"), transform_vis, max_n=preview_max)
    labels_list = [(s.classification if s.classification is not None else -1) for s in dataset.samples]
    plot_labels_distribution(labels_list, os.path.join(preview_dir, "labels_distribution.png"))


# Orquestra o carregamento da ResNet50 e a extração dos embeddings.
def extract_feature_embeddings(
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace
) -> Tuple[np.ndarray, List[SampleInfo]]:
    """Carrega a ResNet50 com os pesos corretos e devolve (embeddings, infos)."""
    model = build_resnet50_feature_extractor(
        device,
        weights_path=args.weights_path,
        avoid_download=args.avoid_download,
        torch_home=args.torch_home
    )
    return extract_embeddings(model, loader, device=device, use_amp=True)


# Persiste embeddings/metadados em disco e devolve um DataFrame ordenado.
def save_embeddings_artifacts(
    feats: np.ndarray,
    infos: List[SampleInfo],
    out_dir: str,
    save_csv: bool
) -> pd.DataFrame:
    """Salva features em NPY/CSV (opcional) e constrói metadata.csv ordenado por idx."""
    emb_path = os.path.join(out_dir, "features.npy")
    np.save(emb_path, feats)
    print(f"[ok] Salvou embeddings em: {emb_path} ({human_readable_size(os.path.getsize(emb_path))})")

    if save_csv:
        feat_cols = [f"f{i}" for i in range(feats.shape[1])]
        df_feats = pd.DataFrame(feats, columns=feat_cols)
        csv_feat_path = os.path.join(out_dir, "features.csv")
        df_feats.to_csv(csv_feat_path, index=False)
        print(f"[ok] Salvou embeddings CSV em: {csv_feat_path} ({human_readable_size(os.path.getsize(csv_feat_path))})")

    meta_records = [
        {
            "idx": s.idx,
            "AccessionNumber": s.accession,
            "Classification": s.classification if s.classification is not None else -1,
            "dicom_path": s.path,
        }
        for s in infos
    ]
    df_meta = pd.DataFrame(meta_records).sort_values("idx").reset_index(drop=True)
    meta_path = os.path.join(out_dir, "metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"[ok] Salvou metadados em: {meta_path}")
    return df_meta


# Executa PCA 2D sobre os embeddings e salva os artefatos resultantes.
def run_pca_analysis(
    feats: np.ndarray,
    infos: List[SampleInfo],
    args: argparse.Namespace,
    out_dir: str
) -> np.ndarray:
    """Retorna as coordenadas PCA 2D, salvando CSV/PNG e variância explicada."""
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[0], 2), dtype=np.float32)

    print("[info] PCA 2D...")
    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(feats)
    df_pca = pd.DataFrame({"pca_x": coords[:, 0], "pca_y": coords[:, 1]})
    df_pca.to_csv(os.path.join(out_dir, "pca_2d.csv"), index=False)
    labels = [s.classification if s.classification is not None else -1 for s in infos]
    plot_scatter(coords, labels, clusters=None, title="PCA (colorido por classificação, sem cluster)", out_path=os.path.join(out_dir, "pca_2d.png"))
    with open(os.path.join(out_dir, "pca_explained_variance.json"), "w") as fp:
        json.dump({"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, fp, indent=2)
    return coords


# Calcula t-SNE opcional para explorar vizinhanças locais dos embeddings.
def run_tsne_analysis(
    feats: np.ndarray,
    infos: List[SampleInfo],
    args: argparse.Namespace,
    out_dir: str
) -> np.ndarray:
    """Gera t-SNE 2D (quando habilitado) e salva CSV/PNG correspondentes."""
    if not args.tsne or feats.shape[0] < 2:
        return np.zeros((feats.shape[0], 2), dtype=np.float32)

    print("[info] t-SNE 2D... (pode levar alguns minutos)")
    perplex = min(30, max(5, feats.shape[0] // 50))
    tsne = TSNE(n_components=2, perplexity=perplex, learning_rate="auto", init="pca", random_state=args.seed)
    coords = tsne.fit_transform(feats)
    df_tsne = pd.DataFrame({"tsne_x": coords[:, 0], "tsne_y": coords[:, 1]})
    df_tsne.to_csv(os.path.join(out_dir, "tsne_2d.csv"), index=False)
    labels = [s.classification if s.classification is not None else -1 for s in infos]
    plot_scatter(coords, labels, clusters=None, title="t-SNE (colorido por classificação, sem cluster)", out_path=os.path.join(out_dir, "tsne_2d.png"))
    return coords


# Aplica k-means, salva métricas e gráficos, anexando rótulos ao DataFrame.
def run_clustering_analysis(
    feats: np.ndarray,
    df_meta: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: str,
    pca_coords: np.ndarray,
    tsne_coords: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], List[int]]:
    """Executa k-means (2..8) com seleção automática de K e produz gráficos de apoio."""
    cluster_labels_list: List[int] = []
    history: List[Dict[str, Optional[float]]] = []
    metrics_out: Dict[str, Optional[float]]

    if feats.shape[0] >= 3:
        print("[info] k-means (seleção automática de K por silhouette, 2..8)...")
        k_labels, metrics_data, history = auto_kmeans(feats, k_min=2, k_max=8, random_state=args.seed)
        df_meta["cluster_k"] = k_labels
        cluster_labels_list = k_labels.tolist()
        metrics_out = dict(metrics_data)
        metrics_out["history"] = history
        sil_val = metrics_out.get("silhouette")
        db_val = metrics_out.get("davies_bouldin")
        sil_txt = f"{sil_val:.4f}" if isinstance(sil_val, (float, int)) else "n/a"
        db_txt = f"{db_val:.4f}" if isinstance(db_val, (float, int)) else "n/a"
        print(f"[ok] Clustering: K={metrics_out['best_k']} | silhouette={sil_txt} | davies_bouldin={db_txt}")

        labels_for_plots = df_meta["Classification"].tolist()
        plot_scatter(pca_coords, labels_for_plots, clusters=k_labels.tolist(), title=f"PCA (clusters K={metrics_out['best_k']}, cor=classe)", out_path=os.path.join(out_dir, "pca_2d_clusters.png"))
        if args.tsne and feats.shape[0] >= 2:
            plot_scatter(tsne_coords, labels_for_plots, clusters=k_labels.tolist(), title=f"t-SNE (clusters K={metrics_out['best_k']}, cor=classe)", out_path=os.path.join(out_dir, "tsne_2d_clusters.png"))
    else:
        df_meta["cluster_k"] = -1
        metrics_out = {"best_k": None, "silhouette": None, "davies_bouldin": None, "history": []}

    with open(os.path.join(out_dir, "clustering_metrics.json"), "w") as fp:
        json.dump(metrics_out, fp, indent=2)

    plot_kmeans_history(metrics_out.get("history", []), os.path.join(out_dir, "kmeans_metrics.png"))
    plot_cluster_counts(cluster_labels_list, os.path.join(out_dir, "cluster_distribution.png"))
    return df_meta, metrics_out, cluster_labels_list


# Combina metadados, projeções e clusters em um CSV consolidado.
def save_joined_table(
    df_meta: pd.DataFrame,
    pca_coords: np.ndarray,
    tsne_coords: np.ndarray,
    feats: np.ndarray,
    args: argparse.Namespace,
    out_dir: str
) -> pd.DataFrame:
    """Monta joined.csv com metadados + projeções + clusters para exploração posterior."""
    df_join = df_meta.copy()
    df_join["pca_x"] = pca_coords[:, 0] if pca_coords.shape[0] else np.nan
    df_join["pca_y"] = pca_coords[:, 1] if pca_coords.shape[0] else np.nan

    if args.tsne and feats.shape[0] >= 2 and os.path.exists(os.path.join(out_dir, "tsne_2d.csv")):
        df_join["tsne_x"] = tsne_coords[:, 0]
        df_join["tsne_y"] = tsne_coords[:, 1]
    else:
        df_join["tsne_x"] = np.nan
        df_join["tsne_y"] = df_join.get("tsne_y", np.nan)

    joined_path = os.path.join(out_dir, "joined.csv")
    df_join.to_csv(joined_path, index=False)
    print(f"[ok] Salvou consolidado em: {joined_path}")
    return df_join


# Escreve em JSON os metadados completos da execução (versões, tempos, args).
def write_session_info(
    run_dir: str,
    device: torch.device,
    dataset: MammoDicomDataset,
    feats: np.ndarray,
    args: argparse.Namespace,
    t0: float,
    t1: float,
    t2: float
) -> None:
    """Salva run/session_info.json com dados de ambiente e tempos da execução."""
    t3 = time.time()
    session = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__ if _HAS_TORCHVISION else None,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "n_samples": int(len(dataset)),
        "embedding_dim": int(feats.shape[1]) if feats.size else 2048,
        "timings_sec": {
            "total": float(t3 - t0),
            "feature_extraction": float(t2 - t1),
        },
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "session_info.json"), "w") as fp:
        json.dump(session, fp, indent=2)


# Imprime um resumo do volume de exames por classe clínica.
def log_class_summary(df_join: pd.DataFrame) -> None:
    """Exibe contagem por classe clínica (incluindo rótulos ausentes) para sanity check final."""
    try:
        counts = df_join["Classification"].value_counts(dropna=False).sort_index()
        print("\n[resumo] Quantidade por classificação:")
        for cls, ct in counts.items():
            nome = CLASS_MAP.get(int(cls), "desconhecida") if int(cls) != -1 else "sem rótulo"
            print(f"  Classe {cls}: {ct}  ({nome})")
    except Exception:
        pass


# ------------------- Rotina principal -------------------

# Orquestra todas as etapas do pipeline de extração, visualização e clustering.
def main():
    """Coordena todas as etapas da pipeline: argumentos, extração, análises e salvamento."""
    # Etapa 1: interpretar argumentos da CLI e montar as opções de execução.
    args = parse_arguments()

    # Etapa 2: preparar ambiente (tempo inicial, seeds e diretórios principais).
    t0 = time.time()
    seed_everything(args.seed)
    data_dir = find_best_data_dir(args.data_dir)
    preview_dir, run_dir = prepare_output_dirs(args.out_dir)

    # Etapa 3: escolher dinamicamente o dispositivo de execução.
    device = resolve_device(args.device)
    print(f"[info] Dispositivo: {device}")

    # Etapa 4: carregar rótulos clínicos e construir dataset/DataLoader.
    labels_map = load_labels_dict(args.csv_path)
    transform_model, transform_vis = get_transforms()
    dataset, loader = build_dataset_and_loader(data_dir, labels_map, args, transform_model)
    print(f"[info] Amostras consideradas: {len(dataset)} (após filtros).")
    if len(dataset) == 0:
        print("[erro] Nenhuma amostra encontrada. Abortando.")
        return

    # Etapa 5: gerar visualizações diagnósticas do pré-processamento.
    print("[info] Gerando visualizações obrigatórias do pré-processamento e amostras...")
    generate_mandatory_previews(dataset, preview_dir, transform_vis, args.preview_max)

    # Etapa 6: extrair embeddings com a ResNet50.
    t1 = time.time()
    feats, infos = extract_feature_embeddings(loader, device, args)
    t2 = time.time()
    if feats.shape[0] != len(infos):
        raise RuntimeError("Número de embeddings não coincide com metadados.")

    # Etapa 7: salvar embeddings/metadados estruturados.
    df_meta = save_embeddings_artifacts(feats, infos, args.out_dir, args.save_csv)

    # Etapa 8: executar análises de projeção (PCA/t-SNE).
    pca_coords = run_pca_analysis(feats, infos, args, args.out_dir)
    tsne_coords = run_tsne_analysis(feats, infos, args, args.out_dir)

    # Etapa 9: aplicar k-means, salvar métricas e gráficos auxiliares.
    df_meta, _, _ = run_clustering_analysis(feats, df_meta, args, args.out_dir, pca_coords, tsne_coords)

    # Etapa 10: consolidar tudo em um CSV amigável para exploração externa.
    df_join = save_joined_table(df_meta, pca_coords, tsne_coords, feats, args, args.out_dir)

    # Etapa 11: registrar metadados da execução e apresentar resumo final.
    write_session_info(run_dir, device, dataset, feats, args, t0, t1, t2)
    log_class_summary(df_join)

    print("\n[Finalizado] Arquivos gerados em:", os.path.abspath(args.out_dir))



if __name__ == "__main__":
    main()
