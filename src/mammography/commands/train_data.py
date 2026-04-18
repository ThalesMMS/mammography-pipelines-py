#!/usr/bin/env python3
#
# train_data.py
# mammography-pipelines
#
# Trains EfficientNetB0/ResNet50/ViT density classifiers with optional caching, AMP, Grad-CAM, and evaluation exports.
#
# Thales Matheus Mendonça Santos - April 2026
#
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
"""Train EfficientNetB0/ResNet50/ViT for breast density with optional caches and AMP."""

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import DataLoader

from mammography.data.csv_loader import (
    load_dataset_dataframe,
)
from mammography.data.dataset import EmbeddingStore, load_embedding_store
from mammography.data.splits import (
    create_splits,
    create_three_way_split,
    load_splits_from_csvs,
)
from mammography.io.dicom import is_dicom_path
from mammography.utils.class_modes import (
    get_label_mapper as build_label_mapper,
)
from mammography.utils.class_modes import (
    get_num_classes,
)
from mammography.utils.common import (
    parse_float_list,
)


@dataclass
class PreparedTrainingData:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame | None
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]] | None
    mean: list[float]
    std: list[float]
    num_classes: int
    split_group_column: str | None
    views_to_train: list[str | None]
    view_column: str
    embedding_store: EmbeddingStore | None
    cache_dir: str
    mapper: Callable[[int], int] | None


def get_label_mapper(mode):
    """Return a mapper function to collapse classes when running binary experiments."""
    return build_label_mapper(mode)


def _normalize_patient_id(value):
    """
    Normalize a patient identifier value to a cleaned string or `None`.

    Parameters:
        value: The raw patient identifier (any type) to normalize.

    Returns:
        The identifier as a stripped string, or `None` if `value` is `None`, NaN, or an empty/whitespace-only string.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _unique_paths(paths: Sequence[str]) -> list[str]:
    """
    Deduplicate a sequence of path strings, preserving first-seen order and skipping falsy values.

    Returns:
        list[str]: Unique, truthy paths in the original order they were first encountered.
    """
    uniq = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        uniq.append(path)
    return uniq


def _collect_dicom_paths(df: pd.DataFrame) -> list[str]:
    """
    Collects and returns unique DICOM file paths found in a DataFrame's `image_path` column preserving first-seen order.

    Parameters:
        df (pd.DataFrame): DataFrame that may contain an `image_path` column with file path values.

    Returns:
        list[str]: List of unique paths from `df["image_path"]` that satisfy `is_dicom_path`, preserving original order and excluding null/falsy entries.
    """
    if "image_path" not in df.columns:
        return []
    paths = [str(p) for p in df["image_path"].dropna().tolist()]
    dicom_paths = [p for p in paths if is_dicom_path(p)]
    return _unique_paths(dicom_paths)


def _preflight_dicom_headers(
    dicom_paths: Sequence[str],
    seed: int,
    logger: logging.Logger,
    max_samples: int = 1000,
    error_threshold: float = 0.01,
) -> None:
    """
    Validate a sample of DICOM files can be read and raise if the read error rate exceeds a threshold.

    Reads up to `max_samples` paths from `dicom_paths` (deterministically sampled using `seed` when the list is larger than `max_samples`) and attempts to read each file's DICOM header. Logs a summary of the number and fraction of read errors; if the fraction of failures is greater than `error_threshold` a RuntimeError is raised that includes example failing paths. If `dicom_paths` is empty the function logs and returns without error.

    Parameters:
        dicom_paths (Sequence[str]): Sequence of file paths to validate.
        seed (int): RNG seed used for deterministic sampling when the number of paths exceeds `max_samples`.
        max_samples (int): Maximum number of paths to sample and check.
        error_threshold (float): Fraction threshold (0.0-1.0) of read failures above which a RuntimeError is raised.
    """
    if not dicom_paths:
        logger.info("Preflight DICOM: nenhum caminho DICOM detectado; pulando.")
        return
    total = len(dicom_paths)
    sample_paths = list(dicom_paths)
    sampled = False
    if total > max_samples:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(total, size=max_samples, replace=False)
        sample_paths = [dicom_paths[i] for i in idxs]
        sampled = True
    errors = []
    error_types = []
    for path in sample_paths:
        try:
            pydicom.dcmread(path, stop_before_pixels=True, force=True)
        # Preflight must count any unreadable DICOM-ish file, including IO,
        # pydicom parsing, and malformed binary errors.
        except Exception as e:
            errors.append(path)
            if len(error_types) < 5:
                error_types.append((path, type(e).__name__))
    err_count = len(errors)
    err_rate = err_count / max(1, len(sample_paths))
    if sampled:
        logger.info(
            "Preflight DICOM: %d/%d erros (%.2f%%) na amostra %d/%d.",
            err_count,
            len(sample_paths),
            err_rate * 100.0,
            len(sample_paths),
            total,
        )
    else:
        logger.info(
            "Preflight DICOM: %d/%d erros (%.2f%%).",
            err_count,
            len(sample_paths),
            err_rate * 100.0,
        )
    if err_rate > error_threshold:
        examples = errors[:3]
        raise RuntimeError(
            "CRITICO: preflight DICOM falhou. "
            f"{err_count}/{len(sample_paths)} ({err_rate:.2%}) com erro "
            f"(ex: {examples}; tipos: {error_types})."
        )


def _select_patient_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Selects the first column that contains at least one normalized patient identifier.

    Parameters:
        df (pd.DataFrame): DataFrame to inspect for patient identifier columns.

    Returns:
        str | None: The name of the first column ('patient_id' or 'PatientID') that has at least one non-null normalized patient ID, or `None` if neither column contains valid IDs.
    """
    for col in ("patient_id", "PatientID"):
        if col in df.columns:
            series = df[col].apply(_normalize_patient_id)
            if series.notna().any():
                return col
    return None


def _missing_id_examples(df: pd.DataFrame) -> list[str]:
    """
    Return up to three example values from common identifier columns when patient IDs are missing.

    Returns:
        list[str]: Up to three stringified values from the first available column among `"accession"` or `"image_path"`. Returns an empty list if neither column is present.
    """
    for col in ("accession", "image_path"):
        if col in df.columns:
            return df[col].astype(str).head(3).tolist()
    return []


def _patient_ids_from_column(
    df: pd.DataFrame,
    column: str,
    split_label: str,
    *,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> set[str]:
    """
    Collect normalized patient IDs from a DataFrame column and validate missing values.

    If the column contains no missing patient IDs, returns the set of normalized IDs.
    If there are missing values and `strict` is False, logs a warning (if `logger` is provided)
    and returns the set of non-missing normalized IDs.
    If there are missing values and `strict` is True, raises a RuntimeError describing the
    number of missing entries and providing example rows.

    Parameters:
        df (pd.DataFrame): DataFrame containing the column to extract.
        column (str): Name of the column that holds patient identifiers to normalize.
        split_label (str): Label used in log and error messages to identify the split.
        strict (bool): If True, treat missing patient IDs as an error; if False, warn and continue.

    Returns:
        set[str]: Set of normalized patient IDs present in the specified column.
    """
    series = df[column].apply(_normalize_patient_id)
    missing_mask = series.isna()
    if missing_mask.any():
        examples = _missing_id_examples(df[missing_mask])
        if not strict:
            if logger is not None:
                logger.warning(
                    "Leakage check parcial: coluna '%s' com %d valores vazios no split %s; "
                    "ignorando esses patient_ids (ex: %s).",
                    column,
                    missing_mask.sum(),
                    split_label,
                    examples,
                )
            return set(series.dropna().tolist())
        raise RuntimeError(
            f"CRITICO: coluna '{column}' com {missing_mask.sum()} valores vazios "
            f"no split {split_label}; nao e possivel validar vazamento (ex: {examples})."
        )
    return set(series.tolist())


def _patient_ids_from_dicom(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    logger: logging.Logger,
    test_df: pd.DataFrame | None = None,
    *,
    strict: bool = True,
) -> tuple[set[str], set[str], set[str]]:
    """
    Extract patient identifiers from DICOM headers referenced by split image_path columns.

    Parameters:
        train_df (pd.DataFrame): Training split containing an `image_path` column with file paths.
        val_df (pd.DataFrame): Validation split containing an `image_path` column with file paths.
        logger (logging.Logger): Logger used for warnings and informational messages.
        test_df (pd.DataFrame | None): Optional test split containing an `image_path` column.
        strict (bool): If True, unreadable headers or missing PatientID values raise RuntimeError; otherwise warnings are logged and readable IDs are returned.

    Returns:
        tuple[set[str], set[str], set[str]]: Patient ID sets for train, val, and test (empty test set when `test_df` is None). If non-DICOM paths are present, returns empty sets after logging a warning.

    Raises:
        RuntimeError: If either DataFrame lacks an `image_path` column.
        RuntimeError: If no valid image paths are found to inspect.
        RuntimeError: If strict mode is enabled and DICOM header reads fail for one or more files (includes count and examples).
        RuntimeError: If strict mode is enabled and one or more DICOM files lack a `PatientID` (includes count and examples).
    """
    if "image_path" not in train_df.columns or "image_path" not in val_df.columns:
        raise RuntimeError(
            "CRITICO: coluna image_path ausente; nao e possivel validar vazamento."
        )
    if test_df is not None and "image_path" not in test_df.columns:
        raise RuntimeError(
            "CRITICO: coluna image_path ausente; nao e possivel validar vazamento."
        )
    all_paths = _unique_paths(
        (
            str(p)
            for frame in (train_df, val_df, test_df)
            if frame is not None
            for p in frame["image_path"].dropna().tolist()
        )
    )
    if not all_paths:
        raise RuntimeError("CRITICO: nenhum image_path valido para validar vazamento.")
    non_dicom = [p for p in all_paths if not is_dicom_path(p)]
    if non_dicom:
        examples = non_dicom[:3]
        logger.warning(
            "Amostras nao-DICOM detectadas (%d/%d); pulando verificacao de vazamento por PatientID (ex: %s).",
            len(non_dicom),
            len(all_paths),
            examples,
        )
        return set(), set(), set()
    logger.info(
        "Lendo cabecalhos DICOM para obter PatientID (%d arquivos).", len(all_paths)
    )
    patient_by_path: dict[str, str] = {}
    read_errors = []
    missing_ids = []
    for path in all_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as exc:
            read_errors.append((path, exc))
            continue
        patient_id = _normalize_patient_id(getattr(ds, "PatientID", None))
        if not patient_id:
            missing_ids.append(path)
            continue
        patient_by_path[path] = patient_id
    if read_errors:
        err_rate = len(read_errors) / max(1, len(all_paths))
        examples = [p for p, _ in read_errors[:3]]
        message = (
            "CRITICO: falha ao ler cabecalhos DICOM para obter PatientID. "
            f"{len(read_errors)}/{len(all_paths)} ({err_rate:.2%}) com erro (ex: {examples})."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning("%s Prosseguindo porque split_mode=random.", message)
    if missing_ids:
        err_rate = len(missing_ids) / max(1, len(all_paths))
        examples = missing_ids[:3]
        message = (
            "CRITICO: PatientID ausente em arquivos DICOM; "
            f"{len(missing_ids)}/{len(all_paths)} ({err_rate:.2%}) amostras sem PatientID (ex: {examples})."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning("%s Prosseguindo porque split_mode=random.", message)

    def _ids_for(df: pd.DataFrame | None) -> set[str]:
        """
        Collects patient IDs referenced by each row's image_path using the module's path-to-patient mapping.

        Parameters:
            df (pd.DataFrame): DataFrame with an "image_path" column whose values are looked up.

        Returns:
            set[str]: Set of patient IDs found for image paths present in the `patient_by_path` mapping.
        """
        ids = set()
        if df is None:
            return ids
        for path in df["image_path"].tolist():
            patient_id = patient_by_path.get(str(path))
            if patient_id:
                ids.add(patient_id)
        return ids

    return _ids_for(train_df), _ids_for(val_df), _ids_for(test_df)


def _assert_no_patient_leakage(
    train_patients: set[str],
    val_patients: set[str],
    *,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """
    Validate that there is no patient ID overlap between training and validation sets.

    If any patient IDs are present in both `train_patients` and `val_patients` this function
    either raises a RuntimeError (when `strict` is True) or logs a warning and returns (when
    `strict` is False).

    Parameters:
        train_patients (set[str]): Set of patient IDs assigned to the training split.
        val_patients (set[str]): Set of patient IDs assigned to the validation split.
        strict (bool): If True, raise RuntimeError on detected overlap; if False, log a warning and continue.
        logger (logging.Logger | None): Optional logger used for warning messages; a module logger is used if omitted.
    """
    intersec = train_patients.intersection(val_patients)
    if not intersec:
        return
    sample = sorted(intersec)[:3]
    message = (
        "CRITICO: vazamento de dados detectado! "
        f"{len(intersec)} pacientes aparecem em train e val (ex: {sample})."
    )
    if strict:
        raise RuntimeError(message)
    warn_logger = logger or logging.getLogger("mammography")
    warn_logger.warning("%s Prosseguindo porque split_mode=random.", message)


def _resolve_split_group_column(
    df: pd.DataFrame,
    split_mode: str,
) -> str | None:
    """
    Determine which dataframe column should be used to group examples when creating splits.

    If `split_mode` is "random" or "preset", no grouping column is required and the function returns `None`. For other split modes, selects the dataset's patient identifier column ("patient_id" or "PatientID") and returns its name.

    Parameters:
        df (pd.DataFrame): Dataset to inspect for a patient identifier column.
        split_mode (str): Split mode string; grouping is required for modes other than "random" or "preset".

    Returns:
        str | None: The name of the patient identifier column to use for group-based splitting, or `None` when no grouping is needed.

    Raises:
        SystemExit: If a group-based split mode is requested but no patient identifier column is found.
    """
    mode = (split_mode or "random").lower()
    if mode in {"random", "preset"}:
        return None
    patient_col = _select_patient_id_column(df)
    if patient_col is None:
        raise SystemExit(
            "split-mode=patient requer coluna 'patient_id' ou 'PatientID' no dataset carregado."
        )
    return patient_col


def resolve_loader_runtime(args, device: torch.device):
    """
    Adjust DataLoader worker, prefetch, and persistence settings based on CLI options, environment variables, and device/platform heuristics.

    Parameters:
        args (argparse.Namespace): CLI arguments with attributes `num_workers`, `prefetch_factor`, `persistent_workers`, and `loader_heuristics`. Environment variable `MAMMO_NUM_WORKERS` may override `num_workers`.
        device (torch.device): Target device used for training (e.g., CPU, CUDA, MPS).

    Returns:
        tuple: (num_workers, prefetch_factor_or_None, persistent_workers)
            - num_workers (int): Number of worker processes to use (clamped or forced to 0 in restricted environments).
            - prefetch_factor_or_None (int | None): `prefetch_factor` to pass to DataLoader, or `None` when disabled.
            - persistent_workers (bool): Whether to enable persistent workers.
    """
    num_workers = args.num_workers
    prefetch = (
        args.prefetch_factor
        if args.prefetch_factor and args.prefetch_factor > 0
        else None
    )
    persistent = args.persistent_workers
    override_workers = os.environ.get("MAMMO_NUM_WORKERS")
    if override_workers:
        try:
            num_workers = max(0, int(override_workers))
        except ValueError:
            logging.getLogger("mammography").warning(
                "MAMMO_NUM_WORKERS invalido (%s). Usando valor de linha de comando.",
                override_workers,
            )
    if not args.loader_heuristics:
        if num_workers == 0:
            return 0, None, False
        return num_workers, prefetch, persistent
    if num_workers == 0:
        return 0, None, False
    if device.type == "mps":
        return 0, None, False
    if os.name == "nt" and device.type == "cuda" and sys.version_info >= (3, 13):
        logging.getLogger("mammography").warning(
            (
                "Desabilitando DataLoader multiprocessado em Windows + CUDA + Python %s "
                "para evitar deadlocks observados durante treino oficial."
            ),
            ".".join(str(part) for part in sys.version_info[:3]),
        )
        return 0, None, False
    if device.type == "cpu":
        num_workers = max(0, min(num_workers, os.cpu_count() or 0))
        if num_workers == 0:
            return 0, None, False
        return num_workers, prefetch, persistent
    return num_workers, prefetch, persistent


def _build_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    shuffle: bool,
    sampler: Optional[torch.utils.data.Sampler],
    dl_kwargs: dict[str, Any],
    logger: logging.Logger,
) -> DataLoader:
    """
    Builds a torch DataLoader for the given dataset, retrying with safer worker settings if a PermissionError occurs.

    Parameters:
        dataset: The dataset to wrap in a DataLoader.
        shuffle: Whether to shuffle data each epoch.
        sampler: Optional sampler to draw indices from.
        dl_kwargs: Keyword arguments forwarded to torch.utils.data.DataLoader (e.g., batch_size, num_workers, prefetch_factor, persistent_workers).
        logger: Logger used to emit a warning if a fallback configuration is required.

    Returns:
        A configured torch.utils.data.DataLoader instance. If creating the DataLoader raises a PermissionError and `dl_kwargs["num_workers"]` is greater than 0, the function will log a warning and retry with `num_workers=0`, `persistent_workers=False`, and without `prefetch_factor`.

    Raises:
        PermissionError: If DataLoader construction fails with a PermissionError and `dl_kwargs["num_workers"]` is 0.
    """
    try:
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            sampler=sampler,
            **dl_kwargs,
        )
        if dl_kwargs.get("num_workers", 0) > 0:
            _ = iter(dataloader)
        return dataloader
    except PermissionError as exc:
        if dl_kwargs.get("num_workers", 0) == 0:
            raise
        logger.warning(
            "Falha ao iniciar DataLoader com num_workers=%s (%s). Repetindo com num_workers=0.",
            dl_kwargs.get("num_workers"),
            exc,
        )
        safe_kwargs = dict(dl_kwargs)
        safe_kwargs["num_workers"] = 0
        safe_kwargs["persistent_workers"] = False
        safe_kwargs.pop("prefetch_factor", None)
        return DataLoader(
            dataset,
            shuffle=shuffle,
            sampler=sampler,
            **safe_kwargs,
        )


def prepare_training_data(
    args: argparse.Namespace,
    csv_path: str,
    dicom_root: str | None,
    outdir_root: Path,
    logger: logging.Logger,
) -> PreparedTrainingData:
    # Load Data
    """
    Prepare and validate dataset splits and related training artifacts for mammography classifier training.

    Loads a dataset CSV, filters labeled samples, parses normalization stats, creates or loads train/val(/test) splits, validates patient-level leakage (using a patient-id column or DICOM PatientID headers), configures optional view-specific training and optional embedding-store fusion, and returns a PreparedTrainingData container with frames, row records, normalization, class metadata, split configuration, and optional resources.

    Parameters:
        args (argparse.Namespace): CLI arguments controlling dataset loading, splitting, view and embedding options (e.g., split_mode, seed, val_frac, test_frac, view_specific_training, embeddings_dir, arch, mean, std, classes, cache_dir).
        csv_path (str): Path to the dataset CSV used to build the DataFrame.
        dicom_root (str | None): Optional root directory used to resolve DICOM image paths when loading the dataset.
        outdir_root (Path): Output directory root used to derive a cache directory when not provided in args.
        logger (logging.Logger): Logger for informational messages and warnings.

    Returns:
        PreparedTrainingData: Dataclass containing the prepared dataset and metadata:
            - df, train_df, val_df, optional test_df (pd.DataFrame)
            - train_rows, val_rows, optional test_rows (list[dict])
            - mean, std (list[float])
            - num_classes (int)
            - split_group_column (str | None)
            - views_to_train (list[str | None]), view_column (str)
            - embedding_store (optional), cache_dir (str), mapper (label mapping)

    Raises:
        SystemExit: On missing or invalid CLI configuration (e.g., bad mean/std parsing, missing preset CSVs, missing view specification, unsupported embedding fusion).
        RuntimeError: On critical validation failures (e.g., no image paths for leakage check, unreadable DICOM headers beyond threshold, missing DICOM PatientID, detected patient-level leakage between splits).
    """
    df = load_dataset_dataframe(
        csv_path,
        dicom_root,
        exclude_class_5=not args.include_class_5,
        dataset=args.dataset,
        auto_detect=args.auto_detect,
    )
    logger.info(
        "Loaded %d samples from dataset '%s'.",
        len(df),
        args.dataset or "custom",
    )
    df = df[df["professional_label"].notna()]
    logger.info("Valid samples (with label): %d", len(df))
    if args.subset and args.subset > 0:
        subset_count = min(args.subset, len(df))
        df = df.sample(n=subset_count, random_state=args.seed).reset_index(drop=True)
        logger.info("Subset selecionado: %d amostras.", subset_count)

    try:
        mean = parse_float_list(args.mean, expected_len=3, name="mean")
        std = parse_float_list(args.std, expected_len=3, name="std")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    num_classes = get_num_classes(args.classes)

    split_group_column = _resolve_split_group_column(df, args.split_mode)

    # Route to appropriate split function based on configuration
    test_df = None
    if args.split_mode == "preset":
        # Load pre-defined splits from CSV files
        if not args.train_csv or not args.val_csv:
            raise SystemExit("Modo 'preset' requer --train-csv e --val-csv.")
        train_df, val_df, test_df = load_splits_from_csvs(
            args.train_csv,
            args.val_csv,
            args.test_csv,
        )
    elif args.test_frac > 0:
        # Create three-way split (train/val/test)
        train_df, val_df, test_df = create_three_way_split(
            df,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            num_classes=num_classes,
            ensure_all_splits_have_all_classes=args.split_ensure_all_classes,
            max_tries=args.split_max_tries,
            group_col=split_group_column,
        )
    else:
        # Default two-way split (train/val) for backward compatibility
        train_df, val_df = create_splits(
            df,
            val_frac=args.val_frac,
            seed=args.seed,
            num_classes=num_classes,
            ensure_val_has_all_classes=args.split_ensure_all_classes,
            max_tries=args.split_max_tries,
            group_col=split_group_column,
        )

    split_frames = [train_df, val_df]
    if test_df is not None:
        split_frames.append(test_df)
    strict_patient_ids = args.split_mode != "random"

    common_columns = set(split_frames[0].columns)
    for frame in split_frames[1:]:
        common_columns &= set(frame.columns)

    patient_col = None
    for col in ("patient_id", "PatientID"):
        if col in common_columns and any(
            frame[col].apply(_normalize_patient_id).notna().any()
            for frame in split_frames
        ):
            patient_col = col
            break

    if patient_col:
        logger.info(
            "Leakage check: usando coluna comum '%s' para patient_id.",
            patient_col,
        )
        train_patients = _patient_ids_from_column(
            train_df,
            patient_col,
            "train",
            strict=strict_patient_ids,
            logger=logger,
        )
        val_patients = _patient_ids_from_column(
            val_df,
            patient_col,
            "val",
            strict=strict_patient_ids,
            logger=logger,
        )
        test_patients = (
            _patient_ids_from_column(
                test_df,
                patient_col,
                "test",
                strict=strict_patient_ids,
                logger=logger,
            )
            if test_df is not None
            else set()
        )
    else:
        logger.info(
            "Leakage check: nenhuma coluna patient_id comum a todos os splits; "
            "usando PatientID dos cabecalhos DICOM."
        )
        all_paths = _unique_paths(
            [
                str(path)
                for frame in split_frames
                if "image_path" in frame.columns
                for path in frame["image_path"].dropna().tolist()
            ]
        )
        if not all_paths:
            raise RuntimeError(
                "CRITICO: nenhum image_path valido para validar vazamento."
            )
        non_dicom = [path for path in all_paths if not is_dicom_path(path)]
        if non_dicom:
            if args.split_mode == "random":
                logger.warning(
                    (
                        "Leakage check ignorado: split random sem patient_id contem %d amostras "
                        "nao-DICOM."
                    ),
                    len(non_dicom),
                )
                train_patients = set()
                val_patients = set()
                test_patients = set()
            else:
                examples = non_dicom[:3]
                raise RuntimeError(
                    "CRITICO: amostras nao-DICOM sem patient_id; nao e possivel validar vazamento "
                    f"(ex: {examples})."
                )
        else:
            dicom_paths = _unique_paths(
                path for frame in split_frames for path in _collect_dicom_paths(frame)
            )
            _preflight_dicom_headers(dicom_paths, args.seed, logger)
            train_patients, val_patients, test_patients = _patient_ids_from_dicom(
                train_df,
                val_df,
                logger,
                test_df,
                strict=strict_patient_ids,
            )
    _assert_no_patient_leakage(
        train_patients,
        val_patients,
        strict=strict_patient_ids,
        logger=logger,
    )
    if test_patients:
        # Also check for train/test and val/test leakage
        train_test_intersec = train_patients.intersection(test_patients)
        if train_test_intersec:
            sample = sorted(train_test_intersec)[:3]
            message = (
                "CRITICO: vazamento de dados detectado! "
                f"{len(train_test_intersec)} pacientes aparecem em train e test (ex: {sample})."
            )
            if strict_patient_ids:
                raise RuntimeError(message)
            logger.warning(
                "%s Prosseguindo porque split_mode=%s.",
                message,
                args.split_mode,
            )
        val_test_intersec = val_patients.intersection(test_patients)
        if val_test_intersec:
            sample = sorted(val_test_intersec)[:3]
            message = (
                "CRITICO: vazamento de dados detectado! "
                f"{len(val_test_intersec)} pacientes aparecem em val e test (ex: {sample})."
            )
            if strict_patient_ids:
                raise RuntimeError(message)
            logger.warning(
                "%s Prosseguindo porque split_mode=%s.",
                message,
                args.split_mode,
            )

    train_rows = train_df.to_dict("records")
    val_rows = val_df.to_dict("records")
    test_rows = test_df.to_dict("records") if test_df is not None else None

    def _has_view_values(df: pd.DataFrame | None, column: str) -> bool:
        """
        Check whether a DataFrame column contains at least one non-empty value.

        Parameters:
            df (pd.DataFrame | None): DataFrame to inspect; `None` is treated as absent.
            column (str): Column name to check.

        Returns:
            True if `column` exists in `df` and contains at least one non-null, non-empty string after stripping whitespace, False otherwise.
        """
        if df is None or column not in df.columns:
            return False
        values = df[column].dropna().astype(str).str.strip()
        return bool(values.ne("").any())

    # View-specific training setup
    views_to_train = []
    view_column = args.view_column
    if args.view_specific_training:
        view_column_candidates = [
            args.view_column,
            "ViewPosition",
            "view_position",
            "View",
        ]
        selected_requested_column = False
        for candidate in view_column_candidates:
            # Test splits may be absent or unlabeled; only train/val must drive selection.
            if _has_view_values(train_df, candidate) and _has_view_values(
                val_df, candidate
            ):
                view_column = candidate
                selected_requested_column = candidate == args.view_column
                break
        else:
            raise SystemExit(
                f"View column '{args.view_column}' nao encontrada no DataFrame."
            )
        if selected_requested_column:
            logger.info(
                "Using requested view column '%s' for view-specific training.",
                view_column,
            )
        else:
            logger.info(
                "Using fallback view column '%s' for view-specific training (requested '%s').",
                view_column,
                args.view_column,
            )
        raw_views = getattr(args, "views", None)
        if not raw_views or not str(raw_views).strip():
            raise SystemExit("Nenhuma view especificada em --views.")
        views_to_train = [v.strip() for v in str(raw_views).split(",") if v.strip()]
        if not views_to_train:
            raise SystemExit("Nenhuma view especificada em --views.")
        logger.info("View-specific training habilitado para views: %s", views_to_train)
    else:
        views_to_train = [None]  # Single training run without view filtering

    embedding_store = None
    if args.embeddings_dir:
        _emb_root = Path(args.embeddings_dir)
        if (_emb_root / "features.npy").exists() and (
            _emb_root / "metadata.csv"
        ).exists():
            if args.arch not in ["efficientnet_b0", "resnet50"]:
                raise SystemExit(
                    "Fusao de embeddings so esta disponivel para efficientnet_b0 e resnet50."
                )
            embedding_store = load_embedding_store(args.embeddings_dir)
            logger.info("Embeddings carregadas de %s", args.embeddings_dir)

            def _count_missing(rows):
                """
                Count rows whose embedding is missing from the embedding store.

                Parameters:
                    rows (Iterable[dict]): Sequence of row records to check with `embedding_store.lookup`.

                Returns:
                    int: Number of rows for which `embedding_store.lookup(row)` returns `None`.
                """
                return sum(1 for r in rows if embedding_store.lookup(r) is None)  # type: ignore[union-attr]

            missing_train = _count_missing(train_rows)
            missing_val = _count_missing(val_rows)
            missing_test = _count_missing(test_rows) if test_rows else 0
            if missing_train or missing_val or missing_test:
                if test_rows:
                    logger.warning(
                        "Embeddings ausentes: train=%s, val=%s, test=%s (total train=%s, val=%s, test=%s).",
                        missing_train,
                        missing_val,
                        missing_test,
                        len(train_rows),
                        len(val_rows),
                        len(test_rows),
                    )
                else:
                    logger.warning(
                        "Embeddings ausentes: train=%s, val=%s (total train=%s, val=%s).",
                        missing_train,
                        missing_val,
                        len(train_rows),
                        len(val_rows),
                    )
        else:
            logger.info(
                "Embeddings dir '%s' especificado mas features.npy/metadata.csv nao encontrados; "
                "prosseguindo sem fusao de embeddings.",
                args.embeddings_dir,
            )

    cache_dir = args.cache_dir or str(outdir_root / "cache")

    mapper = get_label_mapper(args.classes)
    return PreparedTrainingData(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        mean=mean,
        std=std,
        num_classes=num_classes,
        split_group_column=split_group_column,
        views_to_train=views_to_train,
        view_column=view_column,
        embedding_store=embedding_store,
        cache_dir=cache_dir,
        mapper=mapper,
    )
