#!/usr/bin/env python3
# ruff: noqa
#
# train.py
# mammography-pipelines
#
# Trains EfficientNetB0/ResNet50/ViT density classifiers with optional caching, AMP, Grad-CAM, and evaluation exports.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Train EfficientNetB0/ResNet50/ViT for breast density with optional caches and AMP."""
import os
import signal
import sys
import hashlib
import argparse
import json
import logging
import time
import threading
import shlex
from datetime import datetime, timezone
from typing import Any, Sequence, Optional, Tuple, List, Dict
import torch
from torch import profiler
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


try:
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import ValidationError

from mammography.config import HP, TrainConfig
from mammography.utils.class_modes import (
    CLASS_MODE_HELP,
    VISIBLE_CLASS_MODES_METAVAR,
    get_label_mapper as build_label_mapper,
    get_num_classes,
    parse_classes_mode_arg,
)
from mammography.utils.common import (
    seed_everything,
    resolve_device,
    configure_runtime,
    setup_logging,
    increment_path,
    parse_float_list,
    get_reproducibility_info,
)
from mammography.utils import bool_flags as bool_flag_utils
from mammography.utils import export_formats as export_format_utils
from mammography.data.csv_loader import (
    load_dataset_dataframe,
    resolve_dataset_cache_mode,
    DATASET_PRESETS,
    resolve_paths_from_preset,
)
from mammography.data.dataset import MammoDensityDataset, mammo_collate, load_embedding_store
from mammography.data.splits import create_splits, create_three_way_split, load_splits_from_csvs
from mammography.io.dicom import is_dicom_path
from mammography.models.nets import build_model
from mammography.training.engine import (
    train_one_epoch,
    validate,
    save_metrics_figure,
    extract_embeddings,
    plot_history,
    save_predictions,
    save_atomic,
)

from mammography.commands.train_args import _normalize_bool_flags, _parse_bool_literal, parse_args
from mammography.commands.train_artifacts import (
    GracefulKiller,
    _clean_top_k,
    _export_figure_multi_format,
    _find_resume_checkpoint,
    _load_checkpoint_for_eval,
    _metrics_artifact_base,
    _parse_export_formats,
    _plot_history_format,
    _resolve_best_model_path,
    _resolve_checkpoint_candidate,
    _save_metrics_figure_format,
    _sort_top_k,
    _summarize_metrics_for_summary,
    _update_top_k,
    _write_metrics_artifacts,
    _write_split_artifacts,
    get_file_hash,
)
from mammography.commands import train_data as _train_data
from mammography.commands.train_data import (
    _assert_no_patient_leakage,
    _build_dataloader,
    _collect_dicom_paths,
    _missing_id_examples,
    _normalize_patient_id,
    _patient_ids_from_column,
    _patient_ids_from_dicom,
    _preflight_dicom_headers,
    _resolve_split_group_column,
    _select_patient_id_column,
    _unique_paths,
    get_label_mapper,
    prepare_training_data,
    resolve_loader_runtime,
)
from mammography.commands.train_ensemble import run_view_specific_ensemble
from mammography.commands.train_modeling import (
    _parse_class_weights,
    _resolve_backbone_module,
    _resolve_head_module,
    build_param_groups,
    freeze_backbone,
    unfreeze_last_block,
)
from mammography.commands.train_tracking import (
    ExperimentTracker,
    _init_tracker,
    _maybe_register_training_run,
    _sanitize_tracking_params,
)

_patch_lock = threading.Lock()


def _prepare_training_data_with_facade_patches(
    args: argparse.Namespace,
    csv_path: str,
    dicom_root: str | None,
    outdir_root: Path,
    logger: logging.Logger,
):
    """
    Temporarily route training-data helpers through this facade for compatibility.

    This is a monkeypatch compatibility layer for tests and legacy callers that
    replace helpers on mammography.commands.train. It saves the original methods
    on _train_data, replaces _train_data.load_dataset_dataframe,
    _train_data.create_splits, _train_data.create_three_way_split,
    _train_data.load_splits_from_csvs, and
    _train_data._assert_no_patient_leakage with this facade module's globals,
    calls _train_data.prepare_training_data(...), then restores the saved
    originals from ``original`` in a finally block.

    This helper depends on every name in ``patch_names`` being present in this
    module's ``globals()``. It mutates _train_data module attributes during the
    call, so concurrent training runs in the same Python process are
    unsupported and guarded by ``_patch_lock``.

    Removing this before callers migrate would make monkeypatches against
    mammography.commands.train.* land on the wrong module, causing tests or
    external code that rely on the historical facade import path to fail. It can
    be removed once those callers patch mammography.commands.train_data directly.
    
    Parameters:
        args (argparse.Namespace): Parsed training arguments used by prepare_training_data.
        csv_path (str): Path to the input CSV or preset identifier resolved for dataset preparation.
        dicom_root (str | None): Root directory for DICOM files, or None if not applicable.
        outdir_root (Path): Base output directory for prepared artifacts and temporary files.
        logger (logging.Logger): Logger to use for preparation progress and messages.
    
    Returns:
        The prepared training-data artifact returned by _train_data.prepare_training_data (typically a namespace or mapping containing dataframes, rows, normalization stats, label mapping, embedding store, and related metadata).
    """
    if not _patch_lock.acquire(blocking=False):
        raise RuntimeError(
            "Concurrent training-data facade patching is unsupported; "
            "_train_data.prepare_training_data is already running."
        )
    patch_names = (
        "load_dataset_dataframe",
        "create_splits",
        "create_three_way_split",
        "load_splits_from_csvs",
        "_assert_no_patient_leakage",
    )
    try:
        original = {name: getattr(_train_data, name) for name in patch_names}
        for name in patch_names:
            setattr(_train_data, name, globals()[name])
        try:
            return _train_data.prepare_training_data(
                args, csv_path, dicom_root, outdir_root, logger
            )
        finally:
            for name, value in original.items():
                setattr(_train_data, name, value)
    finally:
        _patch_lock.release()
from mammography.commands.train_loop import TrainingLoopResult, run_training_loop
from mammography.commands.train_setup import TrainCommandSetup, build_train_command, prepare_train_command_setup


def main(argv: Optional[Sequence[str]] = None):
    facade = sys.modules[__name__]
    setup = prepare_train_command_setup(argv, facade)
    prepared = _prepare_training_data_with_facade_patches(
        setup.args,
        setup.csv_path,
        setup.dicom_root,
        setup.outdir_root,
        setup.logger,
    )
    loop_result = run_training_loop(
        args=setup.args,
        prepared=prepared,
        csv_path=setup.csv_path,
        dicom_root=setup.dicom_root,
        outdir_root=setup.outdir_root,
        outdir_path=setup.outdir_path,
        metrics_dir=setup.metrics_dir,
        export_formats=setup.export_formats,
        logger=setup.logger,
        device=setup.device,
        killer=setup.killer,
    )
    run_view_specific_ensemble(
        args=setup.args,
        logger=setup.logger,
        views_to_train=prepared.views_to_train,
        outdir_path=setup.outdir_path,
        num_classes=prepared.num_classes,
        embedding_store=prepared.embedding_store,
        device=setup.device,
        val_rows=prepared.val_rows,
        view_column=prepared.view_column,
        cache_mode_val=loop_result.cache_mode_val,
        cache_dir=prepared.cache_dir,
        mapper=prepared.mapper,
        mean=prepared.mean,
        std=prepared.std,
        nw=loop_result.nw,
        persistent=loop_result.persistent,
        prefetch=loop_result.prefetch,
        export_formats=setup.export_formats,
    )
    _maybe_register_training_run(
        args=setup.args,
        outdir_root=setup.outdir_root,
        command=build_train_command(setup.argv_list),
        logger=setup.logger,
    )
