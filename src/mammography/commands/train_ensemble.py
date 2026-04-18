#!/usr/bin/env python3
# ruff: noqa
#
# train_ensemble.py
# mammography-pipelines
#
# Trains EfficientNetB0/ResNet50/ViT density classifiers with optional caching, AMP, Grad-CAM, and evaluation exports.
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Train EfficientNetB0/ResNet50/ViT for breast density with optional caches and AMP."""

import argparse
import json
import logging
from typing import Any
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from mammography.data.dataset import MammoDensityDataset
from mammography.models.nets import build_model
from mammography.training.engine import (
    save_metrics_figure,
)

from mammography.commands.train_artifacts import _save_metrics_figure_format


def _empty_ensemble_metrics(num_classes: int) -> dict[str, Any]:
    labels = [str(i) for i in range(num_classes)]
    empty_report = {
        label: {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 0,
        }
        for label in labels
    }
    empty_report["accuracy"] = 0.0
    empty_report["macro avg"] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0,
    }
    empty_report["weighted avg"] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0,
    }
    return {
        "acc": 0.0,
        "kappa_quadratic": 0.0,
        "auc_ovr": 0.0,
        "num_samples": 0,
        "f1_macro": 0.0,
        "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
        "classification_report": empty_report,
    }


def _write_ensemble_metrics(
    ensemble_metrics: dict[str, Any],
    *,
    outdir_path: Path,
    export_formats: list[str],
    logger: logging.Logger,
) -> None:
    ensemble_metrics_dir = outdir_path / "metrics"
    ensemble_metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(
        ensemble_metrics_dir / "ensemble_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(ensemble_metrics, f, indent=2, default=str)
    save_metrics_figure(
        ensemble_metrics, str(ensemble_metrics_dir / "ensemble_metrics.png")
    )

    if export_formats:
        base_name = "ensemble_metrics"
        figures_dir = outdir_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for fmt in export_formats:
            _save_metrics_figure_format(
                ensemble_metrics,
                str(figures_dir / f"{base_name}.{fmt}"),
            )

    logger.info(
        "Metricas de ensemble salvas em %s",
        ensemble_metrics_dir / "ensemble_metrics.json",
    )


def run_view_specific_ensemble(
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
    views_to_train: list[str | None],
    outdir_path: Path,
    num_classes: int,
    embedding_store: Any | None,
    device: torch.device,
    val_rows: list[dict[str, Any]],
    view_column: str,
    cache_mode_val: str,
    cache_dir: str,
    mapper: Any,
    mean: list[float],
    std: list[float],
    nw: int,
    persistent: bool,
    prefetch: int | None,
    export_formats: list[str],
) -> None:
    # Ensemble evaluation after all view-specific training completes
    """
    Run post-training ensemble evaluation across per-view models and export ensemble metrics and figures.

    This function (when view-specific training is enabled and an ensemble method is selected) attempts to load each view's best checkpoint, builds an ensemble predictor from available view models, evaluates the ensemble on validation patients that have predictions for every required view, computes classification metrics (accuracy, quadratic-weighted kappa, multi-class OVR AUC, confusion matrix, classification report, macro F1), and writes metrics and figures to disk under the provided output directory. If fewer than two view models are available or no complete patients produce predictions, an all-zero metrics payload is generated and saved.

    Parameters:
        args (argparse.Namespace): CLI/runtime arguments that control behavior (must contain view_specific_training, ensemble_method, arch, batch_size, img_size, amp, train_backbone, unfreeze_last_block, and related flags).
        views_to_train (list[str | None]): Views that were trained; used to locate per-view output directories and checkpoints.
        outdir_path (Path): Base output directory for the training run; per-view subdirectories and the metrics output directory are derived from this path.
        num_classes (int): Number of target classes for metric computation and model construction.
        embedding_store (Any | None): Optional embedding store; its feature_dim is used to build models when present.
        device (torch.device): Device used for model inference.
        val_rows (list[dict[str, Any]]): Validation rows (one dict per sample) used to build per-sample datasets and group samples by patient.
        view_column (str): Name of the column in each validation row that indicates the sample's view.
        cache_mode_val (str): Cache mode passed to dataset construction for validation samples.
        cache_dir (str): Path to cache directory passed to dataset construction.
        mapper (callable | None): Optional label mapping callable applied to raw labels; if None, labels are converted by subtracting 1.
        mean (list[float]): Per-channel mean normalization for dataset construction.
        std (list[float]): Per-channel std normalization for dataset construction.
        nw (int): Number of data loader workers used when building the validation dataloader.
        persistent (bool): Whether to use persistent workers in the validation dataloader.
        prefetch (int | None): Optional prefetch factor for the dataloader.
        export_formats (list[str]): Additional figure export formats; if non-empty, metric figures are exported in these formats as well.

    Side effects:
        - Loads model checkpoints from per-view output directories derived from outdir_path.
        - Writes ensemble_metrics.json and metric figure files under outdir_path/metrics (and optionally outdir_path/figures).
    """
    if args.view_specific_training and args.ensemble_method != "none":
        logger.info("Evaluando ensemble com metodo '%s'...", args.ensemble_method)

        # Import EnsemblePredictor
        from mammography.models.cancer_models import EnsemblePredictor

        # Load best models for each view
        view_models = {}
        for view in views_to_train:
            if view is None:
                logger.warning(
                    "View nula encontrada durante ensemble; pulando modelo default."
                )
                continue
            view_name = str(view)
            view_outdir_path = outdir_path.parent / f"{outdir_path.name}_{view_name}"
            best_model_name = f"best_model_{view_name.lower()}.pt"
            best_model_path = view_outdir_path / best_model_name

            if not best_model_path.exists():
                logger.warning(
                    "Modelo para view '%s' nao encontrado em %s; pulando ensemble.",
                    view_name,
                    best_model_path,
                )
                continue

            # Create a new model instance and load weights
            view_model = build_model(
                args.arch,
                num_classes=num_classes,
                train_backbone=args.train_backbone,
                unfreeze_last_block=args.unfreeze_last_block,
                pretrained=False,
                extra_feature_dim=embedding_store.feature_dim if embedding_store else 0,
            ).to(device)

            state = torch.load(best_model_path, map_location=device, weights_only=True)
            try:
                view_model.load_state_dict(state, strict=True)
            except RuntimeError:
                logger.exception(
                    "Falha ao carregar checkpoint de ensemble com strict=True: %s",
                    best_model_path,
                )
                raise
            view_model.eval()
            view_models[view_name] = view_model
            logger.info(
                "Carregado modelo para view '%s' de %s", view_name, best_model_path
            )

        if len(view_models) < 2:
            logger.warning(
                "Ensemble requer pelo menos 2 modelos treinados; encontrado %d. Pulando ensemble.",
                len(view_models),
            )
            _write_ensemble_metrics(
                _empty_ensemble_metrics(num_classes),
                outdir_path=outdir_path,
                export_formats=export_formats,
                logger=logger,
            )
            logger.info("Arquivo de metricas vazio do ensemble gerado como fallback.")
        else:
            if args.ensemble_method == "weighted":
                raise ValueError(
                    "View-specific weighted ensemble requires explicit per-view "
                    "weights, but training orchestration does not configure them."
                )
            # Create ensemble predictor
            ensemble = EnsemblePredictor(view_models, method=args.ensemble_method)

            # Group validation samples by patient for ensemble evaluation
            logger.info("Agrupando amostras de validacao por paciente para ensemble...")
            patient_samples = {}
            for row in val_rows:
                patient_id = (
                    row.get("patient_id")
                    or row.get("PatientID")
                    or row.get("accession")
                )
                if not patient_id:
                    continue
                view = row.get(view_column)
                if not view or view not in view_models:
                    continue

                if patient_id not in patient_samples:
                    patient_samples[patient_id] = {}
                patient_samples[patient_id].setdefault(view, []).append(row)

            # Filter to patients with all required views
            required_views = set(view_models.keys())
            complete_patients = {
                pid: samples
                for pid, samples in patient_samples.items()
                if set(samples.keys()) == required_views
            }

            logger.info(
                "Encontrados %d pacientes com todas as views (%s) para ensemble de %d pacientes totais.",
                len(complete_patients),
                ", ".join(sorted(required_views)),
                len(patient_samples),
            )

            all_y = []
            all_p = []
            all_prob = []

            if len(complete_patients) == 0:
                logger.warning(
                    "Nenhum paciente com todas as views; nao e possivel avaliar ensemble."
                )
            else:
                # Evaluate ensemble on patients with all views

                with torch.no_grad():
                    for patient_id, view_samples in tqdm(
                        complete_patients.items(), desc="Ensemble Val", leave=False
                    ):
                        # Get predictions from each view
                        view_probs = {}
                        label = None

                        for view, rows in view_samples.items():
                            sample_probs = []
                            for row in rows:
                                # Get sample label (should be same across views for same patient)
                                if label is None:
                                    raw_label = row.get("professional_label")
                                    if raw_label is not None and not (
                                        isinstance(raw_label, float)
                                        and np.isnan(raw_label)
                                    ):
                                        label_val = int(raw_label)
                                        label = (
                                            mapper(label_val)
                                            if mapper
                                            else (label_val - 1)
                                        )

                                # Create single-sample dataset to load image
                                sample_ds = MammoDensityDataset(
                                    [row],
                                    args.img_size,
                                    train=False,
                                    augment=False,
                                    cache_mode=cache_mode_val,
                                    cache_dir=cache_dir,
                                    split_name="val_ensemble",
                                    label_mapper=mapper,
                                    embedding_store=embedding_store,
                                    mean=mean,
                                    std=std,
                                )

                                if len(sample_ds) == 0:
                                    continue
                                sample_data = sample_ds[0]
                                if sample_data is None:
                                    continue

                                if len(sample_data) == 4:
                                    x, y, meta, extra_feat = sample_data
                                else:
                                    x, y, meta = sample_data
                                    extra_feat = None

                                x = x.unsqueeze(0).to(
                                    device=device, memory_format=torch.channels_last
                                )
                                extra_tensor = None
                                if extra_feat is not None:
                                    extra_tensor = extra_feat.unsqueeze(0).to(
                                        device=device
                                    )

                                # Get prediction from view-specific model
                                with torch.autocast(
                                    device_type=device.type,
                                    enabled=args.amp and device.type in ["cuda", "mps"],
                                ):
                                    logits = view_models[view](x, extra_tensor)

                                sample_probs.append(torch.softmax(logits, dim=1))

                            if sample_probs:
                                view_probs[view] = torch.mean(
                                    torch.cat(sample_probs, dim=0),
                                    dim=0,
                                    keepdim=True,
                                )

                        # Combine predictions using ensemble
                        if len(view_probs) == len(required_views) and label is not None:
                            ensemble_probs = ensemble.predict(view_probs)
                            ensemble_pred = torch.argmax(ensemble_probs, dim=1)

                            all_y.append(label)
                            all_p.append(ensemble_pred.cpu().item())
                            all_prob.append(ensemble_probs.cpu().numpy())

            if len(all_y) > 0:
                from sklearn.metrics import (
                    accuracy_score,
                    confusion_matrix,
                    classification_report,
                    cohen_kappa_score,
                    roc_auc_score,
                )

                acc = accuracy_score(all_y, all_p)
                kappa = cohen_kappa_score(all_y, all_p, weights="quadratic")
                labels = list(range(num_classes))
                cm = confusion_matrix(all_y, all_p, labels=labels)
                report = classification_report(
                    all_y,
                    all_p,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )

                try:
                    all_prob_concat = np.concatenate(all_prob, axis=0)
                    auc = roc_auc_score(
                        all_y, all_prob_concat, multi_class="ovr", average="macro"
                    )
                except (ValueError, TypeError) as exc:
                    logger.debug("Nao foi possivel calcular AUC do ensemble: %s", exc)
                    auc = 0.0

                ensemble_metrics = {
                    "acc": float(acc),
                    "kappa_quadratic": float(kappa),
                    "auc_ovr": float(auc),
                    "num_samples": len(all_y),
                    "f1_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
                    "confusion_matrix": cm.tolist(),
                    "classification_report": report,
                }

                logger.info(
                    "Ensemble Metrics | Acc: %.4f | Kappa: %.4f | AUC: %.4f (n=%d)",
                    acc,
                    kappa,
                    auc,
                    len(all_y),
                )
            else:
                logger.warning(
                    "Nenhuma predicao de ensemble gerada; gerando metricas vazias."
                )
                ensemble_metrics = _empty_ensemble_metrics(num_classes)

            # Save ensemble metrics
            _write_ensemble_metrics(
                ensemble_metrics,
                outdir_path=outdir_path,
                export_formats=export_formats,
                logger=logger,
            )
