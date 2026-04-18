#
# cv_engine.py
# mammography-pipelines
#
# Cross-validation training engine with fold orchestration and metrics aggregation.
#
# Thales Matheus Mendonça Santos - February 2026
#
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from mammography.config import TrainConfig
from mammography.data.splits import create_kfold_splits
from mammography.training.engine import train_one_epoch, validate, save_atomic
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.data.csv_loader import load_dataset_dataframe, resolve_dataset_cache_mode, resolve_paths_from_preset
from mammography.models.nets import build_model
from mammography.utils.class_modes import get_label_mapper, get_num_classes
from mammography.utils.common import seed_everything, resolve_device
from mammography.utils.statistics import aggregate_cv_metrics


LOGGER = logging.getLogger("mammography")


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""
    fold_idx: int
    train_size: int
    val_size: int
    best_epoch: int
    best_val_acc: float
    best_val_kappa: float
    best_val_macro_f1: float
    best_val_auc: Optional[float]
    final_train_loss: float
    final_train_acc: float
    checkpoint_path: Path
    metrics_path: Path


class CrossValidationEngine:
    """K-fold cross-validation training engine.

    Orchestrates training across multiple folds using existing train_one_epoch/validate
    infrastructure. Saves per-fold checkpoints and aggregates metrics across folds.

    Args:
        config: TrainConfig with training parameters
        n_folds: Number of cross-validation folds (default 5)
        cv_seed: Random seed for fold splitting (default 42)
        save_all_folds: If True, save checkpoints for all folds. If False, only save best fold (default False)

    Example:
        >>> config = TrainConfig(csv="data.csv", epochs=10, batch_size=32)
        >>> cv_engine = CrossValidationEngine(config, n_folds=5)
        >>> results = cv_engine.run()
        >>> print(f"Mean accuracy: {results['mean_val_acc']:.3f} ± {results['std_val_acc']:.3f}")
    """

    def __init__(
        self,
        config: TrainConfig,
        n_folds: int = 5,
        cv_seed: int = 42,
        save_all_folds: bool = False,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds deve ser >= 2, recebido: {n_folds}")

        self.config = config
        self.n_folds = n_folds
        self.cv_seed = cv_seed
        self.save_all_folds = save_all_folds

        # Create output directory
        self.output_root = Path(config.outdir)
        self.output_root.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "CrossValidationEngine inicializado: n_folds=%d, cv_seed=%d, output=%s",
            n_folds,
            cv_seed,
            self.output_root,
        )

    def run(self) -> Dict[str, Any]:
        """Execute k-fold cross-validation training.

        Returns:
            Dictionary with aggregated results including mean, std, and per-fold metrics.
            Keys include: 'mean_val_acc', 'std_val_acc', 'mean_val_kappa', 'fold_results', etc.

        Raises:
            RuntimeError: If dataset loading fails or insufficient samples for k-fold
        """
        LOGGER.info("Iniciando cross-validation com %d folds...", self.n_folds)

        # Load full dataset
        full_df = self._load_dataset()
        if full_df is None or full_df.empty:
            raise RuntimeError("Dataset vazio ou invalido para cross-validation")

        LOGGER.info("Dataset carregado: %d amostras", len(full_df))

        # Create k-fold splits
        num_classes = get_num_classes(self.config.classes)
        folds = create_kfold_splits(
            full_df,
            n_splits=self.n_folds,
            seed=self.cv_seed,
            num_classes=num_classes,
        )

        if len(folds) != self.n_folds:
            raise RuntimeError(
                f"Esperado {self.n_folds} folds, obtido {len(folds)}. "
                "Verifique se ha amostras suficientes."
            )

        # Train each fold
        fold_results: List[FoldResult] = []
        for fold_idx, (train_df, val_df) in enumerate(folds):
            LOGGER.info("\n" + "=" * 80)
            LOGGER.info("FOLD %d/%d", fold_idx + 1, self.n_folds)
            LOGGER.info("=" * 80)

            fold_result = self._train_fold(fold_idx, train_df, val_df)
            fold_results.append(fold_result)

        # Aggregate results
        aggregated = self._aggregate_results(fold_results)

        # Save aggregated results
        self._save_aggregated_results(aggregated, fold_results)

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("CROSS-VALIDATION COMPLETO")
        LOGGER.info("=" * 80)
        LOGGER.info(
            "Accuracy: %.3f ± %.3f  [95%% CI: %.3f, %.3f]",
            aggregated["mean_val_acc"],
            aggregated["std_val_acc"],
            aggregated["ci_lower_val_acc"],
            aggregated["ci_upper_val_acc"],
        )
        LOGGER.info(
            "Kappa:    %.3f ± %.3f  [95%% CI: %.3f, %.3f]",
            aggregated["mean_val_kappa"],
            aggregated["std_val_kappa"],
            aggregated["ci_lower_val_kappa"],
            aggregated["ci_upper_val_kappa"],
        )
        LOGGER.info(
            "Macro-F1: %.3f ± %.3f  [95%% CI: %.3f, %.3f]",
            aggregated["mean_val_macro_f1"],
            aggregated["std_val_macro_f1"],
            aggregated["ci_lower_val_macro_f1"],
            aggregated["ci_upper_val_macro_f1"],
        )
        if aggregated["mean_val_auc"] is not None:
            LOGGER.info(
                "AUC:      %.3f ± %.3f  [95%% CI: %.3f, %.3f]",
                aggregated["mean_val_auc"],
                aggregated["std_val_auc"],
                aggregated["ci_lower_val_auc"],
                aggregated["ci_upper_val_auc"],
            )

        return aggregated

    def _load_dataset(self) -> Optional[pd.DataFrame]:
        """Load dataset DataFrame using config parameters.

        Returns:
            DataFrame with dataset samples, or None if loading fails
        """
        try:
            # Resolve paths from preset if dataset is specified
            if self.config.dataset:
                csv_path, dicom_root = resolve_paths_from_preset(
                    self.config.csv,
                    self.config.dataset,
                    self.config.dicom_root,
                )
                if csv_path and not self.config.csv:
                    self.config.csv = csv_path
                if dicom_root and not self.config.dicom_root:
                    self.config.dicom_root = dicom_root

            # Load dataset
            df = load_dataset_dataframe(
                csv_path=self.config.csv,
                dicom_root=self.config.dicom_root,
                exclude_class_5=not self.config.include_class_5,
                dataset=self.config.dataset,
            )
            if self.config.csv and "image_path" in df.columns:
                csv_dir = Path(str(self.config.csv)).expanduser().resolve().parent
                df = df.copy()

                def _resolve_image_path(value: Any) -> Any:
                    if pd.isna(value):
                        return value
                    path = Path(str(value))
                    return str(path if path.is_absolute() else csv_dir / path)

                df["image_path"] = df["image_path"].apply(_resolve_image_path)

            # Apply subset if specified
            if self.config.subset > 0 and len(df) > self.config.subset:
                LOGGER.info("Aplicando subset: %d amostras", self.config.subset)
                df = df.head(self.config.subset)

            return df
        except Exception as exc:
            LOGGER.error("Falha ao carregar dataset: %s", exc, exc_info=True)
            return None

    def _train_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> FoldResult:
        """Train model on a single fold.

        Args:
            fold_idx: Zero-indexed fold number
            train_df: Training DataFrame for this fold
            val_df: Validation DataFrame for this fold

        Returns:
            FoldResult with metrics and checkpoint paths
        """
        # Create fold-specific output directory
        fold_dir = self.output_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Fold %d: Train=%d | Val=%d", fold_idx, len(train_df), len(val_df))

        # Set reproducibility
        seed_everything(self.config.seed + fold_idx, deterministic=self.config.deterministic)

        # Get device
        device = resolve_device(self.config.device)

        # Create datasets
        num_classes = get_num_classes(self.config.classes)
        mapper = get_label_mapper(self.config.classes)
        train_rows = train_df.to_dict("records")
        val_rows = val_df.to_dict("records")
        cache_mode_train = resolve_dataset_cache_mode(self.config.cache_mode, train_rows)
        cache_mode_val = resolve_dataset_cache_mode(self.config.cache_mode, val_rows)
        cache_dir = self.config.cache_dir or str(fold_dir / "cache")

        train_dataset = MammoDensityDataset(
            rows=train_rows,
            img_size=self.config.img_size,
            train=True,
            augment=self.config.augment,
            augment_vertical=self.config.augment_vertical,
            augment_color=self.config.augment_color,
            rotation_deg=self.config.augment_rotation_deg,
            cache_mode=cache_mode_train,
            cache_dir=cache_dir,
            split_name="train",
            label_mapper=mapper,
            mean=self.config.mean,
            std=self.config.std,
        )

        val_dataset = MammoDensityDataset(
            rows=val_rows,
            img_size=self.config.img_size,
            train=False,
            augment=False,
            cache_mode=cache_mode_val,
            cache_dir=cache_dir,
            split_name="val",
            label_mapper=mapper,
            mean=self.config.mean,
            std=self.config.std,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=mammo_collate,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=mammo_collate,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

        # Create model
        model = build_model(
            arch=self.config.arch,
            num_classes=num_classes,
            pretrained=self.config.pretrained,
        ).to(device)

        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Create loss function
        loss_fn = nn.CrossEntropyLoss()

        # Create gradient scaler for AMP
        scaler = GradScaler(enabled=self.config.amp) if self.config.amp else None

        # Training loop
        best_val_acc = -float("inf")
        best_val_kappa = 0.0
        best_val_macro_f1 = 0.0
        best_val_auc = None
        best_epoch = 0
        final_train_loss = 0.0
        final_train_acc = 0.0

        for epoch in range(1, self.config.epochs + 1):
            LOGGER.info("Fold %d | Epoch %d/%d", fold_idx, epoch, self.config.epochs)

            # Train
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_fn=loss_fn,
                scaler=scaler,
                amp_enabled=self.config.amp,
            )

            final_train_loss = train_loss
            final_train_acc = train_acc

            # Validate
            val_metrics, _ = validate(
                model=model,
                loader=val_loader,
                device=device,
                amp_enabled=self.config.amp,
                loss_fn=loss_fn,
                collect_preds=False,
            )

            val_acc = val_metrics.get("accuracy", 0.0)
            val_kappa = val_metrics.get("kappa", 0.0)
            val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            val_auc = val_metrics.get("auc", None)

            LOGGER.info(
                "Fold %d | Epoch %d | Train Loss: %.4f | Train Acc: %.3f | Val Acc: %.3f | Val Kappa: %.3f",
                fold_idx,
                epoch,
                train_loss,
                train_acc,
                val_acc,
                val_kappa,
            )

            # Track best epoch
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_kappa = val_kappa
                best_val_macro_f1 = val_macro_f1
                best_val_auc = val_auc
                best_epoch = epoch

                # Save best model checkpoint if save_all_folds or this is the last fold
                if self.save_all_folds or fold_idx == self.n_folds - 1:
                    best_checkpoint_path = fold_dir / "best_model.pt"
                    state = {
                        "epoch": epoch,
                        "fold": fold_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_kappa": val_kappa,
                        "config": self.config.model_dump(),
                    }
                    save_atomic(state, best_checkpoint_path)
                    LOGGER.debug("Fold %d: Best model salvado em %s", fold_idx, best_checkpoint_path)

        # Save final checkpoint
        checkpoint_path = fold_dir / "checkpoint.pt"
        if self.save_all_folds or fold_idx == self.n_folds - 1:
            state = {
                "epoch": self.config.epochs,
                "fold": fold_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_kappa": best_val_kappa,
                "best_epoch": best_epoch,
                "config": self.config.model_dump(),
            }
            save_atomic(state, checkpoint_path)
            LOGGER.info("Fold %d: Checkpoint salvo em %s", fold_idx, checkpoint_path)

        # Save fold metrics
        metrics_path = fold_dir / "metrics.json"
        fold_metrics = {
            "fold_idx": fold_idx,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "best_epoch": best_epoch,
            "best_val_acc": float(best_val_acc),
            "best_val_kappa": float(best_val_kappa),
            "best_val_macro_f1": float(best_val_macro_f1),
            "best_val_auc": float(best_val_auc) if best_val_auc is not None else None,
            "final_train_loss": float(final_train_loss),
            "final_train_acc": float(final_train_acc),
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2, ensure_ascii=False)

        LOGGER.info("Fold %d: Metricas salvas em %s", fold_idx, metrics_path)

        return FoldResult(
            fold_idx=fold_idx,
            train_size=len(train_df),
            val_size=len(val_df),
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            best_val_kappa=best_val_kappa,
            best_val_macro_f1=best_val_macro_f1,
            best_val_auc=best_val_auc,
            final_train_loss=final_train_loss,
            final_train_acc=final_train_acc,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        )

    def _aggregate_results(self, fold_results: List[FoldResult]) -> Dict[str, Any]:
        """Aggregate metrics across all folds using statistical utilities.

        Args:
            fold_results: List of FoldResult objects

        Returns:
            Dictionary with aggregated statistics (mean, std, min, max, CI)
        """
        if not fold_results:
            raise ValueError("fold_results nao pode estar vazio")

        # Prepare fold metrics in format expected by aggregate_cv_metrics
        fold_metrics = []
        for fr in fold_results:
            metrics = {
                "val_acc": fr.best_val_acc,
                "val_kappa": fr.best_val_kappa,
                "val_macro_f1": fr.best_val_macro_f1,
            }
            # Only include AUC if available
            if fr.best_val_auc is not None:
                metrics["val_auc"] = fr.best_val_auc
            fold_metrics.append(metrics)

        # Use aggregate_cv_metrics to compute statistics with confidence intervals
        aggregated_stats = aggregate_cv_metrics(fold_metrics)

        # Transform to flat dictionary format for backward compatibility
        aggregated = {
            "n_folds": len(fold_results),
            # Accuracy
            "mean_val_acc": aggregated_stats["val_acc"]["mean"],
            "std_val_acc": aggregated_stats["val_acc"]["std"],
            "min_val_acc": aggregated_stats["val_acc"]["min"],
            "max_val_acc": aggregated_stats["val_acc"]["max"],
            "ci_lower_val_acc": aggregated_stats["val_acc"]["ci_lower"],
            "ci_upper_val_acc": aggregated_stats["val_acc"]["ci_upper"],
            # Kappa
            "mean_val_kappa": aggregated_stats["val_kappa"]["mean"],
            "std_val_kappa": aggregated_stats["val_kappa"]["std"],
            "min_val_kappa": aggregated_stats["val_kappa"]["min"],
            "max_val_kappa": aggregated_stats["val_kappa"]["max"],
            "ci_lower_val_kappa": aggregated_stats["val_kappa"]["ci_lower"],
            "ci_upper_val_kappa": aggregated_stats["val_kappa"]["ci_upper"],
            # Macro F1
            "mean_val_macro_f1": aggregated_stats["val_macro_f1"]["mean"],
            "std_val_macro_f1": aggregated_stats["val_macro_f1"]["std"],
            "min_val_macro_f1": aggregated_stats["val_macro_f1"]["min"],
            "max_val_macro_f1": aggregated_stats["val_macro_f1"]["max"],
            "ci_lower_val_macro_f1": aggregated_stats["val_macro_f1"]["ci_lower"],
            "ci_upper_val_macro_f1": aggregated_stats["val_macro_f1"]["ci_upper"],
        }

        # Add AUC statistics if available
        if "val_auc" in aggregated_stats:
            aggregated.update({
                "mean_val_auc": aggregated_stats["val_auc"]["mean"],
                "std_val_auc": aggregated_stats["val_auc"]["std"],
                "min_val_auc": aggregated_stats["val_auc"]["min"],
                "max_val_auc": aggregated_stats["val_auc"]["max"],
                "ci_lower_val_auc": aggregated_stats["val_auc"]["ci_lower"],
                "ci_upper_val_auc": aggregated_stats["val_auc"]["ci_upper"],
            })
        else:
            aggregated.update({
                "mean_val_auc": None,
                "std_val_auc": None,
                "min_val_auc": None,
                "max_val_auc": None,
                "ci_lower_val_auc": None,
                "ci_upper_val_auc": None,
            })

        # Add per-fold details
        aggregated["fold_results"] = [
            {
                "fold_idx": fr.fold_idx,
                "best_val_acc": fr.best_val_acc,
                "best_val_kappa": fr.best_val_kappa,
                "best_val_macro_f1": fr.best_val_macro_f1,
                "best_val_auc": fr.best_val_auc,
                "best_epoch": fr.best_epoch,
                "train_size": fr.train_size,
                "val_size": fr.val_size,
            }
            for fr in fold_results
        ]

        # Add detailed statistics structure for downstream use
        aggregated["detailed_stats"] = aggregated_stats

        return aggregated

    def _save_aggregated_results(
        self,
        aggregated: Dict[str, Any],
        fold_results: List[FoldResult],
    ) -> None:
        """Save aggregated cross-validation results to cv_summary.json.

        Args:
            aggregated: Aggregated statistics dictionary
            fold_results: List of FoldResult objects
        """
        summary_path = self.output_root / "cv_summary.json"
        detailed_stats = aggregated.get("detailed_stats", {})

        def _metric_summary(source_name: str) -> dict[str, Any]:
            stats = detailed_stats.get(source_name, {})
            return {
                key: stats.get(key)
                for key in ("mean", "std", "min", "max", "ci_lower", "ci_upper")
            }

        # Add metadata
        summary = {
            "n_folds": self.n_folds,
            "cv_seed": self.cv_seed,
            "config": self.config.model_dump(mode="json"),
            "results": aggregated,
            "aggregated_metrics": {
                "accuracy": _metric_summary("val_acc"),
                "kappa": _metric_summary("val_kappa"),
                "macro_f1": _metric_summary("val_macro_f1"),
            },
        }
        if "val_auc" in detailed_stats:
            summary["aggregated_metrics"]["auc"] = _metric_summary("val_auc")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        LOGGER.info("Resultados agregados salvos em %s", summary_path)
