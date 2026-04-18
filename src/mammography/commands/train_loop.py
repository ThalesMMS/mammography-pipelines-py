# ruff: noqa
"""Per-view training loop for the density training command."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import profiler
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import WeightedRandomSampler

from mammography.commands.train_artifacts import (
    _clean_top_k,
    _load_checkpoint_for_eval,
    _plot_history_format,
    _resolve_best_model_path,
    _save_metrics_figure_format,
    _summarize_metrics_for_summary,
    _update_top_k,
    _write_metrics_artifacts,
    _write_split_artifacts,
    get_file_hash,
)
from mammography.commands.train_data import (
    PreparedTrainingData,
    _build_dataloader,
    resolve_loader_runtime,
)
from mammography.commands.train_modeling import (
    _parse_class_weights,
    build_param_groups,
    freeze_backbone,
    unfreeze_last_block,
)
from mammography.commands.train_resume import (
    _checkpoint_model,
    _resolve_view_resume_path,
)
from mammography.commands.train_tracking import _init_tracker
from mammography.data.csv_loader import resolve_dataset_cache_mode
from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.training.engine import (
    extract_embeddings,
    plot_history,
    save_atomic,
    save_metrics_figure,
    save_predictions,
    train_one_epoch,
    validate,
)
from mammography.utils.common import get_reproducibility_info


@dataclass(frozen=True)
class TrainingLoopResult:
    cache_mode_val: str
    nw: int
    prefetch: int | None
    persistent: bool


def run_training_loop(
    *,
    args,
    prepared: PreparedTrainingData,
    csv_path: str,
    dicom_root: str | None,
    outdir_root: Path,
    outdir_path: Path,
    metrics_dir: Path,
    export_formats: list[str],
    logger,
    device,
    killer,
) -> TrainingLoopResult:
    loop_result: TrainingLoopResult | None = None
    df = prepared.df
    train_df = prepared.train_df
    val_df = prepared.val_df
    test_df = prepared.test_df
    train_rows = prepared.train_rows
    val_rows = prepared.val_rows
    test_rows = prepared.test_rows
    mean = prepared.mean
    std = prepared.std
    num_classes = prepared.num_classes
    split_group_column = prepared.split_group_column
    views_to_train = prepared.views_to_train
    view_column = prepared.view_column
    embedding_store = prepared.embedding_store
    cache_dir = prepared.cache_dir
    mapper = prepared.mapper

    # Loop over views (or single iteration if view-specific training is disabled)
    for current_view in views_to_train:
        # Filter data by view if view-specific training is enabled
        if current_view is not None:
            view_train_df = train_df[train_df[view_column] == current_view].reset_index(
                drop=True
            )
            view_val_df = val_df[val_df[view_column] == current_view].reset_index(
                drop=True
            )
            view_test_df = (
                test_df[test_df[view_column] == current_view].reset_index(drop=True)
                if test_df is not None and view_column in test_df.columns
                else None
            )
            view_train_rows = view_train_df.to_dict("records")
            view_val_rows = view_val_df.to_dict("records")
            view_test_rows = (
                view_test_df.to_dict("records") if view_test_df is not None else []
            )
            logger.info(
                "Training view-specific model for view '%s': %d train samples, %d val samples, %d test samples",
                current_view,
                len(view_train_rows),
                len(view_val_rows),
                len(view_test_rows),
            )
            if not view_train_rows:
                logger.warning(
                    "Skipping view '%s' because it has no training samples.",
                    current_view,
                )
                continue
            # Create view-specific output directory
            view_outdir_path = outdir_path.parent / f"{outdir_path.name}_{current_view}"
            view_outdir_path.mkdir(parents=True, exist_ok=True)
            view_metrics_dir = view_outdir_path / "metrics"
            view_metrics_dir.mkdir(parents=True, exist_ok=True)
            # Create figures directory for view-specific training if export formats are specified
            if export_formats:
                view_figures_dir = view_outdir_path / "figures"
                view_figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use full dataset (no view filtering)
            view_train_df = train_df
            view_val_df = val_df
            view_test_df = test_df
            view_train_rows = train_rows
            view_val_rows = val_rows
            view_test_rows = test_rows if test_rows is not None else []
            view_outdir_path = outdir_path
            view_metrics_dir = metrics_dir

        _write_split_artifacts(
            train_df=view_train_df,
            val_df=view_val_df,
            test_df=view_test_df,
            outdir=view_outdir_path,
            split_mode=args.split_mode,
            group_column=split_group_column,
            logger=logger,
        )

        cache_mode_train = resolve_dataset_cache_mode(args.cache_mode, view_train_rows)
        cache_mode_val = resolve_dataset_cache_mode(args.cache_mode, view_val_rows)
        cache_mode_test = (
            resolve_dataset_cache_mode(args.cache_mode, view_test_rows)
            if view_test_rows
            else None
        )

        train_ds = MammoDensityDataset(
            view_train_rows,
            args.img_size,
            train=True,
            augment=args.augment,
            augment_vertical=args.augment_vertical,
            augment_color=args.augment_color,
            rotation_deg=args.augment_rotation_deg,
            cache_mode=cache_mode_train,
            cache_dir=cache_dir,
            split_name="train",
            label_mapper=mapper,
            embedding_store=embedding_store,
            mean=mean,
            std=std,
        )
        val_ds = MammoDensityDataset(
            view_val_rows,
            args.img_size,
            train=False,
            augment=False,
            cache_mode=cache_mode_val,
            cache_dir=cache_dir,
            split_name="val",
            label_mapper=mapper,
            embedding_store=embedding_store,
            mean=mean,
            std=std,
        )

        # Sampler/loader settings
        def _map_row_label(row):
            raw = row.get("professional_label")
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                return None
            val = int(raw)
            if mapper:
                return mapper(val)
            return val - 1

        sampler = None
        mapped_train_labels = None
        if args.sampler_weighted:
            mapped_train_labels = [_map_row_label(row) for row in view_train_rows]
            mapped = [label for label in mapped_train_labels if label is not None]
            counts = np.bincount(np.array(mapped, dtype=int), minlength=num_classes)
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
            class_weights = torch.tensor(
                inv ** float(args.sampler_alpha), dtype=torch.float
            )
            sample_weights = torch.tensor(
                [
                    class_weights[label] if label is not None else 0.0
                    for label in mapped_train_labels
                ],
                dtype=torch.float,
            )
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

        nw, prefetch, persistent = resolve_loader_runtime(args, device)
        dl_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": nw,
            "persistent_workers": bool(persistent and nw > 0),
            "pin_memory": device.type == "cuda",
            "collate_fn": mammo_collate,
        }
        if prefetch is not None and nw > 0:
            dl_kwargs["prefetch_factor"] = prefetch
        loop_result = TrainingLoopResult(
            cache_mode_val=cache_mode_val,
            nw=nw,
            prefetch=prefetch,
            persistent=persistent,
        )

        train_loader = _build_dataloader(
            train_ds,
            shuffle=sampler is None,
            sampler=sampler,
            dl_kwargs=dl_kwargs,
            logger=logger,
        )
        val_loader = _build_dataloader(
            val_ds,
            shuffle=False,
            sampler=None,
            dl_kwargs=dl_kwargs,
            logger=logger,
        )
        test_loader = None
        if view_test_rows:
            test_ds = MammoDensityDataset(
                view_test_rows,
                args.img_size,
                train=False,
                augment=False,
                cache_mode=cache_mode_test or "none",
                cache_dir=cache_dir,
                split_name="test",
                label_mapper=mapper,
                embedding_store=embedding_store,
                mean=mean,
                std=std,
            )
            test_loader = _build_dataloader(
                test_ds,
                shuffle=False,
                sampler=None,
                dl_kwargs=dl_kwargs,
                logger=logger,
            )

        # Model
        model = build_model(
            args.arch,
            num_classes=num_classes,
            train_backbone=args.train_backbone,
            unfreeze_last_block=args.unfreeze_last_block,
            pretrained=args.pretrained,
            extra_feature_dim=embedding_store.feature_dim if embedding_store else 0,
        ).to(device)

        if args.torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
                logger.info("torch.compile ativado.")
            except Exception as exc:
                logger.warning("torch.compile falhou; seguindo sem compile: %s", exc)
        checkpoint_model = _checkpoint_model(model)

        optim_params = build_param_groups(model, args.arch, args.lr, args.backbone_lr)
        optim_kwargs = {"weight_decay": args.weight_decay}
        if args.fused_optim and device.type == "cuda":
            optim_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(
            optim_params if optim_params else model.parameters(), **optim_kwargs
        )
        weights = None
        class_weights = _parse_class_weights(args.class_weights, num_classes)
        if class_weights == "auto":
            if mapped_train_labels is None:
                mapped_train_labels = [_map_row_label(row) for row in view_train_rows]
            mapped = [label for label in mapped_train_labels if label is not None]
            counts = np.bincount(np.array(mapped, dtype=int), minlength=num_classes)
            if np.any(counts == 0):
                logger.warning(
                    "class_weights=auto ignorado: alguma classe sem amostras."
                )
            else:
                total = float(np.sum(counts))
                weights_np = (total / counts) ** float(args.class_weights_alpha)
                weights_np = weights_np / np.mean(weights_np)
                weights = torch.tensor(weights_np, dtype=torch.float32).to(device)
                logger.info("Pesos de classe (auto): %s", weights_np.tolist())
        elif isinstance(class_weights, list):
            weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        scaler = GradScaler() if args.amp and device.type == "cuda" else None
        scheduler = None
        scheduler_mode = args.scheduler
        if scheduler_mode == "auto":
            scheduler_mode = (
                "plateau"
                if args.lr_reduce_patience and args.lr_reduce_patience > 0
                else "none"
            )
        if scheduler_mode == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.lr_reduce_factor,
                patience=args.lr_reduce_patience,
                min_lr=args.lr_reduce_min_lr,
                cooldown=args.lr_reduce_cooldown,
            )
        elif scheduler_mode == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs),
                eta_min=args.scheduler_min_lr,
            )
        elif scheduler_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, args.scheduler_step_size),
                gamma=args.scheduler_gamma,
            )

        best_acc = -1.0
        best_metric = -1.0
        best_epoch = -1
        patience_ctr = 0
        history = []
        top_k_metric = "macro_f1"
        top_k_limit = 3
        top_k_dir = view_outdir_path / "top_k"
        top_k: list[dict[str, Any]] = []
        gradcam_dir = view_outdir_path / "gradcam" if args.gradcam else None
        summary_path = view_outdir_path / "summary.json"
        # Add view suffix to checkpoint filename for view-specific training
        checkpoint_name = (
            f"checkpoint_{current_view.lower()}.pt" if current_view else "checkpoint.pt"
        )
        checkpoint_path = view_outdir_path / checkpoint_name
        resume_epoch = 1
        resume_path: Optional[Path] = None
        if args.resume_from:
            resume_path = _resolve_view_resume_path(
                args.resume_from,
                current_view=current_view,
                checkpoint_name=checkpoint_name,
                view_outdir_path=view_outdir_path,
                views_to_train=views_to_train,
            )
            logger.info("Retomando treino de %s", resume_path)
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                checkpoint_model.load_state_dict(checkpoint["model_state"])
                if "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                else:
                    logger.warning(
                        "Checkpoint sem optimizer_state; retomando apenas pesos."
                    )
                if scheduler is not None:
                    sched_state = checkpoint.get("scheduler_state")
                    if sched_state is not None:
                        scheduler.load_state_dict(sched_state)
                elif checkpoint.get("scheduler_state") is not None:
                    logger.warning(
                        "Checkpoint contem scheduler_state, mas nenhum scheduler ativo."
                    )
                if scaler is not None:
                    scaler_state = checkpoint.get("scaler_state")
                    if scaler_state is not None:
                        scaler.load_state_dict(scaler_state)
                elif checkpoint.get("scaler_state") is not None:
                    logger.warning(
                        "Checkpoint contem scaler_state, mas AMP desativado."
                    )
                best_acc = float(checkpoint.get("best_acc", best_acc))
                best_metric = float(
                    checkpoint.get(
                        "best_metric", checkpoint.get("best_acc", best_metric)
                    )
                )
                best_epoch = int(checkpoint.get("best_epoch", best_epoch))
                patience_ctr = int(checkpoint.get("patience_ctr", patience_ctr))
                resume_epoch = int(checkpoint.get("epoch", 0)) + 1
                top_k = _clean_top_k(checkpoint.get("top_k", []))
            else:
                checkpoint_model.load_state_dict(checkpoint)
                logger.warning(
                    "Checkpoint sem metadados; retomando apenas pesos do modelo."
                )
            if resume_epoch < 1:
                resume_epoch = 1
        if resume_epoch > args.epochs:
            logger.warning(
                "Checkpoint epoch %s >= epochs solicitadas (%s); nenhum epoch sera executado.",
                resume_epoch - 1,
                args.epochs,
            )
        repro_info = get_reproducibility_info()
        summary_payload = {
            "run_id": view_outdir_path.name,
            "seed": args.seed,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "reproducibility": repro_info,
            "arch": args.arch,
            "classes": args.classes,
            "dataset": args.dataset,
            "csv": str(csv_path),
            "dicom_root": str(dicom_root) if dicom_root else None,
            "data_hashes": {
                "csv": get_file_hash(str(csv_path)),
            },
            "embeddings_dir": args.embeddings_dir,
            "outdir": str(view_outdir_path),
            "outdir_root": str(outdir_root),
            "subset": args.subset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "pretrained": args.pretrained,
            # Optimizer hyperparameters
            "optimizer": "AdamW",
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "fused_optim": args.fused_optim,
            # Scheduler hyperparameters
            "scheduler": scheduler_mode,
            "lr_reduce_patience": args.lr_reduce_patience,
            "lr_reduce_factor": args.lr_reduce_factor,
            "lr_reduce_min_lr": args.lr_reduce_min_lr,
            "lr_reduce_cooldown": args.lr_reduce_cooldown,
            "scheduler_min_lr": args.scheduler_min_lr,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            # Augmentation settings
            "augment": args.augment,
            "augment_vertical": args.augment_vertical,
            "augment_color": args.augment_color,
            "augment_rotation_deg": args.augment_rotation_deg,
            # Model training settings
            "train_backbone": args.train_backbone,
            "unfreeze_last_block": args.unfreeze_last_block,
            "warmup_epochs": args.warmup_epochs,
            # Runtime settings
            "amp": args.amp,
            "deterministic": args.deterministic,
            "allow_tf32": args.allow_tf32,
            "torch_compile": args.torch_compile,
            # Data loading settings
            "cache_mode": args.cache_mode,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
            "loader_heuristics": args.loader_heuristics,
            # Normalization
            "mean": mean,
            "std": std,
            # Split settings
            "split_mode": args.split_mode,
            "split_group_column": split_group_column,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "split_ensure_all_classes": args.split_ensure_all_classes,
            "split_max_tries": args.split_max_tries,
            # Class weighting
            "class_weights": args.class_weights,
            "class_weights_alpha": args.class_weights_alpha,
            "sampler_weighted": args.sampler_weighted,
            "sampler_alpha": args.sampler_alpha,
            # Early stopping
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            # Resume settings
            "resume_from": str(resume_path) if resume_path else None,
            "resume_epoch": resume_epoch,
            # Tracker settings
            "tracker": args.tracker,
            "tracker_project": args.tracker_project,
            "tracker_run_name": args.tracker_run_name,
            "tracker_uri": args.tracker_uri,
            # Evaluation settings
            "save_val_preds": args.save_val_preds,
            "export_val_embeddings": args.export_val_embeddings,
            "top_k_metric": top_k_metric,
            "top_k_limit": top_k_limit,
            # View-specific training
            "view": current_view,
            "view_specific_training": args.view_specific_training,
            "view_column": view_column if args.view_specific_training else None,
            "views": args.views if args.view_specific_training else None,
            "ensemble_method": args.ensemble_method
            if args.view_specific_training
            else None,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        tracker = _init_tracker(args, summary_payload, view_outdir_path, logger)

        logger.info(
            "Starting training%s...",
            f" for view '{current_view}'" if current_view else "",
        )
        for epoch in range(resume_epoch, args.epochs + 1):
            if args.warmup_epochs and epoch <= args.warmup_epochs:
                freeze_backbone(model, args.arch)
            elif args.train_backbone:
                for p in model.parameters():
                    p.requires_grad = True
                if args.unfreeze_last_block:
                    unfreeze_last_block(model, args.arch)

            prof_ctx = None
            if args.profile and epoch == resume_epoch:
                Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
                activities = [profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(profiler.ProfilerActivity.CUDA)
                prof_ctx = profiler.profile(activities=activities, record_shapes=False)
            with prof_ctx if prof_ctx is not None else nullcontext():
                t_loss, t_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    loss_fn=loss_fn,
                    scaler=scaler,
                    amp_enabled=args.amp and device.type in ["cuda", "mps"],
                )

            val_metrics, pred_rows = validate(
                model,
                val_loader,
                device,
                amp_enabled=args.amp and device.type in ["cuda", "mps"],
                loss_fn=loss_fn,
                collect_preds=args.save_val_preds,
                gradcam=args.gradcam,
                gradcam_dir=gradcam_dir,
                gradcam_limit=args.gradcam_limit,
            )
            v_acc = float(val_metrics.get("acc", 0.0) or 0.0)
            v_loss = float(val_metrics.get("loss", 0.0) or 0.0)
            v_macro_f1 = float(val_metrics.get("macro_f1", 0.0) or 0.0)
            if not np.isfinite(v_macro_f1):
                v_macro_f1 = 0.0

            if prof_ctx is not None:
                try:
                    trace_path = Path(args.profile_dir) / "trace.json"
                    prof_ctx.export_chrome_trace(trace_path)
                    logger.info(f"Trace salvo em {trace_path}")
                except Exception as exc:
                    logger.warning("Falha ao salvar trace do profiler: %s", exc)

            logger.info(
                "Epoch %s/%s%s | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f Macro-F1: %.4f",
                epoch,
                args.epochs,
                f" (view={current_view})" if current_view else "",
                t_loss,
                t_acc,
                v_loss,
                v_acc,
                v_macro_f1,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": t_loss,
                    "train_acc": t_acc,
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "val_auc": val_metrics.get("auc_ovr", 0.0) or 0.0,
                    "val_kappa": val_metrics.get("kappa_quadratic", 0.0) or 0.0,
                    "val_macro_f1": val_metrics.get("macro_f1", 0.0) or 0.0,
                    "val_bal_acc": val_metrics.get("bal_acc", 0.0) or 0.0,
                    "val_bal_acc_adj": val_metrics.get("bal_acc_adj", 0.0) or 0.0,
                }
            )
            plot_history(history, view_outdir_path)

            # Export figures in publication formats if requested
            if export_formats:
                for fmt in export_formats:
                    _plot_history_format(
                        history, view_outdir_path / "figures", "train_history", fmt
                    )

            if tracker:
                # Comprehensive per-epoch metric logging
                metrics_payload = {
                    # Training metrics
                    "train_loss": float(t_loss),
                    "train_acc": float(t_acc),
                    # Validation metrics
                    "val_loss": float(v_loss),
                    "val_acc": float(v_acc),
                    "val_f1": float(v_macro_f1),
                    "val_kappa": float(val_metrics.get("kappa_quadratic", 0.0) or 0.0),
                    "balanced_acc": float(val_metrics.get("bal_acc", 0.0) or 0.0),
                    # Additional metrics (kept for backward compatibility)
                    "val_macro_f1": float(v_macro_f1),
                    "val_bal_acc": float(val_metrics.get("bal_acc", 0.0) or 0.0),
                    "val_bal_acc_adj": float(
                        val_metrics.get("bal_acc_adj", 0.0) or 0.0
                    ),
                }
                # Add AUC if available
                auc_val = val_metrics.get("auc_ovr", None)
                if auc_val is not None and np.isfinite(auc_val):
                    metrics_payload["val_auc"] = float(auc_val)
                    metrics_payload["val_auc_ovr"] = float(
                        auc_val
                    )  # Keep for backward compatibility
                tracker.log_metrics(metrics_payload, step=epoch)

            # Persist latest metrics
            val_metrics_filename = (
                f"val_metrics_{current_view.lower()}.json"
                if current_view
                else "val_metrics.json"
            )
            val_metrics_fig_name = (
                f"val_metrics_{current_view.lower()}.png"
                if current_view
                else "val_metrics.png"
            )
            with open(
                view_metrics_dir / val_metrics_filename, "w", encoding="utf-8"
            ) as f:
                json.dump(val_metrics, f, indent=2, default=str)
            save_metrics_figure(
                val_metrics, str(view_metrics_dir / val_metrics_fig_name)
            )

            # Export metrics figure in publication formats if requested
            if export_formats:
                base_name = (
                    f"val_metrics_{current_view.lower()}"
                    if current_view
                    else "val_metrics"
                )
                for fmt in export_formats:
                    _save_metrics_figure_format(
                        val_metrics,
                        str(view_outdir_path / "figures" / f"{base_name}.{fmt}"),
                    )

            if args.save_val_preds:
                save_predictions(pred_rows, view_outdir_path)

            if v_acc > best_acc:
                best_acc = v_acc

            top_k, new_entry = _update_top_k(
                top_k,
                v_macro_f1,
                epoch,
                checkpoint_model.state_dict(),
                top_k_dir,
                top_k_limit,
                top_k_metric,
            )
            if new_entry and tracker:
                tracker.log_artifact(
                    Path(new_entry["path"]), name=f"topk_{top_k_metric}_epoch{epoch}"
                )

            improved = v_macro_f1 > best_metric + args.early_stop_min_delta
            if improved:
                best_metric = v_macro_f1
                best_epoch = epoch
                # Add view suffix to best model filename for view-specific training
                best_model_name = (
                    f"best_model_{current_view.lower()}.pt"
                    if current_view
                    else "best_model.pt"
                )
                best_model_path = view_outdir_path / best_model_name
                save_atomic(checkpoint_model.state_dict(), best_model_path)
                if tracker:
                    tracker.log_artifact(
                        best_model_path, name=f"best_{top_k_metric}_epoch{epoch}"
                    )
                best_metrics_filename = (
                    f"best_metrics_{current_view.lower()}.json"
                    if current_view
                    else "best_metrics.json"
                )
                best_metrics_fig_name = (
                    f"best_metrics_{current_view.lower()}.png"
                    if current_view
                    else "best_metrics.png"
                )
                with open(
                    view_metrics_dir / best_metrics_filename, "w", encoding="utf-8"
                ) as f:
                    json.dump(val_metrics, f, indent=2, default=str)
                save_metrics_figure(
                    val_metrics, str(view_metrics_dir / best_metrics_fig_name)
                )

                # Export best metrics figure in publication formats if requested
                if export_formats:
                    base_name = (
                        f"best_metrics_{current_view.lower()}"
                        if current_view
                        else "best_metrics"
                    )
                    for fmt in export_formats:
                        _save_metrics_figure_format(
                            val_metrics,
                            str(view_outdir_path / "figures" / f"{base_name}.{fmt}"),
                        )

                patience_ctr = 0
            else:
                patience_ctr += 1

            if scheduler_mode == "plateau" and scheduler is not None:
                scheduler.step(v_macro_f1)
            elif scheduler is not None:
                scheduler.step()

            checkpoint_state = {
                "epoch": epoch,
                "model_state": checkpoint_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_acc": float(best_acc),
                "best_metric": float(best_metric),
                "best_metric_name": top_k_metric,
                "best_epoch": int(best_epoch),
                "patience_ctr": int(patience_ctr),
                "top_k": top_k,
                "mean": mean,
                "std": std,
            }
            save_atomic(checkpoint_state, checkpoint_path)

            if killer.kill_now:
                if tracker:
                    tracker.finish()
                print("[INFO] Estado salvo com sucesso. Encerrando execucao.")
                sys.exit(0)

            if args.early_stop_patience and patience_ctr >= args.early_stop_patience:
                logger.info(
                    f"Early stopping ativado após {patience_ctr} épocas sem melhoria."
                )
                break

        logger.info(
            "Done%s. Best %s: %.4f (epoch %s) | Best Acc: %.4f",
            f" for view '{current_view}'" if current_view else "",
            top_k_metric,
            best_metric,
            best_epoch,
            best_acc,
        )
        best_model_path = _resolve_best_model_path(
            results_dir=view_outdir_path,
            current_view=current_view,
            top_k=top_k,
            resume_path=resume_path,
        )
        if best_model_path is not None:
            logger.info(
                "Carregando melhor checkpoint%s para avaliacao final: %s",
                f" (view={current_view})" if current_view else "",
                best_model_path,
            )
            _load_checkpoint_for_eval(checkpoint_model, best_model_path, device)
        else:
            logger.warning(
                "Melhor checkpoint%s nao encontrado; avaliacao final usara o estado atual do modelo.",
                f" (view={current_view})" if current_view else "",
            )

        test_metrics_summary = None
        if test_loader is not None:
            logger.info(
                "Avaliando conjunto de teste%s (%d amostras)...",
                f" (view={current_view})" if current_view else "",
                len(view_test_rows),
            )
            test_metrics, test_pred_rows = validate(
                model,
                test_loader,
                device,
                amp_enabled=args.amp and device.type in ["cuda", "mps"],
                loss_fn=loss_fn,
                collect_preds=True,
                progress_label="Test",
            )
            test_metrics["epoch"] = int(best_epoch)
            test_metrics["num_samples"] = int(len(view_test_rows))
            _write_metrics_artifacts(
                metrics=test_metrics,
                metrics_dir=view_metrics_dir,
                outdir=view_outdir_path,
                split_name="test",
                current_view=current_view,
                export_formats=export_formats,
            )
            save_predictions(
                test_pred_rows, view_outdir_path, filename="test_predictions.csv"
            )
            test_metrics_summary = _summarize_metrics_for_summary(test_metrics)

            test_acc = float(test_metrics.get("acc", 0.0) or 0.0)
            test_macro_f1 = float(test_metrics.get("macro_f1", 0.0) or 0.0)
            test_auc_raw = test_metrics.get("auc_ovr", None)
            test_auc = (
                float(test_auc_raw)
                if isinstance(test_auc_raw, (int, float, np.integer, np.floating))
                and np.isfinite(test_auc_raw)
                else None
            )
            logger.info(
                "Test metrics%s | Acc: %.4f | Macro-F1: %.4f | AUC: %s",
                f" (view={current_view})" if current_view else "",
                test_acc,
                test_macro_f1,
                f"{test_auc:.4f}" if test_auc is not None else "n/a",
            )

            if tracker:
                test_payload = {
                    "test_loss": float(test_metrics.get("loss", 0.0) or 0.0),
                    "test_acc": test_acc,
                    "test_f1": test_macro_f1,
                    "test_kappa": float(
                        test_metrics.get("kappa_quadratic", 0.0) or 0.0
                    ),
                    "test_bal_acc": float(test_metrics.get("bal_acc", 0.0) or 0.0),
                    "test_bal_acc_adj": float(
                        test_metrics.get("bal_acc_adj", 0.0) or 0.0
                    ),
                }
                if test_auc is not None:
                    test_payload["test_auc"] = test_auc
                    test_payload["test_auc_ovr"] = test_auc
                tracker.log_metrics(test_payload, step=max(best_epoch, 0))
        elif args.test_frac > 0:
            logger.warning(
                "Split de teste solicitado%s, mas sem amostras disponiveis; pulando avaliacao final.",
                f" (view={current_view})" if current_view else "",
            )
        summary_payload.update(
            {
                "best_acc": float(best_acc),
                "val_acc": float(best_acc),
                "best_metric": float(best_metric),
                "best_metric_name": top_k_metric,
                "best_epoch": int(best_epoch),
                "top_k": top_k,
                "test_metrics": test_metrics_summary,
                "finished_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        if args.export_val_embeddings:
            logger.info(
                "Extraindo embeddings do conjunto de validação%s...",
                f" (view={current_view})" if current_view else "",
            )
            if best_model_path is not None:
                _load_checkpoint_for_eval(checkpoint_model, best_model_path, device)
            feats, metas = extract_embeddings(
                model,
                val_loader,
                device,
                amp_enabled=args.amp and device.type in ["cuda", "mps"],
            )
            np.save(view_outdir_path / "embeddings_val.npy", feats)
            pd.DataFrame(metas).to_csv(
                view_outdir_path / "embeddings_val.csv", index=False
            )

        if tracker:
            tracker.finish()

    if loop_result is None:
        cache_mode_val = resolve_dataset_cache_mode(args.cache_mode, val_rows)
        nw, prefetch, persistent = resolve_loader_runtime(args, device)
        loop_result = TrainingLoopResult(
            cache_mode_val=cache_mode_val,
            nw=nw,
            prefetch=prefetch,
            persistent=persistent,
        )
    return loop_result
