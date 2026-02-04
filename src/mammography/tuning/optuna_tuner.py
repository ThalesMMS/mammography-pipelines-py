#
# optuna_tuner.py
# mammography-pipelines
#
# Optuna integration for automated hyperparameter optimization with trial objective wrapper and pruning.
#
# Thales Matheus Mendonça Santos - January 2026
#
"""Optuna integration for automated hyperparameter tuning with MedianPruner."""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.trial import Trial, TrialState
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from mammography.config import HP
from mammography.data.dataset import MammoDensityDataset
from mammography.models.nets import build_model
from mammography.training.engine import train_one_epoch, validate
from mammography.tuning.search_space import SearchSpace, CategoricalParam, FloatParam, IntParam
from mammography.utils.common import seed_everything, resolve_device


class OptunaTuner:
    """
    Wraps the training pipeline in an Optuna objective function.

    Enables automated hyperparameter search with MedianPruner for efficient trial pruning.
    Best hyperparameters are automatically saved to JSON after optimization completes.

    Usage:
        tuner = OptunaTuner(
            search_space=SearchSpace.from_yaml("configs/tune.yaml"),
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            base_config=base_config,
            dataloader_kwargs=dl_kwargs
        )
        study = tuner.optimize(n_trials=50, study_name="density_tuning")
        best_params = study.best_params
    """

    def __init__(
        self,
        search_space: SearchSpace,
        train_dataset: MammoDensityDataset,
        val_dataset: MammoDensityDataset,
        device: torch.device,
        base_config: Dict[str, Any],
        num_classes: int = 4,
        outdir: str = "outputs/tune",
        amp_enabled: bool = False,
        arch: str = "efficientnet_b0",
        pretrained: bool = True,
        fixed_epochs: Optional[int] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OptunaTuner with training infrastructure.

        Args:
            search_space: SearchSpace defining hyperparameter search bounds
            train_dataset: MammoDensityDataset for training data
            val_dataset: MammoDensityDataset for validation data
            device: torch.device for training (cuda/mps/cpu)
            base_config: Base training configuration (dataset, augmentation, etc.)
            num_classes: Number of output classes (default: 4 for BI-RADS density)
            outdir: Output directory for saving results and checkpoints
            amp_enabled: Whether to use automatic mixed precision
            arch: Model architecture (efficientnet_b0 or resnet50)
            pretrained: Whether to use ImageNet pretrained weights
            fixed_epochs: If set, override search space epochs (useful for quick trials)
            dataloader_kwargs: Additional kwargs for DataLoader (num_workers, pin_memory, etc.)
        """
        self.search_space = search_space
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.base_config = base_config
        self.num_classes = num_classes
        self.outdir = Path(outdir)
        self.amp_enabled = amp_enabled
        self.arch = arch
        self.pretrained = pretrained
        self.fixed_epochs = fixed_epochs
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.logger = logging.getLogger("mammography.tuning")

        # Ensure output directory exists
        self.outdir.mkdir(parents=True, exist_ok=True)

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from the search space using Optuna trial.

        Args:
            trial: Optuna trial object for suggesting parameters

        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}
        for name, param_config in self.search_space.parameters.items():
            if isinstance(param_config, CategoricalParam):
                params[name] = trial.suggest_categorical(name, param_config.choices)
            elif isinstance(param_config, FloatParam):
                params[name] = trial.suggest_float(
                    name,
                    param_config.low,
                    param_config.high,
                    step=param_config.step,
                    log=param_config.log,
                )
            elif isinstance(param_config, IntParam):
                params[name] = trial.suggest_int(
                    name,
                    param_config.low,
                    param_config.high,
                    step=param_config.step,
                    log=param_config.log,
                )
        return params

    def _objective(self, trial: Trial) -> float:
        """
        Trial objective function that wraps the training loop.

        This is the core function called by Optuna for each trial. It:
        1. Samples hyperparameters from the search space
        2. Builds and trains a model with those hyperparameters
        3. Reports validation metrics for pruning decisions
        4. Returns the final validation accuracy for optimization

        Args:
            trial: Optuna trial object

        Returns:
            Validation accuracy (metric to maximize)
        """
        # Sample hyperparameters from search space
        hparams = self._suggest_hyperparameters(trial)
        self.logger.info(f"Trial {trial.number}: Testing hyperparameters: {hparams}")

        # Extract hyperparameters with defaults from base config
        lr = hparams.get("lr", self.base_config.get("lr", HP.LR))
        backbone_lr = hparams.get("backbone_lr", self.base_config.get("backbone_lr", HP.BACKBONE_LR))
        batch_size = hparams.get("batch_size", self.base_config.get("batch_size", HP.BATCH_SIZE))
        warmup_epochs = hparams.get("warmup_epochs", self.base_config.get("warmup_epochs", HP.WARMUP_EPOCHS))
        early_stop_patience = hparams.get("early_stop_patience", self.base_config.get("early_stop_patience", HP.EARLY_STOP_PATIENCE))
        unfreeze_last_block = hparams.get("unfreeze_last_block", self.base_config.get("unfreeze_last_block", HP.UNFREEZE_LAST_BLOCK))

        # Determine number of epochs (use fixed if provided, else from config)
        epochs = self.fixed_epochs if self.fixed_epochs else self.base_config.get("epochs", HP.EPOCHS)

        # Create DataLoaders with sampled batch_size
        loader_kwargs = self.dataloader_kwargs.copy()
        loader_kwargs["batch_size"] = batch_size

        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            **loader_kwargs
        )
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            **loader_kwargs
        )

        self.logger.debug(f"Trial {trial.number}: Created DataLoaders with batch_size={batch_size}")

        # Build model with sampled hyperparameters
        model = build_model(
            arch=self.arch,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            train_backbone=self.base_config.get("train_backbone", HP.TRAIN_BACKBONE),
            unfreeze_last_block=unfreeze_last_block,
        ).to(self.device)

        # Setup optimizer with differential learning rates
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "fc" in name or "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.base_config.get("weight_decay", 1e-4),
        )

        # Setup learning rate warmup scheduler
        warmup_scheduler = None
        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of target LR
                total_iters=warmup_epochs,
            )
            self.logger.debug(
                f"Trial {trial.number}: Warmup scheduler created for {warmup_epochs} epochs "
                f"(start_factor=0.1)"
            )

        # Setup loss function and AMP scaler
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler() if self.amp_enabled and self.device.type in ["cuda", "mps"] else None

        # Training loop with pruning
        best_val_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=self.device,
                loss_fn=loss_fn,
                scaler=scaler,
                amp_enabled=self.amp_enabled,
            )

            # Validation phase
            val_metrics, _ = validate(
                model=model,
                loader=val_loader,
                device=self.device,
                amp_enabled=self.amp_enabled,
                loss_fn=loss_fn,
                collect_preds=False,
            )

            val_acc = val_metrics.get("accuracy", 0.0)
            val_loss = val_metrics.get("loss", float("inf"))

            # Apply warmup scheduler for first N epochs
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr_head = optimizer.param_groups[0]["lr"]
                current_lr_backbone = optimizer.param_groups[1]["lr"]
                self.logger.debug(
                    f"Trial {trial.number} Epoch {epoch + 1}: "
                    f"Warmup LR - head={current_lr_head:.6f}, backbone={current_lr_backbone:.6f}"
                )

            # Report intermediate results for pruning
            trial.report(val_acc, epoch)

            # Check if trial should be pruned based on intermediate results
            if trial.should_prune():
                self.logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}/{epochs}")
                raise optuna.TrialPruned()

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping based on patience
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                self.logger.info(
                    f"Trial {trial.number} early stopped at epoch {epoch + 1}/{epochs} "
                    f"(patience={early_stop_patience})"
                )
                break

            self.logger.debug(
                f"Trial {trial.number} Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Return best validation accuracy as the objective value
        self.logger.info(f"Trial {trial.number} completed with best_val_acc={best_val_acc:.4f}")
        return best_val_acc

    def optimize(
        self,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        pruner_warmup_steps: int = 5,
        pruner_startup_trials: int = 3,
        direction: str = "maximize",
        timeout: Optional[int] = None,
    ) -> optuna.Study:
        """
        Run Optuna optimization to find the best hyperparameters.

        Args:
            n_trials: Number of trials to run
            study_name: Name for the Optuna study (default: auto-generated)
            storage: Database URL for persistent storage (default: in-memory)
            pruner_warmup_steps: Minimum epochs before pruning can occur
            pruner_startup_trials: Number of trials before median pruning starts
            direction: Optimization direction ("maximize" or "minimize")
            timeout: Maximum time in seconds for the study (default: no limit)

        Returns:
            Completed Optuna study object with results
        """
        if study_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            study_name = f"tune_{self.arch}_{timestamp}"

        # Create study with MedianPruner for efficient trial pruning
        pruner = MedianPruner(
            n_startup_trials=pruner_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
        )

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            pruner=pruner,
            load_if_exists=True,
        )

        self.logger.info(
            f"Starting Optuna optimization: {n_trials} trials, study={study_name}, "
            f"pruner=MedianPruner(warmup={pruner_warmup_steps}, startup={pruner_startup_trials})"
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Log results
        self.logger.info(f"Optimization complete! Best trial: {study.best_trial.number}")
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best validation accuracy: {study.best_value:.4f}")

        # Save best hyperparameters to JSON
        best_params_path = self.outdir / f"{study_name}_best_params.json"
        self._save_best_params(study, best_params_path)

        # Save study statistics
        stats_path = self.outdir / f"{study_name}_stats.json"
        self._save_study_stats(study, stats_path)

        return study

    def _save_best_params(self, study: optuna.Study, filepath: Path) -> None:
        """Save best hyperparameters to JSON file."""
        result = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "datetime": datetime.now(timezone.utc).isoformat(),
            "study_name": study.study_name,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        self.logger.info(f"Best hyperparameters saved to {filepath}")

    def _save_study_stats(self, study: optuna.Study, filepath: Path) -> None:
        """Save detailed study statistics to JSON file."""
        trials_data = []
        for trial in study.trials:
            trial_info = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "duration": trial.duration.total_seconds() if trial.duration else None,
            }
            trials_data.append(trial_info)

        stats = {
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "datetime": datetime.now(timezone.utc).isoformat(),
            "trials": trials_data,
            "pruned_trials": len([t for t in study.trials if t.state == TrialState.PRUNED]),
            "completed_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            "failed_trials": len([t for t in study.trials if t.state == TrialState.FAIL]),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Study statistics saved to {filepath}")
