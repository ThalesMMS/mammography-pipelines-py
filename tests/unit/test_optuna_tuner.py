#
# test_optuna_tuner.py
# mammography-pipelines
#
# Tests for Optuna integration and hyperparameter tuning.
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from optuna.trial import Trial

from mammography.data.dataset import MammoDensityDataset
from mammography.tuning.optuna_tuner import OptunaTuner
from mammography.tuning.search_space import SearchSpace, CategoricalParam, FloatParam


class TestOptunaTunerAugmentation:
    """Test that OptunaTuner correctly applies sampled augmentation parameters to dataset."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock MammoDensityDataset for testing."""
        dataset = Mock(spec=MammoDensityDataset)
        dataset.train = True
        dataset.augment = True
        dataset.augment_vertical = False
        dataset.augment_color = False
        dataset.rotation_deg = 5.0
        dataset.__len__ = Mock(return_value=10)
        return dataset

    @pytest.fixture
    def search_space_with_augmentation(self):
        """Create a search space with augmentation parameters."""
        return SearchSpace(
            parameters={
                "augment": CategoricalParam(choices=[True, False]),
                "augment_vertical": CategoricalParam(choices=[True, False]),
                "augment_color": CategoricalParam(choices=[True, False]),
                "augment_rotation_deg": FloatParam(low=0.0, high=15.0, step=2.5),
            }
        )

    @pytest.fixture
    def tuner_with_augmentation(self, mock_dataset, search_space_with_augmentation):
        """Create an OptunaTuner with augmentation search space."""
        device = torch.device("cpu")
        base_config = {"epochs": 1, "lr": 1e-4, "batch_size": 8}
        return OptunaTuner(
            search_space=search_space_with_augmentation,
            train_dataset=mock_dataset,
            val_dataset=mock_dataset,
            device=device,
            base_config=base_config,
            num_classes=4,
            outdir="outputs/test_tune",
            amp_enabled=False,
            dataloader_kwargs={"num_workers": 0},
        )

    def test_augmentation_params_extracted_from_trial(self, tuner_with_augmentation):
        """Test that augmentation parameters are correctly extracted from Optuna trial."""
        # Create mock trial that suggests augmentation parameters
        mock_trial = Mock(spec=Trial)
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: {
            "augment": True,
            "augment_vertical": True,
            "augment_color": False,
        }.get(name, choices[0]))
        mock_trial.suggest_float = Mock(return_value=10.0)

        # Extract hyperparameters using the tuner's method
        hparams = tuner_with_augmentation._suggest_hyperparameters(mock_trial)

        # Verify augmentation parameters were extracted
        assert "augment" in hparams
        assert "augment_vertical" in hparams
        assert "augment_color" in hparams
        assert "augment_rotation_deg" in hparams

    def test_augmentation_params_applied_to_dataset(self, tuner_with_augmentation, mock_dataset):
        """Test that sampled augmentation parameters are applied to train_dataset."""
        # Create mock trial with specific augmentation values
        mock_trial = Mock(spec=Trial)
        mock_trial.number = 0
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: {
            "augment": True,
            "augment_vertical": True,
            "augment_color": True,
        }.get(name, choices[0]))
        mock_trial.suggest_float = Mock(return_value=12.5)
        mock_trial.report = Mock()
        mock_trial.should_prune = Mock(return_value=True)  # Prune early to speed up test

        # Mock the build_model, optimizer, loss, and training functions
        with patch("mammography.tuning.optuna_tuner.build_model") as mock_build_model, \
             patch("mammography.tuning.optuna_tuner.train_one_epoch") as mock_train, \
             patch("mammography.tuning.optuna_tuner.validate") as mock_validate:

            # Setup mocks
            mock_model = Mock()
            mock_model.named_parameters = Mock(return_value=[])
            mock_model.to = Mock(return_value=mock_model)
            mock_build_model.return_value = mock_model

            mock_train.return_value = (0.5, 0.7)
            mock_validate.return_value = ({"accuracy": 0.8, "loss": 0.3}, None)

            # Run objective (should prune after first epoch)
            try:
                tuner_with_augmentation._objective(mock_trial)
            except Exception:
                pass  # Expected to raise TrialPruned

            # Verify augmentation parameters were applied to train_dataset
            assert mock_dataset.augment is True
            assert mock_dataset.augment_vertical is True
            assert mock_dataset.augment_color is True
            assert mock_dataset.rotation_deg == 12.5

    def test_augmentation_rotation_deg_parameter_mapping(self, tuner_with_augmentation, mock_dataset):
        """Test that 'augment_rotation_deg' from search space is mapped to 'rotation_deg' in dataset."""
        # Create mock trial with specific rotation value
        mock_trial = Mock(spec=Trial)
        mock_trial.number = 0
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: {
            "augment": True,
            "augment_vertical": False,
            "augment_color": False,
        }.get(name, choices[0]))
        mock_trial.suggest_float = Mock(return_value=7.5)
        mock_trial.report = Mock()
        mock_trial.should_prune = Mock(return_value=True)

        # Mock the build_model, optimizer, loss, and training functions
        with patch("mammography.tuning.optuna_tuner.build_model") as mock_build_model, \
             patch("mammography.tuning.optuna_tuner.train_one_epoch") as mock_train, \
             patch("mammography.tuning.optuna_tuner.validate") as mock_validate:

            # Setup mocks
            mock_model = Mock()
            mock_model.named_parameters = Mock(return_value=[])
            mock_model.to = Mock(return_value=mock_model)
            mock_build_model.return_value = mock_model

            mock_train.return_value = (0.5, 0.7)
            mock_validate.return_value = ({"accuracy": 0.8, "loss": 0.3}, None)

            # Run objective (should prune after first epoch)
            try:
                tuner_with_augmentation._objective(mock_trial)
            except Exception:
                pass

            # Verify rotation_deg was correctly mapped from augment_rotation_deg
            assert mock_dataset.rotation_deg == 7.5

    def test_augmentation_disabled_when_augment_false(self, tuner_with_augmentation, mock_dataset):
        """Test that augment is disabled when sampled as False."""
        # Create mock trial with augment=False
        mock_trial = Mock(spec=Trial)
        mock_trial.number = 0
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: {
            "augment": False,
            "augment_vertical": True,
            "augment_color": True,
        }.get(name, choices[0]))
        mock_trial.suggest_float = Mock(return_value=10.0)
        mock_trial.report = Mock()
        mock_trial.should_prune = Mock(return_value=True)

        # Mock the build_model, optimizer, loss, and training functions
        with patch("mammography.tuning.optuna_tuner.build_model") as mock_build_model, \
             patch("mammography.tuning.optuna_tuner.train_one_epoch") as mock_train, \
             patch("mammography.tuning.optuna_tuner.validate") as mock_validate:

            # Setup mocks
            mock_model = Mock()
            mock_model.named_parameters = Mock(return_value=[])
            mock_model.to = Mock(return_value=mock_model)
            mock_build_model.return_value = mock_model

            mock_train.return_value = (0.5, 0.7)
            mock_validate.return_value = ({"accuracy": 0.8, "loss": 0.3}, None)

            # Run objective (should prune after first epoch)
            try:
                tuner_with_augmentation._objective(mock_trial)
            except Exception:
                pass

            # Verify augment is disabled (False AND train flag)
            assert mock_dataset.augment is False

    def test_augmentation_defaults_when_not_in_search_space(self):
        """Test that augmentation falls back to defaults when not in search space."""
        # Create search space WITHOUT augmentation parameters
        search_space = SearchSpace(
            parameters={
                "lr": FloatParam(low=1e-5, high=1e-3, log=True),
                "batch_size": CategoricalParam(choices=[8, 16, 32]),
            }
        )

        # Create mock dataset
        mock_dataset = Mock(spec=MammoDensityDataset)
        mock_dataset.train = True
        mock_dataset.augment = False
        mock_dataset.augment_vertical = False
        mock_dataset.augment_color = False
        mock_dataset.rotation_deg = 0.0
        mock_dataset.__len__ = Mock(return_value=10)

        device = torch.device("cpu")
        base_config = {
            "epochs": 1,
            "lr": 1e-4,
            "batch_size": 8,
            "augment": True,
            "augment_vertical": True,
            "augment_color": False,
            "rotation_deg": 8.0,
        }

        tuner = OptunaTuner(
            search_space=search_space,
            train_dataset=mock_dataset,
            val_dataset=mock_dataset,
            device=device,
            base_config=base_config,
            num_classes=4,
            outdir="outputs/test_tune",
            amp_enabled=False,
            dataloader_kwargs={"num_workers": 0},
        )

        # Create mock trial
        mock_trial = Mock(spec=Trial)
        mock_trial.number = 0
        mock_trial.suggest_categorical = Mock(return_value=16)
        mock_trial.suggest_float = Mock(return_value=5e-4)
        mock_trial.report = Mock()
        mock_trial.should_prune = Mock(return_value=True)

        # Mock the build_model, optimizer, loss, and training functions
        with patch("mammography.tuning.optuna_tuner.build_model") as mock_build_model, \
             patch("mammography.tuning.optuna_tuner.train_one_epoch") as mock_train, \
             patch("mammography.tuning.optuna_tuner.validate") as mock_validate:

            # Setup mocks
            mock_model = Mock()
            mock_model.named_parameters = Mock(return_value=[])
            mock_model.to = Mock(return_value=mock_model)
            mock_build_model.return_value = mock_model

            mock_train.return_value = (0.5, 0.7)
            mock_validate.return_value = ({"accuracy": 0.8, "loss": 0.3}, None)

            # Run objective (should prune after first epoch)
            try:
                tuner._objective(mock_trial)
            except Exception:
                pass

            # Verify defaults from base_config were used
            assert mock_dataset.augment is True
            assert mock_dataset.augment_vertical is True
            assert mock_dataset.augment_color is False
            assert mock_dataset.rotation_deg == 8.0
