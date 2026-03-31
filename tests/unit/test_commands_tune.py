from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
try:
    from pydantic import ValidationError
except ModuleNotFoundError:
    from mammography.utils.pydantic_fallback import ValidationError

from mammography.commands import tune as tune_cmd


def test_parse_args_minimal_valid() -> None:
    """Test parsing with minimal valid arguments."""
    args = tune_cmd.parse_args([
        "--dataset", "archive",
    ])
    assert args.dataset == "archive"
    assert args.arch == "efficientnet_b0"
    assert args.classes == "multiclass"
    assert args.n_trials == 50
    assert args.epochs == 100
    assert args.img_size == 512
    assert args.seed == 42


def test_parse_args_custom_options() -> None:
    """Test parsing with custom options."""
    args = tune_cmd.parse_args([
        "--dataset", "mamografias",
        "--arch", "resnet50",
        "--classes", "binary",
        "--n-trials", "100",
        "--study-name", "my_study",
        "--epochs", "10",
        "--img-size", "224",
        "--seed", "123",
        "--outdir", "outputs/custom_tune",
        "--tune-config", "configs/custom_tune.yaml",
    ])
    assert args.dataset == "mamografias"
    assert args.arch == "resnet50"
    assert args.classes == "binary"
    assert args.n_trials == 100
    assert args.study_name == "my_study"
    assert args.epochs == 10
    assert args.img_size == 224
    assert args.seed == 123
    assert args.outdir == "outputs/custom_tune"
    assert args.tune_config == "configs/custom_tune.yaml"


def test_parse_args_arch_choices() -> None:
    """Test that arch argument only accepts valid choices."""
    valid_archs = ["efficientnet_b0", "resnet50"]
    for arch in valid_archs:
        args = tune_cmd.parse_args([
            "--dataset", "archive",
            "--arch", arch,
        ])
        assert args.arch == arch

    with pytest.raises(SystemExit):
        tune_cmd.parse_args([
            "--dataset", "archive",
            "--arch", "invalid_arch",
        ])


def test_parse_args_classes_choices() -> None:
    """Test that classes argument only accepts valid choices."""
    valid_classes = ["binary", "multiclass"]
    for classes in valid_classes:
        args = tune_cmd.parse_args([
            "--dataset", "archive",
            "--classes", classes,
        ])
        assert args.classes == classes

    with pytest.raises(SystemExit):
        tune_cmd.parse_args([
            "--dataset", "archive",
            "--classes", "invalid_task",
        ])


def test_parse_args_density_alias_warns_and_normalizes() -> None:
    """Test that the density alias emits a deprecation warning."""
    with pytest.warns(FutureWarning, match="density"):
        args = tune_cmd.parse_args([
            "--dataset", "archive",
            "--classes", "density",
        ])

    assert args.classes == "multiclass"


def test_parse_args_task_alias() -> None:
    """Test that --task works as alias for --classes."""
    args = tune_cmd.parse_args([
        "--dataset", "archive",
        "--task", "binary",
    ])
    assert args.classes == "binary"


def test_parse_args_pruner_options() -> None:
    """Test pruner-related arguments."""
    args = tune_cmd.parse_args([
        "--dataset", "archive",
        "--pruner-warmup-steps", "10",
        "--pruner-startup-trials", "5",
    ])
    assert args.pruner_warmup_steps == 10
    assert args.pruner_startup_trials == 5


def test_parse_args_storage_and_timeout() -> None:
    """Test storage and timeout options."""
    args = tune_cmd.parse_args([
        "--dataset", "archive",
        "--storage", "sqlite:///outputs/optuna.db",
        "--timeout", "3600",
    ])
    assert args.storage == "sqlite:///outputs/optuna.db"
    assert args.timeout == 3600


def test_parse_args_tracker_choices() -> None:
    """Test tracker argument validation."""
    valid_trackers = ["none", "local", "mlflow", "wandb"]
    for tracker in valid_trackers:
        args = tune_cmd.parse_args([
            "--dataset", "archive",
            "--tracker", tracker,
        ])
        assert args.tracker == tracker

    with pytest.raises(SystemExit):
        tune_cmd.parse_args([
            "--dataset", "archive",
            "--tracker", "invalid_tracker",
        ])


def test_parse_args_dry_run() -> None:
    """Test dry-run flag."""
    args = tune_cmd.parse_args([
        "--dataset", "archive",
        "--dry-run",
    ])
    assert args.dry_run is True


def test_parse_args_augmentation_options() -> None:
    """Test augmentation-related arguments."""
    args = tune_cmd.parse_args([
        "--dataset", "archive",
        "--augment",
        "--augment-vertical",
        "--augment-color",
        "--augment-rotation-deg", "10.0",
    ])
    assert args.augment is True
    assert args.augment_vertical is True
    assert args.augment_color is True
    assert args.augment_rotation_deg == 10.0


def test_get_label_mapper_density() -> None:
    """Test label mapper for density mode."""
    mapper = tune_cmd.get_label_mapper("density")
    assert mapper is None  # No mapping for density mode


def test_get_label_mapper_binary() -> None:
    """Test label mapper for binary mode."""
    mapper = tune_cmd.get_label_mapper("binary")
    assert mapper is not None
    # Test mapping: 1,2 -> 0; 3,4 -> 1
    assert mapper(1) == 0
    assert mapper(2) == 0
    assert mapper(3) == 1
    assert mapper(4) == 1


def test_get_label_mapper_multiclass() -> None:
    """Test label mapper for multiclass mode."""
    mapper = tune_cmd.get_label_mapper("multiclass")
    assert mapper is None  # Same as density


def test_get_label_mapper_none() -> None:
    """Test label mapper with None input."""
    mapper = tune_cmd.get_label_mapper(None)
    assert mapper is None


def test_resolve_loader_runtime_cuda() -> None:
    """Test loader runtime configuration for CUDA device."""
    args = MagicMock()
    args.num_workers = 4
    args.prefetch_factor = 2
    args.persistent_workers = True
    args.loader_heuristics = True
    device = torch.device("cuda")

    nw, prefetch, persistent = tune_cmd.resolve_loader_runtime(args, device)
    assert nw == 4
    assert prefetch == 2
    assert persistent is True


def test_resolve_loader_runtime_mps() -> None:
    """Test loader runtime configuration for MPS device."""
    args = MagicMock()
    args.num_workers = 4
    args.prefetch_factor = 2
    args.persistent_workers = True
    args.loader_heuristics = True
    device = torch.device("mps")

    nw, prefetch, persistent = tune_cmd.resolve_loader_runtime(args, device)
    assert nw == 0  # MPS forces num_workers=0
    assert prefetch == 2
    assert persistent is False


def test_resolve_loader_runtime_cpu() -> None:
    """Test loader runtime configuration for CPU device."""
    args = MagicMock()
    args.num_workers = 16
    args.prefetch_factor = 3
    args.persistent_workers = True
    args.loader_heuristics = True
    device = torch.device("cpu")

    nw, prefetch, persistent = tune_cmd.resolve_loader_runtime(args, device)
    # CPU should clamp to cpu_count
    import os
    expected_workers = min(16, os.cpu_count() or 0)
    assert nw == max(0, expected_workers)
    assert prefetch == 3
    assert persistent is True


def test_resolve_loader_runtime_no_heuristics() -> None:
    """Test loader runtime without heuristics enabled."""
    args = MagicMock()
    args.num_workers = 4
    args.prefetch_factor = 2
    args.persistent_workers = True
    args.loader_heuristics = False
    device = torch.device("mps")

    nw, prefetch, persistent = tune_cmd.resolve_loader_runtime(args, device)
    # Should return args as-is when heuristics disabled
    assert nw == 4
    assert prefetch == 2
    assert persistent is True


def test_resolve_optuna_db_path_sqlite_triple_slash() -> None:
    """Test resolving SQLite path with triple slash."""
    storage = "sqlite:///outputs/archive_tune/optuna.db"
    path = tune_cmd._resolve_optuna_db_path(storage)
    assert path == Path("outputs/archive_tune/optuna.db")


def test_resolve_optuna_db_path_sqlite_double_slash() -> None:
    """Test resolving SQLite path with double slash."""
    storage = "sqlite://outputs/optuna.db"
    path = tune_cmd._resolve_optuna_db_path(storage)
    assert path == Path("outputs/optuna.db")


def test_resolve_optuna_db_path_non_sqlite() -> None:
    """Test resolving non-SQLite storage returns None."""
    storage = "mysql://user:pass@localhost/db"
    path = tune_cmd._resolve_optuna_db_path(storage)
    assert path is None


def test_resolve_optuna_db_path_none() -> None:
    """Test resolving None storage returns None."""
    path = tune_cmd._resolve_optuna_db_path(None)
    assert path is None


def test_load_json_payload_valid(tmp_path: Path) -> None:
    """Test loading valid JSON payload."""
    json_path = tmp_path / "data.json"
    payload = {"key": "value", "number": 42}
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = tune_cmd._load_json_payload(json_path)
    assert loaded == payload


def test_load_json_payload_missing_file() -> None:
    """Test loading JSON from non-existent file returns None."""
    loaded = tune_cmd._load_json_payload(Path("nonexistent.json"))
    assert loaded is None


def test_load_json_payload_none_path() -> None:
    """Test loading JSON with None path returns None."""
    loaded = tune_cmd._load_json_payload(None)
    assert loaded is None


def test_load_json_payload_invalid_json(tmp_path: Path) -> None:
    """Test loading invalid JSON returns None."""
    json_path = tmp_path / "invalid.json"
    json_path.write_text("not valid json", encoding="utf-8")

    loaded = tune_cmd._load_json_payload(json_path)
    assert loaded is None


def test_load_json_payload_non_dict(tmp_path: Path) -> None:
    """Test loading JSON that's not a dict returns None."""
    json_path = tmp_path / "list.json"
    json_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    loaded = tune_cmd._load_json_payload(json_path)
    assert loaded is None


def test_coerce_int_valid() -> None:
    """Test coercing valid integer values."""
    assert tune_cmd._coerce_int(42) == 42
    assert tune_cmd._coerce_int("123") == 123
    assert tune_cmd._coerce_int(3.14) == 3


def test_coerce_int_invalid() -> None:
    """Test coercing invalid values returns None."""
    assert tune_cmd._coerce_int("not_a_number") is None
    assert tune_cmd._coerce_int(None) is None
    assert tune_cmd._coerce_int([1, 2, 3]) is None


def test_coerce_float_valid() -> None:
    """Test coercing valid float values."""
    assert tune_cmd._coerce_float(3.14) == 3.14
    assert tune_cmd._coerce_float("2.5") == 2.5
    assert tune_cmd._coerce_float(42) == 42.0


def test_coerce_float_invalid() -> None:
    """Test coercing invalid values returns None."""
    assert tune_cmd._coerce_float("not_a_number") is None
    assert tune_cmd._coerce_float(None) is None
    assert tune_cmd._coerce_float([1.0, 2.0]) is None


def test_main_dry_run_mode(tmp_path: Path, monkeypatch) -> None:
    """Test dry-run mode validates configuration without running optimization."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,1\nACC002,2\n",
        encoding="utf-8"
    )

    search_space_path = tmp_path / "tune_config.yaml"
    search_space_yaml = """
parameters:
  batch_size:
    type: int
    low: 8
    high: 32
    step: 8
  lr:
    type: float
    low: 0.0001
    high: 0.01
    log: true
"""
    search_space_path.write_text(search_space_yaml, encoding="utf-8")

    outdir = tmp_path / "outputs" / "tune"

    # Mock dependencies
    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())

    # Mock dataset loading
    import pandas as pd
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png", "img3.png", "img4.png"] * 10,
        "professional_label": [1, 2, 3, 4] * 10,
    })
    monkeypatch.setattr(tune_cmd, "load_dataset_dataframe", lambda *a, **k: df)
    monkeypatch.setattr(tune_cmd, "create_splits", lambda *a, **k: (df[:30], df[30:]))
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    # Mock MammoDensityDataset
    monkeypatch.setattr(tune_cmd, "MammoDensityDataset", lambda *a, **k: object())

    exit_code = tune_cmd.main([
        "--csv", str(csv_path),
        "--outdir", str(outdir),
        "--tune-config", str(search_space_path),
        "--n-trials", "10",
        "--epochs", "2",
        "--dry-run",
    ])

    assert exit_code == 0
    # Config should be saved even in dry-run
    assert (outdir / "results" / "tune_config.json").exists()


def test_main_missing_search_space_config(tmp_path: Path, monkeypatch) -> None:
    """Test error handling when search space config doesn't exist."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,1\n",
        encoding="utf-8"
    )

    outdir = tmp_path / "outputs" / "tune"

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    with pytest.raises(SystemExit, match="Search space config not found"):
        tune_cmd.main([
            "--csv", str(csv_path),
            "--outdir", str(outdir),
            "--tune-config", "nonexistent.yaml",
        ])


def test_main_invalid_search_space_config(tmp_path: Path, monkeypatch) -> None:
    """Test error handling for invalid search space configuration."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,1\n",
        encoding="utf-8"
    )

    search_space_path = tmp_path / "invalid_tune.yaml"
    # Invalid YAML - missing parameter type
    search_space_yaml = """
parameters:
  batch_size:
    low: 8
    high: 32
"""
    search_space_path.write_text(search_space_yaml, encoding="utf-8")

    outdir = tmp_path / "outputs" / "tune"

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    with pytest.raises(SystemExit, match="Failed to load search space config"):
        tune_cmd.main([
            "--csv", str(csv_path),
            "--outdir", str(outdir),
            "--tune-config", str(search_space_path),
        ])


def test_main_missing_csv_error(tmp_path: Path, monkeypatch) -> None:
    """Test error handling when CSV is not provided."""
    outdir = tmp_path / "outputs" / "tune"

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (None, None))

    # Create mock TrainConfig that returns None for csv
    class MockConfig:
        csv = None
        dicom_root = None

        @classmethod
        def from_args(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(tune_cmd, "TrainConfig", MockConfig)

    with pytest.raises(SystemExit, match="Informe --csv ou --dataset"):
        tune_cmd.main([
            "--outdir", str(outdir),
            "--arch", "efficientnet_b0",
        ])


def test_main_config_validation_error(tmp_path: Path, monkeypatch) -> None:
    """Test error handling for Pydantic validation errors."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text("AccessionNumber,Classification\nACC001,1\n", encoding="utf-8")

    outdir = tmp_path / "outputs" / "tune"

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    def _raise_validation_error(*args, **kwargs):
        raise ValidationError.from_exception_data("TrainConfig", [])

    monkeypatch.setattr(tune_cmd.TrainConfig, "from_args", _raise_validation_error)

    with pytest.raises(SystemExit, match="Config invalida"):
        tune_cmd.main([
            "--csv", str(csv_path),
            "--outdir", str(outdir),
        ])


def test_main_skips_existing_study_with_sufficient_trials(tmp_path: Path, monkeypatch) -> None:
    """Test that optimization is skipped when existing study has enough trials."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,1\nACC002,2\n",
        encoding="utf-8"
    )

    search_space_path = tmp_path / "tune_config.yaml"
    search_space_yaml = """
parameters:
  batch_size:
    type: int
    low: 8
    high: 32
"""
    search_space_path.write_text(search_space_yaml, encoding="utf-8")

    outdir = tmp_path / "outputs" / "tune"
    outdir.mkdir(parents=True, exist_ok=True)

    # Create existing best_params.json
    best_params_path = outdir / "best_params.json"
    best_params_payload = {
        "best_trial": 15,
        "best_value": 0.92,
        "best_params": {"batch_size": 16, "lr": 0.001},
        "n_trials": 50,
        "study_name": "test_study",
    }
    best_params_path.write_text(json.dumps(best_params_payload), encoding="utf-8")

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    import pandas as pd
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png"],
        "professional_label": [1, 2],
    })
    monkeypatch.setattr(tune_cmd, "load_dataset_dataframe", lambda *a, **k: df)

    # Mock study summary to return None (no Optuna study in storage)
    monkeypatch.setattr(tune_cmd, "load_study_summary", lambda *a, **k: None)

    registry_called = False

    def _fake_register(**kwargs):
        nonlocal registry_called
        registry_called = True
        return "run-123"

    monkeypatch.setattr(tune_cmd.tune_registry, "register_tune_run", _fake_register)

    exit_code = tune_cmd.main([
        "--csv", str(csv_path),
        "--outdir", str(outdir),
        "--tune-config", str(search_space_path),
        "--n-trials", "50",
    ])

    assert exit_code == 0
    assert registry_called  # Should register existing results


def test_main_registers_tune_run_after_optimization(tmp_path: Path, monkeypatch) -> None:
    """Test that tune run is registered after successful optimization."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,1\nACC002,2\nACC003,3\nACC004,4\n",
        encoding="utf-8"
    )

    search_space_path = tmp_path / "tune_config.yaml"
    search_space_yaml = """
parameters:
  batch_size:
    type: int
    low: 8
    high: 16
  lr:
    type: float
    low: 0.001
    high: 0.01
    log: true
"""
    search_space_path.write_text(search_space_yaml, encoding="utf-8")

    outdir = tmp_path / "outputs" / "tune"

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    import pandas as pd
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png", "img3.png", "img4.png"] * 5,
        "professional_label": [1, 2, 3, 4] * 5,
    })
    monkeypatch.setattr(tune_cmd, "load_dataset_dataframe", lambda *a, **k: df)
    monkeypatch.setattr(tune_cmd, "create_splits", lambda *a, **k: (df[:15], df[15:]))
    monkeypatch.setattr(tune_cmd, "MammoDensityDataset", lambda *a, **k: object())
    monkeypatch.setattr(tune_cmd, "load_study_summary", lambda *a, **k: None)

    # Mock OptunaTuner
    class MockTrial:
        def __init__(self, number, state, value, params):
            self.number = number
            self.state = MagicMock()
            self.state.name = state
            self.value = value
            self.params = params

    class MockStudy:
        def __init__(self):
            self.study_name = "test_study"
            self.trials = [
                MockTrial(0, "COMPLETE", 0.85, {"batch_size": 8, "lr": 0.001}),
                MockTrial(1, "COMPLETE", 0.90, {"batch_size": 16, "lr": 0.005}),
                MockTrial(2, "PRUNED", None, {}),
            ]
            self.best_trial = self.trials[1]
            self.best_value = 0.90
            self.best_params = {"batch_size": 16, "lr": 0.005}

    class MockOptunaTuner:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            return MockStudy()

    monkeypatch.setattr(tune_cmd, "OptunaTuner", MockOptunaTuner)

    captured_registry: dict = {}

    def _fake_register(**kwargs):
        captured_registry.update(kwargs)
        return "run-456"

    monkeypatch.setattr(tune_cmd.tune_registry, "register_tune_run", _fake_register)

    exit_code = tune_cmd.main([
        "--csv", str(csv_path),
        "--outdir", str(outdir),
        "--tune-config", str(search_space_path),
        "--n-trials", "3",
        "--epochs", "1",
        "--study-name", "integration_test_study",
    ])

    assert exit_code == 0
    assert captured_registry["run_name"] == "integration_test_study"
    assert captured_registry["arch"] == "efficientnet_b0"
    assert captured_registry["classes"] == "multiclass"
    assert captured_registry["study_name"] == "test_study"
    assert captured_registry["n_trials"] == 3
    assert captured_registry["completed_trials"] == 2
    assert captured_registry["pruned_trials"] == 1
    assert captured_registry["best_trial"] == 1
    assert captured_registry["best_value"] == 0.90
    assert captured_registry["best_params"] == {"batch_size": 16, "lr": 0.005}


def test_main_registry_error_handling(tmp_path: Path, monkeypatch, caplog) -> None:
    """Test that registry errors are logged but don't crash the command."""
    csv_path = tmp_path / "classificacao.csv"
    csv_path.write_text(
        "AccessionNumber,Classification\nACC001,1\nACC002,2\n",
        encoding="utf-8"
    )

    search_space_path = tmp_path / "tune_config.yaml"
    search_space_yaml = """
parameters:
  batch_size:
    type: int
    low: 8
    high: 16
"""
    search_space_path.write_text(search_space_yaml, encoding="utf-8")

    outdir = tmp_path / "outputs" / "tune"

    monkeypatch.setattr(tune_cmd, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(tune_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tune_cmd, "setup_logging", lambda *a, **k: tune_cmd.logging.getLogger())
    monkeypatch.setattr(tune_cmd, "resolve_paths_from_preset", lambda *a, **k: (str(csv_path), None))

    import pandas as pd
    df = pd.DataFrame({
        "path": ["img1.png", "img2.png"] * 5,
        "professional_label": [1, 2] * 5,
    })
    monkeypatch.setattr(tune_cmd, "load_dataset_dataframe", lambda *a, **k: df)
    monkeypatch.setattr(tune_cmd, "create_splits", lambda *a, **k: (df[:8], df[8:]))
    monkeypatch.setattr(tune_cmd, "MammoDensityDataset", lambda *a, **k: object())
    monkeypatch.setattr(tune_cmd, "load_study_summary", lambda *a, **k: None)

    class MockTrial:
        def __init__(self):
            self.number = 0
            self.state = MagicMock()
            self.state.name = "COMPLETE"
            self.value = 0.88
            self.params = {"batch_size": 8}

    class MockStudy:
        def __init__(self):
            self.study_name = "test_study"
            self.trials = [MockTrial()]
            self.best_trial = self.trials[0]
            self.best_value = 0.88
            self.best_params = {"batch_size": 8}

    class MockOptunaTuner:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            return MockStudy()

    monkeypatch.setattr(tune_cmd, "OptunaTuner", MockOptunaTuner)

    def _failing_register(**kwargs):
        raise RuntimeError("Registry database unavailable")

    monkeypatch.setattr(tune_cmd.tune_registry, "register_tune_run", _failing_register)

    exit_code = tune_cmd.main([
        "--csv", str(csv_path),
        "--outdir", str(outdir),
        "--tune-config", str(search_space_path),
        "--n-trials", "1",
        "--epochs", "1",
    ])

    # Should complete successfully despite registry error
    assert exit_code == 0
