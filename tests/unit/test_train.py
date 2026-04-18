from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from torch import nn
from torch.utils.data import DataLoader

import mammography.commands.train as train_module
from mammography.commands.train import (
    _resolve_best_model_path,
    _summarize_metrics_for_summary,
    freeze_backbone,
    resolve_loader_runtime,
    unfreeze_last_block,
)
from mammography.commands.train_artifacts import _find_resume_checkpoint
from mammography.commands.train_resume import (
    _checkpoint_model,
    _resolve_view_resume_path,
)
from mammography.config import TrainConfig
from mammography.training.engine import train_one_epoch, validate


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:
        return self.net(x)


class _DummyTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _DummyTransformerBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [_DummyTransformerBlock(), _DummyTransformerBlock()]
        )


class _DummyVitWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _DummyTransformerBackbone()
        self.classifier = nn.Linear(4, 2)


def test_parse_args_train_config_maps_views_to_parser_namespace() -> None:
    config = TrainConfig(
        dataset="archive",
        view_specific_training=True,
        views_to_train=["CC", "MLO"],
    )

    args = train_module.parse_args(config)

    assert args.views == "CC,MLO"


def test_parse_args_train_config_applies_namespace_only_fields() -> None:
    config = TrainConfig(dataset="archive", auto_normalize=True)

    args = train_module.parse_args(config)

    assert args.auto_normalize is True


def test_parse_args_train_config_rejects_unmapped_explicit_fields() -> None:
    config = TrainConfig(dataset="archive")
    object.__setattr__(
        config,
        "model_dump",
        lambda mode="python": {"dataset": "archive", "unsupported_field": True},
    )
    object.__setattr__(
        config,
        "__pydantic_fields_set__",
        {"dataset", "unsupported_field"},
    )

    with pytest.raises(ValueError, match="unsupported_field"):
        train_module.parse_args(config)


def test_resolve_view_resume_path_uses_matching_sibling_checkpoint(
    tmp_path: Path,
) -> None:
    cc_dir = tmp_path / "results_CC"
    mlo_dir = tmp_path / "results_MLO"
    cc_dir.mkdir()
    mlo_dir.mkdir()
    cc_checkpoint = cc_dir / "checkpoint_cc.pt"
    mlo_checkpoint = mlo_dir / "checkpoint_mlo.pt"
    cc_checkpoint.write_text("cc", encoding="utf-8")
    mlo_checkpoint.write_text("mlo", encoding="utf-8")

    resolved = _resolve_view_resume_path(
        str(cc_checkpoint),
        current_view="MLO",
        checkpoint_name="checkpoint_mlo.pt",
        view_outdir_path=mlo_dir,
        views_to_train=["CC", "MLO"],
    )

    assert resolved == mlo_checkpoint


def test_resolve_view_resume_path_rejects_single_checkpoint_for_multiple_views(
    tmp_path: Path,
) -> None:
    cc_dir = tmp_path / "results_CC"
    mlo_dir = tmp_path / "results_MLO"
    cc_dir.mkdir()
    mlo_dir.mkdir()
    cc_checkpoint = cc_dir / "checkpoint_cc.pt"
    cc_checkpoint.write_text("cc", encoding="utf-8")

    with pytest.raises(SystemExit):
        _resolve_view_resume_path(
            str(cc_checkpoint),
            current_view="MLO",
            checkpoint_name="checkpoint_mlo.pt",
            view_outdir_path=mlo_dir,
            views_to_train=["CC", "MLO"],
        )


def test_find_resume_checkpoint_detects_view_specific_checkpoint(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results_CC"
    results_dir.mkdir()
    checkpoint = results_dir / "checkpoint_cc.pt"
    checkpoint.write_text("cc", encoding="utf-8")

    assert _find_resume_checkpoint(str(tmp_path)) == checkpoint


def test_checkpoint_model_unwraps_compiled_orig_mod() -> None:
    original = object()
    wrapper = SimpleNamespace(_orig_mod=original)

    assert _checkpoint_model(wrapper) is original


def _make_loader() -> DataLoader:
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 4, (8,))
    meta = [{"id": i} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    return DataLoader(dataset, batch_size=4, collate_fn=collate)


def test_train_one_epoch_and_validate() -> None:
    model = DummyModel()
    loader = _make_loader()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, acc = train_one_epoch(model, loader, optimizer, device)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0

    metrics, pred_rows = validate(model, loader, device)
    assert "acc" in metrics
    assert isinstance(metrics["acc"], float)
    assert isinstance(pred_rows, list)


def test_freeze_backbone_keeps_vit_classifier_trainable() -> None:
    model = _DummyVitWrapper()

    freeze_backbone(model, "vit_b_16")

    assert all(not param.requires_grad for param in model.backbone.parameters())
    assert all(param.requires_grad for param in model.classifier.parameters())


def test_unfreeze_last_block_supports_vit_architecture() -> None:
    model = _DummyVitWrapper()
    freeze_backbone(model, "vit_b_16")

    unfreeze_last_block(model, "vit_b_16")

    first_block_params = list(model.backbone.encoder.layers[0].parameters())
    last_block_params = list(model.backbone.encoder.layers[-1].parameters())
    assert all(not param.requires_grad for param in first_block_params)
    assert all(param.requires_grad for param in last_block_params)


def test_resolve_best_model_path_prefers_top_k_checkpoint(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    top_k_dir = results_dir / "top_k"
    top_k_dir.mkdir()
    best_top_k = top_k_dir / "model_epoch007_macro_f10.8123.pt"
    best_top_k.write_bytes(b"weights")
    (results_dir / "best_model.pt").write_bytes(b"fallback")

    resolved = _resolve_best_model_path(
        results_dir=results_dir,
        current_view=None,
        top_k=[{"score": 0.8123, "epoch": 7, "path": str(best_top_k)}],
        resume_path=None,
    )

    assert resolved == best_top_k


def test_summarize_metrics_for_summary_skips_non_finite_values() -> None:
    summary = _summarize_metrics_for_summary(
        {
            "acc": 0.91,
            "macro_f1": 0.89,
            "auc_ovr": float("nan"),
            "loss": float("inf"),
            "epoch": 12,
            "num_samples": 42,
        }
    )

    assert summary == {
        "acc": 0.91,
        "macro_f1": 0.89,
        "epoch": 12,
        "num_samples": 42,
    }


def test_resolve_loader_runtime_disables_windows_cuda_workers_on_python313(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        loader_heuristics=True,
    )
    monkeypatch.setattr(train_module.os, "name", "nt")
    monkeypatch.setattr(train_module.sys, "version_info", (3, 13, 3))

    num_workers, prefetch, persistent = resolve_loader_runtime(
        args,
        torch.device("cuda"),
    )

    assert num_workers == 0
    assert prefetch is None
    assert persistent is False


def test_resolve_loader_runtime_normalizes_zero_workers_without_heuristics() -> None:
    args = SimpleNamespace(
        num_workers=0,
        prefetch_factor=4,
        persistent_workers=True,
        loader_heuristics=False,
    )

    num_workers, prefetch, persistent = resolve_loader_runtime(
        args,
        torch.device("cpu"),
    )

    assert num_workers == 0
    assert prefetch is None
    assert persistent is False


def test_resolve_loader_runtime_normalizes_cpu_clamped_zero_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        loader_heuristics=True,
    )
    monkeypatch.setattr(train_module.os, "cpu_count", lambda: 0)

    num_workers, prefetch, persistent = resolve_loader_runtime(
        args,
        torch.device("cpu"),
    )

    assert num_workers == 0
    assert prefetch is None
    assert persistent is False


def test_resolve_loader_runtime_keeps_cuda_workers_outside_windows_python313(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        loader_heuristics=True,
    )
    monkeypatch.setattr(train_module.os, "name", "posix")
    monkeypatch.setattr(train_module.sys, "version_info", (3, 13, 3))

    num_workers, prefetch, persistent = resolve_loader_runtime(
        args,
        torch.device("cuda"),
    )

    assert num_workers == 4
    assert prefetch == 4
    assert persistent is True
