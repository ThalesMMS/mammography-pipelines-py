from __future__ import annotations

from pathlib import Path

import torch

from mammography.commands import inference as inference_cmd


def test_inference_registers_run_with_metrics(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "patches_completo"
    input_dir.mkdir()
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_text("weights", encoding="utf-8")
    output_path = tmp_path / "preds.csv"

    monkeypatch.setattr(
        inference_cmd,
        "_iter_inputs",
        lambda _root: [str(input_dir / "a.png"), str(input_dir / "b.png")],
    )
    monkeypatch.setattr(inference_cmd, "MammoDensityDataset", lambda *a, **k: object())

    def _fake_dataloader(_dataset, batch_size, shuffle, collate_fn):
        imgs = torch.zeros((2, 3, 8, 8))
        metas = [{"path": "a.png"}, {"path": "b.png"}]
        return [(imgs, None, metas, None)]

    monkeypatch.setattr(inference_cmd, "DataLoader", _fake_dataloader)

    class _DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], self.num_classes), device=x.device)

    def _fake_build_model(*, arch, num_classes, train_backbone, unfreeze_last_block, pretrained):
        return _DummyModel(num_classes)

    monkeypatch.setattr(inference_cmd, "build_model", _fake_build_model)
    monkeypatch.setattr(inference_cmd.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(inference_cmd, "resolve_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(inference_cmd, "configure_runtime", lambda *a, **k: None)

    times = iter([1.0, 5.0])
    monkeypatch.setattr(inference_cmd.time, "perf_counter", lambda: next(times))

    captured: dict[str, object] = {}

    def _fake_register_inference_run(**kwargs):
        captured.update(kwargs)
        return "run-123"

    monkeypatch.setattr(
        inference_cmd.inference_registry,
        "register_inference_run",
        _fake_register_inference_run,
    )

    inference_cmd.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_dir),
            "--arch",
            "efficientnet_b0",
            "--classes",
            "density",
            "--output",
            str(output_path),
            "--run-name",
            "patches_inference_effnet",
            "--no-mlflow",
        ]
    )

    assert output_path.exists()
    assert captured["run_name"] == "patches_inference_effnet"
    assert captured["checkpoint_path"] == checkpoint_path

    metrics = captured["metrics"]
    assert metrics.total_images == 2
    assert metrics.duration_sec == 4.0
    assert metrics.images_per_sec == 0.5
