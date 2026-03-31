from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mammography.commands import explain as explain_cmd


def test_parse_args_required_arguments() -> None:
    """Test that required arguments are validated."""
    with pytest.raises(SystemExit):
        explain_cmd.parse_args([])


def test_parse_args_minimal_valid() -> None:
    """Test parsing with minimal valid arguments."""
    args = explain_cmd.parse_args([
        "--model-path", "model.pth",
        "--images-dir", "data/",
    ])
    assert args.model_path == "model.pth"
    assert args.images_dir == "data/"
    assert args.method == "gradcam"
    assert args.model_type == "resnet50"
    assert args.num_classes == 4


def test_parse_args_custom_options() -> None:
    """Test parsing with custom options."""
    args = explain_cmd.parse_args([
        "--model-path", "vit_model.pth",
        "--images-dir", "images/",
        "--method", "attention",
        "--model-type", "vit",
        "--batch-size", "16",
        "--output-dir", "outputs/custom",
        "--alpha", "0.5",
        "--colormap", "viridis",
    ])
    assert args.model_path == "vit_model.pth"
    assert args.method == "attention"
    assert args.model_type == "vit"
    assert args.batch_size == 16
    assert args.output_dir == "outputs/custom"
    assert args.alpha == 0.5
    assert args.colormap == "viridis"


def test_parse_args_method_choices() -> None:
    """Test that method argument only accepts valid choices."""
    valid_methods = ["gradcam", "attention", "both"]
    for method in valid_methods:
        args = explain_cmd.parse_args([
            "--model-path", "model.pth",
            "--images-dir", "data/",
            "--method", method,
        ])
        assert args.method == method

    with pytest.raises(SystemExit):
        explain_cmd.parse_args([
            "--model-path", "model.pth",
            "--images-dir", "data/",
            "--method", "invalid",
        ])


def test_parse_args_registry_options() -> None:
    """Test registry-related arguments."""
    args = explain_cmd.parse_args([
        "--model-path", "model.pth",
        "--images-dir", "data/",
        "--run-name", "my_explain_run",
        "--no-mlflow",
        "--no-registry",
    ])
    assert args.run_name == "my_explain_run"
    assert args.no_mlflow is True
    assert args.no_registry is True


def test_discover_images_from_directory(tmp_path: Path) -> None:
    """Test image discovery from a directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "img1.png").write_text("fake", encoding="utf-8")
    (img_dir / "img2.jpg").write_text("fake", encoding="utf-8")
    (img_dir / "img3.dcm").write_text("fake", encoding="utf-8")

    images = explain_cmd.discover_images(str(img_dir))
    assert len(images) == 3
    assert str(img_dir / "img1.png") in images
    assert str(img_dir / "img2.jpg") in images
    assert str(img_dir / "img3.dcm") in images


def test_discover_images_from_csv(tmp_path: Path) -> None:
    """Test image discovery from CSV file."""
    csv_path = tmp_path / "images.csv"
    csv_path.write_text("image_path\nimg1.png\nimg2.jpg\n", encoding="utf-8")

    images = explain_cmd.discover_images(str(csv_path))
    assert len(images) == 2
    assert "img1.png" in images
    assert "img2.jpg" in images


def test_discover_images_from_csv_alternate_columns(tmp_path: Path) -> None:
    """Test CSV discovery with alternate column names."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("path,label\na.png,1\nb.png,2\n", encoding="utf-8")

    images = explain_cmd.discover_images(str(csv_path))
    assert len(images) == 2
    assert "a.png" in images
    assert "b.png" in images


def test_discover_images_from_csv_no_valid_column(tmp_path: Path) -> None:
    """Test CSV discovery raises error when no valid column found."""
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("label,density\n1,A\n2,B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No image path column found"):
        explain_cmd.discover_images(str(csv_path))


def test_discover_images_single_file(tmp_path: Path) -> None:
    """Test discovery of single image file."""
    img_path = tmp_path / "single.png"
    img_path.write_text("fake", encoding="utf-8")

    images = explain_cmd.discover_images(str(img_path))
    assert len(images) == 1
    assert str(img_path) in images


def test_load_model_with_model_state_dict(tmp_path: Path, monkeypatch) -> None:
    """Test loading model from checkpoint with model_state_dict key."""
    checkpoint_path = tmp_path / "model.pth"
    device = torch.device("cpu")

    class _DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes)
            self.num_classes = num_classes

    dummy_model = _DummyModel(4)
    state_dict = dummy_model.state_dict()

    checkpoint = {"model_state_dict": state_dict, "epoch": 10}

    def _fake_build_model(*, arch, num_classes, pretrained):
        return _DummyModel(num_classes)

    monkeypatch.setattr(explain_cmd, "build_model", _fake_build_model)
    monkeypatch.setattr(explain_cmd.torch, "load", lambda *a, **k: checkpoint)

    model = explain_cmd.load_model(
        str(checkpoint_path),
        "resnet50",
        4,
        device,
    )
    assert model is not None
    assert model.num_classes == 4


def test_load_model_with_state_dict(tmp_path: Path, monkeypatch) -> None:
    """Test loading model from checkpoint with state_dict key."""
    checkpoint_path = tmp_path / "model.pth"
    device = torch.device("cpu")

    class _DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes)
            self.num_classes = num_classes

    dummy_model = _DummyModel(4)
    state_dict = dummy_model.state_dict()

    checkpoint = {"state_dict": state_dict}

    def _fake_build_model(*, arch, num_classes, pretrained):
        return _DummyModel(num_classes)

    monkeypatch.setattr(explain_cmd, "build_model", _fake_build_model)
    monkeypatch.setattr(explain_cmd.torch, "load", lambda *a, **k: checkpoint)

    model = explain_cmd.load_model(
        str(checkpoint_path),
        "efficientnet_b0",
        4,
        device,
    )
    assert model is not None


def test_load_model_with_raw_state_dict(tmp_path: Path, monkeypatch) -> None:
    """Test loading model from raw state dict checkpoint."""
    checkpoint_path = tmp_path / "model.pth"
    device = torch.device("cpu")

    class _DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes)

    dummy_model = _DummyModel(4)
    state_dict = dummy_model.state_dict()

    def _fake_build_model(*, arch, num_classes, pretrained):
        return _DummyModel(num_classes)

    monkeypatch.setattr(explain_cmd, "build_model", _fake_build_model)
    monkeypatch.setattr(explain_cmd.torch, "load", lambda *a, **k: state_dict)

    model = explain_cmd.load_model(
        str(checkpoint_path),
        "resnet50",
        4,
        device,
    )
    assert model is not None


def test_get_target_layer_resnet() -> None:
    """Test target layer detection for ResNet."""
    class _MockResNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, 3),
                torch.nn.Conv2d(512, 512, 3),
            )

    model = _MockResNet()
    layer = explain_cmd.get_target_layer(model, "resnet50")
    assert layer == model.layer4[-1]


def test_get_target_layer_efficientnet() -> None:
    """Test target layer detection for EfficientNet."""
    class _MockEfficientNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3),
                torch.nn.Conv2d(32, 64, 3),
            )

    model = _MockEfficientNet()
    layer = explain_cmd.get_target_layer(model, "efficientnet_b0")
    assert layer == model.features[-1]


def test_get_target_layer_custom_layer() -> None:
    """Test custom layer resolution."""
    class _MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3),
            )

    model = _MockModel()
    layer = explain_cmd.get_target_layer(model, "resnet50", custom_layer="encoder.0")
    assert isinstance(layer, torch.nn.Conv2d)


def test_get_target_layer_invalid_custom_layer() -> None:
    """Test error handling for invalid custom layer."""
    class _MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential()

    model = _MockModel()
    with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
        explain_cmd.get_target_layer(model, "resnet50", custom_layer="nonexistent")


def test_get_target_layer_missing_layer4() -> None:
    """Test error handling for ResNet without layer4."""
    class _MockModel(torch.nn.Module):
        pass

    model = _MockModel()
    with pytest.raises(ValueError, match="ResNet model missing 'layer4'"):
        explain_cmd.get_target_layer(model, "resnet50")


def test_get_target_layer_missing_features() -> None:
    """Test error handling for EfficientNet without features."""
    class _MockModel(torch.nn.Module):
        pass

    model = _MockModel()
    with pytest.raises(ValueError, match="EfficientNet model missing 'features'"):
        explain_cmd.get_target_layer(model, "efficientnet_b0")


def test_get_target_layer_unknown_model_type() -> None:
    """Test error handling for unknown model type."""
    model = torch.nn.Module()
    with pytest.raises(ValueError, match="Unknown model type for GradCAM"):
        explain_cmd.get_target_layer(model, "unknown_model")


def test_main_generates_explanations_and_registers(tmp_path: Path, monkeypatch) -> None:
    """Test full explain workflow with GradCAM and registry integration."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "img1.png").write_text("fake", encoding="utf-8")
    (input_dir / "img2.png").write_text("fake", encoding="utf-8")

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("weights", encoding="utf-8")

    output_dir = tmp_path / "explanations"
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    # Mock dependencies
    monkeypatch.setattr(
        explain_cmd,
        "discover_images",
        lambda _path: [str(input_dir / "img1.png"), str(input_dir / "img2.png")],
    )

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3))
            self.fc = torch.nn.Linear(512, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 4), device=x.device)

    dummy_model = _DummyModel()
    monkeypatch.setattr(explain_cmd, "load_model", lambda *a, **k: dummy_model)
    monkeypatch.setattr(explain_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(explain_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(explain_cmd, "seed_everything", lambda _s: None)
    monkeypatch.setattr(explain_cmd, "setup_logging", lambda *a, **k: explain_cmd.logging.getLogger())

    # Mock image loading to return simple tensors
    def _fake_load_images(paths, img_size, logger):
        return [torch.rand(3, img_size, img_size) for _ in paths]

    monkeypatch.setattr(explain_cmd, "load_images_as_tensors", _fake_load_images)

    # Mock explainer
    class _FakeExplainer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def save_batch_overlays(self, images, heatmaps, output_dir, alpha, colormap, index_offset):
            return len(images)

    monkeypatch.setattr(explain_cmd, "GradCAMExplainer", _FakeExplainer)

    # Mock generate_explanations_batch
    def _fake_generate_explanations(images, model, explainer_type, target_classes, device, batch_size):
        return [torch.rand(1, 224, 224) for _ in images]

    monkeypatch.setattr(explain_cmd, "generate_explanations_batch", _fake_generate_explanations)

    # Capture registry call
    captured: dict[str, object] = {}

    def _fake_register(**kwargs):
        captured.update(kwargs)
        return "run-456"

    monkeypatch.setattr(
        explain_cmd.explain_registry,
        "register_explain_run",
        _fake_register,
    )

    # Run command
    exit_code = explain_cmd.main([
        "--model-path", str(checkpoint_path),
        "--images-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--model-type", "resnet50",
        "--method", "gradcam",
        "--batch-size", "2",
        "--run-name", "test_gradcam_run",
        "--registry-csv", str(registry_csv),
        "--registry-md", str(registry_md),
        "--no-mlflow",
    ])

    assert exit_code == 0
    assert output_dir.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "gradcam").exists()

    # Verify summary content
    with open(output_dir / "summary.json") as f:
        summary = json.load(f)
    assert summary["total_images"] == 2
    assert summary["loaded_images"] == 2
    assert "gradcam" in summary["explainers"]

    # Verify registry call
    assert captured["run_name"] == "test_gradcam_run"
    assert captured["checkpoint_path"] == checkpoint_path
    assert captured["arch"] == "resnet50"
    assert captured["method"] == "gradcam"
    metrics = captured["metrics"]
    assert metrics.total_images == 2
    assert metrics.loaded_images == 2


def test_main_error_model_not_found(tmp_path: Path, monkeypatch) -> None:
    """Test error handling when model file doesn't exist."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()

    monkeypatch.setattr(explain_cmd, "seed_everything", lambda _s: None)
    monkeypatch.setattr(explain_cmd, "setup_logging", lambda *a, **k: explain_cmd.logging.getLogger())

    exit_code = explain_cmd.main([
        "--model-path", "nonexistent.pth",
        "--images-dir", str(input_dir),
        "--quiet",
    ])

    assert exit_code == 1


def test_main_error_images_not_found(tmp_path: Path, monkeypatch) -> None:
    """Test error handling when images directory doesn't exist."""
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(explain_cmd, "seed_everything", lambda _s: None)
    monkeypatch.setattr(explain_cmd, "setup_logging", lambda *a, **k: explain_cmd.logging.getLogger())

    exit_code = explain_cmd.main([
        "--model-path", str(checkpoint_path),
        "--images-dir", "nonexistent/",
        "--quiet",
    ])

    assert exit_code == 1


def test_main_error_no_images_discovered(tmp_path: Path, monkeypatch) -> None:
    """Test error handling when no images are found."""
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(explain_cmd, "seed_everything", lambda _s: None)
    monkeypatch.setattr(explain_cmd, "setup_logging", lambda *a, **k: explain_cmd.logging.getLogger())
    monkeypatch.setattr(explain_cmd, "discover_images", lambda _path: [])

    exit_code = explain_cmd.main([
        "--model-path", str(checkpoint_path),
        "--images-dir", str(input_dir),
        "--quiet",
    ])

    assert exit_code == 1


def test_main_handles_model_loading_failure(tmp_path: Path, monkeypatch) -> None:
    """Test error handling for model loading failures."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(explain_cmd, "seed_everything", lambda _s: None)
    monkeypatch.setattr(explain_cmd, "setup_logging", lambda *a, **k: explain_cmd.logging.getLogger())
    monkeypatch.setattr(explain_cmd, "discover_images", lambda _path: ["img1.png"])

    def _failing_load_model(*args, **kwargs):
        raise RuntimeError("Invalid checkpoint format")

    monkeypatch.setattr(explain_cmd, "load_model", _failing_load_model)

    exit_code = explain_cmd.main([
        "--model-path", str(checkpoint_path),
        "--images-dir", str(input_dir),
        "--quiet",
    ])

    assert exit_code == 1


def test_main_no_registry_flag(tmp_path: Path, monkeypatch) -> None:
    """Test that --no-registry prevents registry updates."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_text("weights", encoding="utf-8")
    output_dir = tmp_path / "explanations"

    monkeypatch.setattr(explain_cmd, "seed_everything", lambda _s: None)
    monkeypatch.setattr(explain_cmd, "setup_logging", lambda *a, **k: explain_cmd.logging.getLogger())
    monkeypatch.setattr(explain_cmd, "discover_images", lambda _path: ["img1.png"])

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3))

    monkeypatch.setattr(explain_cmd, "load_model", lambda *a, **k: _DummyModel())
    monkeypatch.setattr(explain_cmd, "resolve_device", lambda _v: torch.device("cpu"))
    monkeypatch.setattr(explain_cmd, "configure_runtime", lambda *a, **k: None)
    monkeypatch.setattr(
        explain_cmd,
        "load_images_as_tensors",
        lambda paths, img_size, logger: [torch.rand(3, img_size, img_size)],
    )

    class _FakeExplainer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def save_batch_overlays(self, images, heatmaps, output_dir, alpha, colormap, index_offset):
            return len(images)

    monkeypatch.setattr(explain_cmd, "GradCAMExplainer", _FakeExplainer)
    monkeypatch.setattr(
        explain_cmd,
        "generate_explanations_batch",
        lambda images, model, explainer_type, target_classes, device, batch_size: [torch.rand(1, 224, 224)],
    )

    registry_called = False

    def _fake_register(**kwargs):
        nonlocal registry_called
        registry_called = True
        return "run-789"

    monkeypatch.setattr(
        explain_cmd.explain_registry,
        "register_explain_run",
        _fake_register,
    )

    exit_code = explain_cmd.main([
        "--model-path", str(checkpoint_path),
        "--images-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--no-registry",
        "--quiet",
    ])

    assert exit_code == 0
    assert not registry_called
