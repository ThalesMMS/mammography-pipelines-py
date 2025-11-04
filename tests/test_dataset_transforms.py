import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as legacy_transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RSNA_Mammo_ResNet50_Density import MammoDensityDataset


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _write_test_image(path: Path) -> None:
    arr = np.linspace(0, 255, num=256 * 256, dtype=np.uint8).reshape(256, 256)
    rgb = np.stack([arr, np.flipud(arr), np.rot90(arr)], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


def _legacy_pipeline(img_size: int, train: bool):
    ops = [
        legacy_transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        legacy_transforms.CenterCrop(img_size),
    ]
    if train:
        ops.extend([
            legacy_transforms.RandomHorizontalFlip(p=0.5),
            legacy_transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),
        ])
    ops.extend([
        legacy_transforms.ToTensor(),
        legacy_transforms.Normalize(MEAN, STD),
    ])
    return legacy_transforms.Compose(ops)


@pytest.mark.parametrize("train", [True, False])
def test_dataset_matches_legacy_pipeline(tmp_path, train):
    image_path = Path(tmp_path) / "sample.png"
    _write_test_image(image_path)
    row = {"image_path": str(image_path), "professional_label": 2}

    dataset = MammoDensityDataset([row], img_size=128, train=train, cache_mode="none")

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(1234)
    new_img, new_label, _ = dataset[0]

    torch.manual_seed(1234)
    with Image.open(image_path) as pil_img:
        pil_img = pil_img.convert("RGB")
        legacy_img = _legacy_pipeline(128, train=train)(pil_img)

    diff = (new_img - legacy_img).abs()
    if train:
        interior = diff[:, 8:-8, 8:-8]
        assert interior.max() < 5e-2
        assert diff.max() < 3.5
    else:
        assert torch.allclose(new_img, legacy_img, atol=2e-2)
    assert new_label == 1
    assert new_img.unsqueeze(0).is_contiguous(memory_format=torch.channels_last)


@pytest.mark.skipif(
    not (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())),
    reason="Neither CUDA nor MPS backends are available",
)
def test_transforms_run_equally_on_gpu(tmp_path):
    image_path = Path(tmp_path) / "sample.png"
    _write_test_image(image_path)
    row = {"image_path": str(image_path), "professional_label": 2}
    dataset = MammoDensityDataset([row], img_size=128, train=True, cache_mode="none")

    with Image.open(image_path) as pil_img:
        pil_img = pil_img.convert("RGB")
        base_tensor = dataset._convert_to_tensor(pil_img)

    cpu_device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    def reseed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    reseed(4321)
    cpu_out = dataset._to_channels_last(dataset._apply_transforms(base_tensor.to(device=cpu_device)))

    reseed(4321)
    device_tensor = dataset._to_channels_last(base_tensor.to(device=device))
    device_out = dataset._to_channels_last(dataset._apply_transforms(device_tensor)).to(cpu_device)

    assert torch.allclose(cpu_out, device_out, atol=1e-5)
