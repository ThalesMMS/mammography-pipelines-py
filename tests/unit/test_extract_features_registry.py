import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mammography.commands import extract_features


def _write_embedding_artifacts(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "features.npy", np.zeros((2, 4), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "accession": "000001",
                "raw_label": 1,
                "image_path": "dummy.png",
            }
        ]
    ).to_csv(outdir / "metadata.csv", index=False)
    (outdir / "pca_label.png").write_text("dummy", encoding="utf-8")
    (outdir / "clustering_metrics.json").write_text("{}", encoding="utf-8")


def test_sanitize_json_metrics_replaces_non_finite_values() -> None:
    metrics = extract_features._sanitize_json_metrics(
        {"k": np.int64(3), "silhouette": np.float64("-inf"), "davies_bouldin": np.inf}
    )

    assert metrics == {"k": 3, "silhouette": None, "davies_bouldin": None}


def test_run_reduction_respects_explicit_no_flags() -> None:
    args = argparse.Namespace(run_reduction=True, pca=False, tsne=None, umap=True)

    extract_features._resolve_reduction_flags(args)

    assert args.pca is False
    assert args.tsne is True
    assert args.umap is True


def test_register_embedding_run_defaults_run_name(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "mamografias_embeddings"
    _write_embedding_artifacts(outdir)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    args = argparse.Namespace(
        arch="resnet50",
        layer_name="avgpool",
        batch_size=32,
        img_size=512,
        run_name="",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri="",
        experiment="",
        no_mlflow=True,
    )

    run_name, run_id = extract_features._register_embedding_run(
        outdir=outdir,
        args=args,
        dataset_name="mamografias",
        command="mammography embed --dataset mamografias",
    )

    assert run_name == "mamografias_embed_resnet50"
    assert run_id is None
    assert registry_csv.exists()
    assert registry_md.exists()

    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_name"] == "mamografias_embed_resnet50"
    assert rows[0]["dataset"] == "mamografias"
    assert "pca_label.png" in rows[0]["reduction_outputs"]
    assert "clustering_metrics.json" in rows[0]["clustering_outputs"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "mamografias_embed_resnet50" in md_text


class _DummyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict, None]:
        return torch.zeros(1), 0, {}, None


def test_extract_embeddings_fallbacks_to_single_worker(monkeypatch) -> None:
    calls: list[int] = []

    def _fake_extract_embeddings(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        amp_enabled: bool,
        layer_name: str,
    ) -> tuple[np.ndarray, list[dict]]:
        calls.append(loader.num_workers)
        if loader.num_workers > 0:
            raise PermissionError("semlock denied")
        return np.zeros((1, 2), dtype=np.float32), [{"accession": "1"}]

    monkeypatch.setattr(extract_features, "extract_embeddings", _fake_extract_embeddings)

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 1,
        "collate_fn": lambda batch: batch,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }

    features, metadata = extract_features._extract_embeddings_with_fallback(
        model=torch.nn.Identity(),
        dataset=_DummyDataset(),
        loader_kwargs=loader_kwargs,
        device=torch.device("cpu"),
        amp_enabled=False,
        layer_name="avgpool",
        logger=logging.getLogger("tests"),
    )

    assert calls == [1, 0]
    assert features.shape == (1, 2)
    assert metadata == [{"accession": "1"}]


def test_serialize_args_converts_paths() -> None:
    args = argparse.Namespace(
        registry_csv=Path("results/registry.csv"),
        registry_md=Path("results/registry.md"),
        values=["a", Path("outputs/embeddings")],
    )

    serialized = extract_features._serialize_args(args)

    assert serialized["registry_csv"] == "results/registry.csv"
    assert serialized["registry_md"] == "results/registry.md"
    assert serialized["values"] == ["a", "outputs/embeddings"]


def test_resolve_output_dir_reuse(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "reuse"
    outdir.mkdir(parents=True, exist_ok=True)

    reused = extract_features._resolve_output_dir(str(outdir), reuse_outdir=True)
    assert reused == str(outdir)

    incremented = extract_features._resolve_output_dir(str(outdir), reuse_outdir=False)
    assert incremented != str(outdir)
