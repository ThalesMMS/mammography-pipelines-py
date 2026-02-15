import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mammography.tools import embedding_registry as registry


def _write_embedding_artifacts(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "features.npy", np.zeros((2, 4), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "accession": "000001",
                "raw_label": 1,
                "image_path": "dummy.dcm",
            }
        ]
    ).to_csv(outdir / "metadata.csv", index=False)
    (outdir / "joined.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    session_info = {
        "device": "cpu",
        "num_samples": 2,
        "features_shape": [2, 4],
        "args": {
            "dataset": "archive",
            "arch": "resnet50",
            "layer_name": "avgpool",
            "batch_size": 32,
            "img_size": 512,
        },
    }
    (outdir / "session_info.json").write_text(
        json.dumps(session_info), encoding="utf-8"
    )
    (outdir / "example_embedding.json").write_text("{}", encoding="utf-8")
    for name in ["pca_label.png", "tsne_label.png", "umap_label.png"]:
        (outdir / name).write_bytes(b"dummy")
    (outdir / "clustering_metrics.json").write_text("{}", encoding="utf-8")
    (outdir / "clustering_metrics.png").write_bytes(b"dummy")
    (outdir / "pca_cluster.png").write_bytes(b"dummy")
    (outdir / "tsne_cluster.png").write_bytes(b"dummy")
    preview_dir = outdir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "first_image_loaded.png").write_bytes(b"dummy")


def test_default_run_name_uses_dataset_alias() -> None:
    assert registry.default_run_name("patches_completo", "resnet50") == "patches_embed_resnet50"


def test_default_run_name_uses_arch_alias_for_effnet() -> None:
    assert (
        registry.default_run_name("patches_completo", "efficientnet_b0")
        == "patches_embed_effnet"
    )


def test_register_embedding_run_writes_registry_and_mlflow_fallback(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "archive_embeddings"
    _write_embedding_artifacts(outdir)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_embedding_run(
        outdir=outdir,
        dataset="archive",
        model="resnet50",
        layer="avgpool",
        batch_size=32,
        img_size=512,
        run_name="archive_embed_resnet50",
        command="mammography embed --dataset archive",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    assert registry_csv.exists()
    assert registry_md.exists()

    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset"] == "archive"
    assert rows[0]["workflow"] == "embed"
    assert rows[0]["model"] == "resnet50"
    assert rows[0]["layer"] == "avgpool"
    assert rows[0]["batch_size"] == "32"
    assert rows[0]["img_size"] == "512"
    assert "pca_label.png" in rows[0]["reduction_outputs"]
    assert "clustering_metrics.json" in rows[0]["clustering_outputs"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_embed_resnet50" in md_text
    assert "Model: resnet50" in md_text
    assert "Reduction outputs" in md_text
    assert "Clustering outputs" in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "features.npy").exists()
    assert (artifacts_dir / "metadata.csv").exists()
    assert (artifacts_dir / "joined.csv").exists()
    assert (artifacts_dir / "session_info.json").exists()
    assert (artifacts_dir / "preview" / "first_image_loaded.png").exists()


def test_main_uses_session_info_when_args_missing(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "archive_embeddings"
    _write_embedding_artifacts(outdir)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    exit_code = registry.main(
        [
            "--outdir",
            str(outdir),
            "--run-name",
            "archive_embed_effnet",
            "--command",
            "mammography embed --dataset archive",
            "--registry-csv",
            str(registry_csv),
            "--registry-md",
            str(registry_md),
            "--tracking-uri",
            str(tracking_root),
        ]
    )

    assert exit_code == 0
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset"] == "archive"
    assert rows[0]["model"] == "resnet50"
    assert rows[0]["layer"] == "avgpool"
