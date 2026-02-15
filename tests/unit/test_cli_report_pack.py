from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mammography import cli


def test_run_report_pack_passes_run_name_and_outputs(tmp_path: Path) -> None:
    assets_dir = tmp_path / "Article" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "density_train_seed42.png").write_text("asset", encoding="utf-8")
    (assets_dir / "density_confusion_seed42.png").write_text("asset", encoding="utf-8")
    tex_path = tmp_path / "Article" / "sections" / "density_model.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("tex", encoding="utf-8")

    run_dir = tmp_path / "outputs" / "mamografias_density_effnet" / "results_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        runs=[run_dir],
        assets_dir=assets_dir,
        tex_path=tex_path,
        gradcam_limit=4,
        run_name="mamografias_report_pack",
        tracking_uri="",
        experiment="",
        registry_csv=tmp_path / "results" / "registry.csv",
        registry_md=tmp_path / "results" / "registry.md",
        no_mlflow=False,
        no_registry=False,
    )

    summarized = [
        SimpleNamespace(
            assets={
                "train_curve": "density_train_seed42.png",
                "confusion": "density_confusion_seed42.png",
                "gradcam": None,
                "explanations": None,
            }
        )
    ]

    with patch(
        "mammography.tools.report_pack.package_density_runs",
        return_value=summarized,
    ), patch(
        "mammography.tools.report_pack_registry.register_report_pack_run"
    ) as mock_register:
        exit_code = cli._run_report_pack(args, [])

    assert exit_code == 0
    kwargs = mock_register.call_args.kwargs
    assert kwargs["run_name"] == "mamografias_report_pack"
    assert assets_dir / "density_train_seed42.png" in kwargs["output_paths"]
    assert assets_dir / "density_confusion_seed42.png" in kwargs["output_paths"]
    assert tex_path in kwargs["output_paths"]
    assert "--run-name" in kwargs["command"]
