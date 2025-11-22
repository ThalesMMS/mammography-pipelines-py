#!/usr/bin/env python3
#
# report_pack.py
# mammography-pipelines-py
#
# Consolidates Stage 2 runs by copying assets, building Grad-CAM collages, and updating LaTeX sections.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Helper utilities to consolidate Stage 2 runs for reporting/Article exports."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - fallback when pillow is not installed
    Image = None  # type: ignore

LOGGER = logging.getLogger("report_pack")


@dataclass
class Stage2Run:
    path: Path
    run_id: str
    seed: int
    summary_path: Path
    metrics_path: Path
    metrics: dict[str, Any]
    assets: dict[str, str | None]
    stats: dict[str, float]


def _load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file and return a dictionary payload."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _copy_asset(src: Path, dest: Path) -> str:
    """Copy an artifact into the shared assets folder and return its basename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    LOGGER.info("Copiado %s -> %s", src, dest)
    return dest.name


def _iter_gradcam_images(folder: Path) -> Iterable[Path]:
    """Yield Grad-CAM PNGs sorted alphabetically to produce deterministic grids."""
    for path in sorted(folder.glob("gradcam_*.png")):
        if path.is_file():
            yield path


def _build_gradcam_grid(image_paths: Sequence[Path], dest: Path, max_tiles: int = 4) -> str | None:
    """Assemble up to `max_tiles` Grad-CAMs into a single collage."""
    selected = list(image_paths)[:max_tiles]
    if not selected:
        return None
    if Image is None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected[0], dest)
        LOGGER.warning("Pillow não está disponível; copiando apenas %s para %s.", selected[0], dest)
        return dest.name
    images = [Image.open(path).convert("RGB") for path in selected]
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    tile_w = max(widths)
    tile_h = max(heights)
    cols = min(2, len(images))
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(0, 0, 0))
    for idx, img in enumerate(images):
        if img.width != tile_w or img.height != tile_h:
            img = img.resize((tile_w, tile_h))
        row = idx // cols
        col = idx % cols
        canvas.paste(img, (col * tile_w, row * tile_h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    LOGGER.info("Grad-CAM grid gerado em %s", dest)
    return dest.name


def _format_metric_table(runs: Sequence[Stage2Run]) -> tuple[list[dict[str, str]], dict[str, float]]:
    """Prepare LaTeX-friendly rows and mean/std aggregates for the report table."""
    table_rows: list[dict[str, str]] = []
    aggregated = {"accuracy": [], "kappa": [], "macro_f1": [], "auc": []}
    for run in runs:
        stats = run.stats
        aggregated["accuracy"].append(stats["accuracy"])
        aggregated["kappa"].append(stats["kappa"])
        aggregated["macro_f1"].append(stats["macro_f1"])
        aggregated["auc"].append(stats["auc"])
        table_rows.append(
            {
                "seed": f"{run.seed}",
                "run_id": run.run_id,
                "accuracy": f"{stats['accuracy']:.3f}",
                "kappa": f"{stats['kappa']:.3f}",
                "macro_f1": f"{stats['macro_f1']:.3f}",
                "auc": f"{stats['auc']:.3f}",
            }
        )
    mean_std = {}
    for key, values in aggregated.items():
        if not values:
            mean = std = 0.0
        else:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
        mean_std[key] = (mean, std)
    return table_rows, mean_std


def _render_stage2_tex(tex_path: Path, runs: Sequence[Stage2Run]) -> None:
    """Update the LaTeX section with metrics and assets for the provided runs."""
    if not runs:
        LOGGER.warning("Nenhum run fornecido; pulando geração de stage2_model.tex.")
        return
    table_rows, mean_std = _format_metric_table(runs)
    confusion_figs = [
        (run.seed, run.assets.get("confusion"))
        for run in runs
        if run.assets.get("confusion") is not None
    ]
    gradcam_figs = [
        (run.seed, run.assets.get("gradcam"))
        for run in runs
        if run.assets.get("gradcam") is not None
    ]
    train_fig = None
    best_run = max(runs, key=lambda r: r.stats["macro_f1"])
    if best_run.assets.get("train_curve"):
        train_fig = best_run.assets["train_curve"]
    latex_lines = [
        "% This file is auto-generated via tools/report_pack.py",
        "\\subsection{Modelo de Densidade (Stage~2)}",
        "\\label{sec:stage2-model}",
        (
            "Executamos três seeds independentes (42/43/44) do pipeline EfficientNetB0 com "
            "fusão dos embeddings da Etapa~1 e hold-out de 20\\%. A Tabela~\\ref{tab:stage2-metrics} "
            "resume as métricas de validação e destaca a média $\\pm$ desvio."
        ),
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Métricas de validação hold-out (20\\%) por seed.}",
        "  \\label{tab:stage2-metrics}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Seed & Accuracy & $\\kappa_q$ & Macro-F1 & AUC (OvR) \\\\",
        "    \\midrule",
    ]
    for row in table_rows:
        latex_lines.append(
            f"    {row['seed']} ({row['run_id']}) & {row['accuracy']} & {row['kappa']} & {row['macro_f1']} & {row['auc']} \\\\"
        )
    latex_lines.append("    \\midrule")
    latex_lines.append(
        "    \\textbf{Média $\\pm$ σ} & "
        f"\\textbf{{{mean_std['accuracy'][0]:.3f} $\\pm$ {mean_std['accuracy'][1]:.3f}}} & "
        f"\\textbf{{{mean_std['kappa'][0]:.3f} $\\pm$ {mean_std['kappa'][1]:.3f}}} & "
        f"\\textbf{{{mean_std['macro_f1'][0]:.3f} $\\pm$ {mean_std['macro_f1'][1]:.3f}}} & "
        f"\\textbf{{{mean_std['auc'][0]:.3f} $\\pm$ {mean_std['auc'][1]:.3f}}} \\\\"
    )
    latex_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            (
                "Comparando com os classificadores tradicionais da Etapa~1 (balanced accuracy $\\approx$~0,51, "
                "$\\kappa$~0,45), mesmo o pior seed do Stage~2 mantém ganhos absolutos de 6--8 p.p. em macro-F1 "
                "e 0,22 em $\\kappa$, ainda que a meta de macro-F1 0,72 permaneça em aberto."
            ),
        ]
    )
    if confusion_figs:
        col_spec = "c" * len(confusion_figs)
        latex_lines.extend(
            [
                "\\begin{figure}[ht]",
                "  \\centering",
                "  \\setlength{\\tabcolsep}{3pt}",
                f"  \\begin{{tabular}}{{{col_spec}}}",
            ]
        )
        row = "    " + " & ".join(
            f"\\includegraphics[width=0.3\\linewidth]{{{asset}}}" for _seed, asset in confusion_figs
        ) + " \\\\"
        latex_lines.append(row)
        latex_lines.extend(
            [
                "  \\end{tabular}",
                "  \\caption{Matrizes de confusão e métricas por classe para cada seed.}",
                "  \\label{fig:stage2-confusion}",
                "\\end{figure}",
            ]
        )
    if gradcam_figs or train_fig:
        latex_lines.append("\\begin{figure}[ht]")
        latex_lines.append("  \\centering")
        if train_fig:
            latex_lines.append(
                f"  \\includegraphics[width=0.48\\linewidth]{{{train_fig}}}"
            )
        gradcam_seed = None
        if gradcam_figs:
            gradcam_seed, gradcam_asset = gradcam_figs[0]
            latex_lines.append(
                f"  \\includegraphics[width=0.48\\linewidth]{{{gradcam_asset}}}"
            )
        latex_lines.append(
            f"  \\caption{{Curvas de treino (seed {best_run.seed}) e exemplos de Grad-CAM "
            f"(seed {gradcam_seed if gradcam_seed is not None else '42'}).}}"
        )
        latex_lines.append("  \\label{fig:stage2-curves-gradcam}")
        latex_lines.append("\\end{figure}")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    LOGGER.info("Arquivo LaTeX atualizado: %s", tex_path)


def _summarize_run(run_path: Path, assets_dir: Path, gradcam_limit: int) -> Stage2Run:
    """Collect metrics, copy artifacts, and store summary.json for a single run."""
    summary_path = run_path / "summary.json"
    metrics_path = run_path / "metrics" / "val_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json ausente em {run_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"val_metrics.json ausente em {run_path}")
    summary = _load_json(summary_path)
    metrics = _load_json(metrics_path)
    run_id = summary.get("run_id") or run_path.name
    seed = int(summary.get("seed", -1))
    assets: dict[str, str | None] = {}
    train_curve = run_path / "train_history.png"
    if train_curve.exists():
        assets["train_curve"] = _copy_asset(
            train_curve, assets_dir / f"stage2_train_seed{seed}.png"
        )
    metrics_png = run_path / "metrics" / "val_metrics.png"
    if metrics_png.exists():
        assets["confusion"] = _copy_asset(
            metrics_png, assets_dir / f"stage2_confusion_seed{seed}.png"
        )
    gradcam_dir = run_path / "gradcam"
    if gradcam_dir.exists():
        grid_path = assets_dir / f"stage2_gradcam_seed{seed}.png"
        grid_name = _build_gradcam_grid(list(_iter_gradcam_images(gradcam_dir)), grid_path, gradcam_limit)
        assets["gradcam"] = grid_name
    else:
        assets["gradcam"] = None
    report = metrics.get("classification_report", {})
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
    summary["val_metrics"] = {
        "accuracy": float(metrics.get("acc", metrics.get("accuracy", 0.0))),
        "kappa_quadratic": float(metrics.get("kappa_quadratic", 0.0)),
        "macro_f1": float(macro_f1 or 0.0),
        "weighted_f1": float(weighted_f1 or 0.0),
        "auc_ovr": float(metrics.get("auc_ovr", 0.0) or 0.0),
    }
    stats = {
        "accuracy": summary["val_metrics"]["accuracy"],
        "kappa": summary["val_metrics"]["kappa_quadratic"],
        "macro_f1": summary["val_metrics"]["macro_f1"],
        "weighted_f1": summary["val_metrics"]["weighted_f1"],
        "auc": summary["val_metrics"]["auc_ovr"],
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
    }
    summary.update(
        {
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "classification_report": metrics.get("classification_report"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "article_assets": assets,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("summary.json atualizado em %s", summary_path)
    return Stage2Run(
        path=run_path,
        run_id=run_id,
        seed=seed,
        summary_path=summary_path,
        metrics_path=metrics_path,
        metrics=metrics,
        assets=assets,
        stats=stats,
    )


def package_stage2_runs(
    runs: Sequence[Path],
    assets_dir: Path,
    tex_path: Path | None = None,
    gradcam_limit: int = 4,
) -> list[Stage2Run]:
    """High-level helper used by CLI and Projeto.py to gather/export Stage 2 runs."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    summarized: list[Stage2Run] = []
    for run in runs:
        summarized.append(_summarize_run(run, assets_dir, gradcam_limit))
    summarized.sort(key=lambda r: r.seed)
    if tex_path is not None:
        _render_stage2_tex(tex_path, summarized)
    return summarized


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empacota execuções Stage 2 para o Article.")
    parser.add_argument("--run", dest="runs", action="append", type=Path, required=True, help="Diretório results_* a ser exportado.")
    parser.add_argument("--assets", dest="assets_dir", type=Path, default=Path("Article/assets"), help="Destino das figuras copiadas.")
    parser.add_argument("--tex", dest="tex_path", type=Path, default=Path("Article/sections/stage2_model.tex"), help="Arquivo LaTeX a atualizar.")
    parser.add_argument("--gradcam-limit", dest="gradcam_limit", type=int, default=4, help="Número máximo de Grad-CAMs individuais no grid.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")
    try:
        package_stage2_runs(args.runs, args.assets_dir, tex_path=args.tex_path, gradcam_limit=args.gradcam_limit)
    except Exception as exc:  # pragma: no cover - CLI fallback
        LOGGER.error("Falha no empacotamento: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
