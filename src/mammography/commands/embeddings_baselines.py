#!/usr/bin/env python3
#
# embeddings_baselines.py
# mammography-pipelines
#
# Compare ResNet50 embeddings versus handcrafted descriptors using classical classifiers.
#
"""Baseline classifiers for mammography embeddings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy, kurtosis, skew, ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

from mammography.io.dicom import dicom_to_pil_rgb, is_dicom_path
from mammography.utils.numpy_warnings import (
    suppress_numpy_matmul_warnings,
    resolve_pca_svd_solver,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLASS_LABELS = {1, 2, 3, 4}


def _coerce_label(val: object) -> int | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        raw = val.strip().upper()
        if raw in {"A", "B", "C", "D"}:
            return {"A": 1, "B": 2, "C": 3, "D": 4}[raw]
        try:
            val = int(raw)
        except Exception:
            return None
    if isinstance(val, (int, np.integer)):
        ival = int(val)
        if ival in {0, 1, 2, 3}:
            return ival + 1
        return ival
    try:
        return int(val)
    except Exception:
        return None


def _resolve_label_column(meta: pd.DataFrame) -> str:
    for col in ("raw_label", "professional_label", "density_label", "label", "Classification"):
        if col in meta.columns:
            return col
    raise ValueError("metadata.csv nao possui coluna de labels reconhecida.")


def _resolve_path_column(meta: pd.DataFrame) -> str:
    for col in ("path", "image_path", "dicom_path"):
        if col in meta.columns:
            return col
    raise ValueError("metadata.csv nao possui coluna de caminho (path/image_path/dicom_path).")


def _normalize_path(raw: str, roots: List[Path]) -> Path:
    candidate = Path(str(raw).replace("\\", "/")).expanduser()
    if candidate.is_absolute():
        return candidate
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (roots[0] / candidate).resolve()


def _compute_basic_stats(arr: np.ndarray) -> List[float]:
    flat = arr.reshape(-1).astype(np.float32)
    hist = np.histogram(flat, bins=32, range=(0, 255), density=True)[0] + 1e-9
    return [
        float(flat.mean()),
        float(flat.std(ddof=1)),
        float(flat.min()),
        float(flat.max()),
        float(np.percentile(flat, 10)),
        float(np.percentile(flat, 25)),
        float(np.percentile(flat, 50)),
        float(np.percentile(flat, 75)),
        float(np.percentile(flat, 90)),
        float(skew(flat)),
        float(kurtosis(flat)),
        float(entropy(hist)),
    ]


def _load_grayscale(path: Path) -> Image.Image:
    if is_dicom_path(str(path)):
        img = dicom_to_pil_rgb(str(path))
    else:
        img = Image.open(path).convert("RGB")
    return img.convert("L")


def load_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = np.load(embeddings_dir / "features.npy")
    if features.ndim == 1:
        features = features.reshape(1, -1)
    metadata = pd.read_csv(embeddings_dir / "metadata.csv")
    label_col = _resolve_label_column(metadata)
    labels = metadata[label_col].apply(_coerce_label)
    mask = labels.isin(CLASS_LABELS)
    filtered = metadata.loc[mask].reset_index(drop=True)
    return features[mask.values], labels[mask].to_numpy(dtype=np.int64), filtered


def compute_handcrafted_features(
    metadata: pd.DataFrame,
    cache_path: Path,
    img_size: int,
    embeddings_dir: Path,
    pca_svd_solver: str | None = "auto",
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(metadata):
            return cached

    roots = [Path.cwd(), REPO_ROOT, embeddings_dir]
    path_col = _resolve_path_column(metadata)
    stats_features: List[List[float]] = []
    resized_vectors: List[np.ndarray] = []

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Handcrafted features"):
        raw_path = getattr(row, path_col)
        path = _normalize_path(str(raw_path), roots)
        img_gray = _load_grayscale(path)
        arr = np.asarray(img_gray, dtype=np.float32)
        stats_features.append(_compute_basic_stats(arr))

        resized = img_gray.resize((img_size, img_size), resample=Image.BICUBIC)
        resized_vectors.append(np.asarray(resized, dtype=np.float32).reshape(-1))

    stats_matrix = np.array(stats_features, dtype=np.float32)
    resized_matrix = np.stack(resized_vectors, axis=0) / 255.0
    components = min(32, resized_matrix.shape[1])
    solver = resolve_pca_svd_solver(
        resized_matrix.shape[0], resized_matrix.shape[1], components, pca_svd_solver
    )
    pca = PCA(n_components=components, random_state=42, svd_solver=solver)
    with suppress_numpy_matmul_warnings():
        pca_matrix = pca.fit_transform(resized_matrix)
    handcrafted = np.concatenate([stats_matrix, pca_matrix], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, handcrafted)
    return handcrafted


def build_models() -> Dict[str, Callable[[], Pipeline]]:
    return {
        "logreg": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial")),
            ]
        ),
        "svm-linear": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(max_iter=5000)),
            ]
        ),
        "svm-rbf": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", gamma="scale")),
            ]
        ),
        "rf": lambda: Pipeline(
            [
                ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42)),
            ]
        ),
    }


def evaluate_feature_set(name: str, X: np.ndarray, y: np.ndarray, out_dir: Path) -> Dict[str, Dict[str, float]]:
    models = build_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    summary: Dict[str, Dict[str, float]] = {}

    for model_name, builder in models.items():
        fold_metrics: List[Dict[str, float]] = []
        for train_idx, test_idx in skf.split(X, y):
            model = builder()
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            fold_metrics.append(
                {
                    "balanced_accuracy": balanced_accuracy_score(y[test_idx], preds),
                    "macro_f1": f1_score(y[test_idx], preds, average="macro"),
                    "kappa": cohen_kappa_score(y[test_idx], preds),
                }
            )
        metrics_df = pd.DataFrame(fold_metrics)
        summary[model_name] = {
            "balanced_accuracy_mean": float(metrics_df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(metrics_df["balanced_accuracy"].std(ddof=1)),
            "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
            "macro_f1_std": float(metrics_df["macro_f1"].std(ddof=1)),
            "kappa_mean": float(metrics_df["kappa"].mean()),
            "kappa_std": float(metrics_df["kappa"].std(ddof=1)),
            "fold_balanced_accuracy": metrics_df["balanced_accuracy"].tolist(),
        }

    out_path = out_dir / f"{name}_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def paired_t_test(
    deep_scores: List[float],
    classical_scores: List[float],
    out_path: Path,
) -> None:
    stat, pval = ttest_rel(deep_scores, classical_scores)
    payload = {"t_statistic": float(stat), "p_value": float(pval)}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_report(
    embeddings_summary: Dict[str, Dict[str, float]],
    classical_summary: Dict[str, Dict[str, float]],
    report_path: Path,
) -> None:
    def _lines(summary: Dict[str, Dict[str, float]], label: str) -> List[str]:
        lines = [f"### {label}"]
        for model, metrics in summary.items():
            lines.append(
                f"- **{model}** — BA: {metrics['balanced_accuracy_mean']:.3f} ± {metrics['balanced_accuracy_std']:.3f}, "
                f"macro-F1: {metrics['macro_f1_mean']:.3f}, kappa: {metrics['kappa_mean']:.3f}"
            )
        return lines

    rows = ["# Baselines de Embeddings", ""]
    rows += _lines(embeddings_summary, "Embeddings (ResNet50)") + [""]
    rows += _lines(classical_summary, "Classicos (PCA + textura)") + [""]
    report_path.write_text("\n".join(rows), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baselines classicos (embeddings vs descritores).")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("outputs/embeddings_resnet50"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/embeddings_baselines"))
    parser.add_argument("--cache-classic", type=Path, default=Path("outputs/embeddings_baselines/classic_features.npy"))
    parser.add_argument("--img-size", type=int, default=224, help="Tamanho do resize para features classicas (default: 224).")
    parser.add_argument(
        "--pca-svd-solver",
        default="auto",
        choices=["auto", "full", "randomized", "arpack"],
        help="Solver do PCA (auto/full/randomized/arpack).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    X_embed, y, meta = load_embeddings(args.embeddings_dir)
    handcrafted = compute_handcrafted_features(
        meta,
        args.cache_classic,
        args.img_size,
        args.embeddings_dir,
        pca_svd_solver=args.pca_svd_solver,
    )

    embed_summary = evaluate_feature_set("embeddings", X_embed, y, args.outdir)
    classic_summary = evaluate_feature_set("handcrafted", handcrafted, y, args.outdir)

    best_embed = max(embed_summary.items(), key=lambda kv: kv[1]["balanced_accuracy_mean"])
    best_classic = max(classic_summary.items(), key=lambda kv: kv[1]["balanced_accuracy_mean"])
    paired_t_test(
        best_embed[1]["fold_balanced_accuracy"],
        best_classic[1]["fold_balanced_accuracy"],
        args.outdir / "paired_t_test.json",
    )
    render_report(embed_summary, classic_summary, args.outdir / "report.md")

    print("Best embedding model:", best_embed[0], best_embed[1]["balanced_accuracy_mean"])
    print("Best classical model:", best_classic[0], best_classic[1]["balanced_accuracy_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
