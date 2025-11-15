#!/usr/bin/env python3
"""Stage 1 classical baselines for mammography embeddings.

This script compares deep embeddings (ResNet50) versus handcrafted descriptors
computed directly from the DICOMs. It trains Logistic Regression, Linear/RBF SVM,
and RandomForest classifiers with 5-fold stratified CV and records balanced
accuracy, macro-F1, and quadratic kappa. A paired t-test evaluates whether the
best deep model significantly outperforms the best classical model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew, ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extract_mammo_resnet50 import dicom_debug_preprocess  # type: ignore  # noqa: E402

CLASS_LABELS = [1, 2, 3, 4]


def _normalize_path(raw: str) -> Path:
    candidate = Path(raw.replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def load_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = np.load(embeddings_dir / "features.npy")
    metadata = pd.read_csv(embeddings_dir / "metadata.csv")
    mask = metadata["Classification"].isin(CLASS_LABELS)
    filtered = metadata.loc[mask].reset_index(drop=True)
    return features[mask.values], filtered["Classification"].to_numpy(), filtered


def _compute_basic_stats(arr: np.ndarray) -> List[float]:
    flat = arr.reshape(-1).astype(np.float32)
    stats = [
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
        float(entropy(np.histogram(flat, bins=32, range=(0, 255), density=True)[0] + 1e-9)),
    ]
    return stats


def compute_handcrafted_features(metadata: pd.DataFrame, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    stats_features: List[List[float]] = []
    resized_vectors: List[np.ndarray] = []

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Handcrafted features"):
        dicom_path = _normalize_path(row.dicom_path)
        info = dicom_debug_preprocess(str(dicom_path))
        win = info["win_uint8"].astype(np.float32)
        stats_features.append(_compute_basic_stats(win))

        pil_resized = info["pil_resized_rgb"].convert("L")
        resized_vectors.append(np.asarray(pil_resized, dtype=np.float32).reshape(-1))

    stats_matrix = np.array(stats_features, dtype=np.float32)
    resized_matrix = np.stack(resized_vectors, axis=0) / 255.0
    components = min(32, resized_matrix.shape[1])
    pca = PCA(n_components=components, random_state=42)
    pca_matrix = pca.fit_transform(resized_matrix)
    handcrafted = np.concatenate([stats_matrix, pca_matrix], axis=1)
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
                f"macro-F1: {metrics['macro_f1_mean']:.3f}, κ: {metrics['kappa_mean']:.3f}"
            )
        return lines

    rows = ["# Stage 1 Baselines", ""]
    rows += _lines(embeddings_summary, "Embeddings (ResNet50)") + [""]
    rows += _lines(classical_summary, "Clássicos (PCA + textura)") + [""]
    report_path.write_text("\n".join(rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 baseline classifiers.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("outputs/embeddings_resnet50"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/stage1_baselines"))
    parser.add_argument("--cache-classic", type=Path, default=Path("outputs/stage1_baselines/classic_features.npy"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    X_embed, y, meta = load_embeddings(args.embeddings_dir)
    handcrafted = compute_handcrafted_features(meta, args.cache_classic)

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
