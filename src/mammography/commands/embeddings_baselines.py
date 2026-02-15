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
import shlex
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy, kurtosis, skew, ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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


def _limit_samples(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    max_samples: int | None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not max_samples or max_samples <= 0:
        return features, labels, metadata
    if max_samples >= len(labels):
        return features, labels, metadata
    n_classes = len(np.unique(labels))
    if max_samples < n_classes:
        raise ValueError(
            "max_samples precisa ser maior ou igual ao numero de classes."
        )
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=max_samples, random_state=42
    )
    indices, _ = next(splitter.split(features, labels))
    return (
        features[indices],
        labels[indices],
        metadata.iloc[indices].reset_index(drop=True),
    )


def load_embeddings(
    embeddings_dir: Path, max_samples: int | None = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = np.load(embeddings_dir / "features.npy")
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if not np.isfinite(features).all():
        warnings.warn(
            "Embeddings contem valores NaN/inf; substituindo por zeros.",
            RuntimeWarning,
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = pd.read_csv(embeddings_dir / "metadata.csv")
    label_col = _resolve_label_column(metadata)
    labels = metadata[label_col].apply(_coerce_label)
    mask = labels.isin(CLASS_LABELS)
    filtered = metadata.loc[mask].reset_index(drop=True)
    filtered_features = features[mask.values]
    filtered_labels = labels[mask].to_numpy(dtype=np.int64)
    return _limit_samples(filtered_features, filtered_labels, filtered, max_samples)


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
    components = min(32, resized_matrix.shape[1], resized_matrix.shape[0])
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


def _extract_auc_scores(model: Pipeline, X_test: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def _compute_auc(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> float:
    scores = _extract_auc_scores(model, X_test)
    if scores is None:
        return float("nan")
    scores_arr = np.asarray(scores)
    try:
        if scores_arr.ndim == 1:
            return float(roc_auc_score(y_test, scores_arr))
        return float(roc_auc_score(y_test, scores_arr, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


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
                    "accuracy": accuracy_score(y[test_idx], preds),
                    "balanced_accuracy": balanced_accuracy_score(y[test_idx], preds),
                    "macro_f1": f1_score(y[test_idx], preds, average="macro"),
                    "auc": _compute_auc(model, X[test_idx], y[test_idx]),
                    "kappa": cohen_kappa_score(y[test_idx], preds),
                }
            )
        metrics_df = pd.DataFrame(fold_metrics)
        summary[model_name] = {
            "accuracy_mean": float(metrics_df["accuracy"].mean()),
            "accuracy_std": float(metrics_df["accuracy"].std(ddof=1)),
            "balanced_accuracy_mean": float(metrics_df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(metrics_df["balanced_accuracy"].std(ddof=1)),
            "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
            "macro_f1_std": float(metrics_df["macro_f1"].std(ddof=1)),
            "auc_mean": float(metrics_df["auc"].mean()),
            "auc_std": float(metrics_df["auc"].std(ddof=1)),
            "kappa_mean": float(metrics_df["kappa"].mean()),
            "kappa_std": float(metrics_df["kappa"].std(ddof=1)),
            "fold_accuracy": metrics_df["accuracy"].tolist(),
            "fold_balanced_accuracy": metrics_df["balanced_accuracy"].tolist(),
            "fold_macro_f1": metrics_df["macro_f1"].tolist(),
            "fold_auc": metrics_df["auc"].tolist(),
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
                f"- **{model}** — Acc: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}, "
                f"macro-F1: {metrics['macro_f1_mean']:.3f}, AUC: {metrics['auc_mean']:.3f}, "
                f"BA: {metrics['balanced_accuracy_mean']:.3f}, kappa: {metrics['kappa_mean']:.3f}"
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
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Limite opcional de amostras para acelerar o baseline "
            "(0 = sem limite)."
        ),
    )
    parser.add_argument(
        "--pca-svd-solver",
        default="auto",
        choices=["auto", "full", "randomized", "arpack"],
        help="Solver do PCA (auto/full/randomized/arpack).",
    )
    parser.add_argument("--dataset", default="", help="Identificador do dataset (opcional).")
    parser.add_argument("--run-name", default="", help="Nome do run no MLflow")
    parser.add_argument("--tracking-uri", default="", help="Tracking URI para MLflow")
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    parser.add_argument("--experiment", default="", help="Nome opcional do experimento no MLflow")
    parser.add_argument("--no-mlflow", action="store_true", help="Nao registrar no MLflow")
    parser.add_argument("--no-registry", action="store_true", help="Nao atualizar registry local")
    return parser.parse_args(argv)


def _register_baselines_run(
    *,
    report_path: Path,
    embeddings_dir: Path,
    outdir: Path,
    args: argparse.Namespace,
    command: str,
) -> Tuple[str, str | None]:
    from mammography.tools import baselines_registry

    dataset = args.dataset or baselines_registry.infer_dataset_name(embeddings_dir, outdir)
    run_name = args.run_name or baselines_registry.default_run_name(embeddings_dir, outdir)
    run_id = baselines_registry.register_baselines_run(
        report_path=report_path,
        dataset=dataset,
        run_name=run_name,
        command=command,
        registry_csv=args.registry_csv,
        registry_md=args.registry_md,
        tracking_uri=args.tracking_uri or None,
        experiment=args.experiment or None,
        log_mlflow=not args.no_mlflow,
    )
    return run_name, run_id


def _build_baselines_report(
    embeddings_summary: Dict[str, Dict[str, float]],
    classical_summary: Dict[str, Dict[str, float]],
    best_embed: Tuple[str, Dict[str, float]],
    best_classic: Tuple[str, Dict[str, float]],
    embeddings_dir: Path,
    outdir: Path,
    max_samples: int | None = None,
) -> Dict[str, object]:
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "embeddings_dir": str(embeddings_dir),
        "outdir": str(outdir),
        "max_samples": max_samples or 0,
        "feature_sets": {
            "embeddings": embeddings_summary,
            "handcrafted": classical_summary,
        },
        "best_models": {
            "embeddings": {
                "model": best_embed[0],
                "balanced_accuracy_mean": best_embed[1]["balanced_accuracy_mean"],
            },
            "handcrafted": {
                "model": best_classic[0],
                "balanced_accuracy_mean": best_classic[1]["balanced_accuracy_mean"],
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    max_samples = args.max_samples
    if max_samples <= 0:
        env_limit = os.getenv("MAMMOGRAPHY_BASELINES_MAX_SAMPLES", "").strip()
        if env_limit:
            try:
                max_samples = int(env_limit)
            except ValueError:
                raise ValueError(
                    "MAMMOGRAPHY_BASELINES_MAX_SAMPLES deve ser um inteiro."
                ) from None

    X_embed, y, meta = load_embeddings(args.embeddings_dir, max_samples=max_samples)
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
    baselines_report = _build_baselines_report(
        embed_summary,
        classic_summary,
        best_embed,
        best_classic,
        args.embeddings_dir,
        args.outdir,
        max_samples=max_samples,
    )
    report_path = args.outdir / "baselines_report.json"
    report_path.write_text(json.dumps(baselines_report, indent=2), encoding="utf-8")

    print("Best embedding model:", best_embed[0], best_embed[1]["balanced_accuracy_mean"])
    print("Best classical model:", best_classic[0], best_classic[1]["balanced_accuracy_mean"])

    if not args.no_registry:
        try:
            run_name, _run_id = _register_baselines_run(
                report_path=report_path,
                embeddings_dir=args.embeddings_dir,
                outdir=args.outdir,
                args=args,
                command=shlex.join(sys.argv),
            )
            print(f"[ok] Registry atualizado (run_name={run_name}).")
        except Exception as exc:
            print(f"[warn] Falha ao registrar baselines: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
