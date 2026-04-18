#!/usr/bin/env python3
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
"""Checkpoint resume helpers for training workflows."""

from __future__ import annotations

from pathlib import Path


def _checkpoint_model(model):
    return getattr(model, "_orig_mod", model)


def _view_resume_sibling(
    resume_file: Path,
    *,
    current_view: str | None,
    checkpoint_name: str,
    views_to_train,
) -> Path | None:
    if current_view is None:
        return None
    parent_name = resume_file.parent.name
    for view in views_to_train:
        if view is None:
            continue
        suffix = f"_{str(view).lower()}"
        if parent_name.lower().endswith(suffix):
            base_name = parent_name[: -len(suffix)]
            return (
                resume_file.parent.parent
                / f"{base_name}_{current_view}"
                / checkpoint_name
            )
    return None


def _resolve_view_resume_path(
    raw_resume: str,
    *,
    current_view: str | None,
    checkpoint_name: str,
    view_outdir_path: Path,
    views_to_train,
) -> Path:
    resume_path = Path(raw_resume)
    view_count = sum(1 for view in views_to_train if view is not None)

    if resume_path.is_dir():
        candidates = [resume_path / checkpoint_name, view_outdir_path / checkpoint_name]
        if current_view is not None:
            candidates.append(
                resume_path / f"checkpoint_{str(current_view).lower()}.pt"
            )
            sibling = _view_resume_sibling(
                resume_path / checkpoint_name,
                current_view=current_view,
                checkpoint_name=checkpoint_name,
                views_to_train=views_to_train,
            )
            if sibling is not None:
                candidates.append(sibling)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise SystemExit(
            f"Checkpoint nao encontrado para view {current_view or 'default'} em {resume_path}"
        )

    if not resume_path.exists():
        raise SystemExit(f"Checkpoint nao encontrado: {resume_path}")
    if not resume_path.is_file():
        raise SystemExit(f"Checkpoint invalido: {resume_path}")

    if current_view is None or view_count <= 1:
        return resume_path

    candidates: list[Path] = []
    sibling = _view_resume_sibling(
        resume_path,
        current_view=current_view,
        checkpoint_name=checkpoint_name,
        views_to_train=views_to_train,
    )
    if sibling is not None:
        candidates.append(sibling)
    candidates.extend(
        [resume_path.parent / checkpoint_name, view_outdir_path / checkpoint_name]
    )
    if resume_path.name == checkpoint_name:
        candidates.append(resume_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise SystemExit(
        "--resume-from apontou para um unico checkpoint, mas multiplas views serao treinadas. "
        "Informe um diretorio ou checkpoints checkpoint_<view>.pt correspondentes a cada view."
    )
