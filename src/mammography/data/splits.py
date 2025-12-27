import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

LOGGER = logging.getLogger("mammography")


def create_splits(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
    num_classes: int = 4,
    ensure_val_has_all_classes: bool = True,
    max_tries: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation splits with group-aware stratification when possible."""
    if "professional_label" not in df.columns:
        raise ValueError("DataFrame deve conter a coluna 'professional_label'.")

    df = df.copy()

    def _map_label(lbl) -> Optional[int]:
        if pd.isna(lbl):
            return None
        try:
            val = int(lbl)
        except Exception:
            return None
        if num_classes == 2:
            if val in (1, 2):
                return 0
            if val in (3, 4):
                return 1
            return None
        target = val - 1
        return target if 0 <= target < num_classes else None

    df["_target"] = df["professional_label"].apply(_map_label)
    df = df[df["_target"].notna()].copy()
    if df.empty:
        raise RuntimeError("Nenhuma amostra valida apos mapear os rotulos.")
    df["_target"] = df["_target"].astype(int)

    # Split without groups if accession is missing.
    if "accession" not in df.columns or df["accession"].isna().all():
        LOGGER.warning("Coluna 'accession' ausente; split sem agrupamento.")
        y = df["_target"].values
        strat = y if len(np.unique(y)) > 1 else None
        train_idx, val_idx = train_test_split(
            df.index.to_numpy(),
            test_size=val_frac,
            random_state=seed,
            stratify=strat,
        )
        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy()
    else:
        group_targets = df.groupby("accession")["_target"].agg(lambda s: s.value_counts().idxmax())
        group_ids = group_targets.index.to_numpy()
        group_y = group_targets.values

        if len(group_ids) < 2:
            raise RuntimeError("Grupos insuficientes para dividir train/val.")

        group_counts = pd.Series(group_y).value_counts().to_dict()
        required_train = [c for c, n in group_counts.items() if n >= 1]
        required_val = [c for c, n in group_counts.items() if n >= 2] if ensure_val_has_all_classes else []

        def _has_required(sub_df: pd.DataFrame, required) -> bool:
            counts = sub_df["_target"].value_counts().to_dict()
            return all(counts.get(c, 0) > 0 for c in required)

        train_df = val_df = None
        for attempt in range(max_tries):
            rs = seed + attempt
            strat = group_y if len(np.unique(group_y)) > 1 else None
            try:
                tr_g, va_g = train_test_split(
                    group_ids,
                    test_size=val_frac,
                    random_state=rs,
                    stratify=strat,
                )
            except ValueError:
                tr_g, va_g = train_test_split(group_ids, test_size=val_frac, random_state=rs, shuffle=True)

            tr_df = df[df["accession"].isin(tr_g)]
            va_df = df[df["accession"].isin(va_g)]
            if _has_required(tr_df, required_train) and _has_required(va_df, required_val):
                train_df, val_df = tr_df.copy(), va_df.copy()
                break

        if train_df is None or val_df is None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            train_idx, val_idx = next(splitter.split(df, df["_target"], groups=df["accession"]))
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

    def _counts(sub_df: pd.DataFrame) -> dict[int, int]:
        counts = sub_df["_target"].value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.items()}

    LOGGER.info(
        "Split criado: Train=%d (groups=%s) | Val=%d (groups=%s) | Train counts=%s | Val counts=%s",
        len(train_df),
        train_df["accession"].nunique() if "accession" in train_df.columns else "NA",
        len(val_df),
        val_df["accession"].nunique() if "accession" in val_df.columns else "NA",
        _counts(train_df),
        _counts(val_df),
    )

    train_df = train_df.drop(columns=["_target"])
    val_df = val_df.drop(columns=["_target"])
    return train_df, val_df
