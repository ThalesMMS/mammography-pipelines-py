import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

LOGGER = logging.getLogger("mammography")


@dataclass
class SplitConfig:
    """Configuration for train/validation/test splitting."""

    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    ensure_all_classes: bool = True
    max_tries: int = 200


def load_splits_from_csvs(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load pre-defined train/val/test splits from independent CSV files.

    Args:
        train_csv: Path to training set CSV file
        val_csv: Path to validation set CSV file
        test_csv: Optional path to test set CSV file

    Returns:
        Tuple of (train_df, val_df, test_df). test_df is None if test_csv not provided.

    Raises:
        FileNotFoundError: If any CSV file does not exist
        ValueError: If CSV files are missing required columns or have overlapping samples
    """
    # Validate paths exist
    train_path = Path(train_csv)
    val_path = Path(val_csv)
    test_path = Path(test_csv) if test_csv else None

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo de treino nao encontrado: {train_csv}")
    if not val_path.exists():
        raise FileNotFoundError(f"Arquivo de validacao nao encontrado: {val_csv}")
    if test_path and not test_path.exists():
        raise FileNotFoundError(f"Arquivo de teste nao encontrado: {test_csv}")

    # Load CSV files
    LOGGER.info("Carregando splits de CSVs: train=%s, val=%s, test=%s", train_csv, val_csv, test_csv or "None")
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path) if test_path else None
    except Exception as exc:
        raise ValueError(f"Erro ao ler arquivos CSV: {exc}") from exc

    # Validate required columns
    required_cols = {"image_path"}
    for name, df in [("train", train_df), ("val", val_df)]:
        if df is None:
            continue
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV de {name} esta faltando colunas obrigatorias: {missing}")

    if test_df is not None:
        missing = required_cols - set(test_df.columns)
        if missing:
            raise ValueError(f"CSV de teste esta faltando colunas obrigatorias: {missing}")

    # Check for overlaps using image_path (primary) or accession (fallback)
    def _get_identifier_set(df: pd.DataFrame) -> set:
        if "image_path" in df.columns:
            return set(df["image_path"].dropna())
        if "accession" in df.columns:
            return set(df["accession"].dropna())
        raise ValueError("DataFrame deve conter 'image_path' ou 'accession' para verificacao de sobreposicao")

    train_ids = _get_identifier_set(train_df)
    val_ids = _get_identifier_set(val_df)

    # Check train/val overlap
    train_val_overlap = train_ids & val_ids
    if train_val_overlap:
        sample = list(train_val_overlap)[:5]
        raise ValueError(
            f"Sobreposicao encontrada entre train e val ({len(train_val_overlap)} amostras). "
            f"Exemplos: {sample}"
        )

    # Check test overlaps if test set provided
    if test_df is not None:
        test_ids = _get_identifier_set(test_df)

        train_test_overlap = train_ids & test_ids
        if train_test_overlap:
            sample = list(train_test_overlap)[:5]
            raise ValueError(
                f"Sobreposicao encontrada entre train e test ({len(train_test_overlap)} amostras). "
                f"Exemplos: {sample}"
            )

        val_test_overlap = val_ids & test_ids
        if val_test_overlap:
            sample = list(val_test_overlap)[:5]
            raise ValueError(
                f"Sobreposicao encontrada entre val e test ({len(val_test_overlap)} amostras). "
                f"Exemplos: {sample}"
            )

    LOGGER.info(
        "Splits carregados com sucesso: Train=%d | Val=%d | Test=%s",
        len(train_df),
        len(val_df),
        len(test_df) if test_df is not None else "None",
    )

    return train_df, val_df, test_df


def create_three_way_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    num_classes: int = 4,
    ensure_all_splits_have_all_classes: bool = True,
    max_tries: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits with group-aware stratification when possible.

    Args:
        df: DataFrame with 'professional_label' and optionally 'accession' columns
        val_frac: Fraction of data to use for validation (default 0.15)
        test_frac: Fraction of data to use for test (default 0.15)
        seed: Random seed for reproducibility
        num_classes: Number of classes (2 or 4)
        ensure_all_splits_have_all_classes: If True, ensure val and test have all classes
        max_tries: Maximum attempts to find valid split with all classes

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        ValueError: If DataFrame is missing required columns
        RuntimeError: If no valid samples after label mapping or insufficient groups
    """
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

        # First split off test set
        train_val_idx, test_idx = train_test_split(
            df.index.to_numpy(),
            test_size=test_frac,
            random_state=seed,
            stratify=strat,
        )
        test_df = df.loc[test_idx].copy()
        train_val_df = df.loc[train_val_idx].copy()

        # Then split train_val into train and val
        y_train_val = train_val_df["_target"].values
        strat_train_val = y_train_val if len(np.unique(y_train_val)) > 1 else None
        # Adjust val_frac to be relative to train_val size
        val_frac_adjusted = val_frac / (1 - test_frac)
        train_idx, val_idx = train_test_split(
            train_val_df.index.to_numpy(),
            test_size=val_frac_adjusted,
            random_state=seed,
            stratify=strat_train_val,
        )
        train_df = train_val_df.loc[train_idx].copy()
        val_df = train_val_df.loc[val_idx].copy()
    else:
        group_targets = df.groupby("accession")["_target"].agg(lambda s: s.value_counts().idxmax())
        group_ids = group_targets.index.to_numpy()
        group_y = group_targets.values

        if len(group_ids) < 3:
            raise RuntimeError("Grupos insuficientes para dividir train/val/test (minimo: 3).")

        group_counts = pd.Series(group_y).value_counts().to_dict()
        required_train = [c for c, n in group_counts.items() if n >= 1]
        if ensure_all_splits_have_all_classes:
            required_val = [c for c, n in group_counts.items() if n >= 2]
            required_test = [c for c, n in group_counts.items() if n >= 3]
        else:
            required_val = []
            required_test = []

        def _has_required(sub_df: pd.DataFrame, required) -> bool:
            if not required:
                return True
            counts = sub_df["_target"].value_counts().to_dict()
            return all(counts.get(c, 0) > 0 for c in required)

        train_df = val_df = test_df = None
        for attempt in range(max_tries):
            rs = seed + attempt
            strat = group_y if len(np.unique(group_y)) > 1 else None

            # First split off test groups
            try:
                train_val_g, test_g = train_test_split(
                    group_ids,
                    test_size=test_frac,
                    random_state=rs,
                    stratify=strat,
                )
            except ValueError:
                train_val_g, test_g = train_test_split(
                    group_ids, test_size=test_frac, random_state=rs, shuffle=True
                )

            # Get targets for train_val groups for stratification
            train_val_targets = group_targets.loc[train_val_g].values
            strat_train_val = train_val_targets if len(np.unique(train_val_targets)) > 1 else None

            # Then split train_val into train and val
            val_frac_adjusted = val_frac / (1 - test_frac)
            try:
                tr_g, va_g = train_test_split(
                    train_val_g,
                    test_size=val_frac_adjusted,
                    random_state=rs,
                    stratify=strat_train_val,
                )
            except ValueError:
                tr_g, va_g = train_test_split(
                    train_val_g, test_size=val_frac_adjusted, random_state=rs, shuffle=True
                )

            tr_df = df[df["accession"].isin(tr_g)]
            va_df = df[df["accession"].isin(va_g)]
            te_df = df[df["accession"].isin(test_g)]

            if (_has_required(tr_df, required_train) and
                _has_required(va_df, required_val) and
                _has_required(te_df, required_test)):
                train_df, val_df, test_df = tr_df.copy(), va_df.copy(), te_df.copy()
                break

        if train_df is None or val_df is None or test_df is None:
            # Fallback: use GroupShuffleSplit for test, then split remaining
            LOGGER.warning(
                "Nao foi possivel criar split com todas as classes em %d tentativas. "
                "Usando fallback GroupShuffleSplit.",
                max_tries
            )
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            train_val_idx, test_idx = next(splitter.split(df, df["_target"], groups=df["accession"]))
            test_df = df.iloc[test_idx].copy()
            train_val_df = df.iloc[train_val_idx].copy()

            # Split train_val
            val_frac_adjusted = val_frac / (1 - test_frac)
            splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_adjusted, random_state=seed)
            train_idx, val_idx = next(
                splitter2.split(train_val_df, train_val_df["_target"], groups=train_val_df["accession"])
            )
            train_df = train_val_df.iloc[train_idx].copy()
            val_df = train_val_df.iloc[val_idx].copy()

    def _counts(sub_df: pd.DataFrame) -> dict[int, int]:
        counts = sub_df["_target"].value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.items()}

    LOGGER.info(
        "3-way split criado: Train=%d (groups=%s) | Val=%d (groups=%s) | Test=%d (groups=%s) | "
        "Train counts=%s | Val counts=%s | Test counts=%s",
        len(train_df),
        train_df["accession"].nunique() if "accession" in train_df.columns else "NA",
        len(val_df),
        val_df["accession"].nunique() if "accession" in val_df.columns else "NA",
        len(test_df),
        test_df["accession"].nunique() if "accession" in test_df.columns else "NA",
        _counts(train_df),
        _counts(val_df),
        _counts(test_df),
    )

    train_df = train_df.drop(columns=["_target"])
    val_df = val_df.drop(columns=["_target"])
    test_df = test_df.drop(columns=["_target"])
    return train_df, val_df, test_df


def create_splits(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
    num_classes: int = 4,
    ensure_val_has_all_classes: bool = True,
    max_tries: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation splits with group-aware stratification when possible.

    Args:
        df: DataFrame with 'professional_label' and optionally 'accession' columns
        val_frac: Fraction of data to use for validation (default 0.2)
        seed: Random seed for reproducibility
        num_classes: Number of classes (2 or 4)
        ensure_val_has_all_classes: If True, ensure validation set has all classes
        max_tries: Maximum attempts to find valid split with all classes

    Returns:
        Tuple of (train_df, val_df)

    Raises:
        ValueError: If DataFrame is missing required columns
        RuntimeError: If no valid samples after label mapping or insufficient groups
    """
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


def create_kfold_splits(
    df: Optional[pd.DataFrame],
    n_splits: int = 5,
    seed: int = 42,
    num_classes: int = 4,
) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create k-fold cross-validation splits with group-aware stratification when possible.

    Args:
        df: DataFrame with 'professional_label' and optionally 'accession' columns.
            If None, returns empty list for validation purposes.
        n_splits: Number of folds (default 5)
        seed: Random seed for reproducibility
        num_classes: Number of classes (2 or 4)

    Returns:
        List of (train_df, val_df) tuples, one per fold

    Raises:
        ValueError: If DataFrame is missing required columns or n_splits < 2
        RuntimeError: If no valid samples after label mapping or insufficient groups
    """
    # Handle None input for validation
    if df is None:
        return []

    if n_splits < 2:
        raise ValueError(f"n_splits deve ser >= 2, recebido: {n_splits}")

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

    folds = []

    # Split without groups if accession is missing
    if "accession" not in df.columns or df["accession"].isna().all():
        LOGGER.warning("Coluna 'accession' ausente; k-fold sem agrupamento.")
        y = df["_target"].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            def _counts(sub_df: pd.DataFrame) -> dict[int, int]:
                counts = sub_df["_target"].value_counts().sort_index()
                return {int(k): int(v) for k, v in counts.items()}

            LOGGER.info(
                "Fold %d/%d: Train=%d | Val=%d | Train counts=%s | Val counts=%s",
                fold_idx + 1,
                n_splits,
                len(train_df),
                len(val_df),
                _counts(train_df),
                _counts(val_df),
            )

            train_df = train_df.drop(columns=["_target"])
            val_df = val_df.drop(columns=["_target"])
            folds.append((train_df, val_df))
    else:
        # Group-aware k-fold
        group_targets = df.groupby("accession")["_target"].agg(lambda s: s.value_counts().idxmax())
        unique_groups = group_targets.index.to_numpy()
        group_y = group_targets.values

        if len(unique_groups) < n_splits:
            raise RuntimeError(
                f"Grupos insuficientes para {n_splits} folds. "
                f"Requer pelo menos {n_splits} grupos, tem {len(unique_groups)}."
            )

        # Use StratifiedGroupKFold for seed-dependent, stratified splits
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(unique_groups, group_y, groups=unique_groups)):
            train_groups = unique_groups[train_idx]
            val_groups = unique_groups[val_idx]

            train_df = df[df["accession"].isin(train_groups)].copy()
            val_df = df[df["accession"].isin(val_groups)].copy()

            def _counts(sub_df: pd.DataFrame) -> dict[int, int]:
                counts = sub_df["_target"].value_counts().sort_index()
                return {int(k): int(v) for k, v in counts.items()}

            LOGGER.info(
                "Fold %d/%d: Train=%d (groups=%d) | Val=%d (groups=%d) | "
                "Train counts=%s | Val counts=%s",
                fold_idx + 1,
                n_splits,
                len(train_df),
                train_df["accession"].nunique(),
                len(val_df),
                val_df["accession"].nunique(),
                _counts(train_df),
                _counts(val_df),
            )

            train_df = train_df.drop(columns=["_target"])
            val_df = val_df.drop(columns=["_target"])
            folds.append((train_df, val_df))

    LOGGER.info("K-fold splits criados: %d folds", n_splits)
    return folds


def filter_by_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """Filter DataFrame to include only rows with specified mammography view.

    Args:
        df: DataFrame with 'view' column containing mammography view labels
        view: View label to filter by (e.g., 'CC', 'MLO')

    Returns:
        Filtered DataFrame containing only rows matching the specified view

    Raises:
        ValueError: If DataFrame is missing 'view' column
    """
    if "view" not in df.columns:
        raise ValueError("DataFrame deve conter a coluna 'view'.")

    filtered_df = df[df["view"] == view].copy()

    LOGGER.info(
        "Filtro de view aplicado: view=%s | Original=%d | Filtrado=%d",
        view,
        len(df),
        len(filtered_df),
    )

    return filtered_df
