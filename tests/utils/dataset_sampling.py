from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, TypeVar

DEFAULT_SAMPLE_RATIO = 0.05
DEFAULT_SAMPLE_SEED = 42

T = TypeVar("T")


def sample_size(total: int, ratio: float = DEFAULT_SAMPLE_RATIO) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    size = int(round(total * ratio))
    if size < 1:
        size = 1
    if size > total:
        size = total
    return size


def sample_sequence(
    items: Sequence[T],
    ratio: float = DEFAULT_SAMPLE_RATIO,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> List[T]:
    items_list = list(items)
    total = len(items_list)
    count = sample_size(total, ratio)
    if count == 0:
        return []
    if count == total:
        return items_list
    rng = random.Random(seed)
    return rng.sample(items_list, count)


def sample_paths_by_extension(
    root: Path,
    extensions: Iterable[str],
    ratio: float = DEFAULT_SAMPLE_RATIO,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> List[Path]:
    if not root.exists():
        return []
    ext_set = {ext.lower() for ext in extensions}
    matches: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in ext_set:
            matches.append(path)
    matches.sort()
    return sample_sequence(matches, ratio=ratio, seed=seed)


def sample_dataframe(
    df,
    ratio: float = DEFAULT_SAMPLE_RATIO,
    seed: int = DEFAULT_SAMPLE_SEED,
):
    total = len(df)
    count = sample_size(total, ratio)
    if count == 0 or count == total:
        return df
    return df.sample(n=count, random_state=seed)
