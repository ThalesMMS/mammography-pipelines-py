from __future__ import annotations

import pickle

from mammography.utils.class_modes import get_label_mapper


def test_binary_label_mapper_is_picklable() -> None:
    mapper = get_label_mapper("binary")

    payload = pickle.dumps(mapper)
    restored = pickle.loads(payload)

    assert restored(1) == 0
    assert restored(2) == 0
    assert restored(3) == 1
    assert restored(4) == 1


def test_multiclass_label_mapper_is_none() -> None:
    assert get_label_mapper("multiclass") is None
