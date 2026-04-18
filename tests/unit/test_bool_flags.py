from __future__ import annotations

import pytest

from mammography.utils.bool_flags import normalize_bool_flags, parse_bool_literal


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("off", False),
        ("0", False),
        ("no", False),
        ("maybe", None),
        (None, None),
    ],
)
def test_parse_bool_literal(value: str | None, expected: bool | None) -> None:
    assert parse_bool_literal(value) is expected


def test_normalize_bool_flags_rewrites_known_literals() -> None:
    assert normalize_bool_flags(
        [
            "--sampler-weighted",
            "false",
            "--augment",
            "true",
            "--epochs",
            "3",
        ]
    ) == ["--no-sampler-weighted", "--augment", "--epochs", "3"]


def test_normalize_bool_flags_leaves_unknown_literals() -> None:
    assert normalize_bool_flags(["--sampler-weighted", "auto"]) == [
        "--sampler-weighted",
        "auto",
    ]


def test_normalize_bool_flags_accepts_none() -> None:
    assert normalize_bool_flags(None) is None
