import pytest

from mammography.commands import train as train_command


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_parse_args_accepts_sampler_weighted_literals(value: str, expected: bool) -> None:
    args = train_command.parse_args(["--sampler-weighted", value])
    assert args.sampler_weighted is expected


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_parse_args_accepts_unfreeze_last_block_literals(value: str, expected: bool) -> None:
    args = train_command.parse_args(["--unfreeze-last-block", value])
    assert args.unfreeze_last_block is expected


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_parse_args_accepts_augment_literals(value: str, expected: bool) -> None:
    args = train_command.parse_args(["--augment", value])
    assert args.augment is expected
