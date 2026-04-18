from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mammography.commands import train as train_command
from mammography.commands import train_data


class _StopAfterSplit(RuntimeError):
    """Raised by test doubles to stop train.main after split dispatch."""


def _patch_train_prelude(
    monkeypatch: pytest.MonkeyPatch,
    *,
    df: pd.DataFrame,
) -> None:
    csv_path = Path("dummy.csv")

    monkeypatch.setattr(
        train_command,
        "resolve_paths_from_preset",
        lambda csv, dataset, dicom_root: (str(csv_path), None),
    )
    monkeypatch.setattr(
        train_command.TrainConfig,
        "from_args",
        lambda args, csv=None, dicom_root=None: SimpleNamespace(csv=csv, dicom_root=dicom_root),
    )
    monkeypatch.setattr(train_command, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_command, "increment_path", lambda path: str(path))
    monkeypatch.setattr(
        train_command,
        "setup_logging",
        lambda outdir, log_level: logging.getLogger("test-train-split-modes"),
    )
    monkeypatch.setattr(
        train_command,
        "resolve_device",
        lambda device: SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr(train_command, "configure_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_command,
        "GracefulKiller",
        lambda register_signals=False: SimpleNamespace(kill_now=False),
    )
    monkeypatch.setattr(train_command, "load_dataset_dataframe", lambda *args, **kwargs: df)


def test_main_random_split_passes_group_col_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(12)],
            "professional_label": [1, 2, 3, 4] * 3,
            "accession": [f"ACC{idx:03d}" for idx in range(12)],
            "patient_id": [f"PAT{idx // 2:03d}" for idx in range(12)],
        }
    )
    _patch_train_prelude(monkeypatch, df=df)

    def _fake_create_splits(*args, **kwargs):
        assert kwargs["group_col"] is None
        raise _StopAfterSplit()

    monkeypatch.setattr(train_command, "create_splits", _fake_create_splits)

    with pytest.raises(_StopAfterSplit):
        train_command.main(
            [
                "--csv",
                "dummy.csv",
                "--outdir",
                str(tmp_path / "run"),
                "--split-mode",
                "random",
            ]
        )


def test_main_patient_split_passes_patient_group_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(12)],
            "professional_label": [1, 2, 3, 4] * 3,
            "accession": [f"ACC{idx:03d}" for idx in range(12)],
            "patient_id": [f"PAT{idx // 2:03d}" for idx in range(12)],
        }
    )
    _patch_train_prelude(monkeypatch, df=df)

    def _fake_create_splits(*args, **kwargs):
        assert kwargs["group_col"] == "patient_id"
        raise _StopAfterSplit()

    monkeypatch.setattr(train_command, "create_splits", _fake_create_splits)

    with pytest.raises(_StopAfterSplit):
        train_command.main(
            [
                "--csv",
                "dummy.csv",
                "--outdir",
                str(tmp_path / "run"),
                "--split-mode",
                "patient",
            ]
        )


def test_patient_column_leakage_check_skips_dicom_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "image_path": [f"img_{idx}.dcm" for idx in range(12)],
            "professional_label": [1, 2, 3, 4] * 3,
            "patient_id": [f"PAT{idx // 2:03d}" for idx in range(12)],
        }
    )
    _patch_train_prelude(monkeypatch, df=df)

    def _fake_create_splits(*args, **kwargs):
        return (
            df.iloc[:8].reset_index(drop=True),
            df.iloc[8:].reset_index(drop=True),
        )

    def _fail_if_dicom_read(*args, **kwargs):
        raise AssertionError("DICOM preflight should not run for column leakage checks")

    def _fake_assert_no_patient_leakage(train_patients, val_patients, strict, logger):
        assert train_patients == {f"PAT{idx // 2:03d}" for idx in range(8)}
        assert val_patients == {f"PAT{idx // 2:03d}" for idx in range(8, 12)}
        assert strict is True
        raise _StopAfterSplit()

    monkeypatch.setattr(train_command, "create_splits", _fake_create_splits)
    monkeypatch.setattr(train_data.pydicom, "dcmread", _fail_if_dicom_read)
    monkeypatch.setattr(
        train_command,
        "_assert_no_patient_leakage",
        _fake_assert_no_patient_leakage,
    )

    with pytest.raises(_StopAfterSplit):
        train_command.main(
            [
                "--csv",
                "dummy.csv",
                "--outdir",
                str(tmp_path / "run"),
                "--split-mode",
                "patient",
            ]
        )


def test_main_patient_split_requires_patient_id_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(12)],
            "professional_label": [1, 2, 3, 4] * 3,
            "accession": [f"ACC{idx:03d}" for idx in range(12)],
        }
    )
    _patch_train_prelude(monkeypatch, df=df)

    with pytest.raises(SystemExit, match="split-mode=patient requer coluna"):
        train_command.main(
            [
                "--csv",
                "dummy.csv",
                "--outdir",
                str(tmp_path / "run"),
                "--split-mode",
                "patient",
            ]
        )


def test_main_random_three_way_split_skips_patient_leakage_for_nondicom_without_patient_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(12)],
            "professional_label": [1, 2, 3, 4] * 3,
            "accession": [f"ACC{idx:03d}" for idx in range(12)],
        }
    )
    _patch_train_prelude(monkeypatch, df=df)

    def _fake_create_three_way_split(*args, **kwargs):
        assert kwargs["group_col"] is None
        return (
            df.iloc[:8].reset_index(drop=True),
            df.iloc[8:10].reset_index(drop=True),
            df.iloc[10:].reset_index(drop=True),
        )

    def _fake_assert_no_patient_leakage(train_patients, val_patients, strict, logger):
        assert train_patients == set()
        assert val_patients == set()
        assert strict is False
        raise _StopAfterSplit()

    monkeypatch.setattr(train_command, "create_three_way_split", _fake_create_three_way_split)
    monkeypatch.setattr(train_command, "_assert_no_patient_leakage", _fake_assert_no_patient_leakage)

    with pytest.raises(_StopAfterSplit):
        train_command.main(
            [
                "--csv",
                "dummy.csv",
                "--outdir",
                str(tmp_path / "run"),
                "--split-mode",
                "random",
                "--test-frac",
                "0.1",
            ]
        )


def test_patient_ids_from_dicom_warns_in_non_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    train_df = pd.DataFrame({"image_path": ["train_ok.dcm", "train_bad.dcm"]})
    val_df = pd.DataFrame({"image_path": ["val_ok.dcm"]})
    test_df = pd.DataFrame({"image_path": ["test_missing.dcm"]})

    def _fake_dcmread(path, *args, **kwargs):
        if path == "train_bad.dcm":
            raise OSError("unreadable")
        patient_id = None if path == "test_missing.dcm" else f"PAT-{path}"
        return SimpleNamespace(PatientID=patient_id)

    monkeypatch.setattr(train_data.pydicom, "dcmread", _fake_dcmread)

    with caplog.at_level(logging.WARNING):
        train_patients, val_patients, test_patients = train_data._patient_ids_from_dicom(
            train_df,
            val_df,
            logging.getLogger("test-train-split-modes"),
            test_df,
            strict=False,
        )

    assert train_patients == {"PAT-train_ok.dcm"}
    assert val_patients == {"PAT-val_ok.dcm"}
    assert test_patients == set()
    assert "falha ao ler cabecalhos DICOM" in caplog.text
    assert "PatientID ausente" in caplog.text
