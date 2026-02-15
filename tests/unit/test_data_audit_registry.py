import csv
import json
from pathlib import Path

from mammography.tools import data_audit_registry as registry


def _write_manifest(path: Path) -> None:
    data = {
        "generated_at": "2025-02-01T00:00:00Z",
        "total_dicom_files": 10,
        "total_readable_files": 8,
        "class_histogram": {"1": 3, "2": 1, "missing": 2},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_audit_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "accession,classification,dicom_count,readable_count,unreadable_count,"
        "sample_checksum,notes\n",
        encoding="utf-8",
    )


def _write_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Audit log\n", encoding="utf-8")


def test_load_manifest_counts_and_format(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data_manifest.json"
    _write_manifest(manifest_path)
    counts = registry.load_manifest_counts(manifest_path)
    assert counts.total_dicom_files == 10
    assert counts.valid_dicom_files == 8
    assert counts.invalid_dicom_files == 2
    assert counts.class_histogram["1"] == 3
    distribution = registry.format_class_distribution(counts.class_histogram)
    assert distribution == "1=3, 2=1, missing=2"


def test_register_data_audit_writes_registry_and_mlflow_fallback(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "outputs" / "archive_audit" / "data_manifest.json"
    audit_csv_path = tmp_path / "outputs" / "archive_audit" / "data_audit.csv"
    log_path = tmp_path / "Article" / "assets" / "data_qc.log"
    _write_manifest(manifest_path)
    _write_audit_csv(audit_csv_path)
    _write_log(log_path)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_data_audit(
        manifest_path=manifest_path,
        audit_csv_path=audit_csv_path,
        log_path=log_path,
        dataset="archive",
        workflow="data-audit",
        run_name="archive_data_audit",
        command="mammography data-audit --archive archive",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    assert registry_csv.exists()
    assert registry_md.exists()

    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset"] == "archive"
    assert rows[0]["workflow"] == "data-audit"
    assert rows[0]["run_name"] == "archive_data_audit"
    assert rows[0]["total_dicom_files"] == "10"
    assert rows[0]["valid_dicom_files"] == "8"
    assert rows[0]["invalid_dicom_files"] == "2"
    assert rows[0]["class_distribution"] == "1=3, 2=1, missing=2"

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_data_audit" in md_text
    assert "Total DICOMs: 10" in md_text
    assert "Valid DICOMs: 8" in md_text
    assert "Invalid DICOMs: 2" in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / manifest_path.name).exists()
    assert (artifacts_dir / audit_csv_path.name).exists()
    assert (artifacts_dir / log_path.name).exists()
