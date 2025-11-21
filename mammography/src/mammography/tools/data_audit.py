#!/usr/bin/env python3
"""Utility script that inventories the archive/ directory and emits audit artifacts.

Outputs:
  - data_manifest.json: summary of accession coverage and class distribution.
  - Article/assets/data_qc.log: human-readable snapshot for the report.
  - outputs/embeddings_resnet50/data_audit.csv: per-accession quality table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import pydicom  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(f"pydicom não disponível: {exc}") from exc


ACCEPTED_CLASSES = {"1", "2", "3", "4"}


@dataclass
class AccessionAudit:
    accession: str
    classification: str | None
    dicom_count: int
    readable_count: int
    unreadable_count: int
    sample_checksum: str | None
    notes: str = ""

    @property
    def ok(self) -> bool:
        if self.dicom_count == 0:
            return False
        if self.unreadable_count > 0:
            return False
        return True


def _parse_csv(csv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            acc = row.get("AccessionNumber")
            cls = row.get("Classification")
            if not acc or not cls:
                continue
            mapping[acc.zfill(6)] = cls.strip()
    return mapping


def _read_dicom(path: Path) -> bool:
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def _hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _iter_dicom_paths(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*.dcm")):
        if path.is_file():
            yield path


def audit_accessions(root: Path, csv_map: dict[str, str]) -> list[AccessionAudit]:
    audits: list[AccessionAudit] = []
    for acc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        accession = acc_dir.name
        dicom_paths = list(_iter_dicom_paths(acc_dir))
        readable = 0
        for dicom_path in dicom_paths:
            if _read_dicom(dicom_path):
                readable += 1
        checksum = _hash_file(dicom_paths[0]) if dicom_paths else None
        classification = csv_map.get(accession)
        notes = ""
        if classification not in ACCEPTED_CLASSES:
            notes = f"Classe inválida: {classification}" if classification else "Sem classificação."
        audits.append(
            AccessionAudit(
                accession=accession,
                classification=classification,
                dicom_count=len(dicom_paths),
                readable_count=readable,
                unreadable_count=len(dicom_paths) - readable,
                sample_checksum=checksum,
                notes=notes,
            )
        )
    return audits


def _write_manifest(audits: list[AccessionAudit], csv_map: dict[str, str], out_path: Path) -> None:
    coverage = Counter(a.classification or "missing" for a in audits)
    total_readable = sum(a.readable_count for a in audits)
    total_files = sum(a.dicom_count for a in audits)
    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_accessions": len(audits),
        "total_csv_entries": len(csv_map),
        "exams_missing_dir": sorted(set(csv_map) - {a.accession for a in audits}),
        "dirs_missing_csv": sorted(a.accession for a in audits if a.classification is None),
        "class_histogram": coverage,
        "total_dicom_files": total_files,
        "total_readable_files": total_readable,
        "readability_ratio": total_readable / total_files if total_files else 0.0,
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_audit_csv(audits: list[AccessionAudit], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "accession",
                "classification",
                "dicom_count",
                "readable_count",
                "unreadable_count",
                "sample_checksum",
                "notes",
            ],
        )
        writer.writeheader()
        for audit in audits:
            writer.writerow(asdict(audit))


def _write_log(audits: list[AccessionAudit], log_path: Path) -> None:
    readable_exams = sum(1 for a in audits if a.ok)
    total = len(audits)
    log_lines = [
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        f"Exames na pasta archive/: {total}",
        f"Exames com DICOMs lidos: {readable_exams} ({(readable_exams/total*100 if total else 0):.1f}%)",
    ]
    top_notes = [a for a in audits if a.notes][:10]
    if top_notes:
        log_lines.append("Observações:")
        for item in top_notes:
            log_lines.append(f"- {item.accession}: {item.notes}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita DICOMs no diretório archive/.")
    parser.add_argument("--archive", type=Path, default=Path("archive"))
    parser.add_argument("--csv", type=Path, default=Path("classificacao.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest.json"))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/embeddings_resnet50/data_audit.csv"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("Article/assets/data_qc.log"),
    )
    args = parser.parse_args()

    if not args.archive.exists():
        print(f"Diretório não encontrado: {args.archive}", file=sys.stderr)
        return 1
    csv_map = _parse_csv(args.csv)
    audits = audit_accessions(args.archive, csv_map)
    _write_manifest(audits, csv_map, args.manifest)
    _write_audit_csv(audits, args.audit_csv)
    _write_log(audits, args.log)
    total_files = sum(a.dicom_count for a in audits)
    total_readable = sum(a.readable_count for a in audits)
    ratio = total_readable / total_files * 100 if total_files else 0.0
    print(f"Arquivos DICOM lidos: {total_readable}/{total_files} ({ratio:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
