from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_LINES = 1000


def test_python_files_stay_under_line_budget() -> None:
    oversized: list[tuple[str, int]] = []
    for root_name in ("src", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            line_count = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
            if line_count > MAX_LINES:
                oversized.append((str(path.relative_to(ROOT)), line_count))

    assert oversized == []
