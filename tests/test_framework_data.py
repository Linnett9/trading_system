from __future__ import annotations

import csv
import sys

from core.research.framework import data


def test_csv_field_size_limit_falls_back_to_platform_accepted_value(monkeypatch):
    calls: list[int] = []
    accepted = sys.maxsize // 100

    def fake_field_size_limit(limit: int) -> int:
        calls.append(limit)
        if limit > accepted:
            raise OverflowError("Python int too large to convert to C long")
        return limit

    monkeypatch.setattr(data.csv, "field_size_limit", fake_field_size_limit)

    assert data._set_max_csv_field_size_limit() == accepted
    assert calls == [sys.maxsize, sys.maxsize // 10, accepted]


def test_csv_row_repository_reads_large_field(tmp_path):
    path = tmp_path / "rows.csv"
    large_value = "x" * (csv.field_size_limit() + 1)
    path.write_text(f"name,payload\nrow,{large_value}\n", encoding="utf-8")

    rows = data.CsvRowRepository().read(path)

    assert rows == [{"name": "row", "payload": large_value}]
