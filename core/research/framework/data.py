from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _set_max_csv_field_size_limit() -> int:
    limit = sys.maxsize
    while True:
        try:
            return csv.field_size_limit(limit)
        except OverflowError:
            limit //= 10


class CsvRowRepository:
    def read(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        _set_max_csv_field_size_limit()
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


class JsonRepository:
    def read(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
