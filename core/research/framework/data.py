from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class CsvRowRepository:
    def read(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


class JsonRepository:
    def read(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
