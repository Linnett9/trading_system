from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


class ResearchArtifactWriter:
    """Filesystem report adapter shared by research-only pipelines."""

    def write_json(self, path: Path, payload: Any) -> Path:
        self._ensure_parent(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_markdown(self, path: Path, content: str) -> Path:
        return self.write_text(path, content)

    def write_text(self, path: Path, content: str) -> Path:
        self._ensure_parent(path)
        path.write_text(content, encoding="utf-8")
        return path

    def write_csv(
        self,
        path: Path,
        rows: Iterable[dict[str, Any]],
        *,
        fieldnames: Sequence[str],
        extrasaction: str = "ignore",
    ) -> Path:
        self._ensure_parent(path)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction=extrasaction,
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
