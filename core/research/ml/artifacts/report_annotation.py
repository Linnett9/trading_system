from __future__ import annotations

import csv
import json
from pathlib import Path


def annotate_report_artifacts(output_dir: Path, research_label: str) -> None:
    warning = (
        "Short-history ML smoke test only. Not valid for production conclusions."
        if research_label == "SMOKE_TEST_NOT_PRODUCTION_VALIDATED"
        else None
    )
    for path in output_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["research_label"] = research_label
            payload["production_validated"] = False
            if warning:
                payload["warning"] = warning
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for path in output_dir.glob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        if "research_label" not in fieldnames:
            fieldnames.append("research_label")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row["research_label"] = research_label
                writer.writerow(row)
