from __future__ import annotations

import csv
import json

from core.research.ml.artifacts.report_annotation import annotate_report_artifacts


def test_report_annotation_adds_research_metadata_to_json(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")

    annotate_report_artifacts(tmp_path, "UNIT_RESEARCH")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "accuracy": 0.5,
        "research_label": "UNIT_RESEARCH",
        "production_validated": False,
    }


def test_report_annotation_adds_research_label_to_csv(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_csv(path, ["date", "prediction"], [{"date": "2024-01-01", "prediction": "1"}])

    annotate_report_artifacts(tmp_path, "UNIT_RESEARCH")

    rows = _read_csv(path)
    assert rows == [
        {
            "date": "2024-01-01",
            "prediction": "1",
            "research_label": "UNIT_RESEARCH",
        }
    ]


def test_report_annotation_preserves_existing_csv_columns(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_csv(
        path,
        ["date", "prediction", "research_label"],
        [{"date": "2024-01-01", "prediction": "1", "research_label": "OLD"}],
    )

    annotate_report_artifacts(tmp_path, "UNIT_RESEARCH")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames == ["date", "prediction", "research_label"]
    assert rows == [
        {
            "date": "2024-01-01",
            "prediction": "1",
            "research_label": "UNIT_RESEARCH",
        }
    ]


def test_report_annotation_adds_smoke_test_warning_to_json(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"model_type": "noop"}), encoding="utf-8")

    annotate_report_artifacts(tmp_path, "SMOKE_TEST_NOT_PRODUCTION_VALIDATED")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["research_label"] == "SMOKE_TEST_NOT_PRODUCTION_VALIDATED"
    assert payload["production_validated"] is False
    assert payload["warning"] == (
        "Short-history ML smoke test only. Not valid for production conclusions."
    )


def _write_csv(
    path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
