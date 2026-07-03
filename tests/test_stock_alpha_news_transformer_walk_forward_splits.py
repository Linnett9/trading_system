import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_walk_forward_splits import (
    ASSIGNMENTS_FILENAME,
    REPORT_FILENAME,
    build_walk_forward_splits_report_only,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _approved_label_report(**overrides) -> dict:
    report = {
        "labels_attached": True,
        "blocking_reasons": [],
        "duplicate_event_key_count": 0,
        "future_timestamp_count": 0,
        "leakage_violation_count": 0,
    }
    report.update(overrides)
    return report


def _labeled_rows(count: int = 10) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        day = index + 1
        rows.append(
            {
                "event_key": f"event-{index:03d}",
                "symbol": "AAA" if index % 2 == 0 else "BBB",
                "available_at_timestamp": f"2024-01-{day:02d}T13:00:00Z",
                "label_date": f"2024-01-{day:02d}",
                "future_return_1d": "0.0100000000",
                "future_return_5d": "0.0200000000",
                "future_return_20d": "0.0300000000",
                "future_drawdown_20d": "-0.0100000000",
                "reduce_exposure_label": "false",
            }
        )
    return rows


def _run(tmp_path: Path, *, rows=None, report=None, output_dir=None):
    reports_root = tmp_path / "reports"
    labeled_path = _write_csv(reports_root / "labeled" / "events.csv", rows or _labeled_rows())
    report_path = _write_json(reports_root / "labeled" / "report.json", report or _approved_label_report())
    return build_walk_forward_splits_report_only(
        labeled_dataset_path=labeled_path,
        label_report_path=report_path,
        output_dir=output_dir or reports_root / "splits",
        reports_root=reports_root,
        train_dates=3,
        validation_dates=2,
        test_dates=2,
        step_dates=2,
    )


def test_split_builder_refuses_unapproved_or_unlabelled_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labels_attached"):
        _run(tmp_path, report=_approved_label_report(labels_attached=False))

    with pytest.raises(ValueError, match="blocking_reasons"):
        _run(tmp_path, report=_approved_label_report(blocking_reasons=["duplicate_event_keys"]))


def test_split_builder_uses_chronological_splits_only(tmp_path: Path) -> None:
    report = _run(tmp_path, rows=list(reversed(_labeled_rows())))

    assert report["fold_count"] == 2
    first = report["folds"][0]
    assert first["train_date_min"] == "2024-01-01"
    assert first["train_date_max"] == "2024-01-03"
    assert first["validation_date_min"] == "2024-01-04"
    assert first["validation_date_max"] == "2024-01-05"
    assert first["test_date_min"] == "2024-01-06"
    assert first["test_date_max"] == "2024-01-07"


def test_split_builder_reports_no_date_overlap_between_splits(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report["overlap_violation_count"] == 0
    for fold in report["folds"]:
        assert fold["train_date_max"] < fold["validation_date_min"]
        assert fold["validation_date_max"] < fold["test_date_min"]


def test_split_builder_reports_no_future_leakage(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report["leakage_violation_count"] == 0


def test_split_builder_output_stays_under_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _run(tmp_path, output_dir=tmp_path / "outside")

    report = _run(tmp_path)
    output_dir = tmp_path / "reports" / "splits"
    assert (output_dir / REPORT_FILENAME).exists()
    assert (output_dir / ASSIGNMENTS_FILENAME).exists()
    assert report["rows_total"] == 10
    assert report["rows_used_for_splits"] == 10


def test_split_builder_does_not_emit_training_or_transformer_paths(tmp_path: Path) -> None:
    report = _run(tmp_path)
    rendered = json.dumps(report).lower()
    assignments = (tmp_path / "reports" / "splits" / ASSIGNMENTS_FILENAME).read_text(encoding="utf-8").lower()

    assert "model_training" not in rendered
    assert "transformer_training" not in rendered
    assert "broker" not in rendered
    assert "paper" not in rendered
    assert "live" not in rendered
    assert "transformer" not in assignments
