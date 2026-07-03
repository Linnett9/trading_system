from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_LABEL_COLUMNS = (
    "label_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_drawdown_20d",
    "reduce_exposure_label",
)

REPORT_FILENAME = "news_transformer_walk_forward_splits_report.json"
ASSIGNMENTS_FILENAME = "news_transformer_walk_forward_split_assignments.csv"


def build_walk_forward_splits_report_only(
    *,
    labeled_dataset_path: str | Path,
    label_report_path: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
    train_dates: int = 504,
    validation_dates: int = 63,
    test_dates: int = 63,
    step_dates: int = 63,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    label_report = _read_json(Path(label_report_path))
    _validate_label_report(label_report)

    rows = _read_csv(Path(labeled_dataset_path))
    rows_with_required_labels = [row for row in rows if _has_required_labels(row)]
    rows_sorted = sorted(rows_with_required_labels, key=_split_sort_key)
    folds, assignments = _build_folds(
        rows_sorted,
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
        step_dates=step_dates,
    )
    leakage_violation_count = _leakage_violation_count(assignments)
    overlap_violation_count = _overlap_violation_count(folds)
    report = _report(
        rows=rows,
        rows_used_for_splits=len(rows_sorted),
        folds=folds,
        leakage_violation_count=leakage_violation_count,
        overlap_violation_count=overlap_violation_count,
    )
    _write_outputs(output_dir_path, report, assignments)
    return report


def _validate_label_report(report: Mapping[str, Any]) -> None:
    blocking_reasons = list(report.get("blocking_reasons") or [])
    guardrails = {
        "labels_attached": report.get("labels_attached") is True,
        "blocking_reasons": not blocking_reasons,
        "duplicate_event_key_count": int(report.get("duplicate_event_key_count", 0)) == 0,
        "future_timestamp_count": int(report.get("future_timestamp_count", 0)) == 0,
        "leakage_violation_count": int(report.get("leakage_violation_count", 0)) == 0,
    }
    failed = [name for name, passed in guardrails.items() if not passed]
    if failed:
        raise ValueError(f"label report is not approved for walk-forward splits: {', '.join(failed)}")


def _build_folds(
    rows: Sequence[Mapping[str, str]],
    *,
    train_dates: int,
    validation_dates: int,
    test_dates: int,
    step_dates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if min(train_dates, validation_dates, test_dates, step_dates) <= 0:
        raise ValueError("split window sizes must be positive")

    unique_dates = sorted({_parse_date(str(row["label_date"])) for row in rows})
    folds: list[dict[str, Any]] = []
    assignments: list[dict[str, str]] = []
    fold_index = 0
    start = train_dates
    while start + validation_dates + test_dates <= len(unique_dates):
        train_set = set(unique_dates[:start])
        validation_set = set(unique_dates[start:start + validation_dates])
        test_set = set(unique_dates[start + validation_dates:start + validation_dates + test_dates])

        train_rows = [row for row in rows if _parse_date(str(row["label_date"])) in train_set]
        validation_rows = [row for row in rows if _parse_date(str(row["label_date"])) in validation_set]
        test_rows = [row for row in rows if _parse_date(str(row["label_date"])) in test_set]
        if train_rows and validation_rows and test_rows:
            fold_id = f"fold_{fold_index:03d}"
            folds.append(_fold_summary(fold_id, train_rows, validation_rows, test_rows))
            assignments.extend(_assignment_rows(fold_id, "train", train_rows))
            assignments.extend(_assignment_rows(fold_id, "validation", validation_rows))
            assignments.extend(_assignment_rows(fold_id, "test", test_rows))
            fold_index += 1
        start += step_dates
    return folds, assignments


def _fold_summary(
    fold_id: str,
    train_rows: Sequence[Mapping[str, str]],
    validation_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "train_date_min": _date_min(train_rows),
        "train_date_max": _date_max(train_rows),
        "validation_date_min": _date_min(validation_rows),
        "validation_date_max": _date_max(validation_rows),
        "test_date_min": _date_min(test_rows),
        "test_date_max": _date_max(test_rows),
        "symbols": sorted({str(row.get("symbol", "")) for row in train_rows + validation_rows + test_rows if row.get("symbol")}),
    }


def _assignment_rows(fold_id: str, split: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "fold_id": fold_id,
            "split": split,
            "event_key": str(row.get("event_key", "")),
            "symbol": str(row.get("symbol", "")),
            "available_at_timestamp": str(row.get("available_at_timestamp", "")),
            "label_date": str(row.get("label_date", "")),
        }
        for row in rows
    ]


def _report(
    *,
    rows: Sequence[Mapping[str, str]],
    rows_used_for_splits: int,
    folds: Sequence[Mapping[str, Any]],
    leakage_violation_count: int,
    overlap_violation_count: int,
) -> dict[str, Any]:
    return {
        "rows_total": len(rows),
        "rows_used_for_splits": rows_used_for_splits,
        "rows_excluded_from_splits": len(rows) - rows_used_for_splits,
        "fold_count": len(folds),
        "folds": list(folds),
        "train_date_min": _field_min(folds, "train_date_min"),
        "train_date_max": _field_max(folds, "train_date_max"),
        "validation_date_min": _field_min(folds, "validation_date_min"),
        "validation_date_max": _field_max(folds, "validation_date_max"),
        "test_date_min": _field_min(folds, "test_date_min"),
        "test_date_max": _field_max(folds, "test_date_max"),
        "leakage_violation_count": leakage_violation_count,
        "overlap_violation_count": overlap_violation_count,
        "symbols_per_fold": {str(fold["fold_id"]): len(fold["symbols"]) for fold in folds},
        "next_allowed_step": "build_news_transformer_baseline_model_report_only",
    }


def _leakage_violation_count(assignments: Sequence[Mapping[str, str]]) -> int:
    violations = 0
    for row in assignments:
        available_at = _parse_timestamp(row.get("available_at_timestamp", ""))
        label_date = _parse_date(str(row.get("label_date", "")))
        if available_at and label_date < available_at.date():
            violations += 1
    return violations


def _overlap_violation_count(folds: Sequence[Mapping[str, Any]]) -> int:
    violations = 0
    for fold in folds:
        train_max = _parse_date(str(fold["train_date_max"]))
        validation_min = _parse_date(str(fold["validation_date_min"]))
        validation_max = _parse_date(str(fold["validation_date_max"]))
        test_min = _parse_date(str(fold["test_date_min"]))
        if train_max >= validation_min:
            violations += 1
        if validation_max >= test_min:
            violations += 1
    return violations


def _has_required_labels(row: Mapping[str, str]) -> bool:
    return all(row.get(column) for column in REQUIRED_LABEL_COLUMNS)


def _split_sort_key(row: Mapping[str, str]) -> tuple[date, datetime, str]:
    return (
        _parse_date(str(row["label_date"])),
        _parse_timestamp(str(row.get("available_at_timestamp", ""))) or datetime.min.replace(tzinfo=timezone.utc),
        str(row.get("event_key", "")),
    )


def _date_min(rows: Sequence[Mapping[str, str]]) -> str:
    return min(str(row["label_date"]) for row in rows)


def _date_max(rows: Sequence[Mapping[str, str]]) -> str:
    return max(str(row["label_date"]) for row in rows)


def _field_min(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    values = [str(row[field]) for row in rows if row.get(field)]
    return min(values) if values else None


def _field_max(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    values = [str(row[field]) for row in rows if row.get(field)]
    return max(values) if values else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(output_dir: Path, report: Mapping[str, Any], assignments: Sequence[Mapping[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / ASSIGNMENTS_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["fold_id", "split", "event_key", "symbol", "available_at_timestamp", "label_date"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build report-only walk-forward splits for labelled news transformer events.")
    parser.add_argument("--labeled-dataset", required=True)
    parser.add_argument("--label-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--train-dates", type=int, default=504)
    parser.add_argument("--validation-dates", type=int, default=63)
    parser.add_argument("--test-dates", type=int, default=63)
    parser.add_argument("--step-dates", type=int, default=63)
    args = parser.parse_args(argv)

    report = build_walk_forward_splits_report_only(
        labeled_dataset_path=args.labeled_dataset,
        label_report_path=args.label_report,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
        train_dates=args.train_dates,
        validation_dates=args.validation_dates,
        test_dates=args.test_dates,
        step_dates=args.step_dates,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
