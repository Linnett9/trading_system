from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_FILENAME = "news_transformer_text_feature_readiness_report.json"
TEXT_COLUMNS = ("title", "summary_or_text")
REQUIRED_LABEL_COLUMNS = (
    "label_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_drawdown_20d",
    "reduce_exposure_label",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_text_feature_readiness_report_only(
    *,
    labeled_dataset_path: str | Path,
    baseline_report_path: str | Path,
    split_assignments_path: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    baseline_report = _read_json(Path(baseline_report_path))
    _validate_baseline_report(baseline_report)

    rows = _read_csv(Path(labeled_dataset_path))
    assignments = _read_csv(Path(split_assignments_path))
    rows_by_event_key = {str(row.get("event_key", "")): row for row in rows if row.get("event_key")}
    assigned_rows = [rows_by_event_key[row["event_key"]] for row in assignments if row.get("event_key") in rows_by_event_key]

    report = _report(
        rows=rows,
        assigned_rows=assigned_rows,
        assignments=assignments,
        baseline_report=baseline_report,
    )
    _write_report(output_dir_path, report)
    return report


def _validate_baseline_report(report: Mapping[str, Any]) -> None:
    guardrails = {
        "mode": report.get("mode") == "news_transformer_baseline_model_report_only",
        "research_only": report.get("research_only") is True,
        "trading_impact": report.get("trading_impact") == "none",
        "transformer_training_started": report.get("transformer_training_started") is False,
        "leakage_violation_count": int(report.get("leakage_violation_count", 0)) == 0,
        "overlap_violation_count": int(report.get("overlap_violation_count", 0)) == 0,
        "next_allowed_step": report.get("next_allowed_step") == "review_news_transformer_baseline_model_report",
    }
    failed = [name for name, passed in guardrails.items() if not passed]
    if failed:
        raise ValueError(f"baseline report is not approved for text readiness: {', '.join(failed)}")


def _report(
    *,
    rows: Sequence[Mapping[str, str]],
    assigned_rows: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, str]],
    baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    rows_with_required_labels = [row for row in rows if _has_required_labels(row)]
    text_stats = _text_stats(rows)
    blocking_reasons = _blocking_reasons(text_stats)
    warnings = _warnings(text_stats, baseline_report)
    return {
        "mode": "news_transformer_text_feature_readiness_report_only",
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "transformer_training_started": False,
        "rows_total": len(rows),
        "rows_with_required_labels": len(rows_with_required_labels),
        "assignment_rows": len(assignments),
        "assigned_rows_matched": len(assigned_rows),
        "fold_count": len({row.get("fold_id", "") for row in assignments if row.get("fold_id")}),
        "baseline_review": _baseline_review(baseline_report),
        "text_columns": text_stats,
        "fold_text_coverage": _fold_text_coverage(assignments, assigned_rows),
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "readiness_available": not blocking_reasons,
        "next_allowed_step": (
            "build_news_transformer_text_baseline_report_only"
            if not blocking_reasons
            else "resolve_news_transformer_text_feature_readiness_blockers"
        ),
    }


def _baseline_review(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(report.get("selected_model_average_test_metrics", {}) or {})
    balanced_accuracy = float(metrics.get("balanced_accuracy", 0.0))
    return {
        "fold_count": report.get("fold_count"),
        "models_evaluated": report.get("models_evaluated", []),
        "selected_model_counts": report.get("selected_model_counts", {}),
        "selected_model_average_test_metrics": metrics,
        "metadata_baseline_signal": (
            "weak"
            if balanced_accuracy < 0.55
            else "moderate_or_better"
        ),
    }


def _text_stats(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for column in TEXT_COLUMNS:
        values = [str(row.get(column, "")).strip() for row in rows]
        nonempty = [value for value in values if value]
        token_counts = [_token_count(value) for value in nonempty]
        stats[column] = {
            "rows_present": len(nonempty),
            "rows_missing": len(rows) - len(nonempty),
            "coverage": _safe_div(len(nonempty), len(rows)),
            "average_character_length": _mean([len(value) for value in nonempty]),
            "average_token_count": _mean(token_counts),
            "max_token_count": max(token_counts) if token_counts else 0,
            "unique_value_count": len(set(nonempty)),
        }
    combined_text = [_combined_text(row) for row in rows]
    combined_nonempty = [value for value in combined_text if value]
    token_counts = [_token_count(value) for value in combined_nonempty]
    vocabulary = Counter(token for value in combined_nonempty for token in _tokens(value))
    stats["combined_text"] = {
        "rows_present": len(combined_nonempty),
        "rows_missing": len(rows) - len(combined_nonempty),
        "coverage": _safe_div(len(combined_nonempty), len(rows)),
        "average_token_count": _mean(token_counts),
        "max_token_count": max(token_counts) if token_counts else 0,
        "unique_value_count": len(set(combined_nonempty)),
        "vocabulary_size": len(vocabulary),
        "top_tokens": [{"token": token, "count": count} for token, count in vocabulary.most_common(20)],
    }
    return stats


def _fold_text_coverage(assignments: Sequence[Mapping[str, str]], assigned_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rows_by_fold_split: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for assignment, row in zip(assignments, assigned_rows):
        rows_by_fold_split[(str(assignment.get("fold_id", "")), str(assignment.get("split", "")))].append(row)
    coverage: dict[str, dict[str, Any]] = {}
    for (fold_id, split), rows in sorted(rows_by_fold_split.items()):
        coverage.setdefault(fold_id, {})[split] = {
            "rows": len(rows),
            "combined_text_rows_present": sum(1 for row in rows if _combined_text(row)),
            "combined_text_coverage": _safe_div(sum(1 for row in rows if _combined_text(row)), len(rows)),
        }
    return coverage


def _blocking_reasons(text_stats: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if float(text_stats["combined_text"]["coverage"]) < 0.95:
        reasons.append("combined_text_coverage_below_minimum")
    if int(text_stats["combined_text"]["vocabulary_size"]) < 3:
        reasons.append("combined_text_vocabulary_too_small")
    return reasons


def _warnings(text_stats: Mapping[str, Any], baseline_report: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if int(text_stats["summary_or_text"]["rows_present"]) == 0:
        warnings.append("summary_or_text is empty for all rows; first text baseline will rely on titles only")
    metrics = baseline_report.get("selected_model_average_test_metrics", {}) or {}
    if float(metrics.get("balanced_accuracy", 0.0)) < 0.55:
        warnings.append("metadata baseline is weak; compare text baseline against it before transformer work")
    return warnings


def _has_required_labels(row: Mapping[str, str]) -> bool:
    return all(row.get(column) for column in REQUIRED_LABEL_COLUMNS)


def _combined_text(row: Mapping[str, str]) -> str:
    return " ".join(str(row.get(column, "")).strip() for column in TEXT_COLUMNS if str(row.get(column, "")).strip())


def _token_count(value: str) -> int:
    return len(_tokens(value))


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(value)]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build report-only text-feature readiness for labelled news transformer events.")
    parser.add_argument("--labeled-dataset", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--split-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args(argv)

    report = build_text_feature_readiness_report_only(
        labeled_dataset_path=args.labeled_dataset,
        baseline_report_path=args.baseline_report,
        split_assignments_path=args.split_assignments,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
