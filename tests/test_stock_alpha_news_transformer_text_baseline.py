import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_text_baseline import (
    PREDICTIONS_FILENAME,
    REPORT_FILENAME,
    build_text_baseline_report_only,
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


def _readiness_report(**overrides) -> dict:
    report = {
        "mode": "news_transformer_text_feature_readiness_report_only",
        "research_only": True,
        "trading_impact": "none",
        "transformer_training_started": False,
        "readiness_available": True,
        "blocking_reasons": [],
        "next_allowed_step": "build_news_transformer_text_baseline_report_only",
        "baseline_review": {
            "selected_model_average_test_metrics": {
                "balanced_accuracy": 0.51,
                "log_loss": 0.70,
            }
        },
        "text_columns": {
            "title": {"coverage": 1.0},
            "summary_or_text": {"coverage": 0.0},
            "combined_text": {"vocabulary_size": 10},
        },
        "warnings": ["summary_or_text is empty for all rows; first text baseline will rely on titles only"],
    }
    report.update(overrides)
    return report


def _labeled_rows() -> list[dict[str, str]]:
    specs = [
        ("event-1", "AAA", "risk warning filed by AAA", "true"),
        ("event-2", "AAA", "risk loss filed by AAA", "true"),
        ("event-3", "BBB", "growth filed by BBB", "false"),
        ("event-4", "BBB", "growth update filed by BBB", "false"),
        ("event-5", "AAA", "risk update filed by AAA", "true"),
        ("event-6", "BBB", "growth plan filed by BBB", "false"),
    ]
    rows = []
    for index, (event_key, symbol, title, label) in enumerate(specs, start=1):
        rows.append(
            {
                "event_key": event_key,
                "symbol": symbol,
                "title": title,
                "summary_or_text": "",
                "available_at_timestamp": f"2024-01-0{index}T12:00:00Z",
                "label_date": f"2024-01-0{index}",
                "future_return_1d": "0.0100000000",
                "future_return_5d": "0.0200000000",
                "future_return_20d": "-0.0300000000" if label == "true" else "0.0300000000",
                "future_drawdown_20d": "-0.0600000000" if label == "true" else "-0.0100000000",
                "reduce_exposure_label": label,
            }
        )
    return rows


def _assignments(leaky: bool = False) -> list[dict[str, str]]:
    split_by_event = {
        "event-1": "train",
        "event-2": "train",
        "event-3": "train",
        "event-4": "validation",
        "event-5": "test",
        "event-6": "test",
    }
    rows = []
    for row in _labeled_rows():
        available_at = "2024-01-09T12:00:00Z" if leaky and row["event_key"] == "event-5" else row["available_at_timestamp"]
        rows.append(
            {
                "fold_id": "fold_000",
                "split": split_by_event[row["event_key"]],
                "event_key": row["event_key"],
                "symbol": row["symbol"],
                "available_at_timestamp": available_at,
                "label_date": row["label_date"],
            }
        )
    return rows


def _run(tmp_path: Path, *, readiness=None, assignments=None, output_dir=None):
    reports_root = tmp_path / "reports"
    labeled_path = _write_csv(reports_root / "labeled" / "events.csv", _labeled_rows())
    readiness_path = _write_json(reports_root / "readiness" / "report.json", readiness or _readiness_report())
    assignments_path = _write_csv(reports_root / "splits" / "assignments.csv", assignments or _assignments())
    return build_text_baseline_report_only(
        labeled_dataset_path=labeled_path,
        text_readiness_report_path=readiness_path,
        split_assignments_path=assignments_path,
        output_dir=output_dir or reports_root / "text_baseline",
        reports_root=reports_root,
    )


def test_text_baseline_refuses_unapproved_readiness(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="next_allowed_step"):
        _run(tmp_path, readiness=_readiness_report(next_allowed_step="train_transformer"))

    with pytest.raises(ValueError, match="readiness_available"):
        _run(tmp_path, readiness=_readiness_report(readiness_available=False))


def test_text_baseline_uses_walk_forward_assignments_and_reports_metrics(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report["research_only"] is True
    assert report["trading_impact"] == "none"
    assert report["fold_count"] == 1
    assert report["models_evaluated"] == [
        "text_global_prior",
        "text_token_naive_bayes_min1",
        "text_token_naive_bayes_min5",
    ]
    assert report["folds"][0]["train_date_max"] == "2024-01-03"
    assert report["folds"][0]["validation_date_min"] == "2024-01-04"
    assert report["folds"][0]["test_date_min"] == "2024-01-05"
    assert report["selected_model_average_test_metrics"]["log_loss"] >= 0
    assert "balanced_accuracy_delta" in report["text_vs_metadata_baseline"]


def test_text_baseline_refuses_future_leakage_in_assignments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future leakage"):
        _run(tmp_path, assignments=_assignments(leaky=True))


def test_text_baseline_output_stays_under_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _run(tmp_path, output_dir=tmp_path / "outside")

    _run(tmp_path)
    output_dir = tmp_path / "reports" / "text_baseline"
    assert (output_dir / REPORT_FILENAME).exists()
    assert (output_dir / PREDICTIONS_FILENAME).exists()


def test_text_baseline_does_not_import_transformer_or_trading_paths() -> None:
    source = Path("scripts/stock_alpha_news_transformer_text_baseline.py").read_text(encoding="utf-8")

    assert "torch" not in source
    assert "sklearn" not in source
    assert "broker" not in source
    assert "paper_trading" not in source
    assert "live" not in source
