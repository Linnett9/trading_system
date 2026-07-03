import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_text_feature_readiness import (
    REPORT_FILENAME,
    build_text_feature_readiness_report_only,
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


def _baseline_report(**overrides) -> dict:
    report = {
        "mode": "news_transformer_baseline_model_report_only",
        "research_only": True,
        "trading_impact": "none",
        "transformer_training_started": False,
        "leakage_violation_count": 0,
        "overlap_violation_count": 0,
        "fold_count": 1,
        "models_evaluated": ["global_prior", "symbol_prior", "symbol_form_prior"],
        "selected_model_counts": {"global_prior": 1},
        "selected_model_average_test_metrics": {"balanced_accuracy": 0.51, "log_loss": 0.70},
        "next_allowed_step": "review_news_transformer_baseline_model_report",
    }
    report.update(overrides)
    return report


def _rows(summary: str = "") -> list[dict[str, str]]:
    return [
        {
            "event_key": f"event-{index}",
            "symbol": "AAA",
            "title": f"{form} filed by AAA",
            "summary_or_text": summary,
            "available_at_timestamp": f"2024-01-0{index}T12:00:00Z",
            "label_date": f"2024-01-0{index}",
            "future_return_1d": "0.0100000000",
            "future_return_5d": "0.0200000000",
            "future_return_20d": "0.0300000000",
            "future_drawdown_20d": "-0.0100000000",
            "reduce_exposure_label": "false",
        }
        for index, form in enumerate(["8-K", "10-Q", "10-K"], start=1)
    ]


def _assignments() -> list[dict[str, str]]:
    return [
        {
            "fold_id": "fold_000",
            "split": split,
            "event_key": f"event-{index}",
            "symbol": "AAA",
            "available_at_timestamp": f"2024-01-0{index}T12:00:00Z",
            "label_date": f"2024-01-0{index}",
        }
        for index, split in enumerate(["train", "validation", "test"], start=1)
    ]


def _run(tmp_path: Path, *, baseline=None, rows=None, output_dir=None):
    reports_root = tmp_path / "reports"
    rows_path = _write_csv(reports_root / "labeled" / "events.csv", rows or _rows())
    baseline_path = _write_json(reports_root / "baseline" / "report.json", baseline or _baseline_report())
    assignments_path = _write_csv(reports_root / "splits" / "assignments.csv", _assignments())
    return build_text_feature_readiness_report_only(
        labeled_dataset_path=rows_path,
        baseline_report_path=baseline_path,
        split_assignments_path=assignments_path,
        output_dir=output_dir or reports_root / "text_readiness",
        reports_root=reports_root,
    )


def test_text_readiness_refuses_unapproved_baseline_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="next_allowed_step"):
        _run(tmp_path, baseline=_baseline_report(next_allowed_step="train_transformer"))

    with pytest.raises(ValueError, match="transformer_training_started"):
        _run(tmp_path, baseline=_baseline_report(transformer_training_started=True))


def test_text_readiness_accepts_title_only_dataset_with_warning(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report["readiness_available"] is True
    assert report["text_columns"]["title"]["coverage"] == 1.0
    assert report["text_columns"]["summary_or_text"]["coverage"] == 0.0
    assert report["text_columns"]["combined_text"]["coverage"] == 1.0
    assert "summary_or_text is empty for all rows; first text baseline will rely on titles only" in report["warnings"]
    assert report["baseline_review"]["metadata_baseline_signal"] == "weak"
    assert report["next_allowed_step"] == "build_news_transformer_text_baseline_report_only"


def test_text_readiness_blocks_when_combined_text_is_missing(tmp_path: Path) -> None:
    rows = _rows()
    for row in rows:
        row["title"] = ""
    report = _run(tmp_path, rows=rows)

    assert report["readiness_available"] is False
    assert "combined_text_coverage_below_minimum" in report["blocking_reasons"]


def test_text_readiness_output_stays_under_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _run(tmp_path, output_dir=tmp_path / "outside")

    _run(tmp_path)
    assert (tmp_path / "reports" / "text_readiness" / REPORT_FILENAME).exists()


def test_text_readiness_does_not_import_transformer_or_trading_paths() -> None:
    source = Path("scripts/stock_alpha_news_transformer_text_feature_readiness.py").read_text(encoding="utf-8")

    assert "torch" not in source
    assert "sklearn" not in source
    assert "broker" not in source
    assert "paper_trading" not in source
    assert "live" not in source
