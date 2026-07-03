import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_attach_price_labels import attach_price_labels_report_only


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _event_rows(**overrides: str) -> list[dict[str, str]]:
    row = {
        "event_id": "event-1",
        "event_key": "sec|AAA|url|2024-01-02T13:00:00Z",
        "symbol": "AAA",
        "available_at_timestamp": "2024-01-02T13:00:00Z",
        "title": "10-Q filed by AAA",
    }
    row.update(overrides)
    return [row]


def _price_rows(symbol: str = "AAA") -> list[dict[str, str]]:
    closes = [100, 102, 101, 104, 105, 110, 108, 107, 106, 104, 103, 102, 99, 98, 97, 96, 95, 94, 93, 92, 91]
    return [
        {"symbol": symbol, "date": f"2024-01-{index + 2:02d}", "close": str(close)}
        for index, close in enumerate(closes)
    ]


def _run(tmp_path: Path, *, events=None, prices=None, output_dir=None):
    reports = tmp_path / "reports"
    event_path = _write_csv(reports / "features" / "events.csv", events or _event_rows())
    price_path = _write_csv(tmp_path / "prices.csv", prices or _price_rows())
    return attach_price_labels_report_only(
        event_dataset_path=event_path,
        price_csv_path=price_path,
        output_dir=output_dir or reports / "labeled",
        reports_root=reports,
    )


def test_refuses_duplicate_event_keys_and_future_timestamps(tmp_path: Path) -> None:
    duplicate = _event_rows()[0]
    report = _run(tmp_path, events=[duplicate, dict(duplicate)])

    assert report["labels_attached"] is False
    assert report["duplicate_event_key_count"] == 1
    assert "duplicate_event_keys" in report["blocking_reasons"]

    future_report = _run(
        tmp_path,
        events=_event_rows(available_at_timestamp="2999-01-01T00:00:00Z"),
    )
    assert future_report["future_timestamp_count"] == 1
    assert "future_timestamps" in future_report["blocking_reasons"]


def test_joins_next_valid_trading_date_and_computes_returns_and_drawdown(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        events=_event_rows(available_at_timestamp="2024-01-01T22:00:00Z"),
    )
    labeled_path = tmp_path / "reports" / "labeled" / "news_transformer_event_features_labeled.csv"
    row = next(csv.DictReader(labeled_path.open(encoding="utf-8")))

    assert report["labels_attached"] is True
    assert row["label_date"] == "2024-01-02"
    assert row["future_return_1d"] == "0.0200000000"
    assert row["future_return_5d"] == "0.1000000000"
    assert row["future_return_20d"] == "-0.0900000000"
    assert row["future_drawdown_20d"] == "-0.0900000000"
    assert row["reduce_exposure_label"] == "true"
    assert report["next_allowed_step"] == "build_news_transformer_walk_forward_splits_report_only"


def test_emits_blocking_report_when_price_loader_is_missing(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    event_path = _write_csv(reports / "features" / "events.csv", _event_rows())

    report = attach_price_labels_report_only(
        event_dataset_path=event_path,
        output_dir=reports / "labeled",
        reports_root=reports,
    )

    assert report["labels_attached"] is False
    assert report["blocking_reasons"] == ["price_loader_not_found"]
    assert report["next_allowed_step"] == "implement_or_select_canonical_price_loader"


def test_attaches_labels_from_adjusted_price_directory(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    event_path = _write_csv(
        reports / "features" / "events.csv",
        _event_rows(available_at_timestamp="2024-01-01T22:00:00Z"),
    )
    adjusted_dir = tmp_path / "adjusted"
    _write_csv(
        adjusted_dir / "AAA.csv",
        [
            {"Date": row["date"], "Adj Close": row["close"]}
            for row in _price_rows()
        ],
    )

    report = attach_price_labels_report_only(
        event_dataset_path=event_path,
        adjusted_price_dir=adjusted_dir,
        output_dir=reports / "labeled",
        reports_root=reports,
    )

    row = next(csv.DictReader((reports / "labeled" / "news_transformer_event_features_labeled.csv").open(encoding="utf-8")))
    assert report["labels_attached"] is True
    assert row["label_date"] == "2024-01-02"
    assert row["future_return_20d"] == "-0.0900000000"


def test_outputs_must_stay_under_reports_and_no_training_paths_are_emitted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _run(tmp_path, output_dir=tmp_path / "outside")

    report = _run(tmp_path)
    rendered = json.dumps(report).lower()

    assert "model_training" not in rendered
    assert "transformer_training" not in rendered
    assert "broker" not in rendered
