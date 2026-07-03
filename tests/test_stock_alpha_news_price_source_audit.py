import csv
from pathlib import Path

from scripts.stock_alpha_news_price_source_audit import build_stock_alpha_news_price_source_audit


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_price_source_audit_detects_candidate_and_symbol_coverage(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "reports" / "events.csv",
        [
            {"symbol": "AAA", "event_key": "1"},
            {"symbol": "BBB", "event_key": "2"},
        ],
    )
    adjusted = tmp_path / "adjusted"
    _write_csv(adjusted / "AAA.csv", [{"Date": "2024-01-01", "Adj Close": "10"}])

    report = build_stock_alpha_news_price_source_audit(
        event_dataset_path=events,
        candidate_paths=[adjusted],
    )

    assert report["canonical_price_source_found"] is False
    assert report["symbol_coverage_count"] == 1
    assert report["missing_news_symbols"] == ["BBB"]
    assert report["candidate_price_sources"][0]["required_columns_present"] is True
    assert report["recommended_next_step"] == "extend_adjusted_price_source_to_full_news_symbol_universe"


def test_price_source_audit_reports_missing_required_columns(tmp_path: Path) -> None:
    events = _write_csv(tmp_path / "reports" / "events.csv", [{"symbol": "AAA", "event_key": "1"}])
    adjusted = tmp_path / "adjusted"
    _write_csv(adjusted / "AAA.csv", [{"symbol": "AAA", "price": "10"}])

    report = build_stock_alpha_news_price_source_audit(
        event_dataset_path=events,
        candidate_paths=[adjusted],
    )

    candidate = report["candidate_price_sources"][0]
    assert candidate["required_columns_present"] is False
    assert candidate["missing_columns"] == ["close_or_adjusted_close", "date"]
    assert report["canonical_price_source_found"] is False
