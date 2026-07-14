from __future__ import annotations

import csv
from pathlib import Path

from infrastructure.data.alpaca_daily_reclassification_diagnostics import (
    large_row_triggers,
    run_reclassification_diagnostics,
)


def test_large_row_triggers_distinguish_absolute_relative_return_and_volume():
    row = _row("AAPL", "2026-01-02", 100, 102, 100, 101, classification="LARGE_UNEXPLAINED_DIFFERENCE")
    row["close_abs_diff"] = "2.0"
    row["close_rel_diff"] = "0.02"
    row["return_abs_diff"] = "0.003"
    row["volume_rel_diff"] = "0.4"

    triggers = large_row_triggers(row)

    assert "close absolute threshold" in triggers
    assert "close relative threshold" in triggers
    assert "return threshold" in triggers
    assert "volume threshold" in triggers


def test_stable_price_offset_is_not_genuine_return_disagreement(tmp_path):
    path = tmp_path / "provider_reconciliation.csv"
    rows = []
    for index in range(30):
        session = f"2026-01-{index + 1:02d}"
        rows.append(_row("OFFSET", session, 100 + index, 102 + index * 1.02, 1000, 1000, classification="LARGE_UNEXPLAINED_DIFFERENCE"))
    _write_rows(path, rows)

    result = run_reclassification_diagnostics(reconciliation_path=path, report_root=tmp_path / "reports")

    counts = result["summary"]["revised_classification_counts"]
    assert counts["STABLE_PRICE_LEVEL_ADJUSTMENT_DIFFERENCE"] == 30
    assert counts.get("GENUINE_LARGE_RETURN_DISAGREEMENT", 0) == 0


def test_sudden_ratio_change_and_corporate_like_return_are_detected(tmp_path):
    path = tmp_path / "provider_reconciliation.csv"
    rows = []
    for index in range(10):
        rows.append(_row("JUMP", f"2026-01-{index + 1:02d}", 100 + index, 100 + index, 1000, 1000))
    rows.append(_row("JUMP", "2026-01-11", 110, 55, 1000, 1000, classification="LARGE_UNEXPLAINED_DIFFERENCE"))
    rows[-1]["return_abs_diff"] = "0.5"
    _write_rows(path, rows)

    result = run_reclassification_diagnostics(reconciliation_path=path, report_root=tmp_path / "reports")

    assert result["price_ratio_regime_changes"]
    assert result["summary"]["revised_classification_counts"]["POSSIBLE_CORPORATE_ACTION"] >= 1


def test_boundaries_extensions_dry_run_and_idempotence(tmp_path):
    path = tmp_path / "provider_reconciliation.csv"
    rows = [_row("AAPL", "2026-01-01", "", 100, "", 1000, classification="STOOQ_ONLY", alpaca=False)]
    for day in range(2, 25):
        rows.append(_row("AAPL", f"2026-01-{day:02d}", 100 + day, 100 + day, 1000, 1000))
    rows.append(_row("AAPL", "2026-01-25", 125, "", 1000, "", classification="ALPACA_ONLY", stooq=False))
    _write_rows(path, rows)
    before = path.read_text(encoding="utf-8")
    report_root = tmp_path / "reports"

    dry = run_reclassification_diagnostics(reconciliation_path=path, report_root=report_root, dry_run=True)
    first = run_reclassification_diagnostics(reconciliation_path=path, report_root=report_root)
    second = run_reclassification_diagnostics(reconciliation_path=path, report_root=report_root)

    assert dry["dry_run"] is True
    assert path.read_text(encoding="utf-8") == before
    assert first["summary"] == second["summary"]
    assert first["summary"]["revised_classification_counts"]["STOOQ_ONLY_ARCHIVE_BOUNDARY"] == 1
    assert first["summary"]["revised_classification_counts"]["ALPACA_ONLY_RECENT_EXTENSION"] == 1
    assert (report_root / "revised_provider_reconciliation.csv").exists()


def test_every_row_gets_one_revised_classification_and_triggers(tmp_path):
    path = tmp_path / "provider_reconciliation.csv"
    rows = [_row("AAPL", f"2026-01-{day:02d}", 100 + day, 100 + day, 1000, 1000) for day in range(1, 25)]
    rows.append(_row("AAPL", "2026-01-25", 125, 125, 1000, 1000, classification="LARGE_UNEXPLAINED_DIFFERENCE"))
    rows[-1]["return_abs_diff"] = "0.003"
    _write_rows(path, rows)

    result = run_reclassification_diagnostics(reconciliation_path=path, report_root=tmp_path / "reports")
    revised = list(csv.DictReader((tmp_path / "reports" / "revised_provider_reconciliation.csv").open(encoding="utf-8")))

    assert len(revised) == len(rows)
    assert all(row["classification"] for row in revised)
    assert all(row["trigger_reasons"] for row in revised)
    assert result["extreme_return_review"]


def _row(symbol, session, alpaca_close, stooq_close, alpaca_volume, stooq_volume, *, classification="MATCH", alpaca=True, stooq=True):
    alpaca_close_value = "" if alpaca_close == "" else float(alpaca_close)
    stooq_close_value = "" if stooq_close == "" else float(stooq_close)
    close_abs = "" if "" in (alpaca_close_value, stooq_close_value) else abs(alpaca_close_value - stooq_close_value)
    close_rel = "" if close_abs == "" or stooq_close_value == 0 else close_abs / abs(stooq_close_value)
    volume_abs = "" if "" in (alpaca_volume, stooq_volume) else abs(float(alpaca_volume) - float(stooq_volume))
    volume_rel = "" if volume_abs == "" or float(stooq_volume) == 0 else volume_abs / abs(float(stooq_volume))
    return {
        "asset_id": f"asset_{symbol}",
        "canonical_symbol": symbol,
        "session_date": session,
        "alpaca_present": str(alpaca).lower(),
        "stooq_present": str(stooq).lower(),
        "alpaca_open": alpaca_close_value,
        "stooq_open": stooq_close_value,
        "open_abs_diff": close_abs,
        "open_rel_diff": close_rel,
        "alpaca_high": alpaca_close_value,
        "stooq_high": stooq_close_value,
        "high_abs_diff": close_abs,
        "high_rel_diff": close_rel,
        "alpaca_low": alpaca_close_value,
        "stooq_low": stooq_close_value,
        "low_abs_diff": close_abs,
        "low_rel_diff": close_rel,
        "alpaca_close": alpaca_close_value,
        "stooq_close": stooq_close_value,
        "close_abs_diff": close_abs,
        "close_rel_diff": close_rel,
        "alpaca_volume": alpaca_volume,
        "stooq_volume": stooq_volume,
        "volume_abs_diff": volume_abs,
        "volume_rel_diff": volume_rel,
        "alpaca_return": "0.01" if alpaca and stooq else "",
        "stooq_return": "0.01" if alpaca and stooq else "",
        "return_abs_diff": "0.0" if alpaca and stooq else "",
        "missing_source_indicator": "present_both" if alpaca and stooq else ("missing_alpaca" if not alpaca else "missing_stooq"),
        "classification": classification,
    }


def _write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
