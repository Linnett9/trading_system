from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


LABEL_COLUMNS = (
    "label_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_drawdown_20d",
    "reduce_exposure_label",
)


def attach_price_labels_report_only(
    *,
    event_dataset_path: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
    price_csv_path: str | Path | None = None,
    reduce_exposure_drawdown_threshold: float = -0.05,
) -> dict[str, Any]:
    reports_root_path = Path(reports_root)
    output_dir_path = Path(output_dir)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    rows = _read_csv(Path(event_dataset_path))
    duplicate_event_key_count = _duplicate_count(row.get("event_key", "") for row in rows)
    future_timestamp_count = sum(1 for row in rows if _is_future(row.get("available_at_timestamp", "")))
    blocking_reasons: list[str] = []
    if duplicate_event_key_count:
        blocking_reasons.append("duplicate_event_keys")
    if future_timestamp_count:
        blocking_reasons.append("future_timestamps")

    if price_csv_path is None:
        blocking_reasons.append("price_loader_not_found")
        report = _report(
            rows=rows,
            labeled_rows=[],
            duplicate_event_key_count=duplicate_event_key_count,
            future_timestamp_count=future_timestamp_count,
            leakage_violation_count=0,
            missing_price_symbols=sorted({row.get("symbol", "") for row in rows if row.get("symbol")}),
            blocking_reasons=blocking_reasons,
            warnings=["no canonical project price loader was found for stock-alpha news labels"],
            labels_attached=False,
        )
        _write_reports(output_dir_path, [], report)
        return report

    prices_by_symbol = _read_prices(Path(price_csv_path))
    labeled_rows, missing_symbols, leakage_violations = _label_rows(
        rows,
        prices_by_symbol=prices_by_symbol,
        reduce_exposure_drawdown_threshold=reduce_exposure_drawdown_threshold,
    )
    if leakage_violations:
        blocking_reasons.append("label_date_precedes_available_at_timestamp")
    labels_attached = not blocking_reasons
    report = _report(
        rows=rows,
        labeled_rows=labeled_rows,
        duplicate_event_key_count=duplicate_event_key_count,
        future_timestamp_count=future_timestamp_count,
        leakage_violation_count=leakage_violations,
        missing_price_symbols=missing_symbols,
        blocking_reasons=blocking_reasons,
        warnings=[] if labels_attached else ["labels were computed only for rows with sufficient future price bars"],
        labels_attached=labels_attached,
    )
    _write_reports(output_dir_path, labeled_rows, report)
    return report


def _label_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    prices_by_symbol: Mapping[str, Sequence[tuple[date, float]]],
    reduce_exposure_drawdown_threshold: float,
) -> tuple[list[dict[str, str]], list[str], int]:
    labeled: list[dict[str, str]] = []
    missing_symbols: set[str] = set()
    leakage_violations = 0
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        available_at = _parse_timestamp(str(row.get("available_at_timestamp", "")))
        prices = list(prices_by_symbol.get(symbol, []))
        labeled_row = dict(row)
        for column in LABEL_COLUMNS:
            labeled_row[column] = ""
        if not available_at or not prices:
            if symbol:
                missing_symbols.add(symbol)
            labeled.append(labeled_row)
            continue
        price_dates = [item[0] for item in prices]
        index = bisect_left(price_dates, available_at.date())
        if index >= len(prices):
            missing_symbols.add(symbol)
            labeled.append(labeled_row)
            continue
        label_date = prices[index][0]
        if label_date < available_at.date():
            leakage_violations += 1
        close_t = prices[index][1]
        labeled_row["label_date"] = label_date.isoformat()
        labeled_row["future_return_1d"] = _future_return(prices, index, 1)
        labeled_row["future_return_5d"] = _future_return(prices, index, 5)
        labeled_row["future_return_20d"] = _future_return(prices, index, 20)
        labeled_row["future_drawdown_20d"] = _future_drawdown(prices, index, 20)
        return_20d = _float_or_none(labeled_row["future_return_20d"])
        drawdown_20d = _float_or_none(labeled_row["future_drawdown_20d"])
        if return_20d is not None and drawdown_20d is not None:
            labeled_row["reduce_exposure_label"] = str(
                return_20d < 0 or drawdown_20d <= reduce_exposure_drawdown_threshold
            ).lower()
        elif close_t <= 0:
            labeled_row["reduce_exposure_label"] = ""
        labeled.append(labeled_row)
    return labeled, sorted(missing_symbols), leakage_violations


def _future_return(prices: Sequence[tuple[date, float]], index: int, horizon: int) -> str:
    if index + horizon >= len(prices) or prices[index][1] <= 0:
        return ""
    return _format_float((prices[index + horizon][1] / prices[index][1]) - 1)


def _future_drawdown(prices: Sequence[tuple[date, float]], index: int, horizon: int) -> str:
    if index + 1 >= len(prices) or prices[index][1] <= 0:
        return ""
    end = min(index + horizon, len(prices) - 1)
    if end <= index:
        return ""
    min_future = min(close for _, close in prices[index + 1:end + 1])
    return _format_float((min_future / prices[index][1]) - 1)


def _report(
    *,
    rows: Sequence[Mapping[str, str]],
    labeled_rows: Sequence[Mapping[str, str]],
    duplicate_event_key_count: int,
    future_timestamp_count: int,
    leakage_violation_count: int,
    missing_price_symbols: Sequence[str],
    blocking_reasons: Sequence[str],
    warnings: Sequence[str],
    labels_attached: bool,
) -> dict[str, Any]:
    rows_labeled = sum(1 for row in labeled_rows if row.get("future_return_20d"))
    symbols_in = sorted({row.get("symbol", "") for row in rows if row.get("symbol")})
    symbols_labeled = sorted({row.get("symbol", "") for row in labeled_rows if row.get("future_return_20d")})
    return {
        "labels_attached": labels_attached,
        "rows_in": len(rows),
        "rows_labeled": rows_labeled,
        "rows_unlabeled": len(rows) - rows_labeled,
        "symbols_in": symbols_in,
        "symbols_labeled": symbols_labeled,
        "missing_price_symbols": list(missing_price_symbols),
        "duplicate_event_key_count": duplicate_event_key_count,
        "future_timestamp_count": future_timestamp_count,
        "leakage_violation_count": leakage_violation_count,
        "blocking_reasons": list(blocking_reasons),
        "warnings": list(warnings),
        "next_allowed_step": (
            "build_news_transformer_walk_forward_splits_report_only"
            if labels_attached
            else "implement_or_select_canonical_price_loader"
        ),
    }


def _write_reports(output_dir: Path, labeled_rows: Sequence[Mapping[str, str]], report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if labeled_rows:
        columns = [column for column in labeled_rows[0]]
        with (output_dir / "news_transformer_event_features_labeled.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(labeled_rows)
    (output_dir / "news_transformer_price_label_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "news_transformer_label_leakage_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_prices(path: Path) -> dict[str, list[tuple[date, float]]]:
    prices: dict[str, list[tuple[date, float]]] = {}
    for row in _read_csv(path):
        symbol = str(row.get("symbol", "")).strip().upper()
        date_text = str(row.get("date") or row.get("timestamp") or "").strip()[:10]
        close_text = str(row.get("adj_close") or row.get("close") or "").strip()
        if not symbol or not date_text or not close_text:
            continue
        prices.setdefault(symbol, []).append((date.fromisoformat(date_text), float(close_text)))
    return {symbol: sorted(items) for symbol, items in prices.items()}


def _duplicate_count(values: Sequence[str]) -> int:
    materialized = [value for value in values if value]
    return len(materialized) - len(set(materialized))


def _is_future(value: str) -> bool:
    parsed = _parse_timestamp(value)
    return bool(parsed and parsed > datetime.now(timezone.utc))


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


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def _format_float(value: float) -> str:
    return f"{value:.10f}"


def _float_or_none(value: str) -> float | None:
    if not value:
        return None
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach report-only price labels to news transformer events.")
    parser.add_argument("--event-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--price-csv")
    parser.add_argument("--reduce-exposure-drawdown-threshold", type=float, default=-0.05)
    args = parser.parse_args(argv)

    report = attach_price_labels_report_only(
        event_dataset_path=args.event_dataset,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
        price_csv_path=args.price_csv,
        reduce_exposure_drawdown_threshold=args.reduce_exposure_drawdown_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
