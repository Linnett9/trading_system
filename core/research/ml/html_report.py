from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def write_research_html_report(output_path: Path, artifact_dir: Path) -> None:
    """Write a static research report from already-generated ML artifacts."""
    metrics = _read_json(artifact_dir / "metrics.json")
    calibration = _read_json(artifact_dir / "probability_calibration.json")
    calibrated = _read_json(artifact_dir / "calibrated_probability_calibration.json")
    overlay = _read_json(artifact_dir / "holdout_shadow_overlay.json")
    overlay_result = overlay.get("result") if isinstance(overlay, dict) else None
    overlay_payload = overlay_result or overlay or {"status": "not_available"}
    walk_forward = _read_json(artifact_dir / "walk_forward_metrics.json")
    confusion_rows = _read_csv(artifact_dir / "confusion_matrix.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join([
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<title>ML Research Report</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:32px;color:#1f2933}",
            "section{margin:0 0 28px}",
            "table{border-collapse:collapse;width:100%;max-width:980px}",
            "th,td{border:1px solid #d9e2ec;padding:7px 9px;text-align:left}",
            "th{background:#f0f4f8}",
            "code{background:#f0f4f8;padding:2px 4px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>ML Research Report</h1>",
            "<p>Research only. Trading impact: none. Production validated: false.</p>",
            _section("Model Performance", _dict_table(_flatten_metrics(metrics))),
            _section("Calibration", _calibration_table(calibration, calibrated)),
            _section("Confusion Matrix", _rows_table(confusion_rows)),
            _section("Shadow Overlay", _dict_table(overlay_payload)),
            _section("Walk-Forward Results", _walk_forward_table(walk_forward)),
            "</body>",
            "</html>",
        ]),
        encoding="utf-8",
    )


def _section(title: str, body: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{body}</section>"


def _flatten_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return {**{k: v for k, v in payload.items() if k != "metrics"}, **metrics}
    return payload


def _calibration_table(calibration: dict[str, Any], calibrated: dict[str, Any]) -> str:
    raw = calibration.get("calibration", calibration)
    rows = [{
        "method": "raw",
        "brier_score": raw.get("brier_score"),
        "expected_calibration_error": raw.get("expected_calibration_error"),
    }]
    for item in calibrated.get("method_comparison", {}).get("ranked_methods", []):
        if item.get("method") == "raw":
            continue
        rows.append(item)
    return _rows_table(rows)


def _walk_forward_table(payload: dict[str, Any]) -> str:
    rows = []
    for fold in payload.get("folds", []):
        metrics = fold.get("metrics", {})
        rows.append({
            "fold": fold.get("fold"),
            "test_start_date": fold.get("test_start_date"),
            "test_sample_count": fold.get("test_sample_count"),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
        })
    return _rows_table(rows)


def _dict_table(payload: dict[str, Any] | None) -> str:
    if not payload:
        payload = {"status": "not_available"}
    rows = [{"metric": key, "value": value} for key, value in payload.items()]
    return _rows_table(rows)


def _rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    headers = list(rows[0])
    lines = [
        "<table>",
        "<thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_format(row.get(header)))}</td>"
                for header in headers
            )
            + "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "".join(lines)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
