from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_sec_window_rerun_plan(summary: Mapping[str, Any]) -> str:
    window_months = int(summary["window_months"])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run locally outside Codex.",
        "# Inspect the coverage summary after each small wave.",
        "# Stop on timeout, provider failure, rate limit, raw write, or output outside reports/.",
        "# Do not commit reports/ or data/news/.",
        "PY=${PY:-/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python}",
        "",
    ]
    items = list(summary.get("items", []) or [])
    for item in _prioritized_items(items):
        lines.extend(_commands_for_item(item, window_months))
    return "\n".join(lines).rstrip() + "\n"


def _prioritized_items(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    priority = {
        "success_missing_event_rows": 0,
        "timeout_failure": 1,
        "provider_failure": 1,
        "rate_limited": 1,
        "unattempted_symbols": 1,
        "missing_report": 2,
        "unsafe_output_path": 3,
    }
    return sorted(
        (item for item in items if item.get("classification") in priority),
        key=lambda item: (priority[str(item.get("classification"))], str(item.get("family_name"))),
    )


def _commands_for_item(item: Mapping[str, Any], window_months: int) -> list[str]:
    classification = str(item.get("classification", ""))
    config_path = str(item.get("config_path", ""))
    requested = [str(symbol).strip().upper() for symbol in item.get("requested_symbols", []) if str(symbol).strip()]
    family = str(item.get("family_name", ""))
    lines = [
        f"# {classification}: {family} requested={','.join(requested)}",
    ]
    if classification == "success_missing_event_rows" and config_path:
        lines.append(_run_command(config_path))
        lines.append("sleep 5")
        return lines
    if classification in {"timeout_failure", "provider_failure", "rate_limited", "unattempted_symbols"}:
        for symbol in requested:
            retry_config = _retry_config_path(window_months, family, symbol)
            lines.append(f"# Create/use one-symbol retry config for {symbol}; never rerun failed grouped 120mo configs blindly.")
            lines.append(_run_command(retry_config))
            lines.append("sleep 10")
        return lines
    if classification == "missing_report" and config_path:
        if window_months == 120 and len(requested) > 1:
            for symbol in requested:
                retry_config = _retry_config_path(window_months, family, symbol)
                lines.append(f"# Missing grouped report; prefer one-symbol retry for {symbol}.")
                lines.append(_run_command(retry_config))
                lines.append("sleep 10")
        else:
            lines.append(_run_command(config_path))
            lines.append("sleep 5")
    return lines


def _retry_config_path(window_months: int, family: str, symbol: str) -> str:
    source_part = "unknown"
    marker = f"_{window_months}mo_part_"
    if marker in family:
        source_part = family.split(marker, 1)[1].split("_", 1)[0]
    safe_symbol = symbol.replace(".", "_")
    return (
        "config/"
        f"config.stock_alpha_news_collect_sec_company_filings_{window_months}mo_retry_part_{source_part}_{safe_symbol}_dry_run.yaml"
    )


def _run_command(config_path: str) -> str:
    return f'PYTHONDONTWRITEBYTECODE=1 "$PY" main.py --mode ml-stock-alpha-news-collect-free-sources --config "{config_path}"'


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate local rerun commands from SEC event-row coverage summary.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    plan = build_sec_window_rerun_plan(summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
