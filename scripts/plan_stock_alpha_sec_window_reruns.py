from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_sec_window_rerun_plan(
    summary: Mapping[str, Any],
    *,
    mode: str = "unresolved-only",
    include_missing_grouped_configs: bool = False,
) -> str:
    window_months = int(summary["window_months"])
    config_dir = Path(str(summary.get("config_dir") or "config"))
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
    successful_symbols = set(_successful_event_symbols(items))
    planned_symbols: set[str] = set()
    for item in _prioritized_items(items, mode):
        lines.extend(
            _commands_for_item(
                item,
                window_months,
                config_dir=config_dir,
                mode=mode,
                include_missing_grouped_configs=include_missing_grouped_configs,
                successful_symbols=successful_symbols,
                planned_symbols=planned_symbols,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _prioritized_items(items: Sequence[Mapping[str, Any]], mode: str) -> list[Mapping[str, Any]]:
    if mode == "unresolved-only":
        priority = {
            "timeout_failure": 0,
            "provider_failure": 0,
        }
    elif mode == "all-actionable":
        priority = {
            "success_missing_event_rows": 0,
            "timeout_failure": 1,
            "provider_failure": 1,
            "rate_limited": 1,
            "unattempted_symbols": 1,
            "missing_report": 2,
            "unsafe_output_path": 3,
        }
    else:
        raise ValueError(f"Unsupported planner mode: {mode}")
    return sorted(
        (item for item in items if item.get("classification") in priority),
        key=lambda item: (priority[str(item.get("classification"))], str(item.get("family_name"))),
    )


def _commands_for_item(
    item: Mapping[str, Any],
    window_months: int,
    *,
    config_dir: Path,
    mode: str,
    include_missing_grouped_configs: bool,
    successful_symbols: set[str],
    planned_symbols: set[str],
) -> list[str]:
    classification = str(item.get("classification", ""))
    config_path = str(item.get("config_path", ""))
    requested = [
        str(symbol).strip().upper()
        for symbol in item.get("requested_symbols", [])
        if (
            str(symbol).strip()
            and str(symbol).strip().upper() not in successful_symbols
            and str(symbol).strip().upper() not in planned_symbols
        )
    ]
    family = str(item.get("family_name", ""))
    lines = [
        f"# {classification}: {family} requested={','.join(requested)}",
    ]
    if not requested and classification in {"timeout_failure", "provider_failure", "rate_limited", "unattempted_symbols"}:
        lines.append("# Already recovered by a successful event-row report; no command emitted.")
        return lines
    if classification == "success_missing_event_rows" and config_path and mode == "all-actionable":
        if _path_exists(config_path):
            lines.append(_run_command(config_path))
            lines.append("sleep 5")
        else:
            lines.append(f"# TODO missing config for event-row rerun: {config_path}")
        return lines
    if classification in {"timeout_failure", "provider_failure", "rate_limited", "unattempted_symbols"}:
        for symbol in requested:
            retry_config = _retry_config_path(window_months, family, symbol, config_dir)
            if retry_config is None:
                lines.append(f"# TODO cannot derive one-symbol retry config for {symbol}; no executable command emitted.")
                planned_symbols.add(symbol)
                continue
            if _path_exists(retry_config):
                lines.append(_run_command(retry_config))
                lines.append("sleep 10")
            else:
                lines.append(f"# TODO create/use one-symbol retry config for {symbol}: {retry_config}")
            planned_symbols.add(symbol)
        return lines
    if classification == "missing_report" and config_path and mode == "all-actionable":
        if window_months == 120 and len(requested) > 1 and not include_missing_grouped_configs:
            lines.append("# Missing grouped 120mo config is not expanded by default.")
            lines.append("# Re-run planner with --include-missing-grouped-configs only after manual review.")
            return lines
        if window_months == 120 and len(requested) > 1:
            for symbol in requested:
                retry_config = _retry_config_path(window_months, family, symbol, config_dir)
                if retry_config is None:
                    lines.append(f"# TODO cannot derive one-symbol retry config for {symbol}; no executable command emitted.")
                elif _path_exists(retry_config):
                    lines.append(_run_command(retry_config))
                    lines.append("sleep 10")
                else:
                    lines.append(f"# TODO create/use one-symbol retry config for {symbol}: {retry_config}")
                planned_symbols.add(symbol)
        elif _path_exists(config_path):
            lines.append(_run_command(config_path))
            lines.append("sleep 5")
        else:
            lines.append(f"# TODO missing config for missing report: {config_path}")
    return lines


def _retry_config_path(window_months: int, family: str, symbol: str, config_dir: Path) -> str | None:
    source_part = _source_part(window_months, family)
    if source_part is None:
        return None
    safe_symbol = symbol.replace(".", "_")
    filename = (
        f"config.stock_alpha_news_collect_sec_company_filings_{window_months}mo_"
        f"retry_part_{source_part}_{safe_symbol}_dry_run.yaml"
    )
    return str(config_dir / filename)


def _source_part(window_months: int, family: str) -> str | None:
    markers = (f"_{window_months}mo_retry_part_", f"_{window_months}mo_part_")
    for marker in markers:
        if marker in family:
            return family.split(marker, 1)[1].split("_", 1)[0]
    return None


def _successful_event_symbols(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({
        str(symbol).strip().upper()
        for item in items
        if item.get("classification") == "success_with_event_rows"
        for symbol in item.get("requested_symbols", [])
        if str(symbol).strip()
    })


def _path_exists(path: str) -> bool:
    return Path(path).exists()


def _run_command(config_path: str) -> str:
    return f'PYTHONDONTWRITEBYTECODE=1 "$PY" main.py --mode ml-stock-alpha-news-collect-free-sources --config "{config_path}"'


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate local rerun commands from SEC event-row coverage summary.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("unresolved-only", "all-actionable"), default="unresolved-only")
    parser.add_argument("--include-missing-grouped-configs", action="store_true")
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    plan = build_sec_window_rerun_plan(
        summary,
        mode=args.mode,
        include_missing_grouped_configs=args.include_missing_grouped_configs,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
