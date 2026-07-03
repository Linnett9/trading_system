from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_MARKERS = (
    "data/news",
    "transformer",
    "feature",
    "training",
    "readiness",
    "broker",
    "paper",
    "live",
    "trading",
)


def build_missing_window_wave_plans(
    summary: Mapping[str, Any],
    *,
    wave_size: int = 5,
) -> list[str]:
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    items = _missing_report_items(summary)
    waves = [items[index:index + wave_size] for index in range(0, len(items), wave_size)]
    return [_render_wave(wave, wave_number=index + 1, total_waves=len(waves)) for index, wave in enumerate(waves)]


def _missing_report_items(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = [
        item
        for item in summary.get("items", []) or []
        if item.get("classification") == "missing_report"
    ]
    return sorted(items, key=lambda item: str(item.get("family_name", "")))


def _render_wave(wave: Sequence[Mapping[str, Any]], *, wave_number: int, total_waves: int) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Missing 120-month SEC grouped config wave {wave_number:02d} of {total_waves:02d}.",
        "# Run locally outside Codex.",
        "# Stop on timeout, provider failure, or rate limit.",
        "# Inspect the SEC event-row coverage summary after each wave.",
        "# Do not commit reports/.",
        "# This plan does not decompose grouped configs into one-symbol retries.",
        "PY=${PY:-/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python}",
        "",
    ]
    for item in wave:
        config_path = str(item.get("config_path", "")).strip()
        family = str(item.get("family_name", "")).strip()
        symbols = ",".join(str(symbol).strip().upper() for symbol in item.get("requested_symbols", []) if str(symbol).strip())
        lines.append(f"# missing_report: {family} requested={symbols}")
        if not config_path:
            lines.append("# TODO missing config path; no command emitted.")
            continue
        if _has_forbidden_marker(config_path):
            lines.append(f"# TODO unsafe config path skipped: {config_path}")
            continue
        lines.append(_run_command(config_path))
        lines.append("sleep 10")
    return "\n".join(lines).rstrip() + "\n"


def _has_forbidden_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in FORBIDDEN_MARKERS)


def _run_command(config_path: str) -> str:
    return f'PYTHONDONTWRITEBYTECODE=1 "$PY" main.py --mode ml-stock-alpha-news-collect-free-sources --config "{config_path}"'


def write_missing_window_wave_plans(
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path,
    output_prefix: str = "sec_120mo_missing_wave",
    wave_size: int = 5,
    max_waves: int | None = None,
) -> list[Path]:
    plans = build_missing_window_wave_plans(summary, wave_size=wave_size)
    if max_waves is not None:
        plans = plans[:max_waves]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, plan in enumerate(plans, start=1):
        path = output_dir_path / f"{output_prefix}_{index:02d}.sh"
        path.write_text(plan, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write local waves for missing 120-month SEC grouped configs.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="sec_120mo_missing_wave")
    parser.add_argument("--wave-size", type=int, default=5)
    parser.add_argument("--max-waves", type=int)
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    paths = write_missing_window_wave_plans(
        summary,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        wave_size=args.wave_size,
        max_waves=args.max_waves,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
