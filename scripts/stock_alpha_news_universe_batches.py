from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def build_universe_batches(universe_path: str | Path, batch_size: int) -> list[list[str]]:
    if batch_size < 1 or batch_size > 500:
        raise ValueError("batch_size must be between 1 and 500")
    payload: Any = yaml.safe_load(Path(universe_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("universe file must contain a YAML mapping")
    symbols = [
        str(value).strip().upper()
        for value in payload.get("symbols", []) or []
        if str(value).strip()
    ]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("universe symbols must be non-empty and unique")
    expected_count = int(payload.get("available_count", len(symbols)))
    if expected_count != len(symbols):
        raise ValueError("universe available_count does not match symbols")
    return [
        symbols[index : index + batch_size]
        for index in range(0, len(symbols), batch_size)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List deterministic, read-only batches from a stock-alpha universe."
    )
    parser.add_argument("--universe", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--batch-index",
        type=int,
        help="Print one 1-based batch as a YAML only_symbols value.",
    )
    args = parser.parse_args(argv)

    batches = build_universe_batches(args.universe, args.batch_size)
    if args.batch_index is not None:
        if args.batch_index < 1 or args.batch_index > len(batches):
            parser.error(f"--batch-index must be between 1 and {len(batches)}")
        print(yaml.safe_dump({"only_symbols": batches[args.batch_index - 1]}, sort_keys=False).strip())
        return 0

    for index, symbols in enumerate(batches, start=1):
        print(
            yaml.safe_dump(
                {f"batch_{index:02d}": symbols},
                default_flow_style=True,
                sort_keys=False,
                width=100_000,
            ).strip()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
