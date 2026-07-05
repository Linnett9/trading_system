from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.audits.adjusted_data_config import (
    _comparison_config,
    _output_dir,
    _validation_config,
)
from core.research.ml.audits.adjusted_data_analysis import (
    build_adjusted_data_comparison,
    _symbols_to_compare,
    detect_split_like_adjustment_ratio,
)
from core.research.ml.audits.adjusted_data_loading import (
    _adjusted_close_by_date,
    _load_adjusted_rows_by_symbol,
    _load_raw_stooq_rows_by_symbol,
    _number,
    _raw_close_by_date,
    _read_json,
)
from core.research.ml.audits.adjusted_data_types import (
    AdjustedDataComparisonPaths,
    AdjustedPriceReplayPaths,
    NOTICE,
    RESEARCH_METADATA,
)
from core.research.ml.audits.adjusted_price_replay import (
    build_adjusted_price_replay,
)
from core.research.ml.audits.adjusted_data_reporting import (
    _comparison_json_payload,
    _comparison_markdown,
    _replay_markdown,
    _write_comparison_csv,
    _write_replay_csv,
)


def write_adjusted_data_comparison(
    config: dict[str, Any],
) -> AdjustedDataComparisonPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    comparison_config = _comparison_config(config)
    symbols = _symbols_to_compare(canonical, comparison_config)
    raw_rows = _load_raw_stooq_rows_by_symbol(
        Path(str(comparison_config["stooq_parquet_dir"])),
        symbols,
    )
    adjusted_rows = _load_adjusted_rows_by_symbol(comparison_config, symbols)
    payload = build_adjusted_data_comparison(
        raw_rows_by_symbol=raw_rows,
        adjusted_rows_by_symbol=adjusted_rows,
        canonical_replay=canonical,
        comparison_config=comparison_config,
    )
    paths = AdjustedDataComparisonPaths(
        csv_path=output_dir / "adjusted_data_comparison.csv",
        json_path=output_dir / "adjusted_data_comparison.json",
        markdown_path=output_dir / "adjusted_data_comparison.md",
    )
    paths.json_path.write_text(
        json.dumps(_comparison_json_payload(payload), indent=2),
        encoding="utf-8",
    )
    _write_comparison_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_comparison_markdown(payload), encoding="utf-8")
    return paths


def write_adjusted_price_replay(
    config: dict[str, Any],
) -> AdjustedPriceReplayPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    selected_optimizer = _read_json(output_dir / "selected_optimizer_exposure_path.json")
    comparison = _read_json(output_dir / "adjusted_data_comparison.json")
    comparison_config = _comparison_config(config)
    symbols = _symbols_to_compare(canonical, comparison_config)
    raw_rows = _load_raw_stooq_rows_by_symbol(
        Path(str(comparison_config["stooq_parquet_dir"])),
        symbols,
    )
    adjusted_rows = _load_adjusted_rows_by_symbol(comparison_config, symbols)
    payload = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion_audit,
        selected_optimizer=selected_optimizer,
        adjusted_comparison=comparison,
        raw_closes_by_symbol={
            symbol: _raw_close_by_date(rows)
            for symbol, rows in raw_rows.items()
        },
        adjusted_closes_by_symbol={
            symbol: _adjusted_close_by_date(rows)
            for symbol, rows in adjusted_rows.items()
        },
        validation_config=_validation_config(config),
    )
    paths = AdjustedPriceReplayPaths(
        csv_path=output_dir / "adjusted_price_replay.csv",
        json_path=output_dir / "adjusted_price_replay.json",
        markdown_path=output_dir / "adjusted_price_replay.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_replay_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_replay_markdown(payload), encoding="utf-8")
    return paths
