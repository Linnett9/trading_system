from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.interfaces.data_feed import IDataFeed
from core.research.ml.audits.benchmark_relative_validation import (
    _load_required_closes as _load_benchmark_required_closes,
)
from core.research.ml.audits.data_adjustment_validation_adjustments import (
    _adjusted_status_acceptable,
    _adjustment_red_flags,
    _candidate_suspicious_dependencies,
    _daily_row_dependencies,
    _normalized_price_rows,
    _overall_adjusted_status,
    _period_anomaly_dependencies,
    _period_anomaly_rows,
    _raw_adjusted_comparison,
    _split_like_factor,
    _suspicious_rebalance_dates_from_daily_rows,
    _symbol_adjusted_status,
    _symbol_adjustment_report,
    _symbols_to_audit,
    _unique_dependency_rows,
    build_data_adjustment_audit,
    detect_split_like_jumps,
)
from core.research.ml.audits.data_adjustment_validation_clean_replay import (
    _excluded_period_count,
    _validation_candidates_by_name,
    _validation_summary,
    build_clean_data_replay,
    build_independent_period_validation,
)
from core.research.ml.audits.data_adjustment_validation_config import (
    _audit_config,
    _normalize_audit_config,
    _output_dir,
    _validation_config,
)
from core.research.ml.audits.data_adjustment_validation_loading import (
    _load_stooq_price_rows,
    _load_stooq_price_rows_by_symbol,
    _read_json,
)
from core.research.ml.audits.data_adjustment_validation_reporting import (
    _adjustment_markdown,
    _clean_replay_markdown,
    _independent_period_markdown,
    _write_adjustment_csv,
    _write_clean_replay_csv,
)
from core.research.ml.audits.data_adjustment_validation_types import (
    COMMON_SPLIT_FACTORS,
    DEFAULT_INSPECT_SYMBOLS,
    NOTICE,
    REPORT_CANDIDATES,
    RESEARCH_METADATA,
    CleanDataReplayPaths,
    DataAdjustmentAuditPaths,
    IndependentPeriodValidationPaths,
)
from core.research.ml.audits.data_adjustment_validation_utils import (
    _date,
    _date_string,
    _first_number,
    _first_present,
    _fmt,
    _number,
    _numbers_close,
)


def write_data_adjustment_audit(
    config: dict[str, Any],
) -> DataAdjustmentAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    audit_config = _audit_config(config)
    symbols = _symbols_to_audit(canonical, champion_audit, audit_config)
    symbol_rows = _load_stooq_price_rows_by_symbol(
        Path(str(audit_config["stooq_parquet_dir"])),
        symbols,
    )
    payload = build_data_adjustment_audit(
        symbol_rows_by_symbol=symbol_rows,
        canonical_replay=canonical,
        champion_audit=champion_audit,
        audit_config=audit_config,
    )
    paths = DataAdjustmentAuditPaths(
        csv_path=output_dir / "data_adjustment_audit.csv",
        json_path=output_dir / "data_adjustment_audit.json",
        markdown_path=output_dir / "data_adjustment_audit.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_adjustment_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_adjustment_markdown(payload), encoding="utf-8")
    return paths


def write_clean_data_replay(
    config: dict[str, Any],
    data_feed: IDataFeed,
) -> CleanDataReplayPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    selected_optimizer = _read_json(output_dir / "selected_optimizer_exposure_path.json")
    adjustment_audit = _read_json(output_dir / "data_adjustment_audit.json")
    closes = _load_benchmark_required_closes(config, data_feed, canonical)
    payload = build_clean_data_replay(
        canonical_replay=canonical,
        champion_audit=champion_audit,
        selected_optimizer=selected_optimizer,
        adjustment_audit=adjustment_audit,
        closes_by_symbol=closes,
        validation_config=_validation_config(config),
    )
    paths = CleanDataReplayPaths(
        csv_path=output_dir / "clean_data_replay.csv",
        json_path=output_dir / "clean_data_replay.json",
        markdown_path=output_dir / "clean_data_replay.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_clean_replay_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_clean_replay_markdown(payload), encoding="utf-8")
    return paths


def write_independent_period_validation(
    config: dict[str, Any],
) -> IndependentPeriodValidationPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    payload = build_independent_period_validation(
        canonical_replay=canonical,
        validation_config=_validation_config(config),
    )
    paths = IndependentPeriodValidationPaths(
        json_path=output_dir / "independent_period_validation.json",
        markdown_path=output_dir / "independent_period_validation.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(
        _independent_period_markdown(payload),
        encoding="utf-8",
    )
    return paths
