from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.audits.historical_coverage_audit_analysis import (
    _blockers,
    _bottleneck,
    _recommendations,
    _required_history,
    _row,
    _rows,
)
from core.research.ml.audits.historical_coverage_audit_io import (
    _audit_config,
    _markdown,
    _output_dir,
    _read_json,
    _write_csv,
)
from core.research.ml.audits.historical_coverage_audit_math import _date, _number
from core.research.ml.audits.historical_coverage_audit_ranges import (
    _aggregate_ranges,
    _csv_date_range,
    _load_adjusted_price_ranges,
    _load_raw_price_ranges,
    _parquet_date_range,
    _prediction_artifact_range,
    _range_summary,
)
from core.research.ml.audits.historical_coverage_audit_replay import (
    _adjusted_replay_summary,
    _canonical_summary,
    _median_label_window_days,
    _non_overlap_count,
    _possible_non_overlap_windows,
)
from core.research.ml.audits.historical_coverage_audit_types import (
    RESEARCH_METADATA,
    TARGET_PERIODS,
    HistoricalCoverageAuditPaths,
)


def write_historical_coverage_audit(
    config: dict[str, Any],
) -> HistoricalCoverageAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        Path(str(path))
        for path in config.get("ml", {}).get("source_prediction_dirs", []) or []
    ]
    payload = build_historical_coverage_audit(
        raw_price_ranges=_load_raw_price_ranges(
            Path(
                str(
                    config.get("ml", {}).get(
                        "stooq_parquet_dir",
                        "data/processed/stooq_parquet",
                    )
                )
            )
        ),
        adjusted_price_ranges=_load_adjusted_price_ranges(
            Path(
                str(
                    config.get("ml", {})
                    .get("adjusted_data_source", {})
                    .get("adjusted_data_dir", "data/reference/adjusted_prices")
                )
            )
        ),
        source_prediction_ranges=[
            _prediction_artifact_range(path / "prediction_artifacts.csv")
            for path in source_dirs
        ],
        meta_prediction_range=_prediction_artifact_range(
            output_dir / "meta_auxiliary_predictions.csv"
        ),
        canonical_replay=_read_json(output_dir / "canonical_continuous_equity_replay.json"),
        adjusted_price_replay=_read_json(output_dir / "adjusted_price_replay.json"),
        config=_audit_config(config),
    )
    paths = HistoricalCoverageAuditPaths(
        csv_path=output_dir / "historical_coverage_audit.csv",
        json_path=output_dir / "historical_coverage_audit.json",
        markdown_path=output_dir / "historical_coverage_audit.md",
    )
    _write_csv(paths.csv_path, payload)
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_historical_coverage_audit(
    *,
    raw_price_ranges: list[dict[str, Any]],
    adjusted_price_ranges: list[dict[str, Any]],
    source_prediction_ranges: list[dict[str, Any]],
    meta_prediction_range: dict[str, Any],
    canonical_replay: dict[str, Any],
    adjusted_price_replay: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    minimum = int(config.get("min_independent_periods", 36))
    label_window_days = _median_label_window_days(canonical_replay)
    raw = _aggregate_ranges(raw_price_ranges, "raw_stooq_parquet")
    adjusted = _aggregate_ranges(adjusted_price_ranges, "adjusted_reference_csv")
    source_predictions = _aggregate_ranges(
        [row for row in source_prediction_ranges if row.get("available")],
        "source_prediction_artifacts",
    )
    meta_predictions = _range_summary(meta_prediction_range, "meta_auxiliary_predictions")
    canonical = _canonical_summary(canonical_replay)
    adjusted_replay = _adjusted_replay_summary(adjusted_price_replay)
    possible = _possible_non_overlap_windows(canonical_replay)
    needed = {
        str(target): _required_history(
            latest=canonical.get("latest_canonical_replay_date")
            or meta_predictions.get("latest_date")
            or source_predictions.get("latest_date"),
            target_count=target,
            label_window_days=label_window_days,
        )
        for target in TARGET_PERIODS
    }
    bottleneck = _bottleneck(
        raw=raw,
        adjusted=adjusted,
        source_predictions=source_predictions,
        meta_predictions=meta_predictions,
        canonical=canonical,
        minimum=minimum,
    )
    recommendations = _recommendations(
        bottleneck=bottleneck,
        raw=raw,
        adjusted=adjusted,
        source_predictions=source_predictions,
        meta_predictions=meta_predictions,
        needed=needed,
    )
    return {
        "mode": "historical_coverage_audit_research_only",
        "minimum_independent_periods": minimum,
        "target_independent_periods": list(TARGET_PERIODS),
        "label_window_days_median": label_window_days,
        "raw_prices": raw,
        "adjusted_prices": adjusted,
        "source_prediction_artifacts": source_predictions,
        "meta_prediction_artifacts": meta_predictions,
        "canonical_replay": canonical,
        "adjusted_replay": adjusted_replay,
        "possible_leakage_safe_non_overlap_windows": possible,
        "history_required_for_targets": needed,
        "historical_bottleneck": bottleneck,
        "blockers": _blockers(
            bottleneck=bottleneck,
            canonical=canonical,
            adjusted_replay=adjusted_replay,
            minimum=minimum,
        ),
        "recommendations": recommendations,
        "full_model_rerun_required": recommendations["full_model_rerun_required"],
        "overnight_command_if_rerun_justified": (
            "python3.10 main.py --mode ml-research-batch "
            "--config configs/research/regime_transformer_meta_ensemble_v1.yaml "
            "--profile benchmark"
        ),
        "rows": _rows(raw, adjusted, source_predictions, meta_predictions, canonical),
        **RESEARCH_METADATA,
    }
