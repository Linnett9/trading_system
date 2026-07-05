from __future__ import annotations

import json
from typing import Any

from core.interfaces.data_feed import IDataFeed
from core.research.ml.audits.benchmark_relative_validation_types import COST_STRESS_BPS, RESEARCH_METADATA, BenchmarkRelativeValidationPaths
from core.research.ml.audits.benchmark_relative_validation_builder import build_benchmark_relative_validation
from core.research.ml.audits.benchmark_relative_validation_baselines import _baseline_row, _canonical_candidate, _canonical_schedule, _market_baseline, _price_return, _selected_universe_baseline
from core.research.ml.audits.benchmark_relative_validation_scoring import _merge_existing_concentration, _score_candidate, _symbol_contributions, _turnover_by_row
from core.research.ml.audits.benchmark_relative_validation_gates import _adjusted_price_status_acceptable, _apply_gates, _external_promotion_gate_context
from core.research.ml.audits.benchmark_relative_validation_io import _load_required_closes, _output_dir, _read_json
from core.research.ml.audits.benchmark_relative_validation_math import _compound, _equity_curve, _fmt, _max_drawdown, _number, _periods_per_year, _return, _sharpe, _sortino
from core.research.ml.audits.benchmark_relative_validation_reporting import _markdown, _promotion_markdown, _write_csv


def write_benchmark_relative_validation(
    config: dict[str, Any],
    data_feed: IDataFeed,
) -> BenchmarkRelativeValidationPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    anomaly = _read_json(output_dir / "anomaly_quarantine_report.json")
    concentration = _read_json(output_dir / "profit_concentration_audit.json")
    external_reports = {
        "data_adjustment_audit": _read_json(output_dir / "data_adjustment_audit.json"),
        "clean_data_replay": _read_json(output_dir / "clean_data_replay.json"),
        "independent_period_validation": _read_json(
            output_dir / "independent_period_validation.json"
        ),
        "adjusted_data_comparison": _read_json(
            output_dir / "adjusted_data_comparison.json"
        ),
        "adjusted_price_replay": _read_json(output_dir / "adjusted_price_replay.json"),
        "adjusted_replay_alignment_audit": _read_json(
            output_dir / "adjusted_replay_alignment_audit.json"
        ),
    }
    closes = _load_required_closes(config, data_feed, canonical)
    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report=anomaly,
        concentration_report=concentration,
        closes_by_symbol=closes,
        validation_config=config.get("ml", {}).get(
            "benchmark_relative_validation",
            {},
        ),
        external_reports=external_reports,
    )
    paths = BenchmarkRelativeValidationPaths(
        csv_path=output_dir / "benchmark_relative_validation.csv",
        json_path=output_dir / "benchmark_relative_validation.json",
        markdown_path=output_dir / "benchmark_relative_validation.md",
        promotion_readiness_path=output_dir / "promotion_readiness_report.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload.get("candidates", []))
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    paths.promotion_readiness_path.write_text(
        _promotion_markdown(payload),
        encoding="utf-8",
    )
    return paths
