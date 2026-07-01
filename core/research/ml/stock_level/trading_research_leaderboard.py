from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.trading_research_leaderboard_augmentation import (
    _augment_with_benchmark_validation,
    _augment_with_canonical_metrics,
    _refresh_robustness_metrics,
    _scenario_return,
)
from core.research.ml.trading_research_leaderboard_io import (
    _markdown,
    _read_json,
    _write_csv,
)
from core.research.ml.trading_research_leaderboard_math import (
    _ascending,
    _descending,
    _drawdown_magnitude,
    _first_number,
    _format,
    _number,
)
from core.research.ml.trading_research_leaderboard_ranking import _ranking_key
from core.research.ml.trading_research_leaderboard_rows import (
    _allocation_trading_rows,
    _benchmark_validation_rows,
    _canonical_only_rows,
    _classification_diagnostics,
    _classification_rows,
    _classification_trading_rows,
    _optimizer_trading_row,
    _trading_row,
)
from core.research.ml.trading_research_leaderboard_types import (
    NOTICE,
    RANKING_BASIS,
    RESEARCH_METADATA,
    TradingResearchLeaderboardPaths,
)


def write_trading_research_leaderboard(
    output_dir: Path,
    classification_leaderboard_path: Path,
    allocation_comparison_path: Path,
    optimizer_results_path: Path,
    auxiliary_metrics_path: Path,
) -> TradingResearchLeaderboardPaths:
    """Combine research reports without changing their source selection logic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classification = _read_json(classification_leaderboard_path)
    allocation = _read_json(allocation_comparison_path)
    optimizer = _read_json(optimizer_results_path)
    auxiliary = _read_json(auxiliary_metrics_path)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    concentration = _read_json(output_dir / "profit_concentration_audit.json")
    benchmark_validation = _read_json(
        output_dir / "benchmark_relative_validation.json"
    )

    classification_rows = _classification_rows(classification)
    trading_rows = [
        *_classification_trading_rows(classification_rows),
        *_allocation_trading_rows(allocation),
    ]
    optimizer_row = _optimizer_trading_row(optimizer)
    if optimizer_row is not None:
        trading_rows.append(optimizer_row)
    trading_rows.extend(_canonical_only_rows(canonical, trading_rows))
    trading_rows.extend(
        _benchmark_validation_rows(benchmark_validation, trading_rows)
    )
    _augment_with_canonical_metrics(trading_rows, canonical, concentration)
    _augment_with_benchmark_validation(trading_rows, benchmark_validation)
    canonical_ranking_available = any(
        _number(row.get("canonical_continuous_return")) is not None
        for row in trading_rows
    )
    for row in trading_rows:
        row["canonical_ranking_available"] = canonical_ranking_available

    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(sorted(trading_rows, key=_ranking_key), start=1)
    ]
    payload = {
        "mode": "trading_research_leaderboard",
        "ranking_basis": RANKING_BASIS,
        "canonical_ranking_available": canonical_ranking_available,
        "classification_metrics_role": "diagnostics_only",
        "leaderboard": ranked_rows,
        "classification_diagnostics": _classification_diagnostics(
            classification_rows
        ),
        "meta_auxiliary_forecast_metrics": auxiliary.get("targets", {}),
        "meta_auxiliary_available_targets": auxiliary.get(
            "available_targets", []
        ),
        "source_artifacts": {
            "base_and_meta_classification": str(classification_leaderboard_path),
            "allocation_v2": str(allocation_comparison_path),
            "allocation_optimizer": str(optimizer_results_path),
            "meta_auxiliary": str(auxiliary_metrics_path),
            "canonical_continuous_equity_replay": str(
                output_dir / "canonical_continuous_equity_replay.json"
            ),
            "profit_concentration_audit": str(
                output_dir / "profit_concentration_audit.json"
            ),
            "benchmark_relative_validation": str(
                output_dir / "benchmark_relative_validation.json"
            ),
        },
        "optimizer_status": {
            "sampler_requested": optimizer.get("sampler_requested"),
            "sampler_used": optimizer.get("sampler_used"),
            "objective_mode": optimizer.get("objective_mode"),
            "fallback_reason": optimizer.get("fallback_reason"),
            "skip_reason": optimizer.get("skip_reason"),
        },
        **RESEARCH_METADATA,
    }

    paths = TradingResearchLeaderboardPaths(
        csv_path=output_dir / "trading_research_leaderboard.csv",
        json_path=output_dir / "trading_research_leaderboard.json",
        markdown_path=output_dir / "trading_research_leaderboard.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, ranked_rows)
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths
