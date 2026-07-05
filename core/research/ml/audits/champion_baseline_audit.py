from __future__ import annotations

import json
from typing import Any

from core.research.ml.audits.champion_baseline_audit_io import (
    _champion_config_path,
    _expanded_dataset_path,
    _meta_dataset_path,
    _meta_output_dir,
    _read_csv,
    _read_json,
    _read_yaml,
)
from core.research.ml.audits.champion_baseline_audit_diagnostics import (
    _attach_expanded_symbols,
    _configured_evaluation_dates,
    _diagnostic_baseline_rows,
    _evaluation_periods,
    _expanded_rows_by_date,
    _late_period_dominance,
    _red_flags,
    _return_audit_candidate,
    _stooq_adjustment_audit,
    _top_date_report,
    _v2_vs_exact,
)
from core.research.ml.audits.champion_baseline_audit_math import (
    _annualized_return,
    _compound_returns,
    _equity_curve,
    _fmt,
    _max_drawdown,
    _number,
    _observed_periods_per_year,
    _sharpe,
    _sortino,
)
from core.research.ml.audits.champion_baseline_audit_reporting import (
    _markdown,
    _write_csv,
)
from core.research.ml.audits.champion_baseline_audit_replay import (
    _active_champion_config,
    _continuous_summary,
    _load_replay_candles,
    _period_grid_summary,
    _read_parquet_candles,
    _required_replay_symbols,
    _selection_at_or_before,
    _selection_lookup,
    _symbol_return_anomalies,
    _top_periods,
    _try_exact_champion_replay,
    _unavailable_exact_replay,
    exact_champion_replay_from_equity,
)
from core.research.ml.audits.champion_baseline_audit_types import (
    ChampionBaselineAuditPaths,
    RESEARCH_METADATA,
)


def write_champion_baseline_audit(config: dict[str, Any]) -> ChampionBaselineAuditPaths:
    output_dir = _meta_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_rows = _read_csv(_expanded_dataset_path(config))
    meta_rows = _read_csv(_meta_dataset_path(config))
    return_audit = _read_json(output_dir / "benchmark_return_audit.json")
    periods = _evaluation_periods(config, meta_rows, expanded_rows)
    diagnostic_rows = _diagnostic_baseline_rows(return_audit)
    exact_replay = _try_exact_champion_replay(config, periods)
    top_date_report = _top_date_report(
        return_audit,
        exact_replay,
        expanded_rows,
    )
    payload = {
        "mode": "champion_baseline_audit_research_only",
        "baseline_semantics": {
            "champion_return_next_period_created_in": (
                "core/research/ml/rebalance_dataset.py::build_champion_rebalance_rows"
            ),
            "champion_return_next_period_represents": (
                "dual-momentum backtester equity-curve return for each expanded "
                "variant row over the configured label horizon"
            ),
            "current_allocation_champion_baseline": (
                "full allocation exposure applied to the date-averaged expanded "
                "variant return series"
            ),
            "current_champion_baseline_is_exact_champion_replay": False,
            "why_champion_baseline_equals_always_full_exposure": (
                "both use constant allocation exposure of 1.0 in allocation_v2"
            ),
            "equality_is_misleading": True,
        },
        "baseline_rows": diagnostic_rows + [exact_replay["summary"]],
        "exact_champion_replay": exact_replay,
        "v2_vs_exact_champion": _v2_vs_exact(return_audit, exact_replay),
        "top_date_report": top_date_report,
        "stooq_adjustment_audit": _stooq_adjustment_audit(
            config,
            exact_replay,
        ),
        "red_flags": _red_flags(exact_replay, return_audit),
        **RESEARCH_METADATA,
    }
    paths = ChampionBaselineAuditPaths(
        csv_path=output_dir / "champion_baseline_audit.csv",
        json_path=output_dir / "champion_baseline_audit.json",
        markdown_path=output_dir / "champion_baseline_audit.md",
    )
    _write_csv(paths.csv_path, payload["baseline_rows"])
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths
