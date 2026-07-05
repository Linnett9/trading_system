from __future__ import annotations

import json
from typing import Any

from core.research.ml.independent_period_expansion_audit_candidates import (
    _candidate_adjusted_rows,
    _candidate_coverage,
    _candidate_summary,
    _no_selected_symbol_rows,
    _no_selected_symbol_summary,
)
from core.research.ml.independent_period_expansion_audit_config import (
    _expansion_config,
    _normalized_expansion_config,
    _output_dir,
    _validation_config,
)
from core.research.ml.independent_period_expansion_audit_io import _markdown, _write_csv
from core.research.ml.independent_period_expansion_audit_metrics import (
    _red_flags,
    _safest_expansion,
    _setting_metrics,
)
from core.research.ml.independent_period_expansion_audit_selection import _select_periods
from core.research.ml.independent_period_expansion_audit_sources import (
    _load_adjusted_closes,
    _read_json,
)
from core.research.ml.independent_period_expansion_audit_types import (
    REPORT_CANDIDATES,
    RESEARCH_METADATA,
    IndependentPeriodExpansionAuditPaths,
)


def write_independent_period_expansion_audit(
    config: dict[str, Any],
) -> IndependentPeriodExpansionAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    adjusted_replay = _read_json(output_dir / "adjusted_price_replay.json")
    canonical_replay = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    data_adjustment = _read_json(output_dir / "data_adjustment_audit.json")
    benchmark_validation = _read_json(output_dir / "benchmark_relative_validation.json")
    adjusted_alignment = _read_json(output_dir / "adjusted_replay_alignment_audit.json")
    adjusted_closes = _load_adjusted_closes(config)
    payload = build_independent_period_expansion_audit(
        adjusted_price_replay=adjusted_replay,
        canonical_replay=canonical_replay,
        data_adjustment_audit=data_adjustment,
        benchmark_relative_validation=benchmark_validation,
        adjusted_replay_alignment_audit=adjusted_alignment,
        adjusted_closes_by_symbol=adjusted_closes,
        validation_config=_validation_config(config),
        expansion_config=_expansion_config(config),
    )
    paths = IndependentPeriodExpansionAuditPaths(
        csv_path=output_dir / "independent_period_expansion_audit.csv",
        json_path=output_dir / "independent_period_expansion_audit.json",
        markdown_path=output_dir / "independent_period_expansion_audit.md",
    )
    _write_csv(paths.csv_path, payload["rows"])
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_independent_period_expansion_audit(
    *,
    adjusted_price_replay: dict[str, Any],
    canonical_replay: dict[str, Any],
    data_adjustment_audit: dict[str, Any] | None = None,
    benchmark_relative_validation: dict[str, Any] | None = None,
    adjusted_replay_alignment_audit: dict[str, Any] | None = None,
    adjusted_closes_by_symbol: dict[str, dict[str, float]] | None = None,
    validation_config: dict[str, Any] | None = None,
    expansion_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config or {}
    expansion = _normalized_expansion_config(expansion_config or {})
    minimum = int(config.get("min_independent_periods", 36))
    suspicious_dates = set(
        (data_adjustment_audit or {}).get("suspicious_rebalance_dates", []) or []
    )
    alignment = (adjusted_replay_alignment_audit or {}).get("alignment", {})
    adjusted_closes_by_symbol = adjusted_closes_by_symbol or {}
    no_selection_rows = _no_selected_symbol_rows(
        adjusted_price_replay=adjusted_price_replay,
        canonical_replay=canonical_replay,
    )
    rows: list[dict[str, Any]] = []
    settings: dict[str, dict[str, Any]] = {}
    for candidate_name in REPORT_CANDIDATES:
        adjusted_rows = _candidate_adjusted_rows(adjusted_price_replay, candidate_name)
        coverage = _candidate_coverage(adjusted_price_replay, candidate_name)
        candidate_settings = []
        for setting in expansion["settings"]:
            selected, skipped = _select_periods(adjusted_rows, setting)
            metrics = _setting_metrics(
                candidate_name=candidate_name,
                setting=setting,
                selected_rows=selected,
                skipped_rows=skipped,
                coverage=coverage,
                suspicious_dates=suspicious_dates,
                adjusted_closes_by_symbol=adjusted_closes_by_symbol,
                minimum_independent_periods=minimum,
                adjusted_alignment=alignment,
                benchmark_relative_validation=benchmark_relative_validation or {},
            )
            rows.append(metrics)
            candidate_settings.append(metrics)
        settings[candidate_name] = {
            "candidate_name": candidate_name,
            "settings": candidate_settings,
            "safest_expansion": _safest_expansion(candidate_settings, minimum),
            **RESEARCH_METADATA,
        }
    return {
        "mode": "independent_adjusted_period_expansion_audit_research_only",
        "purpose": (
            "Compare leakage-safe adjusted replay period selections without "
            "lowering promotion gates or rerunning models."
        ),
        "minimum_independent_periods": minimum,
        "promotion_thresholds_changed": False,
        "leakage_safety_rule": (
            "A selected period must start after the previous selected label window "
            "ends plus the configured minimum gap."
        ),
        "current_valid_adjusted_independent_periods": {
            name: int(
                _candidate_summary(adjusted_price_replay, name).get(
                    "valid_adjusted_independent_period_count"
                )
                or 0
            )
            for name in REPORT_CANDIDATES
        },
        "no_selected_symbols": no_selection_rows,
        "no_selected_symbol_summary": _no_selected_symbol_summary(no_selection_rows),
        "candidate_settings": settings,
        "rows": rows,
        "red_flags": _red_flags(rows, minimum),
        **RESEARCH_METADATA,
    }
