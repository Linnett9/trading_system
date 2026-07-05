from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.audits.adjusted_data_comparison import (
    RESEARCH_METADATA,
    _adjusted_close_by_date,
    _comparison_config,
    _load_adjusted_rows_by_symbol,
    _load_raw_stooq_rows_by_symbol,
    _output_dir,
    _raw_close_by_date,
    _read_json,
    _symbols_to_compare,
)
from core.research.ml.audits.adjusted_replay_alignment_config import (
    _alignment_config,
    _normalize_audit_config,
)
from core.research.ml.audits.adjusted_replay_alignment_math import (
    _close,
    _count,
    _delta,
    _expected_adjusted_return,
    _fmt,
    _max_abs,
    _mismatch,
    _period_return,
    _ratio,
    _top_rows,
)
from core.research.ml.audits.adjusted_replay_alignment_reporting import (
    _markdown,
    _write_csv,
)
from core.research.ml.audits.adjusted_replay_alignment_rows import (
    _alignment_row,
    _candidate_alignment_rows,
    _rows_by_date,
    _symbols,
)
from core.research.ml.audits.adjusted_replay_alignment_summary import (
    _alignment_summary,
    _candidate_summary,
    _explanation_verdict,
    _red_flags,
)
from core.research.ml.audits.adjusted_replay_alignment_types import (
    REPORT_CANDIDATES,
    AdjustedReplayAlignmentAuditPaths,
)


def write_adjusted_replay_alignment_audit(
    config: dict[str, Any],
) -> AdjustedReplayAlignmentAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    adjusted_replay = _read_json(output_dir / "adjusted_price_replay.json")
    comparison_config = _comparison_config(config)
    alignment_config = _alignment_config(config)
    symbols = _symbols_to_compare(canonical, comparison_config)
    raw_rows = _load_raw_stooq_rows_by_symbol(
        Path(str(comparison_config["stooq_parquet_dir"])),
        symbols,
    )
    adjusted_rows = _load_adjusted_rows_by_symbol(comparison_config, symbols)
    payload = build_adjusted_replay_alignment_audit(
        canonical_replay=canonical,
        adjusted_price_replay=adjusted_replay,
        raw_closes_by_symbol={
            symbol: _raw_close_by_date(rows)
            for symbol, rows in raw_rows.items()
        },
        adjusted_closes_by_symbol={
            symbol: _adjusted_close_by_date(rows)
            for symbol, rows in adjusted_rows.items()
        },
        audit_config=alignment_config,
    )
    paths = AdjustedReplayAlignmentAuditPaths(
        csv_path=output_dir / "adjusted_replay_alignment_audit.csv",
        json_path=output_dir / "adjusted_replay_alignment_audit.json",
        markdown_path=output_dir / "adjusted_replay_alignment_audit.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_adjusted_replay_alignment_audit(
    *,
    canonical_replay: dict[str, Any],
    adjusted_price_replay: dict[str, Any],
    raw_closes_by_symbol: dict[str, dict[str, float]],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    audit_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _normalize_audit_config(audit_config or {})
    adjusted_canonical = adjusted_price_replay.get("adjusted_canonical_replay", {})
    replay_candidates = adjusted_price_replay.get("candidates", {}) or {}
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in REPORT_CANDIDATES:
        candidate_rows = _candidate_alignment_rows(
            candidate,
            canonical_replay,
            adjusted_canonical,
            raw_closes_by_symbol,
            adjusted_closes_by_symbol,
            config,
        )
        rows.extend(candidate_rows)
        summaries[candidate] = _candidate_summary(
            candidate,
            candidate_rows,
            replay_candidates.get(candidate, {})
            if isinstance(replay_candidates, dict)
            else {},
        )
    alignment = _alignment_summary(rows, summaries)
    return {
        "mode": "adjusted_replay_alignment_audit_research_only",
        "replay_semantics": (
            "raw canonical replay rows compared against adjusted canonical replay "
            "rows using the same rebalance windows, symbols, exposures, and "
            "non-overlap flags"
        ),
        "audit_config": config,
        "candidate_summaries": summaries,
        "alignment": alignment,
        "biggest_return_deltas": _top_rows(
            rows,
            "return_delta",
            limit=int(config["top_delta_rows"]),
        ),
        "biggest_candidate_net_return_deltas": _top_rows(
            rows,
            "candidate_net_return_delta",
            limit=int(config["top_delta_rows"]),
        ),
        "rows": rows,
        "red_flags": _red_flags(alignment),
        **RESEARCH_METADATA,
    }
