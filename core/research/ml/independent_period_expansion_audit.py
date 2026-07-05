from __future__ import annotations

from core.research.ml.audits.independent_period_expansion_audit import (
    REPORT_CANDIDATES,
    RESEARCH_METADATA,
    IndependentPeriodExpansionAuditPaths,
    build_independent_period_expansion_audit,
    write_independent_period_expansion_audit,
)
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
from core.research.ml.independent_period_expansion_audit_io import _fmt, _markdown, _write_csv
from core.research.ml.independent_period_expansion_audit_math import (
    _compound,
    _date,
    _max_drawdown,
    _number,
    _top_positive_share,
)
from core.research.ml.independent_period_expansion_audit_metrics import (
    _benchmark_return,
    _red_flags,
    _safest_expansion,
    _setting_metrics,
    _source_gates_preserved,
)
from core.research.ml.independent_period_expansion_audit_selection import (
    _first_periods_by_bucket,
    _select_periods,
)
from core.research.ml.independent_period_expansion_audit_sources import (
    _load_adjusted_closes,
    _read_json,
)

__all__ = [
    "REPORT_CANDIDATES",
    "RESEARCH_METADATA",
    "IndependentPeriodExpansionAuditPaths",
    "build_independent_period_expansion_audit",
    "write_independent_period_expansion_audit",
    "_benchmark_return",
    "_candidate_adjusted_rows",
    "_candidate_coverage",
    "_candidate_summary",
    "_compound",
    "_date",
    "_expansion_config",
    "_first_periods_by_bucket",
    "_fmt",
    "_load_adjusted_closes",
    "_markdown",
    "_max_drawdown",
    "_no_selected_symbol_rows",
    "_no_selected_symbol_summary",
    "_normalized_expansion_config",
    "_number",
    "_output_dir",
    "_read_json",
    "_red_flags",
    "_safest_expansion",
    "_select_periods",
    "_setting_metrics",
    "_source_gates_preserved",
    "_top_positive_share",
    "_validation_config",
    "_write_csv",
]
