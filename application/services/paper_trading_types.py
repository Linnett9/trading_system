from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperTradingRunResult:
    run_id: str
    candidate_id: str
    dry_run: bool
    decision: Any
    fill_record: dict[str, Any] | None
    risk_checks: list[Any]
    post_trade_checks: list[Any]
    risk_report_path: Path
    reconciliation_report_path: Path | None
    order_preview_path: Path
    journal_path: Path
    fill_log_path: Path
    event_log_path: Path
    dashboard_path: Path
    metrics_summary_path: Path
    approval_path: Path | None
    blocked_reason: str | None = None
