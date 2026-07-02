from __future__ import annotations

from core.risk.paper_risk_types import RiskCheckResult, RiskSeverity


def risk_status(checks: list[RiskCheckResult]) -> str:
    if any(check.severity == RiskSeverity.CRITICAL for check in checks):
        return "CRITICAL"
    if any(check.severity == RiskSeverity.ERROR for check in checks):
        return "ERROR"
    if any(check.severity == RiskSeverity.WARNING for check in checks):
        return "WARNING"
    return "PASS"
def risk_blocks_submission(checks: list[RiskCheckResult]) -> bool:
    return any(
        check.severity in {RiskSeverity.ERROR, RiskSeverity.CRITICAL}
        for check in checks
    )
