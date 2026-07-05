from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    severity: RiskSeverity
    reason: str
    details: dict[str, Any]
