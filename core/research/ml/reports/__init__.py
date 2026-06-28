from __future__ import annotations

from typing import Any

__all__ = ["MLCalibrationReportWriter"]


def __getattr__(name: str) -> Any:
    if name == "MLCalibrationReportWriter":
        from core.research.ml.reports.calibration_reports import (
            MLCalibrationReportWriter,
        )

        return MLCalibrationReportWriter
    raise AttributeError(name)
