from __future__ import annotations

from typing import Any

__all__ = [
    "MLCalibrationReportWriter",
    "MLDiagnosticReportWriter",
    "MLOverlayReportWriter",
]


def __getattr__(name: str) -> Any:
    if name == "MLCalibrationReportWriter":
        from core.research.ml.reports.calibration_reports import (
            MLCalibrationReportWriter,
        )

        return MLCalibrationReportWriter
    if name == "MLDiagnosticReportWriter":
        from core.research.ml.reports.diagnostic_reports import (
            MLDiagnosticReportWriter,
        )

        return MLDiagnosticReportWriter
    if name == "MLOverlayReportWriter":
        from core.research.ml.reports.overlay_reports import MLOverlayReportWriter

        return MLOverlayReportWriter
    raise AttributeError(name)
