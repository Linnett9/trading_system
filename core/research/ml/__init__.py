"""ML research package.

Keep this package initializer lightweight. Importing submodules such as
``core.research.ml.stock_level`` must not eagerly import the entire meta,
allocation, registry, and prediction-artifact graph; several production
builders import framework config during module initialisation.
"""

__all__ = [
    "CoverageMonitoringConfig",
    "CoverageMonitoringIdentity",
    "IMLModel",
    "MLExperimentConfig",
    "NoOpMLModel",
    "cross_sectional_ranking_diagnostics",
    "meta_auxiliary",
    "meta_ensemble",
    "run_forecast_interval_coverage_monitoring",
]


def __getattr__(name):
    if name == "MLExperimentConfig":
        from core.research.ml.config import MLExperimentConfig

        return MLExperimentConfig
    if name in {"IMLModel", "NoOpMLModel"}:
        from core.research.ml.models import IMLModel, NoOpMLModel

        return {"IMLModel": IMLModel, "NoOpMLModel": NoOpMLModel}[name]
    if name in {"meta_auxiliary", "meta_ensemble"}:
        from core.research.ml.meta import meta_auxiliary, meta_ensemble

        return {"meta_auxiliary": meta_auxiliary, "meta_ensemble": meta_ensemble}[name]
    if name == "cross_sectional_ranking_diagnostics":
        from core.research.ml.metrics import cross_sectional_ranking_diagnostics

        return cross_sectional_ranking_diagnostics
    if name in {
        "CoverageMonitoringConfig",
        "CoverageMonitoringIdentity",
        "run_forecast_interval_coverage_monitoring",
    }:
        from core.research.ml.forecast_interval_coverage_monitoring import (
            CoverageMonitoringConfig,
            CoverageMonitoringIdentity,
            run_forecast_interval_coverage_monitoring,
        )

        return {
            "CoverageMonitoringConfig": CoverageMonitoringConfig,
            "CoverageMonitoringIdentity": CoverageMonitoringIdentity,
            "run_forecast_interval_coverage_monitoring": run_forecast_interval_coverage_monitoring,
        }[name]
    raise AttributeError(name)
