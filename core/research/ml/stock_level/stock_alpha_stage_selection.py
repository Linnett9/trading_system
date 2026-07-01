from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.research.framework.config import StockLevelResearchConfig


STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "stock_artifact": (),
    "alpha_features": ("stock_artifact",),
    "baseline_benchmark": ("stock_artifact",),
    "enriched_benchmark": ("alpha_features",),
    "target_comparison": ("enriched_benchmark",),
    "portfolio_replay": ("enriched_benchmark",),
    "portfolio_policy_sweep": ("enriched_benchmark",),
    "experiment_report": (),
    "attribution": ("enriched_benchmark",),
}


@dataclass(frozen=True)
class StockAlphaStageSpec:
    name: str
    dependencies: tuple[str, ...] = ()
    downstream_stale_on_change: bool = False


@dataclass(frozen=True)
class StockAlphaStageSelection:
    requested: dict[str, bool]
    configured_stage_names: tuple[str, ...] = ()

    def enabled(self, stage_name: str) -> bool:
        return self.requested[stage_name]

    @property
    def requested_stages(self) -> list[str]:
        return [name for name, enabled in self.requested.items() if enabled]

    @property
    def skipped_by_user(self) -> list[str]:
        configured = set(self.configured_stage_names)
        return [name for name, enabled in self.requested.items() if not enabled and name in configured]

    def payload(self) -> dict[str, Any]:
        return {
            "configured": bool(self.configured_stage_names),
            "requested_stages": self.requested_stages,
            "skipped_by_user": self.skipped_by_user,
            "stage_flags": dict(self.requested),
        }


class StockAlphaStageSelector:
    def __init__(
        self,
        config: Mapping[str, Any],
        settings: StockLevelResearchConfig,
        *,
        output_exists: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.specs = {
            name: StockAlphaStageSpec(name=name, dependencies=dependencies)
            for name, dependencies in STAGE_DEPENDENCIES.items()
        }
        self._output_exists = output_exists

    def resolve(self) -> StockAlphaStageSelection:
        defaults = self._default_stage_flags()
        configured = dict(self.config.get("ml", {}) or {}).get("stock_alpha_stages")
        if configured is None:
            return StockAlphaStageSelection(defaults)
        if not isinstance(configured, Mapping):
            raise ValueError("ml.stock_alpha_stages must be a mapping of stage name to boolean")
        unknown = sorted(set(configured) - set(self.specs))
        if unknown:
            raise ValueError(f"Unknown stock-alpha stage name(s): {', '.join(unknown)}")
        selected = dict(defaults)
        for name, enabled in configured.items():
            selected[str(name)] = bool(enabled)
        self._validate_dependencies(selected)
        return StockAlphaStageSelection(selected, tuple(str(name) for name in configured))

    def _default_stage_flags(self) -> dict[str, bool]:
        ml = dict(self.config.get("ml", {}) or {})
        return {
            "stock_artifact": True,
            "alpha_features": True,
            "baseline_benchmark": True,
            "enriched_benchmark": True,
            "target_comparison": self.settings.target_comparison_enabled,
            "portfolio_replay": self.settings.overnight_run_portfolio_replay,
            "portfolio_policy_sweep": bool(
                ml.get(
                    "stock_alpha_overnight_run_portfolio_policy_sweep",
                    self.settings.run_size != "full",
                )
            ),
            "experiment_report": bool(ml.get("stock_alpha_experiment_report_enabled", True)),
            "attribution": self.settings.overnight_run_attribution,
        }

    def _validate_dependencies(self, selected: dict[str, bool]) -> None:
        for stage_name, enabled in selected.items():
            if not enabled:
                continue
            for dependency in self.specs[stage_name].dependencies:
                if selected.get(dependency, False):
                    continue
                if stage_name == "enriched_benchmark" and dependency == "alpha_features":
                    if self._explicit_enriched_artifact_exists():
                        continue
                    raise ValueError(
                        "Stock-alpha stage 'enriched_benchmark' requires alpha_features "
                        "or an existing ml.stock_level_prediction_artifacts_path pointing "
                        "to an enriched artifact."
                    )
                if self._dependency_output_exists(dependency):
                    continue
                raise ValueError(
                    f"Stock-alpha stage '{stage_name}' requires upstream stage "
                    f"'{dependency}', but '{dependency}' is disabled and no compatible "
                    "existing output is available."
                )

    def _dependency_output_exists(self, stage_name: str) -> bool:
        if self._output_exists is not None:
            return self._output_exists(stage_name)
        from core.research.ml.stock_level.run_manifest.paths import expected_stage_output_paths

        output_dir = self.settings.output_dir
        paths = expected_stage_output_paths(self.config, output_dir).get(stage_name, {})
        return bool(paths) and all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths.values())

    def _explicit_enriched_artifact_exists(self) -> bool:
        ml = dict(self.config.get("ml", {}) or {})
        raw_path = ml.get("stock_level_prediction_artifacts_path")
        if not raw_path:
            return False
        path = Path(str(raw_path))
        if path == self.settings.output_dir / "stock_level_prediction_artifacts.csv":
            return False
        if path.name != "stock_level_prediction_artifacts_enriched.csv":
            return False
        return path.exists() and path.stat().st_size > 0
