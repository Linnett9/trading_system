from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.artifacts import MLFeatureCache
from core.research.ml.config import MLExperimentConfig
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.rule_overlay import (
    run_drawdown_risk_diagnostics,
    run_rule_exposure_study,
    run_volatility_managed_walk_forward,
)
from core.research.ml.sector_reference import load_sector_by_symbol


class MLRebalancePipeline:
    """Build and write rebalance research datasets and rule-study artifacts."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        *,
        champion_equity_curve: list[Any],
        champion_selections: list[Any],
        feature_cache: MLFeatureCache | None = None,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._champion_equity_curve = champion_equity_curve
        self._champion_selections = champion_selections
        self._feature_cache = feature_cache or MLFeatureCache(config)

    def build_expanded_rebalance_features(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> MLFeatureBuildResult:
        ml_config = self._ml_config()
        cache_path = Path(
            ml_config.get(
                "expanded_rebalance_dataset_path",
                Path(self._config.get("cache", {}).get("ml_dir", "cache/ml"))
                / "expanded_rebalance_dataset.csv",
            )
        )
        if bool(ml_config.get("read_existing_expanded_rebalance_dataset", False)):
            if not cache_path.exists():
                raise RuntimeError(
                    "ML research batch requires existing expanded rebalance dataset: "
                    f"{cache_path}"
                )
            with cache_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            date_range = None
            if rows:
                date_range = (
                    str(rows[0].get("feature_date", "")),
                    str(rows[-1].get("feature_date", "")),
                )
            return MLFeatureBuildResult(
                rows=rows,
                dropped_rows=feature_result.dropped_rows,
                date_range=date_range,
            )

        expanded_cache_key = self.expanded_rebalance_cache_key(
            feature_result,
            candles_by_symbol,
        )
        if bool(ml_config.get("cache_expanded_rebalance_dataset", False)):
            cached = self._feature_cache.load_expanded_rebalance_rows(
                cache_path,
                expanded_cache_key,
                feature_result.dropped_rows,
            )
            if cached is not None:
                return cached

        from core.research.ml.rebalance_dataset import (
            build_expanded_rebalance_rows,
            write_expanded_rebalance_audit,
            write_rebalance_dataset,
        )

        benchmark = str(self._config.get("ml", {}).get("benchmark_symbols", ["SPY"])[0])
        rows, audit = build_expanded_rebalance_rows(
            dict(self._config),
            feature_result.rows,
            candles_by_symbol,
            benchmark,
            self._experiment_config.label_horizon_days,
            sector_by_symbol=self.sector_by_symbol(),
        )
        audit_path = Path(
            self._config.get("ml", {}).get(
                "expanded_rebalance_audit_path",
                "reports/ml/expanded_rebalance_dataset_audit.json",
            )
        )
        write_rebalance_dataset(cache_path, rows)
        write_expanded_rebalance_audit(audit_path, audit)
        if bool(ml_config.get("cache_expanded_rebalance_dataset", False)):
            self._feature_cache.write_metadata(
                cache_path,
                expanded_cache_key,
                {
                    "cache_type": "expanded_rebalance_dataset",
                    "row_count": len(rows),
                    "date_range": (
                        [str(rows[0]["feature_date"]), str(rows[-1]["feature_date"])]
                        if rows
                        else None
                    ),
                    "audit_path": str(audit_path),
                    "research_only": True,
                    "trading_impact": "none",
                },
            )
        date_range = None
        if rows:
            date_range = (str(rows[0]["feature_date"]), str(rows[-1]["feature_date"]))
        return MLFeatureBuildResult(
            rows=rows,
            dropped_rows=feature_result.dropped_rows,
            date_range=date_range,
        )

    def write_rebalance_dataset(
        self,
        path: Path,
        audit_path: Path,
        feature_rows: list[dict[str, float | str]],
        candles_by_symbol: dict[str, list[Any]],
        rule_study_path: Path,
    ) -> list[dict[str, float | str]]:
        from core.research.ml.rebalance_dataset import (
            build_champion_rebalance_rows,
            write_rebalance_dataset,
        )

        if self._experiment_config.label_type == "should_reduce_exposure":
            write_rebalance_dataset(path, feature_rows)
            audit_path.write_text(json.dumps({
                "row_count": len(feature_rows),
                "should_reduce_exposure_rate": self.row_rate(
                    feature_rows,
                    "should_reduce_exposure",
                ),
                "drawdown_event_rate": self.row_rate(feature_rows, "drawdown_event"),
                "underperforms_spy_rate": self.row_rate(
                    feature_rows,
                    "underperforms_spy",
                ),
                "source": "expanded_rebalance_dataset",
                "research_only": True,
                "trading_impact": "none",
            }, indent=2), encoding="utf-8")
            rule_study_path.write_text(json.dumps({
                "mode": "expanded_rebalance_rule_based_research_only",
                "rules": run_rule_exposure_study(feature_rows, transaction_cost_bps=5.0),
                "volatility_managed_walk_forward": run_volatility_managed_walk_forward(
                    feature_rows,
                    transaction_cost_bps=5.0,
                ),
                "drawdown_risk_diagnostics": run_drawdown_risk_diagnostics(feature_rows),
                "trading_impact": "none",
            }, indent=2), encoding="utf-8")
            return feature_rows

        benchmark = str(self._config.get("ml", {}).get("benchmark_symbols", ["SPY"])[0])
        rows = build_champion_rebalance_rows(
            feature_rows,
            self._champion_selections,
            self._champion_equity_curve,
            candles_by_symbol.get(benchmark, []),
            self._experiment_config.label_horizon_days,
            candles_by_symbol=candles_by_symbol,
            sector_by_symbol=self.sector_by_symbol(),
        )
        write_rebalance_dataset(path, rows)
        audit_path.write_text(json.dumps({
            "row_count": len(rows),
            "good_period_rate": self.row_rate(rows, "good_period"),
            "bad_period_rate": self.row_rate(rows, "bad_period"),
            "underperforms_spy_rate": self.row_rate(rows, "underperforms_spy"),
            "drawdown_event_rate": self.row_rate(rows, "drawdown_event"),
            "history_years": self._config.get("backtest", {}).get("years"),
            "recommended_generalization_years": self._config.get("ml", {}).get(
                "research_years", 10,
            ),
            "minimum_history_years": self._config.get("ml", {}).get(
                "minimum_history_years",
            ),
            "sector_reference_path": self._config.get("ml", {}).get(
                "sector_reference_path",
            ),
            "research_only": True,
        }, indent=2), encoding="utf-8")
        rule_study_path.write_text(json.dumps({
            "mode": "rule_based_research_only",
            "rules": run_rule_exposure_study(rows, transaction_cost_bps=5.0),
            "volatility_managed_walk_forward": run_volatility_managed_walk_forward(
                rows,
                transaction_cost_bps=5.0,
            ),
            "drawdown_risk_diagnostics": run_drawdown_risk_diagnostics(rows),
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")
        return rows

    def expanded_rebalance_cache_key(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_symbol: dict[str, list[Any]],
    ) -> str:
        ml_config = self._ml_config()
        return MLFeatureCache.hash_payload({
            "cache_version": 1,
            "cache_type": "expanded_rebalance_dataset",
            "label_type": self._experiment_config.label_type,
            "label_horizon_days": self._experiment_config.label_horizon_days,
            "expanded_rebalance_dataset": ml_config.get(
                "expanded_rebalance_dataset", {}
            ),
            "benchmark_symbols": ml_config.get("benchmark_symbols", ["SPY", "QQQ"]),
            "sector_reference_path": ml_config.get("sector_reference_path"),
            "sector_by_symbol": ml_config.get("sector_by_symbol", {}),
            "feature_rows_hash": MLFeatureCache.rows_hash(feature_result.rows),
            "feature_row_count": len(feature_result.rows),
            "feature_date_range": feature_result.date_range,
            "history": MLFeatureCache.candles_cache_summary(candles_by_symbol),
        })

    def sector_by_symbol(self) -> dict[str, str]:
        ml_config = self._ml_config()
        return load_sector_by_symbol(
            ml_config.get("sector_reference_path"),
            inline_mapping=dict(ml_config.get("sector_by_symbol", {})),
        )

    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}

    @staticmethod
    def row_rate(rows: list[dict[str, float | str]], key: str) -> float | None:
        return sum(int(row[key]) for row in rows) / len(rows) if rows else None
