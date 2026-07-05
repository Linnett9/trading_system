from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.config import MLExperimentConfig
from core.research.ml.artifacts import MLExperimentPathBuilder, MLFeatureCache
from core.research.ml.features import (
    HistoricalFeatureBuilder,
    MLFeatureBuildResult,
    add_champion_state_features,
)
from core.research.ml.history_coverage import (
    assess_history_coverage,
    write_history_coverage_report,
)


@dataclass(frozen=True)
class MLFeaturePipelineResult:
    feature_result: MLFeatureBuildResult
    candles_by_symbol: dict[str, list[Any]]
    champion_equity_curve: list[Any]
    champion_rebalance_dates: set[str]
    champion_selections: list[Any]
    history_data_metadata: dict[str, dict]
    multi_timeframe_candles_by_timeframe: dict[str, dict[str, list[Any]]] | None = None
    multi_timeframe_metadata_by_timeframe: dict[str, dict[str, dict]] | None = None
    champion_state_updated: bool = False
    history_data_metadata_updated: bool = False


class MLFeaturePipeline:
    """Build historical ML features and related runner state."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        *,
        feed: Any = None,
        research_label: str = "UNSPECIFIED_RESEARCH",
        feature_cache: MLFeatureCache | None = None,
        path_builder: MLExperimentPathBuilder | None = None,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._feed = feed
        self._research_label = research_label
        self._feature_cache = feature_cache or MLFeatureCache(config)
        self._path_builder = path_builder or MLExperimentPathBuilder(
            config,
            experiment_config,
        )

    def build(self) -> MLFeaturePipelineResult:
        if self._feed is None:
            return MLFeaturePipelineResult(
                feature_result=MLFeatureBuildResult(
                    rows=[],
                    dropped_rows=0,
                    date_range=None,
                ),
                candles_by_symbol={},
                champion_equity_curve=[],
                champion_rebalance_dates=set(),
                champion_selections=[],
                history_data_metadata={},
                multi_timeframe_candles_by_timeframe={},
                multi_timeframe_metadata_by_timeframe={},
            )

        symbols = self.feature_symbols()
        loaded_candles = self._load_candles(symbols)
        candles_by_symbol = {
            symbol: result.candles for symbol, result in loaded_candles.items()
        }
        history_data_metadata = {
            symbol: result.metadata for symbol, result in loaded_candles.items()
        }
        ml_config = self._ml_config()
        self.validate_history_coverage(
            candles_by_symbol,
            ml_config,
            history_data_metadata,
        )
        benchmark_symbols = tuple(ml_config.get("benchmark_symbols", ["SPY", "QQQ"]))
        if len(benchmark_symbols) < 2:
            raise ValueError("ml.benchmark_symbols must contain at least SPY and QQQ")

        builder = HistoricalFeatureBuilder(
            benchmark_symbols=tuple(str(symbol) for symbol in benchmark_symbols),
            lookback_days=int(ml_config.get("feature_lookback_days", 252)),
        )
        feature_result = self._build_or_load_historical_features(
            symbols,
            benchmark_symbols,
            builder,
            candles_by_symbol,
        )
        multi_timeframe_candles_by_timeframe = {}
        multi_timeframe_metadata_by_timeframe = {}
        if self._multi_timeframe_enabled(ml_config):
            loaded_by_timeframe = self._load_multi_timeframe_candles(
                symbols,
                ml_config,
            )
            multi_timeframe_candles_by_timeframe = {
                timeframe: {
                    symbol: result.candles for symbol, result in loaded.items()
                }
                for timeframe, loaded in loaded_by_timeframe.items()
            }
            multi_timeframe_metadata_by_timeframe = {
                timeframe: {
                    symbol: result.metadata for symbol, result in loaded.items()
                }
                for timeframe, loaded in loaded_by_timeframe.items()
            }
            feature_result = self._add_multi_timeframe_features(
                feature_result,
                multi_timeframe_candles_by_timeframe,
                ml_config,
            )
        champion_equity_curve: list[Any] = []
        champion_rebalance_dates: set[str] = set()
        champion_selections: list[Any] = []
        champion_state_updated = False
        if (
            ml_config.get("include_champion_state_features", True)
            and self._config.get("research", {}).get("dual_momentum")
        ):
            champion_result = self._run_champion_research(candles_by_symbol)
            champion_state_updated = True
            champion_equity_curve = champion_result.result.equity_curve
            champion_selections = champion_result.selections
            champion_rebalance_dates = {
                selection.timestamp.date().isoformat()
                for selection in champion_result.selections
            }
            feature_result = MLFeatureBuildResult(
                rows=add_champion_state_features(
                    feature_result.rows,
                    champion_result.selections,
                ),
                dropped_rows=feature_result.dropped_rows,
                date_range=feature_result.date_range,
            )
        return MLFeaturePipelineResult(
            feature_result=feature_result,
            candles_by_symbol=candles_by_symbol,
            champion_equity_curve=champion_equity_curve,
            champion_rebalance_dates=champion_rebalance_dates,
            champion_selections=champion_selections,
            history_data_metadata=history_data_metadata,
            multi_timeframe_candles_by_timeframe=multi_timeframe_candles_by_timeframe,
            multi_timeframe_metadata_by_timeframe=multi_timeframe_metadata_by_timeframe,
            champion_state_updated=champion_state_updated,
            history_data_metadata_updated=True,
        )

    def feature_symbols(self) -> list[str]:
        dual_momentum = self._config.get("research", {}).get("dual_momentum", {})
        symbols = dual_momentum.get(
            "symbols",
            self._config.get("backtest", {}).get("symbols", []),
        )
        universe_path = dual_momentum.get("universe_path")
        if universe_path:
            try:
                import yaml

                payload = yaml.safe_load(Path(str(universe_path)).read_text(
                    encoding="utf-8"
                )) or {}
                symbols = payload.get("symbols", symbols)
            except FileNotFoundError:
                symbols = []
        if self._experiment_config.label_type == "should_reduce_exposure":
            symbols = [
                *symbols,
                *self.expanded_rebalance_universe_symbols(dual_momentum),
            ]
        benchmarks = self._ml_config().get("benchmark_symbols", ["SPY", "QQQ"])
        return list(dict.fromkeys([*symbols, *benchmarks]))

    def expanded_rebalance_universe_symbols(
        self,
        dual_momentum: Mapping[str, Any],
    ) -> list[str]:
        expanded_config = self._ml_config().get("expanded_rebalance_dataset", {})
        universe_paths = expanded_config.get(
            "universe_paths",
            [
                "data/reference/universes/current_32.yaml",
                "data/reference/universes/us_liquid_100.yaml",
            ],
        )
        output = []
        for universe_path in universe_paths:
            path = Path(str(universe_path))
            if not path.exists():
                continue
            try:
                import yaml

                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except FileNotFoundError:
                continue
            output.extend(str(symbol).upper() for symbol in payload.get("symbols", []))
        if not output:
            output = [
                str(symbol).upper()
                for symbol in dual_momentum.get("symbols", [])
            ]
        max_symbols = expanded_config.get("max_symbols")
        if max_symbols:
            return output[: int(max_symbols)]
        return output

    def validate_history_coverage(
        self,
        candles_by_symbol: dict[str, list[Any]],
        ml_config: Mapping[str, Any],
        source_metadata: dict[str, dict],
    ) -> None:
        required_years = ml_config.get("minimum_history_years")
        if required_years is None:
            return
        report = assess_history_coverage(
            candles_by_symbol,
            required_years=int(required_years),
            tolerance_days=int(ml_config.get("history_coverage_tolerance_days", 10)),
            source_metadata=source_metadata,
        )
        report_path = Path(self._experiment_config.output_dir) / "history_coverage.json"
        allow_short_history = bool(
            ml_config.get("allow_short_history_for_smoke_test", False)
        )
        report["research_label"] = self._research_label
        report["short_history_allowed_for_smoke_test"] = allow_short_history
        write_history_coverage_report(report_path, report)
        if not report["coverage_sufficient"] and not allow_short_history:
            raise RuntimeError(
                "ML research stopped: historical coverage is insufficient. "
                f"Required {report['required_years']} years, but the common range is "
                f"{report['common_start_date']} to {report['common_end_date']}. "
                f"See {report_path}."
            )

    def _build_or_load_historical_features(
        self,
        symbols: list[str],
        benchmark_symbols: tuple[Any, ...],
        builder: HistoricalFeatureBuilder,
        candles_by_symbol: dict[str, list[Any]],
    ) -> MLFeatureBuildResult:
        feature_cache_path = self._path_builder.features_path()
        feature_cache_key = self._feature_cache.feature_cache_key(
            symbols,
            benchmark_symbols,
            builder.lookback_days,
            candles_by_symbol,
        )
        cached_feature_result = self._feature_cache.load_feature_rows(
            feature_cache_path,
            feature_cache_key,
        )
        if cached_feature_result is not None:
            return cached_feature_result
        feature_result = builder.build(candles_by_symbol)
        self._feature_cache.write_feature_rows(
            feature_cache_path,
            feature_result,
            feature_cache_key,
        )
        return feature_result

    def _load_candles(self, symbols: list[str]) -> dict[str, Any]:
        from application.services.market_data_loader import load_candles_with_metadata

        feature_workers = int(self._ml_config().get("feature_workers", 1))
        if feature_workers == 1 or len(symbols) <= 1:
            return {
                symbol: load_candles_with_metadata(symbol, self._config, self._feed)
                for symbol in symbols
            }
        max_workers = min(feature_workers, len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                lambda symbol: load_candles_with_metadata(
                    symbol, self._config, self._feed
                ),
                symbols,
            )
            return dict(zip(symbols, results))

    def _load_multi_timeframe_candles(
        self,
        symbols: list[str],
        ml_config: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        from application.services.market_data_loader import load_candles_with_metadata

        market_config = ml_config.get("market_data", {})
        base_timeframe = str(
            market_config.get("alignment", {}).get("base_timeframe", "1Day")
        )
        timeframes = [
            str(timeframe)
            for timeframe in market_config.get("enabled_timeframes", [])
            if str(timeframe) != base_timeframe
        ]
        loaded = {}
        for timeframe in timeframes:
            timeframe_config = dict(self._config)
            timeframe_config["backtest"] = {
                **self._config.get("backtest", {}),
                "provider": "market_parquet",
                "data_dir": market_config.get("processed_root", "data/processed"),
                "timeframe": timeframe,
            }
            feature_workers = int(ml_config.get("feature_workers", 1))
            if feature_workers == 1 or len(symbols) <= 1:
                loaded[timeframe] = {
                    symbol: load_candles_with_metadata(
                        symbol, timeframe_config, self._feed
                    )
                    for symbol in symbols
                }
            else:
                max_workers = min(feature_workers, len(symbols))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = executor.map(
                        lambda symbol: load_candles_with_metadata(
                            symbol, timeframe_config, self._feed
                        ),
                        symbols,
                    )
                    loaded[timeframe] = dict(zip(symbols, results))
        return loaded

    def _add_multi_timeframe_features(
        self,
        feature_result: MLFeatureBuildResult,
        candles_by_timeframe: dict[str, dict[str, list[Any]]],
        ml_config: Mapping[str, Any],
    ) -> MLFeatureBuildResult:
        from core.research.ml.multi_timeframe import add_intraday_summary_features

        market_config = ml_config.get("market_data", {})
        rows = add_intraday_summary_features(
            feature_result.rows,
            candles_by_timeframe,
            benchmark_symbols=[
                str(symbol)
                for symbol in ml_config.get("benchmark_symbols", ["SPY", "QQQ"])
            ],
            session_close_utc=str(
                market_config.get("alignment", {}).get("session_close_utc", "21:00")
            ),
        )
        return MLFeatureBuildResult(
            rows=rows,
            dropped_rows=feature_result.dropped_rows,
            date_range=feature_result.date_range,
        )

    def _multi_timeframe_enabled(self, ml_config: Mapping[str, Any]) -> bool:
        market_config = ml_config.get("market_data", {})
        enabled = market_config.get("enabled_timeframes", ["1Day"])
        return (
            self._config.get("backtest", {}).get("provider") == "market_parquet"
            and len(enabled) > 1
        )

    def _run_champion_research(self, candles_by_symbol: dict[str, list[Any]]) -> Any:
        from application.services.dual_momentum_config import (
            active_dual_momentum_config,
        )
        from core.research.dual_momentum_factory import build_dual_momentum_tester

        champion_config = active_dual_momentum_config(self._config)
        return build_dual_momentum_tester(
            self._config,
            champion_config,
        ).run(candles_by_symbol)

    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}
