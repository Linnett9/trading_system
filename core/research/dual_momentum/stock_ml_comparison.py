from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import median

from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.research.dual_momentum.score_providers import (
    HybridScoreProvider,
    MomentumScoreProvider,
    OOSArtifactScoreProvider,
    RankWeightedEnsembleScoreProvider,
)


ELASTIC_NET_COLUMN = "stock_level_predicted_forward_return_10d_elastic_net"
RANDOM_FOREST_COLUMN = "stock_level_predicted_forward_return_10d_random_forest"
DEFAULT_TARGET_COLUMN = "actual_forward_return_10d"
PREDICTION_PREFIX = "stock_level_predicted_forward_return_10d_"
DEFAULT_MODEL_KEYS = ("elastic_net", "random_forest")


@dataclass(frozen=True)
class StockMLComparisonPaths:
    json_path: Path
    csv_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class UniverseDiagnostics:
    mode: str
    requested_symbol_count: int
    artifact_symbol_count: int
    market_data_symbol_count: int
    shared_eligible_symbol_count: int
    missing_market_data_symbols: tuple[str, ...]
    missing_score_symbols: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "requested_symbol_count": self.requested_symbol_count,
            "artifact_symbol_count": self.artifact_symbol_count,
            "market_data_symbol_count": self.market_data_symbol_count,
            "shared_eligible_symbol_count": self.shared_eligible_symbol_count,
            "missing_market_data_symbols": list(self.missing_market_data_symbols),
            "missing_score_symbols": list(self.missing_score_symbols),
        }


def write_stock_ml_dual_momentum_comparison(
    config: dict,
    dual_config: dict,
    candles_by_symbol: dict[str, list],
    strategy_names: list[str] | None = None,
) -> StockMLComparisonPaths:
    comparison_config = _comparison_config(config)
    artifact_path = Path(comparison_config["artifact_path"])
    output_dir = Path(comparison_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_providers = _model_providers(artifact_path, comparison_config)
    providers = _comparison_providers(model_providers, comparison_config)
    target_returns = _target_returns(
        artifact_path,
        date_column=comparison_config["date_column"],
        symbol_column=comparison_config["symbol_column"],
        target_column=comparison_config["target_column"],
    )
    selected_names = strategy_names or list(providers)
    unknown = sorted(set(selected_names) - set(providers))
    if unknown:
        raise ValueError(f"Unknown stock ML comparison strategies: {unknown}")

    universe = _resolve_universe(
        candles_by_symbol,
        list(model_providers.values()),
        mode=comparison_config["mode"],
        requested_symbols=comparison_config.get("requested_symbols", []),
    )
    shared_symbols = list(universe.shared_symbols)
    shared_dates = sorted(
        set.intersection(
            *[
                provider.dates
                for provider in model_providers.values()
            ]
        )
    )
    filtered_candles = {
        symbol: candles
        for symbol, candles in candles_by_symbol.items()
        if symbol in shared_symbols
    }
    if not filtered_candles:
        raise ValueError("No shared symbols between candles and OOS artifact")
    if not shared_dates:
        raise ValueError("No shared OOS dates between configured providers")

    rows = []
    payload_results = {}
    start_at = datetime.combine(shared_dates[0], datetime.min.time())
    end_at = datetime.combine(shared_dates[-1], datetime.min.time())

    for top_n in _top_n_values(comparison_config, dual_config):
        for name in selected_names:
            result_key = f"{name}|top_n_{top_n}"
            experiment_config = _experiment_dual_config(
                dual_config,
                comparison_config,
                top_n,
                result_key,
            )
            tester = build_dual_momentum_tester(
                config,
                experiment_config,
                score_provider=providers[name],
                rebalance_dates=set(shared_dates),
            )
            result = tester.run(
                filtered_candles,
                start_at=start_at,
                end_at=end_at,
            )
            diagnostics = _ranking_diagnostics(result.selections, target_returns)
            row = _result_row(
                name,
                result,
                diagnostics,
                shared_symbols,
                shared_dates,
                top_n=top_n,
            )
            rows.append(row)
            payload_results[result_key] = {
                **row,
                "config": result.config,
                "selections": [
                    selection.to_dict()
                    for selection in result.selections
                ],
            }

    payload = {
        "mode": "stock_ml_dual_momentum_score_comparison",
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "artifact_contract": {
            "path": str(artifact_path),
            "date_column": comparison_config["date_column"],
            "symbol_column": comparison_config["symbol_column"],
            "fold_column": comparison_config["fold_column"],
            "target_column": comparison_config["target_column"],
            "elastic_net_signal_column": (
                comparison_config["elastic_net_signal_column"]
            ),
            "random_forest_signal_column": (
                comparison_config["random_forest_signal_column"]
            ),
            "model_score_columns": _model_score_columns(comparison_config),
        },
        "fairness_controls": {
            "mode": comparison_config["mode"],
            "universe_diagnostics": universe.diagnostics.to_dict(),
            "shared_symbol_count": len(shared_symbols),
            "shared_symbols": shared_symbols,
            "shared_oos_date_count": len(shared_dates),
            "shared_oos_start": shared_dates[0].isoformat(),
            "shared_oos_end": shared_dates[-1].isoformat(),
            "same_downstream_engine": True,
        },
        "leakage_controls": {
            "score_lookup": "exact timestamp-symbol only",
            "forward_fill": False,
            "future_lookup": False,
            "target_columns_allowed_as_scores": False,
            "retrained_models": False,
        },
        "provider_diagnostics": {
            name: provider.diagnostics()
            for name, provider in model_providers.items()
        },
        "top_n_values": _top_n_values(comparison_config, dual_config),
        "top_n_sweep_policy": {
            "max_selected_assets": comparison_config[
                "top_n_sweep_max_selected_assets_policy"
            ],
        },
        "results": payload_results,
        "winners": _winners(rows),
    }

    json_path = output_dir / "stock_ml_dual_momentum_score_comparison.json"
    csv_path = output_dir / "stock_ml_dual_momentum_score_comparison.csv"
    markdown_path = output_dir / "stock_ml_dual_momentum_score_comparison.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, payload, rows)
    return StockMLComparisonPaths(
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )


def stock_ml_comparison_config(config: dict) -> dict:
    return _comparison_config(config)


def stock_ml_comparison_artifact_dates(config: dict) -> set[date]:
    comparison_config = _comparison_config(config)
    path = Path(comparison_config["artifact_path"])
    dates = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        date_column = comparison_config["date_column"]
        if date_column not in columns:
            raise ValueError(
                f"Missing OOS artifact date column: {date_column}"
            )
        for row in reader:
            dates.add(datetime.fromisoformat(row[date_column]).date())
    return dates


def stock_ml_comparison_artifact_symbols(config: dict) -> set[str]:
    comparison_config = _comparison_config(config)
    path = Path(comparison_config["artifact_path"])
    symbols = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        symbol_column = comparison_config["symbol_column"]
        if symbol_column not in columns:
            raise ValueError(
                f"Missing OOS artifact symbol column: {symbol_column}"
            )
        for row in reader:
            symbols.add(str(row[symbol_column]).strip().upper())
    return symbols


def _comparison_config(config: dict) -> dict:
    ml = config.get("ml", {})
    defaults = {
        "artifact_path": (
            "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
            "stock_alpha/benchmark/enriched/"
            "stock_level_model_oos_predictions.csv"
        ),
        "output_dir": (
            "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
            "stock_alpha/benchmark/dual_momentum_score_comparison"
        ),
        "mode": "matched_universe_comparison",
        "universe_mode": None,
        "date_column": "rebalance_date",
        "symbol_column": "symbol",
        "fold_column": "fold_id",
        "target_column": DEFAULT_TARGET_COLUMN,
        "elastic_net_signal_column": ELASTIC_NET_COLUMN,
        "random_forest_signal_column": RANDOM_FOREST_COLUMN,
        "local_data_provider": "market_parquet",
        "market_parquet_dir": "data/processed",
        "timeframe": "1Day",
        "models": list(DEFAULT_MODEL_KEYS),
        "model_score_columns": {},
        "top_n_values": None,
        "top_n_sweep_max_selected_assets_policy": "match_requested_top_n",
        "ensembles": {
            "hybrid_momentum_elastic_net": {
                "members": {"dual_momentum": 0.5, "elastic_net": 0.5},
            },
            "hybrid_momentum_random_forest": {
                "members": {"dual_momentum": 0.5, "random_forest": 0.5},
            },
        },
    }
    resolved = {
        **defaults,
        **ml.get("stock_ml_dual_momentum_comparison", {}),
    }
    resolved["mode"] = _canonical_universe_mode(
        resolved.get("universe_mode") or resolved.get("mode")
    )
    return resolved


@dataclass(frozen=True)
class ResolvedUniverse:
    shared_symbols: tuple[str, ...]
    diagnostics: UniverseDiagnostics


def _resolve_universe(
    candles_by_symbol: dict[str, list],
    providers: list[OOSArtifactScoreProvider],
    mode: str,
    requested_symbols: list[str] | tuple[str, ...] | set[str],
) -> ResolvedUniverse:
    canonical_mode = _canonical_universe_mode(mode)
    market_symbols = {symbol.upper() for symbol in candles_by_symbol}
    artifact_symbols = set.intersection(*[provider.symbols for provider in providers])
    artifact_symbols = {symbol.upper() for symbol in artifact_symbols}
    requested = {str(symbol).upper() for symbol in requested_symbols or []}
    if not requested:
        requested = market_symbols if canonical_mode == "matched_operational_universe" else artifact_symbols
    shared = sorted(requested & artifact_symbols & market_symbols)
    diagnostics = UniverseDiagnostics(
        mode=canonical_mode,
        requested_symbol_count=len(requested),
        artifact_symbol_count=len(artifact_symbols),
        market_data_symbol_count=len(market_symbols),
        shared_eligible_symbol_count=len(shared),
        missing_market_data_symbols=tuple(sorted((requested & artifact_symbols) - market_symbols)),
        missing_score_symbols=tuple(sorted(requested - artifact_symbols)),
    )
    return ResolvedUniverse(tuple(shared), diagnostics)


def _canonical_universe_mode(mode: str | None) -> str:
    aliases = {
        None: "matched_operational_universe",
        "matched_universe_comparison": "matched_operational_universe",
        "matched_operational_universe": "matched_operational_universe",
        "expanded_ml_universe_research": "broad_research_universe",
        "broad_research_universe": "broad_research_universe",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported comparison universe mode: {mode}")
    return aliases[mode]


def _model_providers(
    artifact_path: Path,
    comparison_config: dict,
) -> dict[str, OOSArtifactScoreProvider]:
    providers = {}
    for model_key in comparison_config.get("models", DEFAULT_MODEL_KEYS):
        column = _score_column_for_model(str(model_key), comparison_config)
        providers[str(model_key)] = OOSArtifactScoreProvider(
            artifact_path,
            column,
            date_column=comparison_config["date_column"],
            symbol_column=comparison_config["symbol_column"],
            fold_column=comparison_config["fold_column"],
            rank_normalize=True,
            name=f"{model_key}_oos",
        )
    return providers


def _comparison_providers(
    model_providers: dict[str, OOSArtifactScoreProvider],
    comparison_config: dict,
) -> dict[str, object]:
    providers: dict[str, object] = {"dual_momentum": None}
    for model_key, provider in model_providers.items():
        providers[f"{model_key}_oos"] = provider

    momentum_provider = MomentumScoreProvider()
    for ensemble_name, spec in comparison_config.get("ensembles", {}).items():
        members = {}
        for member_name, weight in dict(spec.get("members", {})).items():
            if member_name == "dual_momentum":
                members[member_name] = (momentum_provider, float(weight))
                continue
            provider = model_providers.get(member_name)
            if provider is None:
                raise ValueError(
                    f"Ensemble {ensemble_name} references unknown model "
                    f"'{member_name}'"
                )
            members[member_name] = (provider, float(weight))
        providers[ensemble_name] = RankWeightedEnsembleScoreProvider(
            ensemble_name,
            members,
        )
    return providers


def _model_score_columns(comparison_config: dict) -> dict[str, str]:
    return {
        str(model_key): _score_column_for_model(str(model_key), comparison_config)
        for model_key in comparison_config.get("models", DEFAULT_MODEL_KEYS)
    }


def _score_column_for_model(model_key: str, comparison_config: dict) -> str:
    explicit = dict(comparison_config.get("model_score_columns", {}))
    legacy = {
        "elastic_net": comparison_config["elastic_net_signal_column"],
        "random_forest": comparison_config["random_forest_signal_column"],
    }
    return explicit.get(model_key) or legacy.get(model_key) or f"{PREDICTION_PREFIX}{model_key}"


def _top_n_values(comparison_config: dict, dual_config: dict) -> list[int]:
    configured = comparison_config.get("top_n_values")
    if configured is None:
        return [int(dual_config.get("top_n", 5))]
    values = [int(value) for value in configured]
    if not values:
        raise ValueError("top_n_values must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("top_n_values must be positive integers")
    return values


def _experiment_dual_config(
    dual_config: dict,
    comparison_config: dict,
    top_n: int,
    result_key: str,
) -> dict:
    experiment_config = {
        **dual_config,
        "top_n": top_n,
        "experiment_name": result_key,
    }
    has_explicit_sweep = comparison_config.get("top_n_values") is not None
    policy = comparison_config.get(
        "top_n_sweep_max_selected_assets_policy",
        "match_requested_top_n",
    )
    if not has_explicit_sweep:
        return experiment_config
    if policy == "match_requested_top_n":
        experiment_config["max_selected_assets"] = top_n
    elif policy == "preserve_dual_config":
        pass
    else:
        raise ValueError(
            "Unsupported top_n_sweep_max_selected_assets_policy: "
            f"{policy}"
        )
    return experiment_config


def _target_returns(
    path: Path,
    date_column: str,
    symbol_column: str,
    target_column: str,
) -> dict[tuple[date, str], float]:
    target_by_key = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {date_column, symbol_column, target_column}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Missing target diagnostic columns: {missing}")
        for row in reader:
            value = float(row[target_column])
            if math.isfinite(value):
                target_by_key[(
                    datetime.fromisoformat(row[date_column]).date(),
                    row[symbol_column],
                )] = value
    return target_by_key


def _ranking_diagnostics(selections, target_returns: dict[tuple[date, str], float]) -> dict:
    spearman_values = []
    spreads = []
    missing_scores = 0
    scored_pairs = 0

    for selection in selections:
        row_date = selection.timestamp.date()
        score_target_pairs = []
        for symbol, score in selection.scores.items():
            target = target_returns.get((row_date, symbol))
            if target is None:
                missing_scores += 1
                continue
            score_target_pairs.append((score, target))
            scored_pairs += 1

        if len(score_target_pairs) >= 2:
            spearman_values.append(_spearman(score_target_pairs))
            sorted_pairs = sorted(score_target_pairs, key=lambda item: item[0])
            bucket = max(1, len(sorted_pairs) // 10)
            bottom = sorted_pairs[:bucket]
            top = sorted_pairs[-bucket:]
            spreads.append(
                sum(item[1] for item in top) / len(top)
                - sum(item[1] for item in bottom) / len(bottom)
            )

    return {
        "mean_spearman_ic": (
            sum(spearman_values) / len(spearman_values)
            if spearman_values
            else None
        ),
        "top_minus_bottom_spread": (
            sum(spreads) / len(spreads)
            if spreads
            else None
        ),
        "scored_target_pair_count": scored_pairs,
        "missing_target_pair_count": missing_scores,
    }


def _spearman(score_target_pairs: list[tuple[float, float]]) -> float:
    score_ranks = _rank_values([item[0] for item in score_target_pairs])
    target_ranks = _rank_values([item[1] for item in score_target_pairs])
    mean_score = sum(score_ranks) / len(score_ranks)
    mean_target = sum(target_ranks) / len(target_ranks)
    numerator = sum(
        (score - mean_score) * (target - mean_target)
        for score, target in zip(score_ranks, target_ranks)
    )
    score_var = sum((score - mean_score) ** 2 for score in score_ranks)
    target_var = sum((target - mean_target) ** 2 for target in target_ranks)
    denominator = math.sqrt(score_var * target_var)
    return numerator / denominator if denominator else 0


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        rank = (index + end) / 2
        for item_index in range(index, end + 1):
            ranks[indexed[item_index][0]] = rank
        index = end + 1
    return ranks


def _result_row(
    strategy_name: str,
    result,
    diagnostics: dict,
    shared_symbols: list[str],
    shared_dates: list[date],
    top_n: int,
) -> dict:
    holding_diagnostics = _holding_diagnostics(result.selections)
    return {
        "strategy": strategy_name,
        "top_n": top_n,
        "requested_top_n": top_n,
        "effective_top_n": _effective_top_n(result.config),
        **holding_diagnostics,
        "total_return": result.result.total_return,
        "annualized_return": result.cagr,
        "benchmark_return": result.benchmark_return,
        "excess_return": result.excess_return,
        "sharpe": result.result.sharpe,
        "max_drawdown": result.result.max_drawdown,
        "calmar": result.calmar,
        "turnover": result.turnover_percent,
        "annualized_turnover": result.annualized_turnover_percent,
        "cost_drag": result.cost_drag_percent,
        "estimated_cost": result.estimated_cost,
        "trade_count": (
            result.result.closed_trades + result.result.open_trades
        ),
        "rebalance_count": result.rebalance_count,
        "date_coverage": len(shared_dates),
        "symbol_coverage": len(shared_symbols),
        "missing_score_coverage": 0,
        "missing_target_pair_count": diagnostics["missing_target_pair_count"],
        "mean_spearman_ic": diagnostics["mean_spearman_ic"],
        "top_minus_bottom_spread": diagnostics["top_minus_bottom_spread"],
    }


def _effective_top_n(result_config: dict) -> int | None:
    top_n = result_config.get("top_n")
    max_selected_assets = result_config.get("max_selected_assets")
    if (
        result_config.get("selection_mode") == "all_positive"
        and max_selected_assets is None
    ):
        return None
    if top_n is None:
        return max_selected_assets
    if max_selected_assets is None:
        return int(top_n)
    return min(int(top_n), int(max_selected_assets))


def _holding_diagnostics(selections) -> dict:
    holding_counts = [len(selection.symbols) for selection in selections]
    candidate_counts = [
        selection.candidate_count
        for selection in selections
        if selection.candidate_count is not None
    ]
    selected_counts = [
        selection.selected_count_before_hysteresis
        for selection in selections
        if selection.selected_count_before_hysteresis is not None
    ]
    final_counts = [
        selection.final_holding_count
        for selection in selections
        if selection.final_holding_count is not None
    ]
    return {
        "average_holding_count": _mean(holding_counts),
        "median_holding_count": (
            median(holding_counts)
            if holding_counts
            else 0
        ),
        "min_holding_count": min(holding_counts) if holding_counts else 0,
        "max_holding_count": max(holding_counts) if holding_counts else 0,
        "average_candidate_count": _mean(candidate_counts),
        "average_selected_count_before_hysteresis": _mean(selected_counts),
        "average_final_holding_count_after_hysteresis": _mean(final_counts),
    }


def _mean(values: list[int | float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _winners(rows: list[dict]) -> dict:
    return {
        "portfolio_return_winner": _max_row(rows, "total_return"),
        "risk_adjusted_winner": _max_row(rows, "sharpe"),
        "drawdown_winner": _max_row(rows, "max_drawdown"),
        "ranking_quality_winner": _max_row(rows, "mean_spearman_ic"),
    }


def _max_row(rows: list[dict], field: str) -> dict | None:
    valid = [row for row in rows if row.get(field) is not None]
    if not valid:
        return None
    row = max(valid, key=lambda item: item[field])
    return {"strategy": row["strategy"], field: row[field]}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict, rows: list[dict]) -> None:
    lines = [
        "# Stock ML Dual-Momentum Score Comparison",
        "",
        "mode=research_only | trading_impact=none | production_validated=false",
        "",
        f"Artifact: `{payload['artifact_contract']['path']}`",
        f"Shared symbols: {payload['fairness_controls']['shared_symbol_count']}",
        f"Shared OOS dates: {payload['fairness_controls']['shared_oos_date_count']}",
        f"Universe mode: `{payload['fairness_controls']['universe_diagnostics']['mode']}`",
        "",
        "| Strategy | Requested Top N | Effective Top N | Avg Holdings | "
        "Total Return | Sharpe | Max Drawdown | Turnover | Mean Spearman IC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {strategy} | {requested_top_n} | {effective_top_n} | "
            "{average_holding_count:.2f} | {total_return:.6f} | "
            "{sharpe:.6f} | {max_drawdown:.6f} | {turnover:.6f} | "
            "{mean_spearman_ic} |".format(
                **{
                    **row,
                    "effective_top_n": (
                        row["effective_top_n"]
                        if row["effective_top_n"] is not None
                        else ""
                    ),
                    "mean_spearman_ic": (
                        f"{row['mean_spearman_ic']:.6f}"
                        if row["mean_spearman_ic"] is not None
                        else ""
                    ),
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
