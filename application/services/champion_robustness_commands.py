from copy import deepcopy
import json
from pathlib import Path

from application.services.dual_momentum_config import active_dual_momentum_config
from application.services.market_data_loader import load_candles
from core.research.champion_robustness import (
    build_champion_robustness_report,
    period_exclusion_summary,
)
from core.research.dual_momentum.diagnostics import (
    aggregate_contributors,
    dual_momentum_diagnosis,
)
from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.research.ml.sector_reference import load_sector_by_symbol


def run_champion_robustness(config: dict, feed) -> None:
    dual_config = active_dual_momentum_config(config)
    benchmark_symbols = config.get("ml", {}).get("benchmark_symbols", ["SPY", "QQQ"])
    investable_symbols = dual_config.get("symbols", config["backtest"]["symbols"])
    symbols = list(dict.fromkeys([*investable_symbols, *benchmark_symbols]))
    candles = {symbol: load_candles(symbol, config, feed) for symbol in symbols}
    sector_map = load_sector_by_symbol(
        config.get("ml", {}).get("sector_reference_path"),
        config.get("ml", {}).get("sector_by_symbol", {}),
    )
    baseline = build_dual_momentum_tester(config, dual_config).run(candles)
    costs = {str(cost): _run(config, dual_config, candles, transaction_cost_bps=cost) for cost in (0, 5, 10, 25, 50)}
    diagnosis = dual_momentum_diagnosis(baseline, candles)
    contributors = [item["symbol"] for item in aggregate_contributors(diagnosis["monthly"], limit=3)]
    stresses = {
        "remove_top_1_contributor": _run(config, dual_config, _without(candles, contributors[:1])),
        "remove_top_3_contributors": _run(config, dual_config, _without(candles, contributors)),
        "exclude_technology_sector": _run(config, dual_config, _without(candles, [symbol for symbol in investable_symbols if sector_map.get(symbol) == "Information Technology"])),
    }
    report = build_champion_robustness_report(baseline, candles, sector_map, costs, stresses)
    report["top_contributors_removed"] = contributors
    report["period_exclusion_stress_tests"] = {
        "exclude_2020_2021": period_exclusion_summary(baseline, candles, "SPY", "2020-01-01", "2021-12-31"),
        "exclude_2022": period_exclusion_summary(baseline, candles, "SPY", "2022-01-01", "2022-12-31"),
    }
    report["paper_candidate_checks"]["period_exclusion_audit_passed"] = all(
        "period_overlapped_backtest" in item
        and "excluded_point_count" in item
        and "warning" in item
        for item in report["period_exclusion_stress_tests"].values()
    )
    if report["paper_candidate_checks"]["period_exclusion_audit_passed"] and report["robustness_pass"]:
        report["production_readiness"] = "paper_candidate"
    path = Path("reports/robustness/champion_stooq_30_robustness.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Champion robustness report: {path}")


def _run(config, dual_config, candles, transaction_cost_bps=None):
    scenario = deepcopy(dual_config)
    if transaction_cost_bps is not None:
        scenario.update({"transaction_cost_bps": transaction_cost_bps, "commission_bps": 0, "slippage_bps": 0, "spread_cost_bps": 0})
    return build_dual_momentum_tester(config, scenario).run(candles)


def _without(candles, removed):
    return {symbol: data for symbol, data in candles.items() if symbol not in set(removed)}
