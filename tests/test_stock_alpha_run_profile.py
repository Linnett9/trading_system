from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile


def _settings(**overrides):
    ml = {"stock_alpha_run_size": "dev", "stock_alpha_dev_max_dates": 2, "stock_alpha_dev_max_symbols": 2, **overrides}
    return StockLevelResearchConfig.from_mapping({"ml": ml})


def test_dev_subset_is_deterministic_bounded_and_chronological():
    rows = [
        {"rebalance_date": date, "symbol": symbol}
        for date in ("2024-01-01", "2024-02-01", "2024-03-01")
        for symbol in ("CCC", "AAA", "BBB")
    ]
    first, audit = apply_stock_alpha_run_profile(list(reversed(rows)), _settings())
    second, _ = apply_stock_alpha_run_profile(rows, _settings())

    assert first == second
    assert [row["rebalance_date"] for row in first] == sorted(row["rebalance_date"] for row in first)
    assert {row["rebalance_date"] for row in first} == {"2024-02-01", "2024-03-01"}
    assert {row["symbol"] for row in first} == {"AAA", "BBB"}
    assert audit == {"run_size": "dev", "effective_row_count": 4, "effective_date_count": 2, "effective_symbol_count": 2}


def test_benchmark_profile_does_not_change_rows():
    rows = [{"rebalance_date": "2024-01-01", "symbol": "AAA"}]
    selected, audit = apply_stock_alpha_run_profile(rows, _settings(stock_alpha_run_size="benchmark"))
    assert selected is rows
    assert audit["run_size"] == "benchmark"


def test_dev_subset_preserves_target_columns():
    rows = [{"rebalance_date": "2024-03-01", "symbol": "AAA", "actual_market_residual_return_10d": 0.1, "actual_vol_adjusted_forward_return_10d": 1.2, "actual_rank_normalized_forward_return_10d": 0.5}]
    selected, _ = apply_stock_alpha_run_profile(rows, _settings())
    assert selected[0]["actual_market_residual_return_10d"] == 0.1
    assert selected[0]["actual_vol_adjusted_forward_return_10d"] == 1.2
    assert selected[0]["actual_rank_normalized_forward_return_10d"] == 0.5
