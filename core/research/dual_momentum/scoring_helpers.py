


def _bull_capture(result):
    if result.benchmark_return > 0:
        return result.result.total_return / result.benchmark_return

    return 0
def _annual_return(result, year):
    return (getattr(result, "annual_returns", {}) or {}).get(year, 0)
def _annual_return_values(result):
    return list((getattr(result, "annual_returns", {}) or {}).values())
def _annual_consistency_penalty(result):
    annual_values = _annual_return_values(result)

    if not annual_values:
        return 0

    annual_mean = sum(annual_values) / len(annual_values)

    annual_variance = (
        sum(
            (value - annual_mean) ** 2
            for value in annual_values
        )
        / len(annual_values)
    )

    return annual_variance ** 0.5
def _negative_year_count(result):
    return sum(
        1
        for value in _annual_return_values(result)
        if value < 0
    )
