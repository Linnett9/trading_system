from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from core.research.ml.ds24.clean_v2_contracts import stable_hash


INCREMENTAL_SUMMARY_STATE_VERSION = "DS24_INCREMENTAL_V3_SUMMARY_STATE_V1"
DEFAULT_HAC_LAG = 12
TRADING_SESSIONS_PER_YEAR = 252


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class ExactSeriesState:
    """Incremental exact-order statistics for one finite numeric series.

    The two heaps retain an exact median.  Raw lagged products retain the
    sufficient statistics required by the existing Newey-West mean interval
    without revisiting historical values.
    """

    maximum_lag: int = DEFAULT_HAC_LAG
    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    positive_count: int = 0
    lower_heap: list[float] = field(default_factory=list)
    upper_heap: list[float] = field(default_factory=list)
    first_values: list[float] = field(default_factory=list)
    last_values: deque[float] = field(default_factory=deque)
    lagged_products: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.maximum_lag < 0:
            raise ValueError("maximum_lag must be non-negative")
        if not self.lagged_products:
            self.lagged_products = [0.0] * (self.maximum_lag + 1)
        if len(self.lagged_products) != self.maximum_lag + 1:
            raise ValueError("lagged_products length does not match maximum_lag")
        if not isinstance(self.last_values, deque):
            self.last_values = deque(self.last_values, maxlen=self.maximum_lag)
        else:
            self.last_values = deque(self.last_values, maxlen=self.maximum_lag)

    def add(self, value: Any) -> bool:
        numeric = _finite(value)
        if numeric is None:
            return False
        previous = list(self.last_values)
        for lag in range(1, min(self.maximum_lag, len(previous)) + 1):
            self.lagged_products[lag] += numeric * previous[-lag]
        self.count += 1
        self.total += numeric
        self.total_squares += numeric * numeric
        self.positive_count += int(numeric > 0.0)
        if len(self.first_values) < self.maximum_lag:
            self.first_values.append(numeric)
        self.last_values.append(numeric)
        if not self.lower_heap or numeric <= -self.lower_heap[0]:
            heapq.heappush(self.lower_heap, -numeric)
        else:
            heapq.heappush(self.upper_heap, numeric)
        if len(self.lower_heap) > len(self.upper_heap) + 1:
            heapq.heappush(self.upper_heap, -heapq.heappop(self.lower_heap))
        elif len(self.upper_heap) > len(self.lower_heap):
            heapq.heappush(self.lower_heap, -heapq.heappop(self.upper_heap))
        return True

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    @property
    def median(self) -> float | None:
        if not self.count:
            return None
        if len(self.lower_heap) == len(self.upper_heap):
            return (-self.lower_heap[0] + self.upper_heap[0]) / 2.0
        return -self.lower_heap[0]

    def sample_standard_deviation(self) -> float | None:
        if self.count <= 1:
            return None
        variance = (
            self.total_squares - (self.total * self.total / self.count)
        ) / (self.count - 1)
        return math.sqrt(max(variance, 0.0))

    def newey_west_mean_ci(
        self,
        *,
        lag: int = DEFAULT_HAC_LAG,
        z: float = 1.96,
    ) -> dict[str, Any]:
        if self.count == 0:
            return {
                "mean": None,
                "lower": None,
                "upper": None,
                "standard_error": None,
                "observations": 0,
                "lag": 0,
            }
        mean = float(self.total / self.count)
        if self.count == 1:
            return {
                "mean": mean,
                "lower": mean,
                "upper": mean,
                "standard_error": 0.0,
                "observations": 1,
                "lag": 0,
            }
        effective_lag = min(max(int(lag), 0), self.count - 1, self.maximum_lag)
        gamma0 = (
            self.total_squares
            - 2.0 * mean * self.total
            + self.count * mean * mean
        ) / self.count
        variance = gamma0
        trailing = list(self.last_values)
        for step in range(1, effective_lag + 1):
            leading_sum = self.total - sum(self.first_values[:step])
            lagged_sum = self.total - sum(trailing[-step:])
            cross = (
                self.lagged_products[step]
                - mean * (leading_sum + lagged_sum)
                + (self.count - step) * mean * mean
            ) / self.count
            variance += 2.0 * (1.0 - step / (effective_lag + 1.0)) * cross
        standard_error = math.sqrt(max(variance / self.count, 0.0))
        return {
            "mean": mean,
            "lower": mean - z * standard_error,
            "upper": mean + z * standard_error,
            "standard_error": standard_error,
            "observations": self.count,
            "lag": effective_lag,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_lag": self.maximum_lag,
            "count": self.count,
            "total": self.total,
            "total_squares": self.total_squares,
            "positive_count": self.positive_count,
            "lower_heap": list(self.lower_heap),
            "upper_heap": list(self.upper_heap),
            "first_values": list(self.first_values),
            "last_values": list(self.last_values),
            "lagged_products": list(self.lagged_products),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExactSeriesState:
        return cls(
            maximum_lag=int(payload.get("maximum_lag", DEFAULT_HAC_LAG)),
            count=int(payload.get("count", 0)),
            total=float(payload.get("total", 0.0)),
            total_squares=float(payload.get("total_squares", 0.0)),
            positive_count=int(payload.get("positive_count", 0)),
            lower_heap=[float(value) for value in payload.get("lower_heap", [])],
            upper_heap=[float(value) for value in payload.get("upper_heap", [])],
            first_values=[float(value) for value in payload.get("first_values", [])],
            last_values=deque(float(value) for value in payload.get("last_values", [])),
            lagged_products=[
                float(value) for value in payload.get("lagged_products", [])
            ],
        )


@dataclass
class IncrementalV3SummaryState:
    """Sufficient statistics for the exact V3 provisional summary contract."""

    rank_ic: ExactSeriesState = field(default_factory=ExactSeriesState)
    daily_net: ExactSeriesState = field(default_factory=ExactSeriesState)
    daily_gross_product: float = 1.0
    daily_net_product: float = 1.0
    daily_equity: float = 1.0
    daily_peak_equity: float = 1.0
    maximum_drawdown: float | None = None
    daily_win_count: int = 0
    last_daily_net_return: float | None = None
    turnover_count: int = 0
    turnover_total: float = 0.0
    total_estimated_costs: float = 0.0
    eligible_asset_total: float = 0.0
    resolved_asset_total: float = 0.0
    rank_rows: int = 0
    daily_rows: int = 0
    terminal_censored_rows: int = 0
    first_resolved_decision_timestamp: str = ""
    last_resolved_decision_timestamp: str = ""
    daily_rank_ic: dict[str, tuple[float, int]] = field(default_factory=dict)
    daily_rank_ic_mean_total: float = 0.0

    def add_rank_rows(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        for row in rows.itertuples(index=False):
            timestamp = pd.Timestamp(getattr(row, "decision_timestamp")).tz_convert(
                "UTC"
            ).isoformat()
            if not self.first_resolved_decision_timestamp:
                self.first_resolved_decision_timestamp = timestamp
            self.last_resolved_decision_timestamp = timestamp
            self.rank_rows += 1
            self.eligible_asset_total += float(
                _finite(getattr(row, "eligible_asset_count", 0.0)) or 0.0
            )
            self.resolved_asset_total += float(
                _finite(getattr(row, "resolved_asset_count", 0.0)) or 0.0
            )
            value = _finite(getattr(row, "spearman_rank_ic", None))
            if value is None:
                continue
            self.rank_ic.add(value)
            session_date = str(getattr(row, "session_date", ""))
            prior_sum, prior_count = self.daily_rank_ic.get(
                session_date, (0.0, 0)
            )
            if prior_count:
                self.daily_rank_ic_mean_total -= prior_sum / prior_count
            next_sum = prior_sum + value
            next_count = prior_count + 1
            self.daily_rank_ic[session_date] = (next_sum, next_count)
            self.daily_rank_ic_mean_total += next_sum / next_count

    def add_sleeve_rows(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        for row in rows.itertuples(index=False):
            turnover = _finite(getattr(row, "turnover", None))
            if turnover is not None:
                self.turnover_count += 1
                self.turnover_total += turnover
            cost = _finite(getattr(row, "transaction_cost_contribution", None))
            if cost is not None:
                self.total_estimated_costs += cost

    def add_daily_rows(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        for row in rows.itertuples(index=False):
            net = _finite(getattr(row, "net_daily_return", None))
            gross = _finite(getattr(row, "gross_daily_return", None))
            if net is None:
                continue
            self.daily_net.add(net)
            self.daily_rows += 1
            self.last_daily_net_return = net
            self.daily_net_product *= 1.0 + net
            if gross is not None:
                self.daily_gross_product *= 1.0 + gross
            self.daily_win_count += int(net > 0.0)
            self.daily_equity *= 1.0 + net
            self.daily_peak_equity = max(
                self.daily_peak_equity, self.daily_equity
            )
            drawdown = self.daily_equity / self.daily_peak_equity - 1.0
            self.maximum_drawdown = (
                drawdown
                if self.maximum_drawdown is None
                else min(self.maximum_drawdown, drawdown)
            )

    def add_terminal_rows(self, rows: pd.DataFrame) -> None:
        self.terminal_censored_rows += int(len(rows))

    def to_summary(
        self,
        *,
        evaluation_contract_id: str,
        evaluation_contract_hash: str,
        evaluation_contract_version: str,
        sleeve_count: int,
        pending_rows: int,
        created_at_utc: str,
    ) -> dict[str, Any]:
        daily_standard_deviation = self.daily_net.sample_standard_deviation()
        annualized_volatility = (
            daily_standard_deviation * math.sqrt(TRADING_SESSIONS_PER_YEAR)
            if daily_standard_deviation is not None
            else None
        )
        annualized_return = (
            self.daily_net.mean * TRADING_SESSIONS_PER_YEAR
            if self.daily_net.mean is not None
            else None
        )
        daily_ic_rows = len(self.daily_rank_ic)
        coverage = (
            self.resolved_asset_total / self.eligible_asset_total
            if self.eligible_asset_total > 0.0
            else None
        )
        return {
            "status": (
                "PROVISIONAL"
                if self.rank_rows or self.daily_rows
                else "NO_RESOLVED_PERFORMANCE_ROWS"
            ),
            "evaluation_contract_id": evaluation_contract_id,
            "evaluation_contract_hash": evaluation_contract_hash,
            "evaluation_contract_version": evaluation_contract_version,
            "first_resolved_decision_timestamp": (
                self.first_resolved_decision_timestamp
            ),
            "last_resolved_decision_timestamp": (
                self.last_resolved_decision_timestamp
            ),
            "resolved_performance_rows": self.rank_rows,
            "rank_ic": {
                "valid_timestamps": self.rank_ic.count,
                "mean_spearman_rank_ic": self.rank_ic.mean,
                "median_spearman_rank_ic": self.rank_ic.median,
                "positive_fraction": (
                    self.rank_ic.positive_count / self.rank_ic.count
                    if self.rank_ic.count
                    else None
                ),
                "dependence_aware_95_ci": (
                    self.rank_ic.newey_west_mean_ci(lag=DEFAULT_HAC_LAG)
                    if self.rank_ic.count
                    else None
                ),
                "inference_method": (
                    "newey_west_hac_lag_12_for_overlapping_60m_targets"
                ),
                "daily_ic_rows": daily_ic_rows,
                "daily_mean_spearman_rank_ic": (
                    self.daily_rank_ic_mean_total / daily_ic_rows
                    if daily_ic_rows
                    else None
                ),
            },
            "returns": {
                "portfolio_contract": "twelve_staggered_equal_capital_sleeves",
                "sleeve_count": int(sleeve_count),
                "simultaneous_capital_limit": 1.0,
                "daily_return_rows": self.daily_net.count,
                "last_daily_net_return": self.last_daily_net_return,
                "cumulative_gross_return": (
                    self.daily_gross_product - 1.0 if self.daily_rows else None
                ),
                "cumulative_net_return": (
                    self.daily_net_product - 1.0 if self.daily_rows else None
                ),
                "annualized_return_from_daily_returns": annualized_return,
                "annualized_volatility_from_daily_returns": annualized_volatility,
                "daily_sharpe": (
                    annualized_return / annualized_volatility
                    if annualized_return is not None
                    and annualized_volatility
                    and annualized_volatility > 0.0
                    else None
                ),
                "maximum_drawdown": self.maximum_drawdown,
                "win_rate": (
                    self.daily_win_count / self.daily_net.count
                    if self.daily_net.count
                    else None
                ),
                "mean_turnover": (
                    self.turnover_total / self.turnover_count
                    if self.turnover_count
                    else None
                ),
                "total_estimated_costs": self.total_estimated_costs,
                "raw_overlapping_forward_returns_annualized": False,
            },
            "coverage": {
                "eligible_resolved_fraction": coverage,
                "pending_score_rows": int(pending_rows),
                "terminal_censored_rows": self.terminal_censored_rows,
            },
            "created_at_utc": created_at_utc,
        }

    def state_identity(self) -> str:
        # This identity is deliberately constant-size.  Persisting or hashing
        # the exact-median heaps on every timestamp would itself reintroduce an
        # O(T^2) lifecycle.  Durable logs remain the restart authority and are
        # scanned once; this digest validates their rebuilt observable state.
        return stable_hash(
            {
                "schema_version": INCREMENTAL_SUMMARY_STATE_VERSION,
                "rank_ic": {
                    "count": self.rank_ic.count,
                    "total": self.rank_ic.total,
                    "total_squares": self.rank_ic.total_squares,
                    "positive_count": self.rank_ic.positive_count,
                    "median": self.rank_ic.median,
                    "newey_west": self.rank_ic.newey_west_mean_ci(
                        lag=DEFAULT_HAC_LAG
                    ),
                },
                "daily_net": {
                    "count": self.daily_net.count,
                    "total": self.daily_net.total,
                    "total_squares": self.daily_net.total_squares,
                    "positive_count": self.daily_net.positive_count,
                },
                "daily_gross_product": self.daily_gross_product,
                "daily_net_product": self.daily_net_product,
                "daily_equity": self.daily_equity,
                "daily_peak_equity": self.daily_peak_equity,
                "maximum_drawdown": self.maximum_drawdown,
                "daily_win_count": self.daily_win_count,
                "last_daily_net_return": self.last_daily_net_return,
                "turnover_count": self.turnover_count,
                "turnover_total": self.turnover_total,
                "total_estimated_costs": self.total_estimated_costs,
                "eligible_asset_total": self.eligible_asset_total,
                "resolved_asset_total": self.resolved_asset_total,
                "rank_rows": self.rank_rows,
                "daily_rows": self.daily_rows,
                "terminal_censored_rows": self.terminal_censored_rows,
                "first_resolved_decision_timestamp": (
                    self.first_resolved_decision_timestamp
                ),
                "last_resolved_decision_timestamp": (
                    self.last_resolved_decision_timestamp
                ),
                "daily_rank_ic_rows": len(self.daily_rank_ic),
                "daily_rank_ic_mean_total": self.daily_rank_ic_mean_total,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INCREMENTAL_SUMMARY_STATE_VERSION,
            "rank_ic": self.rank_ic.to_dict(),
            "daily_net": self.daily_net.to_dict(),
            "daily_gross_product": self.daily_gross_product,
            "daily_net_product": self.daily_net_product,
            "daily_equity": self.daily_equity,
            "daily_peak_equity": self.daily_peak_equity,
            "maximum_drawdown": self.maximum_drawdown,
            "daily_win_count": self.daily_win_count,
            "last_daily_net_return": self.last_daily_net_return,
            "turnover_count": self.turnover_count,
            "turnover_total": self.turnover_total,
            "total_estimated_costs": self.total_estimated_costs,
            "eligible_asset_total": self.eligible_asset_total,
            "resolved_asset_total": self.resolved_asset_total,
            "rank_rows": self.rank_rows,
            "daily_rows": self.daily_rows,
            "terminal_censored_rows": self.terminal_censored_rows,
            "first_resolved_decision_timestamp": (
                self.first_resolved_decision_timestamp
            ),
            "last_resolved_decision_timestamp": (
                self.last_resolved_decision_timestamp
            ),
            "daily_rank_ic": {
                key: [value[0], value[1]]
                for key, value in sorted(self.daily_rank_ic.items())
            },
            "daily_rank_ic_mean_total": self.daily_rank_ic_mean_total,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> IncrementalV3SummaryState:
        if payload.get("schema_version") != INCREMENTAL_SUMMARY_STATE_VERSION:
            raise ValueError("INCREMENTAL_SUMMARY_STATE_SCHEMA_MISMATCH")
        return cls(
            rank_ic=ExactSeriesState.from_dict(payload.get("rank_ic", {})),
            daily_net=ExactSeriesState.from_dict(payload.get("daily_net", {})),
            daily_gross_product=float(payload.get("daily_gross_product", 1.0)),
            daily_net_product=float(payload.get("daily_net_product", 1.0)),
            daily_equity=float(payload.get("daily_equity", 1.0)),
            daily_peak_equity=float(payload.get("daily_peak_equity", 1.0)),
            maximum_drawdown=(
                None
                if payload.get("maximum_drawdown") is None
                else float(payload["maximum_drawdown"])
            ),
            daily_win_count=int(payload.get("daily_win_count", 0)),
            last_daily_net_return=(
                None
                if payload.get("last_daily_net_return") is None
                else float(payload["last_daily_net_return"])
            ),
            turnover_count=int(payload.get("turnover_count", 0)),
            turnover_total=float(payload.get("turnover_total", 0.0)),
            total_estimated_costs=float(
                payload.get("total_estimated_costs", 0.0)
            ),
            eligible_asset_total=float(payload.get("eligible_asset_total", 0.0)),
            resolved_asset_total=float(payload.get("resolved_asset_total", 0.0)),
            rank_rows=int(payload.get("rank_rows", 0)),
            daily_rows=int(payload.get("daily_rows", 0)),
            terminal_censored_rows=int(
                payload.get("terminal_censored_rows", 0)
            ),
            first_resolved_decision_timestamp=str(
                payload.get("first_resolved_decision_timestamp", "")
            ),
            last_resolved_decision_timestamp=str(
                payload.get("last_resolved_decision_timestamp", "")
            ),
            daily_rank_ic={
                str(key): (float(value[0]), int(value[1]))
                for key, value in dict(payload.get("daily_rank_ic", {})).items()
            },
            daily_rank_ic_mean_total=float(
                payload.get("daily_rank_ic_mean_total", 0.0)
            ),
        )


def build_summary_state(
    rank_ic: pd.DataFrame,
    daily_returns: pd.DataFrame,
    sleeves: pd.DataFrame,
    terminal_censored: pd.DataFrame,
) -> IncrementalV3SummaryState:
    state = IncrementalV3SummaryState()
    if not rank_ic.empty:
        ordered_rank = rank_ic.copy()
        ordered_rank["decision_timestamp"] = pd.to_datetime(
            ordered_rank["decision_timestamp"], utc=True
        )
        ordered_rank = ordered_rank.sort_values(
            "decision_timestamp", kind="mergesort"
        )
        state.add_rank_rows(ordered_rank)
    if not sleeves.empty:
        ordered_sleeves = sleeves.copy()
        ordered_sleeves["decision_timestamp"] = pd.to_datetime(
            ordered_sleeves["decision_timestamp"], utc=True
        )
        ordered_sleeves = ordered_sleeves.sort_values(
            "decision_timestamp", kind="mergesort"
        )
        state.add_sleeve_rows(ordered_sleeves)
    if not daily_returns.empty:
        ordered_daily = daily_returns.sort_values(
            "session_date", kind="mergesort"
        )
        state.add_daily_rows(ordered_daily)
    state.add_terminal_rows(terminal_censored)
    return state


def numerical_differences(
    reference: Any,
    candidate: Any,
    *,
    absolute_tolerance: float = 1e-12,
    path: str = "",
) -> list[dict[str, Any]]:
    """Return exact structural differences with a documented float tolerance."""

    differences: list[dict[str, Any]] = []
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        keys = sorted(set(reference) | set(candidate))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in reference or key not in candidate:
                differences.append(
                    {
                        "path": child,
                        "reference": reference.get(key),
                        "candidate": candidate.get(key),
                    }
                )
                continue
            differences.extend(
                numerical_differences(
                    reference[key],
                    candidate[key],
                    absolute_tolerance=absolute_tolerance,
                    path=child,
                )
            )
        return differences
    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            return [{"path": path, "reference": reference, "candidate": candidate}]
        if len(reference) != len(candidate):
            differences.append(
                {
                    "path": path,
                    "reference_length": len(reference),
                    "candidate_length": len(candidate),
                }
            )
        for index, (left, right) in enumerate(zip(reference, candidate)):
            differences.extend(
                numerical_differences(
                    left,
                    right,
                    absolute_tolerance=absolute_tolerance,
                    path=f"{path}[{index}]",
                )
            )
        return differences
    left_number = _finite(reference)
    right_number = _finite(candidate)
    if left_number is not None and right_number is not None:
        if not math.isclose(
            left_number,
            right_number,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            differences.append(
                {"path": path, "reference": reference, "candidate": candidate}
            )
        return differences
    if reference != candidate:
        differences.append(
            {"path": path, "reference": reference, "candidate": candidate}
        )
    return differences

