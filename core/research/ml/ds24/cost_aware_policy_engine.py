from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo


CENT = 0.01
DEFAULT_EQUITY_TOLERANCE = 1e-7


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class InsufficientCashForFees(ValueError):
    """An otherwise executable trade would violate the no-borrowing contract."""


class EvidenceState(str, Enum):
    COMPLETE = "COMPLETE"
    MISSING_EXECUTION_QUOTE = "MISSING_EXECUTION_QUOTE"
    MISSING_MARK = "MISSING_MARK"


class RetentionMode(str, Enum):
    PURE_RETENTION = "PURE_RETENTION"
    CHALLENGER_OVERRIDE = "CHALLENGER_OVERRIDE"


class ResizeExecutionMode(str, Enum):
    FULL_TARGET = "FULL_TARGET"
    TRADE_TO_BAND_BOUNDARY = "TRADE_TO_BAND_BOUNDARY"


@dataclass(frozen=True)
class FeeSchedule:
    authority: str
    retrieval_date: str
    source: str
    sec_sell_notional_rate: float = 0.0
    taf_sell_per_share: float = 0.0
    taf_trade_cap_usd: float | None = None
    cat_per_share: float = 0.0
    commission_per_trade_usd: float = 0.0
    round_up_to_cent: bool = True

    def __post_init__(self) -> None:
        rates = (
            self.sec_sell_notional_rate, self.taf_sell_per_share,
            self.cat_per_share, self.commission_per_trade_usd,
        )
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("fee rates and commission must be finite and non-negative")
        if self.taf_trade_cap_usd is not None and (
            not math.isfinite(self.taf_trade_cap_usd) or self.taf_trade_cap_usd < 0
        ):
            raise ValueError("TAF trade cap must be finite and non-negative")

    def raw_components(self, side: TradeSide, quantity: float, execution_price: float) -> dict[str, float]:
        if not isinstance(side, TradeSide):
            raise ValueError("trade side must be BUY or SELL")
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("quantity must be finite and non-negative")
        if not math.isfinite(execution_price) or execution_price <= 0:
            raise ValueError("execution_price must be finite and positive")
        notional = quantity * execution_price
        components = {"sec": 0.0, "taf": 0.0, "cat": quantity * self.cat_per_share}
        if side is TradeSide.SELL:
            components["sec"] = notional * self.sec_sell_notional_rate
            taf = quantity * self.taf_sell_per_share
            if self.taf_trade_cap_usd is not None:
                taf = min(taf, self.taf_trade_cap_usd)
            components["taf"] = taf
        return components

    def explicit_fee(self, side: TradeSide, quantity: float, execution_price: float) -> float:
        """Single-trade estimate; use DailyFeeAccumulator for account economics."""
        components = self.raw_components(side, quantity, execution_price)
        return self.commission_per_trade_usd + sum(
            _ceil_cent(value) if self.round_up_to_cent and value > 0 else value
            for value in components.values()
        )

    def stable_identity(self) -> dict[str, object]:
        return {
            "authority": self.authority,
            "retrieval_date": self.retrieval_date,
            "source": self.source,
            "sec_sell_notional_rate": self.sec_sell_notional_rate,
            "taf_sell_per_share": self.taf_sell_per_share,
            "taf_trade_cap_usd": self.taf_trade_cap_usd,
            "cat_per_share": self.cat_per_share,
            "commission_per_trade_usd": self.commission_per_trade_usd,
            "round_up_to_cent": self.round_up_to_cent,
        }


@dataclass
class DailyFeeAccumulator:
    """Charge daily changes in each independently rounded regulatory fee."""

    schedule: FeeSchedule
    raw_by_day: dict[date, dict[str, float]] = field(default_factory=dict)
    total_paid: float = 0.0

    def preview(self, side: TradeSide, quantity: float, price: float, timestamp: datetime) -> float:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("fee timestamp must be timezone-aware")
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("quantity must be finite and non-negative")
        if quantity <= 0:
            return 0.0
        day = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        current = self.raw_by_day.get(day, {})
        increments = self.schedule.raw_components(side, quantity, price)
        fee = self.schedule.commission_per_trade_usd
        for name, increment in increments.items():
            before = current.get(name, 0.0)
            after = before + increment
            if self.schedule.round_up_to_cent:
                fee += _ceil_cent(after) - _ceil_cent(before)
            else:
                fee += increment
        return fee

    def record(self, side: TradeSide, quantity: float, price: float, timestamp: datetime) -> float:
        fee = self.preview(side, quantity, price, timestamp)
        if quantity <= 0:
            return fee
        day = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        current = self.raw_by_day.setdefault(day, {"sec": 0.0, "taf": 0.0, "cat": 0.0})
        for name, increment in self.schedule.raw_components(side, quantity, price).items():
            current[name] += increment
        self.total_paid += fee
        return fee


@dataclass(frozen=True)
class CostContract:
    fee_schedule: FeeSchedule
    extra_slippage_bps_per_side: float
    quote_source: str
    stale_after: timedelta
    version: str = "DS24_COST_AWARE_POLICY_ENGINE_V2_DAILY_FEES"

    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "fee_schedule": self.fee_schedule.stable_identity(),
            "extra_slippage_bps_per_side": self.extra_slippage_bps_per_side,
            "quote_source": self.quote_source,
            "stale_after_seconds": self.stale_after.total_seconds(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PolicyContract:
    policy_id: str
    entry_top_n: int
    exit_rank: int
    max_holdings: int | None = None
    minimum_hold: timedelta = timedelta(0)
    rebalance: timedelta = timedelta(minutes=5)
    minimum_rank_improvement: int = 0
    weighting_mode: str = "equal_weight_control"
    retention_mode: RetentionMode = RetentionMode.CHALLENGER_OVERRIDE
    rank_weight_cap: float | None = None
    weight_no_trade_band_bps: float = 0.0
    resize_execution_mode: ResizeExecutionMode = ResizeExecutionMode.FULL_TARGET
    long_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.retention_mode, RetentionMode):
            object.__setattr__(self, "retention_mode", RetentionMode(self.retention_mode))
        if not isinstance(self.resize_execution_mode, ResizeExecutionMode):
            object.__setattr__(self, "resize_execution_mode", ResizeExecutionMode(self.resize_execution_mode))
        if self.entry_top_n < 1:
            raise ValueError("entry_top_n must be positive")
        if self.exit_rank < self.entry_top_n:
            raise ValueError("exit_rank must be greater than or equal to entry_top_n")
        if self.capacity < 1:
            raise ValueError("max_holdings must be positive")
        if self.minimum_rank_improvement < 0:
            raise ValueError("minimum_rank_improvement must be non-negative")
        if self.rank_weight_cap is not None and not 0.0 < self.rank_weight_cap <= 1.0:
            raise ValueError("rank_weight_cap must be in (0, 1]")
        if self.weight_no_trade_band_bps < 0:
            raise ValueError("weight_no_trade_band_bps must be non-negative")

    @property
    def capacity(self) -> int:
        return self.entry_top_n if self.max_holdings is None else self.max_holdings

    def fingerprint(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "entry_top_n": self.entry_top_n,
            "exit_rank": self.exit_rank,
            "max_holdings": self.max_holdings,
            "minimum_hold_seconds": self.minimum_hold.total_seconds(),
            "rebalance_seconds": self.rebalance.total_seconds(),
            "minimum_rank_improvement": self.minimum_rank_improvement,
            "weighting_mode": self.weighting_mode,
            "retention_mode": self.retention_mode.value,
            "rank_weight_cap": self.rank_weight_cap,
            "weight_no_trade_band_bps": self.weight_no_trade_band_bps,
            "resize_execution_mode": self.resize_execution_mode.value,
            "long_only": self.long_only,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RankSignal:
    symbol: str
    rank: int
    score: float


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    regular_session: bool = True
    source: str | None = None

    def validate_for(self, decision_timestamp: datetime, stale_after: timedelta) -> None:
        if self.timestamp < decision_timestamp:
            raise ValueError("quote timestamp precedes decision timestamp")
        if self.timestamp - decision_timestamp > stale_after:
            raise ValueError("quote is stale for the configured tolerance")
        if not self.regular_session:
            raise ValueError("quote is outside the regular trading session")
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("quote must have 0 < bid <= ask")

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class MarkPrice:
    symbol: str
    timestamp: datetime
    price: float


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_cost: float
    opened_at: datetime
    last_rank: int | None = None

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-12


@dataclass(frozen=True)
class Trade:
    account_id: str
    timestamp: datetime
    symbol: str
    side: TradeSide
    quantity: float
    execution_price: float
    reference_price: float
    trade_notional: float
    explicit_fee: float
    spread_cost: float
    extra_slippage_cost: float
    execution_source: str = ""


@dataclass(frozen=True)
class LedgerSnapshot:
    account_id: str
    timestamp: datetime
    cash: float
    position_value: float
    equity: float
    gross_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    explicit_fees: float
    spread_cost: float
    extra_slippage_cost: float
    evidence_state: EvidenceState
    missing_reason: str | None = None
    unresolved_symbols: tuple[str, ...] = ()


@dataclass
class PortfolioLedger:
    account_id: str
    starting_equity: float = 5_000.0
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    explicit_fees: float = 0.0
    spread_cost: float = 0.0
    extra_slippage_cost: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    snapshots: list[LedgerSnapshot] = field(default_factory=list)
    _last_policy_hash: str | None = None
    _last_cost_hash: str | None = None
    _fee_accumulator: DailyFeeAccumulator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cash = self.starting_equity

    def bind_resume_contracts(self, policy: PolicyContract, cost: CostContract) -> None:
        policy_hash = policy.fingerprint()
        cost_hash = cost.fingerprint()
        if self._last_policy_hash is not None and self._last_policy_hash != policy_hash:
            raise ValueError("resume rejected: incompatible policy configuration")
        if self._last_cost_hash is not None and self._last_cost_hash != cost_hash:
            raise ValueError("resume rejected: incompatible cost configuration")
        self._last_policy_hash = policy_hash
        self._last_cost_hash = cost_hash
        if self._fee_accumulator is None:
            self._fee_accumulator = DailyFeeAccumulator(cost.fee_schedule)

    def equity(self, marks: Mapping[str, MarkPrice]) -> float:
        return self.cash + self._position_value(marks)

    def snapshot(self, timestamp: datetime, marks: Mapping[str, MarkPrice]) -> LedgerSnapshot:
        missing = tuple(sorted(symbol for symbol, pos in self.positions.items() if pos.is_open and symbol not in marks))
        if missing:
            snap = LedgerSnapshot(
                account_id=self.account_id,
                timestamp=timestamp,
                cash=self.cash,
                position_value=0.0,
                equity=self.cash,
                gross_exposure=0.0,
                realized_pnl=self.realized_pnl,
                unrealized_pnl=0.0,
                explicit_fees=self.explicit_fees,
                spread_cost=self.spread_cost,
                extra_slippage_cost=self.extra_slippage_cost,
                evidence_state=EvidenceState.MISSING_MARK,
                missing_reason="missing mark for open position",
                unresolved_symbols=missing,
            )
            self.snapshots.append(snap)
            return snap
        position_value = self._position_value(marks)
        equity = self.cash + position_value
        unrealized = sum(
            pos.quantity * (marks[symbol].price - pos.average_cost)
            for symbol, pos in self.positions.items()
            if pos.is_open
        )
        snap = LedgerSnapshot(
            account_id=self.account_id,
            timestamp=timestamp,
            cash=self.cash,
            position_value=position_value,
            equity=equity,
            gross_exposure=abs(position_value) / equity if equity else 0.0,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            explicit_fees=self.explicit_fees,
            spread_cost=self.spread_cost,
            extra_slippage_cost=self.extra_slippage_cost,
            evidence_state=EvidenceState.COMPLETE,
        )
        if not math.isclose(snap.cash + snap.position_value, snap.equity, rel_tol=0.0, abs_tol=DEFAULT_EQUITY_TOLERANCE):
            raise AssertionError("ledger identity violated: cash + positions != equity")
        self.snapshots.append(snap)
        return snap

    def rebalance(
        self,
        *,
        timestamp: datetime,
        signals: Sequence[RankSignal],
        policy: PolicyContract,
        cost: CostContract,
        quotes: Mapping[str, MarketQuote],
        marks: Mapping[str, MarkPrice],
    ) -> LedgerSnapshot:
        self.bind_resume_contracts(policy, cost)
        for symbol, position in self.positions.items():
            if position.is_open and symbol not in marks:
                return self._missing_mark_snapshot(timestamp, symbol, "missing mark for open position")
        desired_symbols = select_symbols(signals, self.positions, timestamp, policy)
        weights = rank_weights(
            signals,
            desired_symbols,
            policy.weighting_mode,
            rank_weight_cap=policy.rank_weight_cap,
        )
        equity = self.equity(marks)
        current_symbols = {symbol for symbol, position in self.positions.items() if position.is_open}
        weights = apply_weight_no_trade_band(
            target_weights=weights,
            current_weights=self._current_weights(marks, equity),
            retained_symbols=current_symbols & set(weights),
            band_bps=policy.weight_no_trade_band_bps,
            mode=policy.resize_execution_mode,
        )
        desired_quantities: dict[str, float] = {}
        missing_target_marks: list[str] = []
        for symbol, weight in weights.items():
            if symbol not in marks:
                missing_target_marks.append(symbol)
                continue
            desired_quantities[symbol] = (equity * weight) / marks[symbol].price
        for symbol in list(self.positions):
            desired_quantities.setdefault(symbol, 0.0)

        sell_symbols = sorted(
            (symbol for symbol, desired in desired_quantities.items() if desired < self.positions.get(symbol, _flat(symbol, timestamp)).quantity - 1e-10),
            key=lambda symbol: symbol,
        )
        buy_symbols = sorted(
            (symbol for symbol, desired in desired_quantities.items() if desired > self.positions.get(symbol, _flat(symbol, timestamp)).quantity + 1e-10),
            key=lambda symbol: symbol,
        )
        missing_executions: list[str] = []
        for symbol in [*sell_symbols, *buy_symbols]:
            if not self._execute_delta(symbol, desired_quantities[symbol], timestamp, cost, quotes):
                missing_executions.append(symbol)
        for signal in signals:
            pos = self.positions.get(signal.symbol)
            if pos and pos.is_open:
                self.positions[signal.symbol] = Position(pos.symbol, pos.quantity, pos.average_cost, pos.opened_at, signal.rank)
        snapshot = self.snapshot(timestamp, marks)
        if missing_target_marks or missing_executions:
            snapshot = replace(
                snapshot,
                evidence_state=(EvidenceState.MISSING_MARK if missing_target_marks
                                else EvidenceState.MISSING_EXECUTION_QUOTE),
                missing_reason=("missing mark for unfilled target" if missing_target_marks
                                else "missing or invalid execution quote"),
                unresolved_symbols=tuple(sorted(set(missing_target_marks + missing_executions))),
            )
            self.snapshots[-1] = snapshot
        return snapshot

    def liquidate_available(
        self,
        *,
        timestamp: datetime,
        policy: PolicyContract,
        cost: CostContract,
        quotes: Mapping[str, MarketQuote],
        marks: Mapping[str, MarkPrice],
    ) -> LedgerSnapshot:
        """Sell fillable matured positions while preserving unresolved exposure."""
        self.bind_resume_contracts(policy, cost)
        for symbol, position in sorted(self.positions.items()):
            if not position.is_open:
                self.positions.pop(symbol, None)
                continue
            if timestamp - position.opened_at < policy.minimum_hold:
                continue
            self._execute_delta(symbol, 0.0, timestamp, cost, quotes)
        return self.snapshot(timestamp, marks)

    def _execute_delta(
        self,
        symbol: str,
        desired_quantity: float,
        timestamp: datetime,
        cost: CostContract,
        quotes: Mapping[str, MarketQuote],
    ) -> bool:
        existing = self.positions.get(symbol, _flat(symbol, timestamp))
        delta = desired_quantity - existing.quantity
        if abs(delta) <= 1e-10:
            if desired_quantity <= 1e-10 and existing.quantity <= 1e-10:
                self.positions.pop(symbol, None)
            return True
        quote = quotes.get(symbol)
        if quote is None:
            return False
        try:
            quote.validate_for(timestamp, cost.stale_after)
        except ValueError:
            return False
        side = TradeSide.BUY if delta > 0 else TradeSide.SELL
        quantity = abs(delta)
        reference = quote.ask if side is TradeSide.BUY else quote.bid
        slippage = reference * cost.extra_slippage_bps_per_side / 10_000.0
        execution_price = reference + slippage if side is TradeSide.BUY else reference - slippage
        if execution_price <= 0:
            raise ValueError("execution price must be positive after slippage")
        trade_notional = quantity * execution_price
        fees = self._fee_accumulator
        if fees is None:
            raise RuntimeError("fee accumulator was not bound to the account")
        execution_timestamp = quote.timestamp
        explicit_fee = fees.preview(side, quantity, execution_price, execution_timestamp)
        spread_cost = abs(reference - quote.midpoint) * quantity
        extra_slippage_cost = slippage * quantity
        if side is TradeSide.BUY:
            total_debit = trade_notional + explicit_fee
            if total_debit > self.cash + 1e-7:
                scale = max(0.0, self.cash / total_debit)
                quantity *= scale
                trade_notional = quantity * execution_price
                explicit_fee = fees.preview(side, quantity, execution_price, execution_timestamp)
                if trade_notional + explicit_fee > self.cash and execution_price > 0:
                    quantity = max(0.0, (self.cash - explicit_fee) / execution_price)
                    trade_notional = quantity * execution_price
                    explicit_fee = fees.preview(side, quantity, execution_price, execution_timestamp)
                spread_cost = abs(reference - quote.midpoint) * quantity
                extra_slippage_cost = slippage * quantity
                total_debit = trade_notional + explicit_fee
            if quantity <= 1e-10:
                return False
            if total_debit > self.cash + 1e-9:
                return False
            self.cash -= total_debit
            if self.cash < 0 and self.cash > -1e-9:
                self.cash = 0.0
            fees.record(side, quantity, execution_price, execution_timestamp)
            new_quantity = existing.quantity + quantity
            new_cost_basis = ((existing.quantity * existing.average_cost) + trade_notional) / new_quantity if new_quantity else 0.0
            opened_at = existing.opened_at if existing.is_open else execution_timestamp
            self.positions[symbol] = Position(symbol, new_quantity, new_cost_basis, opened_at, existing.last_rank)
        else:
            quantity = min(quantity, existing.quantity)
            trade_notional = quantity * execution_price
            explicit_fee = fees.preview(side, quantity, execution_price, execution_timestamp)
            if self.cash + trade_notional < explicit_fee - 1e-9:
                raise InsufficientCashForFees(
                    f"sell of {symbol} cannot settle explicit fees without borrowing"
                )
            fees.record(side, quantity, execution_price, execution_timestamp)
            spread_cost = abs(reference - quote.midpoint) * quantity
            extra_slippage_cost = slippage * quantity
            self.cash += trade_notional - explicit_fee
            if self.cash < 0 and self.cash > -1e-9:
                self.cash = 0.0
            self.realized_pnl += quantity * (execution_price - existing.average_cost) - explicit_fee
            new_quantity = existing.quantity - quantity
            if new_quantity <= 1e-10:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = Position(symbol, new_quantity, existing.average_cost, existing.opened_at, existing.last_rank)
        self.explicit_fees += explicit_fee
        self.spread_cost += spread_cost
        self.extra_slippage_cost += extra_slippage_cost
        self.trades.append(
            Trade(
                account_id=self.account_id,
                timestamp=execution_timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                execution_price=execution_price,
                reference_price=reference,
                trade_notional=trade_notional,
                explicit_fee=explicit_fee,
                spread_cost=spread_cost,
                extra_slippage_cost=extra_slippage_cost,
                execution_source=quote.source or cost.quote_source,
            )
        )
        return True

    def _position_value(self, marks: Mapping[str, MarkPrice]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_open:
                total += pos.quantity * marks[symbol].price
        return total

    def _current_weights(self, marks: Mapping[str, MarkPrice], equity: float) -> dict[str, float]:
        if equity <= 0:
            return {}
        return {
            symbol: position.quantity * marks[symbol].price / equity
            for symbol, position in self.positions.items()
            if position.is_open and symbol in marks
        }

    def _missing_mark_snapshot(self, timestamp: datetime, symbol: str, reason: str) -> LedgerSnapshot:
        snap = LedgerSnapshot(
            account_id=self.account_id,
            timestamp=timestamp,
            cash=self.cash,
            position_value=0.0,
            equity=self.cash,
            gross_exposure=0.0,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=0.0,
            explicit_fees=self.explicit_fees,
            spread_cost=self.spread_cost,
            extra_slippage_cost=self.extra_slippage_cost,
            evidence_state=EvidenceState.MISSING_MARK,
            missing_reason=reason,
            unresolved_symbols=(symbol,),
        )
        self.snapshots.append(snap)
        return snap


def rank_weights(
    signals: Sequence[RankSignal],
    selected_symbols: Sequence[str],
    mode: str,
    *,
    rank_weight_cap: float | None = None,
) -> dict[str, float]:
    if not selected_symbols:
        return {}
    by_symbol = {signal.symbol: signal for signal in signals}
    missing = [symbol for symbol in selected_symbols if symbol not in by_symbol]
    if missing:
        raise ValueError(f"selected symbols missing rank signals: {missing}")
    ordered = sorted((by_symbol[symbol] for symbol in selected_symbols), key=lambda signal: signal.rank)
    if mode == "equal_weight_control":
        raw = {signal.symbol: 1.0 for signal in ordered}
    elif mode == "linear_rank_tilt_1_5x":
        count = len(ordered)
        if count == 1:
            raw = {ordered[0].symbol: 1.0}
        else:
            raw = {signal.symbol: 1.5 - 0.5 * index / (count - 1) for index, signal in enumerate(ordered)}
    elif mode == "sqrt_inverse_rank":
        raw = {signal.symbol: 1.0 / math.sqrt(signal.rank) for signal in ordered}
    elif mode == "capped_inverse_rank":
        raw = {signal.symbol: 1.0 / signal.rank for signal in ordered}
    else:
        raise ValueError(f"unsupported weighting mode: {mode}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("rank weights must have positive total")
    weights = {symbol: value / total for symbol, value in raw.items()}
    if mode == "capped_inverse_rank":
        if rank_weight_cap is None:
            raise ValueError("capped_inverse_rank requires rank_weight_cap")
        weights = _cap_and_redistribute(weights, rank_weight_cap)
    return weights


def apply_weight_no_trade_band(
    *,
    target_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    retained_symbols: set[str],
    band_bps: float,
    mode: ResizeExecutionMode,
) -> dict[str, float]:
    if band_bps <= 0.0 or not retained_symbols:
        return dict(target_weights)
    band = band_bps / 10_000.0
    adjusted = dict(target_weights)
    for symbol in retained_symbols:
        current = current_weights.get(symbol)
        target = target_weights.get(symbol)
        if current is None or target is None:
            continue
        delta = target - current
        if abs(delta) <= band:
            adjusted[symbol] = current
        elif mode is ResizeExecutionMode.TRADE_TO_BAND_BOUNDARY:
            adjusted[symbol] = target - band if delta > 0 else target + band
        elif mode is not ResizeExecutionMode.FULL_TARGET:
            raise ValueError(f"unsupported resize execution mode: {mode}")
    return adjusted


def select_symbols(
    signals: Sequence[RankSignal],
    positions: Mapping[str, Position],
    timestamp: datetime,
    policy: PolicyContract,
) -> tuple[str, ...]:
    by_symbol = {signal.symbol: signal for signal in signals}
    selected: set[str] = set()
    protected: set[str] = set()
    for symbol, position in positions.items():
        signal = by_symbol.get(symbol)
        if not position.is_open:
            continue
        if timestamp - position.opened_at < policy.minimum_hold:
            selected.add(symbol)
            protected.add(symbol)
        elif signal is not None and signal.rank <= policy.exit_rank:
            selected.add(symbol)
    selected = _apply_replacement_buffer(selected, protected, by_symbol, positions, policy)
    return tuple(sorted(selected, key=lambda symbol: by_symbol[symbol].rank if symbol in by_symbol else 10**9))


def validate_policy_admission(
    *,
    candidate_symbols: Sequence[str],
    available_symbols: Sequence[str],
    future_complete_filter_used: bool,
) -> None:
    if future_complete_filter_used:
        raise ValueError("future-completeness filtering is prohibited for policy admission")
    unavailable = sorted(set(candidate_symbols) - set(available_symbols))
    if unavailable:
        raise ValueError(f"policy admission references unavailable current evidence: {unavailable}")


def _apply_replacement_buffer(
    selected: set[str],
    protected: set[str],
    by_symbol: Mapping[str, RankSignal],
    positions: Mapping[str, Position],
    policy: PolicyContract,
) -> set[str]:
    capacity = policy.capacity
    if len(selected) > capacity:
        raise ValueError("open retained positions exceed policy holding capacity")
    open_symbols = {
        symbol for symbol, position in positions.items()
        if position.is_open
    }
    challengers = sorted(
        [signal for signal in by_symbol.values() if signal.symbol not in open_symbols and signal.rank <= policy.entry_top_n],
        key=lambda signal: (signal.rank, signal.symbol),
    )
    for challenger in challengers:
        if policy.retention_mode is RetentionMode.PURE_RETENTION:
            if len(selected) < capacity:
                selected.add(challenger.symbol)
            continue
        replaceable = [symbol for symbol in selected if symbol in positions and symbol not in protected]
        if replaceable:
            incumbent = max(replaceable, key=lambda symbol: (
                by_symbol[symbol].rank if symbol in by_symbol else 10**9, symbol,
            ))
            incumbent_rank = by_symbol[incumbent].rank if incumbent in by_symbol else 10**9
            improvement = incumbent_rank - challenger.rank
            if improvement >= policy.minimum_rank_improvement:
                selected.discard(incumbent)
                selected.add(challenger.symbol)
        elif len(selected) < capacity:
            selected.add(challenger.symbol)
        if len(selected) > capacity:
            raise AssertionError("policy holding capacity exceeded")
    return selected


def _cap_and_redistribute(weights: Mapping[str, float], cap: float) -> dict[str, float]:
    remaining = dict(weights)
    capped: dict[str, float] = {}
    while remaining:
        overweight = {symbol: weight for symbol, weight in remaining.items() if weight > cap}
        if not overweight:
            residual = 1.0 - sum(capped.values())
            total = sum(remaining.values())
            capped.update({symbol: residual * weight / total for symbol, weight in remaining.items()})
            break
        for symbol in overweight:
            capped[symbol] = cap
            remaining.pop(symbol)
        residual = 1.0 - sum(capped.values())
        if residual < -1e-12:
            raise ValueError("rank_weight_cap is too low for the selected holdings")
        total = sum(remaining.values())
        if total <= 0:
            break
        remaining = {symbol: residual * weight / total for symbol, weight in remaining.items()}
    total = sum(capped.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("capped rank weights failed to normalize")
    return capped


def _flat(symbol: str, timestamp: datetime) -> Position:
    return Position(symbol=symbol, quantity=0.0, average_cost=0.0, opened_at=timestamp)


def _ceil_cent(value: float) -> float:
    return math.ceil((value - 1e-12) / CENT) * CENT
