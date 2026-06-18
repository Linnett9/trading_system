"""Strategy construction for research and backtest workflows.

The rest of the application asks for a strategy by name through
``build_strategy(symbol, config)``. Internally, this module keeps strategy
creation behind small registry objects so adding a strategy is a localized
change: define a builder and register it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.interfaces.strategy import IStrategy
from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from strategies.bollinger_mean_reversion_strategy import (
    BollingerMeanReversionStrategy,
)
from strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
from strategies.ema_crossover_strategy import EMACrossoverStrategy
from strategies.ema_rsi_filter_strategy import EMARSIFilterStrategy
from strategies.ema_rsi_pullback_strategy import EMARSIPullbackStrategy
from strategies.ensemble_vote_strategy import EnsembleVoteStrategy
from strategies.rsi_mean_reversion_strategy import RSIMeanReversionStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.trend_pullback_strategy import TrendPullbackStrategy

StrategyBuilder = Callable[[str, dict[str, Any]], IStrategy]


@dataclass(frozen=True)
class StrategyDefinition:
    """Named construction rule for one configured strategy."""

    name: str
    builder: StrategyBuilder

    def build(self, symbol: str, config: dict[str, Any]) -> IStrategy:
        """Create the strategy instance for a symbol and config block."""

        return self.builder(symbol, config)


class StrategyRegistry:
    """Lookup table that owns available strategy definitions.

    The registry keeps construction pluggable and explicit. Callers depend on
    the registry interface, while each concrete strategy remains responsible
    only for its own trading behavior.
    """

    def __init__(self, definitions: list[StrategyDefinition]):
        self._definitions = {
            definition.name: definition
            for definition in definitions
        }

    def build(
        self,
        symbol: str,
        config: dict[str, Any] | None = None,
    ) -> IStrategy:
        """Build the strategy named in config, defaulting to EMA crossover."""

        strategy_config = config or {}
        name = strategy_config.get("name", "ema_crossover")

        try:
            definition = self._definitions[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._definitions))
            raise ValueError(
                f"Unknown strategy: {name}. Available strategies: {available}"
            ) from exc

        return definition.build(symbol, strategy_config)

    def names(self) -> list[str]:
        """Return the configured strategy names in sorted order."""

        return sorted(self._definitions)


def _build_ema_crossover(symbol: str, config: dict[str, Any]) -> IStrategy:
    return EMACrossoverStrategy(
        symbol=symbol,
        fast_period=config.get("ema_fast_period", 50),
        slow_period=config.get("ema_slow_period", 200),
    )


def _build_rsi(symbol: str, config: dict[str, Any]) -> IStrategy:
    return RSIStrategy(
        symbol=symbol,
        period=config.get("rsi_period", 14),
        oversold=config.get("rsi_oversold", 30),
        overbought=config.get("rsi_overbought", 70),
    )


def _build_rsi_mean_reversion(
    symbol: str,
    config: dict[str, Any],
) -> IStrategy:
    return RSIMeanReversionStrategy(
        symbol=symbol,
        period=config.get("rsi_period", 14),
        oversold=config.get("rsi_oversold", 35),
        exit_level=config.get("rsi_exit_level", 50),
        overbought=config.get("rsi_overbought", 70),
        require_sideways_regime=config.get("require_sideways_regime", False),
    )


def _build_sideways_rsi_mean_reversion(
    symbol: str,
    config: dict[str, Any],
) -> IStrategy:
    return RSIMeanReversionStrategy(
        symbol=symbol,
        period=config.get("rsi_period", 14),
        oversold=config.get("rsi_oversold", 40),
        exit_level=config.get("rsi_exit_level", 55),
        overbought=config.get("rsi_overbought", 70),
        require_sideways_regime=True,
    )


def _build_donchian_breakout(
    symbol: str,
    config: dict[str, Any],
) -> IStrategy:
    return DonchianBreakoutStrategy(
        symbol=symbol,
        lookback_period=config.get("donchian_lookback", 20),
        use_volatility_filter=config.get("use_volatility_filter", False),
        use_volume_filter=config.get("use_volume_filter", False),
        min_relative_volume=config.get("min_relative_volume", 1.0),
    )


def _build_volatility_filtered_donchian(
    symbol: str,
    config: dict[str, Any],
) -> IStrategy:
    return DonchianBreakoutStrategy(
        symbol=symbol,
        lookback_period=config.get("donchian_lookback", 20),
        use_volatility_filter=True,
        use_volume_filter=config.get("use_volume_filter", False),
        min_relative_volume=config.get("min_relative_volume", 1.0),
    )


def _build_ema_rsi_filter(symbol: str, config: dict[str, Any]) -> IStrategy:
    return EMARSIFilterStrategy(
        symbol=symbol,
        fast_period=config.get("ema_fast_period", 20),
        slow_period=config.get("ema_slow_period", 50),
        rsi_entry=config.get("rsi_entry", 50),
        rsi_exit=config.get("rsi_exit", 45),
    )


def _build_ema_rsi_pullback(
    symbol: str,
    config: dict[str, Any],
) -> IStrategy:
    return EMARSIPullbackStrategy(
        symbol=symbol,
        fast_period=config.get("ema_fast_period", 20),
        slow_period=config.get("ema_slow_period", 200),
        rsi_pullback=config.get("rsi_pullback", 45),
        rsi_recover=config.get("rsi_recover", 55),
        use_regime_filter=config.get("use_regime_filter", True),
        min_relative_volume=config.get("min_relative_volume"),
    )


def _build_bollinger_mean_reversion(
    symbol: str,
    config: dict[str, Any],
) -> IStrategy:
    return BollingerMeanReversionStrategy(
        symbol=symbol,
        rsi_entry=config.get("rsi_entry", 40),
        rsi_exit=config.get("rsi_exit", 55),
        require_sideways_regime=config.get("require_sideways_regime", False),
        min_bandwidth=config.get("min_bandwidth"),
        max_bandwidth=config.get("max_bandwidth"),
    )


def _build_trend_pullback(symbol: str, config: dict[str, Any]) -> IStrategy:
    return TrendPullbackStrategy(
        symbol=symbol,
        fast_period=config.get("pullback_fast_period", 50),
        trend_fast_period=config.get("trend_fast_period", 50),
        trend_slow_period=config.get("trend_slow_period", 200),
        pullback_tolerance=config.get("pullback_tolerance", 0.02),
        exit_extension=config.get("pullback_exit_extension", 0.08),
        use_regime_filter=config.get("use_regime_filter", True),
        min_adx=config.get("min_adx"),
        min_relative_volume=config.get("min_relative_volume"),
    )


def _build_ensemble_vote(symbol: str, config: dict[str, Any]) -> IStrategy:
    return EnsembleVoteStrategy(
        symbol=symbol,
        min_buy_votes=config.get("ensemble_min_buy_votes", 2),
        min_sell_votes=config.get("ensemble_min_sell_votes", 1),
        rsi_entry=config.get("rsi_entry", 50),
        rsi_exit=config.get("rsi_exit", 45),
        rsi_pullback=config.get("rsi_pullback", 45),
        pullback_tolerance=config.get("pullback_tolerance", 0.02),
        use_regime_filter=config.get("use_regime_filter", True),
        use_breakout_vote=config.get("use_breakout_vote", True),
    )


def _build_buy_and_hold(symbol: str, config: dict[str, Any]) -> IStrategy:
    return BuyAndHoldStrategy(symbol=symbol)


def default_strategy_registry() -> StrategyRegistry:
    """Create the standard strategy registry used by research runs."""

    return StrategyRegistry([
        StrategyDefinition("ema_crossover", _build_ema_crossover),
        StrategyDefinition("rsi", _build_rsi),
        StrategyDefinition("rsi_mean_reversion", _build_rsi_mean_reversion),
        StrategyDefinition(
            "rsi_sideways_mean_reversion",
            _build_sideways_rsi_mean_reversion,
        ),
        StrategyDefinition("donchian_breakout", _build_donchian_breakout),
        StrategyDefinition(
            "donchian_with_volatility_filter",
            _build_volatility_filtered_donchian,
        ),
        StrategyDefinition("ema_rsi_filter", _build_ema_rsi_filter),
        StrategyDefinition("ema_rsi_pullback", _build_ema_rsi_pullback),
        StrategyDefinition(
            "bollinger_mean_reversion",
            _build_bollinger_mean_reversion,
        ),
        StrategyDefinition("trend_pullback", _build_trend_pullback),
        StrategyDefinition("ensemble_vote", _build_ensemble_vote),
        StrategyDefinition("buy_and_hold", _build_buy_and_hold),
    ])


DEFAULT_STRATEGY_REGISTRY = default_strategy_registry()


def build_strategy(symbol: str, config: dict[str, Any]) -> IStrategy:
    """Build a configured strategy using the default registry.

    This compatibility wrapper preserves the public API used by the existing
    backtest runner and tests.
    """

    return DEFAULT_STRATEGY_REGISTRY.build(symbol, config)
