from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from strategies.bollinger_mean_reversion_strategy import (
    BollingerMeanReversionStrategy,
)
from strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
from strategies.ema_rsi_pullback_strategy import EMARSIPullbackStrategy
from strategies.ema_rsi_filter_strategy import EMARSIFilterStrategy
from strategies.ema_crossover_strategy import EMACrossoverStrategy
from strategies.ensemble_vote_strategy import EnsembleVoteStrategy
from strategies.rsi_mean_reversion_strategy import RSIMeanReversionStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.trend_pullback_strategy import TrendPullbackStrategy


def build_strategy(symbol: str, config: dict):
    name = config.get("name", "ema_crossover")

    if name == "ema_crossover":
        return EMACrossoverStrategy(
            symbol=symbol,
            fast_period=config.get("ema_fast_period", 50),
            slow_period=config.get("ema_slow_period", 200),
        )

    if name == "rsi":
        return RSIStrategy(
            symbol=symbol,
            period=config.get("rsi_period", 14),
            oversold=config.get("rsi_oversold", 30),
            overbought=config.get("rsi_overbought", 70),
        )

    if name == "rsi_mean_reversion":
        return RSIMeanReversionStrategy(
            symbol=symbol,
            period=config.get("rsi_period", 14),
            oversold=config.get("rsi_oversold", 35),
            exit_level=config.get("rsi_exit_level", 50),
            overbought=config.get("rsi_overbought", 70),
            require_sideways_regime=config.get(
                "require_sideways_regime",
                False,
            ),
        )

    if name == "rsi_sideways_mean_reversion":
        return RSIMeanReversionStrategy(
            symbol=symbol,
            period=config.get("rsi_period", 14),
            oversold=config.get("rsi_oversold", 40),
            exit_level=config.get("rsi_exit_level", 55),
            overbought=config.get("rsi_overbought", 70),
            require_sideways_regime=True,
        )

    if name == "donchian_breakout":
        return DonchianBreakoutStrategy(
            symbol=symbol,
            lookback_period=config.get("donchian_lookback", 20),
            use_volatility_filter=config.get(
                "use_volatility_filter",
                False,
            ),
            use_volume_filter=config.get("use_volume_filter", False),
            min_relative_volume=config.get("min_relative_volume", 1.0),
        )

    if name == "donchian_with_volatility_filter":
        return DonchianBreakoutStrategy(
            symbol=symbol,
            lookback_period=config.get("donchian_lookback", 20),
            use_volatility_filter=True,
            use_volume_filter=config.get("use_volume_filter", False),
            min_relative_volume=config.get("min_relative_volume", 1.0),
        )

    if name == "ema_rsi_filter":
        return EMARSIFilterStrategy(
            symbol=symbol,
            fast_period=config.get("ema_fast_period", 20),
            slow_period=config.get("ema_slow_period", 50),
            rsi_entry=config.get("rsi_entry", 50),
            rsi_exit=config.get("rsi_exit", 45),
        )

    if name == "ema_rsi_pullback":
        return EMARSIPullbackStrategy(
            symbol=symbol,
            fast_period=config.get("ema_fast_period", 20),
            slow_period=config.get("ema_slow_period", 200),
            rsi_pullback=config.get("rsi_pullback", 45),
            rsi_recover=config.get("rsi_recover", 55),
            use_regime_filter=config.get("use_regime_filter", True),
            min_relative_volume=config.get("min_relative_volume"),
        )

    if name == "bollinger_mean_reversion":
        return BollingerMeanReversionStrategy(
            symbol=symbol,
            rsi_entry=config.get("rsi_entry", 40),
            rsi_exit=config.get("rsi_exit", 55),
            require_sideways_regime=config.get(
                "require_sideways_regime",
                False,
            ),
            min_bandwidth=config.get("min_bandwidth"),
            max_bandwidth=config.get("max_bandwidth"),
        )

    if name == "trend_pullback":
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

    if name == "ensemble_vote":
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

    if name == "buy_and_hold":
        return BuyAndHoldStrategy(symbol=symbol)

    raise ValueError(f"Unknown strategy: {name}")
