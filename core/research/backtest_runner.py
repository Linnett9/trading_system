from copy import deepcopy

from core.engine.backtest_engine import BacktestEngine
from core.engine.execution_engine import ExecutionEngine
from core.execution.simple_execution_model import SimpleExecutionModel
from core.research.strategy_factory import build_strategy
from core.risk.atr_risk_manager import ATRRiskManager
from core.risk.position_sizer import build_position_sizer
from core.risk.simple_risk_manager import SimpleRiskManager
from core.risk.volatility_risk_manager import VolatilityRiskManager
from core.services.portfolio_engine import PortfolioEngine
from core.services.trade_manager import TradeManager
from infrastructure.broker.paper_broker import PaperBroker


def merge_config(config: dict, overrides: dict | None = None) -> dict:
    merged = deepcopy(config)

    for section, values in (overrides or {}).items():
        if section not in merged:
            merged[section] = {}

        merged[section].update(values)

    return merged


def build_risk_manager(config: dict):
    risk_config = config["risk"]
    manager = risk_config.get("manager", "simple")
    position_sizer = build_position_sizer(config)

    if manager == "atr":
        return ATRRiskManager(
            max_risk_per_trade=risk_config["max_risk_per_trade"],
            max_exposure=risk_config["max_exposure"],
            atr_multiplier=risk_config["atr_multiplier"],
            position_sizer=position_sizer,
        )

    if manager == "volatility":
        return VolatilityRiskManager(
            max_risk_per_trade=risk_config["max_risk_per_trade"],
            max_exposure=risk_config["max_exposure"],
            position_sizer=position_sizer,
        )

    return SimpleRiskManager(
        max_risk_per_trade=risk_config["max_risk_per_trade"],
        max_exposure=risk_config["max_exposure"],
        position_sizer=position_sizer,
    )


def run_backtest(
    candles,
    symbol: str,
    config: dict,
    overrides: dict | None = None,
):
    run_config = merge_config(config, overrides)
    backtest_config = run_config["backtest"]
    risk_config = run_config["risk"]
    execution_config = run_config["execution"]
    research_config = run_config.get("research", {})

    trade_manager = TradeManager()
    portfolio = PortfolioEngine(
        starting_cash=backtest_config["starting_equity"],
    )

    execution_model = SimpleExecutionModel(
        spread_bps=execution_config["spread_bps"],
        slippage_bps=execution_config["slippage_bps"],
        seed=execution_config.get("seed"),
    )

    execution_engine = ExecutionEngine(
        broker=PaperBroker(None),
        trade_manager=trade_manager,
        execution_model=execution_model,
    )

    engine = BacktestEngine(
        data_feed=None,
        strategy=build_strategy(symbol, run_config["strategy"]),
        risk_manager=build_risk_manager(run_config),
        execution_engine=execution_engine,
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol=symbol,
        timeframe=backtest_config["timeframe"],
        account_equity=backtest_config["starting_equity"],
        warmup_bars=backtest_config["warmup_bars"],
        atr_stop_multiplier=risk_config["atr_stop_multiplier"],
        atr_take_profit_multiplier=risk_config["atr_take_profit_multiplier"],
        trailing_atr_multiplier=risk_config.get("trailing_atr_multiplier"),
        close_open_trades_at_end=True,
        early_stop_max_drawdown=research_config.get(
            "early_stop_max_drawdown",
        ),
        early_stop_equity_floor=(
            backtest_config["starting_equity"]
            * research_config["early_stop_equity_floor_pct"]
            if research_config.get("early_stop_equity_floor_pct") is not None
            else None
        ),
    )

    return engine.run(candles)
