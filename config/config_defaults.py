from config.config_defaults_ml import ML_DEFAULTS
from config.config_defaults_research import RESEARCH_DEFAULTS
from config.config_defaults_runtime import (
    ALERTS_DEFAULTS,
    BACKTEST_DEFAULTS,
    BROKER_DEFAULTS,
    CACHE_DEFAULTS,
    EXECUTION_DEFAULTS,
    PAPER_CANDIDATE_ID_DEFAULT,
    PAPER_TRADING_DEFAULTS,
    PORTFOLIO_DEFAULTS,
    POSITION_SIZING_DEFAULTS,
    REPORTS_DEFAULTS,
    RISK_DEFAULTS,
    STRATEGY_DEFAULTS,
    TRADING_DEFAULTS,
)


DEFAULT_CONFIG = {
    "paper_candidate_id": PAPER_CANDIDATE_ID_DEFAULT,
    "backtest": BACKTEST_DEFAULTS,
    "strategy": STRATEGY_DEFAULTS,
    "position_sizing": POSITION_SIZING_DEFAULTS,
    "paper_trading": PAPER_TRADING_DEFAULTS,
    "trading": TRADING_DEFAULTS,
    "broker": BROKER_DEFAULTS,
    "portfolio": PORTFOLIO_DEFAULTS,
    "execution": EXECUTION_DEFAULTS,
    "alerts": ALERTS_DEFAULTS,
    "ml": ML_DEFAULTS,
    "risk": RISK_DEFAULTS,
    "research": RESEARCH_DEFAULTS,
    "reports": REPORTS_DEFAULTS,
    "cache": CACHE_DEFAULTS,
}
