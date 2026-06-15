from datetime import datetime

import pytest

from core.entities.risk_context import RiskContext
from core.entities.signal import Signal
from core.risk.atr_risk_manager import ATRRiskManager
from core.risk.simple_risk_manager import SimpleRiskManager
from core.risk.volatility_risk_manager import VolatilityRiskManager


def make_signal(action="BUY"):
    return Signal(
        symbol="AAPL",
        action=action,
        timestamp=datetime(2024, 1, 1),
    )


def test_simple_risk_manager_converts_notional_to_quantity():
    manager = SimpleRiskManager(
        max_risk_per_trade=0.01,
        max_exposure=0.2,
    )

    size = manager.position_size(make_signal(), 10_000, 200)

    assert size == 0.5


def test_simple_risk_manager_rejects_invalid_price():
    manager = SimpleRiskManager()

    with pytest.raises(ValueError):
        manager.position_size(make_signal(), 10_000, 0)


def test_atr_risk_manager_uses_context_atr_and_exposure_cap():
    manager = ATRRiskManager(
        max_risk_per_trade=0.01,
        max_exposure=0.02,
        atr_multiplier=2,
        atr_value=100,
    )

    size = manager.position_size(
        make_signal(),
        account_equity=10_000,
        market_price=100,
        risk_context=RiskContext(atr=5),
    )

    assert size == 2


def test_atr_risk_manager_reduces_size_when_atr_rises():
    manager = ATRRiskManager(
        max_risk_per_trade=0.01,
        max_exposure=1.0,
        atr_multiplier=2,
    )

    low_atr_size = manager.position_size(
        make_signal(),
        10_000,
        100,
        RiskContext(atr=2),
    )
    high_atr_size = manager.position_size(
        make_signal(),
        10_000,
        100,
        RiskContext(atr=10),
    )

    assert high_atr_size < low_atr_size


def test_atr_risk_manager_rejects_invalid_atr():
    manager = ATRRiskManager()

    with pytest.raises(ValueError):
        manager.position_size(
            make_signal(),
            10_000,
            100,
            RiskContext(atr=0),
        )


def test_volatility_risk_manager_reduces_size_when_volatility_rises():
    manager = VolatilityRiskManager(
        max_risk_per_trade=1.0,
        max_exposure=0.2,
    )

    low_vol_size = manager.position_size(
        make_signal(),
        10_000,
        100,
        RiskContext(volatility=0.01),
    )
    high_vol_size = manager.position_size(
        make_signal(),
        10_000,
        100,
        RiskContext(volatility=0.20),
    )

    assert high_vol_size < low_vol_size


def test_volatility_risk_manager_rejects_invalid_volatility():
    manager = VolatilityRiskManager()

    with pytest.raises(ValueError):
        manager.position_size(
            make_signal(),
            10_000,
            100,
            RiskContext(volatility=0),
        )
