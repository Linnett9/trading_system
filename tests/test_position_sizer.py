import pytest

from core.entities.risk_context import RiskContext
from core.risk.position_sizer import (
    ATRPositionSizer,
    FixedDollarSizer,
    FixedFractionalSizer,
    VolatilitySizer,
    build_position_sizer,
)


def test_fixed_fractional_sizer_uses_target_exposure():
    sizer = FixedFractionalSizer(
        target_exposure=0.20,
        max_exposure=0.50,
    )

    assert sizer.size(500, 100) == 1


def test_fixed_fractional_sizer_respects_max_exposure():
    sizer = FixedFractionalSizer(
        target_exposure=0.80,
        max_exposure=0.20,
    )

    assert sizer.size(500, 100) == 1


def test_fixed_dollar_sizer_caps_at_max_exposure():
    sizer = FixedDollarSizer(
        dollar_amount=200,
        max_exposure=0.20,
    )

    assert sizer.size(500, 100) == 1


def test_atr_position_sizer_reduces_size_when_atr_rises():
    sizer = ATRPositionSizer(
        max_risk_per_trade=0.01,
        max_exposure=1.0,
        atr_multiplier=2,
    )

    low_atr = sizer.size(500, 100, RiskContext(atr=1))
    high_atr = sizer.size(500, 100, RiskContext(atr=5))

    assert high_atr < low_atr


def test_volatility_sizer_reduces_size_when_volatility_rises():
    sizer = VolatilitySizer(
        target_exposure=0.20,
        max_exposure=0.20,
    )

    low_vol = sizer.size(500, 100, RiskContext(volatility=0.01))
    high_vol = sizer.size(500, 100, RiskContext(volatility=0.20))

    assert high_vol < low_vol


def test_build_position_sizer_from_config():
    sizer = build_position_sizer({
        "risk": {
            "max_risk_per_trade": 0.01,
            "max_exposure": 0.20,
            "atr_multiplier": 2,
        },
        "position_sizing": {
            "mode": "fixed_fractional",
            "target_exposure": 0.20,
            "max_exposure": 0.20,
        },
    })

    assert sizer.size(500, 100) == 1


def test_position_sizer_rejects_invalid_mode():
    with pytest.raises(ValueError):
        build_position_sizer({
            "risk": {
                "max_risk_per_trade": 0.01,
                "max_exposure": 0.20,
                "atr_multiplier": 2,
            },
            "position_sizing": {
                "mode": "unknown",
            },
        })
