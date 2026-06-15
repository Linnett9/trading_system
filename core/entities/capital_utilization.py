from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalUtilization:
    average_position_value: float = 0
    average_exposure_percent: float = 0
    max_exposure_percent: float = 0
    average_cash_percent: float = 1
    average_leverage: float = 0

    def to_dict(self) -> dict:
        return {
            "average_position_value": self.average_position_value,
            "average_exposure_percent": self.average_exposure_percent,
            "max_exposure_percent": self.max_exposure_percent,
            "average_cash_percent": self.average_cash_percent,
            "average_leverage": self.average_leverage,
        }
