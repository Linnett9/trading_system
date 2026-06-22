from abc import ABC, abstractmethod


class IAlertService(ABC):

    @abstractmethod
    def send_alert(self, title: str, message: str, severity: str) -> None:
        pass
