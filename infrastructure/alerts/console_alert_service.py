from core.interfaces.alert_service import IAlertService


class ConsoleAlertService(IAlertService):

    def send_alert(self, title: str, message: str, severity: str) -> None:
        print(f"ALERT [{severity}] {title}: {message}")
