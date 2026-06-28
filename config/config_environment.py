import os


def _apply_environment_credentials(config):
    """Keep provider credentials out of tracked configuration files."""
    alpaca_config = config.setdefault("alpaca", {})
    api_key = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
    )
    secret_key = (
        os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET")
        or os.environ.get("APCA_API_SECRET_KEY")
    )
    if api_key:
        alpaca_config["api_key"] = api_key
    if secret_key:
        alpaca_config["secret_key"] = secret_key
