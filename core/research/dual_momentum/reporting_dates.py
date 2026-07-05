from datetime import datetime


def parse_config_date(value):
    return datetime.fromisoformat(value)
