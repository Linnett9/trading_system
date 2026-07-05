from __future__ import annotations


def format_percent(value):
    return f"{value * 100:.2f}%"

def format_optional_percent(value):
    if value is None:
        return "n/a"

    return format_percent(value)
