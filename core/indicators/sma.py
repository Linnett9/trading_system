def sma(values, period):

    if len(values) < period:
        return None

    return sum(values[-period:]) / period