from core.research.ml.stock_level.prediction_artifacts import rows as _rows

globals().update({name: value for name, value in vars(_rows).items() if not name.startswith("__")})
