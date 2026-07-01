from core.research.ml.stock_level.prediction_artifacts import targets as _targets

globals().update({name: value for name, value in vars(_targets).items() if not name.startswith("__")})
