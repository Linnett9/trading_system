from core.research.ml.stock_level.prediction_artifacts import sources as _sources

globals().update({name: value for name, value in vars(_sources).items() if not name.startswith("__")})
