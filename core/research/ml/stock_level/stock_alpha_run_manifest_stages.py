from core.research.ml.stock_level.run_manifest import stages as _stages

globals().update({name: value for name, value in vars(_stages).items() if not name.startswith("__")})
