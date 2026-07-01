from core.research.ml.stock_level.run_manifest import paths as _paths

globals().update({name: value for name, value in vars(_paths).items() if not name.startswith("__")})
