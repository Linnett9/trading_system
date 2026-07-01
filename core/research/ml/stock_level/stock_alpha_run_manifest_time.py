from core.research.ml.stock_level.run_manifest import time as _time

globals().update({name: value for name, value in vars(_time).items() if not name.startswith("__")})
