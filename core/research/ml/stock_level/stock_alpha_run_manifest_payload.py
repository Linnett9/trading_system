from core.research.ml.stock_level.run_manifest import payload as _payload

globals().update({name: value for name, value in vars(_payload).items() if not name.startswith("__")})
