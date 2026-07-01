from core.research.ml.stock_level.run_manifest import service as _service

globals().update({name: value for name, value in vars(_service).items() if not name.startswith("__")})
