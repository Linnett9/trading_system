from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.vast_reverse_queue_r1 import main


if __name__ == "__main__":
    raise SystemExit(main())
