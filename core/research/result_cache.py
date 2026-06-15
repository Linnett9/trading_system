import hashlib
import json
from pathlib import Path


class ResultCache:

    def __init__(
        self,
        cache_dir: str = "cache/results",
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key_parts: dict):
        if not self.enabled:
            return None

        path = self._path(key_parts)
        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key_parts: dict, value: dict):
        if not self.enabled:
            return

        path = self._path(key_parts)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _path(self, key_parts: dict) -> Path:
        payload = json.dumps(
            key_parts,
            default=str,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"
