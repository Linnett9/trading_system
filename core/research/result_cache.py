import hashlib
import json
from json import JSONDecodeError
from pathlib import Path


class ResultCache:

    def __init__(
        self,
        cache_dir: str = "cache/results",
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.disabled_reason = ""

        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                self._disable(f"cache directory unavailable: {error}")

    def get(self, key_parts: dict):
        if not self.enabled:
            return None

        path = self._path(key_parts)
        try:
            if not path.exists():
                return None

            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError):
            return None

    def set(self, key_parts: dict, value: dict):
        if not self.enabled:
            return

        path = self._path(key_parts)
        try:
            path.write_text(
                json.dumps(value, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
        except OSError as error:
            self._disable(f"cache write failed: {error}")

    def _path(self, key_parts: dict) -> Path:
        payload = json.dumps(
            key_parts,
            default=str,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _disable(self, reason: str):
        self.enabled = False
        self.disabled_reason = reason
