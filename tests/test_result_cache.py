from core.research.result_cache import ResultCache


class FailingCachePath:

    def exists(self):
        return False

    def write_text(self, *_args, **_kwargs):
        raise OSError(28, "No space left on device")


def test_result_cache_disables_itself_when_write_fails(tmp_path):
    cache = ResultCache(cache_dir=str(tmp_path), enabled=True)
    cache._path = lambda _key_parts: FailingCachePath()

    cache.set({"symbol": "AAPL"}, {"total_return": 0.1})

    assert not cache.enabled
    assert "cache write failed" in cache.disabled_reason


def test_disabled_result_cache_returns_none(tmp_path):
    cache = ResultCache(cache_dir=str(tmp_path), enabled=False)

    assert cache.get({"symbol": "AAPL"}) is None
