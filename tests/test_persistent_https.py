from __future__ import annotations

import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
from urllib.request import Request

import pytest

from infrastructure.data.persistent_https import (
    BufferedHttpResponse,
    PersistentHttpsSession,
    WorkerLocalHttpsSessionPool,
)


class _Response:
    status = 200
    reason = "OK"
    headers: dict[str, str] = {}

    def __init__(self, body: bytes = b"{}") -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body


class _Connection:
    def __init__(self, responses: list[object], **_kwargs: object) -> None:
        self.responses = responses
        self.requests = 0
        self.closed = False
        self.timeout = 30.0

    def request(self, *_args: object, **_kwargs: object) -> None:
        self.requests += 1

    def getresponse(self) -> _Response:
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, _Response)
        return result

    def close(self) -> None:
        self.closed = True


def test_persistent_session_reuses_successful_connection() -> None:
    connections: list[_Connection] = []

    def factory(*_args: object, **kwargs: object) -> _Connection:
        connection = _Connection([_Response(b"one"), _Response(b"two")], **kwargs)
        connections.append(connection)
        return connection

    session = PersistentHttpsSession(connection_factory=factory)
    request = Request("https://data.alpaca.markets/v2/stocks/bars")
    with session.open(request, timeout=5) as response:
        assert response.read() == b"one"
    with session.open(request, timeout=5) as response:
        assert response.read() == b"two"

    assert len(connections) == 1
    assert connections[0].requests == 2
    assert session.connection_generation == 1


def test_tls_failure_invalidates_connection_and_retry_gets_fresh_connection() -> None:
    connections: list[_Connection] = []
    outcomes = iter(([ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")], [_Response(b"recovered")]))

    def factory(*_args: object, **kwargs: object) -> _Connection:
        connection = _Connection(list(next(outcomes)), **kwargs)
        connections.append(connection)
        return connection

    session = PersistentHttpsSession(connection_factory=factory)
    request = Request("https://data.alpaca.markets/v2/stocks/bars")
    with pytest.raises(URLError, match="UNEXPECTED_EOF"):
        session.open(request, timeout=5)
    with session.open(request, timeout=5) as response:
        assert response.read() == b"recovered"

    assert len(connections) == 2
    assert connections[0].closed
    assert session.connection_generation == 2


def test_persistent_session_rejects_cross_thread_use() -> None:
    session = PersistentHttpsSession(
        connection_factory=lambda *_args, **kwargs: _Connection([_Response(), _Response()], **kwargs)
    )
    request = Request("https://data.alpaca.markets/v2/stocks/bars")
    session.open(request, timeout=5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        error = executor.submit(session.open, request, timeout=5).exception()
    assert isinstance(error, RuntimeError)
    assert "cannot be shared" in str(error)


def test_worker_pool_owns_independent_sessions_and_closes_all() -> None:
    barrier = threading.Barrier(2)

    class FakeSession:
        def __init__(self) -> None:
            self.owner = threading.get_ident()
            self.closed = False
            self.connection_generation = 1

        def open(self, _request: Request, *, timeout: float) -> BufferedHttpResponse:
            assert timeout == 5
            assert self.owner == threading.get_ident()
            barrier.wait(timeout=2)
            return BufferedHttpResponse(str(self.owner).encode())

        def close(self) -> None:
            self.closed = True

    sessions: list[FakeSession] = []

    def factory() -> FakeSession:
        session = FakeSession()
        sessions.append(session)
        return session

    pool = WorkerLocalHttpsSessionPool(session_factory=factory)  # type: ignore[arg-type]
    request = Request("https://data.alpaca.markets/v2/stocks/bars")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pool.open, request, timeout=5) for _ in range(2)]
        owners = {future.result().read() for future in futures}

    assert len(owners) == 2
    assert pool.session_count == 2
    pool.close()
    assert all(session.closed for session in sessions)


def test_one_failed_worker_session_does_not_poison_other_worker() -> None:
    barrier = threading.Barrier(2)
    outcomes = iter(("fail", "success"))

    class FakeSession:
        connection_generation = 1

        def __init__(self, outcome: str) -> None:
            self.outcome = outcome

        def open(self, _request: Request, *, timeout: float) -> BufferedHttpResponse:
            barrier.wait(timeout=2)
            if self.outcome == "fail":
                raise URLError("TLS EOF")
            return BufferedHttpResponse(b"healthy")

        def close(self) -> None:
            pass

    pool = WorkerLocalHttpsSessionPool(
        session_factory=lambda: FakeSession(next(outcomes))  # type: ignore[arg-type]
    )
    request = Request("https://data.alpaca.markets/v2/stocks/bars")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pool.open, request, timeout=5) for _ in range(2)]
        results = []
        for future in futures:
            try:
                results.append(future.result().read())
            except URLError:
                results.append(b"failed")
    assert sorted(results) == [b"failed", b"healthy"]
