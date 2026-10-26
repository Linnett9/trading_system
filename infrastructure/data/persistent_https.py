"""Worker-local persistent HTTPS transport with fail-closed session invalidation."""

from __future__ import annotations

import http.client
import io
import ssl
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request


class BufferedHttpResponse:
    """Minimal context-managed response contract consumed by provider adapters."""

    def __init__(self, body: bytes) -> None:
        self._body = io.BytesIO(body)

    def read(self) -> bytes:
        return self._body.read()

    def __enter__(self) -> "BufferedHttpResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._body.close()


class PersistentHttpsSession:
    """Own one keep-alive HTTPS connection on exactly one worker thread."""

    def __init__(
        self,
        *,
        connection_factory: Callable[..., http.client.HTTPSConnection] = http.client.HTTPSConnection,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._ssl_context = ssl_context or ssl.create_default_context()
        self._connection: http.client.HTTPSConnection | None = None
        self._origin: tuple[str, int] | None = None
        self._owner_thread_id: int | None = None
        self.connection_generation = 0
        self.closed = False

    def open(self, request: Request, *, timeout: float) -> BufferedHttpResponse:
        self._assert_owner()
        if self.closed:
            raise RuntimeError("Persistent HTTPS session is closed")
        parsed = urlsplit(request.full_url)
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            raise ValueError(f"Persistent session requires an HTTPS URL: {request.full_url}")
        port = parsed.port or 443
        origin = (parsed.hostname, port)
        if self._origin is not None and self._origin != origin:
            raise ValueError("Persistent HTTPS session cannot cross origins")
        connection = self._connection_for(origin, timeout)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        try:
            connection.request(request.get_method(), path, headers=dict(request.header_items()))
            response = connection.getresponse()
            body = response.read()
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            self.invalidate()
            raise URLError(f"persistent HTTPS transport failure: {exc}") from exc
        if response.status >= 400:
            raise HTTPError(
                request.full_url,
                response.status,
                response.reason,
                response.headers,
                io.BytesIO(body),
            )
        return BufferedHttpResponse(body)

    def invalidate(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def close(self) -> None:
        self.invalidate()
        self.closed = True

    def _connection_for(
        self,
        origin: tuple[str, int],
        timeout: float,
    ) -> http.client.HTTPSConnection:
        if self._connection is None:
            self._origin = origin
            self._connection = self._connection_factory(
                origin[0],
                origin[1],
                timeout=timeout,
                context=self._ssl_context,
            )
            self.connection_generation += 1
        else:
            self._connection.timeout = timeout
        return self._connection

    def _assert_owner(self) -> None:
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif self._owner_thread_id != thread_id:
            raise RuntimeError("Persistent HTTPS session cannot be shared across worker threads")


class WorkerLocalHttpsSessionPool:
    """Lazily create and close one persistent HTTPS session per worker thread."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], PersistentHttpsSession] = PersistentHttpsSession,
    ) -> None:
        self._session_factory = session_factory
        self._local = threading.local()
        self._sessions: list[PersistentHttpsSession] = []
        self._lock = threading.Lock()
        self._closed = False

    def open(self, request: Request, *, timeout: float) -> BufferedHttpResponse:
        if self._closed:
            raise RuntimeError("Worker-local HTTPS session pool is closed")
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._session_factory()
            self._local.session = session
            with self._lock:
                self._sessions.append(session)
        return session.open(request, timeout=timeout)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    @property
    def connection_generations(self) -> int:
        with self._lock:
            return sum(session.connection_generation for session in self._sessions)

    def close(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions)
            self._closed = True
        for session in sessions:
            session.close()

    def __enter__(self) -> "WorkerLocalHttpsSessionPool":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = ["BufferedHttpResponse", "PersistentHttpsSession", "WorkerLocalHttpsSessionPool"]
