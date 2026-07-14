from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from infrastructure.data.market_sessions import session_type


NORMALIZER_VERSION = "historical_bar_provider_v1"
ALPACA_API_KEY_ALIASES = ("ALPACA_API_KEY", "APCA_API_KEY_ID")
ALPACA_SECRET_KEY_ALIASES = ("ALPACA_SECRET_KEY", "ALPACA_SECRET", "APCA_API_SECRET_KEY")
_ATOMIC_WRITE_LOCK = threading.Lock()
_MAX_RAW_BATCH_PATH_PART_LENGTH = 120


@dataclass(frozen=True)
class HistoricalBarProviderCapabilities:
    provider: str
    supported_timeframes: tuple[str, ...]
    feed: str
    coverage_note: str
    request_window_limit: str
    max_symbols_per_request: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class HistoricalBarRequest:
    symbols: tuple[str, ...]
    timeframe: str
    start: datetime
    end: datetime
    feed: str
    adjustment: str = "all"
    extended_hours: bool = False
    page_token: str | None = None
    limit: int = 10_000
    raw_chunk_id: str | None = None
    canonical_symbols: tuple[str, ...] | None = None
    provider_symbol_by_canonical: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class HistoricalBarPage:
    bars: list[dict[str, Any]]
    next_page_token: str | None
    raw_payload: dict[str, Any]
    latency_seconds: float


@dataclass
class HistoricalBarMetrics:
    requests_attempted: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    requests_retried: int = 0
    http_429_count: int = 0
    pages_downloaded: int = 0
    rows_downloaded: int = 0
    throttle_sleep_time_seconds: float = 0.0
    response_latency_seconds: list[float] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)
    configured_requests_per_minute: int | None = None
    current_requests_per_minute: int | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = max(0.000001, time.monotonic() - self.started_monotonic)
        return {
            "requests_attempted": self.requests_attempted,
            "requests_successful": self.requests_successful,
            "requests_failed": self.requests_failed,
            "requests_retried": self.requests_retried,
            "http_429_count": self.http_429_count,
            "effective_requests_per_minute": self.requests_attempted / elapsed * 60.0,
            "configured_requests_per_minute": self.configured_requests_per_minute,
            "current_requests_per_minute": self.current_requests_per_minute,
            "average_response_latency": (
                sum(self.response_latency_seconds) / len(self.response_latency_seconds)
                if self.response_latency_seconds
                else None
            ),
            "pages_downloaded": self.pages_downloaded,
            "rows_downloaded": self.rows_downloaded,
            "rows_per_request": (
                self.rows_downloaded / self.requests_attempted
                if self.requests_attempted
                else 0.0
            ),
            "throttle_sleep_time_seconds": self.throttle_sleep_time_seconds,
        }


@dataclass(frozen=True)
class ResolvedAlpacaCredentials:
    api_key: str
    secret_key: str
    credentials_available: bool
    credential_source: str
    api_key_alias_used: str | None = None
    secret_key_alias_used: str | None = None

    def public_report(self) -> dict[str, Any]:
        return {
            "credential_source": self.credential_source,
            "credentials_available": self.credentials_available,
            "api_key_alias_used": self.api_key_alias_used,
            "secret_key_alias_used": self.secret_key_alias_used,
        }


def resolve_alpaca_credentials(
    config: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedAlpacaCredentials:
    """Resolve Alpaca credentials using the repository's existing aliases.

    Values are intentionally not exposed by the returned public report.
    """
    env = os.environ if environ is None else environ
    alpaca_config = dict((config or {}).get("alpaca", {}) or {})
    if alpaca_config.get("api_key") and alpaca_config.get("secret_key"):
        return ResolvedAlpacaCredentials(
            str(alpaca_config["api_key"]),
            str(alpaca_config["secret_key"]),
            True,
            "config.alpaca",
        )
    api_key = ""
    api_alias = None
    for alias in ALPACA_API_KEY_ALIASES:
        if env.get(alias):
            api_key = str(env[alias])
            api_alias = alias
            break
    secret_key = ""
    secret_alias = None
    for alias in ALPACA_SECRET_KEY_ALIASES:
        if env.get(alias):
            secret_key = str(env[alias])
            secret_alias = alias
            break
    return ResolvedAlpacaCredentials(
        api_key,
        secret_key,
        bool(api_key and secret_key),
        "environment" if api_key or secret_key else "absent",
        api_alias,
        secret_alias,
    )


class HistoricalBarProvider(Protocol):
    name: str

    def capabilities(self) -> HistoricalBarProviderCapabilities:
        ...

    def check_authentication(self) -> dict[str, Any]:
        ...

    def fetch_page(self, request: HistoricalBarRequest) -> HistoricalBarPage:
        ...


class SharedRateLimiter:
    """Global request limiter shared by all workers for a provider account."""

    def __init__(
        self,
        *,
        requests_per_minute: int = 180,
        max_in_flight_requests: int = 4,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        if max_in_flight_requests < 1:
            raise ValueError("max_in_flight_requests must be positive")
        self.requests_per_minute = int(requests_per_minute)
        self.configured_requests_per_minute = int(requests_per_minute)
        self.max_in_flight_requests = int(max_in_flight_requests)
        self._sleeper = sleeper
        self._clock = clock
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max_in_flight_requests)
        self._request_times: list[float] = []
        self._next_allowed_request_time = 0.0
        self.sleep_time_seconds = 0.0

    def acquire(self) -> None:
        self._semaphore.acquire()
        with self._lock:
            now = self._clock()
            window_start = now - 60.0
            self._request_times = [value for value in self._request_times if value > window_start]
            sleep_for_window = 0.0
            if len(self._request_times) >= self.requests_per_minute:
                sleep_for_window = max(0.0, 60.0 - (now - self._request_times[0]))
            min_interval_seconds = 60.0 / float(self.requests_per_minute)
            sleep_for_pacing = max(0.0, self._next_allowed_request_time - now)
            sleep_for = max(sleep_for_window, sleep_for_pacing)
            if sleep_for:
                self.sleep_time_seconds += sleep_for
                self._sleeper(sleep_for)
                now = self._clock()
                self._request_times = [
                    value for value in self._request_times if value > now - 60.0
                ]
            self._request_times.append(now)
            self._next_allowed_request_time = now + min_interval_seconds

    def release(self) -> None:
        self._semaphore.release()

    def reduce_pressure_after_429(self) -> None:
        with self._lock:
            self.requests_per_minute = max(1, int(self.requests_per_minute * 0.75))


class AlpacaBasicHistoricalBarProvider:
    name = "alpaca"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: str = "iex",
        rate_limiter: SharedRateLimiter | None = None,
        opener: Callable = urlopen,
        timeout_seconds: float = 30.0,
        metrics: HistoricalBarMetrics | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        resolved = resolve_alpaca_credentials(config)
        self.api_key = api_key if api_key is not None else resolved.api_key
        self.secret_key = secret_key if secret_key is not None else resolved.secret_key
        self.credentials = (
            ResolvedAlpacaCredentials(
                self.api_key,
                self.secret_key,
                bool(self.api_key and self.secret_key),
                "explicit",
            )
            if api_key is not None or secret_key is not None
            else resolved
        )
        self.feed = feed
        self.rate_limiter = rate_limiter or SharedRateLimiter(
            requests_per_minute=180,
            max_in_flight_requests=4,
        )
        self._opener = opener
        self._timeout_seconds = timeout_seconds
        self.metrics = metrics or HistoricalBarMetrics()
        self.metrics.configured_requests_per_minute = self.rate_limiter.configured_requests_per_minute
        self.metrics.current_requests_per_minute = self.rate_limiter.requests_per_minute
        self._metrics_lock = threading.Lock()

    def capabilities(self) -> HistoricalBarProviderCapabilities:
        return HistoricalBarProviderCapabilities(
            provider=self.name,
            supported_timeframes=("1Min", "5Min", "15Min", "1Hour", "1Day", "5m", "1h"),
            feed=self.feed,
            coverage_note=(
                "Alpaca Basic/free should be probed live for actual account depth; "
                "free stock feed is configured as IEX, not paid SIP."
            ),
            request_window_limit="bounded by configured date windows and page limit",
            max_symbols_per_request=50,
            provenance={
                "free_access_only": True,
                "default_free_feed": "iex",
                "sip_historical_status": "probe_required_before_operational_use",
                "auth_env": list(ALPACA_API_KEY_ALIASES + ALPACA_SECRET_KEY_ALIASES),
            },
        )

    def check_authentication(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "feed": self.feed,
            **self.credentials.public_report(),
            "can_attempt_authenticated_request": bool(self.api_key and self.secret_key),
        }

    def fetch_page(self, request: HistoricalBarRequest) -> HistoricalBarPage:
        params = {
            "symbols": ",".join(request.symbols),
            "timeframe": _alpaca_timeframe(request.timeframe),
            "start": _utc_z(request.start),
            "end": _utc_z(request.end),
            "feed": request.feed or self.feed,
            "adjustment": request.adjustment,
            "limit": int(request.limit),
            "sort": "asc",
        }
        if request.page_token:
            params["page_token"] = request.page_token
        url = "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params)
        http_request = Request(
            url,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
            },
        )
        self.rate_limiter.acquire()
        with self._metrics_lock:
            self.metrics.requests_attempted += 1
        started = time.monotonic()
        try:
            with self._opener(http_request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            with self._metrics_lock:
                self.metrics.requests_failed += 1
            if exc.code == 429:
                self.metrics.http_429_count += 1
            if exc.code == 429:
                self.rate_limiter.reduce_pressure_after_429()
                self.metrics.current_requests_per_minute = self.rate_limiter.requests_per_minute
            raise AlpacaHistoricalBarError.from_http_error(exc) from exc
        finally:
            self.rate_limiter.release()
        latency = time.monotonic() - started
        bars = _normalize_alpaca_bars(payload, request)
        with self._metrics_lock:
            self.metrics.requests_successful += 1
            self.metrics.response_latency_seconds.append(latency)
            self.metrics.pages_downloaded += 1
            self.metrics.rows_downloaded += len(bars)
            self.metrics.throttle_sleep_time_seconds = self.rate_limiter.sleep_time_seconds
            self.metrics.current_requests_per_minute = self.rate_limiter.requests_per_minute
        return HistoricalBarPage(
            bars=bars,
            next_page_token=payload.get("next_page_token"),
            raw_payload=payload,
            latency_seconds=latency,
        )


class AlpacaHistoricalBarError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        response_body: str | None = None,
        response_json: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.response_body = response_body
        self.response_json = dict(response_json or {})

    @property
    def classification(self) -> str:
        if self.status_code == 401:
            return "permanent_authentication_failure"
        if self.status_code == 403:
            return "entitlement_failure"
        if self.status_code == 429:
            return "retryable_failure"
        if self.status_code is not None and 500 <= self.status_code <= 599:
            return "retryable_failure"
        return "validation_failure"

    @property
    def retryable(self) -> bool:
        return self.classification == "retryable_failure"

    @classmethod
    def from_http_error(cls, error: HTTPError) -> "AlpacaHistoricalBarError":
        details = error.read().decode("utf-8", errors="replace")
        response_json = None
        try:
            parsed = json.loads(details)
            if isinstance(parsed, dict):
                response_json = parsed
        except json.JSONDecodeError:
            response_json = None
        retry_after = None
        raw_retry_after = error.headers.get("Retry-After") if error.headers else None
        if raw_retry_after:
            try:
                retry_after = float(raw_retry_after)
            except ValueError:
                retry_after = None
        return cls(
            f"Alpaca historical bars request failed ({error.code}): {details}",
            status_code=error.code,
            retry_after=retry_after,
            response_body=details,
            response_json=response_json,
        )


class BackfillChunkStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"completed_chunks": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def mark_completed(self, chunk_id: str, metadata: Mapping[str, Any]) -> None:
        state = self.load()
        completed = list(state.get("completed_chunks", []))
        if not any(row.get("chunk_id") == chunk_id for row in completed):
            completed.append({"chunk_id": chunk_id, **dict(metadata)})
        state["completed_chunks"] = completed
        _atomic_write_text(self.path, json.dumps(state, indent=2, default=str))

    def is_completed(self, chunk_id: str) -> bool:
        return any(row.get("chunk_id") == chunk_id for row in self.load().get("completed_chunks", []))


class CollectionManifest:
    VALID_STATUSES = {
        "planned",
        "in_progress",
        "completed",
        "empty_valid_response",
        "retryable_failure",
        "permanent_authentication_failure",
        "entitlement_failure",
        "validation_failure",
    }

    PRESERVED_INITIALIZE_STATUSES = VALID_STATUSES - {"planned"}

    def __init__(
        self,
        path: str | Path,
        *,
        writer: Callable[[Path, str], None] | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path = Path(journal_path) if journal_path is not None else self.path.with_suffix(self.path.suffix + ".events.jsonl")
        self._writer = writer or _atomic_write_text
        self._lock = threading.Lock()
        self._state_cache: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        with self._lock:
            state = self._load_unlocked()
            return {**state, "chunks": dict(state.get("chunks", {}))}

    def update(self, chunk_id: str, status: str, metadata: Mapping[str, Any] | None = None) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"unsupported chunk status: {status}")
        event = {
            **dict(metadata or {}),
            "chunk_id": chunk_id,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            state = self._load_unlocked()
            chunks = dict(state.get("chunks", {}))
            chunks[chunk_id] = {
                **dict(chunks.get(chunk_id, {})),
                **event,
            }
            state["chunks"] = chunks
            self._append_event_unlocked(event)
            self._state_cache = state

    def initialize_plan(
        self,
        requests: Sequence[HistoricalBarRequest],
        *,
        dry_run: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._load_unlocked()
            chunks = dict(state.get("chunks", {}))
            existing = len(chunks)
            added = 0
            updated_planned = 0
            preserved = 0
            now = datetime.now(timezone.utc).isoformat()
            planned_metadata = dict(metadata or {})
            planned_metadata["dry_run"] = dry_run
            for request in requests:
                chunk_id = request.raw_chunk_id or _chunk_id(request)
                current = dict(chunks.get(chunk_id, {}))
                status = current.get("status")
                if status in self.PRESERVED_INITIALIZE_STATUSES:
                    preserved += 1
                    continue
                planned = {
                    **current,
                    **planned_metadata,
                    "chunk_id": chunk_id,
                    "status": "planned",
                    "updated_at": current.get("updated_at") or now,
                }
                if not current:
                    added += 1
                elif planned != current:
                    updated_planned += 1
                chunks[chunk_id] = planned
            state["chunks"] = _ordered_chunks(chunks)
            changed = added > 0 or updated_planned > 0 or len(chunks) != existing
            if changed:
                self._writer(self.path, json.dumps(state, indent=2, default=str))
                self._truncate_journal_unlocked()
                self._state_cache = state
            return {
                "existing_chunk_count": existing,
                "final_chunk_count": len(chunks),
                "added_planned_count": added,
                "updated_planned_count": updated_planned,
                "preserved_existing_state_count": preserved,
                "write_performed": changed,
            }

    def status(self, chunk_id: str) -> str | None:
        return self.load().get("chunks", {}).get(chunk_id, {}).get("status")

    def is_completed(self, chunk_id: str) -> bool:
        return self.status(chunk_id) in {"completed", "empty_valid_response"}

    def checkpoint(self) -> None:
        with self._lock:
            state = self._load_unlocked()
            state["chunks"] = _ordered_chunks(dict(state.get("chunks", {})))
            self._writer(self.path, json.dumps(state, indent=2, default=str))
            self._truncate_journal_unlocked()
            self._state_cache = state

    def _load_unlocked(self) -> dict[str, Any]:
        if self._state_cache is not None:
            return self._state_cache
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            state = {"chunks": {}}
        chunks = dict(state.get("chunks", {}))
        if self.journal_path.exists():
            journal_text = self.journal_path.read_text(encoding="utf-8")
            journal_lines = journal_text.splitlines()
            journal_ended_cleanly = journal_text.endswith("\n") or journal_text == ""
            for index, line in enumerate(journal_lines, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(journal_lines) and not journal_ended_cleanly:
                        continue
                    raise
                chunk_id = str(event.get("chunk_id", ""))
                status = str(event.get("status", ""))
                if not chunk_id or status not in self.VALID_STATUSES:
                    continue
                chunks[chunk_id] = {**dict(chunks.get(chunk_id, {})), **event}
        state["chunks"] = chunks
        self._state_cache = state
        return state

    def _append_event_unlocked(self, event: Mapping[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(event), default=str, separators=(",", ":")) + "\n"
        with open(self.journal_path, "ab") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    def _truncate_journal_unlocked(self) -> None:
        if self.journal_path.exists():
            _atomic_write_text(self.journal_path, "")


class ImmutableRawChunkStore:
    def __init__(self, root: str | Path = "data/raw/alpaca/stock_bars") -> None:
        self.root = Path(root)

    def chunk_dir(self, request: HistoricalBarRequest) -> Path:
        batch = _raw_batch_path_part(request.symbols)
        window = _safe_path_part(f"{request.start:%Y%m%dT%H%M%SZ}_{request.end:%Y%m%dT%H%M%SZ}")
        return self.root / request.feed / request.timeframe / batch / window

    def write_completed_chunk(
        self,
        request: HistoricalBarRequest,
        *,
        rows: Sequence[Mapping[str, Any]],
        raw_pages: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
        force_refresh: bool = False,
        raw_write: bool = True,
    ) -> dict[str, Any]:
        target = self.chunk_dir(request)
        manifest_path = target / "manifest.json"
        if manifest_path.exists() and not force_refresh:
            raise FileExistsError(f"completed raw chunk already exists: {manifest_path}")
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            _remove_tree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        if raw_write:
            _atomic_write_text(tmp / "provider_pages.json", json.dumps(list(raw_pages), indent=2, default=str))
        _atomic_write_text(tmp / "normalized_rows.json", json.dumps(list(rows), indent=2, default=str))
        timestamps = [row["timestamp"] for row in rows if isinstance(row.get("timestamp"), datetime)]
        manifest = {
            "provider": "alpaca",
            "feed": request.feed,
            "timeframe_requested": request.timeframe,
            "native_timeframe": _alpaca_timeframe(request.timeframe),
            "symbol_batch": list(request.symbols),
            "symbol_batch_path": _raw_batch_path_part(request.symbols),
            "symbol_batch_path_schema": "plain-or-hashed-v2",
            "canonical_symbol_batch": list(request.canonical_symbols or request.symbols),
            "provider_symbol_map": dict(request.provider_symbol_by_canonical),
            "requested_start": request.start.isoformat(),
            "requested_end": request.end.isoformat(),
            "actual_earliest_timestamp": min(timestamps).isoformat() if timestamps else None,
            "actual_latest_timestamp": max(timestamps).isoformat() if timestamps else None,
            "row_count": len(rows),
            "page_count": int(metrics.get("pages", 0)),
            "pagination_stop_reason": "next_page_token_absent",
            "request_count": int(metrics.get("requests_attempted", 0)),
            "retry_count": int(metrics.get("requests_retried", 0)),
            "http_429_count": int(metrics.get("http_429_count", 0)),
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "adjustment_mode": request.adjustment,
            "session_policy": "regular_session_default" if not request.extended_hours else "extended_hours_requested",
            "normalizer_version": NORMALIZER_VERSION,
            "completion_state": "completed" if rows else "empty_valid_response",
        }
        _atomic_write_text(tmp / "manifest.json", json.dumps(manifest, indent=2, default=str))
        if target.exists() and force_refresh:
            _remove_tree(target)
        tmp.replace(target)
        return manifest

    def read_completed_chunk(self, request: HistoricalBarRequest) -> list[dict[str, Any]]:
        path = self.chunk_dir(request) / "normalized_rows.json"
        if not path.exists():
            raise FileNotFoundError(f"completed raw chunk rows not found: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        output = []
        for raw in rows:
            row = dict(raw)
            timestamp = _parse_stored_datetime(row.get("timestamp"))
            if timestamp is not None:
                row["timestamp"] = timestamp
                row["session_type"] = session_type(timestamp)
            output.append(row)
        return output


def fetch_chunk_with_retries(
    provider: HistoricalBarProvider,
    request: HistoricalBarRequest,
    *,
    state_store: BackfillChunkStateStore | None = None,
    max_retries: int = 3,
    base_backoff_seconds: float = 1.0,
    jitter_seconds: float = 0.25,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_id = request.raw_chunk_id or _chunk_id(request)
    if state_store and state_store.is_completed(chunk_id):
        return [], {"chunk_id": chunk_id, "skipped_completed": True}
    rows: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    next_token = request.page_token
    pages = 0
    while True:
        page_request = HistoricalBarRequest(
            symbols=request.symbols,
            timeframe=request.timeframe,
            start=request.start,
            end=request.end,
            feed=request.feed,
            adjustment=request.adjustment,
            extended_hours=request.extended_hours,
            page_token=next_token,
            limit=request.limit,
            raw_chunk_id=chunk_id,
            canonical_symbols=request.canonical_symbols,
            provider_symbol_by_canonical=request.provider_symbol_by_canonical,
        )
        attempt = 0
        while True:
            try:
                page = provider.fetch_page(page_request)
                break
            except AlpacaHistoricalBarError as exc:
                if not exc.retryable:
                    raise
                attempt += 1
                metrics = getattr(provider, "metrics", None)
                if metrics is not None:
                    _increment_metric(metrics, "requests_retried", owner=provider)
                if attempt > max_retries:
                    raise
                sleep_for = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else base_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, jitter_seconds)
                )
                sleeper(float(sleep_for))
        rows.extend(page.bars)
        raw_pages.append(page.raw_payload)
        pages += 1
        next_token = page.next_page_token
        if not next_token:
            break
    if state_store:
        state_store.mark_completed(
            chunk_id,
            {
                "provider": provider.name,
                "symbols": list(request.symbols),
                "timeframe": request.timeframe,
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "rows": len(rows),
                "pages": pages,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return rows, {
        "chunk_id": chunk_id,
        "pages": pages,
        "rows": len(rows),
        "skipped_completed": False,
        "raw_pages": raw_pages,
    }


def free_historical_bar_source_inventory() -> dict[str, Any]:
    return {
        "implemented_operational_free_sources": [
            {
                "provider": "alpaca",
                "feed": "iex",
                "timeframes": ["5m", "1h", "1Day"],
                "use": "primary free historical stock-bar probe/backfill source",
                "requires_payment": False,
                "notes": "Actual historical depth must be measured by authenticated live probe.",
            },
            {
                "provider": "stooq",
                "feed": "daily_csv",
                "timeframes": ["1Day"],
                "use": "daily overlap audit source only",
                "requires_payment": False,
                "notes": "Existing repo adapter supports daily CSV; do not use as 5m operational substitute.",
            },
            {
                "provider": "stooq_bulk_parquet",
                "feed": "local_bulk_import",
                "timeframes": ["1Day"],
                "use": "local daily overlap/inventory source when already imported",
                "requires_payment": False,
                "notes": "Bulk download/import remains explicit and separate from canonical market data.",
            },
        ],
        "rejected_for_immediate_free_implementation": [
            {"provider": "alpaca_sip", "reason": "not enabled as operational source until historical entitlement probe succeeds"},
            {"provider": "massive_polygon", "reason": "meaningful historical stock bars require paid access"},
            {"provider": "tiingo", "reason": "meaningful historical API use requires paid or trial-gated access"},
            {"provider": "alpha_vantage", "reason": "intraday/premium endpoints and rate limits unsuitable for this backfill"},
            {"provider": "twelve_data", "reason": "meaningful historical intraday coverage requires paid/trial-gated access"},
            {"provider": "firstrate_data", "reason": "archives are paid datasets"},
            {"provider": "databento", "reason": "US equity historical datasets are paid/trial-gated"},
        ],
        "future_free_extension_points": [
            {
                "provider": "stooq_intraday_downloads",
                "status": "not implemented",
                "reason": "public pages advertise intraday downloads, but collection needs separate legal/provenance and automation review.",
            }
        ],
    }


def _normalize_alpaca_bars(payload: Mapping[str, Any], request: HistoricalBarRequest) -> list[dict[str, Any]]:
    collected_at = datetime.now(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []
    bars_by_symbol = payload.get("bars", {}) or {}
    for symbol, bars in bars_by_symbol.items():
        canonical_symbol = _canonical_symbol_for_provider(request, str(symbol).upper())
        for bar in bars or []:
            timestamp = datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00"))
            row = {
                "symbol": canonical_symbol,
                "timestamp": timestamp,
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar.get("v", 0) or 0),
                "trade_count": bar.get("n"),
                "vwap": bar.get("vw"),
                "provider": "alpaca",
                "feed": request.feed,
                "collection_timestamp": collected_at,
                "requested_timeframe": request.timeframe,
                "native_timeframe": _alpaca_timeframe(request.timeframe),
                "adjustment_mode": request.adjustment,
                "extended_hours": request.extended_hours,
                "session_policy": "all_returned_bars_preserved",
                "session_type": session_type(timestamp),
                "raw_chunk_identifier": request.raw_chunk_id or _chunk_id(request),
                "normalizer_version": NORMALIZER_VERSION,
            }
            if canonical_symbol != str(symbol).upper():
                row["provider_symbol"] = str(symbol).upper()
            output.append(row)
    output.sort(key=lambda row: (row["symbol"], row["timestamp"]))
    return output


def _alpaca_timeframe(timeframe: str) -> str:
    aliases = {"5m": "5Min", "1h": "1Hour", "1Day": "1Day"}
    return aliases.get(timeframe, timeframe)


def _canonical_symbol_for_provider(request: HistoricalBarRequest, provider_symbol: str) -> str:
    reverse = {provider.upper(): canonical.upper() for canonical, provider in request.provider_symbol_by_canonical}
    return reverse.get(provider_symbol.upper(), provider_symbol.upper())


def _utc_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk_id(request: HistoricalBarRequest) -> str:
    return "|".join(
        [
            request.feed,
            request.timeframe,
            ",".join(request.symbols),
            _utc_z(request.start),
            _utc_z(request.end),
            request.adjustment,
            "extended" if request.extended_hours else "regular",
        ]
    )


def _increment_metric(metrics: Any, name: str, *, owner: Any | None = None) -> None:
    lock = getattr(owner, "_metrics_lock", None)
    if lock is None:
        setattr(metrics, name, getattr(metrics, name) + 1)
        return
    with lock:
        setattr(metrics, name, getattr(metrics, name) + 1)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    max_attempts = 5
    base_sleep_seconds = 0.025
    last_error: PermissionError | None = None
    with _ATOMIC_WRITE_LOCK:
        for attempt in range(max_attempts):
            tmp = path.parent / (
                f".{path.name}.{os.getpid()}.{threading.get_ident()}."
                f"{time.time_ns()}.{attempt}.tmp"
            )
            try:
                with open(tmp, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_error = exc
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                if attempt == max_attempts - 1:
                    break
                time.sleep(base_sleep_seconds * (2 ** attempt) + random.uniform(0.0, base_sleep_seconds))
            except Exception:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                raise
    if last_error is not None:
        raise last_error


def _ordered_chunks(chunks: Mapping[str, Any]) -> dict[str, Any]:
    return {key: chunks[key] for key in sorted(chunks)}


def _parse_stored_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _raw_batch_path_part(symbols: Sequence[str]) -> str:
    plain = _safe_path_part("-".join(symbols))
    if len(plain) <= _MAX_RAW_BATCH_PATH_PART_LENGTH:
        return plain
    digest = hashlib.sha256(plain.encode("utf-8")).hexdigest()[:16]
    first = _safe_path_part(str(symbols[0]))[:16] if symbols else "empty"
    last = _safe_path_part(str(symbols[-1]))[:16] if symbols else "empty"
    return f"batch-{len(symbols)}-{first}-{last}-{digest}"
