from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


STANDARD_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "symbol")
logger = logging.getLogger(__name__)
TIMEFRAME_ALIASES = {
    "1day": "1Day",
    "1d": "1Day",
    "d": "1Day",
    "day": "1Day",
    "daily": "1Day",
    "5m": "5m",
    "5min": "5m",
    "5mins": "5m",
    "5minute": "5m",
    "5minutes": "5m",
    "1h": "1h",
    "h": "1h",
    "hour": "1h",
    "hourly": "1h",
    "1hour": "1h",
    "60m": "1h",
    "60min": "1h",
    "60minute": "1h",
}
SUPPORTED_TIMEFRAMES = {"1Day", "5m", "1h"}


@dataclass(frozen=True)
class MarketParquetImportResult:
    symbol: str
    timeframe: str
    source_path: str
    output_path: str
    row_count: int
    first_timestamp: str
    last_timestamp: str
    duplicate_count: int = 0
    skipped_existing: bool = False


class MarketParquetImporter:
    """Normalize local OHLCV files into canonical per-symbol/timeframe Parquet."""

    def __init__(
        self,
        raw_dir: str | Path,
        output_root: str | Path = "data/processed",
        *,
        timezone_name: str = "UTC",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_root = Path(output_root)
        self.source_timezone = ZoneInfo(timezone_name)

    def import_timeframe(
        self,
        timeframe: str,
        *,
        symbols: Iterable[str] | None = None,
        resume: bool = True,
    ) -> list[MarketParquetImportResult]:
        canonical_timeframe = normalize_timeframe(timeframe)
        requested = {str(symbol).upper() for symbol in symbols or []}
        results = []
        scanned_files = 0
        skipped_files = 0
        failed_files = 0
        scanned_symbols = set()
        candidate_files = self._candidate_files(canonical_timeframe)
        raw_files_discovered = len(candidate_files)
        logger.info(
            "Market Parquet %s raw files discovered: %s from %s",
            canonical_timeframe,
            raw_files_discovered,
            self.raw_dir,
        )
        print(
            f"Market Parquet {canonical_timeframe} raw files discovered: "
            f"{raw_files_discovered} from {self.raw_dir}"
        )
        for path in candidate_files:
            symbol = self._symbol_from_path(path)
            if requested and symbol not in requested:
                continue
            scanned_files += 1
            scanned_symbols.add(symbol)
            try:
                result = self.import_file(
                    path,
                    canonical_timeframe,
                    symbol=symbol,
                    resume=resume,
                )
            except Exception as exc:
                failed_files += 1
                logger.exception(
                    "Failed to import market data file %s for %s %s: %s",
                    path,
                    symbol,
                    canonical_timeframe,
                    exc,
                )
                continue
            if result is None:
                skipped_files += 1
                continue
            results.append(result)
        parsed_files = len(results)
        skipped_existing_files = sum(result.skipped_existing for result in results)
        written_files = parsed_files - skipped_existing_files
        logger.info(
            "Market Parquet %s import summary: raw_files_discovered=%s, "
            "total_files_scanned=%s, successfully_parsed_files=%s, "
            "written_files=%s, skipped_existing_files=%s, skipped_files=%s, "
            "failed_files=%s",
            canonical_timeframe,
            raw_files_discovered,
            scanned_files,
            parsed_files,
            written_files,
            skipped_existing_files,
            skipped_files,
            failed_files,
        )
        print(
            f"Market Parquet {canonical_timeframe} import summary: "
            f"raw_files_discovered={raw_files_discovered}, "
            f"total_files_scanned={scanned_files}, "
            f"successfully_parsed_files={parsed_files}, "
            f"written_files={written_files}, "
            f"skipped_existing_files={skipped_existing_files}, "
            f"skipped_files={skipped_files}, failed_files={failed_files}"
        )
        missing = sorted(requested - scanned_symbols)
        if missing:
            raise FileNotFoundError(
                "Missing raw files for symbols: " + ", ".join(missing)
            )
        return results

    def import_file(
        self,
        path: str | Path,
        timeframe: str,
        *,
        symbol: str | None = None,
        resume: bool = True,
    ) -> MarketParquetImportResult | None:
        source_path = Path(path)
        canonical_timeframe = normalize_timeframe(timeframe)
        normalized_symbol = (symbol or self._symbol_from_path(source_path)).upper()
        try:
            rows = self._read_rows(source_path, normalized_symbol, canonical_timeframe)
        except (csv.Error, KeyError, UnicodeDecodeError, ValueError) as exc:
            logger.warning(
                "Skipping market data file %s for %s %s: %s",
                source_path,
                normalized_symbol,
                canonical_timeframe,
                exc,
            )
            return None
        if not rows:
            logger.warning(
                "Skipping market data file %s for %s %s: contains no OHLCV rows",
                source_path,
                normalized_symbol,
                canonical_timeframe,
            )
            return None
        output_path = market_parquet_path(
            self.output_root,
            normalized_symbol,
            canonical_timeframe,
        )
        assert_market_parquet_output_path(
            output_path,
            self.output_root,
            normalized_symbol,
            canonical_timeframe,
        )
        duplicate_count = _duplicate_count(rows)
        if resume and output_path.exists():
            return MarketParquetImportResult(
                symbol=normalized_symbol,
                timeframe=canonical_timeframe,
                source_path=str(source_path),
                output_path=str(output_path),
                row_count=len(rows),
                first_timestamp=rows[0]["timestamp"].isoformat(),
                last_timestamp=rows[-1]["timestamp"].isoformat(),
                duplicate_count=duplicate_count,
                skipped_existing=True,
            )
        try:
            self._validate_rows(normalized_symbol, canonical_timeframe, rows)
        except ValueError as exc:
            logger.warning(
                "Skipping market data file %s for %s %s: %s",
                source_path,
                normalized_symbol,
                canonical_timeframe,
                exc,
            )
            return None
        self._write_parquet(output_path, rows)
        return MarketParquetImportResult(
            symbol=normalized_symbol,
            timeframe=canonical_timeframe,
            source_path=str(source_path),
            output_path=str(output_path),
            row_count=len(rows),
            first_timestamp=rows[0]["timestamp"].isoformat(),
            last_timestamp=rows[-1]["timestamp"].isoformat(),
            duplicate_count=duplicate_count,
        )

    def _candidate_files(self, timeframe: str) -> list[Path]:
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw market data directory not found: {self.raw_dir}")
        canonical_timeframe = normalize_timeframe(timeframe)
        files_by_folder: dict[str, dict[Path, list[Path]]] = {}
        token_matched_files = []
        for path in sorted(self.raw_dir.rglob("*")):
            if path.suffix.lower() not in {".csv", ".txt"}:
                continue
            source_timeframe, source_folder = _detect_source_timeframe(path, self.raw_dir)
            if source_timeframe:
                files_by_folder.setdefault(source_timeframe, {}).setdefault(
                    source_folder or self.raw_dir,
                    [],
                ).append(path)
                continue
            lower_name = path.name.lower()
            lower_parts = [part.lower() for part in path.parts]
            timeframe_tokens = _timeframe_tokens(canonical_timeframe)
            if any(token in lower_name for token in timeframe_tokens) or any(
                token in part for token in timeframe_tokens for part in lower_parts
            ):
                token_matched_files.append(path)

        files = []
        for source_folder, folder_files in sorted(
            files_by_folder.get(canonical_timeframe, {}).items(),
            key=lambda item: str(item[0]).lower(),
        ):
            logger.info(
                "Detected raw folder: %s -> canonical: %s -> files: %s",
                source_folder.name,
                canonical_timeframe,
                len(folder_files),
            )
            files.extend(folder_files)
        if files:
            return sorted(files)
        if token_matched_files:
            logger.info(
                "Detected raw folder: %s -> canonical: %s -> files: %s",
                self.raw_dir.name,
                canonical_timeframe,
                len(token_matched_files),
            )
            return sorted(token_matched_files)
        top_level_files = sorted(
            path
            for path in self.raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".txt"}
        )
        if top_level_files:
            logger.info(
                "Detected raw folder: %s -> canonical: %s -> files: %s",
                self.raw_dir.name,
                canonical_timeframe,
                len(top_level_files),
            )
        return top_level_files

    def _read_rows(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
    ) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "<DATE>" in fieldnames and "<TIME>" in fieldnames:
                rows = [
                    self._parse_stooq_row(symbol, timeframe, row, line_number)
                    for line_number, row in enumerate(reader, start=2)
                ]
            else:
                rows = [
                    self._parse_standard_row(symbol, row, line_number)
                    for line_number, row in enumerate(reader, start=2)
                ]
        rows = [row for row in rows if row is not None]
        return sorted(rows, key=lambda row: row["timestamp"])

    def _parse_standard_row(
        self,
        fallback_symbol: str,
        row: dict[str, str],
        line_number: int,
    ) -> dict[str, Any]:
        lowered = {str(key).strip().lower(): value for key, value in row.items()}
        timestamp_value = (
            lowered.get("timestamp")
            or lowered.get("datetime")
            or lowered.get("date")
            or lowered.get("time")
        )
        if not timestamp_value:
            raise ValueError(f"row {line_number} is missing a timestamp/date column")
        if "date" in lowered and ("time" in lowered or "bar_time" in lowered):
            timestamp_value = f"{lowered['date']} {lowered.get('time') or lowered.get('bar_time')}"
        symbol = str(lowered.get("symbol") or lowered.get("ticker") or fallback_symbol).upper()
        return {
            "timestamp": parse_utc_timestamp(
                str(timestamp_value),
                source_timezone=self.source_timezone,
            ),
            "open": _float_value(lowered, ("open", "o"), line_number),
            "high": _float_value(lowered, ("high", "h"), line_number),
            "low": _float_value(lowered, ("low", "l"), line_number),
            "close": _float_value(lowered, ("close", "c", "adj close"), line_number),
            "volume": _float_value(lowered, ("volume", "vol", "v"), line_number, default=0.0),
            "symbol": symbol,
        }

    def _parse_stooq_row(
        self,
        symbol: str,
        timeframe: str,
        row: dict[str, str],
        line_number: int,
    ) -> dict[str, Any]:
        expected_periods = {"1Day": {"D"}, "5m": {"5"}, "1h": {"60", "H", "1H"}}[
            timeframe
        ]
        period = str(row.get("<PER>", "")).upper()
        if period not in expected_periods:
            raise ValueError(
                "row "
                f"{line_number} has period {period}; expected one of "
                f"{', '.join(sorted(expected_periods))}"
            )
        timestamp = parse_utc_timestamp(
            f"{row['<DATE>']}{row['<TIME>']}",
            "%Y%m%d%H%M%S",
            source_timezone=self.source_timezone,
        )
        return {
            "timestamp": timestamp,
            "open": float(row["<OPEN>"]),
            "high": float(row["<HIGH>"]),
            "low": float(row["<LOW>"]),
            "close": float(row["<CLOSE>"]),
            "volume": float(row.get("<VOL>", 0) or 0),
            "symbol": symbol,
        }

    def _validate_rows(
        self,
        symbol: str,
        timeframe: str,
        rows: list[dict[str, Any]],
    ) -> None:
        timestamps = [row["timestamp"] for row in rows]
        if timestamps != sorted(timestamps):
            raise ValueError(f"{symbol} {timeframe} timestamps are not sorted")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError(f"{symbol} {timeframe} contains duplicate timestamps")
        for index, row in enumerate(rows, start=1):
            if row["symbol"] != symbol:
                raise ValueError(
                    f"{symbol} {timeframe} row {index} has mismatched symbol {row['symbol']}"
                )
            if (
                min(row["open"], row["high"], row["low"], row["close"]) <= 0
                or row["volume"] < 0
                or row["high"] < row["low"]
                or row["high"] < max(row["open"], row["close"])
                or row["low"] > min(row["open"], row["close"])
            ):
                raise ValueError(f"{symbol} {timeframe} row {index} has invalid OHLCV")

    def _write_parquet(self, path: Path, rows: list[dict[str, Any]]) -> None:
        logger.debug("Writing parquet -> %s", path)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Market Parquet import requires pyarrow. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([{column: row[column] for column in STANDARD_COLUMNS} for row in rows])
        temporary_path = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(path)

    def _symbol_from_path(self, path: Path) -> str:
        stem = path.stem.upper()
        for token in (
            "_5M",
            "-5M",
            "_5MIN",
            "-5MIN",
            "_1H",
            "-1H",
            "_H",
            "-H",
            "_60M",
            "-60M",
            "_60MIN",
            "-60MIN",
        ):
            stem = stem.replace(token, "")
        return stem.split(".")[0]


def migrate_legacy_daily_parquet(
    legacy_dir: str | Path,
    output_root: str | Path,
    *,
    symbols: Iterable[str] | None = None,
    resume: bool = True,
) -> list[MarketParquetImportResult]:
    legacy_root = Path(legacy_dir)
    if not legacy_root.exists():
        raise FileNotFoundError(f"Legacy daily Parquet directory not found: {legacy_root}")
    requested = {str(symbol).upper() for symbol in symbols or []}
    paths = sorted(legacy_root.glob("*.parquet"))
    if requested:
        paths = [path for path in paths if path.stem.upper() in requested]
    results = []
    for path in paths:
        symbol = path.stem.upper()
        output_path = market_parquet_path(output_root, symbol, "1Day")
        assert_market_parquet_output_path(output_path, output_root, symbol, "1Day")
        rows = _read_legacy_daily_rows(path, symbol)
        duplicate_count = _duplicate_count(rows)
        if resume and output_path.exists():
            results.append(
                MarketParquetImportResult(
                    symbol=symbol,
                    timeframe="1Day",
                    source_path=str(path),
                    output_path=str(output_path),
                    row_count=len(rows),
                    first_timestamp=rows[0]["timestamp"].isoformat(),
                    last_timestamp=rows[-1]["timestamp"].isoformat(),
                    duplicate_count=duplicate_count,
                    skipped_existing=True,
                )
            )
            continue
        _validate_standard_rows(symbol, "1Day", rows)
        _write_standard_parquet(output_path, rows)
        results.append(
            MarketParquetImportResult(
                symbol=symbol,
                timeframe="1Day",
                source_path=str(path),
                output_path=str(output_path),
                row_count=len(rows),
                first_timestamp=rows[0]["timestamp"].isoformat(),
                last_timestamp=rows[-1]["timestamp"].isoformat(),
                duplicate_count=duplicate_count,
            )
        )
    missing = sorted(requested - {result.symbol for result in results})
    if missing:
        raise FileNotFoundError(
            "Missing legacy daily Parquet for symbols: " + ", ".join(missing)
        )
    return results


class MarketParquetDataFeed(IDataFeed):
    """Read canonical local Parquet bars for daily and intraday research."""

    def __init__(self, data_root: str | Path = "data/processed"):
        self.data_root = Path(data_root)
        self._last_request_metadata: dict[str, dict] = {}

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        canonical_timeframe = normalize_timeframe(timeframe)
        normalized_symbol = symbol.upper()
        path = market_parquet_path(self.data_root, normalized_symbol, canonical_timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"Market Parquet data not found for {normalized_symbol} "
                f"{canonical_timeframe}: {path}. Run --mode import-market-parquet."
            )
        candles = [
            candle for candle in self._load(path, normalized_symbol)
            if _within_range(candle.timestamp, start, end)
        ]
        self._last_request_metadata[normalized_symbol] = {
            "source": "market_parquet",
            "source_file": str(path),
            "timeframe": canonical_timeframe,
            "timestamp_timezone": "UTC",
            "timestamp_semantics": "bar_close",
            "page_count": 0,
        }
        return candles

    def get_last_request_metadata(self, symbol: str) -> dict:
        return dict(self._last_request_metadata.get(symbol.upper(), {}))

    def _load(self, path: Path, symbol: str) -> list[Candle]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Market Parquet research requires pyarrow. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        table = pq.read_table(path, columns=list(STANDARD_COLUMNS))
        data = table.to_pydict()
        return sorted(
            [
                Candle(
                    symbol=str(row_symbol or symbol).upper(),
                    timestamp=_ensure_utc(timestamp),
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume),
                )
                for timestamp, open_price, high_price, low_price, close_price, volume, row_symbol in zip(
                    data["timestamp"],
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    data["volume"],
                    data["symbol"],
                )
            ],
            key=lambda candle: candle.timestamp,
        )


def market_parquet_path(root: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(root) / symbol.upper() / normalize_timeframe(timeframe) / "bars.parquet"


def assert_market_parquet_output_path(
    path: str | Path,
    root: str | Path,
    symbol: str,
    timeframe: str,
) -> None:
    canonical_timeframe = normalize_timeframe(timeframe)
    expected = market_parquet_path(root, symbol, canonical_timeframe)
    actual = Path(path)
    if actual != expected:
        raise AssertionError(
            "Market Parquet output path must include symbol and timeframe: "
            f"expected {expected}, got {actual}"
        )
    if actual.name != "bars.parquet":
        raise AssertionError(
            f"Market Parquet output filename must be bars.parquet: {actual}"
        )
    if actual.parent.name != canonical_timeframe:
        raise AssertionError(
            "Market Parquet output path is missing timeframe directory "
            f"{canonical_timeframe}: {actual}"
        )
    if actual.parent.parent.name != str(symbol).upper():
        raise AssertionError(
            "Market Parquet output path is missing symbol directory "
            f"{str(symbol).upper()}: {actual}"
        )


def normalize_timeframe(value: str) -> str:
    normalized = str(value).replace("_", "").replace("-", "").replace(" ", "").lower()
    timeframe = TIMEFRAME_ALIASES.get(normalized, str(value))
    if timeframe not in SUPPORTED_TIMEFRAMES:
        allowed = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        raise ValueError(f"Unsupported market timeframe '{value}'. Use one of: {allowed}")
    return timeframe


def normalize_timeframe_source_folder(folder_name: str) -> str:
    normalized = (
        str(folder_name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    source_folder_aliases = {
        "5min": "5m",
        "5mins": "5m",
        "5minute": "5m",
        "5minutes": "5m",
        "5m": "5m",
        "hourly": "1h",
        "1hour": "1h",
        "1hours": "1h",
        "hour": "1h",
        "h": "1h",
        "1h": "1h",
        "daily": "1Day",
        "1day": "1Day",
        "day": "1Day",
        "d": "1Day",
    }
    return source_folder_aliases.get(normalized, "")


def parse_utc_timestamp(
    value: str,
    fmt: str | None = None,
    *,
    source_timezone: ZoneInfo = ZoneInfo("UTC"),
) -> datetime:
    text = value.strip()
    parsed = datetime.strptime(text, fmt) if fmt else datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=source_timezone)
    return _ensure_utc(parsed)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _within_range(value: datetime, start: datetime, end: datetime) -> bool:
    timestamp = _ensure_utc(value)
    return _ensure_utc(start) <= timestamp <= _ensure_utc(end)


def _float_value(
    row: dict[str, str],
    names: tuple[str, ...],
    line_number: int,
    *,
    default: float | None = None,
) -> float:
    for name in names:
        if name in row and row[name] not in {None, ""}:
            return float(row[name])
    if default is not None:
        return default
    raise ValueError(f"row {line_number} is missing one of: {', '.join(names)}")


def _duplicate_count(rows: list[dict[str, Any]]) -> int:
    timestamps = [row["timestamp"] for row in rows]
    return len(timestamps) - len(set(timestamps))


def _detect_source_timeframe(path: Path, raw_root: Path) -> tuple[str, Path | None]:
    for parent in _relative_parents(path.parent, raw_root):
        canonical_timeframe = normalize_timeframe_source_folder(parent.name)
        if canonical_timeframe:
            return canonical_timeframe, parent
    return "", None


def _relative_parents(path: Path, root: Path) -> list[Path]:
    resolved_root = root.resolve()
    parents = [path, *path.parents]
    contained = []
    for parent in parents:
        try:
            parent.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        contained.append(parent)
    return list(reversed(contained))


def _timeframe_tokens(timeframe: str) -> tuple[str, ...]:
    if timeframe == "5m":
        return ("5m", "5min", "5_min", "5-minute", "5minute")
    if timeframe == "1h":
        return ("1h", "hour", "hourly", "60m", "60min", "60_min")
    return ("1day", "daily", "day")


def _read_legacy_daily_rows(path: Path, symbol: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Market Parquet migration requires pyarrow. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    table = pq.read_table(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    data = table.to_pydict()
    return sorted(
        [
            {
                "timestamp": _ensure_utc(timestamp),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
                "symbol": symbol,
            }
            for timestamp, open_price, high_price, low_price, close_price, volume in zip(
                data["timestamp"],
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
            )
        ],
        key=lambda row: row["timestamp"],
    )


def _validate_standard_rows(
    symbol: str,
    timeframe: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"{symbol} {timeframe} contains no rows")
    timestamps = [row["timestamp"] for row in rows]
    if timestamps != sorted(timestamps):
        raise ValueError(f"{symbol} {timeframe} timestamps are not sorted")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{symbol} {timeframe} contains duplicate timestamps")
    for index, row in enumerate(rows, start=1):
        if row["symbol"] != symbol:
            raise ValueError(
                f"{symbol} {timeframe} row {index} has mismatched symbol {row['symbol']}"
            )
        if (
            min(row["open"], row["high"], row["low"], row["close"]) <= 0
            or row["volume"] < 0
            or row["high"] < row["low"]
            or row["high"] < max(row["open"], row["close"])
            or row["low"] > min(row["open"], row["close"])
        ):
            raise ValueError(f"{symbol} {timeframe} row {index} has invalid OHLCV")


def _write_standard_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    logger.debug("Writing parquet -> %s", path)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Market Parquet import requires pyarrow. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [{column: row[column] for column in STANDARD_COLUMNS} for row in rows]
    )
    temporary_path = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary_path, compression="zstd")
    temporary_path.replace(path)
