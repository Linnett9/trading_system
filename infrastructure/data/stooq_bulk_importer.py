from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import io
from pathlib import Path
from typing import TextIO
import zipfile


_REQUIRED_COLUMNS = (
    "<TICKER>", "<PER>", "<DATE>", "<TIME>", "<OPEN>", "<HIGH>",
    "<LOW>", "<CLOSE>", "<VOL>", "<OPENINT>",
)


@dataclass(frozen=True)
class StooqBulkImportResult:
    symbol: str
    source_path: str
    output_path: str
    row_count: int
    first_date: str
    last_date: str
    duplicate_count: int = 0
    missing_trading_day_gaps: int = 0
    max_calendar_gap_days: int = 0
    integrity_validated: bool = True
    skipped_existing: bool = False


@dataclass(frozen=True)
class StooqBulkImportManifest:
    source: str
    output_dir: str
    requested_symbol_count: int
    imported: list[StooqBulkImportResult]
    skipped_existing: list[StooqBulkImportResult]
    missing_symbols: list[dict[str, str]]
    failed_symbols: list[dict[str, str]]
    duplicate_symbols: list[dict[str, str | int]]
    missing_data_symbols: list[dict[str, str | int]]
    generated_at: str
    research_only: bool = True
    trading_impact: str = "none"


@dataclass(frozen=True)
class StooqRawSymbolCandidate:
    symbol: str
    source_path: str
    asset_class: str
    row_count: int


class StooqBulkImporter:
    """Validate immutable Stooq ASCII rows and materialize local Parquet files."""

    def __init__(
        self,
        extracted_dir: str,
        parquet_dir: str,
        zip_path: str | None = None,
        minimum_history_years: int = 9,
        history_tolerance_days: int = 10,
    ):
        self.extracted_dir = Path(extracted_dir)
        self.parquet_dir = Path(parquet_dir)
        self.zip_path = Path(zip_path) if zip_path else None
        self.minimum_history_years = minimum_history_years
        self.history_tolerance_days = history_tolerance_days
        self._symbol_index: dict[str, Path] | None = None

    def raw_symbol_candidates(
        self,
        asset_class: str = "all",
        min_rows: int = 0,
        exclude_warrants_units_rights: bool = False,
    ) -> list[StooqRawSymbolCandidate]:
        normalized_asset_class = asset_class.lower()
        if normalized_asset_class not in {"stocks", "etfs", "all"}:
            raise ValueError(
                "asset_class must be one of: stocks, etfs, all"
            )
        candidates: list[StooqRawSymbolCandidate] = []
        for symbol, path in self._symbol_index_by_symbol().items():
            path_asset_class = self._asset_class_from_path(path)
            if (
                normalized_asset_class != "all"
                and path_asset_class != normalized_asset_class
            ):
                continue
            if (
                exclude_warrants_units_rights
                and self._is_warrant_unit_or_right(symbol)
            ):
                continue
            row_count = self._count_price_rows(path)
            if row_count < min_rows:
                continue
            candidates.append(
                StooqRawSymbolCandidate(
                    symbol=symbol,
                    source_path=str(path),
                    asset_class=path_asset_class,
                    row_count=row_count,
                )
            )
        return sorted(
            candidates,
            key=lambda candidate: (-candidate.row_count, candidate.symbol),
        )

    def select_raw_symbols(
        self,
        top: int | None = None,
        asset_class: str = "all",
        min_rows: int = 0,
        exclude_warrants_units_rights: bool = False,
    ) -> list[StooqRawSymbolCandidate]:
        if top is not None and top <= 0:
            raise ValueError("top must be positive")
        candidates = self.raw_symbol_candidates(
            asset_class=asset_class,
            min_rows=min_rows,
            exclude_warrants_units_rights=exclude_warrants_units_rights,
        )
        return candidates[:top] if top is not None else candidates

    def import_symbols(
        self,
        symbols: list[str],
        resume: bool = False,
    ) -> list[StooqBulkImportResult]:
        return [self.import_symbol(symbol, resume=resume) for symbol in symbols]

    def import_symbols_with_manifest(
        self,
        symbols: list[str],
        manifest_path: str | Path | None = None,
        resume: bool = True,
    ) -> StooqBulkImportManifest:
        imported: list[StooqBulkImportResult] = []
        skipped: list[StooqBulkImportResult] = []
        missing: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []
        duplicate_symbols: list[dict[str, str | int]] = []
        missing_data_symbols: list[dict[str, str | int]] = []

        for symbol in [str(value).upper() for value in symbols]:
            try:
                result = self.import_symbol(symbol, resume=resume)
            except FileNotFoundError as exc:
                missing.append({"symbol": symbol, "reason": str(exc)})
                continue
            except Exception as exc:
                reason = str(exc)
                failed.append({"symbol": symbol, "reason": reason})
                if "duplicate dates" in reason:
                    duplicate_symbols.append({
                        "symbol": symbol,
                        "duplicate_count": 1,
                    })
                continue

            if result.skipped_existing:
                skipped.append(result)
            else:
                imported.append(result)
            if result.duplicate_count:
                duplicate_symbols.append({
                    "symbol": symbol,
                    "duplicate_count": result.duplicate_count,
                })
            if result.missing_trading_day_gaps:
                missing_data_symbols.append({
                    "symbol": symbol,
                    "missing_trading_day_gaps": result.missing_trading_day_gaps,
                    "max_calendar_gap_days": result.max_calendar_gap_days,
                })

        manifest = StooqBulkImportManifest(
            source="stooq_bulk",
            output_dir=str(self.parquet_dir),
            requested_symbol_count=len(symbols),
            imported=imported,
            skipped_existing=skipped,
            missing_symbols=missing,
            failed_symbols=failed,
            duplicate_symbols=duplicate_symbols,
            missing_data_symbols=missing_data_symbols,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
        if manifest_path is not None:
            self.write_manifest(Path(manifest_path), manifest)
        return manifest

    def import_symbol(
        self,
        symbol: str,
        resume: bool = False,
    ) -> StooqBulkImportResult:
        normalized_symbol = symbol.upper()
        rows, source_path = self._read_symbol(normalized_symbol)
        output_path = self.parquet_dir / f"{normalized_symbol}.parquet"
        duplicate_count = self._duplicate_count(rows)
        gap_report = self._missing_data_report(rows)
        self._validate_rows(normalized_symbol, rows)
        if resume and output_path.exists():
            return StooqBulkImportResult(
                symbol=normalized_symbol,
                source_path=source_path,
                output_path=str(output_path),
                row_count=len(rows),
                first_date=rows[0]["timestamp"].date().isoformat(),
                last_date=rows[-1]["timestamp"].date().isoformat(),
                duplicate_count=duplicate_count,
                missing_trading_day_gaps=gap_report["missing_trading_day_gaps"],
                max_calendar_gap_days=gap_report["max_calendar_gap_days"],
                integrity_validated=True,
                skipped_existing=True,
            )
        self._write_parquet(output_path, rows)
        return StooqBulkImportResult(
            symbol=normalized_symbol,
            source_path=source_path,
            output_path=str(output_path),
            row_count=len(rows),
            first_date=rows[0]["timestamp"].date().isoformat(),
            last_date=rows[-1]["timestamp"].date().isoformat(),
            duplicate_count=duplicate_count,
            missing_trading_day_gaps=gap_report["missing_trading_day_gaps"],
            max_calendar_gap_days=gap_report["max_calendar_gap_days"],
            integrity_validated=True,
        )

    def write_manifest(
        self,
        path: Path,
        manifest: StooqBulkImportManifest,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "source": manifest.source,
                "output_dir": manifest.output_dir,
                "requested_symbol_count": manifest.requested_symbol_count,
                "imported_symbol_count": len(manifest.imported),
                "skipped_existing_symbol_count": len(manifest.skipped_existing),
                "missing_symbol_count": len(manifest.missing_symbols),
                "failed_symbol_count": len(manifest.failed_symbols),
                "duplicate_symbol_count": len(manifest.duplicate_symbols),
                "missing_data_symbol_count": len(manifest.missing_data_symbols),
                "symbols": [result.__dict__ for result in manifest.imported],
                "skipped_existing_symbols": [
                    result.__dict__ for result in manifest.skipped_existing
                ],
                "missing_symbols": manifest.missing_symbols,
                "failed_symbols": manifest.failed_symbols,
                "duplicate_symbols": manifest.duplicate_symbols,
                "missing_data_symbols": manifest.missing_data_symbols,
                "generated_at": manifest.generated_at,
                "research_only": manifest.research_only,
                "trading_impact": manifest.trading_impact,
            }, indent=2),
            encoding="utf-8",
        )

    def _read_symbol(self, symbol: str) -> tuple[list[dict], str]:
        extracted_path = self.extracted_dir / f"{symbol}.txt"
        if extracted_path.exists():
            with extracted_path.open("r", encoding="utf-8", newline="") as handle:
                return self._parse_rows(symbol, handle), str(extracted_path)
        recursive_path = self._recursive_symbol_path(symbol)
        if recursive_path is not None:
            with recursive_path.open("r", encoding="utf-8", newline="") as handle:
                return self._parse_rows(symbol, handle), str(recursive_path)
        if self.zip_path and self.zip_path.exists():
            return self._read_zip_member(symbol)
        raise FileNotFoundError(
            "Stooq bulk data not found for "
            f"{symbol}: expected {extracted_path} or recursive *.us.txt under "
            f"{', '.join(str(path) for path in self._symbol_search_roots())}"
        )

    def _recursive_symbol_path(self, symbol: str) -> Path | None:
        return self._symbol_index_by_symbol().get(symbol.upper())

    def _symbol_index_by_symbol(self) -> dict[str, Path]:
        if self._symbol_index is not None:
            return self._symbol_index
        self._symbol_index = self._build_symbol_index()
        return self._symbol_index

    def _build_symbol_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for root in self._symbol_search_roots():
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.txt")):
                symbol = self._symbol_from_stooq_path(path)
                if symbol is None:
                    continue
                index.setdefault(symbol, path)
        return index

    def _symbol_search_roots(self) -> list[Path]:
        roots = [self.extracted_dir]
        if self.extracted_dir.name == "extracted":
            roots.append(self.extracted_dir.parent)
        return list(dict.fromkeys(roots))

    def _symbol_from_stooq_path(self, path: Path) -> str | None:
        stem_parts = path.stem.split(".")
        if len(stem_parts) < 2 or stem_parts[-1].lower() != "us":
            return None
        symbol = stem_parts[0].upper()
        return symbol or None

    def _asset_class_from_path(self, path: Path) -> str:
        parts = [part.lower() for part in path.parts]
        if any("etfs" in part for part in parts):
            return "etfs"
        if any("stocks" in part for part in parts):
            return "stocks"
        return "unknown"

    def _count_price_rows(self, path: Path) -> int:
        with path.open("r", encoding="utf-8", newline="") as handle:
            line_count = sum(1 for _ in handle)
        return max(0, line_count - 1)

    def _is_warrant_unit_or_right(self, symbol: str) -> bool:
        normalized = symbol.upper()
        explicit_suffixes = ("WS", "WT", "WTS", "UN", "UNIT", "RT", "RIGHT")
        if normalized.endswith(explicit_suffixes):
            return True
        return len(normalized) >= 5 and normalized.endswith(("W", "U", "R"))

    def _read_zip_member(self, symbol: str) -> tuple[list[dict], str]:
        with zipfile.ZipFile(self.zip_path) as archive:
            members = [
                name for name in archive.namelist()
                if Path(name).stem.upper() == symbol and name.lower().endswith(".txt")
            ]
            if not members:
                raise FileNotFoundError(
                    f"Stooq bulk ZIP has no TXT member for {symbol}: {self.zip_path}"
                )
            member = members[0]
            with archive.open(member) as raw_handle:
                with io.TextIOWrapper(raw_handle, encoding="utf-8", newline="") as handle:
                    return self._parse_rows(symbol, handle), f"{self.zip_path}!{member}"

    def _parse_rows(self, symbol: str, handle: TextIO) -> list[dict]:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _REQUIRED_COLUMNS:
            raise ValueError(
                f"Stooq bulk {symbol} has invalid columns; expected {_REQUIRED_COLUMNS}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            rows.append(self._parse_row(symbol, row, line_number))
        if not rows:
            raise ValueError(f"Stooq bulk {symbol} contains no price rows")
        return rows

    def _parse_row(self, symbol: str, row: dict[str, str], line_number: int) -> dict:
        try:
            if row["<PER>"] != "D":
                raise ValueError("period must be D")
            timestamp = datetime.strptime(
                f"{row['<DATE>']}{row['<TIME>']}", "%Y%m%d%H%M%S"
            )
            values = {
                "open": float(row["<OPEN>"]),
                "high": float(row["<HIGH>"]),
                "low": float(row["<LOW>"]),
                "close": float(row["<CLOSE>"]),
                "volume": float(row["<VOL>"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Stooq bulk {symbol} row {line_number} is invalid: {exc}"
            ) from exc
        if (
            min(values["open"], values["high"], values["low"], values["close"]) <= 0
            or values["volume"] < 0
            or values["high"] < values["low"]
            or values["high"] < max(values["open"], values["close"])
            or values["low"] > min(values["open"], values["close"])
        ):
            raise ValueError(
                f"Stooq bulk {symbol} row {line_number} has invalid OHLCV values"
            )
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            **values,
            "source": "stooq_bulk",
        }

    def _validate_rows(self, symbol: str, rows: list[dict]) -> None:
        dates = [row["timestamp"] for row in rows]
        if dates != sorted(dates):
            raise ValueError(f"Stooq bulk {symbol} dates are not sorted ascending")
        if len(set(dates)) != len(dates):
            raise ValueError(f"Stooq bulk {symbol} has duplicate dates")
        available_days = (dates[-1] - dates[0]).days
        required_days = self.minimum_history_years * 365 - self.history_tolerance_days
        if available_days < required_days:
            raise ValueError(
                f"Stooq bulk {symbol} has insufficient history: {available_days} days; "
                f"requires {self.minimum_history_years} years"
            )

    def _duplicate_count(self, rows: list[dict]) -> int:
        dates = [row["timestamp"] for row in rows]
        return len(dates) - len(set(dates))

    def _missing_data_report(self, rows: list[dict]) -> dict[str, int]:
        dates = [row["timestamp"].date() for row in rows]
        gaps = [
            (current - previous).days
            for previous, current in zip(dates, dates[1:])
            if (current - previous).days > 4
        ]
        return {
            "missing_trading_day_gaps": len(gaps),
            "max_calendar_gap_days": max(gaps, default=0),
        }

    def _write_parquet(self, path: Path, rows: list[dict]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Stooq bulk Parquet import requires pyarrow. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        temporary_path = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(path)
