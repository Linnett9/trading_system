from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
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

    def import_symbols(self, symbols: list[str]) -> list[StooqBulkImportResult]:
        return [self.import_symbol(symbol) for symbol in symbols]

    def import_symbol(self, symbol: str) -> StooqBulkImportResult:
        normalized_symbol = symbol.upper()
        rows, source_path = self._read_symbol(normalized_symbol)
        self._validate_rows(normalized_symbol, rows)
        output_path = self.parquet_dir / f"{normalized_symbol}.parquet"
        self._write_parquet(output_path, rows)
        return StooqBulkImportResult(
            symbol=normalized_symbol,
            source_path=source_path,
            output_path=str(output_path),
            row_count=len(rows),
            first_date=rows[0]["timestamp"].date().isoformat(),
            last_date=rows[-1]["timestamp"].date().isoformat(),
        )

    def _read_symbol(self, symbol: str) -> tuple[list[dict], str]:
        extracted_path = self.extracted_dir / f"{symbol}.txt"
        if extracted_path.exists():
            with extracted_path.open("r", encoding="utf-8", newline="") as handle:
                return self._parse_rows(symbol, handle), str(extracted_path)
        if self.zip_path and self.zip_path.exists():
            return self._read_zip_member(symbol)
        raise FileNotFoundError(
            "Stooq bulk data not found for "
            f"{symbol}: expected {extracted_path}"
        )

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
