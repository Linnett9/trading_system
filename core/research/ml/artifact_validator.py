from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.research.ml.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    REQUIRED_PREDICTION_ARTIFACT_COLUMNS,
)


@dataclass(frozen=True)
class PredictionArtifactValidationResult:
    csv_path: Path
    metadata_path: Path
    row_count: int
    dataset_hash: str
    artifact_schema_version: str
    legacy_warnings: tuple[str, ...] = ()


def validate_prediction_artifacts(
    csv_path: Path,
    metadata_path: Path | None = None,
) -> PredictionArtifactValidationResult:
    metadata_path = metadata_path or csv_path.with_suffix(".json")
    if not csv_path.exists():
        raise RuntimeError(f"Missing prediction artifact CSV: {csv_path}")
    if not metadata_path.exists():
        raise RuntimeError(f"Missing prediction artifact metadata: {metadata_path}")

    metadata = _read_json(metadata_path)
    metadata_dataset_hash = str(
        metadata.get("dataset_hash") or metadata.get("data_hash") or ""
    )
    if not metadata_dataset_hash:
        raise RuntimeError(
            f"Prediction artifact metadata is missing dataset_hash: {metadata_path}"
        )

    rows, fieldnames = _read_rows(csv_path)
    missing_columns = [
        column
        for column in REQUIRED_PREDICTION_ARTIFACT_COLUMNS
        if column not in fieldnames
    ]
    legacy_warnings: list[str] = []
    if missing_columns:
        legacy_warnings.append(
            "legacy_prediction_artifact_missing_columns:"
            + ",".join(missing_columns)
        )

    row_hashes = {
        str(row.get("dataset_hash", ""))
        for row in rows
        if row.get("split", "holdout") in {"out_of_fold", "holdout", "test"}
    }
    if "" in row_hashes:
        raise RuntimeError(f"Prediction artifact CSV is missing dataset_hash: {csv_path}")
    if row_hashes and row_hashes != {metadata_dataset_hash}:
        raise RuntimeError(
            "Prediction artifact CSV dataset_hash does not match metadata "
            f"for {csv_path}: csv={sorted(row_hashes)} "
            f"metadata={metadata_dataset_hash}"
        )

    row_schema_versions = {
        str(row.get("artifact_schema_version", ""))
        for row in rows
        if row.get("artifact_schema_version")
    }
    metadata_schema_version = str(metadata.get("artifact_schema_version", ""))
    if not metadata_schema_version:
        legacy_warnings.append("legacy_prediction_artifact_missing_metadata_schema")
    elif metadata_schema_version != ARTIFACT_SCHEMA_VERSION:
        legacy_warnings.append(
            f"unexpected_metadata_schema_version:{metadata_schema_version}"
        )
    if row_schema_versions and row_schema_versions != {ARTIFACT_SCHEMA_VERSION}:
        legacy_warnings.append(
            "unexpected_csv_schema_versions:" + ",".join(sorted(row_schema_versions))
        )
    if not row_schema_versions:
        legacy_warnings.append("legacy_prediction_artifact_missing_csv_schema")

    return PredictionArtifactValidationResult(
        csv_path=csv_path,
        metadata_path=metadata_path,
        row_count=len(rows),
        dataset_hash=metadata_dataset_hash,
        artifact_schema_version=metadata_schema_version,
        legacy_warnings=tuple(legacy_warnings),
    )


def validate_prediction_artifact_dirs(
    source_dirs: list[Path],
) -> list[PredictionArtifactValidationResult]:
    missing_dirs = [source_dir for source_dir in source_dirs if not source_dir.exists()]
    if missing_dirs:
        raise RuntimeError(
            "Prediction artifact directories do not exist: "
            + ", ".join(str(path) for path in missing_dirs)
        )

    results = [
        validate_prediction_artifacts(
            source_dir / "prediction_artifacts.csv",
            source_dir / "prediction_artifacts.json",
        )
        for source_dir in source_dirs
    ]
    dataset_hashes = {result.dataset_hash for result in results if result.dataset_hash}
    if len(dataset_hashes) > 1:
        details = ", ".join(
            f"{result.csv_path.parent}={result.dataset_hash}"
            for result in results
        )
        raise RuntimeError(
            "Prediction artifacts were generated from different dataset hashes. "
            f"{details}"
        )
    return results


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Prediction artifact metadata must be an object: {path}")
    return payload
