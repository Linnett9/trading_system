from __future__ import annotations

import hashlib
import json
import os
import fnmatch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT = "stock_alpha_news_assembly_recovery_request.v1"
NOTICE = "READ_ONLY - NOT PRODUCTION EXECUTION AUTHORIZATION"
ASSEMBLY_NAME = "stock_alpha_news_historical_corpus_assembly.csv"
DATA_SUFFIXES = {".csv", ".jsonl", ".parquet"}
METADATA_SUFFIXES = {".json", ".md", ".txt"}
REQUIRED_HEADER_GROUPS = (
    ("provider", "delivery_provider", "source_provider"),
    ("symbol", "ticker"),
    ("published_at_utc", "published_at", "created_at"),
    ("provider_article_id", "article_id", "id"),
)


@dataclass(frozen=True)
class AssemblyRecoveryRequest:
    search_roots: tuple[str, ...]
    target_checksum: str
    expected_filename_patterns: tuple[str, ...]
    expected_provider_scope: tuple[str, ...]
    minimum_size_bytes: int
    maximum_size_bytes: int
    expected_row_count: int
    expected_unique_article_count: int
    expected_symbol_count: int
    expected_published_min: str
    expected_published_max: str
    maximum_depth: int
    maximum_candidates: int
    maximum_metadata_bytes: int
    maximum_full_hashes: int
    output_root: str
    contract_version: str = CONTRACT

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT:
            raise ValueError("Unsupported assembly recovery request")
        if not self.search_roots or any(
            not str(root).strip() for root in self.search_roots
        ):
            raise ValueError("Explicit search roots are required")
        if len(self.target_checksum) != 64:
            raise ValueError("Exact SHA-256 target checksum is required")
        if not self.expected_filename_patterns or not self.expected_provider_scope:
            raise ValueError("Explicit filename patterns and providers are required")
        bounds = (
            self.minimum_size_bytes, self.maximum_size_bytes,
            self.expected_row_count, self.expected_unique_article_count,
            self.expected_symbol_count, self.maximum_depth,
            self.maximum_candidates, self.maximum_metadata_bytes,
            self.maximum_full_hashes,
        )
        if min(bounds) < 1 or self.minimum_size_bytes > self.maximum_size_bytes:
            raise ValueError("Recovery bounds must be positive and ordered")
        if not self.output_root.strip():
            raise ValueError("Explicit output root is required")

    @property
    def identity(self) -> str:
        payload = asdict(self)
        payload["search_roots"] = sorted(
            str(Path(root).resolve()) for root in self.search_roots
        )
        payload["output_root"] = str(Path(self.output_root).resolve())
        return _identity(payload)


def audit_assembly_recovery(
    request: AssemblyRecoveryRequest, *, strict: bool = False,
) -> dict[str, Any]:
    output_root = Path(request.output_root).resolve()
    candidates, root_inventory = discover_recovery_candidates(request)
    plausible = [
        row for row in candidates
        if row["preliminary_hash_eligible"]
    ]
    checksum_results = []
    for row in plausible[:request.maximum_full_hashes]:
        path = Path(row["path"])
        before = _snapshot(path)
        checksum = _sha256(path)
        after = _snapshot(path)
        result = {
            "notice": NOTICE, "path": row["path"], "sha256": checksum,
            "target_match": checksum == request.target_checksum,
            "pre_snapshot": before, "post_snapshot": after,
            "snapshot_unchanged": before == after,
        }
        checksum_results.append(result)
        row["sha256"] = checksum
        if checksum == request.target_checksum:
            row["classification"] = "EXACT_ASSEMBLY_CHECKSUM_MATCH"
            row["reason_codes"] = []
        elif _semantic_identity_match(row, request):
            row["classification"] = "ASSEMBLY_IDENTITY_MATCH_WITHOUT_BYTE_MATCH"
            row["reason_codes"] = ["TARGET_CHECKSUM_MISMATCH"]
        else:
            row["classification"] = "PLAUSIBLE_ASSEMBLY_UNVERIFIED"
            row["reason_codes"] = ["TARGET_CHECKSUM_MISMATCH"]
    for row in plausible[request.maximum_full_hashes:]:
        row["reason_codes"] = sorted(set(
            row["reason_codes"] + ["FULL_HASH_LIMIT_REACHED"]
        ))

    exact = [
        row for row in candidates
        if row["classification"] == "EXACT_ASSEMBLY_CHECKSUM_MATCH"
    ]
    backfill = _backfill_inventory(candidates, request)
    rebuild = _rebuild_readiness(exact, backfill)
    blockers = []
    warnings = []
    if not exact:
        blockers.append({"code": "TARGET_ASSEMBLY_CHECKSUM_NOT_FOUND"})
    if len(exact) > 1:
        warnings.append({
            "code": "MULTIPLE_BYTE_IDENTICAL_ASSEMBLIES",
            "count": len(exact),
        })
    if rebuild["status"] not in {
        "EXACT_ASSEMBLY_AVAILABLE", "ASSEMBLY_REBUILD_READY",
    }:
        blockers.extend({"code": code} for code in rebuild["blocker_codes"])
    if len(plausible) > request.maximum_full_hashes:
        warnings.append({"code": "FULL_HASH_LIMIT_REACHED"})

    selection = None
    if exact:
        selected = sorted(exact, key=lambda row: row["path"])[0]
        selection = {
            "notice": NOTICE,
            "selection": {
                "HISTORICAL_SOURCE_ASSEMBLY": selected["path"],
            },
            "assembly_checksum": request.target_checksum,
            "selection_basis": "EXACT_SHA256_AND_ASSEMBLY_SCHEMA",
            "multiple_identical_copies": len(exact) > 1,
        }
    report = {
        "status": "BLOCKED" if blockers else (
            "READY_WITH_CONDITIONS" if warnings else "READY"
        ),
        "strict_failure": strict and bool(blockers or warnings),
        "request_identity": request.identity,
        "candidate_count": len(candidates),
        "plausible_candidate_count": len(plausible),
        "full_hash_count": len(checksum_results),
        "target_checksum_found": bool(exact),
        "exact_match_count": len(exact),
        "rebuild_readiness": rebuild["status"],
        "blockers": blockers,
        "warnings": warnings,
        "read_only": True,
        "network_access_performed": False,
        "model_activation_performed": False,
        "lease_acquired": False,
        "ledger_or_registry_mutated": False,
        "external_files_mutated": any(
            not row["snapshot_unchanged"] for row in checksum_results
        ),
    }
    _publish(
        output_root, request, root_inventory, candidates, checksum_results,
        selection, backfill, rebuild, blockers, warnings,
    )
    return report


def discover_recovery_candidates(
    request: AssemblyRecoveryRequest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    roots = []
    for root_text in request.search_roots:
        root = Path(root_text).resolve()
        root_row = {
            "notice": NOTICE, "path": str(root),
            "available": root.exists(), "files_considered": 0,
            "candidate_limit_reached": False,
        }
        roots.append(root_row)
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            if depth >= request.maximum_depth:
                directories[:] = []
            directories[:] = sorted(
                name for name in directories
                if not name.startswith(".")
                and name not in {"node_modules", "__pycache__", ".git"}
            )
            for name in sorted(files):
                root_row["files_considered"] += 1
                path = current_path / name
                if not _candidate_name_or_size(path, request):
                    continue
                if len(candidates) >= request.maximum_candidates:
                    root_row["candidate_limit_reached"] = True
                    return sorted(candidates, key=lambda row: row["path"]), roots
                candidates.append(_inspect_candidate(path, request))
    return sorted(candidates, key=lambda row: row["path"]), roots


def _candidate_name_or_size(path, request):
    lower = path.name.lower()
    pattern_match = any(
        _pattern_match(lower, pattern.lower())
        for pattern in request.expected_filename_patterns
    )
    metadata_match = (
        path.suffix.lower() in METADATA_SUFFIXES
        and any(scope in lower for scope in (
            "stock_alpha_news", "historical_backfill", "news_historical",
            "corpus_assembly",
        ))
        and any(kind in lower for kind in (
            "assembly", "backfill", "provider", "inventory",
            "manifest", "checksum", "summary",
        ))
    )
    size_match = (
        path.suffix.lower() in DATA_SUFFIXES
        and request.minimum_size_bytes <= path.stat().st_size
        <= request.maximum_size_bytes
        and any(scope in lower for scope in (
            "news", "historical", "corpus", "assembly",
        ))
    )
    return pattern_match or metadata_match or size_match


def _inspect_candidate(path, request):
    lower = str(path).lower()
    base = {
        "notice": NOTICE, "path": str(path.resolve()),
        "filename": path.name, "size_bytes": path.stat().st_size,
        "modified_timestamp_low_trust": path.stat().st_mtime_ns,
        "classification": "UNRECOGNISED",
        "reason_codes": ["NOT_ASSEMBLY_EVIDENCE"],
        "exact_name_match": path.name.lower() == ASSEMBLY_NAME,
        "approximate_size_match": (
            request.minimum_size_bytes <= path.stat().st_size
            <= request.maximum_size_bytes
        ),
        "header_fields": [], "header_truncated": False,
        "sidecar_evidence": {}, "preliminary_hash_eligible": False,
    }
    if _temporary(lower):
        base.update(
            classification="PARTIAL_OR_TEMPORARY_ARTIFACT",
            reason_codes=["PARTIAL_OR_TEMPORARY_MARKER"],
        )
        return base
    if any(marker in lower for marker in (
        "smoke", "probe", "tiny_fixture",
    )):
        base.update(
            classification="SMOKE_OR_DEVELOPMENT_ARTIFACT",
            reason_codes=["SMOKE_OR_DEVELOPMENT_PATH"],
        )
        return base
    if path.suffix.lower() in METADATA_SUFFIXES:
        base.update(_metadata_evidence(path, request.maximum_metadata_bytes))
        return base
    if path.suffix.lower() not in DATA_SUFFIXES:
        return base
    if "canonical_corpus" in path.name.lower():
        base.update(
            classification="CANONICAL_CORPUS_NOT_ASSEMBLY",
            reason_codes=["CANONICAL_OUTPUT_EXCLUDED"],
        )
        return base
    if any(provider in lower for provider in (
        "alpaca_raw", "benzinga_raw", "raw_provider", "provider_export",
    )):
        base.update(
            classification="RAW_PROVIDER_EXPORT",
            reason_codes=["SINGLE_PROVIDER_OR_RAW_EXPORT"],
        )
        return base
    if path.suffix.lower() == ".csv":
        fields, truncated = _bounded_csv_header(path)
        base["header_fields"] = fields
        base["header_truncated"] = truncated
        schema_ready = all(
            any(name in fields for name in alternatives)
            for alternatives in REQUIRED_HEADER_GROUPS
        )
    else:
        schema_ready = base["exact_name_match"]
    sidecar = _nearby_sidecar(path, request.maximum_metadata_bytes)
    base["sidecar_evidence"] = sidecar
    if "partition" in path.name.lower() or "shard" in path.name.lower():
        base.update(
            classification="HISTORICAL_SOURCE_SHARD",
            reason_codes=[] if schema_ready else ["ASSEMBLY_SCHEMA_NOT_RECOGNISED"],
        )
        return base
    if base["exact_name_match"] or (
        base["approximate_size_match"] and schema_ready
    ):
        reasons = []
        providers = {
            value.lower() for value in sidecar.get("providers", [])
        }
        expected = {
            value.lower() for value in request.expected_provider_scope
        }
        if providers and not expected.issubset(providers):
            reasons.append("EXPECTED_PROVIDER_SCOPE_MISSING")
        if not schema_ready:
            reasons.append("ASSEMBLY_SCHEMA_NOT_RECOGNISED")
        base.update(
            classification="PLAUSIBLE_ASSEMBLY_UNVERIFIED",
            reason_codes=reasons,
            preliminary_hash_eligible=not reasons,
        )
    return base


def _metadata_evidence(path, maximum_bytes):
    if path.stat().st_size > maximum_bytes:
        return {
            "classification": "UNRECOGNISED",
            "reason_codes": ["METADATA_SIZE_LIMIT_EXCEEDED"],
        }
    if path.suffix.lower() != ".json":
        return {
            "classification": "BACKFILL_INTERMEDIATE",
            "reason_codes": ["NON_CONTRACT_SIDECAR"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("not an object")
    except Exception as exc:
        return {
            "classification": "UNRECOGNISED",
            "reason_codes": ["MALFORMED_METADATA"],
            "parse_error_type": type(exc).__name__,
        }
    providers = _providers(payload)
    assembly_contract = bool(
        payload.get("assembly_checksum")
        or payload.get("source_assembly_checksum")
        or "historical_backfill" in str(payload.get("schema_version", ""))
    )
    shard_contract = bool(
        payload.get("partition_count")
        or payload.get("complete_partition_count") is not None
        or payload.get("source_files")
    )
    return {
        "classification": (
            "BACKFILL_INTERMEDIATE" if assembly_contract or shard_contract
            else "UNRECOGNISED"
        ),
        "reason_codes": [] if assembly_contract or shard_contract else [
            "CONTRACT_NOT_RECOGNISED"
        ],
        "metadata_evidence": {
            "schema_version": payload.get("schema_version"),
            "assembly_checksum": (
                payload.get("assembly_checksum")
                or payload.get("source_assembly_checksum")
            ),
            "row_count": payload.get("row_count"),
            "unique_article_count": (
                payload.get("unique_provider_article_count")
                or payload.get("unique_article_count")
            ),
            "symbol_count": payload.get("symbol_count"),
            "published_min": payload.get("min_published_at_utc"),
            "published_max": payload.get("max_published_at_utc"),
            "providers": providers,
            "complete_partition_count":
                payload.get("complete_partition_count"),
            "incomplete_partition_count":
                payload.get("incomplete_partition_count"),
            "source_file_count": (
                len(payload.get("source_files") or [])
                if isinstance(payload.get("source_files"), list) else None
            ),
            "deterministic_ordering": payload.get("ordering_policy"),
        },
    }


def _nearby_sidecar(path, maximum_bytes):
    choices = [
        path.with_suffix(".json"),
        path.parent / "stock_alpha_news_historical_corpus_assembly.json",
        path.parent / "stock_alpha_news_historical_backfill_manifest.json",
    ]
    for candidate in choices:
        if candidate.is_file() and candidate.stat().st_size <= maximum_bytes:
            evidence = _metadata_evidence(candidate, maximum_bytes)
            metadata = evidence.get("metadata_evidence") or {}
            return {
                "path": str(candidate.resolve()),
                **metadata,
            }
    return {}


def _semantic_identity_match(row, request):
    sidecar = row.get("sidecar_evidence") or {}
    expected = {value.lower() for value in request.expected_provider_scope}
    providers = {
        value.lower() for value in sidecar.get("providers", [])
    }
    return all((
        sidecar.get("row_count") == request.expected_row_count,
        sidecar.get("unique_article_count")
            == request.expected_unique_article_count,
        sidecar.get("symbol_count") == request.expected_symbol_count,
        sidecar.get("published_min") == request.expected_published_min,
        sidecar.get("published_max") == request.expected_published_max,
        expected.issubset(providers),
    ))


def _backfill_inventory(candidates, request):
    metadata = [
        row for row in candidates
        if row["classification"] == "BACKFILL_INTERMEDIATE"
        and row.get("metadata_evidence")
    ]
    shards = [
        row for row in candidates
        if row["classification"] == "HISTORICAL_SOURCE_SHARD"
    ]
    providers = sorted({
        provider for row in metadata
        for provider in row["metadata_evidence"].get("providers", [])
    })
    expected = {value.lower() for value in request.expected_provider_scope}
    represented = {value.lower() for value in providers}
    incomplete = [
        row for row in metadata
        if (row["metadata_evidence"].get("incomplete_partition_count") or 0) > 0
    ]
    deterministic = any(
        row["metadata_evidence"].get("deterministic_ordering")
        for row in metadata
    )
    complete_counts = [
        row["metadata_evidence"].get("complete_partition_count")
        for row in metadata
        if row["metadata_evidence"].get("complete_partition_count") is not None
    ]
    return {
        "notice": NOTICE,
        "metadata_candidates": len(metadata),
        "source_shard_candidates": len(shards),
        "providers": providers,
        "expected_provider_scope_complete": expected.issubset(represented),
        "incomplete_metadata_candidates": len(incomplete),
        "maximum_reported_complete_partitions":
            max(complete_counts) if complete_counts else None,
        "deterministic_ordering_evidence": deterministic,
        "all_authoritative_inputs_proven_present": bool(
            shards and metadata and not incomplete
            and expected.issubset(represented) and deterministic
        ),
        "bounded_shard_examples": [
            row["path"] for row in shards[:20]
        ],
    }


def _rebuild_readiness(exact, inventory):
    if exact:
        return {"status": "EXACT_ASSEMBLY_AVAILABLE", "blocker_codes": []}
    if inventory["all_authoritative_inputs_proven_present"]:
        return {"status": "ASSEMBLY_REBUILD_READY", "blocker_codes": []}
    if not inventory["metadata_candidates"] and not inventory[
        "source_shard_candidates"
    ]:
        return {
            "status": "EXTERNAL_RESTORATION_REQUIRED",
            "blocker_codes": ["AUTHORITATIVE_SOURCE_INPUTS_NOT_FOUND"],
        }
    blockers = []
    if not inventory["expected_provider_scope_complete"]:
        blockers.append("EXPECTED_PROVIDER_SCOPE_INCOMPLETE")
    if inventory["incomplete_metadata_candidates"]:
        blockers.append("HISTORICAL_PARTITIONS_INCOMPLETE")
    if not inventory["deterministic_ordering_evidence"]:
        blockers.append("DETERMINISTIC_ASSEMBLY_CONTRACT_NOT_PROVEN")
    if not inventory["source_shard_candidates"]:
        blockers.append("AUTHORITATIVE_SOURCE_SHARDS_NOT_FOUND")
    return {
        "status": "ASSEMBLY_REBUILD_BLOCKED_MISSING_INPUTS",
        "blocker_codes": blockers,
    }


def _publish(output_root, request, roots, candidates, checksums, selection,
             backfill, rebuild, blockers, warnings):
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "recovery_request.json": {
            "notice": NOTICE, **asdict(request),
            "request_identity": request.identity,
        },
        "root_inventory.json": {"notice": NOTICE, "roots": roots},
        "candidate_inventory.json": {
            "notice": NOTICE, "candidates": candidates,
        },
        "candidate_rejections.json": {
            "notice": NOTICE,
            "rejections": [
                row for row in candidates
                if row["classification"] != "EXACT_ASSEMBLY_CHECKSUM_MATCH"
            ],
        },
        "checksum_results.json": {
            "notice": NOTICE, "results": checksums,
        },
        "assembly_selection.json": {
            "notice": NOTICE, "selection": selection,
        },
        "historical_backfill_inventory.json": backfill,
        "rebuild_readiness.json": {"notice": NOTICE, **rebuild},
        "blockers.json": {"notice": NOTICE, "blockers": blockers},
        "warnings.json": {"notice": NOTICE, "warnings": warnings},
    }
    if selection:
        payloads["corpus_evidence_selection.json"] = selection
    for name, payload in payloads.items():
        _assert_private(payload)
        _atomic_json(output_root / name, payload)
    review = _operator_review(
        request, candidates, checksums, selection, backfill, rebuild, blockers,
    )
    _assert_private(review)
    _atomic_text(output_root / "operator_review.md", review)


def _operator_review(request, candidates, checksums, selection, backfill,
                     rebuild, blockers):
    exact_name = [
        row["path"] for row in candidates if row["exact_name_match"]
    ]
    semantic = [
        row["path"] for row in candidates
        if row["classification"] == "ASSEMBLY_IDENTITY_MATCH_WITHOUT_BYTE_MATCH"
    ]
    return "\n".join([
        "# Historical News Assembly Recovery Review", "", NOTICE, "",
        f"1. Exact checksum found: `{selection is not None}`",
        f"2. Exact-name candidates: `{len(exact_name)}`",
        f"3. Semantic non-byte matches: `{len(semantic)}`",
        "4. Approximate 317 MB candidates are listed in candidate_inventory.json.",
        f"5. Historical source shards found: "
        f"`{backfill['source_shard_candidates']}`",
        f"6. Alpaca/Benzinga scope complete: "
        f"`{backfill['expected_provider_scope_complete']}`",
        f"7. Rebuild classification: `{rebuild['status']}`",
        "8. Byte-identical reconstruction is not claimed without exact source "
        "bytes and deterministic ordering evidence.",
        f"9. External restoration required: "
        f"`{rebuild['status'] == 'EXTERNAL_RESTORATION_REQUIRED'}`", "",
        "## Next safe action", "",
        "On the other device or backup, search for the exact filename, calculate "
        f"SHA-256 `{request.target_checksum}`, locate its sidecars, and record "
        "size and path. Do not alter the source. If transferred, verify the "
        "checksum and place it in a new isolated recovery directory; do not "
        "overwrite the incomplete canonical corpus. Rerun 1G, then 1F.", "",
        "## Integrity", "",
        f"- Full hashes calculated: `{len(checksums)}`",
        f"- Blockers: `{[row['code'] for row in blockers]}`",
        "- No searched file was opened for writing.",
        "- No production data was copied, transformed, or materialised.",
        "- No network, model, lease, ledger, or registry operation occurred.",
        "- The news-transformer trainer remains out of scope.", "",
    ])


def _bounded_csv_header(path, limit=65_536):
    with path.open("rb") as handle:
        raw = handle.readline(limit + 1)
    truncated = len(raw) > limit
    text = raw[:limit].decode("utf-8-sig", errors="replace").strip()
    return [field.strip().strip('"') for field in text.split(",")], truncated


def _providers(payload):
    explicit = payload.get("providers") or payload.get("provider_inventory")
    values = set()
    if isinstance(explicit, list):
        values.update(str(value) for value in explicit)
    distribution = payload.get("source_distribution") or {}
    if isinstance(distribution, Mapping):
        values.update(str(value) for value in distribution)
    if "alpaca_benzinga" in str(payload.get("schema_version", "")).lower():
        values.update(("Alpaca", "Benzinga"))
    return sorted(values)


def _pattern_match(name, pattern):
    return fnmatch.fnmatchcase(name, pattern)


def _temporary(value):
    return any(marker in str(value).lower() for marker in (
        ".tmp", ".partial", ".incomplete", "~",
    ))


def _snapshot(path):
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_timestamp_ns": stat.st_mtime_ns,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_private(value):
    forbidden = {
        "headline", "summary", "body", "body_or_full_text",
        "raw_provider_payload", "api_key", "token", "secret",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise ValueError("Private content is prohibited")
            _assert_private(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_private(child)
    elif isinstance(value, str) and len(value) > 32_768:
        raise ValueError("Unbounded evidence string")


def _identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def _atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
