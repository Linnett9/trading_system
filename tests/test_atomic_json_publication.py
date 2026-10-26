from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.research.artifact_publication import (
    ATOMIC_JSON_PUBLICATION_CONTRACT_VERSION,
    JsonByteFormat,
    atomic_write_json,
)
from core.research.ml import ticket63_atomic_json


def test_atomic_json_publication_writes_exact_default_bytes_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    payload = {"z": "omega", "a": 1}

    result = atomic_write_json(path, payload)

    expected = b'{\n  "a": 1,\n  "z": "omega"\n}'
    assert path.read_bytes() == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == hashlib.sha256(expected).hexdigest().upper()
    assert result["contract_version"] == ATOMIC_JSON_PUBLICATION_CONTRACT_VERSION
    assert result["status"] == "PASS"
    assert result["attempts"] == 1


def test_atomic_json_publication_creates_parent_replaces_and_cleans_temp(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "status.json"
    atomic_write_json(path, {"version": 1})

    atomic_write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob(".*.tmp"))


def test_atomic_json_publication_cleans_temp_and_keeps_destination_after_non_transient_replace_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "status.json"
    path.write_text('{"version": 1}', encoding="utf-8")

    def fail_replace(_src: Path, _dst: Path) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        atomic_write_json(path, {"version": 2}, replace_fn=fail_replace)

    assert path.read_text(encoding="utf-8") == '{"version": 1}'
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_atomic_json_publication_cleans_no_temp_after_serialization_failure(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json(path, {"bad": float("nan")})

    assert not path.exists()
    assert not list(tmp_path.glob(".invalid.json.*.tmp"))


def test_atomic_json_publication_supports_explicit_unicode_and_format_options(tmp_path: Path) -> None:
    path = tmp_path / "unicode.json"
    json_format = JsonByteFormat(
        indent=None,
        sort_keys=True,
        default=None,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        trailing_newline=True,
        format_id="compact_utf8_with_newline.test",
    )

    result = atomic_write_json(path, {"word": "cafe", "symbol": "GBP", "name": "José"}, json_format=json_format)

    assert path.read_bytes() == '{"name":"José","symbol":"GBP","word":"cafe"}\n'.encode("utf-8")
    assert result["json_format"] == "compact_utf8_with_newline.test"


def test_atomic_json_publication_supports_legacy_temporary_name_adapter(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    atomic_write_json(
        path,
        {"status": "PASS"},
        temporary_path_fn=lambda final_path: final_path.with_name(f".{final_path.name}.legacy.tmp"),
    )

    assert path.exists()
    assert not (tmp_path / ".manifest.json.legacy.tmp").exists()


def test_ticket63_compatibility_import_preserves_contract_versions(tmp_path: Path) -> None:
    path = tmp_path / "status.json"

    result = ticket63_atomic_json.atomic_write_json(path, {"status": "ok"})
    cleanup = ticket63_atomic_json.cleanup_stale_atomic_json_temps(tmp_path, target_name="status.json", dry_run=True)
    contract = ticket63_atomic_json.atomic_writer_contract()

    assert result["contract_version"] == "ticket63_windows_atomic_json_writer.v1"
    assert cleanup["contract_version"] == "ticket63_stale_atomic_temp_cleanup.v1"
    assert contract["contract_version"] == "ticket63_windows_atomic_json_writer.v1"
