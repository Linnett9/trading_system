from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.research.ml.stock_level.selector_feature_schema import load_feature_schema
from core.research.ml.registries.types import (
    COMMON_REQUIRED_FIELDS, IMPLEMENTATION_STATUSES, RegistryDocument,
    RegistryEntry, RegistryResolution, RegistryValidationError,
)


DEFAULT_REGISTRY_ROOT = Path("config/ml_registries")
REGISTRY_FILES = {
    "equations": "equations.v1.json",
    "indicators": "indicators.v1.json",
    "selector_models": "selector_models.v1.json",
    "exposure": "exposure.v1.json",
    "portfolio_policies": "portfolio_policies.v1.json",
    "target_contracts": "target_contracts.v1.json",
    "ranking_contracts": "ranking_contracts.v1.json",
}


def canonical_json_bytes(payload: Any) -> bytes:
    """Strict, stable JSON: unsupported values fail instead of stringifying."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value for key, value in payload.items()
        if key not in {"registry_file_hash", "entry_hash", "generated_at", "updated_at"}
    }
    if "aliases" in result:
        aliases = result["aliases"]
        if not isinstance(aliases, list) or not all(isinstance(value, str) for value in aliases):
            raise RegistryValidationError("aliases must be a list of strings")
        result["aliases"] = sorted(set(aliases))
    return result


def entry_hash(payload: Mapping[str, Any]) -> str:
    return canonical_hash(_identity_payload(payload))


def registry_hash(payload: Mapping[str, Any]) -> str:
    normalized = _identity_payload(payload)
    entries = normalized.get("entries", [])
    if isinstance(entries, list):
        normalized["entries"] = sorted(
            (_identity_payload(entry) for entry in entries),
            key=lambda entry: str(entry.get("canonical_id", "")),
        )
    return canonical_hash(normalized)


def _validate_common(entry: Mapping[str, Any], kind: str) -> None:
    missing = [field for field in COMMON_REQUIRED_FIELDS if field not in entry]
    if missing:
        raise RegistryValidationError(
            f"Registry entry {entry.get('canonical_id', '<unknown>')} missing required fields: {missing}"
        )
    canonical_id = entry["canonical_id"]
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        raise RegistryValidationError("canonical_id must be a non-empty string")
    if entry["implementation_status"] not in IMPLEMENTATION_STATUSES:
        raise RegistryValidationError(
            f"Invalid implementation status for {canonical_id}: {entry['implementation_status']}"
        )
    if entry["registry_contract_version"] != "ml_registry_entry_v1":
        raise RegistryValidationError(f"Unsupported entry contract for {canonical_id}")
    if not isinstance(entry["research_only"], bool):
        raise RegistryValidationError(f"research_only must be boolean for {canonical_id}")
    if kind == "equations":
        for field in ("description", "input_fields", "output_fields", "availability_rule", "consumers"):
            if field not in entry:
                raise RegistryValidationError(f"Equation {canonical_id} missing {field}")
    if kind == "indicators":
        for field in ("source_fields", "source_dataset", "lookback", "calculation_equation", "available_timestamp_rule", "point_in_time_safe", "allowed_roles", "feature_schema_membership"):
            if field not in entry:
                raise RegistryValidationError(f"Indicator {canonical_id} missing {field}")
    if kind == "target_contracts":
        for field in ("field_name", "horizon", "target_start_rule", "target_end_rule", "label_available_timestamp_field", "label_available_rule", "target_calculation_owner", "target_schema_owner", "allowed_roles", "unit", "data_type", "ranking_objective_contract"):
            if field not in entry:
                raise RegistryValidationError(f"Target contract {canonical_id} missing {field}")
        if not str(entry["horizon"]).strip():
            raise RegistryValidationError(f"Target contract {canonical_id} has an empty horizon")
        if entry["label_available_timestamp_field"] != "label_available_timestamp":
            raise RegistryValidationError(f"Target contract {canonical_id} must use label_available_timestamp")


def load_registry(path: Path, *, expected_kind: str | None = None) -> RegistryDocument:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryValidationError(f"Cannot load registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryValidationError(f"Registry must be an object: {path}")
    kind = payload.get("registry_kind")
    if expected_kind and kind != expected_kind:
        raise RegistryValidationError(f"Expected registry kind {expected_kind}, found {kind}")
    if payload.get("registry_contract_version") != "ml_registry_document_v1":
        raise RegistryValidationError(f"Unsupported registry document contract: {path}")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise RegistryValidationError(f"Registry entries must be a list: {path}")
    seen: set[str] = set()
    aliases: dict[str, str] = {}
    entries: list[RegistryEntry] = []
    defaults = payload.get("entry_defaults", {})
    if not isinstance(defaults, dict):
        raise RegistryValidationError(f"entry_defaults must be an object: {path}")
    for raw_source in raw_entries:
        raw = {**defaults, **raw_source} if isinstance(raw_source, dict) else raw_source
        if isinstance(raw, dict) and kind == "indicators":
            raw.setdefault("feature_schema", None)
        if not isinstance(raw, dict):
            raise RegistryValidationError(f"Registry entry must be an object: {path}")
        _validate_common(raw, str(kind))
        canonical_id = str(raw["canonical_id"])
        if canonical_id in seen:
            raise RegistryValidationError(f"Duplicate canonical ID: {canonical_id}")
        seen.add(canonical_id)
        for alias in raw["aliases"]:
            if alias == canonical_id or alias in aliases or alias in seen:
                raise RegistryValidationError(f"Alias collision: {alias}")
            aliases[alias] = canonical_id
        entries.append(RegistryEntry(str(kind), path, raw, entry_hash(raw)))
    canonical_entries = tuple(sorted(entries, key=lambda entry: entry.canonical_id))
    return RegistryDocument(str(kind), path, str(payload["registry_contract_version"]), canonical_entries, registry_hash(payload))


@dataclass(frozen=True)
class RegistryBundle:
    documents: Mapping[str, RegistryDocument]
    registry_set_hash: str


class RegistryResolver:
    def __init__(self, bundle: RegistryBundle):
        self.bundle = bundle
        self._canonical: dict[tuple[str, str], RegistryEntry] = {}
        self._aliases: dict[tuple[str, str], str] = {}
        for kind, document in bundle.documents.items():
            for entry in document.entries:
                self._canonical[(kind, entry.canonical_id)] = entry
                for alias in entry.aliases:
                    key = (kind, alias)
                    if key in self._aliases or (kind, alias) in self._canonical:
                        raise RegistryValidationError(f"Ambiguous alias: {kind}:{alias}")
                    self._aliases[key] = entry.canonical_id

    def resolve(self, kind: str, requested_id: str, *, role: str | None = None) -> RegistryResolution:
        canonical_id = self._aliases.get((kind, requested_id), requested_id)
        entry = self._canonical.get((kind, canonical_id))
        if entry is None:
            raise KeyError(f"Unknown {kind} registry ID: {requested_id}")
        roles = entry.payload.get("allowed_roles") or [entry.payload.get("model_role")]
        if role and role not in roles:
            raise RegistryValidationError(
                f"Registry ID {requested_id} is incompatible with role {role}; allowed={roles}"
            )
        document = self.bundle.documents[kind]
        return RegistryResolution(requested_id, canonical_id, entry, document.registry_hash, self.bundle.registry_set_hash)

    def verify_feature_schema(self, reference: str | None) -> None:
        if not reference:
            return
        path = Path(reference)
        if not path.exists():
            raise RegistryValidationError(f"Unknown feature schema reference: {reference}")
        load_feature_schema(path)

    def verify_target_contract(self, reference: str | None) -> None:
        if not reference or reference in {"stock_selector_trailing_signals_v1", "should_reduce_exposure", "risk_regime", "drawdown_risk", "champion_success"}:
            return
        try:
            self.resolve("target_contracts", reference)
        except KeyError as exc:
            raise RegistryValidationError(f"Unknown target contract reference: {reference}") from exc

    def target_for_field(self, field_name: str, *, role: str) -> RegistryResolution:
        matches = [entry for entry in self.bundle.documents["target_contracts"].entries if entry.payload.get("field_name") == field_name]
        if len(matches) != 1:
            raise RegistryValidationError(f"Expected one target contract for field {field_name}; found {len(matches)}")
        return self.resolve("target_contracts", matches[0].canonical_id, role=role)

    def validate_references(self) -> None:
        equations = {entry.canonical_id for entry in self.bundle.documents["equations"].entries}
        for document in self.bundle.documents.values():
            for entry in document.entries:
                self.verify_feature_schema(entry.payload.get("feature_schema"))
                self.verify_target_contract(entry.payload.get("target_contract"))
                equation = entry.payload.get("calculation_equation")
                if equation and equation not in equations:
                    raise RegistryValidationError(
                        f"Unknown equation {equation} referenced by {entry.canonical_id}"
                    )


def load_registry_bundle(root: Path = DEFAULT_REGISTRY_ROOT) -> RegistryBundle:
    documents = {
        kind: load_registry(root / filename, expected_kind=kind)
        for kind, filename in REGISTRY_FILES.items()
    }
    set_payload = {
        "registry_contract_version": "ml_registry_set_v1",
        "registries": {kind: document.registry_hash for kind, document in sorted(documents.items())},
    }
    bundle = RegistryBundle(documents, canonical_hash(set_payload))
    RegistryResolver(bundle).validate_references()
    return bundle
