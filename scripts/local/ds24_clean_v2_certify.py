from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_certification import (
    certification_cases,
    source_fixture_summary,
    synthetic_multiyear_raw_frames,
)
from core.research.ml.ds24.clean_v2_contracts import (
    authority_bundle,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_features import (
    REPAIRED_FEATURES,
    SOURCE_TIMESTAMP_SUFFIX,
    certify_future_bar_invariance,
    compute_repaired_session_features,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _logical_hash_valid(payload: dict[str, Any]) -> bool:
    return payload.get("logical_sha256") == stable_hash(
        {key: value for key, value in payload.items() if key != "logical_sha256"}
    )


def _safe_partition(root: Path, relative_value: Any) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe manifest partition path: {relative}")
    return root / relative


def _inventory_valid(
    root: Path,
    rows: Iterable[dict[str, Any]],
    *,
    path_key: str,
    bytes_key: str,
) -> bool:
    for row in rows:
        try:
            path = _safe_partition(root, row[path_key])
            if not path.is_file() or path.stat().st_size != int(row[bytes_key]):
                return False
        except (KeyError, OSError, TypeError, ValueError):
            return False
    return True


def _synthetic_semantic_checks(
    frames: dict[str, pd.DataFrame],
) -> dict[str, bool]:
    frame = frames["AAA"]
    repaired = compute_repaired_session_features(frame)
    sessions = frame["session_date"].drop_duplicates().tolist()
    rth = frame["session_type"].isin(["REGULAR", "EARLY_CLOSE"])
    current_indices = frame.index[
        (frame["session_date"] == sessions[2]) & rth
    ]
    prior = frame[(frame["session_date"] == sessions[1]) & rth]
    two_back = frame[(frame["session_date"] == sessions[0]) & rth]
    row_index = current_indices[10]
    current = frame.loc[row_index]
    formula_checks = [
        np.isclose(
            repaired.loc[row_index, "overnight_gap"],
            frame.loc[current_indices[0], "open"] / prior.iloc[-1]["close"] - 1.0,
        ),
        np.isclose(
            repaired.loc[row_index, "previous_session_return"],
            prior.iloc[-1]["close"] / prior.iloc[0]["open"] - 1.0,
        ),
        np.isclose(
            repaired.loc[row_index, "two_session_return"],
            current["close"] / two_back.iloc[-1]["close"] - 1.0,
        ),
    ]
    extended = frame["session_type"].isin(["PRE_MARKET", "AFTER_HOURS"])
    rth_semantics = bool(
        repaired.loc[extended, list(REPAIRED_FEATURES)].isna().all().all()
    )
    normal = frame[
        (frame["session_date"] == "2017-11-22")
        & (frame["session_type"] == "REGULAR")
    ]
    normal_repaired = repaired.loc[normal.index]
    calendar_semantics = bool(
        np.isclose(normal_repaired.iloc[0]["minutes_since_open"], 5.0)
        and np.isclose(normal_repaired.iloc[0]["minutes_until_close"], 385.0)
        and np.isclose(normal_repaired.iloc[-1]["minutes_until_close"], 0.0)
        and np.isclose(normal_repaired.iloc[-1]["session_progress"], 1.0)
    )
    return {
        "independent_formula_checks_passed": all(formula_checks),
        "rth_extended_hours_semantics_passed": rth_semantics,
        "calendar_state_semantics_passed": calendar_semantics,
    }


def _materialized_inventory_checks() -> dict[str, Any]:
    feature = load_contract("feature_authority.json")
    target = load_contract("target_contract.json")
    sidecar_root = ROOT / feature["sidecar"]["path"]
    base_feature_root = ROOT / feature["base_authority"]["path"]
    target_root = ROOT / target["physical_authority"]["delta_path"]
    base_target_root = ROOT / target["physical_authority"]["base_path"]
    sidecar = _read_json(sidecar_root / "authority_manifest.json")
    target_delta = _read_json(target_root / "authority_manifest.json")
    sidecar_rows = sidecar.get("partitions", [])
    target_rows = target_delta.get("partitions", [])
    base_target_rows = (target_delta.get("base_snapshot") or {}).get(
        "partitions", []
    )
    checks = {
        "sidecar_complete": sidecar.get("complete") is True,
        "sidecar_partition_count": len(sidecar_rows)
        == int(feature["base_authority"]["physical_snapshot_stock_partitions"]),
        "sidecar_row_count": int(sidecar.get("row_count", -1))
        == int(feature["base_authority"]["physical_snapshot_row_count"]),
        "sidecar_self_hash": _logical_hash_valid(sidecar),
        "sidecar_file_inventory": _inventory_valid(
            sidecar_root,
            sidecar_rows,
            path_key="relative_path",
            bytes_key="bytes",
        ),
        "base_feature_file_inventory": _inventory_valid(
            base_feature_root,
            sidecar_rows,
            path_key="base_relative_path",
            bytes_key="base_bytes",
        ),
        "target_delta_complete": target_delta.get("complete") is True,
        "target_delta_symbol_count": len(target_rows)
        == int(feature["base_authority"]["physical_snapshot_asset_count"]),
        "target_delta_self_hash": _logical_hash_valid(target_delta),
        "target_delta_file_inventory": _inventory_valid(
            target_root,
            target_rows,
            path_key="relative_path",
            bytes_key="bytes",
        ),
        "base_target_file_inventory": _inventory_valid(
            base_target_root,
            base_target_rows,
            path_key="base_relative_path",
            bytes_key="base_bytes",
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sidecar_manifest_logical_sha256": sidecar.get("logical_sha256"),
        "target_manifest_logical_sha256": target_delta.get("logical_sha256"),
        "sidecar_partitions": sidecar_rows,
        "target_delta_partitions": target_rows,
        "base_target_partitions": base_target_rows,
        "sidecar_root": sidecar_root,
        "target_root": target_root,
        "base_target_root": base_target_root,
    }


def _full_sidecar_scan(inventory: dict[str, Any]) -> dict[str, Any]:
    row_count = 0
    future_source_rows = 0
    partitions = inventory["sidecar_partitions"]
    columns = [
        "decision_timestamp",
        *[f"{name}{SOURCE_TIMESTAMP_SUFFIX}" for name in REPAIRED_FEATURES],
    ]
    for index, row in enumerate(partitions, start=1):
        path = _safe_partition(inventory["sidecar_root"], row["relative_path"])
        frame = pd.read_parquet(path, columns=columns)
        decision = pd.to_datetime(frame["decision_timestamp"], utc=True)
        row_count += len(frame)
        for column in columns[1:]:
            source = pd.to_datetime(frame[column], utc=True, errors="coerce")
            future_source_rows += int((source.notna() & (source > decision)).sum())
        if index % 100 == 0:
            print(
                f"sidecar certification {index}/{len(partitions)} partitions",
                file=sys.stderr,
                flush=True,
            )
    return {
        "partition_count": len(partitions),
        "row_count": row_count,
        "future_source_rows": future_source_rows,
        "passed": (
            row_count
            == int(
                load_contract("feature_authority.json")["base_authority"][
                    "physical_snapshot_row_count"
                ]
            )
            and future_source_rows == 0
        ),
    }


def _full_target_scan(inventory: dict[str, Any]) -> dict[str, Any]:
    target = load_contract("target_contract.json")
    rows = [
        *[
            (
                inventory["base_target_root"],
                row["base_relative_path"],
            )
            for row in inventory["base_target_partitions"]
        ],
        *[
            (inventory["target_root"], row["relative_path"])
            for row in inventory["target_delta_partitions"]
        ],
    ]
    scanned = 0
    invalid_maturity = 0
    invalid_trainable = 0
    wrong_target = 0
    for index, (root, relative) in enumerate(rows, start=1):
        path = _safe_partition(root, relative)
        frame = pd.read_parquet(
            path,
            columns=[
                "decision_timestamp",
                "target_available_timestamp",
                "target_id",
                "target_value",
                "target_is_trainable",
            ],
        )
        decision = pd.to_datetime(frame["decision_timestamp"], utc=True)
        maturity = pd.to_datetime(
            frame["target_available_timestamp"], utc=True, errors="coerce"
        )
        trainable = frame["target_is_trainable"].fillna(False).astype(bool)
        scanned += len(frame)
        invalid_maturity += int((maturity.notna() & (maturity <= decision)).sum())
        invalid_trainable += int(
            (
                trainable
                & (
                    maturity.isna()
                    | ~np.isfinite(pd.to_numeric(frame["target_value"], errors="coerce"))
                )
            ).sum()
        )
        wrong_target += int(
            (frame["target_id"].astype(str) != target["target_id"]).sum()
        )
        if index % 100 == 0:
            print(
                f"target certification {index}/{len(rows)} partitions",
                file=sys.stderr,
                flush=True,
            )
    return {
        "partition_count": len(rows),
        "row_count": scanned,
        "maturity_not_after_decision_rows": invalid_maturity,
        "invalid_trainable_rows": invalid_trainable,
        "wrong_target_identity_rows": wrong_target,
        "passed": (
            invalid_maturity == 0
            and invalid_trainable == 0
            and wrong_target == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Certify DS24 clean-V2 causality and materialized authorities."
    )
    parser.add_argument(
        "--output",
        default=(
            "research_runs/ds24_clean_v2/"
            "DS24_CLEAN_V2_TOURNAMENT_R1_20260926/causality_certificate.json"
        ),
    )
    parser.add_argument(
        "--full-materialized-authority",
        action="store_true",
        help="Scan every sidecar and target partition; intended for the user terminal.",
    )
    args = parser.parse_args()
    frames = synthetic_multiyear_raw_frames()
    certificate = certify_future_bar_invariance(
        frames, certification_cases(frames)
    )
    certificate["fixture"] = source_fixture_summary(frames)
    certificate.update(_synthetic_semantic_checks(frames))
    certificate["feature_authority_id"] = load_contract("feature_authority.json")[
        "authority_id"
    ]
    certificate["static_authority_bundle_sha256"] = authority_bundle()[
        "bundle_sha256"
    ]
    predictors = load_contract("predictor_manifest.json")["predictors"]
    certificate["zero_target_columns_in_model_inputs"] = not any(
        "target" in str(name).lower() or "forward_return" in str(name).lower()
        for name in predictors
    )
    inventory = _materialized_inventory_checks()
    certificate["physical_inventory_checks"] = {
        "passed": inventory["passed"],
        "checks": inventory["checks"],
        "sidecar_manifest_logical_sha256": inventory[
            "sidecar_manifest_logical_sha256"
        ],
        "target_manifest_logical_sha256": inventory[
            "target_manifest_logical_sha256"
        ],
    }
    if args.full_materialized_authority:
        certificate["full_sidecar_scan"] = _full_sidecar_scan(inventory)
        certificate["full_target_scan"] = _full_target_scan(inventory)
        certificate["certification_scope"] = "FULL_MATERIALIZED_AUTHORITY"
    else:
        certificate["full_sidecar_scan"] = {"passed": False, "status": "NOT_RUN"}
        certificate["full_target_scan"] = {"passed": False, "status": "NOT_RUN"}
        certificate["certification_scope"] = "BOUNDED_SYNTHETIC_ONLY"
    certificate["target_maturity_checks_passed"] = bool(
        certificate["full_target_scan"]["passed"]
    )
    certificate["passed"] = bool(
        certificate.get("passed")
        and certificate["independent_formula_checks_passed"]
        and certificate["rth_extended_hours_semantics_passed"]
        and certificate["calendar_state_semantics_passed"]
        and certificate["zero_target_columns_in_model_inputs"]
        and certificate["physical_inventory_checks"]["passed"]
        and certificate["full_sidecar_scan"]["passed"]
        and certificate["full_target_scan"]["passed"]
    )
    certificate["certificate_sha256"] = stable_hash(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_sha256"
        }
    )
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
