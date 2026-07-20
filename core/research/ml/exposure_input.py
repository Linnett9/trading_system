from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.immutable_runs import file_digest

SELECTOR_DERIVED_SOURCE_TYPE = "selector_derived_replay_v1"
LEGACY_EXPANDED_REBALANCE_SOURCE_TYPE = "legacy_expanded_rebalance_v1"
EXPOSURE_INPUT_CONTRACT_VERSION = 1


def exposure_input_source_type(ml_config: Mapping[str, Any]) -> str:
    configured = str(ml_config.get("exposure_input_source_type", "")).strip()
    if configured:
        return configured
    if str(ml_config.get("feature_set", "")).strip() == "selector_derived_rebalance_v1":
        return SELECTOR_DERIVED_SOURCE_TYPE
    if ml_config.get("stock_selector_rebalance_dataset_path"):
        return SELECTOR_DERIVED_SOURCE_TYPE
    return LEGACY_EXPANDED_REBALANCE_SOURCE_TYPE


def exposure_model_input_source_path(config: Mapping[str, Any]) -> str:
    ml = config.get("ml", {}) or {}
    source_type = exposure_input_source_type(ml)
    if source_type == SELECTOR_DERIVED_SOURCE_TYPE:
        path = ml.get("stock_selector_rebalance_dataset_path")
        if path is None:
            path = (
                Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
                / "stock_selector_rebalance_dataset.csv"
            )
        return str(Path(str(path)).resolve())
    path = ml.get("expanded_rebalance_dataset_path")
    if path is None:
        path = (
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv"
        )
    return str(Path(str(path)).resolve())


def validate_exposure_input_resolution(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = config.get("ml", {}) or {}
    source_type = exposure_input_source_type(ml)
    production = bool(ml.get("exposure_production_campaign", False))
    legacy_label = bool(ml.get("legacy_research_exposure_input", False))
    path = Path(exposure_model_input_source_path(config))
    if source_type == SELECTOR_DERIVED_SOURCE_TYPE:
        if "expanded_rebalance_dataset.csv" in path.name:
            raise RuntimeError(
                "Selector-derived exposure production cannot resolve legacy "
                f"expanded-rebalance input: {path}"
            )
        return validate_selector_derived_exposure_input(path)
    if production:
        raise RuntimeError(
            "Production exposure campaigns must use selector-derived exposure input; "
            f"resolved legacy source={path}"
        )
    if not legacy_label:
        raise RuntimeError(
            "Legacy expanded-rebalance exposure input must be explicitly labelled "
            "with ml.legacy_research_exposure_input=true"
        )
    return {
        "source_type": LEGACY_EXPANDED_REBALANCE_SOURCE_TYPE,
        "contract_version": EXPOSURE_INPUT_CONTRACT_VERSION,
        "dataset_path": str(path),
        "legacy_research_exposure_input": True,
    }


def validate_selector_derived_exposure_input(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    rows = _read_csv(path)
    if not rows:
        raise RuntimeError(f"Selector-derived exposure dataset has no rows: {path}")
    required = {
        "feature_id",
        "feature_date",
        "rebalance_date",
        "label_start_date",
        "label_end_date",
        "label_available_timestamp",
        "outcome_end_date",
        "selector_signal",
        "portfolio_policy",
        "strategy_id",
        "selected_symbols",
        "selected_weights",
        "portfolio_return_next_period",
        "benchmark_return_next_period",
        "transaction_cost_drag",
        "should_reduce_exposure",
        "dataset_hash",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(
            "Selector-derived exposure dataset is missing required columns: "
            f"{missing}; path={path}"
        )
    forbidden = {"dual_momentum_strategy", "strategy_name"}
    if forbidden & set(rows[0]):
        raise RuntimeError(f"Rejected legacy exposure dataset source={path}")
    _validate_selector_rows(rows, path)
    metadata = _read_json(metadata_path)
    if metadata.get("source_type") != SELECTOR_DERIVED_SOURCE_TYPE:
        raise RuntimeError("Selector-derived exposure metadata has incompatible source_type")
    contract = metadata.get("input_source_contract") or {}
    if contract.get("contract_version") != EXPOSURE_INPUT_CONTRACT_VERSION:
        raise RuntimeError("Selector-derived exposure input contract version is incompatible")
    if metadata.get("dataset_hash") != rows[0].get("dataset_hash"):
        raise RuntimeError("Selector-derived exposure dataset_hash does not match metadata")
    if metadata.get("dataset_hash") != contract.get("dataset_identity"):
        raise RuntimeError("Selector-derived exposure dataset identity does not match contract")
    return {
        "source_type": SELECTOR_DERIVED_SOURCE_TYPE,
        "contract_version": EXPOSURE_INPUT_CONTRACT_VERSION,
        "dataset_path": str(path),
        "metadata_path": str(metadata_path),
        "dataset_sha256": file_digest(path),
        "metadata_sha256": file_digest(metadata_path),
        "input_source_contract": contract,
    }


def selector_exposure_input_contract(
    *,
    rows: list[dict[str, Any]],
    predictions_path: Path,
    summary_path: Path,
    equity_curves_path: Path,
    holdings_path: Path,
    selected_signal: str,
    selected_policy: str,
    strategy_id: str,
    source_dataset_hash: str,
    cost_bps: Any = None,
    slippage_bps: Any = None,
) -> dict[str, Any]:
    holdings_identity = _holdings_and_weights_identity(rows)
    contract = {
        "contract_version": EXPOSURE_INPUT_CONTRACT_VERSION,
        "source_type": SELECTOR_DERIVED_SOURCE_TYPE,
        "selector_dataset_identity": source_dataset_hash,
        "selector_model_identity": selected_signal,
        "selector_prediction_identity": file_digest(predictions_path),
        "strict_oos_fold_identity": _strict_oos_fold_identity(predictions_path),
        "replay_identity": file_digest(equity_curves_path),
        "portfolio_policy_identity": selected_policy,
        "holdings_and_weights_checksum": _hash_payload(holdings_identity),
        "return_lineage": {
            "source": str(equity_curves_path),
            "field": "net_return",
            "sha256": file_digest(equity_curves_path),
        },
        "benchmark_lineage": {
            "source": str(equity_curves_path),
            "field": "benchmark_return",
            "sha256": file_digest(equity_curves_path),
        },
        "cost_and_slippage_policy": {
            "source": str(summary_path),
            "transaction_cost_drag_field": "transaction_cost_drag",
            "configured_cost_bps": cost_bps,
            "configured_slippage_bps": slippage_bps,
        },
        "outcome_start_timestamp": min(row["label_start_date"] for row in rows),
        "outcome_end_timestamp": max(row["label_end_date"] for row in rows),
        "outcome_availability_timestamp": max(
            row["label_available_timestamp"] for row in rows
        ),
        "exposure_label_contract": "should_reduce_exposure",
        "strategy_id": strategy_id,
        "selected_signal": selected_signal,
        "selected_policy": selected_policy,
        "source_artifacts": {
            "predictions_path": str(predictions_path),
            "summary_path": str(summary_path),
            "equity_curves_path": str(equity_curves_path),
            "holdings_path": str(holdings_path),
        },
    }
    contract["dataset_identity"] = _hash_payload(
        {
            "source_type": SELECTOR_DERIVED_SOURCE_TYPE,
            "rows": rows,
            "parents": contract["source_artifacts"],
            "lineage": {
                key: contract[key]
                for key in (
                    "selector_dataset_identity",
                    "selector_model_identity",
                    "selector_prediction_identity",
                    "strict_oos_fold_identity",
                    "replay_identity",
                    "portfolio_policy_identity",
                    "holdings_and_weights_checksum",
                    "return_lineage",
                    "benchmark_lineage",
                    "cost_and_slippage_policy",
                    "exposure_label_contract",
                )
            },
        }
    )
    return contract


def _validate_selector_rows(rows: list[dict[str, Any]], path: Path) -> None:
    dataset_hashes = {str(row.get("dataset_hash", "")) for row in rows}
    if len(dataset_hashes) != 1 or "" in dataset_hashes:
        raise RuntimeError(f"Selector-derived exposure dataset has invalid dataset_hash: {path}")
    feature_ids: set[str] = set()
    for row in rows:
        if row["feature_id"] in feature_ids:
            raise RuntimeError(f"Duplicate selector-derived exposure feature_id: {row['feature_id']}")
        feature_ids.add(row["feature_id"])
        if not str(row.get("label_start_date")) <= str(row.get("label_end_date")):
            raise RuntimeError("Selector-derived exposure outcome timestamps are invalid")
        if not str(row.get("feature_date")) < str(row.get("label_start_date")):
            raise RuntimeError("Selector-derived exposure row leaks outcome into features")
        if not str(row.get("label_end_date")) <= str(row.get("label_available_timestamp")):
            raise RuntimeError("Selector-derived exposure outcome is not mature")
        for numeric in (
            "portfolio_return_next_period",
            "benchmark_return_next_period",
            "transaction_cost_drag",
        ):
            float(row[numeric])
        weights = json.loads(str(row["selected_weights"]))
        if sorted(weights) != [symbol for symbol in str(row["selected_symbols"]).split(",") if symbol]:
            raise RuntimeError("Selector-derived exposure holdings and weights mismatch")
        gross = sum(abs(float(value)) for value in weights.values())
        if gross > 1.000001:
            raise RuntimeError("Selector-derived exposure holdings exceed gross exposure bounds")


def _strict_oos_fold_identity(predictions_path: Path) -> dict[str, Any]:
    rows = _read_csv(predictions_path)
    if not rows:
        raise RuntimeError(f"Selector prediction artifact has no rows: {predictions_path}")
    if any(str(row.get("final_fit", "")).lower() == "true" for row in rows):
        raise RuntimeError("Final-fit selector predictions cannot construct historical exposure rows")
    fold_ids = sorted({str(row.get("fold_id", "")).strip() for row in rows})
    if "" in fold_ids:
        raise RuntimeError("Strict-OOS selector predictions require fold_id on every row")
    return {
        "fold_ids": fold_ids,
        "fold_count": len(fold_ids),
        "prediction_row_count": len(rows),
        "prediction_artifact_sha256": file_digest(predictions_path),
    }


def _holdings_and_weights_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rebalance_date": row["rebalance_date"],
            "selected_symbols": row["selected_symbols"],
            "selected_weights": row["selected_weights"],
        }
        for row in sorted(rows, key=lambda item: str(item["rebalance_date"]))
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
