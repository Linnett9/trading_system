from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.target_authority import (
    HALT_AFFECTED,
    INELIGIBLE_SOURCE_BAR,
    MATURED_VALID,
    MISSING_SOURCE_BAR,
    MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION,
    NOT_YET_MATURE,
    QUARANTINED_SOURCE_BAR,
    RIGHT_CENSORED,
    SESSION_BOUNDARY_CONFLICT,
    TARGET_CATALOGUE_VERSION,
    TARGET_RESOLUTION_POLICY_VERSION,
    TARGET_RESOLUTION_STATES,
    UNKNOWN_SOURCE_GAP,
    build_target_manifest,
    calculate_target,
    canonical_hash,
    resolve_target_contract,
    target_catalogue_payload,
    target_code_hash,
    validate_target_availability,
)
from infrastructure.data.calendar_authority import default_calendar_authority
from infrastructure.data.market_sessions import trading_sessions


ASSET = "asset-AAA"
SYMBOL = "AAA"
GENERATED_AT = "2026-07-29T00:00:00Z"
DEFAULT_OUTPUT_DIR = Path("docs/audits/ticket_71")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write Ticket 71 multi-timeframe target authority artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    artifacts = write_ticket_71_artifacts(args.output_dir)
    print(json.dumps({"classification": artifacts["classification"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def write_ticket_71_artifacts(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = _target_surface_inventory()
    conflicts = _target_conflicts(inventory)
    contract_payload = _contract_payload()
    catalogue = target_catalogue_payload()
    daily_results = _daily_examples()
    hourly_results = _hourly_examples()
    five_minute_results = _five_minute_examples()
    all_results = [*daily_results, *hourly_results, *five_minute_results]

    daily_path = output_dir / "daily_target_examples.parquet"
    hourly_path = output_dir / "hourly_target_examples.parquet"
    five_path = output_dir / "five_minute_target_examples.parquet"
    _write_parquet(daily_path, daily_results)
    _write_parquet(hourly_path, hourly_results)
    _write_parquet(five_path, five_minute_results)

    validation = validate_target_availability(all_results)
    matrix = _resolution_matrix(all_results)
    manifest = build_target_manifest(
        five_minute_results,
        selected_target=resolve_target_contract("forward_return_30m__decision_5m"),
        source_cutoff="2024-01-16T21:00:00Z",
        output_paths=(daily_path, hourly_path, five_path),
        configuration={"fixture": "ticket_71_synthetic_reference", "generated_at": GENERATED_AT},
        calendar_identity=default_calendar_authority().identity(
            start="2024-01-02",
            end="2024-07-10",
        ),
        producer_command="python scripts/ticket_71_multi_timeframe_target_authority.py",
        producer_module="scripts.ticket_71_multi_timeframe_target_authority",
    )
    manifest["generated_at"] = GENERATED_AT
    manifest["manifest_hash"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )

    _write_json(output_dir / "target_surface_inventory.json", _inventory_payload(inventory))
    _write_csv(output_dir / "target_conflicts.csv", conflicts, _conflict_fields())
    _write_json(output_dir / "multi_timeframe_target_contract.json", contract_payload)
    _write_json(output_dir / "target_catalogue.json", catalogue)
    _write_csv(output_dir / "target_resolution_matrix.csv", matrix, _matrix_fields())
    _write_json(output_dir / "target_availability_validation.json", validation)
    _write_json(output_dir / "target_manifest_example.json", manifest)

    classification = (
        "MULTI_TIMEFRAME_TARGET_AUTHORITY_IMPLEMENTED"
        if validation["status"] == "PASSED"
        and _required_contracts_present(catalogue)
        and _required_states_covered(all_results)
        else "IMPLEMENTED_WITH_LIMITATIONS"
    )
    summary = _summary_markdown(
        classification=classification,
        inventory=inventory,
        conflicts=conflicts,
        catalogue=catalogue,
        validation=validation,
        manifest=manifest,
        output_dir=output_dir,
    )
    (output_dir / "ticket_71_summary.md").write_text(summary, encoding="utf-8")
    return {
        "classification": classification,
        "output_dir": str(output_dir),
        "artifact_count": len(list(output_dir.glob("*"))),
    }


def _contract_payload() -> dict[str, Any]:
    required_fields = [
        "target_id",
        "legacy_aliases",
        "decision_timeframe",
        "source_bar_timeframe",
        "decision_schedule",
        "horizon_value",
        "horizon_unit",
        "session_boundary_rule",
        "entry_price_rule",
        "exit_price_rule",
        "target_start_rule",
        "target_end_rule",
        "availability_rule",
        "missing_bar_policy",
        "partial_session_policy",
        "early_close_policy",
        "overnight_policy",
        "calendar_authority_version",
        "price_authority_version",
        "target_code_hash",
        "resolution_policy_version",
    ]
    payload = {
        "contract_version": MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION,
        "catalogue_version": TARGET_CATALOGUE_VERSION,
        "generated_at": GENERATED_AT,
        "required_fields": required_fields,
        "supported_horizon_units": [
            "eligible_trading_sessions",
            "elapsed_market_minutes",
            "eligible_bars",
            "to_session_close",
            "to_next_session_open",
        ],
        "supported_result_states": sorted(TARGET_RESOLUTION_STATES),
        "required_invariants": [
            "decision_timeframe is not forecast horizon",
            "source_bar_timeframe is explicit",
            "target_available_timestamp >= target_end_timestamp",
            "target_is_trainable requires target_available_timestamp <= training_cutoff",
            "intraday elapsed-minute targets do not cross market-session boundaries unless contract says so",
            "legacy forward_return_10d maps only to forward_return_10_sessions__decision_1Day",
        ],
        "calendar_authority": default_calendar_authority().identity(
            start="2024-01-02",
            end="2024-07-10",
        ),
        "resolution_policy_version": TARGET_RESOLUTION_POLICY_VERSION,
        "target_code_hash": target_code_hash(),
        "model_training_allowed": False,
        "model_promotion_allowed": False,
    }
    payload["content_hash"] = canonical_hash(payload)
    return payload


def _daily_examples() -> list[dict[str, Any]]:
    sessions = trading_sessions(date(2024, 1, 12), date(2024, 2, 5))
    july_sessions = trading_sessions(date(2024, 7, 3), date(2024, 7, 10))
    bars = [*_daily_bars(sessions, start=50.0), *_daily_bars(july_sessions, start=80.0)]
    missing_intermediate_bars = [
        row for row in bars
        if row["session_date"] != "2024-01-17"
    ]
    cases = [
        ("daily_10_session_holiday_crossing", bars, "2024-01-12", "forward_return_10d", _session_close(sessions[-1])),
        ("daily_5_session_holiday_crossing", bars, "2024-01-12", "forward_return_5_sessions__decision_1Day", _session_close(sessions[-1])),
        ("daily_1_session_early_close", bars, "2024-07-03", "forward_return_1_session__decision_1Day", _session_close(july_sessions[-1])),
        ("daily_not_yet_mature", bars, "2024-01-12", "forward_return_1_session__decision_1Day", _session_close(date(2024, 1, 16))),
        ("daily_intermediate_missing_session", missing_intermediate_bars, "2024-01-12", "forward_return_5_sessions__decision_1Day", _session_close(sessions[-1])),
    ]
    return [_example_row(case_id, calculate_target(
        asset_id=ASSET,
        decision_timestamp=decision,
        bar_source=source_rows,
        target_contract=target,
        source_cutoff=cutoff,
    )) for case_id, source_rows, decision, target, cutoff in cases]


def _hourly_examples() -> list[dict[str, Any]]:
    rows = [
        *_hourly_bars(date(2024, 1, 2)),
        *_hourly_bars(date(2024, 1, 3), start=200.0),
    ]
    cases = [
        ("hourly_60m_valid", "2024-01-02T15:30:00Z", "forward_return_60m__decision_1h", "2024-01-02T21:00:00Z", ()),
        ("hourly_240m_valid", "2024-01-02T15:30:00Z", "forward_return_240m__decision_1h", "2024-01-02T21:00:00Z", ()),
        ("hourly_near_close_conflict", "2024-01-02T20:30:00Z", "forward_return_60m__decision_1h", "2024-01-02T21:00:00Z", ()),
        ("hourly_to_close", "2024-01-02T19:30:00Z", "forward_return_to_close__decision_1h", "2024-01-02T21:00:00Z", ()),
        ("hourly_next_session", "2024-01-02T20:30:00Z", "forward_return_next_session__decision_1h", "2024-01-03T15:30:00Z", ()),
    ]
    return [_example_row(case_id, calculate_target(
        asset_id=ASSET,
        decision_timestamp=decision,
        bar_source=rows,
        target_contract=target,
        source_cutoff=cutoff,
        market_halts=halts,
    )) for case_id, decision, target, cutoff, halts in cases]


def _five_minute_examples() -> list[dict[str, Any]]:
    rows = [
        *_five_minute_bars(date(2024, 1, 2)),
        *_five_minute_bars(date(2024, 1, 3), start=300.0),
        *_five_minute_bars(date(2024, 1, 5), start=400.0),
        *_five_minute_bars(date(2024, 1, 8), start=500.0),
        *_five_minute_bars(date(2024, 1, 12), start=600.0),
        *_five_minute_bars(date(2024, 1, 16), start=700.0),
        *_five_minute_bars(date(2024, 3, 8), start=800.0),
        *_five_minute_bars(date(2024, 3, 11), start=900.0),
        *_five_minute_bars(date(2024, 7, 3), start=1000.0),
    ]
    missing_rows = [
        row for row in rows
        if not (
            row["session_date"] == "2024-01-02"
            and row["bar_end_timestamp"] == "2024-01-02T15:05:00Z"
        )
    ]
    intermediate_missing_rows = [
        row for row in rows
        if not (
            row["session_date"] == "2024-01-02"
            and row["bar_end_timestamp"] == "2024-01-02T14:50:00Z"
        )
    ]
    quarantined_rows = _replace_bar_status(rows, "2024-01-02T15:05:00Z", "QUARANTINED_SOURCE_BAR")
    ineligible_rows = _replace_bar_status(rows, "2024-01-02T15:05:00Z", "INELIGIBLE_SOURCE_BAR")
    gap_rows = _replace_bar_status(rows, "2024-01-02T15:05:00Z", "PROVIDER_GAP")
    cases = [
        ("five_minute_30m_valid", rows, "2024-01-02T14:35:00Z", "forward_return_30m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_60m_valid", rows, "2024-01-02T14:35:00Z", "forward_return_60m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_near_close_conflict", rows, "2024-01-02T20:35:00Z", "forward_return_60m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_to_close", rows, "2024-01-02T19:35:00Z", "forward_return_to_close__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_next_open", rows, "2024-01-02T20:55:00Z", "forward_return_next_open__decision_5m", "2024-01-03T14:35:00Z", ()),
        ("five_minute_weekend_rollover", rows, "2024-01-05T20:55:00Z", "forward_return_next_open__decision_5m", "2024-01-08T14:35:00Z", ()),
        ("five_minute_holiday_rollover", rows, "2024-01-12T20:55:00Z", "forward_return_next_open__decision_5m", "2024-01-16T14:35:00Z", ()),
        ("five_minute_dst_rollover", rows, "2024-03-08T20:55:00Z", "forward_return_next_open__decision_5m", "2024-03-11T13:35:00Z", ()),
        ("five_minute_early_close_to_close", rows, "2024-07-03T16:35:00Z", "forward_return_to_close__decision_5m", "2024-07-03T17:00:00Z", ()),
        ("five_minute_missing_bar", missing_rows, "2024-01-02T14:35:00Z", "forward_return_30m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_intermediate_missing_bar", intermediate_missing_rows, "2024-01-02T14:35:00Z", "forward_return_30m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_quarantined_bar", quarantined_rows, "2024-01-02T14:35:00Z", "forward_return_30m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_ineligible_bar", ineligible_rows, "2024-01-02T14:35:00Z", "forward_return_30m__decision_5m", "2024-01-02T21:00:00Z", ()),
        ("five_minute_provider_gap", gap_rows, "2024-01-02T14:35:00Z", "forward_return_30m__decision_5m", "2024-01-02T21:00:00Z", ()),
        (
            "five_minute_halt_affected",
            rows,
            "2024-01-02T14:35:00Z",
            "forward_return_30m__decision_5m",
            "2024-01-02T21:00:00Z",
            ({"start_timestamp": "2024-01-02T14:45:00Z", "end_timestamp": "2024-01-02T14:55:00Z", "reason": "represented_halt_fixture"},),
        ),
        ("five_minute_right_censored", rows, "2024-01-02T14:35:00Z", "forward_return_60m__decision_5m", "2024-01-02T15:00:00Z", ()),
    ]
    output = []
    for case_id, source_rows, decision, target, cutoff, halts in cases:
        output.append(_example_row(case_id, calculate_target(
            asset_id=ASSET,
            decision_timestamp=decision,
            bar_source=source_rows,
            target_contract=target,
            source_cutoff=cutoff,
            market_halts=halts,
        )))
    return output


def _example_row(case_id: str, result: Any) -> dict[str, Any]:
    payload = result.payload()
    payload["case_id"] = case_id
    payload["row_id"] = f"ticket71-{case_id}"
    return payload


def _target_surface_inventory() -> list[dict[str, Any]]:
    return [
        {
            "name": "forward_return_10d / actual_forward_return_10d",
            "source_timeframe": "1Day",
            "decision_timeframe": "1Day",
            "horizon": 10,
            "horizon_unit": "eligible_trading_sessions",
            "entry_rule": "decision-session close",
            "exit_rule": "tenth eligible future session close",
            "availability_rule": "label_available_timestamp at first later decision close",
            "maturity_rule": "Ticket 57F target_is_trainable true only for MATURED_VALID rows",
            "missing_bar_rule": "legacy target_status plus Ticket 57F target_resolution fields",
            "session_boundary_behavior": "ordered exchange sessions; holidays/weekends skipped; early-close timestamp retained",
            "consumers": [
                "stock-level selector datasets",
                "bounded selector runner",
                "ordinary selector publication",
                "portfolio replay diagnostics",
                "model ranking benchmark",
            ],
            "conflicts_or_ambiguity": ["legacy suffix d can be read as calendar days instead of eligible sessions"],
        },
        {
            "name": "forward_return_1d, forward_return_5d, forward_return_20d registry entries",
            "source_timeframe": "1Day",
            "decision_timeframe": "1Day",
            "horizon": "1/5/20",
            "horizon_unit": "eligible_trading_sessions",
            "entry_rule": "daily close target family",
            "exit_rule": "configured eligible future session close",
            "availability_rule": "per-horizon maturity timestamp where present",
            "maturity_rule": "multi-horizon panels use MATURE/IMMATURE/INVALID state",
            "missing_bar_rule": "per-horizon invalid or immature state",
            "session_boundary_behavior": "daily session order",
            "consumers": ["multi-horizon selector adapters", "target tournament"],
            "conflicts_or_ambiguity": ["legacy d suffix remains less explicit than sessions"],
        },
        {
            "name": "risk-adjusted and benchmark daily labels",
            "source_timeframe": "1Day",
            "decision_timeframe": "1Day",
            "horizon": 10,
            "horizon_unit": "eligible_trading_sessions",
            "entry_rule": "shares raw 10-session close anchor",
            "exit_rule": "shares raw 10-session close endpoint plus benchmark/future path transforms",
            "availability_rule": "inherits raw target availability",
            "maturity_rule": "inherits Ticket 57F target trainability fields when materialized",
            "missing_bar_rule": "blank when benchmark or future path is unavailable",
            "session_boundary_behavior": "daily session order",
            "consumers": ["target comparison", "model ranking benchmark", "portfolio replay diagnostics"],
            "conflicts_or_ambiguity": ["future drawdown and volatility are path labels, not point-in-time features"],
        },
        {
            "name": "hourly intraday features",
            "source_timeframe": "1h",
            "decision_timeframe": "1Day or intraday feature cutoff",
            "horizon": "not a supervised target",
            "horizon_unit": "",
            "entry_rule": "as-of feature fusion only",
            "exit_rule": "not applicable",
            "availability_rule": "bars at or before feature cutoff",
            "maturity_rule": "not applicable",
            "missing_bar_rule": "feature omitted when no bars are available",
            "session_boundary_behavior": "prior implementation did not define target horizons",
            "consumers": ["add_intraday_summary_features"],
            "conflicts_or_ambiguity": ["no authoritative hourly target contract existed before Ticket 71"],
        },
        {
            "name": "Alpaca five-minute final symbol/year archive",
            "source_timeframe": "5m",
            "decision_timeframe": "5m candidate",
            "horizon": "not a supervised target",
            "horizon_unit": "",
            "entry_rule": "source bar timestamp is bar start; target authority derives finalization time",
            "exit_rule": "not previously defined",
            "availability_rule": "finalized symbol/year bars.parquet plus source_row_hash",
            "maturity_rule": "not previously defined for labels",
            "missing_bar_rule": "finalizer records validation and completeness diagnostics",
            "session_boundary_behavior": "session_type from maintained market session wrapper",
            "consumers": ["five-minute archive finalizer", "intraday feature planning"],
            "conflicts_or_ambiguity": ["bar-start timestamp could be mistaken for realised outcome availability"],
        },
        {
            "name": "sequence model labels",
            "source_timeframe": "1Day",
            "decision_timeframe": "1Day",
            "horizon": "consumer-selected daily target",
            "horizon_unit": "eligible_trading_sessions",
            "entry_rule": "inherits selector target",
            "exit_rule": "inherits selector target",
            "availability_rule": "sequence authority rejects target information inside feature window",
            "maturity_rule": "strict sequence context plus target maturity fields",
            "missing_bar_rule": "strict context rejects missing windows when configured",
            "session_boundary_behavior": "daily chronological windows",
            "consumers": ["DLinear", "PatchTST", "Transformer", "TFT", "iTransformer"],
            "conflicts_or_ambiguity": ["sequence length is context history, not forecast horizon"],
        },
        {
            "name": "news transformer report-only price labels",
            "source_timeframe": "1Day adjusted price rows",
            "decision_timeframe": "event available timestamp",
            "horizon": "1/5/20 price observations",
            "horizon_unit": "eligible daily price observations",
            "entry_rule": "first price row on or after event availability date",
            "exit_rule": "N later price row",
            "availability_rule": "report-only labels, not canonical selector target authority",
            "maturity_rule": "rows unlabeled when future bars are insufficient",
            "missing_bar_rule": "separates missing price file from insufficient future bars",
            "session_boundary_behavior": "daily adjusted price order",
            "consumers": ["news transformer report-only diagnostics"],
            "conflicts_or_ambiguity": ["uses price-row observations and remains outside canonical multi-timeframe contract"],
        },
        {
            "name": "portfolio replay exposure label should_reduce_exposure",
            "source_timeframe": "portfolio replay equity curves",
            "decision_timeframe": "daily feature_date",
            "horizon": "configured outcome_horizon_days",
            "horizon_unit": "calendar_days",
            "entry_rule": "portfolio state at feature date",
            "exit_rule": "configured replay outcome endpoint",
            "availability_rule": "label_available_timestamp emitted by replay dataset",
            "maturity_rule": "maximum_target_available_timestamp must pass evidence cutoff",
            "missing_bar_rule": "replay rows validate target contract hash and source equity curve path",
            "session_boundary_behavior": "portfolio replay calendar-day helper",
            "consumers": ["exposure policy dataset", "portfolio replay certification"],
            "conflicts_or_ambiguity": ["calendar-day exposure horizon is not a price-bar forecast horizon"],
        },
    ]


def _target_conflicts(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inventory:
        for conflict in item.get("conflicts_or_ambiguity", []) or []:
            rows.append(
                {
                    "name": item["name"],
                    "conflict_type": _conflict_type(str(conflict)),
                    "description": conflict,
                    "resolution_in_ticket_71": _conflict_resolution(str(conflict), name=str(item["name"])),
                    "blocking": "false",
                    "consumer": "|".join(item.get("consumers") or []),
                }
            )
    return rows


def _conflict_type(conflict: str) -> str:
    text = conflict.lower()
    if "bar-start" in text:
        return "timestamp_semantics"
    if "sequence length" in text:
        return "decision_context_vs_horizon"
    if "calendar-day" in text:
        return "horizon_unit_conflict"
    if "no authoritative" in text:
        return "missing_contract"
    return "ambiguous_legacy_identity"


def _conflict_resolution(conflict: str, *, name: str = "") -> str:
    text = conflict.lower()
    surface = name.lower()
    if "no authoritative" in text:
        return "new hourly and five-minute contracts added; no legacy consumers rewired"
    if "bar-start" in text:
        return "target authority derives bar finalization timestamps before label availability"
    if "sequence length" in text:
        return "sequence-window authority remains context-only; target horizon is explicit contract field"
    if "calendar-day" in text:
        return "left as separate exposure label surface, not aliased to price-bar targets"
    if "report-only" in text or "report-only" in surface:
        return "documented as non-canonical report label surface"
    if "drawdown" in text or "volatility" in text:
        return "kept as separate path-dependent labels, not aliased to forward-return contracts"
    if "forward_return_1d" in surface or "forward_return_20d" in surface:
        return "legacy d-suffixed registry entries remain inventoried; strict research mode requires canonical target IDs"
    return "legacy alias maps only to forward_return_10_sessions__decision_1Day"


def _inventory_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "inventory_version": "target_surface_inventory.v1",
        "generated_at": GENERATED_AT,
        "rows": list(rows),
        "content_hash": canonical_hash(rows),
    }


def _resolution_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(row.get("decision_timeframe") or ""),
            str(row.get("source_bar_timeframe") or ""),
            str(row.get("target_id") or ""),
            str(row.get("target_resolution_classification") or ""),
        )
        for row in rows
    )
    return [
        {
            "decision_timeframe": key[0],
            "source_bar_timeframe": key[1],
            "target_id": key[2],
            "target_resolution_classification": key[3],
            "row_count": value,
        }
        for key, value in sorted(counts.items())
    ]


def _required_contracts_present(catalogue: Mapping[str, Any]) -> bool:
    required = {
        "forward_return_1_session__decision_1Day",
        "forward_return_5_sessions__decision_1Day",
        "forward_return_10_sessions__decision_1Day",
        "forward_return_60m__decision_1h",
        "forward_return_240m__decision_1h",
        "forward_return_to_close__decision_1h",
        "forward_return_next_session__decision_1h",
        "forward_return_30m__decision_5m",
        "forward_return_60m__decision_5m",
        "forward_return_to_close__decision_5m",
        "forward_return_next_open__decision_5m",
    }
    actual = {str(row.get("target_id")) for row in catalogue.get("contracts", [])}
    return required <= actual


def _required_states_covered(rows: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        MATURED_VALID,
        NOT_YET_MATURE,
        RIGHT_CENSORED,
        MISSING_SOURCE_BAR,
        QUARANTINED_SOURCE_BAR,
        INELIGIBLE_SOURCE_BAR,
        SESSION_BOUNDARY_CONFLICT,
        HALT_AFFECTED,
        UNKNOWN_SOURCE_GAP,
    }
    actual = {str(row.get("target_resolution_classification") or "") for row in rows}
    return required <= actual


def _summary_markdown(
    *,
    classification: str,
    inventory: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    catalogue: Mapping[str, Any],
    validation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> str:
    contracts = catalogue.get("contracts", [])
    lines = [
        "# Ticket 71 Multi-Timeframe Target Authority",
        "",
        f"- Classification: `{classification}`",
        f"- Contract version: `{MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION}`",
        f"- Catalogue version: `{TARGET_CATALOGUE_VERSION}`",
        f"- Output directory: `{output_dir}`",
        f"- Existing target surfaces inventoried: `{len(inventory)}`",
        f"- Ambiguities recorded: `{len(conflicts)}`",
        f"- Contracts defined: `{len(contracts)}`",
        f"- Availability validation: `{validation.get('status')}`",
        f"- Manifest trainable example rows: `{(manifest.get('row_counts') or {}).get('trainable')}`",
        "",
        "Daily semantics preserve the legacy `forward_return_10d` value as ten eligible future daily trading sessions while exposing the clearer canonical ID `forward_return_10_sessions__decision_1Day`.",
        "",
        "Hourly and five-minute elapsed-minute targets require maturity inside the same regular session. To-close and next-open contracts declare their session-boundary behavior explicitly.",
        "",
        "Missing-bar, quarantine, ineligible, represented halt, unknown gap, right-censored, and not-yet-mature states are separate target-resolution outcomes.",
        "",
        "No model training, strategy comparison, portfolio replay, paper trading, production deployment, or policy promotion was performed.",
    ]
    return "\n".join(lines) + "\n"


def _daily_bars(sessions: Sequence[date], *, start: float) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": ASSET,
            "canonical_symbol": SYMBOL,
            "session_date": session.isoformat(),
            "timeframe": "1Day",
            "open": start + index - 0.25,
            "close": start + index,
            "source_bar_id": f"{SYMBOL}-1d-{session.isoformat()}",
        }
        for index, session in enumerate(sessions)
    ]


def _five_minute_bars(day: date, *, start: float = 100.0) -> list[dict[str, Any]]:
    return _intraday_bars(day, minutes=5, timeframe="5m", start=start)


def _hourly_bars(day: date, *, start: float = 100.0) -> list[dict[str, Any]]:
    open_ts = _parse(_session_open(day))
    close_ts = _parse(_session_close(day))
    starts = [open_ts + timedelta(hours=index) for index in range(6)]
    starts.append(close_ts - timedelta(hours=1))
    rows = []
    for index, timestamp in enumerate(starts):
        rows.append(_intraday_row(timestamp, minutes=60, timeframe="1h", open_=start + index, close=start + index + 1.0))
    return rows


def _intraday_bars(day: date, *, minutes: int, timeframe: str, start: float) -> list[dict[str, Any]]:
    open_ts = _parse(_session_open(day))
    close_ts = _parse(_session_close(day))
    rows = []
    current = open_ts
    index = 0
    while current < close_ts:
        rows.append(
            _intraday_row(
                current,
                minutes=minutes,
                timeframe=timeframe,
                open_=start + index,
                close=start + index + 0.5,
            )
        )
        current += timedelta(minutes=minutes)
        index += 1
    return rows


def _intraday_row(
    timestamp: datetime,
    *,
    minutes: int,
    timeframe: str,
    open_: float,
    close: float,
) -> dict[str, Any]:
    stamp = _format(timestamp)
    end = timestamp + timedelta(minutes=minutes)
    return {
        "asset_id": ASSET,
        "canonical_symbol": SYMBOL,
        "timestamp_utc": stamp,
        "bar_end_timestamp": _format(end),
        "session_date": timestamp.astimezone(timezone.utc).date().isoformat(),
        "timeframe": timeframe,
        "open": open_,
        "close": close,
        "bar_status": "OK",
        "source_bar_id": f"{SYMBOL}-{timeframe}-{stamp}",
    }


def _replace_bar_status(
    rows: Sequence[Mapping[str, Any]],
    bar_end_timestamp: str,
    status: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["bar_end_timestamp"] == bar_end_timestamp:
            output.append({**row, "bar_status": status})
        else:
            output.append(dict(row))
    return output


def _session_open(day: date) -> str:
    record = default_calendar_authority().session(day)
    if record.open_timestamp is None:
        raise ValueError(f"no open for {day}")
    return _format(record.open_timestamp)


def _session_close(day: date) -> str:
    record = default_calendar_authority().session(day)
    if record.close_timestamp is None:
        raise ValueError(f"no close for {day}")
    return _format(record.close_timestamp)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path, compression="zstd")


def _conflict_fields() -> tuple[str, ...]:
    return (
        "name",
        "conflict_type",
        "description",
        "resolution_in_ticket_71",
        "blocking",
        "consumer",
    )


def _matrix_fields() -> tuple[str, ...]:
    return (
        "decision_timeframe",
        "source_bar_timeframe",
        "target_id",
        "target_resolution_classification",
        "row_count",
    )


if __name__ == "__main__":
    raise SystemExit(main())
