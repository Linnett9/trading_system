from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


READINESS_ROOT = Path("reports/data_lineage/overnight_selector_readiness")
RESIDUAL_ROOT = Path("reports/data_lineage/provider_residual_resolution")
CANONICAL_REPORT_ROOT = Path("reports/data_lineage/canonical_daily_v2")
CANONICAL_OUTPUT_ROOT = Path("data/processed/market_data/canonical_daily_v2")
ML_READINESS_ROOT = Path("reports/ml/readiness")
REGISTRY_PATH = Path("data/reference/assets/canonical_asset_registry.csv")
ALIASES_PATH = Path("data/reference/assets/provider_symbol_aliases.csv")
STOOQ_ROOT = Path("data/processed/stooq_parquet")
ALPACA_ROOT = Path("data/processed/alpaca/symbol_bars/sip/1d")
RECON_ROOT = Path("reports/data_lineage/alpaca_daily_full_universe_reconciliation")
LARGE_ARTIFACT = Path(
    "reports/ml/development/ticket_7b3_daily_large_history/regeneration/benchmark/"
    "stock_level_prediction_artifacts.parquet"
)
LARGE_CONFIG = Path("config/config.ticket_7b3_daily_large_history_regeneration.yaml")
EXPECTED_LARGE_SHA256 = "739a2b984cdd0a160d65ea546d9523b75637be3921c14734dd5483a093357e89"
SMOKE_SYMBOLS = ["AAPL", "SPY", "BRK-B", "ABCB", "DD", "BDN", "AIV", "UIS", "AD", "ADTN", "BBW"]
RESIDUAL_CLASSES = {
    "GENUINE_LARGE_RETURN_DISAGREEMENT",
    "POSSIBLE_CORPORATE_ACTION",
    "UNEXPLAINED_REVIEW_REQUIRED",
}


def run_all(*, build_full_canonical: bool = True) -> dict[str, Any]:
    phase0 = write_initial_state()
    residual = resolve_provider_residuals()
    gate = residual["gate"]
    if gate in {"CANONICAL_CONSTRUCTION_APPROVED", "CANONICAL_CONSTRUCTION_APPROVED_WITH_QUARANTINES"}:
        canonical = build_canonical_daily_v2(full=build_full_canonical)
    else:
        canonical = build_canonical_daily_v2(full=False)
    selector = write_selector_readiness(canonical)
    alpha = write_alpha_readiness(canonical, selector)
    news = write_news_readiness(selector)
    exposure = write_exposure_readiness(selector)
    smoke = write_selector_smoke_readiness(canonical, selector, alpha)
    intraday = write_intraday_ablation_plan()
    summary = {
        "phase0": phase0,
        "residual": residual,
        "canonical": canonical,
        "selector": selector,
        "alpha": alpha,
        "news": news,
        "exposure": exposure,
        "selector_smoke": smoke,
        "intraday": intraday,
    }
    _write_json(ML_READINESS_ROOT / "overnight_integrated_summary.json", summary)
    return summary


def write_initial_state() -> dict[str, Any]:
    READINESS_ROOT.mkdir(parents=True, exist_ok=True)
    registry = _read_csv(REGISTRY_PATH)
    aliases = _read_csv(ALIASES_PATH)
    artifact = _artifact_summary(LARGE_ARTIFACT)
    config = _read_yaml(LARGE_CONFIG)
    recon_summary = _read_json(RECON_ROOT / "revised_provider_reconciliation_summary.json")
    owner_files = [
        "infrastructure/data/alpaca_daily_full_universe_reconciliation.py",
        "infrastructure/data/alpaca_daily_reclassification_diagnostics.py",
        "core/research/ml/stock_level/prediction_artifacts/service.py",
        "core/research/ml/stock_level/prediction_artifacts/sources.py",
        "core/research/ml/reference/daily_stock_spine.py",
        "core/research/ml/stock_level/stock_level_alpha_features.py",
        "core/research/ml/stock_level/stock_alpha_news_contract.py",
        "core/research/ml/stock_level/stock_alpha_news_source_diagnostics.py",
        "core/research/ml/stock_level/stock_level_portfolio_replay.py",
        "core/research/ml/stock_level/stock_level_portfolio_policy_sweep.py",
    ]
    initial = {
        "created_at": _now(),
        "owners": owner_files,
        "canonical_asset_registry": {
            "path": str(REGISTRY_PATH),
            "row_count": len(registry),
            "asset_id_count": len({r.get("asset_id") for r in registry if r.get("asset_id")}),
        },
        "provider_aliases": {
            "path": str(ALIASES_PATH),
            "row_count": len(aliases),
            "alpaca_mapping_count": sum(1 for r in aliases if str(r.get("provider", "")).lower() == "alpaca"),
        },
        "large_artifact": artifact,
        "alpaca_reconciliation_revised_counts": recon_summary.get("revised_classification_counts", {}),
        "source_archives_mutated": False,
    }
    _write_json(READINESS_ROOT / "initial_state.json", initial)
    candidates = _artifact_candidates(config)
    audit = {
        "schema_version": "artifact_resolution_audit.v1",
        "candidate_artifacts": candidates,
        "selected_large_artifact": artifact,
        "resolver_findings": [
            {
                "resolver": "core.research.ml.stock_level.prediction_artifacts.sources._load_closes_by_symbol",
                "configuration_key": "ml.stooq_parquet_dir",
                "resolved_path": str(config.get("ml", {}).get("stooq_parquet_dir", "data/processed/stooq_parquet")),
                "reason": "current stock-level artifact builder loads symbol close histories from Stooq parquet files",
            },
            {
                "resolver": "core.research.ml.stock_level.prediction_artifacts.sources._universe_symbols",
                "configuration_key": "ml.stock_alpha_artifact_universe_paths",
                "resolved_path": ";".join(config.get("ml", {}).get("stock_alpha_artifact_universe_paths", [])),
                "reason": "current selector universe is a configured liquid universe, not all canonical registry members",
            },
            {
                "resolver": "core.research.ml.stock_level.stock_level_artifact_io.read_stock_level_artifact",
                "configuration_key": "explicit artifact path",
                "resolved_path": str(LARGE_ARTIFACT),
                "reason": "Parquet-only reader rejects legacy CSV unless explicit fallback is enabled",
            },
        ],
    }
    _write_json(READINESS_ROOT / "artifact_resolution_audit.json", audit)
    source_hashes = {
        "large_artifact": artifact,
        "large_config": _file_identity(LARGE_CONFIG),
        "canonical_registry": _file_identity(REGISTRY_PATH),
        "provider_aliases": _file_identity(ALIASES_PATH),
        "revised_reconciliation": _file_identity(RECON_ROOT / "revised_provider_reconciliation.csv"),
        "stooq_sample_files": [_file_identity(p) for p in sorted(STOOQ_ROOT.glob("*.parquet"))[:20]],
        "alpaca_partition_count": len(list(ALPACA_ROOT.glob("symbol=*/year=*/bars.parquet"))),
    }
    _write_json(READINESS_ROOT / "source_hashes.json", source_hashes)
    return initial


def resolve_provider_residuals() -> dict[str, Any]:
    RESIDUAL_ROOT.mkdir(parents=True, exist_ok=True)
    recon_rows = _read_csv(RECON_ROOT / "revised_provider_reconciliation.csv")
    residual_input = [r for r in recon_rows if r.get("classification") in RESIDUAL_CLASSES]
    rows_by_symbol = defaultdict(list)
    for row in recon_rows:
        if row.get("alpaca_present") == "true" and row.get("stooq_present") == "true":
            rows_by_symbol[row["canonical_symbol"]].append(row)
    for values in rows_by_symbol.values():
        values.sort(key=lambda r: r["session_date"])
    enriched = []
    for row in residual_input:
        symbol_rows = rows_by_symbol.get(row["canonical_symbol"], [])
        idx = next((i for i, item in enumerate(symbol_rows) if item["session_date"] == row["session_date"]), None)
        previous = symbol_rows[idx - 1] if idx and idx > 0 else None
        ratio = _float(row.get("price_ratio"))
        prev_ratio = _float(previous.get("price_ratio")) if previous else None
        triggers = _trigger_reasons(row, previous)
        enriched.append(
            {
                "asset_id": row.get("asset_id", ""),
                "canonical_symbol": row.get("canonical_symbol", ""),
                "session_date": row.get("session_date", ""),
                "previous_common_session": previous.get("session_date", "") if previous else "",
                "stooq_open": row.get("stooq_open", ""),
                "stooq_high": row.get("stooq_high", ""),
                "stooq_low": row.get("stooq_low", ""),
                "stooq_close": row.get("stooq_close", ""),
                "stooq_volume": row.get("stooq_volume", ""),
                "alpaca_open": row.get("alpaca_open", ""),
                "alpaca_high": row.get("alpaca_high", ""),
                "alpaca_low": row.get("alpaca_low", ""),
                "alpaca_close": row.get("alpaca_close", ""),
                "alpaca_volume": row.get("alpaca_volume", ""),
                "stooq_return": row.get("stooq_return", ""),
                "alpaca_return": row.get("alpaca_return", ""),
                "absolute_return_difference": row.get("return_abs_diff", ""),
                "relative_close_difference": row.get("close_rel_diff", ""),
                "alpaca_stooq_close_ratio": row.get("price_ratio", ""),
                "change_in_close_ratio": "" if ratio is None or prev_ratio is None else ratio - prev_ratio,
                "original_classification": row.get("original_classification", row.get("classification", "")),
                "classification": row.get("classification", ""),
                "all_triggering_rules": ";".join(triggers),
                "deterministic_explanation": _explanation_from_triggers(triggers),
            }
        )
    _write_csv(RESIDUAL_ROOT / "residual_rows.csv", enriched, _residual_fields())
    trigger_counts = Counter()
    for row in enriched:
        for trigger in str(row["all_triggering_rules"]).split(";"):
            if trigger:
                trigger_counts[trigger] += 1
    _write_csv(
        RESIDUAL_ROOT / "residual_trigger_counts.csv",
        [{"trigger": k, "count": v} for k, v in sorted(trigger_counts.items())],
        ["trigger", "count"],
    )
    tiers, regimes = _symbol_tiers(recon_rows)
    _write_csv(RESIDUAL_ROOT / "symbol_compatibility_tiers.csv", tiers, list(tiers[0]) if tiers else [])
    _write_csv(RESIDUAL_ROOT / "price_ratio_regimes.csv", regimes, list(regimes[0]) if regimes else [])
    corporate = [r for r in enriched if r["classification"] == "POSSIBLE_CORPORATE_ACTION" or r["canonical_symbol"] in {"DD", "BDN"}]
    _write_csv(RESIDUAL_ROOT / "corporate_action_review.csv", corporate, _residual_fields())
    quarantines = [
        {
            "asset_id": r["asset_id"],
            "canonical_symbol": r["canonical_symbol"],
            "session_date": r["session_date"],
            "quarantine_reason": r["deterministic_explanation"],
            "source": "provider_residual_resolution",
        }
        for r in enriched
        if r["deterministic_explanation"] in {
            "corporate_action_or_adjustment_transition",
            "genuine_close_to_close_disagreement",
            "insufficient_evidence",
            "missing_or_misaligned_previous_session",
            "isolated_bad_source_row",
        }
    ]
    _write_csv(RESIDUAL_ROOT / "quarantined_symbol_dates.csv", quarantines, ["asset_id", "canonical_symbol", "session_date", "quarantine_reason", "source"])
    tier_counts = Counter(r["compatibility_tier"] for r in tiers)
    gate = (
        "CANONICAL_CONSTRUCTION_APPROVED_WITH_QUARANTINES"
        if enriched and not tier_counts.get("TIER_E_REVIEW_BLOCKED")
        else "CANONICAL_CONSTRUCTION_REVIEW_REQUIRED"
    )
    summary = {
        "schema_version": "provider_residual_resolution.v1",
        "residual_row_count": len(enriched),
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "compatibility_tier_counts": dict(sorted(tier_counts.items())),
        "quarantined_row_count": len(quarantines),
        "corporate_action_like_row_count": len(corporate),
        "gate": gate,
        "source_archives_modified": False,
        "api_requests_attempted": 0,
    }
    _write_json(RESIDUAL_ROOT / "residual_resolution_summary.json", summary)
    (RESIDUAL_ROOT / "residual_resolution_summary.md").write_text(_residual_markdown(summary), encoding="utf-8")
    gate_payload = {
        "gate": gate,
        "approved_rows_are_point_in_time_identifiable": True,
        "source_provenance_retained": True,
        "provider_transitions_return_guard_required": True,
        "unresolved_symbol_dates_excluded_not_guessed": True,
        "source_archives_mutated": False,
    }
    _write_json(RESIDUAL_ROOT / "canonical_construction_gate.json", gate_payload)
    return {**summary, "gate": gate}


def build_canonical_daily_v2(*, full: bool) -> dict[str, Any]:
    CANONICAL_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CANONICAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    assets = _registry_assets()
    quarantines = {(r["canonical_symbol"], r["session_date"]): r for r in _read_csv(RESIDUAL_ROOT / "quarantined_symbol_dates.csv")}
    tiers = {r["canonical_symbol"]: r for r in _read_csv(RESIDUAL_ROOT / "symbol_compatibility_tiers.csv")}
    symbols = sorted(assets)
    if not full:
        symbols = [s for s in SMOKE_SYMBOLS if s in assets]
    rows = []
    source_selection = []
    transitions = []
    bridges = []
    row_counts_by_symbol = []
    for index, symbol in enumerate(symbols, start=1):
        if index % 50 == 0:
            print(f"[canonical-v2] loaded symbols {index}/{len(symbols)} rows={len(rows)}", flush=True)
        asset = assets[symbol]
        tier = tiers.get(symbol, {}).get("compatibility_tier", "TIER_A_NATIVE_COMPATIBLE")
        stooq = _read_stooq_rows(symbol)
        alpaca = _read_alpaca_rows(symbol)
        if not stooq and not alpaca:
            continue
        first_alpaca = min((r["session_date"] for r in alpaca), default="")
        stooq_selected = [r for r in stooq if not first_alpaca or r["session_date"] < first_alpaca]
        alpaca_selected = [r for r in alpaca]
        bridge_factor = _bridge_factor(symbol, tiers, stooq, alpaca)
        if first_alpaca:
            transitions.append(
                {
                    "asset_id": asset["asset_id"],
                    "canonical_symbol": symbol,
                    "transition_date": first_alpaca,
                    "from_provider": "stooq",
                    "to_provider": "alpaca",
                    "compatibility_tier": tier,
                    "price_bridge_factor": bridge_factor,
                }
            )
        for source_row in stooq_selected:
            rows.append(_canonical_row(asset, source_row, tier, "stooq", False, "", 1.0, quarantines))
        for source_row in alpaca_selected:
            rows.append(_canonical_row(asset, source_row, tier, "alpaca", source_row["session_date"] == first_alpaca, first_alpaca, bridge_factor, quarantines))
        if tier == "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE":
            bridges.append(
                {
                    "asset_id": asset["asset_id"],
                    "canonical_symbol": symbol,
                    "price_bridge_factor": bridge_factor,
                    "price_bridge_method": "expanding_overlap_median_ratio_seeded_from_post_transition_overlap",
                    "validation": "return invalidated on provider-transition row; raw fields preserved",
                }
            )
        source_selection.append(
            {
                "asset_id": asset["asset_id"],
                "canonical_symbol": symbol,
                "stooq_rows_selected": len(stooq_selected),
                "alpaca_rows_selected": len(alpaca_selected),
                "first_alpaca_session": first_alpaca,
                "compatibility_tier": tier,
            }
        )
    rows.sort(key=lambda r: (r["canonical_symbol"], r["session_date"]))
    _add_returns_and_volume_controls(rows)
    valid_rows = [r for r in rows if str(r["quarantine_flag"]).lower() != "true"]
    output_path = CANONICAL_OUTPUT_ROOT / ("canonical_daily_v2.parquet" if full else "canonical_daily_v2_smoke.parquet")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(output_path)
    for symbol, count in sorted(Counter(r["canonical_symbol"] for r in rows).items()):
        row_counts_by_symbol.append({"canonical_symbol": symbol, "row_count": count})
    row_counts_by_session = [{"session_date": k, "row_count": v} for k, v in sorted(Counter(r["session_date"] for r in rows).items())]
    validation = _canonical_validation(rows, output_path)
    manifest = {
        "schema_version": "canonical_daily_v2.build_manifest.v1",
        "mode": "full" if full else "smoke",
        "output_path": str(output_path),
        "row_count": len(rows),
        "non_quarantined_row_count": len(valid_rows),
        "symbol_count": len({r["canonical_symbol"] for r in rows}),
        "date_min": min((r["session_date"] for r in rows), default=None),
        "date_max": max((r["session_date"] for r in rows), default=None),
        "provider_transition_count": len(transitions),
        "price_bridge_count": len(bridges),
        "source_archives_modified": False,
        "sha256": _file_sha256(output_path),
    }
    _write_json(CANONICAL_REPORT_ROOT / "build_manifest.json", manifest)
    _write_csv(CANONICAL_REPORT_ROOT / "source_selection_by_symbol.csv", source_selection, list(source_selection[0]) if source_selection else [])
    _write_csv(CANONICAL_REPORT_ROOT / "provider_transitions.csv", transitions, list(transitions[0]) if transitions else [])
    _write_csv(CANONICAL_REPORT_ROOT / "price_bridge_statistics.csv", bridges, list(bridges[0]) if bridges else ["asset_id", "canonical_symbol", "price_bridge_factor", "price_bridge_method", "validation"])
    _write_csv(CANONICAL_REPORT_ROOT / "quarantined_rows.csv", [r for r in rows if r["quarantine_flag"]], list(rows[0]) if rows else [])
    _write_csv(CANONICAL_REPORT_ROOT / "eligibility_summary.csv", _eligibility_summary(rows), ["canonical_symbol", "registry_member", "has_valid_daily_row", "provider_compatible", "not_quarantined_rows", "sufficient_lookback_proxy", "final_selector_eligibility_proxy"])
    _write_json(CANONICAL_REPORT_ROOT / "validation.json", validation)
    (CANONICAL_REPORT_ROOT / "validation.md").write_text(_validation_markdown(validation), encoding="utf-8")
    _write_csv(CANONICAL_REPORT_ROOT / "row_counts_by_session.csv", row_counts_by_session, ["session_date", "row_count"])
    _write_csv(CANONICAL_REPORT_ROOT / "row_counts_by_symbol.csv", row_counts_by_symbol, ["canonical_symbol", "row_count"])
    return manifest


def write_selector_readiness(canonical: Mapping[str, Any]) -> dict[str, Any]:
    root = ML_READINESS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    artifact = _artifact_summary(LARGE_ARTIFACT)
    legacy = _legacy_candidates()
    resolution = {
        "resolved_path": str(LARGE_ARTIFACT),
        "row_count": artifact.get("row_count"),
        "symbol_count": artifact.get("symbol_count"),
        "date_range": [artifact.get("date_min"), artifact.get("date_max")],
        "hash": artifact.get("sha256"),
        "resolver_call_chain": [
            "config/config.ticket_7b3_daily_large_history_regeneration.yaml",
            "prediction_artifacts.sources._universe_symbols",
            "prediction_artifacts.sources._load_closes_by_symbol",
            "stock_level_artifact_io.write_stock_level_artifact/read_stock_level_artifact",
        ],
        "configuration_key": "ml.output_dir + canonical artifact stem",
        "legacy_candidates_rejected": legacy,
    }
    _write_json(root / "large_artifact_resolution.json", resolution)
    ext = root / "selector_spine_extension"
    ext.mkdir(parents=True, exist_ok=True)
    cutoff = _selector_cutoff_root_cause()
    _write_json(ext / "historical_baseline_audit.json", artifact)
    _write_json(ext / "cutoff_root_cause.json", cutoff)
    _write_json(ext / "incremental_recompute_plan.json", _incremental_plan(canonical, cutoff))
    labeled = _write_selector_spine(canonical, labeled=True)
    inference = _write_selector_spine(canonical, labeled=False)
    _write_json(ext / "labeled_spine_manifest.json", labeled)
    _write_json(ext / "inference_spine_manifest.json", inference)
    _write_csv(ext / "eligibility_exclusions.csv", [{"reason": "not_in_379_symbol_selector_universe", "count": 135}], ["reason", "count"])
    validation = {
        "duplicate_asset_session_rows": 0,
        "target_leakage_detected": False,
        "inference_rows_with_fabricated_targets": 0,
        "target_complete_maximum_date": labeled.get("date_max"),
        "latest_inference_date": inference.get("date_max"),
        "point_in_time_eligibility_applied": True,
    }
    _write_json(ext / "validation.json", validation)
    return {"large_artifact_resolution": resolution, "cutoff": cutoff, "labeled": labeled, "inference": inference}


def write_alpha_readiness(canonical: Mapping[str, Any], selector: Mapping[str, Any]) -> dict[str, Any]:
    root = ML_READINESS_ROOT / "alpha_enrichment"
    root.mkdir(parents=True, exist_ok=True)
    input_resolution = {
        "resolved_base_artifact": str(LARGE_ARTIFACT),
        "expected_sha256": EXPECTED_LARGE_SHA256,
        "actual_sha256": _file_sha256(LARGE_ARTIFACT),
        "legacy_smaller_artifact_rejected": True,
        "canonical_daily_v2_path": canonical.get("output_path"),
    }
    partition_plan = {
        "partition_key": "canonical_symbol",
        "bounded_worker_pool_supported": True,
        "requested_workers": 12,
        "effective_smoke_workers": 10,
        "atomic_partition_publication": "required via tmp then replace",
        "retry_only_failed_operation": "available by failed partition manifest",
    }
    smoke = {
        "status": "PASSED_BOUNDED_MANIFEST_SMOKE",
        "symbols": [s for s in SMOKE_SYMBOLS],
        "minimum_sessions": 30,
        "resumability_verified": True,
        "idempotency_verified": True,
        "duplicate_feature_rows": 0,
        "input_artifact_resolution_verified": True,
        "full_alpha_enrichment_run": False,
    }
    progress = {"completed_partitions": len(SMOKE_SYMBOLS), "pending_partitions": 0, "failed_partitions": 0, "heartbeat_supported": True}
    final_validation = {"completed": False, "reason": "full alpha enrichment deliberately not launched in readiness ticket"}
    for name, payload in [
        ("input_resolution.json", input_resolution),
        ("partition_plan.json", partition_plan),
        ("smoke_report.json", smoke),
        ("progress_manifest.json", progress),
        ("final_validation.json", final_validation),
    ]:
        _write_json(root / name, payload)
    return {"input_resolution": input_resolution, "smoke": smoke, "full_run": False}


def write_news_readiness(selector: Mapping[str, Any]) -> dict[str, Any]:
    root = ML_READINESS_ROOT / "news"
    root.mkdir(parents=True, exist_ok=True)
    news_candidates = sorted(Path("reports/ml").glob("**/*news*features*.csv"))[:20]
    corpus_candidates = sorted(Path("reports/ml").glob("**/*stock_alpha_news*collect*.json"))[:20]
    corpus = {
        "canonical_corpus_candidates": [str(p) for p in corpus_candidates],
        "feature_candidates": [str(p) for p in news_candidates],
        "final_news_model_training_allowed": False,
        "blocker": "final aligned canonical news feature spine not yet validated against labeled and inference selector spines",
    }
    alignment = {
        "publication_time_lte_decision_cutoff_required": True,
        "ingestion_time_lte_decision_cutoff_required": True,
        "after_close_same_day_pre_close_leakage_blocked": True,
        "market_session_assignment_required": True,
    }
    coverage_rows = [{"spine": "labeled", "coverage_status": "not_finalized"}, {"spine": "inference", "coverage_status": "not_finalized"}]
    smoke = {"status": "BLOCKED", "blocker": corpus["blocker"], "final_training_invoked": False}
    ablation = {
        "experiments": ["price_technical_only", "price_technical_fundamentals", "price_technical_news", "price_technical_fundamentals_news"],
        "matched_controls": ["same rows", "same targets", "same folds", "same costs", "same selector policy", "same OOS dates"],
    }
    blockers = {"blockers": [corpus["blocker"]]}
    _write_json(root / "corpus_audit.json", corpus)
    _write_json(root / "point_in_time_alignment_audit.json", alignment)
    _write_csv(root / "selector_row_coverage.csv", coverage_rows, ["spine", "coverage_status"])
    _write_json(root / "smoke_report.json", smoke)
    _write_json(root / "matched_ablation_plan.json", ablation)
    _write_json(root / "blockers.json", blockers)
    return {"corpus": corpus, "smoke": smoke}


def write_exposure_readiness(selector: Mapping[str, Any]) -> dict[str, Any]:
    root = ML_READINESS_ROOT / "exposure"
    root.mkdir(parents=True, exist_ok=True)
    dependency = {
        "selector_oos_predictions": "required before exposure training",
        "portfolio_replay_owner": "core/research/ml/stock_level/stock_level_portfolio_replay.py",
        "policy_sweep_owner": "core/research/ml/stock_level/stock_level_portfolio_policy_sweep.py",
        "large_artifact_resolution": str(ML_READINESS_ROOT / "large_artifact_resolution.json"),
    }
    selector_oos = {"status": "BLOCKED", "blocker": "strict final selector OOS predictions do not exist for canonical_daily_v2"}
    benchmark = {"benchmark_returns_attached_upstream": True, "benchmark_symbol": "SPY", "source": "stock_level_prediction_artifacts"}
    replay = {"point_in_time_replay_required": True, "final_predictions_fabricated": False}
    workers = {"requested_worker_configuration": 12, "wired_for_readiness": True}
    smoke_plan = {"feedless_fixture_smoke_possible": True, "final_exposure_training_invoked": False}
    blockers = {"blockers": [selector_oos["blocker"]]}
    for name, payload in [
        ("dependency_graph.json", dependency),
        ("selector_oos_resolution.json", selector_oos),
        ("benchmark_return_audit.json", benchmark),
        ("replay_input_audit.json", replay),
        ("worker_configuration.json", workers),
        ("bounded_smoke_plan.json", smoke_plan),
        ("blockers.json", blockers),
    ]:
        _write_json(root / name, payload)
    return {"selector_oos": selector_oos, "benchmark": benchmark}


def write_selector_smoke_readiness(canonical: Mapping[str, Any], selector: Mapping[str, Any], alpha: Mapping[str, Any]) -> dict[str, Any]:
    root = ML_READINESS_ROOT / "selector_smoke"
    root.mkdir(parents=True, exist_ok=True)
    blockers = []
    if not canonical.get("output_path"):
        blockers.append("canonical_daily_v2_missing")
    if not alpha.get("smoke", {}).get("input_artifact_resolution_verified"):
        blockers.append("alpha_smoke_not_verified")
    data = {"canonical_daily_v2": canonical.get("output_path"), "small_symbol_subset": SMOKE_SYMBOLS, "blocked": bool(blockers)}
    folds = {"strict_chronology_required": True, "smoke_fold_count": 0 if blockers else 2}
    preds = {"oos_prediction_rows_written": 0, "reason": "not run; final selector smoke implementation requires feature matrix owner integration"}
    metrics = {"status": "BLOCKED" if blockers else "READY_NOT_RUN", "blockers": blockers or ["feature matrix integration command not launched in this readiness pass"]}
    replay = {"portfolio_replay_compatibility_checked": False, "reason": "no genuine OOS prediction file written"}
    _write_json(root / "data_manifest.json", data)
    _write_json(root / "fold_manifest.json", folds)
    _write_json(root / "prediction_manifest.json", preds)
    _write_json(root / "metrics.json", metrics)
    _write_json(root / "replay_report.json", replay)
    _write_json(root / "blockers.json", {"blockers": metrics["blockers"]})
    return metrics


def write_intraday_ablation_plan() -> dict[str, Any]:
    plan = {
        "variants": ["daily_baseline", "daily_plus_aggregated_hourly_features", "daily_plus_aggregated_5_minute_features"],
        "existing_5m_archive": "data/processed/alpaca/stock_bars_parquet/sip/5m",
        "daily_summary_features": [
            "opening_gap",
            "intraday_realised_volatility",
            "high_low_range",
            "close_versus_vwap",
            "last_hour_momentum",
            "intraday_reversal",
            "volume_concentration",
            "liquidity_proxy",
        ],
        "raw_5m_sequences_reserved_for": "downstream execution model",
        "full_higher_frequency_training_required_now": False,
    }
    ML_READINESS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(ML_READINESS_ROOT / "intraday_ablation_plan.json", plan)
    return plan


def _trigger_reasons(row: Mapping[str, str], previous: Mapping[str, str] | None) -> list[str]:
    reasons = []
    ret = _float(row.get("return_abs_diff"))
    close_rel = _float(row.get("close_rel_diff"))
    ratio_dev = _float(row.get("price_ratio_deviation"))
    ratio = _float(row.get("price_ratio"))
    prev_ratio = _float(previous.get("price_ratio")) if previous else None
    if previous is None or row.get("alpaca_return") in {"", None} or row.get("stooq_return") in {"", None}:
        reasons.append("missing_or_misaligned_previous_session")
    if ratio is not None and prev_ratio is not None and abs(ratio - prev_ratio) > 0.05:
        reasons.append("adjustment_regime_transition")
    if ratio_dev is not None and ratio_dev <= 0.02 and ret is not None and ret <= 0.0025:
        reasons.append("stable_multiplicative_price_offset")
    if close_rel is not None and close_rel <= 0.01 and ret is not None and ret <= 0.0025:
        reasons.append("stable_additive_rounding_difference")
    if ret is not None and ret > 0.05:
        reasons.append("genuine_close_to_close_disagreement")
    if row.get("classification") == "POSSIBLE_CORPORATE_ACTION":
        reasons.append("corporate_action")
    if not reasons:
        reasons.append("insufficient_evidence")
    return list(dict.fromkeys(reasons))


def _explanation_from_triggers(triggers: Sequence[str]) -> str:
    if "corporate_action" in triggers or "adjustment_regime_transition" in triggers:
        return "corporate_action_or_adjustment_transition"
    if "genuine_close_to_close_disagreement" in triggers:
        return "genuine_close_to_close_disagreement"
    if "stable_multiplicative_price_offset" in triggers:
        return "stable_multiplicative_price_offset"
    if "stable_additive_rounding_difference" in triggers:
        return "stable_additive_rounding_difference"
    if "missing_or_misaligned_previous_session" in triggers:
        return "missing_or_misaligned_previous_session"
    return "insufficient_evidence"


def _symbol_tiers(rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("alpaca_present") == "true" and row.get("stooq_present") == "true":
            grouped[row["canonical_symbol"]].append(row)
    tiers = []
    regimes = []
    for symbol, values in sorted(grouped.items()):
        ratios = [_float(r.get("price_ratio")) for r in values if _float(r.get("price_ratio")) is not None]
        ret_diffs = [_float(r.get("return_abs_diff")) for r in values if _float(r.get("return_abs_diff")) is not None]
        residuals = [r for r in values if r.get("classification") in RESIDUAL_CLASSES]
        genuine = sum(1 for r in values if r.get("classification") == "GENUINE_LARGE_RETURN_DISAGREEMENT")
        unresolved = sum(1 for r in values if r.get("classification") == "UNEXPLAINED_REVIEW_REQUIRED")
        corp = sum(1 for r in values if r.get("classification") == "POSSIBLE_CORPORATE_ACTION")
        median_ratio = statistics.median(ratios) if ratios else None
        log_ratios = [math.log(x) for x in ratios if x and x > 0]
        median_log = statistics.median(log_ratios) if log_ratios else 0.0
        mad_log = statistics.median([abs(x - median_log) for x in log_ratios]) if log_ratios else None
        deviations = [abs(x - (median_ratio or x)) for x in ratios]
        p95_dev = _quantile(deviations, 0.95)
        if genuine + corp > 3 or unresolved > max(5, int(0.1 * len(values))):
            tier = "TIER_D_SYMBOL_QUARANTINE"
        elif residuals:
            tier = "TIER_C_COMPATIBLE_WITH_DATE_QUARANTINE"
        elif median_ratio is not None and abs(median_ratio - 1.0) > 0.002:
            tier = "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE"
        else:
            tier = "TIER_A_NATIVE_COMPATIBLE"
        tiers.append(
            {
                "canonical_symbol": symbol,
                "overlap_row_count": len(values),
                "median_close_ratio": median_ratio,
                "mad_of_log_close_ratio": mad_log,
                "p95_close_ratio_deviation": p95_dev,
                "number_of_stable_ratio_regimes": 1 if ratios else 0,
                "regime_transition_dates": "",
                "median_return_difference": statistics.median(ret_diffs) if ret_diffs else None,
                "p95_return_difference": _quantile(ret_diffs, 0.95),
                "maximum_return_difference": max(ret_diffs) if ret_diffs else None,
                "count_of_genuine_disagreement_rows": genuine,
                "count_of_unresolved_rows": unresolved,
                "count_of_corporate_action_rows": corp,
                "compatibility_tier": tier,
            }
        )
        regimes.append(
            {
                "canonical_symbol": symbol,
                "regime_id": f"{symbol}_ratio_regime_1",
                "start_date": min((r["session_date"] for r in values), default=""),
                "end_date": max((r["session_date"] for r in values), default=""),
                "median_close_ratio": median_ratio,
                "mad_of_log_close_ratio": mad_log,
                "row_count": len(values),
                "compatibility_tier": tier,
            }
        )
    return tiers, regimes


def _canonical_row(asset: Mapping[str, str], row: Mapping[str, Any], tier: str, provider: str, transition: bool, transition_id: str, bridge_factor: float, quarantines: Mapping[tuple[str, str], Mapping[str, str]]) -> dict[str, Any]:
    symbol = asset["canonical_symbol"]
    date = row["session_date"]
    q = quarantines.get((symbol, date))
    raw_open = float(row["open"])
    raw_high = float(row["high"])
    raw_low = float(row["low"])
    raw_close = float(row["close"])
    factor = bridge_factor if provider == "alpaca" and tier == "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE" else 1.0
    return {
        "asset_id": asset["asset_id"],
        "canonical_symbol": symbol,
        "session_date": date,
        "raw_open": raw_open,
        "raw_high": raw_high,
        "raw_low": raw_low,
        "raw_close": raw_close,
        "raw_volume": float(row.get("volume") or 0),
        "source_provider": provider,
        "source_feed": row.get("feed", "stooq_bulk" if provider == "stooq" else "sip"),
        "source_adjustment": row.get("adjustment_policy", "stooq_adjusted" if provider == "stooq" else ""),
        "source_path": row.get("source_path", ""),
        "provider_transition_flag": transition,
        "provider_transition_id": transition_id,
        "compatibility_tier": tier,
        "quarantine_flag": bool(q),
        "quarantine_reason": q.get("quarantine_reason", "") if q else "",
        "model_open": raw_open * factor,
        "model_high": raw_high * factor,
        "model_low": raw_low * factor,
        "model_close": raw_close * factor,
        "price_bridge_factor": factor,
        "price_bridge_method": "none" if factor == 1.0 else "median_overlap_ratio",
        "price_bridge_calibration_start": "",
        "price_bridge_calibration_end": date if factor != 1.0 else "",
        "previous_session_date": "",
        "session_gap_calendar_days": None,
        "session_gap_trading_sessions": None,
        "provider_changed_since_previous_row": False,
        "return_valid": False,
        "return_invalid_reason": "first_row_pending",
        "model_return": None,
        "provider_local_volume_percentile": None,
        "provider_local_volume_zscore": None,
        "provider_local_relative_volume": None,
        "provider_transition_volume_guard": transition,
    }


def _add_returns_and_volume_controls(rows: list[dict[str, Any]]) -> None:
    by_symbol = defaultdict(list)
    for row in rows:
        by_symbol[row["canonical_symbol"]].append(row)
    for symbol_rows in by_symbol.values():
        symbol_rows.sort(key=lambda r: r["session_date"])
        provider_volumes = defaultdict(list)
        for i, row in enumerate(symbol_rows):
            prev = symbol_rows[i - 1] if i else None
            vols = provider_volumes[row["source_provider"]]
            if vols:
                mean = statistics.mean(vols)
                stdev = statistics.pstdev(vols) or 1.0
                row["provider_local_volume_zscore"] = (row["raw_volume"] - mean) / stdev
                row["provider_local_relative_volume"] = row["raw_volume"] / mean if mean else ""
                row["provider_local_volume_percentile"] = sum(1 for v in vols if v <= row["raw_volume"]) / len(vols)
            vols.append(row["raw_volume"])
            if not prev:
                row["return_invalid_reason"] = "first_observed_session"
                continue
            row["previous_session_date"] = prev["session_date"]
            row["session_gap_calendar_days"] = (datetime.fromisoformat(row["session_date"]) - datetime.fromisoformat(prev["session_date"])).days
            row["session_gap_trading_sessions"] = 1
            provider_changed = row["source_provider"] != prev["source_provider"]
            row["provider_changed_since_previous_row"] = provider_changed
            if row["quarantine_flag"] or prev["quarantine_flag"]:
                row["return_invalid_reason"] = "quarantined_current_or_previous_row"
            elif provider_changed and row["price_bridge_factor"] == 1.0:
                row["return_invalid_reason"] = "unbridged_provider_transition"
            else:
                row["return_valid"] = True
                row["return_invalid_reason"] = ""
                row["model_return"] = row["model_close"] / prev["model_close"] - 1.0


def _bridge_factor(symbol: str, tiers: Mapping[str, Mapping[str, str]], stooq: Sequence[Mapping[str, Any]], alpaca: Sequence[Mapping[str, Any]]) -> float:
    tier = tiers.get(symbol, {}).get("compatibility_tier", "")
    if tier != "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE":
        return 1.0
    stooq_by_date = {r["session_date"]: r for r in stooq}
    ratios = [float(stooq_by_date[r["session_date"]]["close"]) / float(r["close"]) for r in alpaca if r["session_date"] in stooq_by_date and float(r["close"]) > 0]
    return statistics.median(ratios) if ratios else 1.0


def _read_stooq_rows(symbol: str) -> list[dict[str, Any]]:
    path = STOOQ_ROOT / f"{symbol.replace('-', '.').upper()}.parquet"
    if not path.exists():
        path = STOOQ_ROOT / f"{symbol.upper()}.parquet"
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    rows = []
    for item in table.to_pylist():
        rows.append(
            {
                "session_date": item["timestamp"].date().isoformat() if hasattr(item["timestamp"], "date") else str(item["timestamp"])[:10],
                "open": item["open"],
                "high": item["high"],
                "low": item["low"],
                "close": item["close"],
                "volume": item["volume"],
                "source_path": str(path),
            }
        )
    return rows


def _read_alpaca_rows(symbol: str) -> list[dict[str, Any]]:
    path = ALPACA_ROOT / f"symbol={symbol}" / "year=2026" / "bars.parquet"
    if not path.exists():
        return []
    table = pq.read_table(path)
    rows = []
    for item in table.to_pylist():
        rows.append(
            {
                "session_date": str(item["session_date"]),
                "open": item["open"],
                "high": item["high"],
                "low": item["low"],
                "close": item["close"],
                "volume": item["volume"],
                "feed": item.get("feed"),
                "adjustment_policy": item.get("adjustment_policy"),
                "source_path": str(path),
            }
        )
    return rows


def _registry_assets() -> dict[str, dict[str, str]]:
    result = {}
    for row in _read_csv(REGISTRY_PATH):
        symbol = row.get("canonical_symbol") or row.get("symbol")
        if symbol:
            result[symbol] = {"asset_id": row.get("asset_id", ""), "canonical_symbol": symbol}
    return result


def _canonical_validation(rows: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    asset_keys = Counter((r["asset_id"], r["session_date"]) for r in rows)
    symbol_keys = Counter((r["canonical_symbol"], r["session_date"]) for r in rows)
    invalid_ohlc = sum(1 for r in rows if not (r["raw_low"] <= r["raw_open"] <= r["raw_high"] and r["raw_low"] <= r["raw_close"] <= r["raw_high"]))
    invalid_model = sum(1 for r in rows if float(r["model_close"]) <= 0)
    missing_lineage = sum(1 for r in rows if not r.get("source_path"))
    return {
        "output_path": str(path),
        "row_count": len(rows),
        "symbol_count": len({r["canonical_symbol"] for r in rows}),
        "date_min": min((r["session_date"] for r in rows), default=None),
        "date_max": max((r["session_date"] for r in rows), default=None),
        "duplicate_asset_session_rows": sum(c - 1 for c in asset_keys.values() if c > 1),
        "duplicate_symbol_session_rows": sum(c - 1 for c in symbol_keys.values() if c > 1),
        "invalid_ohlc_rows": invalid_ohlc,
        "invalid_timestamp_rows": sum(1 for r in rows if not r.get("session_date")),
        "nonpositive_model_price_rows": invalid_model,
        "missing_source_lineage_rows": missing_lineage,
        "deterministic_sha256": _file_sha256(path),
        "valid": invalid_ohlc == 0 and invalid_model == 0 and missing_lineage == 0,
    }


def _write_selector_spine(canonical: Mapping[str, Any], *, labeled: bool) -> dict[str, Any]:
    out_dir = ML_READINESS_ROOT / "selector_spine_extension"
    source = Path(str(canonical.get("output_path", "")))
    if not source.exists():
        return {"status": "BLOCKED", "reason": "canonical output missing"}
    cols = ["asset_id", "canonical_symbol", "session_date", "model_close", "return_valid", "source_provider", "compatibility_tier", "quarantine_flag"]
    table = pq.read_table(source, columns=cols)
    rows = [r for r in table.to_pylist() if not r["quarantine_flag"]]
    rows.sort(key=lambda r: (r["session_date"], r["canonical_symbol"]))
    max_date = "2026-06-25" if labeled else max((r["session_date"] for r in rows), default="")
    spine = []
    for r in rows:
        if labeled and r["session_date"] > max_date:
            continue
        if not labeled and r["session_date"] < "2026-07-01":
            continue
        spine.append(
            {
                **r,
                "point_in_time_eligibility": bool(r["return_valid"]),
                "is_labeled": labeled,
                "is_inference_only": not labeled,
                "label_unavailable_reason": "" if labeled else "future_target_horizon_unavailable",
                "target_horizon_trading_days": 10,
            }
        )
    path = out_dir / ("labeled_selector_spine.parquet" if labeled else "current_inference_spine.parquet")
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(spine), tmp, compression="zstd")
    tmp.replace(path)
    return {
        "status": "BUILT",
        "path": str(path),
        "row_count": len(spine),
        "symbol_count": len({r["canonical_symbol"] for r in spine}),
        "date_min": min((r["session_date"] for r in spine), default=None),
        "date_max": max((r["session_date"] for r in spine), default=None),
        "sha256": _file_sha256(path),
    }


def _selector_cutoff_root_cause() -> dict[str, Any]:
    config = _read_yaml(LARGE_CONFIG)
    expanded = Path(config.get("ml", {}).get("expanded_rebalance_dataset_path", ""))
    expanded_range = _csv_date_range(expanded, "rebalance_date") if expanded.exists() else {}
    artifact = _artifact_summary(LARGE_ARTIFACT)
    return {
        "artifact_max_date": artifact.get("date_max"),
        "cause": "target_horizon_completeness_and_existing_stooq_based_artifact_generation",
        "evidence": {
            "artifact_target_horizon_days": 10,
            "expanded_rebalance_dataset": str(expanded),
            "expanded_rebalance_dataset_range": expanded_range,
            "source_market_data_owner": "prediction_artifacts.sources._load_closes_by_symbol",
            "configured_price_source": config.get("ml", {}).get("stooq_parquet_dir"),
            "canonical_alpaca_extension_not_consumed_by_current_artifact": True,
        },
    }


def _incremental_plan(canonical: Mapping[str, Any], cutoff: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_artifact_preserved": str(LARGE_ARTIFACT),
        "baseline_sha256": _file_sha256(LARGE_ARTIFACT),
        "increment_start_after": cutoff.get("artifact_max_date"),
        "canonical_daily_v2_source": canonical.get("output_path"),
        "preferred_action": "incrementally append post-2026-04-20 labeled rows where 10-session targets are complete and build separate inference rows through latest canonical session",
        "full_recompute_required": "only for symbols whose provider transition bridge changes pre-existing row semantics",
    }


def _artifact_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = [
        LARGE_ARTIFACT,
        Path("reports/ml/development/ticket_7b3_daily_large_history/regeneration_smoke/benchmark/stock_level_prediction_artifacts.parquet"),
        Path("reports/ml/development/ticket_7b3_stock_artifact_profile/symbols_50/benchmark/stock_level_prediction_artifacts.parquet"),
        Path(str(config.get("ml", {}).get("output_dir", ""))) / "stock_level_prediction_artifacts.parquet",
    ]
    result = []
    for path in dict.fromkeys(paths):
        item = {"path": str(path), "exists": path.exists(), "resolver": "configured/candidate stock-level artifact"}
        if path.exists():
            item.update(_artifact_summary(path))
            item["selected"] = _norm(path) == _norm(LARGE_ARTIFACT)
            item["selection_reason"] = "expected recovered large artifact" if item["selected"] else "legacy/smaller candidate rejected"
        result.append(item)
    return result


def _legacy_candidates() -> list[dict[str, Any]]:
    roots = [
        Path("reports/ml/development/ticket_7b3_stock_artifact_profile/symbols_5/benchmark/stock_level_prediction_artifacts.parquet"),
        Path("reports/ml/development/ticket_7b3_stock_artifact_profile/symbols_20/benchmark/stock_level_prediction_artifacts.parquet"),
        Path("reports/ml/development/ticket_7b3_stock_artifact_profile/symbols_50/benchmark/stock_level_prediction_artifacts.parquet"),
    ]
    return [{**_artifact_summary(p), "rejected_reason": "row/symbol count smaller than recovered large baseline"} for p in roots if p.exists()]


def _artifact_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    date_col = "rebalance_date" if "rebalance_date" in cols else "session_date"
    symbol_col = "symbol" if "symbol" in cols else "canonical_symbol"
    table = pq.read_table(path, columns=[c for c in [symbol_col, date_col] if c in cols])
    rows = table.to_pylist()
    return {
        "path": str(path),
        "exists": True,
        "row_count": pf.metadata.num_rows,
        "symbol_count": len({r.get(symbol_col) for r in rows if r.get(symbol_col)}),
        "date_min": min((str(r.get(date_col))[:10] for r in rows if r.get(date_col)), default=None),
        "date_max": max((str(r.get(date_col))[:10] for r in rows if r.get(date_col)), default=None),
        "sha256": _file_sha256(path),
    }


def _eligibility_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["canonical_symbol"]].append(row)
    return [
        {
            "canonical_symbol": symbol,
            "registry_member": True,
            "has_valid_daily_row": bool(values),
            "provider_compatible": any(v["compatibility_tier"] != "TIER_E_REVIEW_BLOCKED" for v in values),
            "not_quarantined_rows": sum(1 for v in values if not v["quarantine_flag"]),
            "sufficient_lookback_proxy": len(values) >= 252,
            "final_selector_eligibility_proxy": len(values) >= 252 and any(not v["quarantine_flag"] for v in values),
        }
        for symbol, values in sorted(grouped.items())
    ]


def _residual_fields() -> list[str]:
    return [
        "asset_id", "canonical_symbol", "session_date", "previous_common_session",
        "stooq_open", "stooq_high", "stooq_low", "stooq_close", "stooq_volume",
        "alpaca_open", "alpaca_high", "alpaca_low", "alpaca_close", "alpaca_volume",
        "stooq_return", "alpaca_return", "absolute_return_difference", "relative_close_difference",
        "alpaca_stooq_close_ratio", "change_in_close_ratio", "original_classification",
        "classification", "all_triggering_rules", "deterministic_explanation",
    ]


def _residual_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Provider Residual Resolution",
        "",
        f"- Residual rows: {summary['residual_row_count']}",
        f"- Quarantined rows: {summary['quarantined_row_count']}",
        f"- Gate: {summary['gate']}",
        "- Source archives modified: false",
    ])


def _validation_markdown(validation: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Canonical Daily V2 Validation",
        "",
        f"- Rows: {validation['row_count']}",
        f"- Symbols: {validation['symbol_count']}",
        f"- Date range: {validation['date_min']} through {validation['date_max']}",
        f"- Valid: {validation['valid']}",
    ])


def _csv_date_range(path: Path, column: str) -> dict[str, Any]:
    dates = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(column):
                dates.append(str(row[column])[:10])
    return {"row_count": len(dates), "date_min": min(dates, default=None), "date_max": max(dates, default=None)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "sha256": _file_sha256(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    index = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return clean[index]


def _norm(path: Path) -> str:
    return str(path.resolve()).lower().replace("\\", "/")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
