from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_finbert_scoring_foundation import (
    authorization_template,
    build_chunk_plan,
    build_production_request,
    build_runtime_configuration,
    deterministic_run_id,
    logical_identity,
    validate_model_package,
    validate_selection,
)

CANONICAL_ROOT = Path(
    r"C:\Users\Brandon\trading_system\reports\ml"
    r"\regime_transformer_meta_ensemble_v1"
    r"\stock_alpha_news_canonical_corpus_alpaca_benzinga_full_compute_adopted_v2"
)
PACKAGE_ROOT = Path(
    r"C:\Users\Brandon\trading_system\cache\ml\production\model_packages"
    r"\ProsusAI--finbert\4556d13015211d73dccd3fdd39d39232506f3e43"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic false-authorized FinBERT scoring plan."
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    selection = validate_selection(_read(args.selection))
    expected = {
        "maximum_sequence_length": 512, "inference_batch_size": 8,
        "chunk_size": 4096, "worker_count": 1, "inner_thread_count": 1,
        "device_policy": "CPU_ONLY", "memory_request_bytes": 10737418240,
        "retry_limit": 0, "text_ineligible_row_policy": "GOVERNED_EXCLUSION",
        "canonical_row_count": 486757, "expected_eligible_row_count": 486756,
        "expected_excluded_row_count": 1, "coverage_unit": "ARTICLE_SYMBOL_PAIR",
        "certification_contract":
            "stock_alpha_finbert_production_score_store.v1",
    }
    for key, value in expected.items():
        if selection[key] != value:
            raise ValueError(f"GOVERNED_SELECTION_MISMATCH:{key}")
    if (
        selection["expected_eligible_row_count"]
        + selection["expected_excluded_row_count"]
        != selection["canonical_row_count"]
    ):
        raise ValueError("SCORING_POPULATION_RECONCILIATION_FAILED")
    package = validate_model_package(
        PACKAGE_ROOT, expected_model_name="ProsusAI/finbert",
        expected_revision="4556d13015211d73dccd3fdd39d39232506f3e43",
    )
    selection_identity = logical_identity(selection)
    request = build_production_request(
        selection=selection, selection_identity=selection_identity,
        model_package=package, canonical_root=str(CANONICAL_ROOT),
        canonical_identity=
            "9941968417d6632e4386635f57b70a0d628d48113527b6c5dd155a21b7509b77",
        canonical_csv_sha256=
            "74844d2997200cfc69729399bbd4729437cc2ec3806ce0408c51ed9846593ce5",
        canonical_manifest_sha256=
            "1ffacef17fdcbda2f6c04b26497a783aefbe6cd13d2f985c4712106097a2d5ea",
    )
    repeat_request = build_production_request(
        selection=selection, selection_identity=selection_identity,
        model_package=package, canonical_root=str(CANONICAL_ROOT),
        canonical_identity=request["canonical_identity"],
        canonical_csv_sha256=request["canonical_csv_sha256"],
        canonical_manifest_sha256=request["canonical_manifest_sha256"],
    )
    runtime = build_runtime_configuration(request)
    runtime_repeat = build_runtime_configuration(repeat_request)
    runtime_checksum = logical_identity(runtime)
    plan = build_chunk_plan(request)
    repeat_plan = build_chunk_plan(repeat_request)
    run_id = deterministic_run_id(request, runtime_checksum)
    item_ids = [
        "chunk-" + logical_identity({
            "run_id": run_id, "chunk_identity": row["chunk_identity"]
        })[:32]
        for row in plan["chunks"]
    ]
    authorization = authorization_template(
        request, runtime_checksum=runtime_checksum, expected_run_id=run_id
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "scoring_selection.json": selection,
        "scoring_selection_identity.json": {
            "selection_identity": selection_identity,
            "scored_at_utc": request["scored_at_utc"],
            "scored_at_selection_identity":
                request["scored_at_selection_identity"],
        },
        "scoring_request.v1.json": request,
        "scoring_request.v1.repeat.json": repeat_request,
        "runtime_configuration.review-draft.yaml": runtime,
        "runtime_configuration_checksum.json": {
            "first": runtime_checksum,
            "repeat": logical_identity(runtime_repeat),
            "stable": runtime == runtime_repeat,
        },
        "chunk_plan.json": plan,
        "chunk_plan.repeat.json": repeat_plan,
        "chunk_plan_identity.json": {
            "first": plan["chunk_plan_identity"],
            "repeat": repeat_plan["chunk_plan_identity"],
            "stable": plan == repeat_plan,
            "shared_run_id": run_id,
            "item_identities": item_ids,
        },
        "authorization.v1.template.json": authorization,
        "authorization.v1.template.identity.json": {
            "authorization_identity": logical_identity(authorization)
        },
        "identity_comparison.json": {
            "request_stable": request == repeat_request,
            "runtime_stable": runtime == runtime_repeat,
            "chunk_plan_stable": plan == repeat_plan,
            "run_id_stable": run_id == deterministic_run_id(
                repeat_request, logical_identity(runtime_repeat)
            ),
        },
        "output_boundary_validation.json": {
            "absolute": Path(request["score_output_root"]).is_absolute(),
            "absent": not Path(request["score_output_root"]).exists(),
            "isolated_from_canonical": Path(request["score_output_root"]).resolve()
                != CANONICAL_ROOT.resolve(),
            "isolated_from_model": Path(request["score_output_root"]).resolve()
                != PACKAGE_ROOT.resolve(),
        },
    }
    for name, payload in artifacts.items():
        _write(output / name, payload)
    _write(output / "authorization.v1.operator-guidance.md",
           "This false template validates the plan only. Setting authorization "
           "true requires a separate operator-reviewed execution ticket.\n")
    cli = REPO_ROOT / "scripts" / "run_authorized_stock_alpha_finbert_scoring.py"
    command = [
        sys.executable, str(cli), "--request",
        str(output / "scoring_request.v1.json"), "--runtime-config",
        str(output / "runtime_configuration.review-draft.yaml"),
        "--authorization", str(output / "authorization.v1.template.json"),
        "--plan-only",
    ]
    results = []
    for name in ("plan_only_result.json", "plan_only_repeat_result.json"):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        payload = json.loads(result.stdout)
        if payload["status"] != "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION":
            raise RuntimeError("PLAN_ONLY_STATUS_MISMATCH")
        _write(output / name, payload)
        results.append(payload)
    _write(output / "blockers.json", {"blockers": [
        "OPERATOR_EXECUTION_AUTHORIZATION_REQUIRED",
        "REAL_EXECUTION_ADAPTER_REMAINS_INTENTIONALLY_UNWIRED",
    ]})
    _write(output / "warnings.json", {"warnings": [
        "ONE_CANONICAL_ROW_GOVERNED_EXCLUSION",
    ]})
    _write(output / "readiness.json", {
        "classification":
            "FINBERT_SCORING_PLAN_VALIDATED_AWAITING_OPERATOR_AUTHORIZATION",
        "execution_authorized": False, "plan_only_results_stable":
            results[0] == results[1],
    })
    _write(output / "operator_review.md",
           "# FinBERT scoring plan review\n\nThe deterministic plan is valid. "
           "Execution remains unauthorized.\n")
    print(json.dumps({
        "request_identity": request["scoring_request_identity"],
        "runtime_checksum": runtime_checksum,
        "chunk_plan_identity": plan["chunk_plan_identity"],
        "chunk_count": len(plan["chunks"]), "run_id": run_id,
        "authorization_identity": logical_identity(authorization),
        "plan_only_status": results[0]["status"],
    }, sort_keys=True))
    return 0


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value):
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    raise SystemExit(main())
