from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (  # noqa: E402
    authority_bundle,
    clean_source_hash,
    load_contract,
)
from core.research.ml.ds24.clean_v2_data import (  # noqa: E402
    CleanV2CompositeData,
    CleanV2DataError,
)
from scripts.local.ds24_clean_v2_family_worker import (  # noqa: E402
    SHORT_LANES,
    build_clean_refit_schedule,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _family_schedule(
    data: CleanV2CompositeData,
    family: str,
) -> tuple[str, int, Sequence[int] | None, list[Any]]:
    models = load_contract("model_registry.json")
    tournament = load_contract("tournament_contract.json")
    if family in models["families"]:
        family_config = models["families"][family]
        lookback = int(family_config["training"]["lookback_sessions"])
        lane = str(family_config["lane"])
    elif family in models["controls"]:
        lookback = int(tournament["refit_policy"]["default_training_sessions"])
        lane = "CONTROLS"
    else:
        raise CleanV2DataError(f"Family is not runnable: {family}")
    qualifier_years: Sequence[int] | None = (
        tuple(int(value) for value in tournament["qualifier"]["calendar_years"])
        if lane in SHORT_LANES
        else None
    )
    schedule = build_clean_refit_schedule(
        data.decision_spine(),
        lookback_sessions=lookback,
        qualifier_years=qualifier_years,
    )
    if not schedule:
        raise CleanV2DataError(f"Empty schedule for {family}")
    return lane, lookback, qualifier_years, schedule


def run(*, family: str, asset_id: str, year: int) -> dict[str, Any]:
    data = CleanV2CompositeData(ROOT)
    lane, lookback, qualifier_years, schedule = _family_schedule(data, family)
    package = next(
        (
            row
            for row in schedule
            if any(int(date[:4]) == year for date in row.score_session_dates)
        ),
        None,
    )
    if package is None:
        raise CleanV2DataError(
            f"No {family} score package intersects requested year {year}"
        )
    partition = next(
        (
            row
            for row in data.partitions
            if row.asset_id == asset_id and row.year == year
        ),
        None,
    )
    if partition is None:
        raise CleanV2DataError(f"Partition is unavailable: {asset_id}/{year}")
    session_dates = [
        *package.training_session_dates,
        *package.score_session_dates,
    ]
    diagnostics = data.diagnose_feature_partition(partition, session_dates)
    # Exercise the same production method that failed, without assembling other
    # assets, fitting a model, or writing an artifact.
    production = data._read_feature_partition(partition, session_dates)
    assembled = data.assemble_sessions(session_dates, asset_ids=(asset_id,))
    assembled_assets = sorted(set(assembled["canonical_symbol"].astype(str)))
    if assembled.empty or assembled_assets != [asset_id]:
        raise CleanV2DataError(
            "Bounded production assembly did not preserve the requested asset: "
            f"expected={[asset_id]}, actual={assembled_assets}"
        )
    return {
        "classification": "DS24_CLEAN_V2_BOUNDED_READER_PREFLIGHT_PASS",
        "run_id": load_contract("tournament_contract.json")["run_id"],
        "family": family,
        "lane": lane,
        "training_lookback_sessions": lookback,
        "qualifier_years": list(qualifier_years) if qualifier_years else None,
        "refit_ordinal": package.ordinal,
        "refit_timestamp": package.refit_T.isoformat(),
        "training_session_dates": package.training_session_dates,
        "score_session_dates": package.score_session_dates,
        "production_reader_rows": int(len(production)),
        "production_assembly_rows": int(len(assembled)),
        "production_assembly_assets": assembled_assets,
        "clean_source_hash": clean_source_hash(),
        "static_authority_bundle_sha256": authority_bundle()["bundle_sha256"],
        "model_fit_performed": False,
        "data_written": False,
        "diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only preflight for one CLEAN V2 first-package feature partition."
        )
    )
    parser.add_argument("--family", default="random_forest")
    parser.add_argument("--asset", default="AAPL")
    parser.add_argument("--year", type=int, default=2016)
    args = parser.parse_args()
    report = run(family=args.family, asset_id=args.asset, year=args.year)
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
