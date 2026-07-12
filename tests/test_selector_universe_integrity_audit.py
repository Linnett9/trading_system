import json

from application.cli_runtime import FEEDLESS_MODES
from core.research.ml.stock_level.selector_universe_integrity_audit import (
    build_selector_universe_integrity_audit,
    write_selector_universe_integrity_audit,
)
from core.research.ml.stock_level.stock_level_artifact_io import write_stock_level_artifact


def _rows():
    rows = []
    for day in range(1, 4):
        for symbol in ("AAA", "BBB"):
            rows.append(
                {
                    "rebalance_date": f"2024-01-0{day}",
                    "symbol": symbol,
                    "average_dollar_volume_21d": 1_000_000 + day,
                    "sector": "Tech" if symbol == "AAA" else "",
                }
            )
    return rows


def _settings(tmp_path, **overrides):
    universe = tmp_path / "universe.yaml"
    universe.write_text("name: current_test\nsource: unit_test_static\nsymbols:\n  - AAA\n  - BBB\n  - CCC\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": "unit_adjusted",
                "download_date": "2024-02-01",
                "symbols": [
                    {"symbol": "AAA", "first_date": "2020-01-01", "last_date": "2024-01-31", "adjusted_ohlc": True},
                    {"symbol": "BBB", "first_date": "2020-01-01", "last_date": "2024-01-31", "adjusted_ohlc": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "sector.json"
    mapping.write_text(json.dumps({"AAA": "Technology"}), encoding="utf-8")
    settings = {
        "enabled": True,
        "source_artifact_path": str(tmp_path / "source.parquet"),
        "universe_path": str(universe),
        "price_manifest_path": str(manifest),
        "classification_mapping_path": str(mapping),
        "output_dir": str(tmp_path / "out"),
        "allow_csv_fallback": False,
        "maximum_decision_dates": None,
        "maximum_symbols": None,
        "require_historical_universe_for_promotion": True,
        "minimum_membership_coverage": 0.95,
        "minimum_observation_coverage": 0.90,
        "unknown_membership_action": "block",
    }
    settings.update(overrides)
    return settings


def test_universe_audit_classifies_static_universe_and_blocks_promotion(tmp_path):
    payload = build_selector_universe_integrity_audit(
        _rows(),
        settings=_settings(tmp_path),
        source_path=None,
    )

    assert payload["universe_classification"] == "CURRENT STATIC UNIVERSE"
    assert "historical_universe_membership_unresolved" in payload["promotion_blockers"]
    assert "delisting_return_coverage_unavailable" in payload["promotion_blockers"]
    assert payload["universe_contract"]["unknown_membership_count"] == 3
    assert {row["membership_status"] for row in payload["membership_audit"]} == {
        "member_observed",
        "member_but_not_in_artifact",
    }


def test_security_identity_and_ticker_change_are_unresolved(tmp_path):
    payload = build_selector_universe_integrity_audit(
        _rows(),
        settings=_settings(tmp_path),
        source_path=None,
    )

    assert all(row["mapping_status"] == "UNRESOLVED_STATIC_TICKER" for row in payload["security_identity_mapping"])
    assert all(row["status"] == "NO_SECURITY_MASTER" for row in payload["ticker_change_audit"])


def test_breadth_coverage_distinguishes_unknown_membership_from_missing_observation(tmp_path):
    payload = build_selector_universe_integrity_audit(
        _rows(),
        settings=_settings(tmp_path),
        source_path=None,
    )

    coverage = payload["breadth_universe_coverage"]
    assert all(row["symbols_with_unknown_membership"] == 3 for row in coverage)
    assert all(row["symbols_missing_price_history"] == 1 for row in coverage)
    assert all(row["promotion_status"] == "block_unknown_historical_membership" for row in coverage)


def test_corporate_action_semantics_use_manifest_identity(tmp_path):
    payload = build_selector_universe_integrity_audit(
        _rows(),
        settings=_settings(tmp_path),
        source_path=None,
    )

    audit = payload["corporate_action_audit"]
    assert audit["price_source"] == "unit_adjusted"
    assert audit["adjusted_ohlc_values"] == [True]
    assert audit["compatible_feature_target_semantics"] is True


def test_write_universe_integrity_outputs(tmp_path):
    artifact = tmp_path / "source.parquet"
    write_stock_level_artifact(
        artifact,
        _rows(),
        fieldnames=list(_rows()[0]),
        config={"ml": {"stock_level_artifact_format": "parquet"}},
    )
    paths = write_selector_universe_integrity_audit(
        {
            "ml": {
                "selector_universe_integrity_audit": {
                    **_settings(tmp_path, source_artifact_path=str(artifact)),
                    "enabled": True,
                }
            }
        }
    )

    assert paths.report_json_path.exists()
    assert paths.historical_membership_audit_path.exists()
    payload = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
    assert payload["training_performed"] is False
    assert payload["trading_impact"] == "none"


def test_universe_integrity_cli_is_feedless():
    assert "ml-selector-universe-integrity-audit" in FEEDLESS_MODES

