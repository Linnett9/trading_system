from __future__ import annotations

import copy

from core.research.ml.selector_campaign_launch_gate import (
    build_selector_campaign_launch_readiness,
)
from core.research.ml.selector_research_campaign import (
    build_selector_research_campaign,
    historical_stage10_baseline_campaign,
)
from core.research.ml.selector_research_protocol import (
    REQUIRED_IDENTITIES,
    freeze_selector_research_protocol,
)


def test_research_gate_admits_phase_c_without_claiming_data_readiness():
    protocol = _protocol()
    campaign = build_selector_research_campaign(protocol)

    report = build_selector_campaign_launch_readiness(
        protocol=protocol,
        campaign=campaign,
        campaign_selection="research",
        source_commit="fixture",
    )

    assert campaign["expected_component_count"] == 75
    assert report["validated_component_count"] == 75
    assert report["readiness_status"] == "READY_FOR_OPERATIONAL_INPUTS"
    assert report["blocked_components"] == []
    assert report["training_performed"] is False
    assert report["dataset_read_performed"] is False


def test_historical_selection_and_operational_identity_states():
    protocol = _protocol()
    campaign = historical_stage10_baseline_campaign()

    waiting = build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=campaign, campaign_selection="historical",
        source_commit="fixture",
    )
    synthetic = build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=campaign, campaign_selection="historical",
        operational_inputs={
            "selector_dataset_identity": "synthetic",
            "selector_dataset_checksum": "synthetic",
            "parent_gate_identity": "synthetic",
            "parent_gate_checksum": "synthetic",
            "operational_input_identity": "synthetic",
            "operational_input_checksum": "synthetic",
            "training_boundary_identity": "synthetic",
            "training_boundary_checksum": "synthetic",
        },
        source_commit="fixture",
    )
    ready = build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=campaign, campaign_selection="historical",
        operational_inputs={
            "selector_dataset_identity": "A" * 64,
            "selector_dataset_checksum": "B" * 64,
            "parent_gate_identity": "C" * 64,
            "parent_gate_checksum": "D" * 64,
            "operational_input_identity": "E" * 64,
            "operational_input_checksum": "F" * 64,
            "training_boundary_identity": "1" * 64,
            "training_boundary_checksum": "2" * 64,
        },
        source_commit="fixture",
    )

    assert waiting["expected_component_count"] == 15
    assert waiting["readiness_status"] == "READY_FOR_OPERATIONAL_INPUTS"
    assert synthetic["readiness_status"] == "READY_FOR_OPERATIONAL_INPUTS"
    assert ready["readiness_status"] == "READY_TO_LAUNCH"


def test_campaign_selection_and_duplicate_ownership_fail_closed():
    protocol = _protocol()
    historical = historical_stage10_baseline_campaign()
    mismatch = build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=historical, campaign_selection="research",
        source_commit="fixture",
    )
    duplicate = copy.deepcopy(historical)
    duplicate["fitted_component_matrix"][-1] = copy.deepcopy(
        duplicate["fitted_component_matrix"][0]
    )
    duplicate["logical_checksum"] = _rehash(duplicate)
    invalid = build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=duplicate, campaign_selection="historical",
        source_commit="fixture",
    )

    assert mismatch["readiness_status"] == "INVALID_CAMPAIGN"
    assert invalid["readiness_status"] == "INVALID_CAMPAIGN"


def _protocol():
    return freeze_selector_research_protocol(
        campaign_identity="synthetic-selector-campaign",
        frozen_identities={
            name: {"identity": f"synthetic-{name}", "checksum": f"checksum-{name}"}
            for name in REQUIRED_IDENTITIES
        },
        source_commit="fixture",
    )


def _rehash(value):
    import hashlib
    import json

    payload = {key: item for key, item in value.items() if key != "logical_checksum"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()
