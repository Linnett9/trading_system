from scripts.stock_alpha_news_feature_generation_gate import build_stock_alpha_news_feature_generation_gate


def _coverage(**overrides):
    payload = {
        "valid_official_rows_by_symbol": {"AEM": 9, "ASML": 8, "MSFT": 25},
        "symbols_with_1_to_9_valid_official_rows": ["AEM", "ASML"],
        "provider_timeout_symbols": ["AMAT", "META"],
        "unresolved_provider_timeout_symbols": [],
        "unsafe_reasons": ["one or more covered symbols have fewer than 10 rows"],
        "event_row_mismatch_count": 0,
        "outputs_outside_reports": 0,
        "touches_data_news": False,
    }
    payload.update(overrides)
    return payload


def _preflight(**overrides):
    payload = {
        "duplicate_event_key_count": 0,
        "invalid_rows_by_provider": {},
        "future_timestamp_count": 0,
        "provider_timeout_symbols": ["AMAT", "META"],
        "unresolved_provider_timeout_symbols": [],
        "unsafe_reasons": ["contract ingest preflight is report-only and has not approved feature generation"],
    }
    payload.update(overrides)
    return payload


def _gate(coverage=None, preflight=None, exceptions=("AEM", "ASML")):
    return build_stock_alpha_news_feature_generation_gate(
        coverage_audit=coverage or _coverage(),
        contract_preflight=preflight or _preflight(),
        audited_thin_symbol_exceptions=exceptions,
    )


def test_gate_approves_with_explicit_aem_asml_thin_exceptions() -> None:
    gate = _gate()

    assert gate["approved"] is True
    assert gate["feature_generation_gate_status"] == "approved"
    assert gate["thin_symbols_before_exceptions"] == ["AEM", "ASML"]
    assert gate["thin_symbols_after_exceptions"] == []
    assert gate["audited_thin_symbol_exceptions"] == ["AEM", "ASML"]
    assert gate["recovered_provider_timeout_symbols"] == ["AMAT", "META"]
    assert gate["unresolved_provider_timeout_symbols"] == []
    assert gate["next_allowed_step"] == "build_news_transformer_feature_dataset_report_only"


def test_gate_blocks_when_thin_symbols_are_not_excepted() -> None:
    gate = _gate(exceptions=())

    assert gate["approved"] is False
    assert gate["thin_symbols_after_exceptions"] == ["AEM", "ASML"]
    assert "one or more thin symbols lack audited exceptions" in gate["blocking_reasons"]
    assert gate["next_allowed_step"] == "resolve_feature_generation_gate_blockers"


def test_gate_blocks_when_unapproved_thin_symbol_remains() -> None:
    coverage = _coverage(
        valid_official_rows_by_symbol={"AEM": 9, "ASML": 8, "XYZ": 3, "MSFT": 25},
        symbols_with_1_to_9_valid_official_rows=["AEM", "ASML", "XYZ"],
    )

    gate = _gate(coverage=coverage)

    assert gate["approved"] is False
    assert gate["thin_symbols_after_exceptions"] == ["XYZ"]


def test_historical_timeouts_do_not_block_but_unresolved_timeouts_do() -> None:
    recovered_gate = _gate()
    unresolved_gate = _gate(
        preflight=_preflight(
            provider_timeout_symbols=["AMAT", "META"],
            unresolved_provider_timeout_symbols=["META"],
        )
    )

    assert recovered_gate["approved"] is True
    assert unresolved_gate["approved"] is False
    assert unresolved_gate["unresolved_provider_timeout_symbols"] == ["META"]
    assert "unresolved provider timeout symbols remain" in unresolved_gate["blocking_reasons"]


def test_hard_contract_quality_blockers() -> None:
    duplicate_gate = _gate(preflight=_preflight(duplicate_event_key_count=1))
    invalid_gate = _gate(preflight=_preflight(invalid_rows_by_provider={"sec_company_filings": 1}))
    future_gate = _gate(preflight=_preflight(future_timestamp_count=1))

    assert duplicate_gate["approved"] is False
    assert "duplicate provider/symbol/url/timestamp event keys detected" in duplicate_gate["blocking_reasons"]
    assert invalid_gate["approved"] is False
    assert "one or more provider rows failed common schema validation" in invalid_gate["blocking_reasons"]
    assert future_gate["approved"] is False
    assert "one or more rows have future published_at_utc timestamps" in future_gate["blocking_reasons"]


def test_gate_does_not_emit_generation_or_training_outputs() -> None:
    gate = _gate()
    rendered = str(gate).lower()

    assert "raw_write" not in rendered
    assert "model_training" not in rendered
    assert "transformer_output" not in rendered
    assert gate["next_allowed_step"] == "build_news_transformer_feature_dataset_report_only"
