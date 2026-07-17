from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import application.cli_dispatch as cli_dispatch
import application.cli_parser as cli_parser
import application.cli_runtime as cli_runtime
from application.services.ml_lineage_commands import run_registry_verify
from core.research.ml.reference.canonical_assets import (
    audit_registry,
    build_registry_from_universe,
    write_audit_reports,
    write_registry_outputs,
)
from core.research.ml.runbook_source_lineage import inspect_runbook_source


MODE = "ml-registry-verify"


def _publication(tmp_path: Path, run_id: str = "synthetic-run") -> Path:
    universe = tmp_path / "universe.txt"
    universe.write_text("\n".join(f"S{number:03d}" for number in range(514)) + "\n", encoding="utf-8")
    assets, aliases, _ = build_registry_from_universe(universe)
    registry = tmp_path / "canonical_asset_registry.csv"
    alias_registry = tmp_path / "provider_symbol_aliases.csv"
    write_registry_outputs(
        assets,
        aliases,
        asset_output=registry,
        alias_output=alias_registry,
        parquet_output=tmp_path / "canonical_asset_registry.parquet",
    )
    audit = audit_registry(assets, aliases, universe_path=universe, repo_root=tmp_path / "empty")
    root = tmp_path / f"run={run_id}"
    write_audit_reports(audit, report_dir=root, registry_path=registry, alias_path=alias_registry)
    return root / "manifest.json"


def test_mode_is_registered_and_parser_accepts_it(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", MODE])
    assert cli_parser.parse_args().mode == MODE


def test_dispatch_routes_only_to_registry_verification_owner(monkeypatch):
    calls = []

    class Commands:
        @staticmethod
        def run_registry_verify(config, args):
            calls.append(("verify", config, args.mode))

        @staticmethod
        def run_base_backtests(*args):
            raise AssertionError("base backtests must not run")

    monkeypatch.setattr(cli_dispatch, "import_module", lambda name: Commands)
    cli_dispatch.dispatch(SimpleNamespace(mode=MODE), {"safe": True}, None)
    assert calls == [("verify", {"safe": True}, MODE)]


def test_runtime_treats_mode_as_feedless(monkeypatch):
    args = SimpleNamespace(mode=MODE, config="synthetic.yaml", profile=None, log_level="info")
    observed = {}
    monkeypatch.setattr(cli_runtime, "parse_args", lambda: args)
    monkeypatch.setattr(cli_runtime, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(cli_runtime, "apply_research_profile", lambda config, profile: config)
    monkeypatch.setattr(cli_runtime, "apply_runtime_overrides", lambda config, parsed: config)
    monkeypatch.setattr(cli_runtime, "build_feed", lambda config: (_ for _ in ()).throw(AssertionError("feed built")))
    monkeypatch.setattr(cli_runtime, "dispatch", lambda parsed, config, feed: observed.update(feed=feed))
    cli_runtime.run_cli()
    assert observed["feed"] is None


def test_valid_synthetic_publication_verifies(tmp_path):
    result = run_registry_verify({}, _args(_publication(tmp_path)))
    assert result["status"] == "READY"
    assert result["canonical_asset_count"] == 514
    assert result["resolved_collection_symbol_count"] == 514
    assert result["feedless"] is True
    assert result["publication_modified"] is False


def test_hash_mismatch_fails_closed(tmp_path):
    manifest = _publication(tmp_path)
    payload = json.loads(manifest.read_text())
    Path(payload["registry_path"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_registry_verify({}, _args(manifest))
    result = json.loads((tmp_path / "verification.json").read_text())
    assert "REGISTRY_ARTIFACT_CHECKSUM_MISMATCH" in result["blockers"]


def test_missing_manifest_fails_closed(tmp_path):
    with pytest.raises(SystemExit):
        run_registry_verify({}, _args(tmp_path / "run=synthetic-run" / "manifest.json"))
    result = json.loads((tmp_path / "verification.json").read_text())
    assert "MANIFEST_MISSING_OR_INVALID" in result["blockers"]


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("unresolved_collection_symbols", ["MISSING"], "UNRESOLVED_COLLECTION_SYMBOLS"),
        ("ambiguous_aliases", [{"provider": "x", "provider_symbol": "S000"}], "AMBIGUOUS_ALIASES"),
    ],
)
def test_bad_audit_state_fails_closed(tmp_path, field, value, blocker):
    manifest = _publication(tmp_path)
    audit_path = manifest.parent / "registry_audit.json"
    audit = json.loads(audit_path.read_text())
    audit[field] = value
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_registry_verify({}, _args(manifest))
    result = json.loads((tmp_path / "verification.json").read_text())
    assert blocker in result["blockers"]


def test_wrong_run_id_fails_closed(tmp_path):
    manifest = _publication(tmp_path)
    with pytest.raises(SystemExit):
        run_registry_verify({}, _args(manifest, run_id="other-run"))
    result = json.loads((tmp_path / "verification.json").read_text())
    assert "RUN_ID_MISMATCH" in result["blockers"]


def test_runbook_stage_three_uses_supported_mode_and_blocks_stage_four():
    text = Path("scripts/selector_parent_publication_runbook.ps1").read_text(encoding="utf-8")
    assert "$stage3Args = @('main.py', '--mode', 'ml-registry-verify', '--config', $selectorConfig" in text
    assert text.count("'--mode', 'ml-registry-verify'") == 1
    assert "'--artifact-manifest', $registryManifest" in text
    assert "'--registry-run-id', $RunId" in text
    stage3 = text.index("            3 {")
    stage4 = text.index("            4 {")
    section = text[stage3:stage4]
    assert "if ($stage3Payload.status -ne 'READY')" in section
    assert "Fail-Stage 3" in section
    assert section.index("if ($stage3Payload.status -ne 'READY')") < section.index("Complete-Stage 3")


def _args(manifest: Path, *, run_id: str = "synthetic-run") -> SimpleNamespace:
    return SimpleNamespace(
        artifact_manifest=str(manifest),
        registry_run_id=run_id,
        verification_output=str(manifest.parents[1] / "verification.json"),
    )


@pytest.mark.parametrize(
    "dirty_relative",
    ["scripts/selector_parent_publication_runbook.ps1", "application/cli_dispatch.py"],
)
def test_source_boundary_blocks_owned_changes_but_ignores_reports(tmp_path, dirty_relative):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    for relative in (
        "scripts/selector_parent_publication_runbook.ps1",
        "application/cli_dispatch.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True)
    report = tmp_path / "reports" / "untracked.json"
    report.parent.mkdir()
    report.write_text("{}", encoding="utf-8")
    assert inspect_runbook_source(tmp_path)["clean_working_tree"] is True
    (tmp_path / dirty_relative).write_text("dirty\n", encoding="utf-8")
    result = inspect_runbook_source(tmp_path)
    assert result["clean_working_tree"] is False
    assert any(dirty_relative in row for row in result["changes"])
