from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import application.cli_dispatch as cli_dispatch


def test_cli_imports_with_legacy_sector_reference_path():
    import application.cli  # noqa: F401
    from core.research.ml.sector_reference import load_sector_by_symbol

    assert callable(load_sector_by_symbol)


def test_cli_import_does_not_load_unrelated_champion_command_module():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import sys

class BlockChampionCommands:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'application.services.champion_robustness_commands':
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockChampionCommands())
import application.cli
import application.cli_dispatch
assert 'application.services.champion_robustness_commands' not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], cwd=project_root, check=True)


def test_paper_trial_dispatch_does_not_load_champion_commands(monkeypatch):
    loaded_modules = []

    class PaperCommands:
        @staticmethod
        def run_paper_dry_run(config, feed):
            return None

    def fake_import_module(name):
        loaded_modules.append(name)
        if name.endswith("champion_robustness_commands"):
            raise ModuleNotFoundError(name)
        return PaperCommands

    monkeypatch.setattr(cli_dispatch, "import_module", fake_import_module)

    cli_dispatch.dispatch(SimpleNamespace(mode="paper-trial"), {}, object())

    assert loaded_modules == ["application.services.paper_commands"]
