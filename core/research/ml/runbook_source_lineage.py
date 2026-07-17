from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNBOOK_SOURCE_BOUNDARY = (
    "scripts/selector_parent_publication_runbook.ps1",
    "application/cli_parser.py",
    "application/cli_dispatch.py",
    "application/cli_runtime.py",
    "application/services/ml_lineage_commands.py",
    "application/services/selector_evaluation_commands.py",
    "core/research/ml/runbook_source_lineage.py",
    "core/research/ml/reference/canonical_assets.py",
    "core/research/ml/reference/daily_stock_spine.py",
    "core/research/ml/selector_publication_gates.py",
    "core/research/ml/selector_component_readiness.py",
    "core/research/ml/stock_level/selector_dataset.py",
    "core/research/ml/stock_level/ordinary_selector_publication.py",
    "scripts/verify_and_register_daily_stock_spine.py",
    "scripts/build_canonical_v2_selector_dataset.py",
    "config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml",
    "config/universes/alpaca_514_symbols.txt",
)


def inspect_runbook_source(repo: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    commit = _git(repo, "rev-parse", "HEAD").strip()
    status = _git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *RUNBOOK_SOURCE_BOUNDARY,
    )
    changes = [line for line in status.splitlines() if line.strip()]
    digest = hashlib.sha256()
    for relative in RUNBOOK_SOURCE_BOUNDARY:
        path = repo / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return {
        "source_commit": commit,
        "clean_working_tree": not changes,
        "source_tree_content_checksum": digest.hexdigest().upper(),
        "source_boundary": list(RUNBOOK_SOURCE_BOUNDARY),
        "changes": changes,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> int:
    payload = inspect_runbook_source(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["clean_working_tree"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
