from __future__ import annotations


def test_cli_imports_with_legacy_sector_reference_path():
    import application.cli  # noqa: F401
    from core.research.ml.sector_reference import load_sector_by_symbol

    assert callable(load_sector_by_symbol)
