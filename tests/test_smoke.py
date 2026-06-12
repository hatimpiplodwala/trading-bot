"""Phase 0 smoke test: the package imports and the test framework is live."""

from __future__ import annotations


def test_package_imports():
    import ict_bot

    assert ict_bot.__version__ == "0.1.0"


def test_entrypoint_callable():
    from ict_bot.main import main

    assert callable(main)
