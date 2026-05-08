import pytest

from app.simulation import pybroker_adapter


def test_pybroker_missing_dependency_fails_gracefully(monkeypatch):
    monkeypatch.setattr(pybroker_adapter, "Strategy", None)
    monkeypatch.setattr(pybroker_adapter, "YFinance", None)
    monkeypatch.setattr(pybroker_adapter, "StrategyConfig", None)

    with pytest.raises(RuntimeError, match="PyBroker is not available"):
        pybroker_adapter.require_pybroker()
