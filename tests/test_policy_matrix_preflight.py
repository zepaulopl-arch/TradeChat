from argparse import Namespace
from pathlib import Path

import pytest

from tools import run_policy_matrix


def _args(**kwargs):
    defaults = {
        "skip_preflight": False,
        "allow_untrained": False,
        "preflight_sample_size": 10,
        "config": None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_artifact_preflight_aborts_on_missing_models(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_policy_matrix,
        "_artifact_profile_for_ticker",
        lambda config, ticker: {
            "ticker": ticker,
            "has_dir": False,
            "has_signal": False,
            "has_model_artifacts": False,
            "model_file_count": 0,
            "zero_signal": False,
            "quality": 0.0,
            "max_abs_prediction": 0.0,
            "ticker_dir": "artifacts/models/x",
        },
    )

    with pytest.raises(SystemExit) as exc:
        run_policy_matrix._run_artifact_preflight(tmp_path, ["ALOS3.SA"], _args())

    assert "Policy matrix preflight failed" in str(exc.value)
    assert (tmp_path / "preflight_artifacts.csv").exists()


def test_artifact_preflight_allows_smoke_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_policy_matrix,
        "_artifact_profile_for_ticker",
        lambda config, ticker: {
            "ticker": ticker,
            "has_dir": False,
            "has_signal": False,
            "has_model_artifacts": False,
            "model_file_count": 0,
            "zero_signal": False,
            "quality": 0.0,
            "max_abs_prediction": 0.0,
            "ticker_dir": "artifacts/models/x",
        },
    )

    run_policy_matrix._run_artifact_preflight(
        tmp_path,
        ["ALOS3.SA"],
        _args(allow_untrained=True),
    )

    assert (tmp_path / "preflight_artifacts.csv").exists()
