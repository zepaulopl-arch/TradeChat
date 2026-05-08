from pathlib import Path

import pandas as pd

from app.refine_service import run_feature_removal


def test_refine_shadow_does_not_touch_operational_artifacts(monkeypatch, tmp_path):
    operational_model = tmp_path / "models" / "PETR4_SA" / "latest_train_d1.json"
    operational_model.parent.mkdir(parents=True)
    operational_model.write_text('{"run_id": "operational"}', encoding="utf-8")

    def fake_load_prices(cfg, ticker, update=False):
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        return pd.DataFrame({ticker: [10.0, 10.1, 10.2, 10.3, 10.4]}, index=idx)

    def fake_build_dataset(cfg, prices, ticker):
        idx = prices.index
        return (
            pd.DataFrame({"ret_1": [0.0, 0.01, 0.01, 0.01, 0.01]}, index=idx),
            pd.DataFrame({"target_return_d1": [0.01, 0.01, 0.01, 0.01, None]}, index=idx),
            {"latest_price": 10.4},
        )

    def fake_train_models(cfg, ticker, X, y, meta, autotune=False, horizon="d1", inner_threads=1):
        artifact_dir = Path(cfg["app"]["artifact_dir"])
        assert "refine" in artifact_dir.parts
        assert artifact_dir != tmp_path
        return {
            "run_id": "shadow",
            "features": ["ret_1"],
            "metrics": {"ridge_arbiter": {"mae_return": 0.01}},
            "latest_prediction_return": 0.01,
            "confidence": 0.5,
        }

    monkeypatch.setattr("app.refine_service.load_prices", fake_load_prices)
    monkeypatch.setattr("app.refine_service.build_dataset", fake_build_dataset)
    monkeypatch.setattr("app.refine_service.train_models", fake_train_models)

    summary = run_feature_removal(
        {
            "app": {"artifact_dir": str(tmp_path)},
            "features": {
                "technical": {"enabled": True},
                "context": {"enabled": True},
                "fundamentals": {"enabled": True},
                "sentiment": {"enabled": True},
            },
        },
        ["PETR4.SA"],
        horizons="d1",
        profiles="full,no_context",
    )

    assert operational_model.read_text(encoding="utf-8") == '{"run_id": "operational"}'
    assert all("refine" in Path(row["artifact_dir"]).parts for row in summary["rows"])
    assert Path(summary["artifacts"]["decision_matrix_csv"]).exists()
