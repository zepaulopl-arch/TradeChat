from pathlib import Path

from app import ranking_service
from app.asset_eligibility import resolve_asset_eligibility


def _write_eligibility(tmp_path: Path) -> dict:
    (tmp_path / "asset_eligibility.yaml").write_text(
        """
asset_eligibility:
  default_status: untested
  aggressive_profiles: [relaxed]
  assets:
    BBAS3.SA:
      status: blocked_aggressive
      best_profile: none
      reason: "relaxed bad in replay"
    PETR4.SA:
      status: observe
      best_profile: relaxed
      reason: "replay positive"
""".strip() + "\n",
        encoding="utf-8",
    )
    return {"_config_dir": str(tmp_path)}


def test_asset_eligibility_blocks_aggressive_profile(tmp_path):
    cfg = _write_eligibility(tmp_path)

    blocked = resolve_asset_eligibility(cfg, "BBAS3.SA", profile="relaxed")
    allowed = resolve_asset_eligibility(cfg, "BBAS3.SA", profile="balanced")

    assert blocked["blocked"] is True
    assert blocked["status"] == "blocked_aggressive"
    assert allowed["blocked"] is False


def test_rank_neutralizes_blocked_eligibility(monkeypatch, tmp_path):
    cfg = _write_eligibility(tmp_path)
    fake_signals = [
        {
            "ticker": "BBAS3.SA",
            "policy": {"label": "BUY", "horizon": "d20", "quality_pct": 80.0},
            "horizons": {"d20": {"prediction_return": 0.02, "quality": 0.8}},
        }
    ]
    monkeypatch.setattr(ranking_service, "iter_latest_signals", lambda cfg: fake_signals)
    monkeypatch.setattr(ranking_service, "active_policy_profile", lambda cfg: "relaxed")
    monkeypatch.setattr(ranking_service, "signal_score", lambda data: 99.0)
    monkeypatch.setattr(ranking_service, "signal_priority", lambda data: 3)
    monkeypatch.setattr(ranking_service, "trigger_horizon", lambda data: data["policy"]["horizon"])
    monkeypatch.setattr(
        ranking_service,
        "trigger_result",
        lambda data: {"prediction_return": 0.02},
    )

    rows = ranking_service.collect_ranked_signals(cfg, tickers=["BBAS3.SA"], limit=10)

    assert rows[0]["signal"] == "NEUTRAL"
    assert rows[0]["priority"] == 0
    assert rows[0]["score"] == 0.0
    assert rows[0]["eligibility_blocked"] is True
    assert "blocked_aggressive" in rows[0]["eligibility"]
