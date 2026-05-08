from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _preset(family: str, preset: str):
    cfg = yaml.safe_load((ROOT / "config" / "features.yaml").read_text(encoding="utf-8"))
    return cfg["families"][family]["presets"][preset]


def test_context_registry_exists_and_cli_is_not_expanded():
    registry = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    assert "PETR4.SA" in registry["assets"]
    assert "^BVSP" in registry["assets"]["PETR4.SA"]["context_tickers"]
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    assert 'sub.add_parser("context"' not in cli
    assert 'sub.add_parser("sentiment"' not in cli
    assert 'sub.add_parser("fundamentals"' not in cli


def test_context_has_temporal_beta_and_no_macro_default():
    technical = _preset("technical", "standard")
    context = _preset("context", "asset_linked")
    assert technical["features"]["macro_features"] is False
    assert context["features"]["beta"] is True
    assert context["windows"]["beta"] == [20, 60]
    text = (ROOT / "app" / "context.py").read_text(encoding="utf-8")
    assert "_rolling_beta" in text
    assert "ctx_{clean}_beta_{w}" in text
    assert "asset_context_registry_from_data_cache" in text


def test_fundamentals_snapshot_not_training_default():
    fund = _preset("fundamentals", "temporal_cvm_safe")
    assert fund["safety"]["require_historical"] is True
    assert fund["safety"]["use_snapshot_as_features"] is False
    assert fund["safety"]["snapshot_only_in_report"] is True
    text = (ROOT / "app" / "fundamentals.py").read_text(encoding="utf-8")
    assert "snapshot_report_only_no_temporal_features" in text
    assert "yfinance_snapshot_as_features" in text


def test_sentiment_temporal_feature_cache():
    sent = _preset("sentiment", "temporal_cache")
    assert sent["enabled"] is True
    assert sent["mode"] == "temporal_feature"
    assert sent["windows"]["mean"] == [1, 3, 7]
    text = (ROOT / "app" / "sentiment.py").read_text(encoding="utf-8")
    assert "load_sentiment_daily_series" in text
    assert "sent_mean_{w}d" in text
    assert "sent_delta_{w}d" in text
    assert "sent_std_{w}d" in text
