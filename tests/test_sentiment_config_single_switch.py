from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_sentiment_has_single_yaml_switch_inside_selection_family():
    cfg = yaml.safe_load((ROOT / "config" / "features.yaml").read_text(encoding="utf-8"))
    default_sentiment = cfg["selection"]["profiles"][cfg["selection"]["active_profile"]][
        "families"
    ]["sentiment"]
    sentiment_preset = cfg["families"]["sentiment"]["presets"][default_sentiment["preset"]]
    assert "enabled" in default_sentiment
    assert "enabled_by_default" not in sentiment_preset
    assert "as_feature" not in sentiment_preset
    assert sentiment_preset["collection"]["max_news_entries"] == 20
    assert sentiment_preset["collection"]["vader_weight"] == 1.0


def test_sentiment_not_controlled_by_cli_flag():
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    assert "--with-news" not in cli
    assert "with_news" not in cli
