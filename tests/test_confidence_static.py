from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_confidence_uses_configurable_agreement_scale_not_prediction_division_only():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["confidence"]["agreement_scale_return"] == 0.010
    models = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "agreement_scale_return" in models
    assert "minimum_when_engines_exist" in models
    assert "mae_reference_return" in models
    assert "discarded_engine_penalty" in models
    assert "dispersion / max(abs(prediction)" not in models
