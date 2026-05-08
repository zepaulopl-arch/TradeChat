from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_requirements_are_split_by_operational_layer():
    expected = [
        "requirements-core.txt",
        "requirements-ml.txt",
        "requirements-sentiment.txt",
        "requirements-dev.txt",
    ]
    main = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for filename in expected:
        assert (ROOT / filename).exists()
        assert f"-r {filename}" in main


def test_heavy_optional_dependencies_are_not_in_core_requirements():
    core = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    ml = (ROOT / "requirements-ml.txt").read_text(encoding="utf-8")
    sentiment = (ROOT / "requirements-sentiment.txt").read_text(encoding="utf-8")
    assert "xgboost" not in core
    assert "catboost" not in core
    assert "nltk" not in core
    assert "lib-pybroker" in ml
    assert "feedparser" in sentiment
