from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feature_audit_helpers_exist():
    text = (ROOT / 'app' / 'feature_audit.py').read_text(encoding='utf-8')
    assert 'def abbreviate_feature_name' in text
    assert 'def top_selected_features' in text
    assert 'def feature_family_profile' in text


def test_train_summary_shows_top_features():
    text = (ROOT / 'app' / 'report.py').read_text(encoding='utf-8')
    assert 'top feats' in text
    assert 'top_5' in text
    assert 'family=' in text
    assert 'relevance=' in text


def test_manifest_stores_feature_audit():
    text = (ROOT / 'app' / 'models.py').read_text(encoding='utf-8')
    assert 'top_features' in text
    assert 'feature_family_profile' in text
