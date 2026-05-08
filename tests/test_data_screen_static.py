from pathlib import Path


def test_data_command_uses_screen_renderer():
    text = Path('app/cli_handlers.py').read_text(encoding='utf-8')
    assert 'print_data_summary' in text
    assert 'data updated | rows=' not in text


def test_data_screen_matches_train_predict_style():
    text = Path('app/report.py').read_text(encoding='utf-8')
    assert 'def print_data_summary' in text
    assert 'banner("TRADECHAT DATA"' in text
    assert 'render_facts' in text
    for label in ['"Status"', '"Rows"', '"Range"', '"Period"', '"Context"', '"Ctx Skipped"', '"Registry"', '"Fundamentals"', '"Sentiment"']:
        assert label in text
