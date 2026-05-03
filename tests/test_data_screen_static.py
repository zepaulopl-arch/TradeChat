from pathlib import Path


def test_data_command_uses_screen_renderer():
    text = Path('app/cli.py').read_text(encoding='utf-8')
    assert 'print_data_summary' in text
    assert 'data updated | rows=' not in text


def test_data_screen_matches_train_predict_style():
    text = Path('app/report.py').read_text(encoding='utf-8')
    assert 'def print_data_summary' in text
    assert 'TRADEGEM DATA |' in text
    for label in ['status      :', 'rows        :', 'range       :', 'period      :', 'context     :', 'registry    :', 'fundamentals:', 'sentiment   :']:
        assert label in text
