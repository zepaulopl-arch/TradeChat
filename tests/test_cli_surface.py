from pathlib import Path

from app.cli import build_parser
from app.commands import signal_command


def _command_action(parser):
    return next(action for action in parser._actions if getattr(action, "dest", None) == "command")


def test_help_shows_only_six_root_commands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["data", "train", "signal", "validate", "refine", "portfolio"]:
        assert command in help_text
    assert "predict" not in help_text
    assert "report" not in help_text


def test_command_count_max_six_public_commands():
    parser = build_parser()
    public = [action.dest for action in _command_action(parser)._choices_actions]
    assert public == ["data", "train", "signal", "validate", "refine", "portfolio"]
    assert len(public) <= 6


def test_predict_deprecated_alias_maps_to_signal_generate():
    parser = build_parser()
    args = parser.parse_args(["predict", "PETR4"])
    assert args.deprecated_alias == "predict"
    assert args.tickers == ["PETR4"]
    args_rank = parser.parse_args(["predict", "PETR4", "--rank"])
    assert args_rank.rank is True


def test_predict_deprecated_alias_emits_warning(monkeypatch, capsys):
    parser = build_parser()
    args = parser.parse_args(["predict", "--rank"])
    monkeypatch.setattr(signal_command, "load_config", lambda path: {})
    monkeypatch.setattr(signal_command, "resolve_cli_tickers", lambda cfg, args, required=False: [])
    monkeypatch.setattr(signal_command, "render_ranking", lambda cfg, limit: [])

    signal_command.run(args)

    assert (
        "Deprecated command: use 'signal generate' instead of 'predict'." in capsys.readouterr().out
    )


def test_report_deprecated_alias_maps_to_signal_report():
    parser = build_parser()
    args = parser.parse_args(["report", "PETR4", "--refresh"])
    assert args.deprecated_alias == "report"
    assert args.refresh is True


def test_report_deprecated_alias_emits_warning(monkeypatch, capsys, tmp_path):
    parser = build_parser()
    args = parser.parse_args(["report", "PETR4"])
    monkeypatch.setattr(signal_command, "load_config", lambda path: {})
    monkeypatch.setattr(
        signal_command, "resolve_cli_tickers", lambda cfg, args, required=True: ["PETR4.SA"]
    )
    monkeypatch.setattr(
        signal_command, "latest_signal_path", lambda cfg, ticker: tmp_path / "missing.json"
    )
    monkeypatch.setattr(
        signal_command, "make_signal", lambda cfg, ticker, update=False: {"ticker": ticker}
    )
    monkeypatch.setattr(signal_command, "write_txt_report", lambda cfg, signal: "report.txt")

    signal_command.run(args)

    out = capsys.readouterr().out
    assert "Deprecated command: use 'signal report' instead of 'report'." in out
    assert "report written: report.txt" in out


def test_signal_has_generate_rank_report_actions():
    parser = build_parser()
    assert parser.parse_args(["signal", "generate", "PETR4"]).signal_action_or_ticker == "generate"
    assert parser.parse_args(["signal", "rank", "PETR4"]).signal_action_or_ticker == "rank"
    assert parser.parse_args(["signal", "report", "PETR4"]).signal_action_or_ticker == "report"


def test_portfolio_default_maps_to_status():
    parser = build_parser()
    args = parser.parse_args(["portfolio"])
    assert args.portfolio_action == "status"


def test_data_default_maps_to_load_if_ticker_given():
    parser = build_parser()
    args = parser.parse_args(["data", "PETR4.SA"])
    assert args.data_action_or_ticker == "PETR4.SA"
    args_load = parser.parse_args(["data", "load", "PETR4.SA"])
    assert args_load.data_action_or_ticker == "load"


def test_no_bat_files_remain():
    root = Path(__file__).resolve().parents[1]
    assert list(root.glob("*.bat")) == []
