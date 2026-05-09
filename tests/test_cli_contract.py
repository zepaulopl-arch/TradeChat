import pytest

from app.cli import ROOT_COMMANDS, build_parser


def _command_action(parser):
    return next(action for action in parser._actions if getattr(action, "dest", None) == "command")


def test_help_shows_exactly_six_root_commands():
    parser = build_parser()
    help_text = parser.format_help()
    public = [action.dest for action in _command_action(parser)._choices_actions]
    assert tuple(public) == ROOT_COMMANDS
    assert tuple(public) == ("data", "train", "signal", "validate", "refine", "portfolio")
    assert "predict" not in help_text
    assert "trade.py report" not in help_text


def test_removed_root_commands_do_not_parse():
    parser = build_parser()
    for command in ("predict", "report"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_signal_subcommands_exist():
    parser = build_parser()
    assert parser.parse_args(["signal", "generate", "PETR4"]).signal_action == "generate"
    assert parser.parse_args(["signal", "rank", "--list", "validacao"]).signal_action == "rank"
    assert parser.parse_args(["signal", "report", "PETR4"]).signal_action == "report"


def test_data_subcommands_exist_without_legacy_default():
    parser = build_parser()
    assert parser.parse_args(["data", "load", "PETR4.SA"]).data_action == "load"
    assert parser.parse_args(["data", "status", "PETR4.SA"]).data_action == "status"
    assert parser.parse_args(["data", "audit", "PETR4.SA"]).data_action == "audit"
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "PETR4.SA"])


def test_portfolio_subcommands_exist():
    parser = build_parser()
    for action in ("status", "plan", "rebalance", "simulate", "live"):
        assert parser.parse_args(["portfolio", action]).portfolio_action == action


def test_validate_requires_explicit_mode():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "PETR4"])
    assert parser.parse_args(["validate", "PETR4", "--mode", "walkforward"]).mode == "walkforward"
