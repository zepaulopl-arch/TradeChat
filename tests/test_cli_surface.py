import pytest

from app.cli import ROOT_COMMANDS, build_parser


def _root_command_action(parser):
    return next(action for action in parser._actions if getattr(action, "dest", None) == "command")


def test_help_shows_only_vnext_root_commands():
    parser = build_parser()
    help_text = parser.format_help()
    public = [action.dest for action in _root_command_action(parser)._choices_actions]
    assert tuple(public) == ROOT_COMMANDS
    assert tuple(public) == ("data", "train", "signal", "validate", "refine", "portfolio")
    assert "predict" not in help_text
    assert "trade.py report" not in help_text


def test_legacy_root_commands_do_not_exist():
    parser = build_parser()
    for command in ("predict", "report"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_signal_and_portfolio_subcommands_exist():
    parser = build_parser()
    assert parser.parse_args(["signal", "generate", "PETR4.SA"]).signal_action == "generate"
    assert parser.parse_args(["signal", "rank", "--list", "validacao"]).signal_action == "rank"
    assert parser.parse_args(["signal", "report", "PETR4.SA"]).signal_action == "report"
    for action in ("status", "plan", "rebalance", "simulate", "live"):
        assert parser.parse_args(["portfolio", action]).portfolio_action == action
