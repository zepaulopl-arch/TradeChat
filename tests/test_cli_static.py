from app.cli import build_parser


def test_parser_has_main_commands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["data", "train", "signal", "validate", "refine", "portfolio"]:
        assert command in help_text
    assert "predict" not in help_text
    assert "trade.py report" not in help_text
    assert "daily" not in help_text
    assert "buy" not in help_text


def test_data_has_explicit_subcommands_without_cvm_flag():
    parser = build_parser()
    data_parser = next(
        action for action in parser._actions if getattr(action, "dest", None) == "command"
    ).choices["data"]
    help_text = data_parser.format_help()
    assert "--cvm" not in help_text
    choices = data_parser._subparsers._group_actions[0].choices
    assert set(choices) == {"load", "status", "audit"}


def test_removed_operational_aliases_are_not_public_commands():
    parser = build_parser()
    choices = next(
        action for action in parser._actions if getattr(action, "dest", None) == "command"
    ).choices
    assert "daily" not in choices
    assert "buy" not in choices
    assert "predict" not in choices
    assert "report" not in choices


def test_signal_validate_refine_and_portfolio_modes_parse():
    parser = build_parser()
    signal = parser.parse_args(["signal", "rank", "PETR4"])
    assert signal.signal_action == "rank"
    port_action = parser.parse_args(["portfolio", "rebalance"])
    assert port_action.portfolio_action == "rebalance"
    live = parser.parse_args(["portfolio", "live"])
    assert live.portfolio_action == "live"
    val = parser.parse_args(["validate", "PETR4", "--mode", "walkforward"])
    assert val.mode == "walkforward"
    refine = parser.parse_args(["refine", "PETR4"])
    assert refine.tickers == ["PETR4"]
    removal = parser.parse_args(
        ["refine", "PETR4", "--removal", "--horizons", "d1", "--profiles", "full,technical_only"]
    )
    assert removal.removal is True
    assert removal.horizons == "d1"
    removal_wf = parser.parse_args(
        [
            "refine",
            "PETR4",
            "--removal",
            "--walkforward",
            "--start",
            "2026-01-01",
            "--end",
            "2026-04-01",
        ]
    )
    assert removal_wf.walkforward is True
    assert removal_wf.start == "2026-01-01"


def test_removed_portfolio_flags_do_not_parse():
    parser = build_parser()
    for flag in ("--live", "--rebalance"):
        try:
            parser.parse_args(["portfolio", flag])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"portfolio {flag} must not parse")
