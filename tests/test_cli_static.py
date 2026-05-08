from app.cli import build_parser


def test_parser_has_main_commands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["data", "train", "predict", "validate", "refine", "report", "portfolio"]:
        assert command in help_text
    assert "daily" not in help_text
    assert "simulate" not in help_text
    assert "buy" not in help_text


def test_data_is_single_command_without_cvm_flag():
    parser = build_parser()
    data_parser = next(action for action in parser._actions if getattr(action, "dest", None) == "command").choices["data"]
    help_text = data_parser.format_help()
    assert "tickers" in help_text
    assert "--cvm" not in help_text
    assert "update" not in data_parser._subparsers._group_actions[0].choices if getattr(data_parser, "_subparsers", None) else True


def test_removed_operational_aliases_are_not_public_commands():
    parser = build_parser()
    choices = next(action for action in parser._actions if getattr(action, "dest", None) == "command").choices
    assert "daily" not in choices
    assert "buy" not in choices
    assert "simulate" not in choices


def test_predict_rank_and_portfolio_rebalance_are_first_class_modes():
    parser = build_parser()
    pred = parser.parse_args(["predict", "PETR4", "--rank"])
    assert pred.rank is True
    port = parser.parse_args(["portfolio", "--rebalance"])
    assert port.rebalance is True
    live = parser.parse_args(["portfolio", "--live"])
    assert live.live is True
    val = parser.parse_args(["validate", "PETR4", "--mode", "walkforward"])
    assert val.mode == "walkforward"
    refine = parser.parse_args(["refine", "PETR4"])
    assert refine.tickers == ["PETR4"]
    removal = parser.parse_args(["refine", "PETR4", "--removal", "--horizons", "d1", "--profiles", "full,technical_only"])
    assert removal.removal is True
    assert removal.horizons == "d1"
    removal_wf = parser.parse_args(["refine", "PETR4", "--removal", "--walkforward", "--start", "2026-01-01", "--end", "2026-04-01"])
    assert removal_wf.walkforward is True
    assert removal_wf.start == "2026-01-01"


def test_portfolio_live_and_rebalance_are_mutually_exclusive():
    parser = build_parser()
    try:
        parser.parse_args(["portfolio", "--live", "--rebalance"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("portfolio --live and --rebalance must be mutually exclusive")
