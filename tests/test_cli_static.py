from app.cli import build_parser


def test_parser_has_main_commands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["data", "train", "predict", "report", "daily"]:
        assert command in help_text


def test_data_is_single_command_without_cvm_flag():
    parser = build_parser()
    data_parser = next(action for action in parser._actions if getattr(action, "dest", None) == "command").choices["data"]
    help_text = data_parser.format_help()
    assert "tickers" in help_text
    assert "--cvm" not in help_text
    assert "update" not in data_parser._subparsers._group_actions[0].choices if getattr(data_parser, "_subparsers", None) else True


def test_daily_has_no_train_or_update_flag():
    parser = build_parser()
    daily_parser = next(action for action in parser._actions if getattr(action, "dest", None) == "command").choices["daily"]
    help_text = daily_parser.format_help()
    assert "--with-news" not in help_text
    assert "--update" not in help_text
    assert "train" not in help_text.lower()
