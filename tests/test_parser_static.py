from app.utils import parse_tickers


def test_parse_tickers_accepts_comma_space_single_string():
    assert parse_tickers("PETR4, VALE3") == ["PETR4.SA", "VALE3.SA"]


def test_parse_tickers_accepts_argparse_split_tokens():
    assert parse_tickers(["PETR4,", "VALE3", "ITUB4"]) == ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]


def test_parse_tickers_accepts_semicolon_and_spaces():
    assert parse_tickers("PETR4; VALE3 ITUB4") == ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
