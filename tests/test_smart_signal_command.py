import argparse

from app.commands import signal_command


def test_smart_signal_uses_runtime_policy_selection(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "PETR4.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        calls["resolved_ticker"] = ticker
        calls["fallback"] = fallback

        return {
            "profile": "aggressive",
            "source": "policy_matrix",
            "evidence": {
                "profit_factor": 1.42,
                "trades": 34,
            },
            "selection": {
                "metric": "profit_factor",
            },
        }

    def fake_apply_policy_profile(
        cfg,
        profile,
    ):
        calls["applied_profile"] = profile

        new_cfg = dict(cfg)
        new_cfg["applied_profile"] = profile
        return new_cfg

    def fake_make_signal(
        cfg,
        ticker,
        update=False,
    ):
        calls["signal_cfg"] = cfg
        calls["signal_ticker"] = ticker
        calls["signal_update"] = update

        return {
            "ticker": ticker,
            "label": "BUY",
            "policy": {
                "profile": cfg["applied_profile"],
            },
        }

    def fake_latest_signal_path(
        cfg,
        ticker,
    ):
        return tmp_path / ticker.replace(".", "_") / "latest_signal.json"

    def fake_print_signal(
        signal,
        verbose=False,
        diagnostic=False,
        cfg=None,
    ):
        calls["printed"] = True
        calls["printed_signal"] = signal
        calls["verbose"] = verbose
        calls["diagnostic"] = diagnostic

    monkeypatch.setattr(
        signal_command,
        "resolve_cli_tickers",
        fake_resolve_cli_tickers,
    )

    monkeypatch.setattr(
        signal_command,
        "resolve_policy_selection",
        fake_resolve_policy_selection,
    )

    monkeypatch.setattr(
        signal_command,
        "apply_policy_profile",
        fake_apply_policy_profile,
    )

    monkeypatch.setattr(
        signal_command,
        "make_signal",
        fake_make_signal,
    )

    monkeypatch.setattr(
        signal_command,
        "latest_signal_path",
        fake_latest_signal_path,
    )

    monkeypatch.setattr(
        signal_command,
        "print_signal",
        fake_print_signal,
    )

    args = argparse.Namespace(
        tickers=[
            "PETR4.SA",
        ],
        asset_list=None,
        policy_profile=None,
        update=True,
        verbose=True,
        diagnostic=True,
    )

    signal_command._smart(
        {},
        args,
    )

    assert calls["resolved_ticker"] == "PETR4.SA"
    assert calls["fallback"] is None
    assert calls["applied_profile"] == "aggressive"
    assert calls["signal_ticker"] == "PETR4.SA"
    assert calls["signal_update"] is True
    assert calls["printed"] is True
    assert calls["verbose"] is True
    assert calls["diagnostic"] is True

    smart_path = tmp_path / "PETR4_SA" / "latest_smart_signal.json"

    assert smart_path.exists()

    written = smart_path.read_text(
        encoding="utf-8",
    )

    assert '"smart_signal"' in written
    assert '"profile": "aggressive"' in written
    assert '"source": "policy_matrix"' in written
    assert '"profit_factor": 1.42' in written


def test_smart_signal_passes_policy_profile_as_fallback(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "VALE3.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        calls["resolved_ticker"] = ticker
        calls["fallback"] = fallback

        return {
            "profile": fallback,
            "source": "fallback",
            "evidence": {},
            "selection": {},
        }

    def fake_apply_policy_profile(
        cfg,
        profile,
    ):
        calls["applied_profile"] = profile

        new_cfg = dict(cfg)
        new_cfg["applied_profile"] = profile
        return new_cfg

    def fake_make_signal(
        cfg,
        ticker,
        update=False,
    ):
        return {
            "ticker": ticker,
            "label": "NEUTRAL",
            "policy": {
                "profile": cfg["applied_profile"],
            },
        }

    def fake_latest_signal_path(
        cfg,
        ticker,
    ):
        return tmp_path / ticker.replace(".", "_") / "latest_signal.json"

    def fake_print_signal(
        signal,
        verbose=False,
        diagnostic=False,
        cfg=None,
    ):
        calls["printed"] = True

    monkeypatch.setattr(
        signal_command,
        "resolve_cli_tickers",
        fake_resolve_cli_tickers,
    )

    monkeypatch.setattr(
        signal_command,
        "resolve_policy_selection",
        fake_resolve_policy_selection,
    )

    monkeypatch.setattr(
        signal_command,
        "apply_policy_profile",
        fake_apply_policy_profile,
    )

    monkeypatch.setattr(
        signal_command,
        "make_signal",
        fake_make_signal,
    )

    monkeypatch.setattr(
        signal_command,
        "latest_signal_path",
        fake_latest_signal_path,
    )

    monkeypatch.setattr(
        signal_command,
        "print_signal",
        fake_print_signal,
    )

    args = argparse.Namespace(
        tickers=[
            "VALE3.SA",
        ],
        asset_list=None,
        policy_profile="balanced",
        update=False,
        verbose=False,
        diagnostic=False,
    )

    signal_command._smart(
        {},
        args,
    )

    assert calls["resolved_ticker"] == "VALE3.SA"
    assert calls["fallback"] == "balanced"
    assert calls["applied_profile"] == "balanced"
    assert calls["printed"] is True


def test_signal_report_prefers_latest_smart_signal(
    tmp_path,
    monkeypatch,
):
    calls = {}
    latest_path = tmp_path / "PETR4_SA" / "latest_signal.json"
    smart_path = tmp_path / "PETR4_SA" / "latest_smart_signal.json"
    smart_path.parent.mkdir(parents=True)
    latest_path.write_text('{"ticker": "PETR4.SA", "plain": true}', encoding="utf-8")
    smart_path.write_text('{"ticker": "PETR4.SA", "smart_signal": {"enabled": true}}', encoding="utf-8")

    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "PETR4.SA",
        ]

    def fake_latest_signal_path(
        cfg,
        ticker,
    ):
        return latest_path

    def fake_write_txt_report(
        cfg,
        signal,
    ):
        calls["signal"] = signal
        return tmp_path / "report.txt"

    def fake_make_signal(
        cfg,
        ticker,
        update=False,
    ):
        raise AssertionError("report should read latest smart signal when available")

    monkeypatch.setattr(
        signal_command,
        "resolve_cli_tickers",
        fake_resolve_cli_tickers,
    )
    monkeypatch.setattr(
        signal_command,
        "latest_signal_path",
        fake_latest_signal_path,
    )
    monkeypatch.setattr(
        signal_command,
        "write_txt_report",
        fake_write_txt_report,
    )
    monkeypatch.setattr(
        signal_command,
        "make_signal",
        fake_make_signal,
    )

    args = argparse.Namespace(
        tickers=[
            "PETR4.SA",
        ],
        asset_list=None,
        refresh=False,
        update=False,
    )

    signal_command._report(
        {},
        args,
    )

    assert calls["signal"]["smart_signal"]["enabled"] is True
