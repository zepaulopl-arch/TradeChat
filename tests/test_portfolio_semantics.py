from __future__ import annotations

from app.commands.portfolio_command import _display_position_side
from app.portfolio_service import position_side
from app.rebalance_service import _signal_position_side


def test_position_side_uses_signed_shares() -> None:
    assert position_side(10) == "LONG"
    assert position_side(-3) == "SHORT"
    assert position_side(0) == "NONE"


def test_status_display_ignores_stale_flat_side() -> None:
    assert _display_position_side({"shares": 10, "side": "FLAT"}) == "LONG"
    assert _display_position_side({"shares": -4, "side": "FLAT"}) == "SHORT"
    assert _display_position_side({"shares": 0, "side": "LONG"}) == "NONE"


def test_rebalance_derives_position_from_actionable_signal_when_plan_is_ambiguous() -> None:
    buy_signal = {"policy": {"label": "BUY"}}
    sell_signal = {"policy": {"label": "SELL"}}

    assert _signal_position_side(buy_signal, {"side": "FLAT"}) == "LONG"
    assert _signal_position_side(sell_signal, {"side": "FLAT"}) == "SHORT"
    assert _signal_position_side(buy_signal, {"side": "SHORT"}) == "SHORT"
    assert _signal_position_side({"policy": {"label": "NEUTRAL"}}, {"side": "FLAT"}) == "NONE"
