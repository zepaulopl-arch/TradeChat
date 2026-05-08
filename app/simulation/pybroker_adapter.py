from __future__ import annotations

import contextlib
import io
from typing import Any

try:
    from pybroker import Strategy, YFinance, disable_progress_bar
    from pybroker.common import FeeMode, PositionMode
    from pybroker.config import StrategyConfig
    from pybroker.slippage import RandomSlippageModel
except Exception:  # pragma: no cover - optional dependency branch
    Strategy = None
    YFinance = None
    FeeMode = None
    PositionMode = None
    StrategyConfig = None
    RandomSlippageModel = None
    disable_progress_bar = None


def pybroker_available() -> bool:
    return Strategy is not None and YFinance is not None and StrategyConfig is not None


def require_pybroker() -> None:
    if not pybroker_available():
        raise RuntimeError(
            "PyBroker is not available. Install lib-pybroker to run validate/replay or "
            "walk-forward simulation. Core data, train, signal and refine commands still work."
        )
    if disable_progress_bar is not None:
        disable_progress_bar()


@contextlib.contextmanager
def quiet_pybroker() -> Any:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
