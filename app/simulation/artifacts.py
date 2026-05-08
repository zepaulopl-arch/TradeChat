from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import write_json


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, summary)
