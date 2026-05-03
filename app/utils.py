from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
import json
import re


def normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not value:
        raise ValueError("empty ticker")
    return value if value.endswith(".SA") or value.startswith("^") or "=" in value else f"{value}.SA"


def safe_ticker(ticker: str) -> str:
    return normalize_ticker(ticker).replace(".", "_").replace("=", "_").replace("^", "")


def run_id(prefix: str, ticker: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{safe_ticker(ticker)}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def parse_tickers(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        text = raw
    else:
        text = " ".join(str(item) for item in raw)
    parts = [p for p in re.split(r"[\s,;]+", text.strip()) if p]
    return [normalize_ticker(p) for p in parts]
