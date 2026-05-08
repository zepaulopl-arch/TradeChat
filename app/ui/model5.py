from __future__ import annotations

import os
import re
import shutil
import textwrap
from typing import Any, Mapping, Sequence


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Tone:
    MAIN = "\033[38;2;215;220;226m"
    MUTED = "\033[38;2;159;168;179m"
    LINE = "\033[38;2;58;64;72m"
    INFO = "\033[38;2;108;142;191m"
    OK = "\033[38;2;111;175;123m"
    BAD = "\033[38;2;194;107;107m"
    WARN = "\033[38;2;199;169;107m"
    RESET = "\033[0m"


STATUS_TONES = {
    "info": Tone.INFO,
    "ok": Tone.OK,
    "success": Tone.OK,
    "positive": Tone.OK,
    "warn": Tone.WARN,
    "warning": Tone.WARN,
    "attention": Tone.WARN,
    "error": Tone.BAD,
    "bad": Tone.BAD,
    "negative": Tone.BAD,
    "neutral": Tone.MUTED,
}


def screen_width(minimum: int = 80, maximum: int = 100) -> int:
    width = shutil.get_terminal_size((96, 24)).columns
    return max(minimum, min(width, maximum))


def _use_color(use_color: bool | None = None) -> bool:
    if use_color is not None:
        return bool(use_color)
    return not bool(os.environ.get("NO_COLOR"))


def strip_ansi(text: Any) -> str:
    return ANSI_RE.sub("", str(text))


def visible_len(text: Any) -> int:
    return len(strip_ansi(text))


def _paint(text: Any, tone: str, *, use_color: bool | None = None) -> str:
    raw = str(text)
    if not _use_color(use_color) or not tone:
        return strip_ansi(raw)
    return f"{tone}{raw}{Tone.RESET}"


def _fit(text: Any, width: int) -> str:
    raw = str(text)
    plain = strip_ansi(raw)
    if len(plain) <= width:
        return raw
    if width <= 3:
        return plain[:width]
    return plain[: width - 3].rstrip() + "..."


def _ljust(text: Any, width: int) -> str:
    raw = _fit(text, width)
    return raw + (" " * max(0, width - visible_len(raw)))


def _rjust(text: Any, width: int) -> str:
    raw = _fit(text, width)
    return (" " * max(0, width - visible_len(raw))) + raw


def _line(width: int, *, use_color: bool | None = None) -> str:
    return _paint("-" * width, Tone.LINE, use_color=use_color)


def render_header(title: str, *, width: int | None = None, use_color: bool | None = None) -> list[str]:
    size = width or screen_width()
    clean = " ".join(str(title).upper().replace("—", "-").split())
    return [
        _line(size, use_color=use_color),
        _paint(_fit(clean, size), Tone.MAIN, use_color=use_color),
        _line(size, use_color=use_color),
    ]


def render_section(title: str, *, width: int | None = None, use_color: bool | None = None) -> list[str]:
    size = width or screen_width()
    label = _paint(str(title).upper(), Tone.INFO, use_color=use_color)
    return ["", _fit(label, size)]


def render_key_values(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    width: int | None = None,
    use_color: bool | None = None,
) -> list[str]:
    size = width or screen_width()
    items = list(values.items()) if isinstance(values, Mapping) else list(values)
    if not items:
        return []
    label_width = min(24, max(10, max(len(strip_ansi(k)) for k, _ in items)))
    value_width = max(20, size - label_width - 3)
    out: list[str] = []
    for key, value in items:
        label = _paint(f"{strip_ansi(key):<{label_width}}", Tone.MUTED, use_color=use_color)
        chunks = textwrap.wrap(strip_ansi(value), width=value_width) or [""]
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                out.append(f"{label} : {_paint(chunk, Tone.MAIN, use_color=use_color)}")
            else:
                out.append(f"{' ' * label_width}   {_paint(chunk, Tone.MAIN, use_color=use_color)}")
    return out


def render_badge(text: Any, status: str = "neutral", *, use_color: bool | None = None) -> str:
    tone = STATUS_TONES.get(str(status).lower(), Tone.MUTED)
    return _paint(f"[{strip_ansi(text).upper()}]", tone, use_color=use_color)


def render_callout(
    text: Any,
    status: str = "info",
    *,
    width: int | None = None,
    use_color: bool | None = None,
) -> list[str]:
    size = width or screen_width()
    tone = STATUS_TONES.get(str(status).lower(), Tone.INFO)
    label = render_badge(status, status, use_color=use_color)
    prefix = f"{label} "
    body_width = max(20, size - visible_len(prefix))
    chunks = textwrap.wrap(strip_ansi(text), width=body_width) or [""]
    out: list[str] = []
    for idx, chunk in enumerate(chunks):
        shown_prefix = prefix if idx == 0 else " " * visible_len(prefix)
        out.append(f"{shown_prefix}{_paint(chunk, tone, use_color=use_color)}")
    return out


def render_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    width: int | None = None,
    aligns: Sequence[str] | None = None,
    min_widths: Sequence[int] | None = None,
    use_color: bool | None = None,
) -> list[str]:
    size = width or screen_width()
    if not columns:
        return []
    aligns = aligns or ["left"] * len(columns)
    mins = list(min_widths or [max(6, min(len(strip_ansi(col)), 14)) for col in columns])
    widths = [max(mins[idx], len(strip_ansi(col))) for idx, col in enumerate(columns)]
    clean_rows = [[str(cell) for cell in row[: len(columns)]] for row in rows]
    for row in clean_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], visible_len(cell))

    sep = "  "

    def total() -> int:
        return sum(widths) + len(sep) * (len(widths) - 1)

    while total() > size:
        candidates = [idx for idx, value in enumerate(widths) if value > mins[idx]]
        if not candidates:
            break
        widest = max(candidates, key=lambda idx: widths[idx])
        widths[widest] -= 1

    def row_line(values: Sequence[Any], header: bool = False) -> str:
        cells: list[str] = []
        for idx, value in enumerate(values):
            align = aligns[idx] if idx < len(aligns) else "left"
            raw = str(value)
            cell = raw if ANSI_RE.search(raw) else _paint(raw, Tone.MUTED if header else Tone.MAIN, use_color=use_color)
            cells.append(_rjust(cell, widths[idx]) if align == "right" else _ljust(cell, widths[idx]))
        return sep.join(cells)

    out = [row_line(columns, header=True), _line(size, use_color=use_color)]
    out.extend(row_line(row) for row in clean_rows)
    return out


def render_operational_closing(
    items: Sequence[Any],
    *,
    width: int | None = None,
    use_color: bool | None = None,
) -> list[str]:
    size = width or screen_width()
    out: list[str] = []
    for idx, item in enumerate(items, start=1):
        prefix = _paint(f"{idx}.", Tone.MUTED, use_color=use_color)
        body_width = max(20, size - visible_len(prefix) - 1)
        chunks = textwrap.wrap(strip_ansi(item), width=body_width) or [""]
        for chunk_idx, chunk in enumerate(chunks):
            shown_prefix = prefix if chunk_idx == 0 else " " * visible_len(prefix)
            out.append(f"{shown_prefix} {_paint(chunk, Tone.MAIN, use_color=use_color)}")
    return out
