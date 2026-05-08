from __future__ import annotations

import re
import shutil
import textwrap
from typing import Any, Sequence


class C:
    """Muted ANSI palette shared by CLI reports and scripts."""

    HEADER = "\033[90m"
    BLUE = "\033[38;5;67m"
    CYAN = "\033[38;5;109m"
    GREEN = "\033[38;5;108m"
    YELLOW = "\033[38;5;144m"
    RED = "\033[38;5;131m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def money_br(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def screen_width(minimum: int = 76, maximum: int = 112) -> int:
    width = shutil.get_terminal_size((96, 24)).columns
    return max(minimum, min(width, maximum))


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def paint(text: Any, tone: str = "", *, use_color: bool = True) -> str:
    out = str(text)
    if not use_color:
        return strip_ansi(out)
    if not tone:
        return out
    return f"{tone}{out}{C.RESET}"


def tone_signal(label: str) -> str:
    upper = str(label).upper()
    if "BUY" in upper:
        return C.GREEN
    if "SELL" in upper:
        return C.RED
    return C.RESET


def tone_delta(value: float) -> str:
    return C.GREEN if value >= 0 else C.RED


def divider(width: int | None = None, *, use_color: bool = True) -> str:
    size = width or screen_width()
    line = "-" * size
    return paint(line, C.DIM, use_color=use_color)


def _ellipsis(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3].rstrip() + "..."


def fit_cell(text: Any, width: int) -> str:
    raw = str(text)
    plain = strip_ansi(raw)
    if len(plain) <= width:
        return raw
    clipped = _ellipsis(plain, width)
    match = re.match(r"^(?P<prefix>(?:\x1b\[[0-9;]*m)*)(?P<body>.*?)(?P<suffix>(?:\x1b\[[0-9;]*m)*)$", raw)
    if match and match.group("prefix"):
        return f"{match.group('prefix')}{clipped}{match.group('suffix') or C.RESET}"
    return clipped


def ljust_ansi(text: Any, width: int) -> str:
    raw = fit_cell(text, width)
    return raw + (" " * max(0, width - visible_len(raw)))


def rjust_ansi(text: Any, width: int) -> str:
    raw = fit_cell(text, width)
    return (" " * max(0, width - visible_len(raw))) + raw


def banner(title: str, *parts: Any, width: int | None = None, use_color: bool = True) -> list[str]:
    size = width or screen_width()
    colored_title = paint(title, C.BOLD, use_color=use_color)
    visible_parts = [str(part) for part in parts if str(part).strip()]
    line = colored_title if not visible_parts else f"{colored_title} | {' | '.join(visible_parts)}"
    return [divider(size, use_color=use_color), fit_cell(line, size), divider(size, use_color=use_color)]


def render_facts(
    facts: Sequence[tuple[Any, Any] | tuple[Any, Any, str]],
    *,
    width: int | None = None,
    max_columns: int = 3,
    use_color: bool = True,
) -> list[str]:
    size = width or screen_width()
    prepared: list[tuple[str, str, str]] = []
    for item in facts:
        if len(item) == 2:
            label, value = item
            tone = ""
        else:
            label, value, tone = item
        prepared.append((strip_ansi(str(label)) if not use_color else str(label), strip_ansi(str(value)) if not use_color else str(value), tone))
    if not prepared:
        return []

    columns = max(1, min(max_columns, size // 32, len(prepared)))
    if size < 90:
        columns = 1
    gap = 3
    col_width = max(22, (size - gap * (columns - 1)) // columns)
    rows: list[str] = []
    for start in range(0, len(prepared), columns):
        chunk = prepared[start : start + columns]
        parts: list[str] = []
        for label, value, tone in chunk:
            value_width = max(8, col_width - 13)
            shown_label = _ellipsis(label, 12)
            shown_value = _ellipsis(value, value_width)
            label_text = paint(f"{shown_label:<12}", C.DIM, use_color=use_color)
            value_text = paint(shown_value, tone, use_color=use_color)
            parts.append(ljust_ansi(f"{label_text} {value_text}", col_width))
        rows.append((" " * gap).join(parts).rstrip())
    return rows


def render_wrapped(label: str, value: Any, *, width: int | None = None, use_color: bool = True) -> list[str]:
    size = width or screen_width()
    prefix = f"{label}: "
    body_width = max(12, size - len(prefix))
    source = strip_ansi(str(value)) if not use_color else str(value)
    chunks = textwrap.wrap(source, width=body_width) or [""]
    out: list[str] = []
    for idx, chunk in enumerate(chunks):
        label_text = paint(prefix if idx == 0 else " " * len(prefix), C.DIM, use_color=use_color)
        out.append(f"{label_text}{chunk}")
    return out


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    width: int | None = None,
    aligns: Sequence[str] | None = None,
    min_widths: Sequence[int] | None = None,
    use_color: bool = True,
) -> list[str]:
    size = width or screen_width()
    if not headers:
        return []

    aligns = aligns or ["left"] * len(headers)
    mins = list(min_widths or [max(5, min(len(h), 12)) for h in headers])
    cell_rows = [
        [strip_ansi(str(cell)) if not use_color else str(cell) for cell in row]
        for row in rows
    ]
    widths = [visible_len(strip_ansi(str(header)) if not use_color else str(header)) for header in headers]
    for row in cell_rows:
        for idx, cell in enumerate(row[: len(widths)]):
            widths[idx] = max(widths[idx], visible_len(cell))

    separator = " | "

    def total_width() -> int:
        return sum(widths) + len(separator) * (len(widths) - 1)

    while total_width() > size:
        shrinkable = [idx for idx, value in enumerate(widths) if value > mins[idx]]
        if not shrinkable:
            break
        widest = max(shrinkable, key=lambda idx: widths[idx])
        widths[widest] -= 1

    def format_row(values: Sequence[str]) -> str:
        cells: list[str] = []
        for idx, value in enumerate(values):
            align = aligns[idx] if idx < len(aligns) else "left"
            if align == "right":
                cells.append(rjust_ansi(value, widths[idx]))
            else:
                cells.append(ljust_ansi(value, widths[idx]))
        return separator.join(cells)

    header_row = format_row([paint(header, C.DIM, use_color=use_color) for header in headers])
    lines = [header_row, divider(min(size, total_width()), use_color=use_color)]
    lines.extend(format_row(row[: len(headers)]) for row in cell_rows)
    return lines
