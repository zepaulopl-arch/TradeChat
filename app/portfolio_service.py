from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ROOT, models_dir
from .utils import normalize_ticker, read_json, safe_ticker


def get_state_db_path(root: Path | None = None) -> Path:
    base = ROOT if root is None else root
    return base / "data" / "tradechat_state.db"


def initial_portfolio(*, capital: float = 10000.0) -> dict[str, Any]:
    amount = float(capital)
    return {
        "account": {"initial_capital": amount, "cash": amount, "currency": "BRL"},
        "positions": {},
        "history": [],
    }


def _resolve_state_path(
    root: Path | None = None,
    db_path: Path | None = None,
) -> Path:
    resolved_db = db_path or get_state_db_path(root)
    resolved_db.parent.mkdir(parents=True, exist_ok=True)
    return resolved_db


def _connect_state_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_state_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            initial_capital REAL NOT NULL,
            cash REAL NOT NULL,
            currency TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS history (
            seq INTEGER PRIMARY KEY,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
    conn.commit()


def _db_has_portfolio(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) AS total FROM account").fetchone()
    return bool(row and int(row["total"]) > 0)


def _portfolio_from_db(conn: sqlite3.Connection) -> dict[str, Any]:
    account_row = conn.execute(
        "SELECT initial_capital, cash, currency FROM account WHERE id = 1"
    ).fetchone()
    if account_row is None:
        raise RuntimeError("portfolio state database has no account row")

    positions: dict[str, Any] = {}
    for row in conn.execute("SELECT ticker, payload FROM positions ORDER BY ticker"):
        payload = json.loads(str(row["payload"]))
        positions[str(row["ticker"])] = payload

    history: list[dict[str, Any]] = []
    for row in conn.execute("SELECT payload FROM history ORDER BY seq"):
        history.append(json.loads(str(row["payload"])))

    return {
        "account": {
            "initial_capital": float(account_row["initial_capital"]),
            "cash": float(account_row["cash"]),
            "currency": str(account_row["currency"]),
        },
        "positions": positions,
        "history": history,
    }


def _write_portfolio_db(conn: sqlite3.Connection, portfolio: dict[str, Any]) -> None:
    account = portfolio.get("account", {}) or {}
    positions = portfolio.get("positions", {}) or {}
    history = portfolio.get("history", []) or []
    now = datetime.now().isoformat(timespec="seconds")

    conn.execute(
        """
        INSERT INTO account (id, initial_capital, cash, currency)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            initial_capital = excluded.initial_capital,
            cash = excluded.cash,
            currency = excluded.currency
        """,
        (
            float(account.get("initial_capital", 0.0) or 0.0),
            float(account.get("cash", 0.0) or 0.0),
            str(account.get("currency", "BRL") or "BRL"),
        ),
    )
    conn.execute("DELETE FROM positions")
    conn.execute("DELETE FROM history")
    for ticker, payload in positions.items():
        conn.execute(
            "INSERT INTO positions (ticker, payload, updated_at) VALUES (?, ?, ?)",
            (
                str(ticker),
                json.dumps(payload, ensure_ascii=False, default=str),
                now,
            ),
        )
    for seq, payload in enumerate(history, start=1):
        created_at = str((payload or {}).get("date") or now)
        conn.execute(
            "INSERT INTO history (seq, payload, created_at) VALUES (?, ?, ?)",
            (
                int(seq),
                json.dumps(payload, ensure_ascii=False, default=str),
                created_at,
            ),
        )
    conn.execute("""
        INSERT INTO metadata (key, value)
        VALUES ('state_format', 'sqlite_v1')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """)
    conn.commit()


def load_portfolio_state(
    *,
    capital: float = 10000.0,
    root: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    resolved_db = _resolve_state_path(root=root, db_path=db_path)
    with _connect_state_db(resolved_db) as conn:
        _ensure_state_schema(conn)
        if _db_has_portfolio(conn):
            return _portfolio_from_db(conn)

        portfolio = initial_portfolio(capital=capital)
        _write_portfolio_db(conn, portfolio)
        return portfolio


def save_portfolio_state(
    portfolio: dict[str, Any],
    *,
    root: Path | None = None,
    db_path: Path | None = None,
) -> Path:
    resolved_db = _resolve_state_path(root=root, db_path=db_path)
    with _connect_state_db(resolved_db) as conn:
        _ensure_state_schema(conn)
        _write_portfolio_db(conn, portfolio)
    return resolved_db


def latest_signal_path(cfg: dict[str, Any], ticker: str) -> Path:
    return models_dir(cfg) / safe_ticker(normalize_ticker(ticker)) / "latest_signal.json"


def load_latest_signal(cfg: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    path = latest_signal_path(cfg, ticker)
    return read_json(path) if path.exists() else None


def iter_latest_signals(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = models_dir(cfg)
    if not root.exists():
        return out
    for ticker_dir in root.iterdir():
        if not ticker_dir.is_dir():
            continue
        signal_path = ticker_dir / "latest_signal.json"
        if signal_path.exists():
            out.append(read_json(signal_path))
    return out


def position_side(shares: int) -> str:
    return "SHORT" if int(shares) < 0 else "LONG"
