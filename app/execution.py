from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ROOT, load_config
from .utils import normalize_ticker, write_json, read_json


def get_portfolio_path(cfg: dict[str, Any]) -> Path:
    path = ROOT / "data" / "portfolio.json"
    return path


def load_portfolio(cfg: dict[str, Any]) -> dict[str, Any]:
    path = get_portfolio_path(cfg)
    if not path.exists():
        initial = {
            "account": {
                "initial_capital": float(cfg.get("trading", {}).get("capital", 10000.0)),
                "cash": float(cfg.get("trading", {}).get("capital", 10000.0)),
                "currency": "BRL"
            },
            "positions": {},
            "history": []
        }
        write_json(path, initial)
        return initial
    return read_json(path)


def save_portfolio(cfg: dict[str, Any], portfolio: dict[str, Any]) -> None:
    path = get_portfolio_path(cfg)
    write_json(path, portfolio)


def virtual_buy(cfg: dict[str, Any], ticker: str, signal: dict[str, Any]) -> dict[str, Any]:
    """Execute a virtual buy order based on a signal."""
    portfolio = load_portfolio(cfg)
    ticker = normalize_ticker(ticker)
    
    if ticker in portfolio["positions"]:
        raise ValueError(f"Already have an active position in {ticker}")
        
    policy = signal.get("policy", {})
    price = float(signal.get("latest_price", 0.0))
    shares = int(policy.get("position_size", 0))
    
    cost = price * shares
    if cost > portfolio["account"]["cash"]:
        # Allow buying with what we have, or raise error? 
        # Let's be strict to simulate real capital constraints.
        shares = int(portfolio["account"]["cash"] / price)
        cost = price * shares
        if shares <= 0:
            raise ValueError(f"Insufficient virtual cash (Need R$ {price:.2f}, have R$ {portfolio['account']['cash']:.2f})")

    new_pos = {
        "ticker": ticker,
        "entry_price": price,
        "shares": shares,
        "target_partial": float(policy.get("target_partial", 0.0)),
        "target_final": float(policy.get("target_price", 0.0)),
        "stop_loss": float(policy.get("stop_loss_price", 0.0)),
        "entry_date": signal.get("latest_date", datetime.now().strftime("%Y-%m-%d")),
        "partial_executed": False,
        "status": "active"
    }
    
    portfolio["positions"][ticker] = new_pos
    portfolio["account"]["cash"] -= cost
    
    log_entry = {
        "type": "BUY",
        "ticker": ticker,
        "price": price,
        "shares": shares,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    portfolio["history"].append(log_entry)
    
    save_portfolio(cfg, portfolio)
    return new_pos


def update_portfolio_prices(cfg: dict[str, Any], current_prices: dict[str, float]) -> list[str]:
    """Check if targets or stops were hit based on current prices."""
    portfolio = load_portfolio(cfg)
    events = []
    to_remove = []
    
    for ticker, pos in portfolio["positions"].items():
        if ticker not in current_prices:
            continue
            
        curr_price = current_prices[ticker]
        shares = pos["shares"]
        
        # Check Stop Loss
        if (pos["stop_loss"] > 0 and curr_price <= pos["stop_loss"]):
            # SELL ALL (Stop Hit)
            gain = (curr_price - pos["entry_price"]) * shares
            portfolio["account"]["cash"] += (curr_price * shares)
            events.append(f"STOP HIT: {ticker} sold at R$ {curr_price:.2f} (Result: R$ {gain:+.2f})")
            to_remove.append(ticker)
            
            portfolio["history"].append({
                "type": "STOP_EXIT", "ticker": ticker, "price": curr_price, "gain": gain, 
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            continue

        # Check Partial Target
        if not pos["partial_executed"] and pos["target_partial"] > 0 and curr_price >= pos["target_partial"]:
            # SELL HALF (Scale out)
            half_shares = int(shares / 2)
            if half_shares > 0:
                gain = (curr_price - pos["entry_price"]) * half_shares
                portfolio["account"]["cash"] += (curr_price * half_shares)
                pos["shares"] -= half_shares
                pos["partial_executed"] = True
                # MOVE STOP TO BREAKEVEN (Trailing Stop simple logic)
                old_stop = pos["stop_loss"]
                pos["stop_loss"] = pos["entry_price"]
                events.append(f"PARTIAL HIT: {ticker} sold 50% at R$ {curr_price:.2f}. Stop moved to R$ {pos['entry_price']:.2f}")
                
                portfolio["history"].append({
                    "type": "PARTIAL_EXIT", "ticker": ticker, "price": curr_price, "gain": gain, 
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        # Check Final Target
        if pos["target_final"] > 0 and curr_price >= pos["target_final"]:
            # SELL REMAINING
            remaining = pos["shares"]
            gain = (curr_price - pos["entry_price"]) * remaining
            portfolio["account"]["cash"] += (curr_price * remaining)
            events.append(f"TARGET HIT: {ticker} closed at R$ {curr_price:.2f} (Result: R$ {gain:+.2f})")
            to_remove.append(ticker)
            
            portfolio["history"].append({
                "type": "FINAL_EXIT", "ticker": ticker, "price": curr_price, "gain": gain, 
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    for t in to_remove:
        del portfolio["positions"][t]
        
    if events:
        save_portfolio(cfg, portfolio)
    return events
