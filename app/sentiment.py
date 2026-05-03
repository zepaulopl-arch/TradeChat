from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import hashlib
import numpy as np
import pandas as pd

from .config import cache_dir
from .utils import normalize_ticker, safe_ticker


def _sentiment_cache_path(cfg: dict[str, Any], ticker: str) -> Path:
    path = cache_dir(cfg) / "sentiment"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{safe_ticker(ticker)}_sentiment_daily.csv"


def _parse_entry_date(entry: Any) -> str:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc).date().isoformat()
        except Exception:
            pass
    return datetime.now().date().isoformat()


def _score_titles(ticker: str, cfg: dict[str, Any]) -> pd.DataFrame:
    sent_cfg = cfg.get("features", {}).get("sentiment", {})
    max_entries = int(sent_cfg.get("max_news_entries", 20))
    try:
        import feedparser
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        from deep_translator import GoogleTranslator
        nltk.download("vader_lexicon", quiet=True)
        sid = SentimentIntensityAnalyzer()
        translator = GoogleTranslator(source="pt", target="en")
        base = ticker.split(".")[0]
        urls = [
            f"https://news.google.com/rss/search?q={base}+Bovespa&hl=pt-BR&gl=BR&ceid=BR:pt-419",
            f"https://www.bing.com/news/search?q={base}+Bovespa&format=rss",
        ]
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for url in urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_entries]:
                title = getattr(entry, "title", "")
                key = hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()
                if not title or key in seen:
                    continue
                seen.add(key)
                try:
                    translated = translator.translate(title)
                    score = float(sid.polarity_scores(translated)["compound"]) * float(sent_cfg.get("vader_weight", 1.0))
                except Exception:
                    continue
                rows.append({"date": _parse_entry_date(entry), "score": score, "title_hash": key})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["date", "score", "title_hash"])


def update_sentiment_cache(ticker: str, cfg: dict[str, Any]) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    sent_cfg = cfg.get("features", {}).get("sentiment", {})
    path = _sentiment_cache_path(cfg, ticker)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date", "score", "count"])
    raw = _score_titles(ticker, cfg)
    if raw.empty:
        return {"enabled": True, "source": "rss_vader_cache", "cache_path": str(path), "new_items": 0, "cache_rows": int(len(old))}
    grouped = raw.groupby("date").agg(score=("score", "mean"), count=("score", "size")).reset_index()
    frames = [df for df in (old, grouped) if df is not None and not df.empty and not df.dropna(how="all").empty]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "score", "count"])
    merged["date"] = pd.to_datetime(merged["date"]).dt.date.astype(str)
    merged = merged.groupby("date", as_index=False).agg(score=("score", "mean"), count=("count", "sum"))
    cache_days = int(sent_cfg.get("cache_days", 365))
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=cache_days)
    merged = merged[pd.to_datetime(merged["date"]) >= cutoff]
    merged.to_csv(path, index=False)
    return {"enabled": True, "source": "rss_vader_cache", "cache_path": str(path), "new_items": int(len(raw)), "cache_rows": int(len(merged))}


def load_sentiment_daily_series(ticker: str, cfg: dict[str, Any], index: pd.Index) -> tuple[pd.DataFrame, dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    sent_cfg = cfg.get("features", {}).get("sentiment", {})
    update_meta = update_sentiment_cache(ticker, cfg)
    path = _sentiment_cache_path(cfg, ticker)
    dates = pd.to_datetime(index).tz_localize(None) if getattr(index, "tz", None) is not None else pd.to_datetime(index)
    base = pd.DataFrame(index=dates)
    base.index.name = "date"
    if path.exists():
        cached = pd.read_csv(path)
    else:
        cached = pd.DataFrame(columns=["date", "score", "count"])
    if not cached.empty:
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date").sort_index()
        daily = cached.reindex(base.index)
    else:
        daily = pd.DataFrame(index=base.index, columns=["score", "count"])
    fallback_zero = bool(sent_cfg.get("fallback_to_zero", True))
    score = daily["score"].astype(float) if "score" in daily else pd.Series(index=base.index, dtype=float)
    count = daily["count"].astype(float) if "count" in daily else pd.Series(index=base.index, dtype=float)
    if fallback_zero:
        score = score.fillna(0.0)
        count = count.fillna(0.0)
    available = (count.fillna(0.0) >= int(sent_cfg.get("min_items_for_feature", 1))).astype(float)
    out = pd.DataFrame(index=index)
    out["sent_score_daily"] = score.to_numpy()
    out["sent_count_daily"] = count.to_numpy()
    out["sent_available"] = available.to_numpy()
    window_groups = sent_cfg.get("window_groups", {}) or {}
    mean_windows = [int(w) for w in window_groups.get("mean", sent_cfg.get("windows", [1, 3, 7])) if int(w) >= 1]
    count_windows = [int(w) for w in window_groups.get("count", sent_cfg.get("windows", [1, 3, 7])) if int(w) >= 1]
    delta_windows = [int(w) for w in window_groups.get("delta", [3]) if int(w) >= 1]
    std_windows = [int(w) for w in window_groups.get("std", [7]) if int(w) >= 2]
    score_s = pd.Series(out["sent_score_daily"].to_numpy(), index=out.index)
    count_s = pd.Series(out["sent_count_daily"].to_numpy(), index=out.index)
    for w in mean_windows:
        out[f"sent_mean_{w}d"] = score_s.rolling(w, min_periods=1).mean()
    for w in count_windows:
        out[f"sent_count_{w}d"] = count_s.rolling(w, min_periods=1).sum()
    for w in delta_windows:
        out[f"sent_delta_{w}d"] = score_s - score_s.shift(w).fillna(0.0)
    for w in std_windows:
        out[f"sent_std_{w}d"] = score_s.rolling(w, min_periods=2).std().fillna(0.0)
    meta = dict(update_meta)
    meta.update({"mode": sent_cfg.get("mode", "temporal_feature"), "columns": list(out.columns)})
    return out, meta


def get_sentiment(ticker: str, cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Current operational sentiment for predict/report."""
    ticker = normalize_ticker(ticker)
    meta = update_sentiment_cache(ticker, cfg)
    path = _sentiment_cache_path(cfg, ticker)
    if path.exists():
        cached = pd.read_csv(path)
        if not cached.empty:
            value = float(cached.sort_values("date")["score"].iloc[-1])
            meta["items"] = int(cached.sort_values("date")["count"].iloc[-1])
            return value, meta
    meta["items"] = 0
    return 0.0, meta
