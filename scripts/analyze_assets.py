import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config, load_data_registry
from app.data import data_status


def main() -> int:
    cfg = load_config()
    registry = load_data_registry(cfg)
    min_rows = int(cfg.get("data", {}).get("min_rows", 150))
    assets = registry.get("assets", {}) or {}
    active = [ticker for ticker, meta in assets.items() if isinstance(meta, dict) and meta.get("registry_status") == "active"]

    rows = []
    for ticker in sorted(active):
        status = data_status(cfg, ticker)
        rows.append((ticker, int(status.get("rows", 0) or 0)))

    print("---RESULTS---")
    for ticker, count in rows:
        print(f"{ticker}: {count}")

    print("---LOW DATA---")
    for ticker, count in rows:
        if count < min_rows:
            print(f"{ticker}: {count}")
    print("---END---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
