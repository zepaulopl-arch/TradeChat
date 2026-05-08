import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.batch_service import diagnose_one_asset, safe_worker_count
from app.config import artifact_dir, load_config, load_data_registry
from app.data import resolve_asset
from app.presentation import C, banner, divider, paint, render_facts, render_table, render_wrapped, screen_width
from app.utils import normalize_ticker, parse_tickers


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _registered_assets(cfg: dict[str, Any]) -> list[str]:
    registry = load_data_registry(cfg)
    assets = registry.get("assets", {}) or {}
    out: list[str] = []
    for ticker, meta in assets.items():
        if not isinstance(meta, dict):
            continue
        if str(meta.get("registry_status", "active")).lower() == "active" and bool(meta.get("use_in_reference_sample", True)):
            out.append(normalize_ticker(ticker))
    return sorted(dict.fromkeys(out))


def _asset_list(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.assets:
        raw_assets = str(args.assets).strip()
        tickers = _registered_assets(cfg) if raw_assets.upper() == "ALL" else parse_tickers([raw_assets])
    else:
        tickers = _registered_assets(cfg)
    tickers = [
        ticker
        for ticker in tickers
        if str((resolve_asset(cfg, ticker).get("profile", {}) or {}).get("registry_status", "active")).lower() == "active"
    ]
    if args.limit:
        tickers = tickers[: int(args.limit)]
    return tickers


def _print_status(row: dict[str, Any]) -> None:
    status = str(row.get("status", "error"))
    if status == "ok":
        print(f"| {paint('OK', C.GREEN)}")
    elif status == "skipped":
        print(f"| {paint('SKIP', C.YELLOW)} at {row.get('failed_stage')}")
    else:
        print(f"| {paint('FAIL', C.RED)} at {row.get('failed_stage')}")


def _write_outputs(cfg: dict[str, Any], rows: list[dict[str, Any]], run_id: str) -> dict[str, Path]:
    out_dir = artifact_dir(cfg) / "diagnostics" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "assets_diagnostic.csv"
    json_path = out_dir / "assets_diagnostic.json"
    txt_path = out_dir / "assets_diagnostic_summary.txt"

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames and key != "traceback":
                fieldnames.append(key)
    if "traceback" not in fieldnames:
        fieldnames.append("traceback")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    ok = [r for r in rows if r.get("status") == "ok"]
    skipped = [r for r in rows if r.get("status") == "skipped"]
    err = [r for r in rows if r.get("status") not in {"ok", "skipped"}]
    width = 88
    lines: list[str] = []
    lines.extend(banner("ASSET DIAGNOSTIC SUMMARY", run_id, width=width, use_color=False))
    lines.extend(
        render_facts(
            [("Total Assets", len(rows)), ("OK", len(ok)), ("Skipped", len(skipped)), ("Errors", len(err))],
            width=width,
            max_columns=4,
            use_color=False,
        )
    )
    lines.append(divider(width, use_color=False))
    issues = skipped + err
    if issues:
        lines.append("ISSUE LOG")
        for row in issues[:30]:
            lines.extend(
                render_wrapped(
                    str(row.get("ticker") or row.get("input_ticker") or "n/a"),
                    f"stage={row.get('failed_stage', 'n/a')} | {row.get('error', 'n/a')}",
                    width=width,
                    use_color=False,
                )
            )
        lines.append(divider(width, use_color=False))

    lines.append("OUTPUT FILES")
    lines.extend(render_wrapped("CSV", csv_path, width=width, use_color=False))
    lines.extend(render_wrapped("JSON", json_path, width=width, use_color=False))
    lines.extend(render_wrapped("TXT", txt_path, width=width, use_color=False))
    lines.append(divider(width, use_color=False))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "txt": txt_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-horizon diagnostics.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--assets", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 uses config default")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tickers = _asset_list(cfg, args)
    run_id = f"diag_{_now_id()}"
    width = screen_width()
    requested_workers = args.workers if args.workers > 0 else int(cfg.get("batch", {}).get("diagnose_workers", 1) or 1)
    workers = safe_worker_count(len(tickers), requested=requested_workers, default=1)

    print()
    for line in banner("MULTI-HORIZON DIAGNOSTIC", f"assets={len(tickers)}", f"run_id={run_id}", width=width):
        print(line)
    for line in render_facts(
        [("Assets", len(tickers)), ("Workers", workers), ("Autotune", bool(args.autotune)), ("No Data", bool(args.no_data))],
        width=width,
        max_columns=4,
    ):
        print(line)

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for idx, ticker in enumerate(tickers, start=1):
            print(f"[{paint(f'{idx:>3}/{len(tickers):<3}', C.DIM)}] {paint(f'{ticker:<12}', C.BOLD)} data -> train -> predict", end=" ", flush=True)
            row = diagnose_one_asset(cfg, ticker, no_data=bool(args.no_data), autotune=bool(args.autotune), inner_threads=None)
            rows.append(row)
            _print_status(row)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, ticker in enumerate(tickers, start=1):
                futures[
                    executor.submit(
                        diagnose_one_asset,
                        cfg,
                        ticker,
                        no_data=bool(args.no_data),
                        autotune=bool(args.autotune),
                        inner_threads=1,
                    )
                ] = (idx, ticker)
            for future in as_completed(futures):
                idx, ticker = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "input_ticker": ticker,
                        "ticker": ticker,
                        "status": "error",
                        "failed_stage": "worker",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": "",
                    }
                rows.append(row)
                print(f"[{paint(f'{idx:>3}/{len(tickers):<3}', C.DIM)}] {paint(f'{ticker:<12}', C.BOLD)} data -> train -> predict", end=" ", flush=True)
                _print_status(row)

    rows.sort(key=lambda row: str(row.get("ticker") or row.get("input_ticker") or ""))
    paths = _write_outputs(cfg, rows, run_id)
    print(divider(width))
    print(f"Summary: {paint(paths['txt'], C.BLUE)}")
    print(divider(width))
    return 0 if all(r.get("status") in {"ok", "skipped"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
