"""Read-only LLM spend rollup over the runs table (the round-2 accounting layer).

Applies a per-model price map (USD per 1M tokens, list price) to the prompt/completion
token counts persisted on every runs row and rolls spend up by model, ablation cell,
station, and day, with a running total. Runs retroactively over any DB with a runs
table -- no schema change needed.

Caveats printed in the report footer:
  - Prices are LIST rates sampled 2026-07-17 and keyed on model alone (no provider
    column yet -- re-key on (model, provider) when the multi-provider matrix lands).
    RE-VERIFY against provider docs before trusting for a new run: prices move weekly.
  - Provider-side prompt caching bills most input below list, so this OVERSTATES the
    true bill; killed/fatal runs persist no token counts, which UNDERSTATES it. The
    provider dashboard is authoritative; this report is for relative cell/station cost.

Usage:
    uv run python scripts/spend_report.py [--db PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from forecaster import store
from forecaster.config import settings

# USD per 1M tokens (input, output), list price, sampled 2026-07-17 (Together rates).
# Keyed on the exact `model` string persisted on runs rows.
PRICES: dict[str, tuple[float, float]] = {
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "google/gemma-4-31B-it": (0.39, 0.97),
    "moonshotai/Kimi-K2.7-Code": (0.95, 4.00),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
}


def _cell(run: dict) -> str:
    """Ablation-cell key derived from persisted provenance: model short name +
    worksheet mode + temperature + prior-TAF access (the run_id suffix)."""
    model_short = (run.get("model") or "?").rsplit("/", 1)[-1]
    mode = run.get("worksheet_mode") or "?"
    temp = run.get("temperature")
    taf = (run.get("run_id") or "").rsplit("_", 1)[-1]  # 'taf' | 'notaf'
    return f"{model_short} ws={mode} t={temp} {taf}"


def _cost(run: dict) -> float | None:
    """List-price cost of one run, or None when the model is unpriced or the row
    carries no token counts (fatal/stub rows)."""
    price = PRICES.get(run.get("model") or "")
    if price is None:
        return None
    pt, ct = run.get("prompt_tokens"), run.get("completion_tokens")
    if pt is None and ct is None:
        return None
    return (pt or 0) / 1e6 * price[0] + (ct or 0) / 1e6 * price[1]


def _table(title: str, rows: list[tuple], header: tuple) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    print(f"\n{title}")
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=settings.db_path)
    args = ap.parse_args()

    con = store.connect(args.db, read_only=True)
    try:
        runs = store.all_runs(con, producer_kind="artificial")
    finally:
        con.close()

    priced, unpriced_models, uncosted = [], set(), 0
    for r in runs:
        c = _cost(r)
        if c is None:
            if (r.get("model") or "") not in PRICES:
                unpriced_models.add(r.get("model") or "?")
            else:
                uncosted += 1
            continue
        priced.append((r, c))

    def rollup(key_fn):
        agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "usd": 0.0, "clean": 0})
        for r, c in priced:
            a = agg[key_fn(r)]
            a["n"] += 1
            a["usd"] += c
            a["clean"] += 1 if r.get("taf_clean") else 0
        return agg

    total = sum(c for _, c in priced)
    print(f"spend_report: {len(runs)} artificial runs, {len(priced)} priced, "
          f"{uncosted} without token counts (fatal/stub), total ${total:.2f} list")

    for title, key_fn in [
        ("BY MODEL", lambda r: r.get("model") or "?"),
        ("BY CELL", _cell),
        ("BY STATION", lambda r: r.get("station") or "?"),
        ("BY DAY (UTC)", lambda r: str(r.get("created_at"))[:10]),
    ]:
        agg = rollup(key_fn)
        rows = []
        for k in sorted(agg, key=lambda k: -agg[k]["usd"]):
            a = agg[k]
            per_run = a["usd"] / a["n"]
            per_clean = f"{a['usd'] / a['clean']:.3f}" if a["clean"] else "--"
            rows.append((k, a["n"], a["clean"], f"{a['usd']:.2f}",
                         f"{per_run:.3f}", per_clean))
        _table(title, rows, ("key", "runs", "clean", "usd", "usd/run", "usd/clean"))

    running = 0.0
    by_day = rollup(lambda r: str(r.get("created_at"))[:10])
    rows = []
    for day in sorted(by_day):
        running += by_day[day]["usd"]
        rows.append((day, f"{by_day[day]['usd']:.2f}", f"{running:.2f}"))
    _table("RUNNING TOTAL", rows, ("day", "usd", "cumulative"))

    if unpriced_models:
        print(f"\nWARNING: no price for model(s) {sorted(unpriced_models)} -- excluded.")
    print("\nList price, keyed on model alone (no provider column yet); caching makes the"
          "\ntrue bill lower and unrecorded fatal-run tokens make it higher -- the provider"
          "\ndashboard is authoritative. Re-verify PRICES before a new collection round.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
