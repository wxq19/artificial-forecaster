"""Collector scheduler: fire collect.py for whichever stations are due this UTC hour.

Each roster station issues on its OWN 8-hourly cycle (stations.py `cycle`), so a single
crontab of per-station times would be brittle to roster edits. Instead this runs once an
hour from ONE cron entry, consults the roster, and dispatches a collection run for every
station whose cycle includes the current UTC hour -- across the configured model x
temperature x worksheet-mode MATRIX.

Issue time is pinned to the TOP of the cycle hour (not the wall-clock fire time), so cron
can fire a couple minutes late (to let the :53 METAR settle) while the obs cutoff stays on
the cycle boundary -- leakage-safe and aligned with the human forecaster's issue time.

Each cell runs as an isolated subprocess (a crash/timeout in one station never aborts the
batch); collect.py serializes its own writes under the single-writer lock.

  uv run python scripts/schedule.py                 # dispatch for the current UTC hour
  uv run python scripts/schedule.py --dry-run       # show what WOULD fire, launch nothing
  uv run python scripts/schedule.py --hour 10       # test a specific cycle hour
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from forecaster import modeldata, stations
from forecaster.config import settings

KIMI = "moonshotai/Kimi-K2.7-Code"
MINIMAX = "MiniMaxAI/MiniMax-M3"
GEMMA = "google/gemma-4-31B-it"
# INKLING = "thinkingmachines/inkling"   # not on Together (404 model_not_available); add later


@dataclass(frozen=True)
class Cell:
    """One matrix cell = one collection run per due station per cycle."""

    label: str
    model: str
    temperature: float
    mode: str           # required (worksheet mandatory) | advisory | off
    taf_access: bool    # give the model the leakage-safe previous TAF


# The benchmark matrix, 5 cells per cycle event (6 once Inkling is added to the control):
#   2-model CONTROL (worksheet required, temp 0, prior TAF) + MiniMax single-factor
#   ablations off that control (temperature, no worksheet, no prior TAF).
MATRIX: list[Cell] = [
    # Cell("ctl-kimi",    KIMI,    0.0, "required", True),    # dropped: worst $/clean TAF (~16x MiniMax), lowest clean rate; revisit with Kimi K3
    Cell("ctl-minimax", MINIMAX, 0.0, "required", True),
    Cell("ctl-gemma",   GEMMA,   0.0, "required", True),
    # Cell("ctl-inkling", INKLING, 0.0, "required", True),   # pending a valid endpoint
    Cell("mm-temp02",   MINIMAX, 0.2, "required", True),    # ablation: temperature
    Cell("mm-nows",     MINIMAX, 0.0, "off",      True),    # ablation: no worksheet
    Cell("mm-notaf",    MINIMAX, 0.0, "required", False),   # ablation: no prior-TAF access
]

# Per-cell wall-clock cap so a hung model/network call can't stall the hour's batch.
# Round-1 evidence (25 events): median cell finished ~9 min after the hour, p90 18, worst
# batch tail 25 min -- but that finish-lag CONFOUNDS queue time with runtime, so the tail
# is not proven runtime. 1500 sits safely above any plausible single-cell runtime; drop to
# 1200 once runs.duration_s (T10) shows max well under 20 min.
CELL_TIMEOUT_S = 1500

# How many collection subprocesses to run concurrently. Each is one python process
# rendering matplotlib charts + calling the API, so this is bounded by the HOST's CPU/RAM,
# NOT the endpoint (Together handles concurrent requests). Conservative default; tune to
# the Pi via --max-parallel. Writes to the one benchmark DB are serialized by the flock,
# so concurrent runs are safe -- they only queue briefly on the final persist.
MAX_PARALLEL = 2

_COLLECT = Path(__file__).resolve().parent / "collect.py"


def due(hour: int) -> list[stations.Station]:
    """Roster stations whose 8-hourly cycle includes this UTC hour."""
    return [s for s in stations.STATIONS if hour in s.cycle]


def _run_one(label: str, cmd: list[str]) -> tuple[str, int | None, str]:
    """Run one collection subprocess, capturing its output so parallel runs don't
    interleave. Returns (label, returncode | None-on-timeout, combined output)."""
    try:
        r = subprocess.run(cmd, timeout=CELL_TIMEOUT_S, capture_output=True, text=True)
        return label, r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "")
        return label, None, (out.decode() if isinstance(out, bytes) else out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dispatch collection runs for due stations.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    ap.add_argument("--hour", type=int, help="override the UTC cycle hour (testing)")
    ap.add_argument("--max-parallel", type=int, default=MAX_PARALLEL,
                    help=f"concurrent collection subprocesses (default {MAX_PARALLEL})")
    ap.add_argument("--db", default=None, help="benchmark DB path (default: settings.db_path)")
    ap.add_argument("--model-data", action=argparse.BooleanOptionalAction, default=None,
                    help="force the GRIBStream model-data tier on/off for this dispatch "
                         "(default: inherit collect.py's MODEL_DATA_ENABLED; BILLS credits when on)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    hour = args.hour if args.hour is not None else now.hour
    issue = now.replace(hour=hour)
    stns = due(hour)

    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}] cycle hour {hour:02d}Z -> "
          f"{len(stns)} due station(s): {', '.join(s.icao for s in stns) or '(none)'}; "
          f"{len(MATRIX)} matrix cell(s)")
    if not stns:
        return 0

    # Whether the model-data tier is on for this dispatch: explicit --model-data wins, else
    # the collector's settings default. Model-data prefetch is done ONCE, BATCHED, in-process
    # (below) -- not per ingest subprocess -- so the union of due stations' neighborhoods is
    # one set of <=500-coord requests instead of N.
    model_data_on = args.model_data if args.model_data is not None else settings.model_data_enabled

    def _md_flag() -> list[str]:
        # None -> inherit collect's settings default; True/False -> force it for this dispatch.
        return [] if args.model_data is None else (
            ["--model-data"] if args.model_data else ["--no-model-data"])

    def _ingest_cmd(st: stations.Station) -> list[str]:
        # Obs banking only -- prefetch is batched in-process, so tell the subprocess NOT to.
        cmd = [sys.executable, str(_COLLECT), "--station", st.icao, "--ingest-only",
               "--no-model-data", "--issue-time", f"{issue:%Y-%m-%dT%H%M}Z"]
        return cmd + (["--db", args.db] if args.db else [])

    def _cell_cmd(st: stations.Station, cell: Cell) -> list[str]:
        cmd = [sys.executable, str(_COLLECT), "--station", st.icao, "--model", cell.model,
               "--temperature", str(cell.temperature), "--mode", cell.mode,
               "--taf-access" if cell.taf_access else "--no-taf-access",
               "--no-ingest",  # obs are pre-banked once per station by the ingest pass below
               "--issue-time", f"{issue:%Y-%m-%dT%H%M}Z"]
        return cmd + _md_flag() + (["--db", args.db] if args.db else [])

    if args.dry_run:
        for st in stns:
            print(f"  DRY-RUN would ingest: {st.icao} (once)")
        if model_data_on:
            print(f"  DRY-RUN would prefetch model-data (batched): {', '.join(s.icao for s in stns)}")
        for st in stns:
            for cell in MATRIX:
                print(f"  DRY-RUN would fire:   {st.icao} {cell.label} ({cell.model.split('/')[-1]})")
        return 0

    # Phase 1 -- ingest each due station's obs ONCE (home + proxy + neighbors) into the
    # benchmark DB. Sequential: the fetch is quick, it de-dups the AWC calls the cells used
    # to each repeat, gives every cell of a station the SAME frozen obs snapshot, and keeps
    # obs fetching off the concurrent cell burst. A station whose ingest fails is skipped --
    # its cells would have no obs (collect.py --ingest-only already retries the transient AWC
    # hiccup, so a hard failure here is real).
    ready: list[stations.Station] = []
    for st in stns:
        label, rc, output = _run_one(f"{st.icao} ingest", _ingest_cmd(st))
        if rc == 0:
            ready.append(st)
            print(f"----- {label}: OK -----")
        else:
            status = f"FAILED rc={rc}" if rc is not None else "TIMEOUT"
            tail = "\n".join(output.strip().splitlines()[-6:]) if output.strip() else "(no output)"
            print(f"----- {label}: {status} -- SKIPPING {st.icao} cells -----\n{tail}\n")

    # Phase 1b -- ONE batched model-data prefetch for all ingested stations (they share this
    # cycle's issue time). Points are free <=500, so the union costs ~1 request/model instead
    # of N. A failure here is non-fatal: the cells with --model-data just copy an empty archive
    # and the get_model_* tools return "not pre-fetched" feedback.
    if model_data_on and ready:
        try:
            md = modeldata.prefetch_many([st.icao for st in ready], as_of=issue, db_path=args.db)
            print(f"----- model-data prefetch: {md['rows_inserted']} rows across "
                  f"{md['coords']} coords in {md['requests']} request(s), "
                  f"{md['credits_charged']} credits -----"
                  + (f"\n  notes: {md['notes']}" if md["notes"] else ""))
        except Exception as e:  # noqa: BLE001 -- prefetch failure must not abort the cells
            print(f"----- model-data prefetch FAILED ({type(e).__name__}: {e}) -- "
                  "cells will find an empty archive -----")

    # Phase 2 -- run the matrix cells (--no-ingest) for the successfully-ingested stations.
    jobs = [(f"{st.icao} {cell.label} ({cell.model.split('/')[-1]})", _cell_cmd(st, cell))
            for st in ready for cell in MATRIX]
    parallel = max(1, args.max_parallel)
    print(f"\ndispatching {len(jobs)} run(s), up to {parallel} in parallel\n")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(_run_one, label, cmd) for label, cmd in jobs]
        for fut in as_completed(futures):
            label, rc, output = fut.result()
            status = "OK" if rc == 0 else (f"FAILED rc={rc}" if rc is not None else "TIMEOUT")
            ok, fail = (ok + 1, fail) if rc == 0 else (ok, fail + 1)
            tail = "\n".join(output.strip().splitlines()[-8:]) if output.strip() else "(no output)"
            print(f"----- {label}: {status} -----\n{tail}\n")

    print(f"done: {ok} ok, {fail} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
