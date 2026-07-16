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

from forecaster import stations

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


# The benchmark matrix, 6 cells per cycle event (7 once Inkling is added to the control):
#   3-model CONTROL (worksheet required, temp 0, prior TAF) + MiniMax single-factor
#   ablations off that control (temperature, no worksheet, no prior TAF).
MATRIX: list[Cell] = [
    Cell("ctl-kimi",    KIMI,    0.0, "required", True),
    Cell("ctl-minimax", MINIMAX, 0.0, "required", True),
    Cell("ctl-gemma",   GEMMA,   0.0, "required", True),
    # Cell("ctl-inkling", INKLING, 0.0, "required", True),   # pending a valid endpoint
    Cell("mm-temp02",   MINIMAX, 0.2, "required", True),    # ablation: temperature
    Cell("mm-nows",     MINIMAX, 0.0, "off",      True),    # ablation: no worksheet
    Cell("mm-notaf",    MINIMAX, 0.0, "required", False),   # ablation: no prior-TAF access
]

# Per-cell wall-clock cap so a hung model/network call can't stall the hour's batch.
CELL_TIMEOUT_S = 1800

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

    # Build every (station, cell) job for this hour, then run up to max_parallel at once.
    jobs: list[tuple[str, list[str]]] = []
    for st in stns:
        for cell in MATRIX:
            cmd = [sys.executable, str(_COLLECT), "--station", st.icao, "--model", cell.model,
                   "--temperature", str(cell.temperature), "--mode", cell.mode,
                   "--taf-access" if cell.taf_access else "--no-taf-access",
                   "--issue-time", f"{issue:%Y-%m-%dT%H%M}Z"]
            if args.db:
                cmd += ["--db", args.db]
            jobs.append((f"{st.icao} {cell.label} ({cell.model.split('/')[-1]})", cmd))

    if args.dry_run:
        for label, _ in jobs:
            print(f"  DRY-RUN would fire: {label}")
        return 0

    parallel = max(1, args.max_parallel)
    print(f"dispatching {len(jobs)} run(s), up to {parallel} in parallel\n")
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
