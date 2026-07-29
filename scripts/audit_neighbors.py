"""Regenerate src/forecaster/neighbor_rates.py -- the measured reporting rate per candidate.

`build_neighbors.py` ranks candidates by DISTANCE from AWC's `stationinfo` catalog, which is
a MEMBERSHIP list: a site is in it if AWC knows it as a METAR site, not if it files METARs.
So `get_nearby_obs` offered dead entries and `get_terrain` drew them blue ("obs-available").
Measured 2026-07-28: 52 of 307 fetchable neighbours filed nothing in 72 hours, across 34 of
the 71 archived stations -- KWRI's KNEL (22.7 km) and all five of RKJK's.

This script measures a RATE over a window rather than probing once, because a single miss is
not proof a site never reports, and a one-shot misread would permanently drop a live field.
The output is a frozen table so `build_neighbors.py --check` stays deterministic: the roster
is then reproducible from (catalog geography + this frozen measurement), not from whatever
AWC happened to serve at build time.

The candidate set comes from `build_neighbors._candidates`, the SAME helper the roster build
uses, so the two cannot drift apart on which sites were considered.

TWO AWC GOTCHAS, both load-bearing:
  - The `metar` product caps a response at ~400 rows REGARDLESS of how many ids were asked
    for. A 40-id batch silently truncates and the ids that fall off look silent. A first scan
    read RJTF as dead this way. `_counts` splits any response that reaches the cap.
  - `collections.Counter.__add__` DROPS zero counts, so summing batches loses exactly the
    silent stations this script exists to find. Plain dicts throughout, on purpose.

RESUMABLE AND CHUNKABLE. A full pass is ~800 throttled AWC requests, about 50 minutes, and
the first attempt lost everything to a timeout. Progress is now cached to
data/neighbor_audit_cache.json after EVERY batch, so a stop costs one batch. Cached entries
older than _STALE_H are re-measured rather than trusted, and a window change discards the
cache outright -- a count only means something against the window it was counted over.

  uv run python scripts/audit_neighbors.py                     # full pass (resumes)
  uv run python scripts/audit_neighbors.py --chunk 10          # next 10 stations, then stop
  uv run python scripts/audit_neighbors.py --stations KWRI,KMIB
  uv run python scripts/audit_neighbors.py --refresh           # ignore the cache
  uv run python scripts/audit_neighbors.py --check             # diff vs committed

Only a pass covering ALL of poll_icaos() writes neighbor_rates.py; a chunk reports how far it
got and stops, because a partial table would silently mark unmeasured sites as silent.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import awc
from forecaster.stations import poll_icaos

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_neighbors import _candidates, _home_info  # noqa: E402  (same candidate set)

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "src" / "forecaster" / "neighbor_rates.py"
# Scratch, in gitignored data/. Holds partial progress so a killed run resumes instead of
# restarting: the first full pass took ~50 min of AWC calls and a timeout lost ALL of it.
_CACHE = _ROOT / "data" / "neighbor_audit_cache.json"
_HOURS = 72        # window; long enough that a routine outage does not read as "dead"
_BATCH = 4         # ids per request; a busy 5-id batch already brushes the ~400-row cap
_CAP = 395         # treat a response this large as truncated and split it
_STALE_H = 24      # a cached measurement older than this is re-measured, not trusted


def _counts(ids: list[str]) -> dict[str, int]:
    """Reports per ICAO over the window. Splits any response that hit AWC's row cap."""
    rows = awc._get("metar", {"ids": ",".join(ids), "format": "json", "hours": _HOURS})
    if len(rows) >= _CAP and len(ids) > 1:
        mid = len(ids) // 2
        return {**_counts(ids[:mid]), **_counts(ids[mid:])}
    out = {i: 0 for i in ids}
    for r in rows:
        icao = (r.get("icaoId") or "").upper()
        if icao in out:
            out[icao] += 1
    return out


def _load_cache() -> dict:
    """Partial progress from earlier runs. A window change invalidates everything, because
    a count is only comparable against the window it was counted over."""
    if not _CACHE.exists():
        return {"window_hours": _HOURS, "candidates": {}, "reports": {}}
    c = json.loads(_CACHE.read_text())
    if c.get("window_hours") != _HOURS:
        print(f"cache was measured over {c.get('window_hours')}h, not {_HOURS}h -- discarding")
        return {"window_hours": _HOURS, "candidates": {}, "reports": {}}
    return c


def _save_cache(cache: dict) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(cache, sort_keys=True))


def _fresh(entry: dict) -> bool:
    at = datetime.fromisoformat(entry["at"])
    return datetime.now(timezone.utc) - at < timedelta(hours=_STALE_H)


def candidates_for(stations: list[str], cache: dict) -> set[str]:
    """Every catalog site near any of `stations`, reusing cached station lookups."""
    out: set[str] = set()
    for home in stations:
        hit = cache["candidates"].get(home)
        if hit is None or not _fresh(hit):
            hlat, hlon, _elev = _home_info(home)
            near = _candidates(hlat, hlon)
            near.pop(home, None)
            hit = {"ids": sorted(near), "at": datetime.now(timezone.utc).isoformat()}
            cache["candidates"][home] = hit
            _save_cache(cache)
            print(f"{home}: {len(hit['ids'])} candidates", flush=True)
        out |= set(hit["ids"])
    return out


def build(stations: list[str], *, refresh: bool = False) -> dict[str, int]:
    """icao -> reports filed over `_HOURS`, for every candidate near any of `stations`.

    Resumable BY DESIGN: the cache is written after every batch, so a timeout, a gateway
    blip or a deliberate stop costs one batch instead of the whole pass. Re-running picks up
    exactly where it stopped, which is also what makes chunking by --stations safe."""
    cache = _load_cache()
    ids = sorted(candidates_for(stations, cache))
    todo = [i for i in ids
            if refresh or i not in cache["reports"] or not _fresh(cache["reports"][i])]
    print(f"\n{len(ids)} candidates near {len(stations)} stations; "
          f"{len(ids) - len(todo)} already measured, {len(todo)} to go", flush=True)

    now = datetime.now(timezone.utc).isoformat()
    for i in range(0, len(todo), _BATCH):
        for icao, n in _counts(todo[i:i + _BATCH]).items():
            cache["reports"][icao] = {"n": n, "at": now}
        _save_cache(cache)
        print(f"  measured {min(i + _BATCH, len(todo))}/{len(todo)}", flush=True)

    rates = {i: cache["reports"][i]["n"] for i in ids if i in cache["reports"]}
    silent = sum(1 for n in rates.values() if n == 0)
    print(f"\n{len(rates)} candidates, {silent} filed nothing in {_HOURS}h")
    return rates


def render(rates: dict[str, int]) -> str:
    lines = [
        '"""Measured METAR reporting rate per candidate site (generated by',
        'scripts/audit_neighbors.py).',
        "",
        "REPORTS[icao] = bulletins AWC served for that site over the measurement window.",
        "AWC's stationinfo catalog lists sites that MAY report; this says which ones DO, so",
        "build_neighbors.py can rank the fetchable set by 'nearest that actually reports'",
        "instead of 'nearest on the list'. Build-time input only -- nothing at runtime reads",
        "it, because filtering at build time makes get_terrain's blue dots true by construction.",
        "",
        "A rate, not a probe: a site missing one ob is not a dead site. Re-measure before a",
        "collection round -- these numbers age, and a station can go offline permanently.",
        '"""',
        "",
        f"WINDOW_HOURS: int = {_HOURS}",
        f"MEASURED_AT: str = {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')!r}",
        "",
        "# icao -> bulletins filed during the window (0 = served nothing at all)",
        "REPORTS: dict[str, int] = {",
    ]
    lines += [f"    {icao!r}: {n}," for icao, n in sorted(rates.items())]
    lines += [
        "}",
        "",
        "",
        "def reports(icao: str) -> int:",
        '    """Bulletins filed during the window; 0 for a silent or unmeasured site."""',
        "    return REPORTS.get(icao.upper(), 0)",
        "",
        "",
        "def measured(icao: str) -> bool:",
        '    """True if this site was actually measured. reports() cannot distinguish a silent',
        "    site from an unmeasured one -- both read 0 -- so build_neighbors.py checks this",
        '    before trusting a 0, or a station added since the audit gets an empty roster."""',
        "    return icao.upper() in REPORTS",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", help="comma-separated ICAOs to measure this pass "
                                       "(default: all of stations.poll_icaos())")
    ap.add_argument("--chunk", type=int, metavar="N",
                    help="measure only the first N unmeasured stations, then stop -- "
                         "re-run to continue where it left off")
    ap.add_argument("--refresh", action="store_true",
                    help="re-measure even sites the cache already holds")
    ap.add_argument("--check", action="store_true",
                    help="re-measure, diff against the committed table, do not write")
    args = ap.parse_args()

    stations = ([s.strip().upper() for s in args.stations.split(",") if s.strip()]
                if args.stations else poll_icaos())
    if args.chunk:
        cache = _load_cache()
        pending = [s for s in stations
                   if s not in cache["candidates"] or not _fresh(cache["candidates"][s])]
        stations = (pending or stations)[:args.chunk]
        print(f"chunk: {len(stations)} station(s) this pass -- {', '.join(stations)}\n")

    rates = build(stations, refresh=args.refresh)
    # Writing the module needs the WHOLE table, else a chunked pass would truncate it.
    # MEMBERSHIP, not length: --stations with 71 ICAOs that are not the roster has the right
    # COUNT and the wrong SET, and would pass a length test and write a table measured over
    # stations the archive does not poll.
    if not set(stations) >= set(poll_icaos()):
        done = len(_load_cache()["candidates"])
        print(f"\npartial pass: {done}/{len(poll_icaos())} stations enumerated. "
              f"Re-run without --stations/--chunk to finish and write {_OUT.name}.")
        return 0
    text = render(rates)
    if args.check:
        # MEASURED_AT always differs, so compare only the table -- the point of --check is
        # whether the same sites still report, not whether the clock moved.
        def table(s: str) -> str:
            return s[s.index("REPORTS: dict"):] if "REPORTS: dict" in s else ""
        current = _OUT.read_text() if _OUT.exists() else ""
        if table(current) == table(text):
            print("\nCHECK PASS: committed neighbor_rates.py matches a fresh measurement.")
            return 0
        print("\nCHECK FAIL: reporting has changed since the committed measurement.")
        return 1
    _OUT.write_text(text)
    print(f"\nwrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
