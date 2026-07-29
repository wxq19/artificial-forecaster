"""Regenerate src/forecaster/station_sites.py -- the frozen ICAO -> (lat, lon) table.

WHY THIS EXISTS. Every other static geographic fact in this repo is committed (neighbors.py,
upper_air_sites.py, radarsites.py). The one that was not is the STATION'S OWN position, which
`awc.station_latlon` fetched from the network on every call. That cost two things:
  - TIME: AWC throttles to 1 req/s, so a 71-station archive sweep spent ~103 s (measured
    2026-07-29: 1.44 s per cold lookup) re-learning coordinates that have not moved.
  - CORRECTNESS UNDER ARCHIVE-ONLY REPLAY: `station_latlon` is called from five places in
    tools.py, each a LIVE network call inside the agent loop. Round 2 serves tools only from
    the archive (owner, 2026-07-28), so a live lookup there is a hole in the seal -- and it
    gates get_map, get_imagery and the radar cascade, i.e. the busiest tools we have.

WHAT IS IN IT. Two sources, unioned, because both kinds of id reach `station_latlon`:
  - the 71 archived stations (`stations.poll_icaos()`), fetched here from AWC's stationinfo
    product -- the genuinely missing data;
  - every distinct NEIGHBOUR already frozen in neighbors.py, which carries their lat/lon.
    Those ids are not roster stations but the agent does pass them: `get_hazard_scan(
    station="KWRI", location="KNEL")` names a neighbour. A roster-only table would still hit
    the network on exactly those paths and defeat the purpose. They cost nothing to include
    because the coordinates are already committed -- this only re-keys them by ICAO.

An id in NEITHER set still falls through to the live lookup in awc.py. That is deliberate:
the table is an accelerator and a seal for the ids we KNOW, not a claim to know every airfield.

  uv run python scripts/build_station_sites.py            # regenerate the committed table
  uv run python scripts/build_station_sites.py --check     # recompute, diff, do not write
"""

import sys
from pathlib import Path

from forecaster import awc, neighbors, stations

_OUT = Path(__file__).resolve().parents[1] / "src" / "forecaster" / "station_sites.py"


def build() -> dict[str, tuple[float, float]]:
    """ICAO -> (lat, lon) for every roster station and every frozen neighbour."""
    icaos = stations.poll_icaos()
    out: dict[str, tuple[float, float]] = {}

    # Roster stations: one batched stationinfo request, not 71 throttled ones.
    rows = awc._get("stationinfo", {"ids": ",".join(icaos), "format": "json"}) or []
    got = {}
    for r in rows:
        icao = (r.get("icaoId") or "").upper()
        if icao and r.get("lat") is not None and r.get("lon") is not None:
            got[icao] = (round(float(r["lat"]), 4), round(float(r["lon"]), 4))
    missing = [i for i in icaos if i not in got]
    if missing:
        # REFUSE rather than emit a short table, the same rule build_neighbors.build() and
        # build_upper_air_sites.build() follow. A station silently absent here would fall
        # through to the live lookup forever and nothing would ever say so.
        raise SystemExit(f"stationinfo returned no position for {len(missing)}: "
                         f"{', '.join(missing)} -- refusing to write a partial table")
    out.update(got)
    print(f"roster: {len(got)} stations in one batched request")

    # Neighbours: already frozen with their positions, just re-keyed by ICAO. A neighbour
    # that IS a roster station keeps the roster value -- same source, no conflict.
    n_added = 0
    for home in icaos:
        for icao, _d, _b, _e, la, lo in neighbors.neighbors_of(home):
            key = icao.upper()
            if key not in out:
                out[key] = (round(float(la), 4), round(float(lo), 4))
                n_added += 1
    print(f"neighbours: {n_added} further ids taken from the frozen neighbors.py table")
    return dict(sorted(out.items()))


def render(table: dict[str, tuple[float, float]], n_roster: int) -> str:
    lines = [
        '"""Frozen ICAO -> (lat, lon). GENERATED -- do not edit by hand.',
        "",
        "Static geography, committed for the same reason neighbors.py and upper_air_sites.py",
        "are: a position does not change, and re-fetching it costs a throttled round trip on",
        "the hottest path in the agent loop. `awc.station_latlon` reads this FIRST and only",
        "falls back to the network for an id that is not here.",
        "",
        f"Covers the {n_roster} archived stations plus every neighbour frozen in neighbors.py,",
        "because the agent passes neighbour ids too (get_hazard_scan location=...).",
        "",
        "Regenerate/verify with scripts/build_station_sites.py [--check].",
        '"""',
        "",
        "# ICAO -> (lat, lon), degrees, 4 dp (~11 m -- far finer than any product we key on)",
        "STATION_LATLON: dict[str, tuple[float, float]] = {",
    ]
    for icao, (la, lo) in table.items():
        lines.append(f"    {icao!r}: ({la}, {lo}),")
    lines += [
        "}",
        "",
        "",
        "def latlon(icao: str) -> tuple[float, float] | None:",
        '    """(lat, lon) if this id is frozen here, else None -- caller decides the fallback."""',
        "    return STATION_LATLON.get(icao.upper())",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    check = "--check" in sys.argv[1:]
    table = build()
    text = render(table, len(stations.poll_icaos()))
    if check:
        current = _OUT.read_text() if _OUT.exists() else ""
        if current == text:
            print(f"\nCHECK PASS: committed station_sites.py matches a fresh build "
                  f"({len(table)} ids).")
            return 0
        print("\nCHECK FAIL: committed station_sites.py differs from a fresh build.")
        return 1
    _OUT.write_text(text)
    print(f"\nwrote {_OUT} ({len(table)} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
