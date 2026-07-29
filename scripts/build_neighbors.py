"""Regenerate src/forecaster/neighbors.py -- the static nearest-neighbor roster.

For EVERY ARCHIVED station (`stations.poll_icaos()` = the 10-station model roster PLUS the
archive-only sites) this freezes two things, mirroring build_radarsites.py. It covers the
whole polled set, not just the roster, because the 2026-07-28 decision freezes input
archives at all of them -- and `neighbors_of` degrades SILENTLY to an empty list off-roster,
so an unbuilt station gets an empty get_nearby_obs and a bare get_terrain map with no
warning anywhere.
  - NEIGHBORS: the 5 closest OTHER METAR airfields within 150 km THAT ACTUALLY REPORT --
    the FETCHABLE set get_nearby_obs reads obs for (dist_km, bearing, elev_delta, plus
    lat/lon so the map can place a marker exactly).
  - AREA_STATIONS: every OTHER box METAR site (nearest first, capped), positions only --
    map CONTEXT so the model sees the full local network, even though obs are only banked
    for the fetchable set. Silent sites land here.
The candidate catalog comes from AWC's `stationinfo` bbox product -- a STABLE membership
list of every known METAR site in the box, so the output does not depend on which sites
happened to answer at build time. Ranking + bearings via geo.py; elev_delta (neighbor minus
home, meters) comes free from stationinfo `elev`.

MEMBERSHIP IS NOT REPORTING. The catalog lists sites that MAY file a METAR, and ranking it
raw put dead entries in the fetchable set: measured 2026-07-28, 52 of 307 neighbors filed
nothing in 72 h, across 34 of the 71 stations -- KWRI's nearest (KNEL, 22.7 km) and ALL FIVE
of RKJK's. So candidates are filtered through the frozen measurement in neighbor_rates.py
(regenerate with scripts/audit_neighbors.py) before ranking. That keeps this build
deterministic -- reproducible from catalog geography plus a frozen table, never from live
reporting -- and makes get_terrain's "blue = obs-available" caption true by construction.

The tools that read this (get_nearby_obs, get_terrain) never touch AWC -- neighbor OBS flow
only through the leakage-safe DB seam (store.copy_obs), and positions are pure geography.

  uv run python scripts/build_neighbors.py            # regenerate the committed table
  uv run python scripts/build_neighbors.py --check     # recompute, diff vs committed, don't write
"""

import sys
from math import cos, radians
from pathlib import Path

from forecaster import awc
from forecaster.geo import haversine_km, nearest_n
from forecaster.stations import poll_icaos

# Tolerated missing on purpose: scripts/audit_neighbors.py imports THIS module to reuse
# _candidates, and it is the script that writes neighbor_rates.py. A hard import would make
# the pair unbootstrappable. build() raises with the fix instead.
try:
    from forecaster import neighbor_rates
except ImportError:                                    # pragma: no cover - bootstrap only
    neighbor_rates = None

_OUT = Path(__file__).resolve().parents[1] / "src" / "forecaster" / "neighbors.py"
_N = 5
_MAX_KM = 150.0
# How often a candidate must file to be FETCHABLE. Expressed as a RATE and scaled by the
# measured window, not as a raw bulletin count -- a hardcoded count silently changes meaning
# the day audit_neighbors.py measures a different window. 4/day is one ob per 6 hours: it
# keeps a 3-hourly synoptic reporter (SCFM, the only live neighbor SCCI has, files ~8/day)
# and drops sites that are dead or nearly so.
_MIN_REPORTS_PER_DAY = 4
_AREA_MAX = 40                     # cap context sites per station (file size + map clutter)
_LAT_PAD = 1.5                     # deg; > 150 km / 111 km-per-deg, with margin


def _min_reports() -> int:
    """Bulletin threshold for the window neighbor_rates.py was actually measured over."""
    return max(1, round(_MIN_REPORTS_PER_DAY * neighbor_rates.WINDOW_HOURS / 24))

# fetchable roster row and context row
NeighborRow = tuple[str, float, str, int, float, float]
AreaRow = tuple[str, float, float]


def _home_info(icao: str) -> tuple[float, float, int]:
    """(lat, lon, elev_m) for a roster station from AWC stationinfo."""
    for r in awc._get("stationinfo", {"ids": icao, "format": "json"}) or []:
        if r.get("lat") is not None and r.get("lon") is not None:
            return float(r["lat"]), float(r["lon"]), int(r.get("elev") or 0)
    raise ValueError(f"AWC has no station info for {icao}")


def _candidates(lat: float, lon: float) -> dict[str, tuple[float, float, int]]:
    """Every METAR-reporting station in a ~150 km bbox: icao -> (lat, lon, elev_m)."""
    lon_pad = min(_LAT_PAD / max(cos(radians(lat)), 0.2), 8.0)   # widen with latitude
    bbox = f"{lat - _LAT_PAD},{lon - lon_pad},{lat + _LAT_PAD},{lon + lon_pad}"
    out: dict[str, tuple[float, float, int]] = {}
    for r in awc._get("stationinfo", {"bbox": bbox, "format": "json"}) or []:
        icao = r.get("icaoId")
        if not icao or "METAR" not in (r.get("siteType") or []):
            continue
        if r.get("lat") is None or r.get("lon") is None:
            continue
        out[icao] = (float(r["lat"]), float(r["lon"]), int(r.get("elev") or 0))
    return out


def build() -> tuple[dict[str, list[NeighborRow]], dict[str, list[AreaRow]]]:
    neighbors_t: dict[str, list[NeighborRow]] = {}
    area_t: dict[str, list[AreaRow]] = {}
    if neighbor_rates is None:
        raise SystemExit("neighbor_rates.py is missing -- run scripts/audit_neighbors.py first")
    floor = _min_reports()
    print(f"fetchable requires >= {floor} bulletins in the "
          f"{neighbor_rates.WINDOW_HOURS} h window measured {neighbor_rates.MEASURED_AT}\n")
    for home in poll_icaos():
        hlat, hlon, helev = _home_info(home)
        cands = _candidates(hlat, hlon)
        cands.pop(home, None)
        # Rank only sites that ACTUALLY report. The catalog is a membership list, so ranking
        # it raw hands get_nearby_obs dead entries and paints them blue on the terrain map.
        # Silent sites are not discarded -- they fall through to context below, where a violet
        # dot claims orientation value only, which is still true of an airfield that is quiet.
        # UNMEASURED IS NOT SILENT. reports() returns 0 for both, so a station added since the
        # last audit scores 0 on every candidate and is emitted with an EMPTY fetchable list --
        # which get_nearby_obs then states as a fact ("no other METAR-reporting airfield within
        # 150 km"). Refuse rather than render a build gap as meteorology. Only candidates
        # INSIDE the search radius are checked; one beyond it could never have ranked anyway.
        unmeasured = sorted(
            icao for icao, (la, lo, _e) in cands.items()
            if not neighbor_rates.measured(icao)
            and haversine_km(hlat, hlon, la, lo) <= _MAX_KM)
        if unmeasured:
            raise SystemExit(
                f"{home}: {len(unmeasured)} candidate(s) within {_MAX_KM:.0f} km are absent "
                f"from neighbor_rates.py ({', '.join(unmeasured[:8])}"
                f"{', ...' if len(unmeasured) > 8 else ''}). Absent means UNMEASURED, not "
                f"silent, and reports() cannot tell the two apart -- ranking them as silent "
                f"would emit an empty or truncated fetchable roster. Re-measure with "
                f"scripts/audit_neighbors.py (it resumes), then build again.")
        catalog = [(icao, la, lo) for icao, (la, lo, _e) in cands.items()
                   if neighbor_rates.reports(icao) >= floor]
        ranked = nearest_n(hlat, hlon, catalog, n=_N, max_km=_MAX_KM)
        fetchable = {icao for icao, _d, _b in ranked}
        skipped = len(cands) - len(catalog)
        neighbors_t[home] = [
            (icao, dist, brg, cands[icao][2] - helev,
             round(cands[icao][0], 4), round(cands[icao][1], 4))
            for icao, dist, brg in ranked
        ]
        # context = every OTHER box site (not fetchable), nearest first, capped
        others = sorted(
            (
                (round(haversine_km(hlat, hlon, la, lo), 1), icao, round(la, 4), round(lo, 4))
                for icao, (la, lo, _e) in cands.items() if icao not in fetchable
            ),
            key=lambda t: (t[0], t[1]),
        )
        area_t[home] = [(icao, la, lo) for _d, icao, la, lo in others[:_AREA_MAX]]
        print(f"{home}: {len(neighbors_t[home])} fetchable "
              f"({', '.join(f'{i}@{d:.0f}km' for i, d, _b, _e, _la, _lo in neighbors_t[home])}), "
              f"{len(area_t[home])} context, {skipped} silent skipped")
    return neighbors_t, area_t


def render(neighbors_t: dict[str, list[NeighborRow]],
           area_t: dict[str, list[AreaRow]]) -> str:
    lines = [
        '"""Static nearest-neighbor roster (generated by scripts/build_neighbors.py).',
        "",
        "For every ARCHIVED station (roster + archive-only), two frozen sets from AWC's",
        "stationinfo catalog:",
        "  - NEIGHBORS: the 5 closest OTHER METAR airfields within 150 km THAT REPORT -- the",
        "    FETCHABLE set. (icao, dist_km, 16-pt bearing FROM home, elev_delta_m = neighbor",
        "    minus home, lat, lon). get_nearby_obs reads OBS for these from the leakage-safe",
        "    DB seam, never a live fetch.",
        "  - AREA_STATIONS: every OTHER box METAR site (nearest first, capped), positions only --",
        "    map CONTEXT for get_terrain so the model sees the full local network; NOT fetchable.",
        "    Catalogued-but-silent airfields land here.",
        "",
        "Geography, filtered by the frozen reporting measurement in neighbor_rates.py -- the",
        "catalog lists sites that MAY report, and ranking it raw put dead entries in the",
        f"fetchable set. A candidate needs >= {_min_reports()} bulletins over the "
        f"{neighbor_rates.WINDOW_HOURS} h",
        f"window measured {neighbor_rates.MEASURED_AT}.",
        "Regenerate/verify with build_neighbors.py [--check]; re-measure with audit_neighbors.py.",
        '"""',
        "",
        "# Search radius the FETCHABLE set was built with. Emitted here so a reader (e.g.",
        "# get_nearby_obs reporting 'no obs in the local region') states the same number the",
        "# table was actually built from, instead of restating a literal that can drift.",
        f"MAX_KM: float = {_MAX_KM}",
        f"MAX_FETCHABLE: int = {_N}",
        "",
        "# home -> [(neighbor_icao, dist_km, bearing, elev_delta_m, lat, lon), ...] nearest first",
        "NEIGHBORS: dict[str, list[tuple[str, float, str, int, float, float]]] = {",
    ]
    for home in neighbors_t:
        lines.append(f"    {home!r}: [")
        for icao, dist, brg, de, la, lo in neighbors_t[home]:
            lines.append(f"        ({icao!r}, {dist}, {brg!r}, {de}, {la}, {lo}),")
        lines.append("    ],")
    lines += [
        "}",
        "",
        "# home -> [(icao, lat, lon), ...] -- context box sites for the map (NOT fetchable)",
        "AREA_STATIONS: dict[str, list[tuple[str, float, float]]] = {",
    ]
    for home in area_t:
        lines.append(f"    {home!r}: [")
        for icao, la, lo in area_t[home]:
            lines.append(f"        ({icao!r}, {la}, {lo}),")
        lines.append("    ],")
    lines += [
        "}",
        "",
        "",
        "def neighbors_of(icao: str) -> list[tuple[str, float, str, int, float, float]]:",
        '    """Fetchable nearest-neighbor rows for a station (empty if none/off-roster)."""',
        "    return NEIGHBORS.get(icao.upper(), [])",
        "",
        "",
        "def area_of(icao: str) -> list[tuple[str, float, float]]:",
        '    """Context box-station positions for the map (empty if none/off-roster)."""',
        "    return AREA_STATIONS.get(icao.upper(), [])",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    check = "--check" in sys.argv[1:]
    neighbors_t, area_t = build()
    text = render(neighbors_t, area_t)
    if check:
        current = _OUT.read_text() if _OUT.exists() else ""
        if current == text:
            print("\nCHECK PASS: committed neighbors.py matches a fresh build.")
            return 0
        print("\nCHECK FAIL: committed neighbors.py differs from a fresh build.")
        return 1
    _OUT.write_text(text)
    print(f"\nwrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
