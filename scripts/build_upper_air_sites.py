"""Regenerate src/forecaster/upper_air_sites.py -- the static nearest-radiosonde roster.

For EVERY archived station (`stations.poll_icaos()`), freeze the nearest upper-air sites so
`get_sounding` can be pointed at a station instead of requiring the caller to know a WMO
number. Mirrors build_neighbors.py / build_radarsites.py: pure geography, deterministic,
`--check` verifies the committed file reproduces.

CATALOG SOURCE: NOAA's IGRA v2 station list (ncei.noaa.gov/pub/data/igra/igra2-station-list.txt)
-- a stable public text file, no auth, no rate limit, with the first/last year of record per
site so a decommissioned station can be excluded rather than silently recommended.

ID SPACE. IGRA ids are `<CC><N><8-digit>`; for network `M` (the WMO network) the last five
digits ARE the WMO number, which is what the `bufr` and `wyoming` sounding sources take.
Non-M networks carry no WMO id and are skipped -- they cannot be fetched by our sources, so
listing them would be a menu of things that do not work. SPC's 3-letter site ids are a
DIFFERENT namespace and are not derivable from this catalog; `get_sounding(source='spc')`
still needs its own id, which is why the tool description names the id space per source.

DISTANCE. No hard cap by default: radiosonde spacing is ~315 km in CONUS but far coarser in
Alaska and OCONUS, and a 600 km sounding is still real information a forecaster would use --
the distance is REPORTED so the model can weigh it, rather than the site being hidden.

  uv run python scripts/build_upper_air_sites.py           # regenerate
  uv run python scripts/build_upper_air_sites.py --check    # verify, do not write
"""

import sys
import urllib.request
from pathlib import Path

from forecaster import awc
from forecaster.geo import nearest_n
from forecaster.stations import poll_icaos

_OUT = Path(__file__).resolve().parents[1] / "src" / "forecaster" / "upper_air_sites.py"
_IGRA = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"

_N = 3                  # nearest sites to keep per station
_MAX_KM = 1500.0        # generous: a guard against nonsense, not a relevance filter
_MIN_LAST_YEAR = 2024   # still reporting recently enough to be worth offering


def _catalog() -> dict[str, tuple[str, float, float, int]]:
    """WMO-numbered, still-active radiosonde sites: wmo -> (name, lat, lon, last_year)."""
    req = urllib.request.Request(_IGRA, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    out: dict[str, tuple[str, float, float, int]] = {}
    for ln in text.splitlines():
        if len(ln) < 81:
            continue
        sid = ln[0:11]
        if sid[2] != "M":                       # non-WMO network -- unfetchable by our sources
            continue
        wmo = sid[6:11]
        try:
            lat, lon = float(ln[12:20]), float(ln[21:30])
            last = int(ln[77:81])          # [72:76] is the FIRST year -- do not confuse them
        except ValueError:
            continue
        if last < _MIN_LAST_YEAR or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue
        name = " ".join(ln[41:71].split())
        prev = out.get(wmo)
        if prev is None or last > prev[3]:      # keep the longest-running duplicate
            out[wmo] = (name, round(lat, 4), round(lon, 4), last)
    return out


def build() -> dict[str, list[tuple[str, str, float, str, float, float]]]:
    cat = _catalog()
    print(f"IGRA: {len(cat)} WMO sites active since {_MIN_LAST_YEAR}")
    points = [(wmo, v[1], v[2]) for wmo, v in cat.items()]
    table: dict[str, list[tuple[str, str, float, str, float, float]]] = {}
    for icao in poll_icaos():
        try:
            lat, lon = awc.station_latlon(icao)
        except Exception as e:  # noqa: BLE001 -- re-raised with the station named
            # REFUSE, do not skip. A skipped station is simply ABSENT from the generated
            # table, and absent degrades silently: sites_for returns [], get_sounding passes
            # the ICAO through as a WMO id, and the run ends in a false "site is not
            # reporting". A network blip would also read as real drift under --check. Same
            # rule as build_neighbors.build() for an unmeasured candidate.
            raise RuntimeError(
                f"{icao}: position lookup failed ({type(e).__name__}: {e}) -- refusing to "
                f"write a table with {icao} missing. Re-run when AWC answers.") from e
        rows = []
        for wmo, dist, brg in nearest_n(lat, lon, points, n=_N, max_km=_MAX_KM):
            name, slat, slon, _last = cat[wmo]
            rows.append((wmo, name, dist, brg, slat, slon))
        table[icao] = rows
        near = ", ".join(f"{w}@{d:.0f}km" for w, _n, d, _b, _la, _lo in rows) or "(none)"
        print(f"{icao}: {near}")
    return table


def render(table: dict[str, list[tuple[str, str, float, str, float, float]]]) -> str:
    lines = [
        '"""Static nearest-radiosonde roster (generated by scripts/build_upper_air_sites.py).',
        "",
        "For every archived station, the closest upper-air sites by great-circle distance:",
        "  (wmo, name, dist_km, 16-pt bearing FROM the station, lat, lon), nearest first.",
        "",
        "`wmo` is the id the `bufr` and `wyoming` get_sounding sources take. SPC's 3-letter",
        "site ids are a DIFFERENT namespace and are NOT derivable from this table.",
        "Distance is reported, not filtered: soundings are sparse outside CONUS and a distant",
        "ascent is still information -- the model weighs it. Pure geography.",
        "Regenerate/verify with build_upper_air_sites.py [--check].",
        '"""',
        "",
        "# station -> [(wmo, name, dist_km, bearing, lat, lon), ...] nearest first",
        "UPPER_AIR: dict[str, list[tuple[str, str, float, str, float, float]]] = {",
    ]
    for icao, rows in table.items():
        lines.append(f"    {icao!r}: [")
        for wmo, name, dist, brg, la, lo in rows:
            lines.append(f"        ({wmo!r}, {name!r}, {dist}, {brg!r}, {la}, {lo}),")
        lines.append("    ],")
    lines += [
        "}",
        "",
        "",
        "def sites_for(icao: str) -> list[tuple[str, str, float, str, float, float]]:",
        '    """Nearest radiosonde sites for a station (empty if the station is unknown)."""',
        "    return UPPER_AIR.get(icao.upper(), [])",
        "",
        "",
        "def nearest_wmo(icao: str) -> str | None:",
        '    """The single closest sounding site\'s WMO id, or None."""',
        "    got = sites_for(icao)",
        "    return got[0][0] if got else None",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    check = "--check" in sys.argv[1:]
    text = render(build())
    if check:
        current = _OUT.read_text() if _OUT.exists() else ""
        if current == text:
            print("\nCHECK PASS: committed upper_air_sites.py matches a fresh build.")
            return 0
        print("\nCHECK FAIL: committed upper_air_sites.py differs from a fresh build.")
        return 1
    _OUT.write_text(text)
    print(f"\nwrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
