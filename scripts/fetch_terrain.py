"""Prewarm + review the OpenTopoMap relief maps for the roster (or a given station).

Terrain is static and OpenTopoMap is a small volunteer tile server, so we fetch each station's
tiles ONCE into the permanent cache (data/terrain/otm) and never again. Run this before a
collection campaign so get_terrain is instant (and polite) at agent runtime. Also writes each
relief map JPEG to data/charts/temp/ for eyeballing.

  uv run python scripts/fetch_terrain.py                 # all roster stations
  uv run python scripts/fetch_terrain.py --station KVBG   # one station
"""

import argparse
from pathlib import Path

from forecaster import awc, stations, terrain

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--station", help="single ICAO (default: whole roster)")
args = ap.parse_args()

icaos = [args.station.upper()] if args.station else stations.icaos()
out = Path("data/charts/temp")
out.mkdir(parents=True, exist_ok=True)

for icao in icaos:
    try:
        lat, lon = awc.station_latlon(icao)
        img = terrain.relief_map(lat, lon, use_cache=True)   # fetch+cache tiles, render JPEG
        p = out / f"terrain_{icao}.jpg"
        p.write_bytes(img)
        print(f"{icao}: {len(img) // 1024} KB -> {p}")
    except Exception as e:  # noqa: BLE001 -- report and continue to the next station
        print(f"{icao}: FAILED ({type(e).__name__}: {e})")
