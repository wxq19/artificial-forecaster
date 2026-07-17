"""Prewarm + review the Esri shaded-relief maps for the roster (or a given station).

Terrain is static, so we fetch each station's tiles ONCE into the permanent cache
(data/terrain/esri_relief) and never again. Run this before a collection campaign so
get_terrain is instant at agent runtime. Renders with the SAME markers + adaptive radius the
tool uses, so the cache covers exactly the tiles get_terrain will need (sparse stations widen
the radius) and the review JPEG matches what the model sees. Writes each map to
data/charts/temp/ for eyeballing.

  uv run python scripts/fetch_terrain.py                 # all roster stations
  uv run python scripts/fetch_terrain.py --station KVBG   # one station
"""

import argparse
from pathlib import Path

from forecaster import awc, neighbors, stations, terrain, tools

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
        neigh = neighbors.neighbors_of(icao)
        markers = [(ic, la, lo) for ic, _d, _b, _e, la, lo in neigh]
        img = terrain.relief_map(lat, lon, markers=markers, context=neighbors.area_of(icao),
                                 radius_mi=tools._map_radius_mi(neigh), use_cache=True)
        p = out / f"terrain_{icao}.jpg"
        p.write_bytes(img)
        print(f"{icao}: {len(img) // 1024} KB -> {p}")
    except Exception as e:  # noqa: BLE001 -- report and continue to the next station
        print(f"{icao}: FAILED ({type(e).__name__}: {e})")
