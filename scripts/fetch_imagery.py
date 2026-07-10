"""Fetch satellite/radar imagery into data/imagery/temp/ for review + cache seeding.

A scripted, repeatable pull that exercises the imagery.py seam directly (no model). With
no args it grabs a representative sweep: geocolor over a few satellite regions, plus the
national/regional/station radar modes. Or target one image with the flags. Provenance URL
is printed for each. Refresh anytime by re-running.

Examples:
  uv run python scripts/fetch_imagery.py                       # the review sweep
  uv run python scripts/fetch_imagery.py --kind satellite --region conus_east --product infrared
  uv run python scripts/fetch_imagery.py --kind radar --station KLSV
  uv run python scripts/fetch_imagery.py --kind radar --region southern_plains
"""

import argparse
from pathlib import Path

from forecaster import awc, imagery

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--kind", choices=["satellite", "radar"])
_ap.add_argument("--product")
_ap.add_argument("--region")
_ap.add_argument("--station")
_ap.add_argument("--outdir", default="data/imagery/temp")
_args = _ap.parse_args()

OUT = Path(_args.outdir)
OUT.mkdir(parents=True, exist_ok=True)


def _ext(data: bytes) -> str:
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    return "png"


def save(name: str, data: bytes, url: str) -> None:
    out = OUT / f"{name}.{_ext(data)}"
    out.write_bytes(data)
    print(f"  OK   {out.name:<34} ({len(data) // 1024:>4} KB)  {url}")


def sat(region: str, product: str) -> None:
    save(f"sat_{region}_{product}",
         imagery.fetch_satellite(region, product),
         imagery.satellite_url(region, product))


def radar_station(icao: str) -> None:
    lat, lon = awc.station_latlon(icao)
    near = imagery.nearest_radar(lat, lon)
    sid, dist = (near[0]["id"], near[1]) if near else ("--", float("nan"))
    print(f"  {icao}: nearest WSR-88D {sid} at {dist:.0f} km "
          f"(guard {imagery.RADAR_STATION_GUARD_KM:.0f} km)")
    save(f"radar_station_{icao}",
         imagery.fetch_radar("station", center=(lat, lon)),
         imagery.radar_url("station", center=(lat, lon)))


def radar_region(region: str) -> None:
    mode = "national" if region == "national" else "regional"
    save(f"radar_{region}",
         imagery.fetch_radar(mode, region=region),
         imagery.radar_url(mode, region=region))


if _args.kind == "satellite":
    sat(_args.region or "conus_east", _args.product or "geocolor")
elif _args.kind == "radar":
    if _args.station:
        radar_station(_args.station.upper())
    else:
        radar_region(_args.region or "national")
else:
    # Default review sweep: satellite products + regions, then radar modes.
    print("satellite:")
    for prod in ("geocolor", "infrared", "water_vapor"):
        sat("conus_east", prod)
    for reg in ("conus_west", "southern_plains", "hawaii"):
        sat(reg, "geocolor")
    print("radar:")
    radar_region("national")
    radar_region("southern_plains")
    for icao in ("KLSV", "KMSP", "KBLV"):
        try:
            radar_station(icao)
        except Exception as e:  # noqa: BLE001 -- record and continue so one bad site doesn't sink the run
            print(f"  FAIL radar_station_{icao}  {type(e).__name__}: {e}")

print(f"\n-> {OUT}")
