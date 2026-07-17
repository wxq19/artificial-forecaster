"""One-off probe: which sea-level-pressure field does HRRR expose on GRIBStream?

    uv run python scripts/probe_hrrr_mslp.py

GFS uses PRMSL @ 'mean sea level'; HRRR is known to use MSLMA or MSLET (grib2 variants).
This pulls one valid time at KMSP for each candidate and reports which returns a finite
value, so modeldata._MSLP can be set correctly. Costs ~1-2 credits (1 time x 1-2 vars).
"""

from datetime import timedelta

from forecaster import awc, gribstream, modeldata

STATION = "KMSP"
CANDIDATES = ["MSLMA", "MSLET", "MSLP"]


def main() -> None:
    lat, lon = awc.station_latlon(STATION)
    anchor = modeldata._utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
    charged = 0
    print(f"probing HRRR MSLP at {STATION} valid {anchor:%Y-%m-%dT%HZ}")
    for name in CANDIDATES:
        try:
            ts = gribstream.fetch_timeseries(
                "hrrr", lat, lon, [gribstream.Var(name, "mean sea level", "mslp")],
                times=[anchor], name=STATION, use_cache=False,
            )
            charged += ts.charged
            val = ts.rows[0].get("mslp") if ts.rows else None
            ok = "OK" if val is not None else "null/masked"
            print(f"  {name:<7} @ 'mean sea level' -> {ok} "
                  f"({val/100:.1f} hPa)" if val is not None else f"  {name:<7} -> {ok}")
        except ValueError as e:
            print(f"  {name:<7} -> ERROR: {e}")
    print(f"credits charged: {charged}")
    print("Set modeldata._MSLP['hrrr'] to the OK candidate.")


if __name__ == "__main__":
    main()
