"""One-off probe: confirm the ECMWF IFS (ifsoper) variable names return data on GRIBStream,
and surface the two enable-time quirks (tcc fraction-vs-percent; 'pl <hPa>' pressure levels).

    uv run python scripts/probe_ifs.py

Names are from the official model page (gribstream.com/models/ifsoper): 2t/2d/10u/10v/msl/tcc
@ 'sfc', pressure fields t/r/u/v/w @ 'pl 500' etc. This pulls one valid time at KMSP and
reports which come back finite, so `modeldata` can flip `_IFS_ENABLED` and add IFS to
gribstream.MODELS with confidence. Costs ~a handful of credits (1 time x ~9 vars).
"""

from datetime import timedelta

from forecaster import awc, gribstream, modeldata

STATION = "KMSP"

SURFACE = [gribstream.Var("2t", "sfc", "t2m"), gribstream.Var("2d", "sfc", "td2m"),
           gribstream.Var("10u", "sfc", "u10"), gribstream.Var("10v", "sfc", "v10"),
           gribstream.Var("msl", "sfc", "mslp"), gribstream.Var("tcc", "sfc", "tcc")]
PRESSURE = [gribstream.Var("t", "pl 500", "t500"), gribstream.Var("r", "pl 500", "r500"),
            gribstream.Var("u", "pl 300", "u300"), gribstream.Var("v", "pl 300", "v300"),
            gribstream.Var("w", "pl 500", "w500")]


def main() -> None:
    lat, lon = awc.station_latlon(STATION)
    anchor = modeldata._utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    print(f"probing IFS (ifsoper) at {STATION} valid {anchor:%Y-%m-%dT%HZ}")
    charged = 0
    for label, vs in [("surface", SURFACE), ("pressure", PRESSURE)]:
        try:
            ts = gribstream.fetch_timeseries("ifsoper", lat, lon, vs, times=[anchor],
                                             name=STATION, use_cache=False)
            charged += ts.charged
            r = ts.rows[0] if ts.rows else {}
            print(f"  {label} (run {ts.runs[0]:%Y-%m-%dT%HZ}):" if ts.runs else f"  {label}:")
            for v in vs:
                val = r.get(v.alias)
                print(f"    {v.name:<4} @ {v.level:<8} -> "
                      + ("null/masked" if val is None else f"{val:.3f}"))
        except ValueError as e:
            print(f"  {label} -> ERROR: {e}")
    print(f"credits charged: {charged}")
    print("If tcc is ~0-1 (not 0-100), the model-state formatter needs a per-model *100 for IFS.")
    print("If all finite: flip modeldata._IFS_ENABLED and add 'ifsoper' to gribstream.MODELS.")


if __name__ == "__main__":
    main()
