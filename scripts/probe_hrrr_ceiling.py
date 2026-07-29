"""One-off probe: which CEILING field does HRRR expose on GRIBStream, and is it real?

    uv run python scripts/probe_hrrr_ceiling.py

ANSWERED 2026-07-29 -- THERE IS NO HRRR CEILING BUG. Keep this script only as the record of
how that was settled; nothing in modeldata.py changed as a result.

THE SUSPICION. At CONUS stations (so the OCONUS-null effect is excluded) `ceil` was 100%
non-null for GFS, 84.6% for NBM and only 21.1% for HRRR -- same stations, same valid times,
which looked like the per-model naming split _MSLP already handles (GFS PRMSL vs HRRR MSLMA,
settled by probe_hrrr_mslp.py).

WHAT THIS PROBE FOUND. `CEIL` at any level returns ZERO ROWS for HRRR -- the field does not
exist there, so HGT @ 'cloud ceiling' was already the right request. The reference models gave
the real answer:
    nbm  CEIL@cloud ceiling ->  88888  3352  88888  1376
    gfs  HGT @cloud ceiling ->  15943  3217   3691  20000
**88888 and 20000 are "no ceiling" SENTINELS, not heights.** GFS and NBM pad unlimited-ceiling
hours with a magic number; HRRR leaves them missing. As stored: 70% of GFS non-null ceilings
are the 20000 fill and 82% of NBM's are sentinels, while HRRR's max is 15,623 ft with no fill
at all. Counting only REAL ceilings: GFS 49,284, HRRR 31,698, NBM 24,334 -- HRRR is mid-pack.

SO THE COMPARISON WAS THE BUG, not the data: raw non-null counts are not comparable across
models with different missing-value conventions. `tools.py` already maps both None and >15000
to "unlimited", so all three models render identically to the agent.

Costs a few credits: 1 point x a handful of valid times x 1-2 vars per model (32 when run).
"""

from datetime import timedelta

from forecaster import awc, gribstream, modeldata

# Two stations, deliberately: one usually-cloudy maritime-influenced field and one that is
# often clear. A single station cannot separate "wrong field" from "no cloud today".
STATIONS = ("KWRI", "KDMA")
CANDIDATES = [("CEIL", "cloud ceiling"), ("HGT", "cloud ceiling"), ("CEIL", "surface")]


def main() -> None:
    anchor = modeldata._utcnow().replace(minute=0, second=0, microsecond=0)
    times = [anchor + timedelta(hours=h) for h in (2, 5, 8, 11)]
    charged = 0
    for station in STATIONS:
        lat, lon = awc.station_latlon(station)
        print(f"\n=== {station} ({lat:.2f},{lon:.2f}) valid "
              f"{times[0]:%dT%HZ}..{times[-1]:%dT%HZ} ===")

        # Reference: what the models that DO work say about cloud here. If GFS reports a
        # ceiling and HRRR does not, the field is wrong; if neither does, it is clear.
        for model, var in (("gfs", gribstream.Var("HGT", "cloud ceiling", "ceil")),
                           ("nbm", gribstream.Var("CEIL", "cloud ceiling", "ceil")),
                           ("gfs", gribstream.Var("TCDC", "entire atmosphere", "tcdc"))):
            try:
                ts = gribstream.fetch_timeseries(model, lat, lon, [var], times=times,
                                                 name=station, use_cache=False)
                charged += ts.charged
                vals = [r.get(var.alias) for r in ts.rows]
                got = [f"{v:.0f}" if v is not None else "--" for v in vals]
                print(f"  REF {model:<5} {var.name:<5}@{var.level:<14} {' '.join(got)}")
            except Exception as e:  # noqa: BLE001
                print(f"  REF {model:<5} {var.name:<5} ERROR: {type(e).__name__}: {e}")

        for name, level in CANDIDATES:
            try:
                ts = gribstream.fetch_timeseries(
                    "hrrr", lat, lon, [gribstream.Var(name, level, "ceil")],
                    times=times, name=station, use_cache=False)
                charged += ts.charged
                vals = [r.get("ceil") for r in ts.rows]
                n_ok = sum(v is not None for v in vals)
                got = [f"{v:.0f}" if v is not None else "--" for v in vals]
                print(f"  hrrr    {name:<5}@{level:<14} {' '.join(got)}"
                      f"   ({n_ok}/{len(vals)} non-null)")
            except Exception as e:  # noqa: BLE001
                print(f"  hrrr    {name:<5}@{level:<14} ERROR: {type(e).__name__}: {e}")

    print(f"\ncredits charged: {charged}")
    print("READ IT LIKE THIS: if a REF model reports a ceiling where every HRRR candidate is "
          "null, the field name is wrong. If REF tcdc is ~0 and everything is null, it is "
          "genuinely clear and HRRR's 21% is honest.")


if __name__ == "__main__":
    main()
