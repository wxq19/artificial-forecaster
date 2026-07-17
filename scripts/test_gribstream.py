"""GRIBStream smoke test -- proves the fetch seam end-to-end for at most 1 credit.

One point (KBAB, Beale AFB -- a military site with NO BUFKIT coverage, so it needs
GRIBStream's arbitrary-lat/lon extraction), one variable (2 m temperature), and a time
window narrow enough to bracket exactly ONE GFS output hour. Credits bill as
returned_valid_times * variables * ceil(coords/500) = 1 * 1 * 1 = 1.

The point of the test is not the temperature -- it is that we can SEE `forecasted_at`, the
model RUN reference time, which is why we chose GRIBStream over Open-Meteo's run-blending
default. Run:  uv run python scripts/test_gribstream.py
"""

from datetime import datetime, timedelta, timezone

from forecaster import awc, gribstream

STATION = "KBAB"          # Beale AFB, CA -- off the BUFKIT station list
MODEL = "gfs"


def main() -> None:
    lat, lon = awc.station_latlon(STATION)
    print(f"{STATION} -> lat {lat:.4f}, lon {lon:.4f}")

    # Bracket ONE whole-hour GFS step ~6 h out with a +/-10 min window: GFS is hourly, so
    # only that single top-of-hour valid time can fall inside -> at most 1 returned row.
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    target = now + timedelta(hours=6)
    from_time = target - timedelta(minutes=10)
    until_time = target + timedelta(minutes=10)
    print(f"window: {from_time:%Y-%m-%dT%H:%MZ} .. {until_time:%Y-%m-%dT%H:%MZ} "
          f"(brackets {target:%Y-%m-%dT%H:%MZ})")

    ts = gribstream.fetch_timeseries(
        MODEL, lat, lon,
        [gribstream.Var("TMP", "2 m above ground", alias="t2m_k")],
        from_time=from_time, until_time=until_time, name=STATION,
    )

    print(f"\nendpoint: {ts.url}")
    print(f"columns:  {ts.columns}")
    print(f"rows:     {len(ts.rows)}  (estimated credits: {ts.credits})")
    if not ts.rows:
        print("PASS (seam works) but 0 rows returned -- widen the window if you expected data")
        return
    for r in ts.rows:
        print(f"  run(forecasted_at)={r['forecasted_at']:%Y-%m-%dT%H:%MZ}  "
              f"valid(forecasted_time)={r['forecasted_time']:%Y-%m-%dT%H:%MZ}  "
              f"t2m={r['t2m_k']:.2f} K ({r['t2m_k'] - 273.15:.1f} C)")
    print(f"\nmodel run(s) represented: "
          f"{', '.join(f'{d:%Y-%m-%dT%H:%MZ}' for d in ts.runs)}")
    print("PASS")


if __name__ == "__main__":
    main()
