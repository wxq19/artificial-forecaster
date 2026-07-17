"""Multi-model state over the FULL 30 h TAF window, sampled 2-hourly (15 steps) to show
the whole span economically. Reuses the field specs + formatters from the full demo.
Prints each model's table + credit cost. Run:  uv run python scripts/gribstream_30h.py
"""

from datetime import timedelta

from forecaster import awc, gribstream
from gribstream_full_demo import (
    _SFC, _ceil_ft, _c, _floor, _kt, _utcnow, _vis_sm, _wind_cell,
)

STATION = "KMSP"


def main():
    lat, lon = awc.station_latlon(STATION)
    now = _floor(_utcnow())
    times = [now + timedelta(hours=h) for h in range(2, 31, 2)]   # 2,4,...,30 -> 15 steps
    charged = 0
    for model in ("gfs", "hrrr", "nbm"):
        # drop MSLP for the wide view: only GFS returns it under PRMSL (naming gap flagged)
        variables = [v for v in _SFC[model] if v.alias != "mslp"]
        ts = gribstream.fetch_timeseries(model, lat, lon, variables, name=STATION, times=times)
        charged += ts.charged
        rows = sorted(ts.rows, key=lambda r: r["forecasted_time"])
        run = ts.runs[0] if ts.runs else None
        print(f"\n{model.upper()} surface forecast for {STATION} -- run {run:%Y-%m-%dT%HZ}, "
              f"30 h at 2-hourly ({'cached' if ts.cached else f'{ts.charged} cr'})")
        print(f"{'Valid (Z)':<15}{'T C':>5}{'Td C':>6}{'Wind':>8}{'Gst':>5}"
              f"{'Cld%':>6}{'Vis':>6}{'Ceil ft':>9}")
        for r in rows:
            t, td, cld = _c(r.get("t2m")), _c(r.get("td2m")), r.get("tcdc")
            gk = _kt(r.get("gust"))
            print(
                f"{r['forecasted_time']:%Y-%m-%dT%HZ}"
                f"{('%5.0f' % t) if t is not None else '   --'}"
                f"{('%6.0f' % td) if td is not None else '    --'}"
                f"{_wind_cell(r, model):>8}"
                f"{('%5.0f' % gk) if gk is not None else '   --'}"
                f"{('%6.0f' % cld) if cld is not None else '    --'}"
                f"{_vis_sm(r.get('vis')):>6}"
                f"{_ceil_ft(r.get('ceil')):>9}"
            )
    print(f"\ncredits billed this run: {charged}")


if __name__ == "__main__":
    main()
