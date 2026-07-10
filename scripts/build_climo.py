"""Build (materialize) the climatology product for a station-month into the persistent
DuckDB, and preview it exactly as the get_climo tool renders it.

    uv run python scripts/build_climo.py --station KLSV --months 7
    uv run python scripts/build_climo.py --station KLSV --months 7,8 --check

The raw multi-year history is fetched into a throwaway scratch DB and discarded; only
the climo_* product rows persist. --check retains the scratch for
one run and recomputes a few cells in pure Python to assert the SQL aggregation is right.
"""

import argparse
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory

from forecaster import climo, store, tools
from forecaster.config import settings


def _months(s: str) -> list[int]:
    out = [int(x) for x in s.split(",") if x.strip()]
    if not out or any(not 1 <= m <= 12 for m in out):
        raise argparse.ArgumentTypeError("months must be 1-12, comma-separated, e.g. 7 or 7,8")
    return out


def _print_summary(result: dict) -> None:
    meta, ingest = result["meta"], result["ingest"]
    print(f"\n=== BUILD {result['station']} months={ingest['months']} ===")
    print(f"metadata: lat={meta['lat']} lon={meta['lon']} tz={meta['tzname']} "
          f"offset(std)={meta['utc_offset_hours']} source={meta['source']}")
    print(f"ingest: inserted {ingest['inserted']} scratch obs across "
          f"{ingest['start_year']}-{ingest['end_year']}")
    print("  per-year obs: " + ", ".join(f"{y}:{n}" for y, n in sorted(ingest["per_year"].items())))
    if ingest["errors"]:
        print(f"  parse errors: {len(ingest['errors'])} (first: {ingest['errors'][0][1]})")
    else:
        print("  parse errors: 0")
    for m in result["rebuilt"]:
        print(f"  month {m['month']}: {m['n_obs_routine']} routine / {m['n_obs_all']} all obs, "
              f"{m['n_days']} days, POR {m['por_start_year']}-{m['por_end_year']} "
              f"({m['n_years_used']} yr)")


def _preview(station: str, months: list[int], db_path: str) -> None:
    con = store.connect(db_path, read_only=True)
    try:
        meta = store.climo_meta(con, station)
        for month in months:
            monthly = store.climo_month(con, station, month)
            hourly = store.climo_hours(con, station, month)
            if monthly:
                print("\n" + "-" * 70)
                print(tools._fmt_climo(meta, monthly, hourly))
    finally:
        con.close()


def _check(station: str, months: list[int], offset: float, scratch_db: str, db_path: str) -> None:
    """Recompute a few cells in pure Python from the retained scratch obs and assert
    equality with the stored climo row (the SQL aggregation is the thing under test)."""
    from forecaster import wxcodes

    sc = store.connect(scratch_db, read_only=True)
    obs = store.window(sc, station, datetime(1900, 1, 1), datetime(2100, 1, 1))
    sc.close()

    def local_month(o):
        return (o["obs_time"] + timedelta(hours=offset)).month

    con = store.connect(db_path, read_only=True)
    print("\n=== --check (pure-Python recompute vs stored climo) ===")
    ok = True
    for month in months:
        mobs = [o for o in obs if local_month(o) == month]
        routine = [o for o in mobs if o["report_type"] == "METAR"]
        # 1) daily TX mean (all obs, local day)
        by_day: dict = {}
        for o in mobs:
            if o["temp_c"] is None:
                continue
            d = (o["obs_time"] + timedelta(hours=offset)).date()
            by_day[d] = max(by_day.get(d, -999), o["temp_c"])
        tx_mean = sum(by_day.values()) / len(by_day) if by_day else None
        # 2) monthly pct_ts (routine)
        pct_ts = 100.0 * sum(any("TS" in w for w in o["weather"]) for o in routine) / len(routine)
        stored = store.climo_month(con, station, month)
        for label, got, exp in [("tx_mean", tx_mean, stored["tx_mean"]),
                                 ("pct_ts", pct_ts, stored["pct_ts"])]:
            good = got is not None and abs(got - exp) < 0.05
            ok &= good
            print(f"  month {month} {label}: python {got:.3f} vs stored {exp:.3f} "
                  f"-> {'PASS' if good else 'FAIL'}")
        # 3) one hourly cell: pct_cig_lt_1000 at the hour with the most obs
        hrs = store.climo_hours(con, station, month)
        target = max(hrs, key=lambda r: r["n_obs"])["hour_utc"]
        hr_routine = [o for o in routine if o["obs_time"].hour == target]
        py_cig = 100.0 * sum((o["ceiling_ft"] or 99999) < 1000 for o in hr_routine) / len(hr_routine)
        stored_cig = next(r["pct_cig_lt_1000"] for r in hrs if r["hour_utc"] == target)
        good = abs(py_cig - stored_cig) < 0.05
        ok &= good
        print(f"  month {month} hour {target:02d}Z pct_cig_lt_1000: python {py_cig:.3f} vs "
              f"stored {stored_cig:.3f} -> {'PASS' if good else 'FAIL'}")
        # 4) wxcodes vs regex TS agreement on a sample
        sample = [o for o in routine if o["weather"]][:200]
        agree = sum(
            any("TS" in w for w in o["weather"])
            == any(wxcodes.classify(w).family == "thunder" for w in o["weather"])
            for o in sample
        )
        ok &= agree == len(sample)
        print(f"  month {month} wxcodes-vs-regex TS: {agree}/{len(sample)} agree "
              f"-> {'PASS' if agree == len(sample) else 'FAIL'}")
    con.close()
    print(f"\n--check {'PASSED' if ok else 'FAILED'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the climatology product for a station-month.")
    ap.add_argument("--station", required=True, help="4-letter ICAO, e.g. KLSV")
    ap.add_argument("--months", required=True, type=_months, help="e.g. 7 or 7,8")
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="recompute a few cells and assert")
    args = ap.parse_args()
    station = args.station.upper()
    db_path = settings.db_path

    if args.check:
        with TemporaryDirectory(prefix="climo_check_") as tmp:
            result = climo.build(station, args.months, start_year=args.start_year,
                                 end_year=args.end_year, db_path=db_path, scratch_dir=tmp)
            _print_summary(result)
            _preview(station, args.months, db_path)
            _check(station, args.months, result["meta"]["utc_offset_hours"],
                   result["scratch_db"], db_path)
    else:
        result = climo.build(station, args.months, start_year=args.start_year,
                             end_year=args.end_year, db_path=db_path)
        _print_summary(result)
        _preview(station, args.months, db_path)


if __name__ == "__main__":
    main()
