"""GRIBStream multi-model MVP -- a minimal reference for three model-data capabilities.

Deliberately tiny (a few hours, three pressure levels) so it costs ~50 credits the first
run and 0 on a same-hour re-run (the local archive). One CONUS military site (KBAB) where
GFS/HRRR/NBM all reach. This is a REFERENCE to react to, not the final shape -- no agent
tool, no DuckDB archive yet. Run:  uv run python scripts/test_gribstream_mvp.py

Capabilities shown:
  1. Multi-model current/forecast state -- same fields from GFS/HRRR/NBM, side by side,
     each with its run stamp, so you can see how the models line up.
  2. Cross-model hazard confirmation -- don't trust one model: check whether the icing
     CONDITIONS (sub-freezing + moist at a level) show up in GFS AND HRRR before believing.
  3. Model-vs-obs verification history -- how a PAST run's forecast compared to what was
     actually observed (GRIBStream asOf pins the run; obs come from AWC METARs).
"""

from datetime import datetime, timedelta, timezone

from forecaster import awc, gribstream, metar

STATION = "KBAB"          # Beale AFB, CA -- CONUS (HRRR ok), off the BUFKIT list
_charged = 0              # running total of credits actually billed this run


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def _round_hour(t: datetime) -> datetime:
    return _floor_hour(t + timedelta(minutes=30))


def _fetch(model, lat, lon, variables, **kw) -> gribstream.TimeSeries:
    """Wrapper that tracks billed credits and prints a live/cached tag per call."""
    global _charged
    ts = gribstream.fetch_timeseries(model, lat, lon, variables, name=STATION, **kw)
    _charged += ts.charged
    tag = "cached  0 cr" if ts.cached else f"live  {ts.charged:>2} cr"
    print(f"    [{model:<4} {tag}]  run={_runs(ts)}")
    return ts


def _runs(ts: gribstream.TimeSeries) -> str:
    return ", ".join(f"{d:%Y-%m-%dT%HZ}" for d in ts.runs) or "-"


# --- Capability 1: multi-model current/forecast state --------------------------------
def cap1_multi_model_state(lat: float, lon: float) -> None:
    print("\n[1] MULTI-MODEL STATE -- 2 m temp/dewpoint, next 3 hours, GFS vs HRRR vs NBM")
    now = _floor_hour(_utcnow())
    variables = [gribstream.Var("TMP", "2 m above ground", "t2m"),
                 gribstream.Var("DPT", "2 m above ground", "td2m")]
    table: dict[datetime, dict[str, str]] = {}
    for model in ("gfs", "hrrr", "nbm"):
        try:
            ts = _fetch(model, lat, lon, variables,
                        from_time=now + timedelta(hours=1), until_time=now + timedelta(hours=3))
        except ValueError as e:
            print(f"    [{model:<4} N/A]  {e}")
            continue
        for r in ts.rows:
            t, td = r.get("t2m"), r.get("td2m")
            cell = (f"{t - 273.15:>4.1f}/{td - 273.15:<4.1f}"
                    if t is not None and td is not None else "  --  ")
            table.setdefault(r["forecasted_time"], {})[model] = cell
    print(f"\n    {'valid (Z)':<16}{'GFS T/Td C':<14}{'HRRR T/Td C':<14}{'NBM T/Td C':<14}")
    for valid in sorted(table):
        row = table[valid]
        print(f"    {valid:%Y-%m-%dT%HZ}   {row.get('gfs',' -- '):<13} "
              f"{row.get('hrrr',' -- '):<13} {row.get('nbm',' -- '):<13}")


# --- Capability 2: cross-model hazard confirmation -----------------------------------
def cap2_cross_model_hazard(lat: float, lon: float) -> None:
    print("\n[2] CROSS-MODEL HAZARD -- icing conditions (sub-freezing + RH>=80%) at "
          "850/700/500 mb, +3h")
    valid = _floor_hour(_utcnow()) + timedelta(hours=3)
    levels = ["850 mb", "700 mb", "500 mb"]
    variables = ([gribstream.Var("TMP", lv, f"t_{lv[:3]}") for lv in levels]
                 + [gribstream.Var("RH", lv, f"rh_{lv[:3]}") for lv in levels])
    verdicts: dict[str, dict[str, bool]] = {}   # level -> {model: icing?}
    for model in ("gfs", "hrrr"):               # both carry pressure levels; NBM is surface-first
        try:
            ts = _fetch(model, lat, lon, variables,
                        from_time=valid - timedelta(minutes=10),
                        until_time=valid + timedelta(minutes=10))
        except ValueError as e:
            print(f"    [{model:<4} N/A]  {e}")
            continue
        if not ts.rows:
            print(f"    [{model:<4}] no row at {valid:%HZ}")
            continue
        r = ts.rows[0]
        print(f"\n    {model.upper()} @ {valid:%Y-%m-%dT%HZ}:")
        for lv in levels:
            t_k, rh = r.get(f"t_{lv[:3]}"), r.get(f"rh_{lv[:3]}")
            if t_k is None or rh is None:
                continue
            t_c = t_k - 273.15
            icing = -15.0 <= t_c <= 0.0 and rh >= 80.0
            verdicts.setdefault(lv, {})[model] = icing
            print(f"      {lv:<7} T={t_c:>5.1f}C  RH={rh:>3.0f}%  -> "
                  f"{'ICING conditions' if icing else 'no icing'}")
    print("\n    cross-model agreement:")
    for lv in levels:
        v = verdicts.get(lv, {})
        if set(v.values()) == {True}:
            print(f"      {lv}: BOTH models show icing conditions -- corroborated")
        elif set(v.values()) == {False}:
            print(f"      {lv}: both models agree -- no icing")
        elif v:
            print(f"      {lv}: MODELS DISAGREE {v} -- single-model signal, treat with caution")


# --- Capability 3: model-vs-obs verification history ---------------------------------
def cap3_model_vs_obs(lat: float, lon: float) -> None:
    print("\n[3] MODEL vs OBS -- a past GFS run's 2 m temp forecast vs observed METAR temps")
    now = _floor_hour(_utcnow())
    as_of = now - timedelta(hours=12)           # pin a run from ~12 h ago
    try:
        ts = _fetch("gfs", lat, lon, [gribstream.Var("TMP", "2 m above ground", "t2m")],
                    from_time=now - timedelta(hours=7), until_time=now - timedelta(hours=1),
                    as_of=as_of)
    except ValueError as e:
        print(f"    GFS N/A: {e}")
        return
    # Observed temps from AWC METARs, keyed by nearest whole hour.
    obs: dict[datetime, int] = {}
    for obs_time, raw, _ in awc.fetch_metar(STATION, hours=8):
        o = metar.parse(raw)
        if o.temp_c is not None:
            obs[_round_hour(obs_time.replace(tzinfo=None))] = o.temp_c
    print(f"\n    run={_runs(ts)}")
    print(f"    {'valid (Z)':<16}{'fcst C':<9}{'obs C':<8}{'error':<8}")
    for r in ts.rows:
        valid = r["forecasted_time"]
        fcst_c = r["t2m"] - 273.15 if r.get("t2m") is not None else None
        obs_c = obs.get(valid)
        err = f"{fcst_c - obs_c:+.1f}" if fcst_c is not None and obs_c is not None else "  --"
        fcst_s = f"{fcst_c:>4.1f}" if fcst_c is not None else "  --"
        obs_s = f"{obs_c}" if obs_c is not None else " --"
        print(f"    {valid:%Y-%m-%dT%HZ}   {fcst_s:<8} {obs_s:<7} {err:<8}")


def main() -> None:
    lat, lon = awc.station_latlon(STATION)
    print(f"{STATION} -> lat {lat:.4f}, lon {lon:.4f}")
    cap1_multi_model_state(lat, lon)
    cap2_cross_model_hazard(lat, lon)
    cap3_model_vs_obs(lat, lon)
    print(f"\ntotal credits billed this run: {_charged}  "
          f"(re-run within the same hour -> 0, served from data/gribstream/)")


if __name__ == "__main__":
    main()
