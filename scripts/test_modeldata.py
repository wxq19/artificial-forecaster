"""Model-data subsystem self-test. No model, no network: exercises the gribstream client
math + body guards, the model_data archive (schema/insert/readers/copy), and the four
read-tool formatters against a temp DB seeded with canned archive rows + obs.

Covers: credit estimate (member excluded, multi-coord ceil(/500)) + charged-on-cache;
fetch_points/fetch_timeseries body guards (<=500, empty coords, no-time); insert idempotency
+ immutability (DO NOTHING); model_data_series window/variable filter; model_data_field
latest-run-per-loc; copy_model_data coord filter; read-only rejects writes; and each tool's
receipt shape (state table + cross-model line, hazard scan icing/turbulence, verification
bias vs seeded obs, nearby-field spatial slice).

Run: uv run python scripts/test_modeldata.py
"""

import math
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from forecaster import gribstream, modeldata, store, tools
from forecaster.metar import parse as metar_parse

TMP = tempfile.mkdtemp(prefix="modeldata_test_")
DB = str(Path(TMP) / "bench.duckdb")
RUN = datetime(2026, 7, 17, 18)          # the model cycle
AS_OF = datetime(2026, 7, 17, 18)
FETCHED = datetime(2026, 7, 17, 18, 5)
LAT, LON = 44.8852, -93.2313             # KTEST site
NB_LAT, NB_LON = 45.0, -93.0             # a neighbor point

checks: list[tuple[str, bool, str]] = []


def check(label, cond, detail=""):
    checks.append((label, bool(cond), "" if cond else f"      {detail}"))


def md_row(model, valid, lat, lon, loc, var, val, run=RUN):
    return {"model": model, "run": run, "valid_time": valid, "lat": lat, "lon": lon,
            "loc_id": loc, "variable": var, "value": val, "member": 0,
            "as_of": AS_OF, "fetched_at": FETCHED}


# --- 1. gribstream client: credit math + body guards (no network) ---------------------
def test_client():
    cols = ["forecasted_at", "forecasted_time", "lat", "lon", "name", "member", "t2m", "td2m"]
    rows = [{"forecasted_time": datetime(2026, 7, 17, h), "lat": la, "lon": -93.0,
             "name": "X", "t2m": 290.0, "td2m": 280.0}
            for h in (18, 19, 20) for la in (44.0, 45.0)]   # 3 times x 2 coords, 2 vars
    ts = gribstream.TimeSeries("gfs", "u", cols, rows)
    check("credits = times*vars*ceil(coords/500)", ts.credits == 3 * 2 * 1, f"got {ts.credits}")
    check("member column not counted as a variable", "member" in cols and ts.credits == 6)
    check("charged == credits when not cached", ts.charged == ts.credits)
    ts.cached = True
    check("charged == 0 on cache hit", ts.charged == 0)
    check("empty rows -> 0 credits", gribstream.TimeSeries("gfs", "u", cols, []).credits == 0)
    # Ensemble: members multiply the bill LINEARLY (verified 2026-07-23, 31 members -> 31 cr).
    ecols = ["forecasted_at", "forecasted_time", "lat", "lon", "name", "member", "t2m"]
    ens = [{"forecasted_time": datetime(2026, 7, 17, 18), "lat": 44.0, "lon": -93.0,
            "name": "X", "member": m, "t2m": 290.0} for m in range(5)]   # 1 time x 1 var x 5 mem
    check("credits multiply by member count",
          gribstream.TimeSeries("gefsatmos", "u", ecols, ens).credits == 5,
          f"got {gribstream.TimeSeries('gefsatmos', 'u', ecols, ens).credits}")

    V = [gribstream.Var("TMP", "2 m above ground", "t2m")]
    for label, fn in [
        (">500 coords raises", lambda: gribstream.fetch_points(
            "gfs", [(0.0, 0.0, str(i)) for i in range(501)], V, times=[RUN])),
        ("empty coords raises", lambda: gribstream.fetch_points("gfs", [], V, times=[RUN])),
        ("no variables raises", lambda: gribstream.fetch_points("gfs", [(0.0, 0.0, "X")], [], times=[RUN])),
        ("unknown model raises", lambda: gribstream.fetch_points("zzz", [(0.0, 0.0, "X")], V, times=[RUN])),
        ("no time selection raises", lambda: gribstream.fetch_points("gfs", [(0.0, 0.0, "X")], V)),
    ]:
        try:
            fn()
            check(label, False, "no ValueError raised")
        except ValueError:
            check(label, True)


# --- 2. archive: schema / insert idempotency+immutability / readers / copy ------------
def test_archive():
    con = store.connect(DB)
    store.init_model_data_schema(con)
    rows = [md_row("gfs", datetime(2026, 7, 17, 18 + h), LAT, LON, "KTEST", var, 290.0 + h)
            for h in range(3) for var in ("t2m", "td2m")]
    added = store.insert_model_data(con, rows)
    check("insert added all rows", added == 6, f"added {added}")
    check("re-insert is idempotent (0 added)", store.insert_model_data(con, rows) == 0)
    # immutability: same PK, different value -> DO NOTHING, original value kept
    store.insert_model_data(con, [md_row("gfs", datetime(2026, 7, 17, 18), LAT, LON, "KTEST", "t2m", 999.0)])
    got = con.execute("SELECT value FROM model_data WHERE variable='t2m' AND valid_time=?",
                      [datetime(2026, 7, 17, 18)]).fetchone()[0]
    check("immutable: value unchanged on PK conflict", got == 290.0, f"got {got}")

    series = store.model_data_series(con, "gfs", LAT, LON,
                                     start=datetime(2026, 7, 17), end=datetime(2026, 7, 18))
    check("series round-trips all rows", len(series) == 6, f"got {len(series)}")
    filt = store.model_data_series(con, "gfs", LAT, LON, start=datetime(2026, 7, 17),
                                   end=datetime(2026, 7, 18), variables=["t2m"])
    check("series variable filter", len(filt) == 3 and all(r["variable"] == "t2m" for r in filt))
    check("series matches by rounded lat/lon equality",
          len(store.model_data_series(con, "gfs", round(LAT, 4), round(LON, 4),
                                      start=datetime(2026, 7, 17), end=datetime(2026, 7, 18))) == 6)

    # field slice across locations (latest run per loc)
    store.insert_model_data(con, [
        md_row("gfs", datetime(2026, 7, 17, 18), NB_LAT, NB_LON, "KNB", "t2m", 285.0),
        md_row("gfs", datetime(2026, 7, 17, 18), NB_LAT, NB_LON, "KNB", "t2m", 286.0,
               run=datetime(2026, 7, 17, 12)),   # older run, same PK-except-run
    ])
    field = store.model_data_field(con, "gfs", "t2m", valid_time=datetime(2026, 7, 17, 18))
    check("field: one row per location", len(field) == 2, f"got {len(field)}")
    knb = next(r for r in field if r["loc_id"] == "KNB")
    check("field: latest run wins per loc", knb["value"] == 285.0 and knb["run"] == RUN)

    vts = store.model_data_valid_times(con, "gfs", LAT, LON)
    check("valid_times distinct + ascending", vts == sorted(set(vts)) and len(vts) == 3)

    locs = store.model_data_locations(con)
    check("locations lists both points", {lc["loc_id"] for lc in locs} == {"KTEST", "KNB"})
    con.close()

    # copy into a scratch per-run DB, filtered to one coordinate
    scratch = str(Path(TMP) / "run.duckdb")
    scon = store.connect(scratch)
    n = store.copy_model_data(scon, DB, coords=[(LAT, LON, "KTEST")])
    check("copy_model_data copies only the filtered coord", n == 6, f"copied {n}")
    check("copy excluded the other location",
          {lc["loc_id"] for lc in store.model_data_locations(scon)} == {"KTEST"})
    scon.close()

    ro = store.connect(DB, read_only=True)
    try:
        ro.execute("INSERT INTO model_data VALUES ('gfs',?,?,0,0,'X','t2m',1,0,?,?)",
                   [RUN, RUN, AS_OF, FETCHED])
        check("read-only rejects writes", False, "insert succeeded on read-only conn")
    except Exception:
        check("read-only rejects writes", True)
    finally:
        ro.close()


# --- 3. formatters via run_tool against the seeded DB ---------------------------------
def seed_state_db(path):
    con = store.connect(path)
    store.init_model_data_schema(con)
    store.init_schema(con)
    rows = []
    for h in range(0, 6, 2):
        vt = datetime(2026, 7, 17, 18 + h)
        # GFS surface at the site
        for var, val in [("t2m", 300.0 + h), ("td2m", 288.0), ("u10", -5.0), ("v10", 3.0),
                         ("gust", 12.0 + h), ("mslp", 101300.0), ("tcdc", 40.0),
                         ("vis", 16000.0), ("ceil", 20000.0)]:
            rows.append(md_row("gfs", vt, LAT, LON, "KTEST", var, val))
        # NBM surface (speed/dir wind, no mslp)
        for var, val in [("t2m", 301.0), ("td2m", 288.0), ("wind", 6.0), ("wdir", 210.0),
                         ("gust", 13.0), ("tcdc", 35.0), ("vis", 16000.0), ("ceil", 20000.0)]:
            rows.append(md_row("nbm", vt, LAT, LON, "KTEST", var, val))
        # a neighbor point (for get_nearby_model_data)
        rows.append(md_row("gfs", vt, NB_LAT, NB_LON, "KNB", "t2m", 296.0 + h))
    # hazard vars at the site (GFS + HRRR), valid 18Z
    hz = datetime(2026, 7, 17, 18)
    for var, val in [("t500", 263.15), ("rh500", 85.0), ("clw500", 0.0003), ("cape", 1200.0),
                     ("cin", -20.0), ("u850", 5.0), ("v850", 2.0), ("u300", 45.0), ("v300", 10.0),
                     ("w500", -3.0), ("hlcy", 220.0)]:
        rows.append(md_row("gfs", hz, LAT, LON, "KTEST", var, val))
    for var, val in [("t500", 264.15), ("rh500", 80.0), ("cape", 900.0), ("cin", -30.0),
                     ("u850", 6.0), ("v850", 1.0), ("u300", 44.0), ("v300", 12.0), ("w500", -2.0)]:
        rows.append(md_row("hrrr", hz, LAT, LON, "KTEST", var, val))
    # An OLDER run covering the SAME 18Z valid time, forecasting worse. This is the case the
    # verification table exists for: one hour, several runs, so "was the fresher run closer?"
    # is answerable. GRIBStream only returns one run per valid time per request, so the
    # archive only ever holds this shape when prefetch_verification pins asOf per run.
    older = RUN - timedelta(hours=6)
    for var, val in [("t2m", 303.0), ("td2m", 286.0), ("u10", -9.0), ("v10", 1.0),
                     ("gust", 25.0), ("mslp", 101100.0)]:
        rows.append(md_row("gfs", datetime(2026, 7, 17, 18), LAT, LON, "KTEST", var, val,
                           run=older))
    store.insert_model_data(con, rows)
    # obs at 18Z for verification (temp 25C so model 300K=26.85C -> +~1.9 bias)
    o = metar_parse("KTEST 171800Z 21008KT 10SM CLR 25/15 A2992")
    store.insert_obs(con, [o], year=2026, month=7, source="test")
    con.close()


def seed_ensemble_db(path):
    """A 10-member GEFS ensemble at one valid time with a KNOWN spread, so the probability
    numbers are checkable: 4/10 ceilings below 200 ft (cat A = 40%), 8/10 gusts >= 15 kt."""
    con = store.connect(path)
    store.init_model_data_schema(con)
    vt = datetime(2026, 7, 23, 18)
    ceil_m = [30, 40, 50, 55, 20000, 20000, 20000, 20000, 20000, 20000]   # 4 low, 6 unlimited
    gust_ms = [5, 8, 10, 12, 13, 14, 8, 9, 20, 6]                         # *1.944 kt; 8 >= 15kt
    rows = []
    for mem in range(10):
        for var, val in [("ceil", ceil_m[mem]), ("vis", 16000.0), ("gust", gust_ms[mem]),
                         ("u10", 3.0), ("v10", 4.0), ("t2m", 293.0 + mem * 0.3), ("td2m", 288.0)]:
            rows.append({"model": "gefsatmos", "run": RUN, "valid_time": vt, "lat": LAT, "lon": LON,
                         "loc_id": "KTEST", "variable": var, "value": val, "member": mem,
                         "as_of": AS_OF, "fetched_at": FETCHED})
    store.insert_model_data(con, rows)
    con.close()


def test_formatters():
    path = str(Path(TMP) / "state.duckdb")
    seed_state_db(path)

    r = tools.run_tool("get_model_state", {"station": "KTEST"}, db_path=path)
    check("get_model_state has GFS + NBM tables",
          "GFS surface forecast" in r.text and "NBM surface forecast" in r.text, r.text[:120])
    check("get_model_state cross-model synopsis", "CROSS-MODEL" in r.text and "peak gust" in r.text)
    check("get_model_state converts K->C (temp ~27)", " 27" in r.text or " 28" in r.text)

    r = tools.run_tool("get_model_state", {"station": "KTEST", "model": "gfs"}, db_path=path)
    check("get_model_state single-model", "GFS surface" in r.text and "NBM surface" not in r.text)

    r = tools.run_tool("get_hazard_scan", {"station": "KTEST"}, db_path=path)
    check("get_hazard_scan reports icing block", "ICING" in r.text and "500 mb" in r.text)
    check("get_hazard_scan reports turbulence + agreement",
          "TURBULENCE" in r.text and "agreement:" in r.text, r.text[-200:])
    check("get_hazard_scan diagnoses convective (CAPE both >500)",
          "convective" in r.text, r.text[-200:])

    r = tools.run_tool("get_model_verification", {"station": "KTEST", "model": "gfs"}, db_path=path)
    check("get_model_verification matches the seeded ob + bias",
          "+1.9" in r.text and "TEMPERATURE (C)" in r.text, r.text[:600])
    check("get_model_verification renders a block per field",
          all(f in r.text for f in ("TEMPERATURE (C)", "DEWPOINT (C)", "ALTIMETER / MSLP (inHg)",
                                    "WIND DIRECTION (deg)", "WIND SPEED (kt)",
                                    "WIND GUST (kt)")), r.text[:300])
    check("get_model_verification puts each RUN in its own column",
          "18Z run" in r.text and "12Z run" in r.text, r.text[:900])
    # The point of the layout: the same hour, two runs, on one line.
    trow = next((ln for ln in r.text.splitlines() if ln.strip().startswith("17/18Z")), "")
    check("get_model_verification shows one hour across several runs on ONE row",
          trow.count("(") >= 2, trow)
    check("get_model_verification separates mean bias from typical error size",
          "mean err" in r.text and "typical" in r.text and "cancelling" in r.text, r.text[:900])

    r = tools.run_tool("get_nearby_model_data", {"station": "KTEST", "variable": "t2m", "model": "gfs"},
                       db_path=path)
    check("get_nearby_model_data lists both points",
          "KTEST" in r.text and "KNB" in r.text, r.text)

    seed_ensemble_db(path + ".ens")
    r = tools.run_tool("get_ensemble_prob", {"station": "KTEST"}, db_path=path + ".ens")
    check("get_ensemble_prob renders category + exceedance + percentile blocks",
          all(s in r.text for s in ("CEILING", "VISIBILITY", "WIND SPEED", "WIND GUST",
                                    "TEMPERATURE / DEWPOINT")), r.text[:300])
    # 4 of 10 members below 200 ft -> ceiling cat A = 40%; 8 of 10 gust >= 15 kt.
    cig = next((ln for ln in r.text.splitlines() if ln.strip().startswith("23/18Z")), "")
    check("get_ensemble_prob ceiling category probability (40% cat A)", " 40" in cig, cig)
    grow = [ln for ln in r.text.splitlines() if ln.strip().startswith("23/18Z")]
    check("get_ensemble_prob gust exceedance present (80% >= 15kt)",
          any(" 80" in ln for ln in grow), grow)
    check("get_ensemble_prob has no data for a non-ensemble station -> feedback",
          "no GEFS ensemble" in tools.run_tool("get_ensemble_prob", {"station": "KTEST"},
                                                db_path=path).text)
    check("get_nearby_model_data converts t2m to C", "(C)" in r.text)

    # not-found feedback (not a crash)
    r = tools.run_tool("get_model_state", {"station": "KZZZ"}, db_path=path)
    check("unknown location -> feedback not crash",
          "not a pre-fetched" in r.text or "no model data" in r.text, r.text)


# --- 4. collect.py data path: benchmark archive -> copy_model_data -> per-run DB -> tools -
def test_collect_path():
    """Mirror what collect.py does with --model-data: prefetch writes the benchmark DB, then
    the per-run DB copies only the station's coordinate neighborhood, and the model cells read
    the tools off THAT DB. Proves the data survives the copy (no LLM needed)."""
    bench = str(Path(TMP) / "bench_collect.duckdb")
    seed_state_db(bench)                      # stands in for a prefetch-populated benchmark DB
    run_db = str(Path(TMP) / "percell.duckdb")
    rcon = store.connect(run_db)
    # exactly the shape collect.py uses: copy the station + neighbor coords
    n = store.copy_model_data(rcon, bench, coords=[(LAT, LON, "KTEST"), (NB_LAT, NB_LON, "KNB")])
    rcon.close()
    check("collect copy moved rows into the per-run DB", n > 0, f"copied {n}")

    r = tools.run_tool("get_model_state", {"station": "KTEST"}, db_path=run_db)
    check("per-run DB: get_model_state renders after copy",
          "GFS surface forecast" in r.text and "CROSS-MODEL" in r.text, r.text[:120])
    r = tools.run_tool("get_nearby_model_data", {"station": "KTEST", "variable": "t2m"}, db_path=run_db)
    check("per-run DB: get_nearby_model_data has both copied points",
          "KTEST" in r.text and "KNB" in r.text)
    r = tools.run_tool("get_hazard_scan", {"station": "KTEST"}, db_path=run_db)
    check("per-run DB: get_hazard_scan renders after copy", "ICING" in r.text)

    # --- LEAKAGE: a run issued AFTER the issue time must never reach the per-run DB -------
    # The archiver accumulates every pull, so the benchmark DB holds many runs at once. The
    # old copy took them all, on the retired assumption that a prefetch pinned to the issue
    # time could only contain runs <= issue time. It cannot hold any more, and the tools now
    # PREFER the newest run, so an uncut copy would serve tomorrow's guidance to today's TAF.
    future_run = RUN + timedelta(hours=12)
    fcon = store.connect(bench)
    store.insert_model_data(fcon, [
        md_row("gfs", RUN + timedelta(hours=h), LAT, LON, "KTEST", "t2m", 310.0,
               run=future_run) for h in range(0, 9, 3)])
    fcon.close()

    cut_db = str(Path(TMP) / "percell_cutoff.duckdb")
    ccon = store.connect(cut_db)
    store.copy_model_data(ccon, bench, coords=[(LAT, LON, "KTEST")], run_at_or_before=RUN)
    runs = {r[0] for r in ccon.execute("SELECT DISTINCT run FROM model_data").fetchall()}
    ccon.close()
    check("cutoff: no run later than the issue time is copied",
          all(x <= RUN for x in runs), f"copied runs {sorted(runs)}")
    check("cutoff: the legitimate run still survives the filter", RUN in runs, f"{sorted(runs)}")

    # And prove the guard is not vacuous: without the cutoff the future run DOES come across,
    # so a passing test above is the filter working rather than the row never existing.
    open_db = str(Path(TMP) / "percell_nocutoff.duckdb")
    ocon = store.connect(open_db)
    store.copy_model_data(ocon, bench, coords=[(LAT, LON, "KTEST")])
    oruns = {r[0] for r in ocon.execute("SELECT DISTINCT run FROM model_data").fetchall()}
    ocon.close()
    check("cutoff: without it the future run really is copied (guard is not vacuous)",
          future_run in oruns, f"copied runs {sorted(oruns)}")

    # The tool layer is what the agent sees: with the cutoff it must read the ISSUE-time run,
    # not the future one. 310 K would be the future row's value.
    r = tools.run_tool("get_point_forecast", {"station": "KTEST"}, db_path=cut_db)
    check("cutoff: get_point_forecast names the issue-time run, not a later one",
          f"{RUN:%Y-%m-%dT%H}Z" in r.text and f"{future_run:%Y-%m-%dT%H}Z" not in r.text,
          r.text.splitlines()[0] if r.text else "")

    # a benchmark DB with NO model_data (tier OFF): copy is a clean 0, tools give feedback
    empty = str(Path(TMP) / "bench_empty.duckdb")
    econ = store.connect(empty)
    store.init_schema(econ)
    econ.close()
    ecell = str(Path(TMP) / "percell_empty.duckdb")
    ercon = store.connect(ecell)
    n0 = store.copy_model_data(ercon, empty)          # source has no model_data table
    store.init_model_data_schema(ercon)               # collect creates the empty schema when OFF
    ercon.close()
    check("copy from a model_data-less benchmark DB is a clean 0", n0 == 0)
    r = tools.run_tool("get_model_state", {"station": "KTEST"}, db_path=ecell)
    check("empty archive -> feedback, not crash",
          "no model data" in r.text or "not a pre-fetched" in r.text, r.text)


# --- 5. grid density + flow-relative + batched coord assembly + IFS scaffold (no network) -
def test_grid_flow_batch_ifs():
    # denser fixed grid: 12 bearings x 3 radii = 36
    base = modeldata._grid_points(45.0, -93.0)
    check("denser fixed grid is 36 points", len(base) == 36, f"got {len(base)}")
    check("no upstream points without flow", not any(n.startswith("u") for _, _, n in base))
    flow = modeldata._grid_points(45.0, -93.0, flow_from=270)
    up = [(la, lo, n) for la, lo, n in flow if n.startswith("u")]
    check("flow adds 6 upstream points (3 bearings x 2 radii)", len(up) == 6, f"got {len(up)}")
    check("upstream points reach farther west (lon < base ring)",
          all(lo < -93.0 - 1.5 for lo, _, _ in [(lo, la, n) for la, lo, n in up]))

    # flow bearing from climo (DB only, no network)
    con = store.connect(str(Path(TMP) / "flow.duckdb"))
    store.init_climo_schema(con)
    con.execute("INSERT INTO climo_hourly (station, month, hour_utc, dir_mode_sector) VALUES "
                "('KWRI', 7, 18, 'W')")
    con.close()
    fdb = str(Path(TMP) / "flow.duckdb")
    check("_flow_from_climo reads prevailing sector -> bearing",
          modeldata._flow_from_climo("KWRI", 7, 18, fdb) == 270.0)
    check("_flow_from_climo picks nearest hour", modeldata._flow_from_climo("KWRI", 7, 19, fdb) == 270.0)
    check("_flow_from_climo None when month absent", modeldata._flow_from_climo("KWRI", 3, 18, fdb) is None)
    check("_flow_from_climo None when climo missing",
          modeldata._flow_from_climo("KWRI", 7, 18, str(Path(TMP) / "nope.duckdb")) is None)
    from datetime import datetime as _dt
    LA, LO = 40.0, -74.6
    check("_resolve_flow honors flow_relative=False",
          modeldata._resolve_flow("KWRI", LA, LO, _dt(2026, 7, 17, 18), fdb, False) is None)
    check("_resolve_flow falls back to climo when no archived steering",
          modeldata._resolve_flow("KWRI", LA, LO, _dt(2026, 7, 17, 18), fdb, True) == 270.0)

    # steering flow: deep-layer vector-mean wind -> wind-FROM bearing
    check("_steering_bearing westerly (u>0) -> 270",
          modeldata._steering_bearing({850: 10.0, 700: 10.0, 500: 10.0},
                                      {850: 0.0, 700: 0.0, 500: 0.0}) == 270.0)
    check("_steering_bearing southerly (v>0) -> 180",
          modeldata._steering_bearing({850: 0.0, 500: 0.0}, {850: 8.0, 500: 8.0}) == 180.0)
    check("_steering_bearing None when no levels present",
          modeldata._steering_bearing({}, {}) is None)
    # archive reproducibility: seed site GFS deep-layer winds -> the SAME bearing prefetch got
    scon = store.connect(str(Path(TMP) / "steer.duckdb"))
    store.init_model_data_schema(scon)
    anchor = _dt(2026, 7, 17, 18)
    swinds = []
    for lv, uu, vv in [(850, 12.0, 4.0), (700, 14.0, 5.0), (500, 20.0, 6.0)]:  # WSW-ish flow
        swinds.append(md_row("gfs", anchor, LA, LO, "KTEST", f"u{lv}", uu))
        swinds.append(md_row("gfs", anchor, LA, LO, "KTEST", f"v{lv}", vv))
    store.insert_model_data(scon, swinds)
    scon.close()
    sdb = str(Path(TMP) / "steer.duckdb")
    expect = modeldata._steering_bearing({850: 12.0, 700: 14.0, 500: 20.0},
                                         {850: 4.0, 700: 5.0, 500: 6.0})
    check("_steering_from_archive recomputes the deep-layer bearing (copy-reproducible)",
          modeldata._steering_from_archive(LA, LO, anchor, sdb) == expect)
    check("_resolve_flow PREFERS archived steering over climo",
          modeldata._resolve_flow("KTEST", LA, LO, anchor, sdb, True) == expect)

    # batched coord assembly (union + dedupe + chunk) -- pure, no network
    a = [(1.0, 1.0, "A"), (2.0, 2.0, "g1")]
    b = [(2.0, 2.0, "B"), (3.0, 3.0, "g2")]   # (2,2) overlaps a
    u = modeldata._dedupe([a, b])
    check("_dedupe unions and dedups overlapping coords", len(u) == 3, f"got {len(u)}")
    check("_dedupe keeps first name for a shared coord",
          next(n for la, lo, n in u if (la, lo) == (2.0, 2.0)) == "g1")
    big = [(float(i), 0.0, str(i)) for i in range(1100)]
    chunks = list(modeldata._chunk(big))
    check("_chunk splits >500 into <=500 batches", [len(c) for c in chunks] == [500, 500, 100])

    # IFS: enabled 2026-07-23 -- in the default MODELS, verified native names, tcc scaled to
    # percent at ingest, still no hazard bundle, and global (kept OCONUS by _applicable_models)
    check("ifsoper is a VALID model", "ifsoper" in gribstream.VALID_MODELS)
    check("ifsoper is in the default prefetch set", "ifsoper" in modeldata.MODELS)
    ifs = {(v.name, v.level, v.alias) for v in modeldata._surface_vars("ifsoper")}
    check("ifs surface uses ECMWF native names @ sfc",
          ("2t", "sfc", "t2m") in ifs and ("msl", "sfc", "mslp") in ifs and ("tcc", "sfc", "tcdc") in ifs)
    check("ifs has no hazard bundle yet", modeldata._hazard_vars("ifsoper") == [])
    # tcc is a 0-1 fraction in IFS; _normalize scales it to percent so the tcdc alias is
    # consistent with GFS/NBM. Other fields and other models are untouched.
    check("ifs tcc fraction scaled to percent at ingest",
          modeldata._normalize("ifsoper", "tcdc", 0.61) == 61.0)
    check("_normalize leaves non-tcdc IFS fields alone",
          modeldata._normalize("ifsoper", "t2m", 293.0) == 293.0)
    check("_normalize leaves other models' tcdc alone",
          modeldata._normalize("gfs", "tcdc", 61.0) == 61.0)
    # IFS is global, so it must survive the OCONUS drop that removes HRRR/NBM.
    keep, drop = modeldata._applicable_models([(64.8, -147.9, "PABI")], modeldata.MODELS)
    check("ifs kept OCONUS while hrrr/nbm dropped",
          "ifsoper" in keep and "gfs" in keep and "hrrr" in drop and "nbm" in drop)
    try:
        gribstream.fetch_points("ifsoper", [(1.0, 1.0, "X")], [gribstream.Var("2t", "sfc", "t2m")])
        check("ifsoper accepted as a model (raises on missing time, not model)", False)
    except ValueError as e:
        check("ifsoper accepted as a model (raises on missing time, not model)",
              "time" in str(e).lower() or "either" in str(e).lower())


# --- 6. vertical profile bundle (the BUFKIT replacement) ------------------------------
def test_profiles():
    # Level coverage, PROBED LIVE 2026-07-28: GFS/HRRR serve all 20 standard levels, IFS a
    # 12-level subset missing 950/900 (the boundary layer), NBM has no pressure levels.
    check("gfs/hrrr serve 20 profile levels",
          len(modeldata.profile_levels("gfs")) == 20 and len(modeldata.profile_levels("hrrr")) == 20)
    check("ifs serves the 12-level subset, without 950/900",
          len(modeldata.profile_levels("ifsoper")) == 12
          and 950 not in modeldata.profile_levels("ifsoper")
          and 900 not in modeldata.profile_levels("ifsoper"))
    check("nbm has no profile (surface-only)", modeldata.profile_levels("nbm") == ())

    # Five variables per level is the cost driver; the model dialects differ.
    gv, iv = modeldata._profile_vars("gfs"), modeldata._profile_vars("ifsoper")
    check("gfs profile = 20 levels x 5 vars", len(gv) == 100, f"got {len(gv)}")
    check("ifs profile = 12 levels x 5 vars", len(iv) == 60, f"got {len(iv)}")
    check("gfs uses GRIB2 names + '<n> mb' levels",
          gv[0].name == "TMP" and gv[0].level == "1000 mb" and gv[0].alias == "t1000")
    check("ifs uses ECMWF shortnames + 'pl <n>' levels",
          iv[0].name == "t" and iv[0].level == "pl 1000" and iv[0].alias == "t1000")
    check("nbm contributes no profile vars", modeldata._profile_vars("nbm") == [])

    # ONE alias namespace: the hazard reader's aliases must be a SUBSET of the profile's, or
    # merging the two bundles would silently break get_hazard_scan.
    prof_aliases = {v.alias for v in gv}
    haz_standalone = {v.alias for v in modeldata._hazard_vars("gfs", profiles=False)}
    haz_merged = {v.alias for v in modeldata._hazard_vars("gfs", profiles=True)}
    check("merged hazard bundle drops exactly what the profile already covers",
          haz_standalone - haz_merged <= prof_aliases and not (haz_merged & prof_aliases),
          f"dropped {sorted(haz_standalone - haz_merged)}")
    check("merged hazard keeps cloud-liquid/omega/CAPE (not in the profile)",
          {"clw500", "w500", "cape", "cin"} <= haz_merged)
    check("1000 mb alias does not collide with 100 mb",
          "t1000" in prof_aliases and "t100" in prof_aliases)

    # The level grid must land on a 00Z-anchored 3-hourly grid whatever the issue hour, or
    # IFS (00/03/06...Z only) silently archives nothing -- no rows, no credits, no error.
    for hr in (0, 1, 2, 5, 17, 23):
        a = datetime(2026, 7, 28, hr)
        la = a.replace(hour=(a.hour // 3) * 3)
        span = math.ceil((30 + (a.hour - la.hour)) / 3) * 3
        grid = modeldata._time_grid(la, span, 3)
        ok = all(x.hour % 3 == 0 for x in grid) and grid[-1] >= a + timedelta(hours=30)
        check(f"level grid snaps to the 3h/00Z grid and still covers +30h (issue {hr:02d}Z)", ok)

    # build_profile: archive rows -> the typed profile charts.skewt draws.
    con = store.connect(DB)
    store.init_model_data_schema(con)
    valid = datetime(2026, 7, 28, 12)
    rows = []
    for i, hpa in enumerate(modeldata.profile_levels("gfs")):
        k = str(hpa)
        rows += [md_row("gfs", valid, LAT, LON, "KTEST", f"t{k}", 300.0 - 2.0 * i),
                 md_row("gfs", valid, LAT, LON, "KTEST", f"rh{k}", 60.0),
                 md_row("gfs", valid, LAT, LON, "KTEST", f"u{k}", 10.0),
                 md_row("gfs", valid, LAT, LON, "KTEST", f"v{k}", 0.0),
                 md_row("gfs", valid, LAT, LON, "KTEST", f"hgt{k}", 100.0 * i)]
    store.insert_model_data(con, rows)
    prof = modeldata.build_profile(con, "KTEST", "gfs", valid, lat=LAT, lon=LON)
    check("build_profile returns every complete level", len(prof.pres) == 20, f"got {len(prof.pres)}")
    check("build_profile is surface-first", prof.pres[0] == 1000.0 and prof.pres[-1] == 100.0)
    check("build_profile converts K -> C", abs(prof.tmpc[0] - 26.85) < 0.01, f"{prof.tmpc[0]}")
    check("build_profile derives Td <= T at every level",
          all(d <= t + 1e-9 for t, d in zip(prof.tmpc, prof.dwpc)))
    check("build_profile converts u/v -> wind FROM direction + knots",
          abs(prof.drct[0] - 270.0) < 0.01 and abs(prof.sknt[0] - 19.44) < 0.05,
          f"{prof.drct[0]}/{prof.sknt[0]}")
    check("build_profile records the run it used", prof.run == RUN)

    # A level missing ANY field is dropped whole rather than plotted half-formed.
    con.execute("DELETE FROM model_data WHERE variable = 'hgt850' AND valid_time = ?", [valid])
    check("a partial level is dropped, not half-plotted",
          len(modeldata.build_profile(con, "KTEST", "gfs", valid, lat=LAT, lon=LON).pres) == 19)

    check("profile_valid_times reports the archived hour",
          modeldata.profile_valid_times(con, "KTEST", "gfs", lat=LAT, lon=LON) == [valid])
    check("store.model_data_as_of returns the pinned issue time",
          store.model_data_as_of(con, "gfs", LAT, LON) == AS_OF)
    try:
        modeldata.build_profile(con, "KTEST", "nbm", valid, lat=LAT, lon=LON)
        check("build_profile refuses a model with no levels", False)
    except ValueError as e:
        check("build_profile refuses a model with no levels", "no vertical profile" in str(e))
    try:
        modeldata.build_profile(con, "KTEST", "gfs", datetime(2026, 7, 28, 21),
                                lat=LAT, lon=LON)
        check("build_profile reports an unarchived hour as feedback", False)
    except ValueError as e:
        check("build_profile reports an unarchived hour as feedback", "no archived" in str(e))
    con.close()


def test_conus_only_coords():
    """A CONUS-only model must be billed over CONUS coordinates ONLY.

    Asked about Ramstein, HRRR does not error -- it returns a full set of NULL-valued rows that
    we pay for and store. Measured live 2026-07-29: every OCONUS station held 5,715 HRRR rows
    and 624 NBM rows at 0% non-null, 844,422 + 612,096 null rows archive-wide, ~952 credits a
    pull. `_applicable_models` was meant to catch this but only fires when NO coordinate is in
    CONUS, and the real batch unions all 71 stations, so it never fired."""
    mixed = [(40.0, -75.0, "KAAA"),        # CONUS
             (49.4, 7.6, "ETAR"),          # Germany
             (35.7, 139.3, "RJTY"),        # Japan
             (-53.0, -70.8, "SCCI")]       # Patagonia
    for m in ("hrrr", "nbm"):
        got = modeldata.model_coords(m, mixed)
        check(f"{m} is billed over CONUS coords only",
              [c[2] for c in got] == ["KAAA"], f"got {[c[2] for c in got]}")
    for m in ("gfs", "ifsoper"):
        check(f"{m} is global and keeps every coord",
              len(modeldata.model_coords(m, mixed)) == 4,
              f"got {len(modeldata.model_coords(m, mixed))}")
    # An all-OCONUS batch leaves a CONUS-only model with nothing -- it must be skipped, not
    # sent an empty request.
    oconus = [c for c in mixed if c[2] != "KAAA"]
    check("a CONUS-only model gets NO coords in an all-OCONUS batch",
          modeldata.model_coords("hrrr", oconus) == [], "got rows")


def test_credit_estimate():
    """The estimator's whole job is not drifting from the fetch, so pin the four ways it
    drifted before: the profile ladder, the pre-anchor tail, the merged-hazard subtraction,
    and CONUS-only model dropping. Offline -- coords are stubbed, no AWC lookups."""
    site = {"KAAA": (40.0, -75.0), "KBBB": (41.0, -76.0)}
    saved = modeldata.site_coord
    modeldata.site_coord = lambda s: (*site[s], s)
    try:
        e = modeldata.estimate_prefetch_many(
            ["KAAA"], as_of=datetime(2026, 7, 28, 11), hours=48, step_h=2, hazard_step_h=3,
            flow_relative=False)

        # (a) The grid is RUN-ANCHORED at the model's own cadence, so `surface_times` is now
        # per model rather than one shared number. GFS is hourly (step_h is a CEILING, and
        # NATIVE_STEP_H['gfs'] is 1), IFS is 3-hourly and NOTHING ELSE -- asking it for the
        # global step is what used to buy 17 rows and pay for 28.
        cyc = modeldata.archive_cycle(datetime(2026, 7, 28, 11))
        check("estimate: the cycle is a 00/06/12/18Z run",
              cyc.hour % 6 == 0 and cyc.minute == 0, f"got {cyc}")
        # This call passes step_h=2, and a COARSER request is honoured: the effective step is
        # max(requested, native), i.e. never finer than the model serves and never finer than
        # asked for. So GFS here is 2-hourly, f000..f048.
        check("estimate: run-anchored grid is f000..f048 at the requested step",
              e["per_model"]["gfs"]["sfc_times"] == 48 // 2 + 1,
              f"got {e['per_model']['gfs']['sfc_times']}")
        # ...but IFS cannot go finer than 3-hourly whatever is asked, which is the waste this
        # replaced: 28 two-hourly times requested, 17 rows returned, 28 paid for.
        if "ifsoper" in e["per_model"]:
            check("estimate: IFS stays 3-hourly even when a finer step is requested",
                  e["per_model"]["ifsoper"]["sfc_times"] == 48 // 3 + 1,
                  f"got {e['per_model']['ifsoper']['sfc_times']}")
        hourly = modeldata.estimate_prefetch_many(
            ["KAAA"], as_of=datetime(2026, 7, 28, 11), hours=48, step_h=1, hazard_step_h=3,
            flow_relative=False)
        check("estimate: step_h=1 gets GFS its native HOURLY ladder",
              hourly["per_model"]["gfs"]["sfc_times"] == 49,
              f"got {hourly['per_model']['gfs']['sfc_times']}")
        check("estimate: step_h=1 does NOT make IFS hourly",
              hourly["per_model"].get("ifsoper", {}).get("sfc_times", 17) == 17,
              f"got {hourly['per_model'].get('ifsoper')}")
        # No pre-anchor tail is needed any more: a run-anchored ladder starts at f000, which is
        # already past by the time the cron fires, so the verification hours arrive anyway --
        # and from the correct run rather than as an older run's 9-row stub.
        s, _h = modeldata._model_times("gfs", datetime(2026, 7, 28, 11), 48, 1, 3, 6, run=cyc)
        check("estimate: the run-anchored grid needs no separate back_h tail",
              s[0] == cyc, f"got {s[0]} want {cyc}")
        # (b) the profile ladder dominates and must be present for every profile model.
        for m in modeldata.PROFILE_MODELS:
            if m in e["per_model"]:
                check(f"estimate includes the {m} profile ladder",
                      e["per_model"][m]["levels"] > 0 and e["per_model"][m]["level_vars"] >= 5 * len(
                          modeldata.profile_levels(m)),
                      f"got {e['per_model'][m]}")
        # (c) merged bundle: level_vars must equal what _fetch_and_insert would build.
        want = len(modeldata._profile_vars("gfs")) + len(modeldata._hazard_vars("gfs", profiles=True))
        check("estimate uses the MERGED level bundle (no double-counted T/RH)",
              e["per_model"]["gfs"]["level_vars"] == want,
              f"got {e['per_model']['gfs']['level_vars']} want {want}")
        # (d) credits are the sum of the parts, and each part is times x vars x chunks -- with
        # times and coords BOTH taken per model now.
        gfs_sfc = e["per_model"]["gfs"]["sfc_times"] * len(modeldata._surface_vars("gfs"))
        check("estimate surface term is times x vars x chunks",
              e["per_model"]["gfs"]["surface"] == gfs_sfc,
              f"got {e['per_model']['gfs']['surface']} want {gfs_sfc}")
        check("estimate total is the sum of its parts",
              e["credits"] == sum(m["surface"] + m["levels"] for m in e["per_model"].values())
              + e["steering_probe"])

        # (e) CONUS-only models are dropped for an OCONUS batch (they bill all-null rows).
        modeldata.site_coord = lambda s: (35.75, 139.35, s)      # Yokota
        o = modeldata.estimate_prefetch_many(["KAAA"], as_of=datetime(2026, 7, 28, 11),
                                             flow_relative=False)
        check("estimate drops CONUS-only models OCONUS",
              "hrrr" not in o["models"] and "nbm" not in o["models"], f"got {o['models']}")
        check("estimate says WHY a model was dropped", any("CONUS" in n for n in o["notes"]))

        # (f) points are free below 500, so credits must not scale with station count.
        modeldata.site_coord = lambda s: (*site[s], s)
        one = modeldata.estimate_prefetch_many(["KAAA"], as_of=datetime(2026, 7, 28, 11),
                                               flow_relative=False)
        two = modeldata.estimate_prefetch_many(["KAAA", "KBBB"], as_of=datetime(2026, 7, 28, 11),
                                               flow_relative=False)
        check("estimate is flat while coords fit one 500-point chunk",
              one["credits"] == two["credits"] and two["coords"] > one["coords"],
              f"{one['credits']} vs {two['credits']}, coords {one['coords']}->{two['coords']}")
        # (g) the steering probe is billed per station when flow-relative is on.
        fr = modeldata.estimate_prefetch_many(["KAAA", "KBBB"], as_of=datetime(2026, 7, 28, 11),
                                              flow_relative=True)
        check("estimate bills one steering probe per station",
              fr["steering_probe"] == 2 * len(modeldata._steer_vars()),
              f"got {fr['steering_probe']}")
    finally:
        modeldata.site_coord = saved


def test_ensemble_batch():
    """The GEFS grid snap (a silent-empty bug that has bitten twice) and the batched pull."""
    # (a) an ODD anchor must snap DOWN onto the model's 00Z-anchored cadence -- an unsnapped
    #     22Z grid gives 22/01/04Z, matches no GEFS time, and returns nothing with no error.
    g = modeldata._snapped_grid(datetime(2026, 7, 17, 22), 30, 3)
    check("snapped grid lands on the 00Z-anchored step", all(t.hour % 3 == 0 for t in g),
          f"got {[t.hour for t in g[:5]]}")
    check("snapped grid starts at or before the anchor", g[0] <= datetime(2026, 7, 17, 22))
    check("snapped grid still reaches anchor+hours",
          g[-1] >= datetime(2026, 7, 17, 22) + timedelta(hours=30), f"ends {g[-1]}")
    check("snapped grid is a no-op on an already-aligned anchor",
          modeldata._snapped_grid(datetime(2026, 7, 17, 21), 30, 3)[0]
          == datetime(2026, 7, 17, 21))

    # (b) the ensemble bill is flat in station count -- one point per station, free below 500.
    one = modeldata.estimate_ensemble(["KAAA"], hours=48)
    many = modeldata.estimate_ensemble([f"K{i:03d}" for i in range(40)], hours=48)
    check("ensemble estimate is flat in station count",
          one["credits"] == many["credits"], f"{one['credits']} vs {many['credits']}")
    full = modeldata.estimate_ensemble(["KAAA"], hours=48,
                                       members=modeldata.GEFS_FULL_MEMBERS)
    check("ensemble estimate scales LINEARLY with members",
          full["credits"] == one["credits"] * 31 // len(modeldata.GEFS_DEFAULT_MEMBERS),
          f"{full['credits']} vs {one['credits']}")
    check("full member set is all 31", len(modeldata.GEFS_FULL_MEMBERS) == 31)

    # (c) the single-station form must DELEGATE, so the grid fix cannot regress on one path.
    seen = {}
    saved = modeldata.prefetch_ensemble_many

    def fake(stations, **kw):
        seen.update(stations=stations, **kw)
        return {"stations": [s.upper() for s in stations], "credits_charged": 7,
                "rows": 1, "members": [0], "notes": []}

    modeldata.prefetch_ensemble_many = fake
    try:
        r = modeldata.prefetch_ensemble("kaaa", hours=48, step_h=3)
    finally:
        modeldata.prefetch_ensemble_many = saved
    check("prefetch_ensemble delegates to the batched form", seen.get("stations") == ["kaaa"])
    check("prefetch_ensemble keeps its single-station return shape",
          r["station"] == "KAAA" and "stations" not in r and r["credits_charged"] == 7, f"got {r}")


def main():
    test_client()
    test_archive()
    test_formatters()
    test_collect_path()
    test_grid_flow_batch_ifs()
    test_profiles()
    test_conus_only_coords()
    test_credit_estimate()
    test_ensemble_batch()
    npass = sum(1 for _, ok, _ in checks if ok)
    for label, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if detail:
            print(detail)
    print(f"\n{npass}/{len(checks)} checks passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
