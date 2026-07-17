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

import shutil
import tempfile
from datetime import datetime
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
    check("member excluded from credits", "member" in cols and ts.credits == 6)
    check("charged == credits when not cached", ts.charged == ts.credits)
    ts.cached = True
    check("charged == 0 on cache hit", ts.charged == 0)
    check("empty rows -> 0 credits", gribstream.TimeSeries("gfs", "u", cols, []).credits == 0)

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
    store.insert_model_data(con, rows)
    # obs at 18Z for verification (temp 25C so model 300K=26.85C -> +~1.9 bias)
    o = metar_parse("KTEST 171800Z 21008KT 10SM CLR 25/15 A2992")
    store.insert_obs(con, [o], year=2026, month=7, source="test")
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
          "mean T bias" in r.text, r.text)

    r = tools.run_tool("get_nearby_model_data", {"station": "KTEST", "variable": "t2m", "model": "gfs"},
                       db_path=path)
    check("get_nearby_model_data lists both points",
          "KTEST" in r.text and "KNB" in r.text, r.text)
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

    # IFS scaffold: fetchable + verified names, out of default MODELS, no hazard bundle
    check("ifsoper is a VALID model", "ifsoper" in gribstream.VALID_MODELS)
    check("ifsoper is NOT in the default prefetch set", "ifsoper" not in modeldata.MODELS)
    ifs = {(v.name, v.level, v.alias) for v in modeldata._surface_vars("ifsoper")}
    check("ifs surface uses ECMWF native names @ sfc",
          ("2t", "sfc", "t2m") in ifs and ("msl", "sfc", "mslp") in ifs and ("tcc", "sfc", "tcdc") in ifs)
    check("ifs has no hazard bundle yet", modeldata._hazard_vars("ifsoper") == [])
    try:
        gribstream.fetch_points("ifsoper", [(1.0, 1.0, "X")], [gribstream.Var("2t", "sfc", "t2m")])
        check("ifsoper accepted as a model (raises on missing time, not model)", False)
    except ValueError as e:
        check("ifsoper accepted as a model (raises on missing time, not model)",
              "time" in str(e).lower() or "either" in str(e).lower())


def main():
    test_client()
    test_archive()
    test_formatters()
    test_collect_path()
    test_grid_flow_batch_ifs()
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
