"""Cross-model consensus ablation experiment (design v2, 2026-07-23).

Tests whether rendering multi-model NWP data as a CONSENSUS (vs raw stacked tables) helps a
VLM forecast hard weather. 3 hard CONUS locations x 5 render arms x 2 models = 30 calls.

Arms (only the model-data block changes; obs + trend note + task are constant):
  control  raw per-model hourly tables (latest run)      -- today's get_model_state shape
  A        consensus digest: per-hour MEAN + range + confidence flag
  B        disagreement-only alerts (names the outlier model)
  C        per-field transpose (models as columns), all fields
  AC       A + C

Consensus value = MEAN (arithmetic; circular for direction); range = min..max across models.
Model data is asOf-pinned to each issue time (leakage-safe). Each location also carries the
LATEST run (hourly, the consensus data) plus 2 PREVIOUS runs (3-hourly) rendered as a
constant run-to-run trend note. Truth + 24h preceding obs come from the benchmark DB.

Stages (run in order):
  --fetch     pull all 3 locations' model data into the scratch archive (BILLS ~4700 credits)
  --preview STATION ARM   print one assembled prompt (free)
  --run       make the 30 LLM calls, save forecasts as JSON (LLM $, no GRIBStream credits)
  --score     score saved forecasts vs obs, write the report
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from forecaster import gribstream, modeldata, store, tools
from forecaster.config import settings
from forecaster.llm import client

# --- configuration ----------------------------------------------------------------------
BENCH_DB = "data/benchmark/forecaster.duckdb"
EXP_DIR = Path("data/consensus_experiment")
SCRATCH_DB = str(EXP_DIR / "archive.duckdb")
FCST_DIR = EXP_DIR / "forecasts"

DET_MODELS = ("gfs", "hrrr", "nbm", "ifsoper")
# HRRR runs hourly but its OFF-cycle runs only reach 18h; only the 6-hourly MAIN cycles
# (00/06/12/18Z) extend to 48h, so a 30h forecast must select HRRR at 6h. NBM's hourly runs
# all go the full distance, so NBM stays hourly.
_CYCLE_H = {"gfs": 6, "hrrr": 6, "nbm": 1, "ifsoper": 6}
_POST_LAG_H = {"gfs": 5, "hrrr": 2, "nbm": 2, "ifsoper": 8}   # realistic availability lag
_TREND_SPACING_H = {"gfs": 6, "hrrr": 6, "nbm": 3, "ifsoper": 6}

FORECAST_HOURS = 30
LATEST_STEP_H = 1        # latest run: hourly (the consensus data)
PREV_STEP_H = 3          # previous runs: 3-hourly (trend only)

LLM_MODELS = {"gemma": "google/gemma-4-31B-it", "minimax": "MiniMaxAI/MiniMax-M3"}
ARMS = ("control", "A", "B", "C", "AC")
TEMPERATURE = 0.2
MAX_TOKENS = 9000        # reasoning models spend heavily before the JSON; too low truncates it


@dataclass
class Location:
    station: str
    issue: datetime
    regime: str
    event: str


LOCATIONS = [
    Location("KDMA", datetime(2026, 7, 17, 12), "SW monsoon microburst",
             "07-17 22Z: 31g49kt +TSRA, T 30->23 outflow, cig 3600, vis 4"),
    Location("KMIB", datetime(2026, 7, 16, 22), "radiation fog / low-IFR",
             "07-17 10-12Z: vis 0.19SM, cig 100ft, calm"),
    Location("KVBG", datetime(2026, 7, 18, 0), "marine stratus",
             "07-18 15Z: cig 200ft, vis 2.0, sea-breeze wind shift"),
]

# --- surface variable set (aviation fields) ---------------------------------------------
_VARS = {
    "gfs":     [("TMP", "2 m above ground", "t2m"), ("DPT", "2 m above ground", "td2m"),
                ("UGRD", "10 m above ground", "u10"), ("VGRD", "10 m above ground", "v10"),
                ("GUST", "surface", "gust"), ("PRMSL", "mean sea level", "mslp"),
                ("TCDC", "entire atmosphere", "tcdc"), ("VIS", "surface", "vis"),
                ("HGT", "cloud ceiling", "ceil")],
    "hrrr":    [("TMP", "2 m above ground", "t2m"), ("DPT", "2 m above ground", "td2m"),
                ("UGRD", "10 m above ground", "u10"), ("VGRD", "10 m above ground", "v10"),
                ("GUST", "surface", "gust"), ("MSLMA", "mean sea level", "mslp"),
                ("TCDC", "entire atmosphere", "tcdc"), ("VIS", "surface", "vis"),
                ("HGT", "cloud ceiling", "ceil")],
    "nbm":     [("TMP", "2 m above ground", "t2m"), ("DPT", "2 m above ground", "td2m"),
                ("WIND", "10 m above ground", "wind"), ("WDIR", "10 m above ground", "wdir"),
                ("GUST", "10 m above ground", "gust"), ("TCDC", "surface", "tcdc"),
                ("VIS", "surface", "vis"), ("CEIL", "cloud ceiling", "ceil")],
    "ifsoper": [("2t", "sfc", "t2m"), ("2d", "sfc", "td2m"), ("10u", "sfc", "u10"),
                ("10v", "sfc", "v10"), ("msl", "sfc", "mslp"), ("tcc", "sfc", "tcdc")],
}


def _vars(model):
    return [gribstream.Var(*t) for t in _VARS[model]]


def station_latlon(station):
    from forecaster import awc
    return awc.station_latlon(station)


# --- unit helpers -----------------------------------------------------------------------
def k2c(k):
    return None if k is None else k - 273.15


def ms2kt(v):
    return None if v is None else v * 1.94384


def vis_sm(m):
    if m is None:
        return None
    return min(m / 1609.34, 10.0)


def ceil_ft(m):
    # HGT@cloud ceiling returns the height of any cloud, incl. high cirrus at 30-49 kft that
    # is not an operational ceiling. Cap at 12000 ft (3658 m): above that = no ceiling, so the
    # consensus is not polluted by high cloud when the aviation question is a LOW ceiling.
    if m is None or m > 3658 or m < 0:
        return None
    return round(m * 3.28084 / 100) * 100


def wind_ds(vm, model):
    """(dir_deg, speed_kt) from whichever wind form the model carries."""
    if model == "nbm":
        d, s = vm.get("wdir"), vm.get("wind")
        return (None if d is None else d % 360), ms2kt(s)
    u, v = vm.get("u10"), vm.get("v10")
    if u is None or v is None:
        return None, None
    return math.degrees(math.atan2(-u, -v)) % 360, ms2kt(math.hypot(u, v))


def circ_mean(degs):
    degs = [d for d in degs if d is not None]
    if not degs:
        return None
    x = sum(math.cos(math.radians(d)) for d in degs) / len(degs)
    y = sum(math.sin(math.radians(d)) for d in degs) / len(degs)
    return math.degrees(math.atan2(y, x)) % 360


def circ_spread(degs):
    degs = [d for d in degs if d is not None]
    m = 0.0
    for i in range(len(degs)):
        for j in range(i + 1, len(degs)):
            d = abs(degs[i] - degs[j]) % 360
            m = max(m, min(d, 360 - d))
    return m


# --- run selection + fetch --------------------------------------------------------------
def target_runs(model, issue):
    """The latest AVAILABLE run at `issue` (given post lag) + 2 previous, at the model's
    trend spacing. Newest first."""
    cycle = _CYCLE_H[model]
    avail = issue - timedelta(hours=_POST_LAG_H[model])
    latest = avail.replace(minute=0, second=0, microsecond=0)
    latest -= timedelta(hours=latest.hour % cycle)
    sp = _TREND_SPACING_H[model]
    return [latest, latest - timedelta(hours=sp), latest - timedelta(hours=2 * sp)]


def fetch_location(loc, con):
    """Pull latest (hourly) + 2 previous (3-hourly) runs per model for one location into the
    scratch archive. Returns credits charged."""
    lat, lon = station_latlon(loc.station)
    coords = [(lat, lon, loc.station)]
    charged, rows = 0, []
    fetched_at = datetime.utcnow()
    win_end = loc.issue + timedelta(hours=FORECAST_HOURS)
    for model in DET_MODELS:
        cycle = _CYCLE_H[model]
        runs = target_runs(model, loc.issue)
        for i, run in enumerate(runs):
            step = LATEST_STEP_H if i == 0 else PREV_STEP_H
            n = int((win_end - loc.issue).total_seconds() // 3600) // step
            times = [loc.issue + timedelta(hours=step * k) for k in range(n + 1)]
            # Pin asOf just AFTER this run so exactly it qualifies. Pinning before the
            # successor (run+cycle) fails for HRRR, whose hourly off-cycle runs would win
            # over the targeted 6-hourly main cycle.
            run_as_of = run + timedelta(minutes=1)
            _ = cycle
            try:
                ts = gribstream.fetch_points(model, coords, _vars(model), times=times,
                                             as_of=run_as_of, use_cache=True)
                charged += ts.charged
                for r in ts.rows:
                    rn, vt = r.get("forecasted_at"), r.get("forecasted_time")
                    if rn is None or vt is None:
                        continue
                    for name, level, alias in _VARS[model]:
                        val = r.get(alias)
                        if model == "ifsoper" and alias == "tcdc" and val is not None:
                            val = val * 100.0
                        rows.append({"model": model, "run": rn, "valid_time": vt,
                                     "lat": lat, "lon": lon, "loc_id": loc.station,
                                     "variable": alias, "value": val, "member": 0,
                                     "as_of": run_as_of, "fetched_at": fetched_at})
            except ValueError as e:
                print(f"  {loc.station} {model} run {run:%m-%dT%HZ}: {e}")
    store.init_model_data_schema(con)
    ins = store.insert_model_data(con, rows)
    print(f"  {loc.station}: {ins} rows, {charged} credits")
    return charged


# --- archive readers --------------------------------------------------------------------
def read_runs(con, loc):
    """{model: {run: {valid_time: {alias: value}}}} for one location, forward window only."""
    lat, lon = station_latlon(loc.station)
    out = {}
    for model in DET_MODELS:
        rows = store.model_data_series(con, model, lat, lon, start=loc.issue,
                                       end=loc.issue + timedelta(hours=FORECAST_HOURS))
        for r in rows:
            out.setdefault(model, {}).setdefault(r["run"], {}) \
               .setdefault(r["valid_time"], {})[r["variable"]] = r["value"]
    return out


def latest_run_pivot(runs_by_model):
    """{model: {valid_time: {alias: value}}} using each model's newest run."""
    out = {}
    for model, by_run in runs_by_model.items():
        if not by_run:
            continue
        newest = max(by_run)
        out[model] = by_run[newest]
    return out


def valid_hours(pivot):
    return sorted({vt for m in pivot.values() for vt in m})


# --- per-hour multi-model derived values ------------------------------------------------
def hour_values(pivot, vt):
    """Per model at one valid time: dict alias->derived (T C, dir, spd, gust kt, vis SM, ceil ft)."""
    out = {}
    for model, series in pivot.items():
        vm = series.get(vt)
        if vm is None:
            continue
        d, s = wind_ds(vm, model)
        out[model] = {"t": k2c(vm.get("t2m")), "td": k2c(vm.get("td2m")),
                      "dir": d, "spd": s, "gust": ms2kt(vm.get("gust")),
                      "vis": vis_sm(vm.get("vis")), "ceil": ceil_ft(vm.get("ceil")),
                      "cld": vm.get("tcdc")}
    return out


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    return sum(vals) / len(vals), min(vals), max(vals)     # mean, lo, hi


def _f(v, dp=0):
    return "--" if v is None else f"{v:.{dp}f}"


# --- ARM RENDERERS ----------------------------------------------------------------------
def render_control(pivot):
    """Raw per-model hourly tables (latest run) -- today's get_model_state shape."""
    out = []
    for model in DET_MODELS:
        series = pivot.get(model)
        if not series:
            continue
        out.append(f"{model.upper()} (latest run):")
        out.append(f"  {'Valid':<8}{'T':>4}{'Td':>4}{'dir':>5}{'spd':>5}{'gst':>5}"
                   f"{'vis':>6}{'ceil':>7}{'cld%':>6}")
        for vt in sorted(series):
            hv = hour_values({model: series}, vt).get(model, {})
            out.append(f"  {vt:%d/%HZ}{_f(hv.get('t')):>4}{_f(hv.get('td')):>4}"
                       f"{_f(hv.get('dir')):>5}{_f(hv.get('spd')):>5}{_f(hv.get('gust')):>5}"
                       f"{_f(hv.get('vis'),1):>6}{(_f(hv.get('ceil')) if hv.get('ceil') else 'none'):>7}"
                       f"{_f(hv.get('cld')):>6}")
        out.append("")
    return "\n".join(out)


def _conf(field, lo, hi, spd_mean=None):
    if lo is None:
        return "--"
    rng = hi - lo
    if field == "dir":
        if spd_mean is not None and spd_mean < 5:
            return "light"
        return "high" if rng <= 30 else ("med" if rng <= 60 else "LOW")
    thr = {"t": (2, 4), "spd": (3, 6), "gust": (4, 8), "vis": (1, 3), "ceil": (500, 1500)}[field]
    return "high" if rng <= thr[0] else ("med" if rng <= thr[1] else "LOW")


def render_A(pivot):
    """Consensus digest: per-hour MEAN + range + confidence flag."""
    out = ["CONSENSUS DIGEST -- per hour, MEAN across models (range in parens); "
           "dir is circular mean, * = light wind (direction unreliable).",
           f"  {'Valid':<7} {'T C':<12} {'dir':>4} {'spd':>3} {'gust kt':<11} "
           f"{'vis':>4} {'ceil':>6}  confidence"]
    for vt in valid_hours(pivot):
        hv = hour_values(pivot, vt)
        tm, tlo, thi = _agg([v["t"] for v in hv.values()])
        dm = circ_mean([v["dir"] for v in hv.values()])
        dspread = circ_spread([v["dir"] for v in hv.values()])
        sm, slo, shi = _agg([v["spd"] for v in hv.values()])
        gm, glo, ghi = _agg([v["gust"] for v in hv.values()])
        vm_, vlo, vhi = _agg([v["vis"] for v in hv.values()])
        cm, clo, chi = _agg([v["ceil"] for v in hv.values()])
        light = sm is not None and sm < 5
        tcell = f"{_f(tm)} ({_f(tlo)}-{_f(thi)})"
        gcell = "--" if gm is None else f"{_f(gm)} ({_f(glo)}-{_f(ghi)})"
        dcell = (_f(dm) + "*") if light else _f(dm)
        conf = (f"T:{_conf('t', tlo, thi)} dir:{'light' if light else _conf('dir', 0, dspread, sm)}"
                f" gust:{_conf('gust', glo, ghi)} cig:{_conf('ceil', clo, chi) if cm else 'clr'}")
        out.append(f"  {vt:%d/%HZ} {tcell:<12} {dcell:>4} {_f(sm):>3} {gcell:<11} "
                   f"{_f(vm_, 1):>4} {(_f(cm) if cm else 'none'):>6}  {conf}")
    return "\n".join(out)


def render_B(pivot):
    """Disagreement-only: flag hours where models diverge past a threshold; name the outlier."""
    out = ["MODEL DISAGREEMENT -- only hours where models diverge materially are listed; "
           "an outlier model is named. Silent hours = models agree."]
    any_line = False
    for vt in valid_hours(pivot):
        hv = hour_values(pivot, vt)
        notes = []
        ts = {m: v["t"] for m, v in hv.items() if v["t"] is not None}
        if ts and max(ts.values()) - min(ts.values()) > 4:
            notes.append(f"T {_f(min(ts.values()))}-{_f(max(ts.values()))} "
                         f"({min(ts, key=ts.get)} cold, {max(ts, key=ts.get)} warm)")
        gs = {m: v["gust"] for m, v in hv.items() if v["gust"] is not None}
        if gs and max(gs.values()) - min(gs.values()) > 8:
            notes.append(f"gust {_f(min(gs.values()))}-{_f(max(gs.values()))} "
                         f"({min(gs, key=gs.get)} low, {max(gs, key=gs.get)} high)")
        ds = {m: v["dir"] for m, v in hv.items() if v["dir"] is not None}
        ss = [v["spd"] for v in hv.values() if v["spd"] is not None]
        if ds and ss and sum(ss) / len(ss) >= 5 and circ_spread(list(ds.values())) > 60:
            notes.append(f"dir spread {circ_spread(list(ds.values())):.0f}deg")
        cs = {m: v["ceil"] for m, v in hv.items()}
        has_cig = [m for m, c in cs.items() if c is not None]
        if has_cig and len(has_cig) != len([m for m in hv]):
            notes.append(f"ceiling: {','.join(has_cig)} show a ceiling, others none")
        elif len(has_cig) >= 2:
            cc = [c for c in cs.values() if c is not None]
            if max(cc) - min(cc) > 1500:
                notes.append(f"ceiling {min(cc):.0f}-{max(cc):.0f}ft")
        if notes:
            any_line = True
            out.append(f"  {vt:%d/%HZ}: " + "; ".join(notes))
    if not any_line:
        out.append("  (models agree at every hour within thresholds)")
    return "\n".join(out)


def render_C(pivot):
    """Per-field transpose: models as columns, one sub-table per field, + mean + range."""
    fields = [("TEMPERATURE (C)", "t", 0), ("WIND DIR (deg)", "dir", 0),
              ("WIND SPEED (kt)", "spd", 0), ("WIND GUST (kt)", "gust", 0),
              ("VISIBILITY (SM)", "vis", 1), ("CEILING (ft)", "ceil", 0)]
    out = ["PER-FIELD MODEL COMPARISON -- models in columns; mean + range at right."]
    cols = [m for m in DET_MODELS if m in pivot]
    for label, key, dp in fields:
        out.append(f"\n{label}")
        out.append("  " + f"{'Valid':<8}" + "".join(f"{m.upper():>7}" for m in cols)
                   + f"{'mean':>7}{'range':>8}")
        for vt in valid_hours(pivot):
            hv = hour_values(pivot, vt)
            vals = {m: hv.get(m, {}).get(key) for m in cols}
            if key == "dir":
                present = [v for v in vals.values() if v is not None]
                mean = circ_mean(present)
                rng = circ_spread(present)
            else:
                mean, lo, hi = _agg(list(vals.values()))
                rng = None if lo is None else hi - lo
            cells = "".join(f"{(_f(vals[m], dp) if vals[m] is not None else '--'):>7}" for m in cols)
            out.append(f"  {vt:%d/%HZ}{cells}{_f(mean, dp):>7}{_f(rng, dp):>8}")
    return "\n".join(out)


def render_trend(runs_by_model, event_vt):
    """Constant run-to-run note: how each model's forecast for the event hour shifted across
    its last 3 runs (oldest->newest). Shows temporal stability alongside the spatial consensus."""
    out = ["RUN-TO-RUN TREND -- each model's forecast for the event window across its last 3 "
           "runs (oldest -> newest). Rising/falling = the model is trending; steady = confident."]
    for model in DET_MODELS:
        by_run = runs_by_model.get(model, {})
        if not by_run:
            continue
        runs = sorted(by_run)[-3:]
        # nearest stored valid time to the event in each run
        def near(series):
            vts = [vt for vt in series if vt in series]
            if not vts:
                return None
            return min(vts, key=lambda vt: abs((vt - event_vt).total_seconds()))
        gline, tline = [], []
        for run in runs:
            vt = near(by_run[run])
            vm = by_run[run].get(vt, {}) if vt else {}
            gline.append(_f(ms2kt(vm.get("gust"))))
            tline.append(_f(k2c(vm.get("t2m"))))
        out.append(f"  {model.upper():8} gust {'->'.join(gline)}   T {'->'.join(tline)}")
    return "\n".join(out)


# --- observations -----------------------------------------------------------------------
def render_obs(bench, station, issue):
    """Decoded 24h preceding obs (the recent trend the forecaster sees at issue)."""
    rows = store.window(bench, station, issue - timedelta(hours=24), issue)
    out = [f"PRECEDING OBSERVATIONS (last 24h before now, {station}):",
           f"  {'Time':<8}{'wind':>9}{'gust':>5}{'vis':>6}{'ceil':>7}{'T/Td':>8}  wx"]
    for o in rows:
        d, s = o.get("wind_dir_deg"), o.get("wind_speed")
        wind = f"{int(d):03d}/{int(s):02d}" if d is not None and s is not None else "calm"
        g = o.get("wind_gust")
        gcell = "" if g is None else f"{int(g)}"
        cig = o.get("ceiling_ft")
        cigcell = f"{int(cig)}" if cig else "none"
        ttd = f"{_f(o.get('temp_c'))}/{_f(o.get('dewpoint_c'))}"
        wx = ",".join(o.get("weather") or [])
        out.append(f"  {o['obs_time']:%d/%HZ}{wind:>9}{gcell:>5}{_f(o.get('vis_sm'), 1):>6}"
                   f"{cigcell:>7}{ttd:>8}  {wx}")
    return "\n".join(out)


# --- prompt assembly --------------------------------------------------------------------
_SCHEMA = """{
  "peak_wind":       {"speed_kt": <int>, "dir_deg": <int>, "time": "DD/HHZ"},
  "peak_gust":       {"gust_kt": <int>, "time": "DD/HHZ"},
  "max_temp":        {"temp_c": <int>, "time": "DD/HHZ"},
  "min_temp":        {"temp_c": <int>, "time": "DD/HHZ"},
  "wind_shift":      {"occurs": <bool>, "to_dir_deg": <int|null>, "time": "DD/HHZ|null"},
  "present_weather": {"occurs": <bool>, "types": ["TS"|"RA"|...], "time": "DD/HHZ|null"},
  "min_visibility":  {"vis_sm": <number>, "time": "DD/HHZ"},
  "min_ceiling":     {"ceiling_ft": <int|null>, "time": "DD/HHZ|null"}
}"""

TASK = """You are a military weather forecaster. Using the recent observations and the model
guidance below, forecast the conditions for {station} over the next 30 hours.

NOW: {issue:%Y-%m-%d %H}00Z. Forecast the window {issue:%d/%H}Z through {end:%d/%H}Z.
All model-guidance and observation times are DD/HHZ (UTC). The observations are the PAST
(before now); the model guidance is the FUTURE (from now forward).

Keep any reasoning to a FEW SENTENCES, then output EXACTLY this JSON object as the LAST thing
in your reply (peak/min = the single most extreme value in the window, with the hour it
occurs; ceiling_ft null means no ceiling). Do not wrap it in code fences.
{schema}
"""


_ARM_LABELS = {"control": "raw per-model tables", "A": "consensus digest",
               "B": "disagreement alerts", "C": "per-field comparison",
               "AC": "consensus digest + per-field comparison",
               "A_gefs": "consensus digest + GEFS 31-member ensemble probabilities"}


def build_prompt(bench, con, loc, arm):
    pivot = latest_run_pivot(read_runs(con, loc))
    runs = read_runs(con, loc)
    event_vt = _event_vt(loc)
    if arm == "A_gefs":
        gefs = tools.run_tool("get_ensemble_prob", {"station": loc.station}, db_path=SCRATCH_DB).text
        blocks = render_A(pivot) + "\n\nGEFS ENSEMBLE (probabilistic, 3-hourly):\n" + gefs
    else:
        blocks = {
            "control": render_control(pivot),
            "A": render_A(pivot),
            "B": render_B(pivot),
            "C": render_C(pivot),
            "AC": render_A(pivot) + "\n\n" + render_C(pivot),
        }[arm]
    parts = [
        TASK.format(station=loc.station, issue=loc.issue,
                    end=loc.issue + timedelta(hours=FORECAST_HOURS), schema=_SCHEMA),
        "",
        render_obs(bench, loc.station, loc.issue),
        "",
        "MODEL GUIDANCE (" + _ARM_LABELS[arm] + "):",
        blocks,
        "",
        render_trend(runs, event_vt),
    ]
    return "\n".join(parts)


def fetch_gefs():
    """Archive the GEFS ensemble (thinned 11 members) for the 3 cases into the experiment
    archive, asOf-pinned to each issue time (leakage-safe). BILLS ~3300 credits."""
    con = store.connect(SCRATCH_DB)
    con.close()
    total = 0
    for loc in LOCATIONS:
        r = modeldata.prefetch_ensemble(loc.station, as_of=loc.issue, hours=FORECAST_HOURS,
                                        step_h=3, db_path=SCRATCH_DB)
        total += r["credits_charged"]
        print(f"  {loc.station}: {r['rows']} rows, {len(r['members'])} members, "
              f"{r['credits_charged']} credits")
    print(f"GEFS TOTAL charged: {total} credits")


def _event_vt(loc):
    return {"KDMA": datetime(2026, 7, 17, 22), "KMIB": datetime(2026, 7, 17, 11),
            "KVBG": datetime(2026, 7, 18, 15)}[loc.station]


# --- LLM call + parse -------------------------------------------------------------------
def call_model(model_id, prompt):
    r = client.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=settings.llm_seed)
    msg = r.choices[0].message
    text = (msg.content or "") + "\n" + (getattr(msg, "reasoning", "") or "")
    return _extract_json(text), text.strip(), (r.usage.completion_tokens if r.usage else 0)


def _extract_json(text):
    """Return the last complete top-level {...} object that parses and carries a forecast key
    (skips the schema echo / partial objects). Balanced-brace scan, so trailing prose is fine."""
    best = None
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start:i + 1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict) and any(k in obj for k in
                                                     ("peak_gust", "peak_wind", "max_temp")):
                        best = obj
                except json.JSONDecodeError:
                    pass
    return best


# --- scoring ----------------------------------------------------------------------------
def observed_extremes(bench, loc):
    rows = store.window(bench, loc.station, loc.issue, loc.issue + timedelta(hours=FORECAST_HOURS))
    def peak(key, want_max=True):
        vals = [(o[key], o["obs_time"], o) for o in rows if o.get(key) is not None]
        if not vals:
            return None, None, None
        v, t, o = (max if want_max else min)(vals, key=lambda x: x[0])
        return v, t, o
    pw, pwt, pwo = peak("wind_speed", True)
    pg, pgt, _ = peak("wind_gust", True)
    mx, mxt, _ = peak("temp_c", True)
    mn, mnt, _ = peak("temp_c", False)
    mv, mvt, _ = peak("vis_sm", False)
    mc, mct, _ = peak("ceiling_ft", False)
    wx = sorted({w for o in rows for w in (o.get("weather") or [])})
    ts_obs = [o for o in rows if any("TS" in w for w in (o.get("weather") or []))]
    return {
        "peak_wind": (pw, pwt, pwo.get("wind_dir_deg") if pwo else None),
        "peak_gust": (pg or 0, pgt),
        "max_temp": (mx, mxt), "min_temp": (mn, mnt),
        "min_visibility": (mv, mvt), "min_ceiling": (mc, mct),
        "present_weather": (bool(wx), wx, ts_obs[0]["obs_time"] if ts_obs else None),
    }


def _hr(s):
    """Parse a 'DD/HHZ' forecast time to an hour-of-window offset key (day, hour)."""
    if not s or not isinstance(s, str):
        return None
    m = re.search(r"(\d{1,2})\D+(\d{1,2})", s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _timing_err(fc_time, obs_time):
    hr = _hr(fc_time)
    if hr is None or obs_time is None:
        return None
    fc = obs_time.replace(day=hr[0], hour=hr[1], minute=0)
    return abs((fc - obs_time).total_seconds()) / 3600.0


def score_forecast(fc, obs):
    """Per-element error dict. Continuous: (err, bias). Timing: hours. Dir: angular."""
    s = {}
    if fc is None:
        return {"parse_fail": True}
    def g(path, *keys):
        d = fc.get(path, {})
        return tuple(d.get(k) for k in keys) if isinstance(d, dict) else (None,) * len(keys)
    ow, owt, owd = obs["peak_wind"]
    fw, fwd, fwt = g("peak_wind", "speed_kt", "dir_deg", "time")
    if fw is not None and ow is not None:
        s["peak_wind_err"] = abs(fw - ow)
        s["peak_wind_bias"] = fw - ow
        s["peak_wind_timing"] = _timing_err(fwt, owt)
    if fwd is not None and owd is not None:
        s["peak_wind_dir_err"] = min(abs(fwd - owd) % 360, 360 - abs(fwd - owd) % 360)
    og, ogt = obs["peak_gust"]
    fg, fgt = g("peak_gust", "gust_kt", "time")
    if fg is not None:
        s["peak_gust_err"] = abs(fg - og)
        s["peak_gust_bias"] = fg - og
        s["peak_gust_timing"] = _timing_err(fgt, ogt)
    for el, okey in (("max_temp", "max_temp"), ("min_temp", "min_temp")):
        ov, ovt = obs[okey]
        fv, ft = g(el, "temp_c", "time")
        if fv is not None and ov is not None:
            s[f"{el}_err"] = abs(fv - ov)
            s[f"{el}_bias"] = fv - ov
            s[f"{el}_timing"] = _timing_err(ft, ovt)
    ov, ovt = obs["min_visibility"]
    fv, ft = g("min_visibility", "vis_sm", "time")
    if fv is not None and ov is not None:
        s["min_vis_err"] = abs(fv - ov)
        s["min_vis_bias"] = fv - ov
        s["min_vis_timing"] = _timing_err(ft, ovt)
    ov, ovt = obs["min_ceiling"]
    fv, ft = g("min_ceiling", "ceiling_ft", "time")
    if ov is not None:                              # only score ceiling where obs had one
        if fv is not None:
            s["min_ceil_err"] = abs(fv - ov)
            s["min_ceil_bias"] = fv - ov
            s["min_ceil_timing"] = _timing_err(ft, ovt)
        else:
            s["min_ceil_missed"] = 1                # obs had a ceiling, forecast said none
    o_occ, o_types, o_tst = obs["present_weather"]
    f_occ, f_types, f_tst = g("present_weather", "occurs", "types", "time")
    if f_occ is not None:
        s["pw_correct"] = int(bool(f_occ) == bool(o_occ))
        o_ts = o_tst is not None
        f_ts = bool(f_types) and any("TS" in str(t).upper() for t in (f_types or []))
        s["ts_hit"] = int(f_ts and o_ts)
        s["ts_miss"] = int(o_ts and not f_ts)
        s["ts_fa"] = int(f_ts and not o_ts)
        if f_ts and o_ts:
            s["ts_timing"] = _timing_err(f_tst, o_tst)
    return s


# --- main -------------------------------------------------------------------------------
def do_fetch():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    con = store.connect(SCRATCH_DB)
    total = 0
    for loc in LOCATIONS:
        print(f"fetch {loc.station} issue {loc.issue:%m-%dT%HZ}")
        total += fetch_location(loc, con)
    con.close()
    print(f"TOTAL charged: {total} credits")


def do_preview(station, arm):
    bench = store.connect(BENCH_DB, read_only=True)
    con = store.connect(SCRATCH_DB, read_only=True)
    loc = next(x for x in LOCATIONS if x.station == station)
    print(build_prompt(bench, con, loc, arm))


def do_run():
    FCST_DIR.mkdir(parents=True, exist_ok=True)
    bench = store.connect(BENCH_DB, read_only=True)
    con = store.connect(SCRATCH_DB, read_only=True)
    for loc in LOCATIONS:
        for arm in ARMS:
            prompt = build_prompt(bench, con, loc, arm)
            for mname, mid in LLM_MODELS.items():
                out = FCST_DIR / f"{loc.station}_{arm}_{mname}.json"
                if out.exists():
                    print(f"skip {out.name} (exists)")
                    continue
                try:
                    parsed, text, tok = call_model(mid, prompt)
                    out.write_text(json.dumps({"station": loc.station, "arm": arm, "model": mname,
                                               "forecast": parsed, "raw": text, "tokens": tok}, indent=2))
                    print(f"{loc.station} {arm:8} {mname:8} tok={tok} parsed={'OK' if parsed else 'FAIL'}")
                except Exception as e:  # noqa: BLE001
                    print(f"{loc.station} {arm} {mname}: ERROR {type(e).__name__}: {e}")


def do_score():
    bench = store.connect(BENCH_DB, read_only=True)
    obs = {loc.station: observed_extremes(bench, loc) for loc in LOCATIONS}
    results = []
    for f in sorted(FCST_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        sc = score_forecast(d["forecast"], obs[d["station"]])
        results.append({**{k: d[k] for k in ("station", "arm", "model")}, "score": sc})
    _report(obs, results)


def _report(obs, results):
    lines = ["# Consensus experiment results", f"_{datetime.now():%Y-%m-%d %H:%M}_", ""]
    lines.append("## Observed extremes (truth)\n")
    for st, o in obs.items():
        pwt = f"{o['peak_wind'][1]:%d/%HZ}" if o['peak_wind'][1] else "--"
        pgt = f"{o['peak_gust'][1]:%d/%HZ}" if o['peak_gust'][1] else "--"
        lines.append(f"- **{st}**: peak wind {o['peak_wind'][0]}kt dir{o['peak_wind'][2]} "
                     f"@{pwt}; peak gust {o['peak_gust'][0]}kt @{pgt}; "
                     f"Tmax {o['max_temp'][0]} Tmin {o['min_temp'][0]}; "
                     f"min vis {o['min_visibility'][0]} min cig {o['min_ceiling'][0]}; "
                     f"wx {o['present_weather'][1]}")
    # aggregate abs error by (arm) across elements, per model
    elements = ["peak_wind_err", "peak_wind_dir_err", "peak_gust_err", "max_temp_err",
                "min_temp_err", "min_vis_err", "min_ceil_err"]
    lines.append("\n## Mean absolute error by arm (lower better), averaged over the 3 locations\n")
    lines.append("| model | arm | " + " | ".join(e.replace("_err", "") for e in elements) + " | gust_bias |")
    lines.append("|" + "---|" * (len(elements) + 3))
    for mname in LLM_MODELS:
        for arm in ARMS:
            rs = [r["score"] for r in results if r["model"] == mname and r["arm"] == arm]
            cells = []
            for e in elements:
                vals = [s[e] for s in rs if e in s]
                cells.append(f"{sum(vals) / len(vals):.1f}" if vals else "--")
            gb = [s["peak_gust_bias"] for s in rs if "peak_gust_bias" in s]
            gbc = f"{sum(gb) / len(gb):+.1f}" if gb else "--"
            lines.append(f"| {mname} | {arm} | " + " | ".join(cells) + f" | {gbc} |")
    out = EXP_DIR / "results.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--preview", nargs=2, metavar=("STATION", "ARM"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        do_fetch()
    elif a.preview:
        do_preview(*a.preview)
    elif a.run:
        do_run()
    elif a.score:
        do_score()
    else:
        ap.print_help()

