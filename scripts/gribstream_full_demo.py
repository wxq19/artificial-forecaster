"""GRIBStream FULL-product scope-out -- the three capabilities at full fidelity for a
bad-weather site, with icing + turbulence diagnosed from model conditions.

This is the REVIEW reference: the formatters here are what would become the agent tools
(lifted into tools.py, data via gribstream.py) if blessed. Not committed, not wired as
tools yet. Writes each product as a text receipt (exactly what the model would see) to a
markdown file for the review artifact.

Site: KMSP (Minneapolis) -- Upper Midwest supercell threat this evening (SPC risk area),
full GFS/HRRR/NBM + pressure-level coverage. Run:
    uv run python scripts/gribstream_full_demo.py
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import awc, gribstream, metar

STATION = "KMSP"
OUT = Path("data/charts/temp/GRIBSTREAM_full_demo_KMSP.md")
_charged = 0

# --- per-model surface field specs (names differ: GFS/HRRR u/v vs NBM speed/dir) --------
_SFC_GFS = [gribstream.Var("TMP", "2 m above ground", "t2m"),
            gribstream.Var("DPT", "2 m above ground", "td2m"),
            gribstream.Var("UGRD", "10 m above ground", "u10"),
            gribstream.Var("VGRD", "10 m above ground", "v10"),
            gribstream.Var("GUST", "surface", "gust"),
            gribstream.Var("PRMSL", "mean sea level", "mslp"),
            gribstream.Var("TCDC", "entire atmosphere", "tcdc"),
            gribstream.Var("VIS", "surface", "vis"),
            gribstream.Var("HGT", "cloud ceiling", "ceil")]
_SFC_NBM = [gribstream.Var("TMP", "2 m above ground", "t2m"),
            gribstream.Var("DPT", "2 m above ground", "td2m"),
            gribstream.Var("WIND", "10 m above ground", "wind"),
            gribstream.Var("WDIR", "10 m above ground", "wdir"),
            gribstream.Var("GUST", "10 m above ground", "gust"),
            gribstream.Var("TCDC", "surface", "tcdc"),
            gribstream.Var("VIS", "surface", "vis"),
            gribstream.Var("CEIL", "cloud ceiling", "ceil")]
_SFC = {"gfs": _SFC_GFS, "hrrr": _SFC_GFS, "nbm": _SFC_NBM}


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _floor(t):
    return t.replace(minute=0, second=0, microsecond=0)


def _fetch(model, lat, lon, variables, **kw):
    global _charged
    ts = gribstream.fetch_timeseries(model, lat, lon, variables, name=STATION, **kw)
    _charged += ts.charged
    return ts


# --- unit helpers ---------------------------------------------------------------------
def _c(k):    return None if k is None else k - 273.15          # noqa: E704
def _kt(ms):  return None if ms is None else ms * 1.94384       # noqa: E704
def _dir(u, v):
    return int(round((270.0 - math.degrees(math.atan2(v, u))) % 360.0 / 10.0) * 10) % 360


def _wind_cell(r, model):
    if model == "nbm":
        spd, d = r.get("wind"), r.get("wdir")
        if spd is None or d is None:
            return "   --"
        return f"{int(round(d)) % 360:03d}/{round(_kt(spd)):02d}"
    u, v = r.get("u10"), r.get("v10")
    if u is None or v is None:
        return "   --"
    return f"{_dir(u, v):03d}/{round(_kt(math.hypot(u, v))):02d}"


def _vis_sm(m):
    if m is None:
        return "--"
    sm = m / 1609.34
    return "P6" if sm >= 6 else f"{sm:.1f}"


def _ceil_ft(m):
    if m is None or m > 15000 or m < 0:     # fill / no ceiling
        return "none"
    return f"{round(m * 3.28084 / 100) * 100:d}"


# --- Capability 1: multi-model current/forecast state ---------------------------------
def cap1(lat, lon):
    now = _floor(_utcnow())
    hrs = 12
    blocks = []
    peaks = {}
    for model in ("gfs", "hrrr", "nbm"):
        try:
            ts = _fetch(model, lat, lon, _SFC[model],
                        from_time=now + timedelta(hours=1), until_time=now + timedelta(hours=hrs))
        except ValueError as e:
            blocks.append(f"{model.upper()}: unavailable ({e})")
            continue
        rows = sorted(ts.rows, key=lambda r: r["forecasted_time"])
        run = ts.runs[0] if ts.runs else None
        lines = [f"{model.upper()} surface forecast for {STATION} -- run "
                 f"{run:%Y-%m-%dT%HZ} (source: {ts.url})",
                 f"{'Valid (Z)':<15}{'T C':>5}{'Td C':>6}{'Wind':>8}{'Gst':>5}"
                 f"{'MSLP':>7}{'Cld%':>6}{'Vis':>6}{'Ceil ft':>9}"]
        gusts = []
        for r in rows:
            gk = _kt(r.get("gust"))
            if gk is not None:
                gusts.append(gk)
            t, td, mslp, cld = _c(r.get("t2m")), _c(r.get("td2m")), r.get("mslp"), r.get("tcdc")
            lines.append(
                f"{r['forecasted_time']:%Y-%m-%dT%HZ}"
                f"{('%5.0f' % t) if t is not None else '   --'}"
                f"{('%6.0f' % td) if td is not None else '    --'}"
                f"{_wind_cell(r, model):>8}"
                f"{('%5.0f' % gk) if gk is not None else '   --'}"
                f"{('%7.0f' % (mslp / 100)) if mslp is not None else '     --'}"
                f"{('%6.0f' % cld) if cld is not None else '    --'}"
                f"{_vis_sm(r.get('vis')):>6}"
                f"{_ceil_ft(r.get('ceil')):>9}"
            )
        blocks.append("\n".join(lines))
        peaks[model] = max(gusts) if gusts else None
    synopsis = "  ".join(f"{m.upper()} peak gust {v:.0f}kt" if v else f"{m.upper()} gust --"
                         for m, v in peaks.items())
    return "\n\n".join(blocks) + f"\n\nCROSS-MODEL: {synopsis}"


# --- Capability 2: cross-model hazard confirmation (ICING + TURBULENCE) ----------------
_ICE_LEVELS = ["650 mb", "600 mb", "550 mb", "500 mb", "450 mb", "400 mb"]
_SHEAR_LEVELS = ["850 mb", "500 mb", "300 mb"]
_VVEL_LEVELS = ["700 mb", "500 mb", "300 mb"]


def _ice_vars(model):
    v = []
    for lv in _ICE_LEVELS:
        p = lv[:3]
        v += [gribstream.Var("TMP", lv, f"t{p}"), gribstream.Var("RH", lv, f"rh{p}")]
        if model == "gfs":
            v.append(gribstream.Var("CLMR", lv, f"clw{p}"))
    return v


def _turb_vars(model):
    v = []
    for lv in _SHEAR_LEVELS:
        p = lv[:3]
        v += [gribstream.Var("UGRD", lv, f"u{p}"), gribstream.Var("VGRD", lv, f"v{p}")]
    for lv in _VVEL_LEVELS:
        v.append(gribstream.Var("VVEL", lv, f"w{lv[:3]}"))
    if model == "gfs":
        v += [gribstream.Var("CAPE", "surface", "cape"), gribstream.Var("CIN", "surface", "cin"),
              gribstream.Var("HLCY", "3000-0 m above ground", "hlcy")]
    else:
        v += [gribstream.Var("CAPE", "180-0 mb above ground", "cape"),
              gribstream.Var("CIN", "180-0 mb above ground", "cin")]
    return v


def cap2(lat, lon):
    valid = _floor(_utcnow()) + timedelta(hours=3)
    win = dict(from_time=valid - timedelta(minutes=10), until_time=valid + timedelta(minutes=10))
    out = [f"Hazard scan for {STATION}, valid {valid:%Y-%m-%dT%HZ} -- conditions diagnosed "
           "from GFS + HRRR (no native icing/turbulence field exists; we confirm the "
           "ENVIRONMENT across models).", ""]

    # ICING: supercooled water (T in [-16,0] C, RH high; GFS cloud-liquid CLMR confirms)
    out.append("ICING (supercooled liquid: T in -16..0 C with RH>=70%, GFS CLMR>0 confirms):")
    ice = {}
    for model in ("gfs", "hrrr"):
        try:
            ts = _fetch(model, lat, lon, _ice_vars(model), **win)
        except ValueError as e:
            out.append(f"  {model.upper()}: unavailable ({e})")
            continue
        if not ts.rows:
            out.append(f"  {model.upper()}: no data at valid time")
            continue
        r = ts.rows[0]
        out.append(f"  {model.upper()} (run {ts.runs[0]:%Y-%m-%dT%HZ}):")
        for lv in _ICE_LEVELS:
            p = lv[:3]
            t, rh = _c(r.get(f"t{p}")), r.get(f"rh{p}")
            if t is None or rh is None:
                continue
            clw = r.get(f"clw{p}")
            flag = (-16.0 <= t <= 0.0) and rh >= 70.0
            clw_s = f" CLW={clw*1000:.2f}g/kg" if clw is not None else ""
            tag = "ICING" if flag else "-"
            ice.setdefault(lv, {})[model] = flag
            out.append(f"    {lv:<7} T={t:>5.1f}C RH={rh:>3.0f}%{clw_s:<16} {tag}")
    out.append("  agreement: " + "; ".join(
        f"{lv} " + ("BOTH icing" if set(v.values()) == {True}
                    else "no icing" if set(v.values()) == {False} else f"DISAGREE {v}")
        for lv, v in ice.items() if v))

    # TURBULENCE: convective (CAPE + ascent) and shear-driven (deep-layer bulk shear)
    out += ["", "TURBULENCE (convective: CAPE + strong ascent; mechanical/CAT: bulk shear):"]
    summ = {}
    for model in ("gfs", "hrrr"):
        try:
            ts = _fetch(model, lat, lon, _turb_vars(model), **win)
        except ValueError as e:
            out.append(f"  {model.upper()}: unavailable ({e})")
            continue
        if not ts.rows:
            continue
        r = ts.rows[0]
        cape, cin = r.get("cape"), r.get("cin")
        w = {lv[:3]: r.get(f"w{lv[:3]}") for lv in _VVEL_LEVELS}
        max_up = min((x for x in w.values() if x is not None), default=None)  # omega<0 = up
        def _v(p): return (r.get(f"u{p}"), r.get(f"v{p}"))                    # noqa: E306,E704
        (u8, v8), (u3, v3) = _v("850"), _v("300")
        deep = (_kt(math.hypot(u3 - u8, v3 - v8))
                if None not in (u8, v8, u3, v3) else None)
        summ[model] = (cape, deep)
        parts = [f"CAPE={cape:.0f}J/kg" if cape is not None else "CAPE=--",
                 f"CIN={cin:.0f}" if cin is not None else "CIN=--",
                 f"max ascent(omega)={max_up:.1f}Pa/s" if max_up is not None else "omega=--",
                 f"850-300mb shear={deep:.0f}kt" if deep is not None else "shear=--"]
        if model == "gfs" and r.get("hlcy") is not None:
            parts.append(f"SRH(0-3km)={r['hlcy']:.0f}m2/s2")
        out.append(f"  {model.upper()} (run {ts.runs[0]:%Y-%m-%dT%HZ}): " + ", ".join(parts))
    if len(summ) == 2:
        cg, ch = summ["gfs"][0], summ["hrrr"][0]
        sg, sh = summ["gfs"][1], summ["hrrr"][1]
        conv = "BOTH show convective potential" if (cg or 0) > 500 and (ch or 0) > 500 else \
               "single-model convective signal" if (cg or 0) > 500 or (ch or 0) > 500 else \
               "neither model convective"
        shr = "deep shear >40kt in both (organized/CAT)" if (sg or 0) > 40 and (sh or 0) > 40 else \
              "modest shear"
        out.append(f"  agreement: {conv}; {shr}")
    return "\n".join(out)


# --- Capability 3: model-vs-obs verification history ----------------------------------
_VER_GFS = [gribstream.Var("TMP", "2 m above ground", "t2m"),
            gribstream.Var("DPT", "2 m above ground", "td2m"),
            gribstream.Var("UGRD", "10 m above ground", "u10"),
            gribstream.Var("VGRD", "10 m above ground", "v10")]
_VER_NBM = [gribstream.Var("TMP", "2 m above ground", "t2m"),
            gribstream.Var("DPT", "2 m above ground", "td2m"),
            gribstream.Var("WIND", "10 m above ground", "wind"),
            gribstream.Var("WDIR", "10 m above ground", "wdir")]
_VER = {"gfs": _VER_GFS, "hrrr": _VER_GFS, "nbm": _VER_NBM}


def cap3(lat, lon):
    now = _floor(_utcnow())
    as_of = now - timedelta(hours=12)
    win = dict(from_time=now - timedelta(hours=7), until_time=now - timedelta(hours=1), as_of=as_of)
    # observed truth from METARs, keyed by nearest whole hour
    obs = {}
    for ot, raw, _ in awc.fetch_metar(STATION, hours=8):
        o = metar.parse(raw)
        obs[_floor(ot.replace(tzinfo=None) + timedelta(minutes=30))] = o
    out = [f"Model-vs-obs verification for {STATION} -- past runs (asOf {as_of:%Y-%m-%dT%HZ}) "
           "vs observed METARs. Positive T/Td error = model too warm/moist.", ""]
    for model in ("gfs", "hrrr", "nbm"):
        try:
            ts = _fetch(model, lat, lon, _VER[model], **win)
        except ValueError as e:
            out.append(f"{model.upper()}: unavailable ({e})\n")
            continue
        rows = sorted(ts.rows, key=lambda r: r["forecasted_time"])
        out.append(f"{model.upper()} (run {ts.runs[0]:%Y-%m-%dT%HZ}):")
        out.append(f"  {'Valid (Z)':<15}{'T f/o':>10}{'Terr':>6}{'Td f/o':>10}{'Tderr':>7}")
        terrs = []
        for r in rows:
            o = obs.get(r["forecasted_time"])
            tf = _c(r.get("t2m"))
            tdf = _c(r.get("td2m"))
            to = o.temp_c if o else None
            tdo = o.dewpoint_c if o else None
            te = f"{tf - to:+.1f}" if tf is not None and to is not None else "--"
            tde = f"{tdf - tdo:+.1f}" if tdf is not None and tdo is not None else "--"
            if tf is not None and to is not None:
                terrs.append(tf - to)
            tfo = f"{tf:.0f}/{to}" if tf is not None and to is not None else "--"
            tdfo = f"{tdf:.0f}/{tdo}" if tdf is not None and tdo is not None else "--"
            out.append(f"  {r['forecasted_time']:%Y-%m-%dT%HZ}{tfo:>10}{te:>6}{tdfo:>10}{tde:>7}")
        if terrs:
            bias = sum(terrs) / len(terrs)
            out.append(f"  -> mean T bias {bias:+.1f}C over {len(terrs)} hrs")
        out.append("")
    return "\n".join(out)


def main():
    lat, lon = awc.station_latlon(STATION)
    p1 = cap1(lat, lon)
    p2 = cap2(lat, lon)
    p3 = cap3(lat, lon)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        f"# GRIBStream full-product demo -- {STATION} ({lat:.3f}, {lon:.3f})\n\n"
        f"## [1] Multi-model current/forecast state\n```\n{p1}\n```\n\n"
        f"## [2] Cross-model hazard confirmation (icing + turbulence)\n```\n{p2}\n```\n\n"
        f"## [3] Model-vs-obs verification history\n```\n{p3}\n```\n\n"
        f"_credits billed this run: {_charged}_\n"
    )
    print(p1); print("\n" + "=" * 80 + "\n")     # noqa: E702
    print(p2); print("\n" + "=" * 80 + "\n")      # noqa: E702
    print(p3)
    print(f"\ncredits billed this run: {_charged}   (written to {OUT})")


if __name__ == "__main__":
    main()
