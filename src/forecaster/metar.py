import re
from datetime import time
from pathlib import Path

from pydantic import BaseModel
from metar_taf_parser.parser.parser import MetarParser

_PARSER = MetarParser()
_PREFIX = re.compile(r"^\s*(METAR|SPECI)\s+")     # library chokes on these; we capture which
_ALT_INHG = re.compile(r"\bA(\d{4})\b")           # US altimeter group, e.g. A2990
_ALT_HPA = re.compile(r"\bQ(\d{4})\b")            # international, e.g. Q1013
# US visibility token, incl. M/P (less/greater-than) and fractions: M1/4SM, 1 1/2SM, P6SM
_VIS_SM = re.compile(r"\b([MP]?(?:\d{1,2} \d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}|\d{1,3})SM)\b")
_HPA_PER_INHG = 33.8638866667
_SKY_CLEAR = {"CLR", "SKC", "NSC", "NCD"}         # not real cloud layers
# A showers group with no phenomenon (VCSH, bare SH) -- the library silently
# drops these (it keeps VCTS/VCFG/-SHRA). We re-attach from the raw body.
_SH_ONLY = re.compile(r"^(?:VC|[+-])?SH$")
# Table 8.1 — reportable visibility, SM <-> meters. A LOOKUP, not physics
# (1/2SM == 800m, not 804m): the values forecasters actually report. M1/8 folds
# into the 0.125 row. 9999m ("10 km or more") is the OCONUS max -> treated as >6SM.
_VIS_TABLE: list[tuple[float, int]] = [
    (0.0, 0), (0.0625, 100), (0.125, 200), (0.1875, 300), (0.25, 400),
    (0.3125, 500), (0.375, 600), (0.5, 800), (0.625, 1000), (0.75, 1200),
    (0.875, 1400), (1.0, 1600), (1.125, 1800), (1.25, 2000), (1.375, 2200),
    (1.5, 2400), (1.625, 2600), (1.75, 2800), (1.875, 3000), (2.0, 3200),
    (2.25, 3600), (2.5, 4000), (2.75, 4400), (3.0, 4800), (4.0, 6000),
    (5.0, 8000), (6.0, 9000),
]


class CloudLayer(BaseModel):
    cover: str                 # BKN, OVC, SCT, FEW
    height_ft: int | None      # feet AGL
    type: str | None           # CB, TCU, or None


class MetarObs(BaseModel):
    """One parsed surface observation. This is OURS — the library object never
    escapes parse(); everything downstream (render, the DuckDB obs table) keys
    off these fields."""

    station: str
    day: int                   # day-of-month (a METAR alone has no month/year)
    time: time                 # UTC
    report_type: str | None    # 'METAR' (routine) or 'SPECI' (weather forced an off-cycle ob)
    auto: bool                 # AUTO — automated station, no human augmentation
    cavok: bool                # CAVOK — vis ≥10 km, no sig cloud/weather
    wind_dir_deg: int | None
    wind_dir_card: str | None
    wind_speed: int | None
    wind_gust: int | None
    wind_unit: str | None
    visibility: str | None     # reported string; AF works in statute miles
    vis_sm: float | None       # numeric statute miles (Table 8.1 lookup)
    vis_m: int | None          # numeric meters (Table 8.1 lookup)
    vis_flag: str | None       # 'M' (less-than), 'P' (greater-than), or None (exact)
    weather: list[str]         # present-weather groups, e.g. ["+RA", "BR"], "VCTS"
    temp_c: int | None
    dewpoint_c: int | None
    altimeter_inhg: float | None   # reported value, taken exact from the raw token
    altimeter_hpa: float | None    # derived from inHg (US METARs only report inHg)
    clouds: list[CloudLayer]
    vertical_visibility_ft: int | None   # VV — indefinite ceiling (fog); None if absent
    ceiling_ft: int | None               # derived: lowest BKN/OVC or VV; None = unlimited
    remarks: str | None        # the RMK section, verbatim
    raw: str                   # original line, untouched


def _sm_to_float(core: str) -> float | None:
    """'10' -> 10.0 | '1/4' -> 0.25 | '1 3/4' -> 1.75"""
    try:
        if " " in core:
            whole, frac = core.split()
            n, d = frac.split("/")
            return int(whole) + int(n) / int(d)
        if "/" in core:
            n, d = core.split("/")
            return int(n) / int(d)
        return float(core)
    except ValueError:
        return None


def _nearest(value: float, key: int, out: int) -> float:
    """Closest Table 8.1 row; ties resolve to the lower visibility (pessimistic)."""
    return min(_VIS_TABLE, key=lambda r: (abs(r[key] - value), r[out]))[out]


def _parse_vis(s: str | None) -> tuple[float | None, int | None, str | None]:
    """(vis_sm, vis_m, vis_flag) — flag is 'M' (<), 'P' (>), or None (exact)."""
    if not s:
        return None, None, None
    t = s.strip()
    if t.upper().endswith("SM"):                          # CONUS — statute miles
        flag = "M" if t[0] == "M" else "P" if t[0] == "P" else None
        sm = _sm_to_float(t[:-2].lstrip("MP"))
        if sm is None:
            return None, None, flag
        m = 9999 if sm > 6 or (flag == "P" and sm >= 6) else int(_nearest(sm, 0, 1))
        return sm, m, flag
    flag = None                                           # OCONUS — meters
    if t[0] in ">P":
        flag, t = "P", t[1:]
    elif t[0] == "M":
        flag, t = "M", t[1:]
    try:
        m = int(t)
    except ValueError:
        return None, None, flag
    if m >= 9999:                       # 9999m = "10 km or more": OCONUS max, i.e. >6 SM
        return 6.0, 9999, "P"
    return _nearest(m, 1, 0), m, flag


def parse(line: str) -> MetarObs:
    raw = line.strip().rstrip("=").strip()
    m = _PARSER.parse(_PREFIX.sub("", raw))        # parse a cleaned copy, keep raw verbatim

    # Report type from the leading keyword, when the source kept it (AWC/Skyvector
    # pastes do; IEM strips it -> None here, and the loader supplies the real tag).
    pm = _PREFIX.match(raw)
    report_type = pm.group(1) if pm else None

    # Pressure: read inHg straight from the raw token rather than trusting the
    # library's lossy integer-hPa conversion. US METARs report inHg only.
    inhg = hpa = None
    if a := _ALT_INHG.search(raw):
        inhg = int(a.group(1)) / 100
        hpa = round(inhg * _HPA_PER_INHG, 1)
    elif q := _ALT_HPA.search(raw):
        hpa = float(int(q.group(1)))
        inhg = round(hpa / _HPA_PER_INHG, 2)

    # Visibility: keep the reported SM token (with M/P) from raw; the library
    # strips the unit. Fall back to its value for non-US (meters) reports.
    if v := _VIS_SM.search(raw):
        visibility = v.group(1)
    elif m.visibility:
        # The library renders 9999 ('10 km or more') as '>10000', which is never
        # how it's reported; show the actual token (9999 = unrestricted).
        visibility = "9999" if m.visibility.distance == ">10000" else m.visibility.distance
    else:
        visibility = None

    # Present weather: rebuild each group's code from the parsed enums, whose
    # .value IS the METAR token (intensity + descriptive + phenomenons).
    weather = [
        (wc.intensity.value if wc.intensity else "")
        + (wc.descriptive.value if wc.descriptive else "")
        + "".join(p.value for p in wc.phenomenons)
        for wc in m.weather_conditions
    ]
    # Re-attach a showers group the library drops (VCSH / bare SH); standard
    # present weather. Scan the body only (pre-RMK) so RMK text can't false-match.
    for tok in raw.split(" RMK ", 1)[0].split():
        if _SH_ONLY.match(tok) and tok not in weather:
            weather.append(tok)

    auto = bool(re.search(r"\bAUTO\b", raw))
    cavok = bool(re.search(r"\bCAVOK\b", raw))
    vv = m.vertical_visibility            # ft AGL, or None

    vis_sm, vis_m, vis_flag = _parse_vis(visibility)
    if cavok and visibility is None:      # CAVOK implies vis >=10 km -> treat as >6 SM
        vis_sm, vis_m, vis_flag = 6.0, 9999, "P"

    clouds = [
        CloudLayer(
            cover=c.quantity.name,
            height_ft=c.height,
            type=c.type.name if c.type else None,
        )
        for c in m.clouds
        if c.quantity.name not in _SKY_CLEAR
    ]
    # Ceiling = lowest broken/overcast layer, or an indefinite (VV) ceiling
    # — VV counts as a ceiling per AFMAN 15-111, 11.4.4.6.
    bases = [c.height_ft for c in clouds if c.cover in ("BKN", "OVC") and c.height_ft is not None]
    if vv is not None:
        bases.append(vv)
    ceiling_ft = min(bases) if bases else None

    remarks = raw.split(" RMK ", 1)[1] if " RMK " in raw else None
    w = m.wind
    return MetarObs(
        station=m.station,
        day=m.day,
        time=m.time,
        report_type=report_type,
        auto=auto,
        cavok=cavok,
        wind_dir_deg=w.degrees if w else None,
        wind_dir_card=w.direction if w else None,
        wind_speed=w.speed if w else None,
        wind_gust=w.gust if w else None,
        wind_unit=w.unit if w else None,
        visibility=visibility,
        vis_sm=vis_sm,
        vis_m=vis_m,
        vis_flag=vis_flag,
        weather=weather,
        temp_c=m.temperature,
        dewpoint_c=m.dew_point,
        altimeter_inhg=inhg,
        altimeter_hpa=hpa,
        clouds=clouds,
        vertical_visibility_ft=vv,
        ceiling_ft=ceiling_ft,
        remarks=remarks,
        raw=raw,
    )


def parse_file(path: str | Path) -> list[MetarObs]:
    return [
        parse(ln)
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def _t(v: int | None) -> str:
    return "—" if v is None else str(v)


def _wind(o: MetarObs) -> str:
    if o.wind_speed is None:
        return "—"
    if o.wind_speed == 0:
        return "calm"
    d = f"{o.wind_dir_deg:03d}" if o.wind_dir_deg is not None else (o.wind_dir_card or "VRB")
    g = f"G{o.wind_gust}" if o.wind_gust else ""
    return f"{d}/{o.wind_speed}{g}{o.wind_unit}"


def _sky(clouds: list[CloudLayer]) -> str:
    if not clouds:
        return "CLR"
    return " ".join(
        c.cover + (f"{c.height_ft}ft" if c.height_ft is not None else "") + (c.type or "")
        for c in clouds
    )


def render(obs: list[MetarObs]) -> str:
    """Decoded, chronological (oldest first) view for the messages array, with the
    raw strings underneath so nothing the decoder dropped is lost to the model.

    Fixed-width columns come first so trends scan down cleanly; the variable-width
    present-weather and sky trail at the end of each row to keep the grid aligned."""
    obs = sorted(obs, key=lambda o: (o.day, o.time))
    if not obs:
        return "(no observations)"
    out = [
        f"{obs[0].station} — {len(obs)} observations (UTC, oldest first)",
        f"  {'DD HHMMZ':<8}  {'wind':<12} {'vis':<8} {'T/Td°C':<8} {'altimeter':<19} "
        "present-wx + sky(ft AGL)",
    ]
    for o in obs:
        t = f"{_t(o.temp_c)}/{_t(o.dewpoint_c)}"
        alt = (
            f"{o.altimeter_inhg:.2f}inHg({o.altimeter_hpa:.0f}hPa)"
            if o.altimeter_inhg
            else "—"
        )
        wx = " ".join(o.weather)
        tail = f"{wx}  {_sky(o.clouds)}" if wx else _sky(o.clouds)
        out.append(
            f"  {o.day:02d} {o.time.strftime('%H%MZ')}  "
            f"{_wind(o):<12} {(o.visibility or '—'):<8} {t:<8} {alt:<19} {tail}"
        )
    out += ["", "Raw:"] + [f"  {o.raw}" for o in obs]
    return "\n".join(out)
