import re
from datetime import time
from pathlib import Path

from pydantic import BaseModel
from metar_taf_parser.parser.parser import MetarParser

_PARSER = MetarParser()
_PREFIX = re.compile(r"^\s*(?:METAR|SPECI)\s+")   # library chokes on these keywords
_ALT_INHG = re.compile(r"\bA(\d{4})\b")           # US altimeter group, e.g. A2990
_ALT_HPA = re.compile(r"\bQ(\d{4})\b")            # international, e.g. Q1013
# US visibility token, incl. M/P (less/greater-than) and fractions: M1/4SM, 1 1/2SM, P6SM
_VIS_SM = re.compile(r"\b([MP]?(?:\d{1,2} \d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}|\d{1,3})SM)\b")
_HPA_PER_INHG = 33.8638866667
_SKY_CLEAR = {"CLR", "SKC", "NSC", "NCD"}         # not real cloud layers


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
    wind_dir_deg: int | None
    wind_dir_card: str | None
    wind_speed: int | None
    wind_gust: int | None
    wind_unit: str | None
    visibility: str | None     # reported string; AF works in statute miles
    weather: list[str]         # present-weather groups, e.g. ["+RA", "BR"], "VCTS"
    temp_c: int | None
    dewpoint_c: int | None
    altimeter_inhg: float | None   # reported value, taken exact from the raw token
    altimeter_hpa: float | None    # derived from inHg (US METARs only report inHg)
    clouds: list[CloudLayer]
    remarks: str | None        # the RMK section, verbatim
    raw: str                   # original line, untouched


def parse(line: str) -> MetarObs:
    raw = line.strip().rstrip("=").strip()
    m = _PARSER.parse(_PREFIX.sub("", raw))        # parse a cleaned copy, keep raw verbatim

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
        visibility = m.visibility.distance
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

    remarks = raw.split(" RMK ", 1)[1] if " RMK " in raw else None
    w = m.wind
    return MetarObs(
        station=m.station,
        day=m.day,
        time=m.time,
        wind_dir_deg=w.degrees if w else None,
        wind_dir_card=w.direction if w else None,
        wind_speed=w.speed if w else None,
        wind_gust=w.gust if w else None,
        wind_unit=w.unit if w else None,
        visibility=visibility,
        weather=weather,
        temp_c=m.temperature,
        dewpoint_c=m.dew_point,
        altimeter_inhg=inhg,
        altimeter_hpa=hpa,
        clouds=[
            CloudLayer(
                cover=c.quantity.name,
                height_ft=c.height,
                type=c.type.name if c.type else None,
            )
            for c in m.clouds
            if c.quantity.name not in _SKY_CLEAR
        ],
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
        "cols: DD HHMMZ | wind | vis | T/Td(°C) | altimeter | present-wx + sky(ft AGL)",
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
