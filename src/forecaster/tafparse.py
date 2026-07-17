"""TAF text <-> typed TafObs (the forecast input seam, symmetric to metar.py).

A TAF is recursive: a header + a PREVAILING period (wind/vis/wx/sky over the
whole validity) + a list of CHANGE groups (FM / BECMG / TEMPO, optionally
PROBxx), each with the very same period shape plus its own validity. So one
TafGroup models both, and a TafObs is a header + prevailing + groups.

Like metar.py: the library object never escapes parse(); everything downstream
keys off OUR fields, and the raw line is retained verbatim. The standard ICAO/US
grammar (incl. US-SM vs intl-meters visibility, CAVOK, PROBxx, TX/TN) is fully
handled by the library; the only military tokens it drops are QNHxxxxINS (read
exact from raw here, like METAR pressure) and NSW (survives in the raw line).
"""

import re
from datetime import datetime, time

from pydantic import BaseModel
from metar_taf_parser.parser.parser import TAFParser

from forecaster.metar import CloudLayer, _parse_vis, _SH_ONLY

_PARSER = TAFParser()
_QNH = re.compile(r"\bQNH(\d{4})INS\b")           # US-military altimeter forecast; library drops it
_SKY_CLEAR = {"CLR", "SKC", "NSC", "NCD"}         # not real cloud layers
# Change-group boundaries, in raw order. PROBxx greedily absorbs a following
# TEMPO/BECMG so 'PROB30 TEMPO' is ONE boundary (the library merges it to one trend).
_BOUNDARY = re.compile(r"\b(?:PROB\d{2}(?:\s+(?:TEMPO|BECMG))?|FM\d{6}|BECMG|TEMPO)\b")
# Same boundaries plus the TX/TN group, for re-inserting conventional TAF line
# breaks in the raw view (the json feed delivers one flat line).
_REFLOW = re.compile(r"\b(?:PROB\d{2}(?:\s+(?:TEMPO|BECMG))?|FM\d{6}|BECMG|TEMPO|TX\d{2}/)")
# AF total-obscuration + hazard groups the library drops or ignores; recovered
# per-group from the raw chunk (like QNH/NSW). Heights are hundreds of feet; the
# bare 6-/5-digit icing/turbulence tokens are safe because nothing else in a TAF
# body is a 6-digit run starting 6/5 with word boundaries (validity has a slash,
# winds end in KT, temps carry TX/TN, AF remark times are 4-digit).
_VV = re.compile(r"\bVV(\d{3})\b")                          # total obscuration (1.3.7.2)
_WS = re.compile(r"\bWS(\d{3})/(\d{3})(\d{2,3})KT\b")       # low-level wind shear (1.3.9.2)
_VA_GRP = re.compile(r"\bVA(\d{3})(\d{3})\b")               # volcanic-ash plume VAbbbttt (1.3.9.1)
_ICING = re.compile(r"\b6(\d)(\d{3})(\d)\b")                # icing 6IchihihitL (1.3.10)
_TURB = re.compile(r"\b5([\dX])(\d{3})(\d)\b")              # turbulence 5BhBhBhBtL (1.3.11)

# Explicit-vs-omitted field detection (scoring-design sec 5.1). A change group omits
# a field to CARRY IT FORWARD; the parser otherwise returns empty collections, which
# are ambiguous with an explicit SKC/NSW. We record which fields each group's raw
# chunk actually STATED so the scoring resolver can tell "inherit" from an explicit
# restatement (omitted sky vs SKC, omitted wx vs NSW, calm vs omitted wind, QNH absent
# on TEMPO). Per-token full-match on the whitespace-split chunk.
_TOK_WIND = re.compile(r"(?:VRB|\d{3})\d{2,3}(?:G\d{2,3})?(?:KT|MPS)")   # incl 00000KT (calm)
_TOK_METERS = re.compile(r"[MP]?\d{4}")                     # meters vis 0000-9999
_TOK_SKY = re.compile(r"(?:FEW|SCT|BKN|OVC)\d{3}(?:CB|TCU)?")
_TOK_QNH = re.compile(r"QNH\d{4}INS")
_WX_DESC = "MI|PR|BC|DR|BL|SH|TS|FZ"                        # WMO present-weather grammar
_WX_PHEN = "DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS"
_TOK_WX = re.compile(rf"(?:[+-]|VC)?(?:(?:{_WX_DESC})(?:{_WX_PHEN})*|(?:{_WX_PHEN})+)")
# The field keys explicit_fields can hold (the resolver's overlay vocabulary).
EXPLICIT_FIELD_KEYS = frozenset({"wind", "gust", "visibility", "weather", "sky", "qnh"})

# AF TAF remarks (AFMAN 15-124). They have NO delimiter, so the library folds them into
# the last group -- corrupting it (and, for WND ... AFT, losing a real wind forecast). The
# scoring path strips them into parse_body before parsing. Extend from doctrine, not
# imagination: anything unrecognized is LEFT in the body (never guessed away).
#   WND ... AFT DDHH can appear ANYWHERE (observed mid-body), so it is grabbed globally;
#   the metwatch/amend-service remarks sit at the TAIL and are stripped iteratively.
_RMK_WND_AFT = re.compile(
    r"\bWND\s+(?:VRB\d{2,3}|\d{3}\d{2,3}(?:G\d{2,3})?)KT\s+AFTR?\s+\d{4}(?:\d{2})?\b")
_RMK_TAIL = (
    re.compile(r"\s+LAST\s+NO\s+AMDS(?:\s+AFT\s+\d{4})?(?:\s+NEXT\s+\d{4})?\s*$"),
    re.compile(r"\s+AMD\s+(?:LTD\s+TO|NOT\s+SKED)\b.*$"),
    re.compile(r"\s+LIMITED\s+METWATCH\b.*$"),
)


def strip_remarks(raw: str) -> tuple[str, str]:
    """Split an AF TAF into (body, remarks). Conservative: pull the WND ... AFT wind remark
    from wherever it sits, then iteratively strip trailing metwatch/amend-service remarks;
    everything unrecognized stays in the body. `body` is safe to parse(); `remarks` feeds
    tafstate.parse_wind_after. A civil TAF with no remarks returns (raw, "")."""
    body = raw.strip().rstrip("=").strip()
    removed: list[str] = []
    body = _RMK_WND_AFT.sub(lambda m: removed.append(m.group(0)) or " ", body)
    changed = True
    while changed:
        changed = False
        for pat in _RMK_TAIL:
            if m := pat.search(body):
                removed.append(m.group(0).strip())
                body = body[:m.start()].rstrip()
                changed = True
    body = re.sub(r"\s{2,}", " ", body).strip()
    return body, " ".join(r.strip() for r in removed)


class TafTemp(BaseModel):
    temp_c: int                # forecast max (TX) or min (TN) temperature
    day: int                   # day-of-month it occurs
    hour: int                  # UTC hour it occurs


class WindOverride(BaseModel):
    """A 'WND <wind> AFT DDHH' human-TAF remark (AFMAN 15-124): the forecaster's actual
    wind forecast for hours at/after `after`. Wind ONLY (dir/speed/gust). Scored as a full
    prevailing overlay from its own time forward (T5). Data-only here (no absolutization);
    tafstate.parse_wind_after builds these with the absolute `after` datetime."""

    after: datetime            # absolute naive UTC from which this wind applies
    wind_dir: int | None = None    # None when variable
    is_vrb: bool = False       # the wind group was VRB
    wind_speed: int
    wind_gust: int | None = None


# AF hazard groups (AFMAN 15-124 1.3.9-1.3.11). Defined here on the PARSE side
# (the seam that owns the typed representation, like metar.CloudLayer); tafgen
# imports these for the OUTPUT side, so generation and parsing share one shape.

class WindShear(BaseModel):
    """Non-convective low-level wind shear WShxhxhx/dddfffKT (1.3.9.2). Surface to
    2000ft AGL; VRB not allowed for the shear direction."""

    height_ft: int             # hxhxhx -- base of the shear, hundreds of feet
    wind_dir: int              # ddd -- tens of degrees true (no VRB)
    wind_speed: int            # ff -- knots above the shear


class VolcanicAsh(BaseModel):
    """Volcanic-ash plume VAbbbttt (1.3.9.1). Surface-based when base_ft == 0, in
    which case VA is also encoded as present weather (1.3.9.1.3)."""

    base_ft: int               # bbb -- base of ash, hundreds of feet AGL
    top_ft: int                # ttt -- top of ash, hundreds of feet AGL


class IcingLayer(BaseModel):
    """Forecast icing 6IchihihitL (1.3.10), repeatable. Ic is the Table 1.5 code:
      0 trace            1 light (mixed)    2 light in cloud (rime)
      3 light in precip  4 mod (mixed)      5 mod in cloud (rime)
      6 mod in precip    7 severe (mixed)   8 severe in cloud (rime)
      9 severe in precip (clear)
    """

    ic: int                    # Table 1.5 type/intensity code figure 0-9
    base_ft: int               # hihihi -- base AGL (Table 1.4)
    thickness_ft: int          # tL -- layer thickness 1000-9000ft (Table 1.6)


class TurbulenceLayer(BaseModel):
    """Forecast turbulence 5BhBhBhBtL (1.3.11), repeatable. B is the Table 1.7 code:
      0 none             1 light            2 mod clear-air, occasional
      3 mod clear-air, frequent             4 mod in cloud, occasional
      5 mod in cloud, frequent              6 severe clear-air, occasional
      7 severe clear-air, frequent          8 severe in cloud, occasional
      9 severe in cloud, frequent           X extreme
    """

    b: int | str               # Table 1.7 code figure 0-9 or 'X' (extreme)
    base_ft: int               # hBhBhB -- base AGL (Table 1.4)
    thickness_ft: int          # tL -- layer thickness 1000-9000ft (Table 1.6)


class TafGroup(BaseModel):
    """One forecast period. change_type=None is the PREVAILING group; otherwise
    FM / BECMG / TEMPO. probability is set for PROB30/PROB40 groups."""

    change_type: str | None    # None=prevailing, else 'FM' | 'BECMG' | 'TEMPO'
    probability: int | None    # 30 or 40 for PROBxx groups, else None
    from_day: int | None       # period start (day/hour); FM also carries minutes
    from_hour: int | None
    from_minute: int | None
    to_day: int | None         # period end; None for FM (runs until the next group)
    to_hour: int | None
    wind_dir_deg: int | None
    wind_dir_card: str | None  # cardinal, or 'VRB'
    wind_speed: int | None
    wind_gust: int | None
    wind_unit: str | None      # KT, or MPS (some intl stations)
    visibility: str | None     # reported token, e.g. 'P6SM', '9999m', '>10000m'
    vis_sm: float | None       # numeric statute miles (metar Table 8.1 lookup)
    vis_m: int | None          # numeric meters
    vis_flag: str | None       # 'M' (<), 'P' (>), or None (exact)
    cavok: bool
    weather: list[str]         # forecast present-weather groups, e.g. ['-SHRA', 'VCTS']
    clouds: list[CloudLayer]
    qnh_inhg: float | None = None   # per-group QNHxxxxINS (US military); None for civil TAFs
    vert_vis_ft: int | None = None  # VVhshshs total obscuration; library drops it, recovered from raw
    wind_shear: "WindShear | None" = None      # AF hazard groups; library ignores them, recovered from raw
    volcanic_ash: "VolcanicAsh | None" = None
    icing: list[IcingLayer] = []
    turbulence: list[TurbulenceLayer] = []
    # Fields this group's raw chunk EXPLICITLY stated (scoring-design sec 5.1); a
    # subset of EXPLICIT_FIELD_KEYS. Empty on a change group means "everything
    # inherited"; empty when chunk alignment failed means "unknown" (parse() only
    # populates it where _split_periods aligns 1:1). Not part of round-trip/render.
    explicit_fields: set[str] = set()


class TafObs(BaseModel):
    """One parsed terminal aerodrome forecast. OURS — the library object never
    escapes parse()."""

    station: str
    issue_day: int             # day-of-month the TAF was issued
    issue_time: time           # UTC issue time
    valid_from_day: int | None
    valid_from_hour: int | None
    valid_to_day: int | None
    valid_to_hour: int | None
    amendment: bool            # AMD
    corrected: bool            # COR
    nil: bool                  # NIL (no forecast)
    canceled: bool             # CNL
    qnh_inhg: float | None     # initial QNHxxxxINS (US military); None for civil TAFs
    prevailing: TafGroup
    groups: list[TafGroup]     # change groups, in TAF order
    max_temp: TafTemp | None   # TX
    min_temp: TafTemp | None   # TN
    wind_overrides: list[WindOverride] = []   # 'WND ... AFT DDHH' remarks (set at score time)
    raw: str                   # original line, untouched


def _weather(conditions) -> list[str]:
    """Rebuild each group's METAR/TAF token from the parsed enums (.value is the
    raw code), exactly like metar.parse does."""
    return [
        (wc.intensity.value if wc.intensity else "")
        + (wc.descriptive.value if wc.descriptive else "")
        + "".join(p.value for p in wc.phenomenons)
        for wc in conditions
    ]


def _clouds(clouds) -> list[CloudLayer]:
    return [
        CloudLayer(
            cover=c.quantity.name,
            height_ft=c.height,
            type=c.type.name if c.type else None,
        )
        for c in clouds
        if c.quantity.name not in _SKY_CLEAR
    ]


def _vis_str(v) -> str | None:
    """Library Visibility -> the token as actually reported: US statute miles
    ('P6SM') or intl/military meters ('9999', '9000'). The library renders 9999 as
    '>10000', which is never how it's reported -- map it back to 9999 (unrestricted)."""
    if v is None:
        return None
    if str(v.unit).endswith("SM"):
        return f"{v.distance}SM"
    return "9999" if str(v.distance) == ">10000" else str(v.distance)


# Table 8.1 fractional statute-mile labels (the reportable rows below 1 SM and the
# eighths above), for formatting the SM equivalent shown beside meters vis.
_SM_FRACS = {
    0.0625: "1/16", 0.125: "1/8", 0.1875: "3/16", 0.25: "1/4", 0.3125: "5/16",
    0.375: "3/8", 0.5: "1/2", 0.625: "5/8", 0.75: "3/4", 0.875: "7/8",
}


def _fmt_sm(sm: float, flag: str | None) -> str:
    """Numeric SM -> a reported-style token: 6.0/'P' -> 'P6SM', 1.25 -> '1 1/4SM'."""
    pre = flag or ""
    whole = int(sm)
    rem = round(sm - whole, 4)
    if rem == 0:
        body = str(whole)
    elif (frac := _SM_FRACS.get(rem)) is None:
        body = f"{sm:g}"
    else:
        body = frac if whole == 0 else f"{whole} {frac}"
    return f"{pre}{body}SM"


def _vis_numeric(v) -> tuple[float | None, int | None, str | None]:
    """Numeric (sm, m, flag) via metar's Table 8.1 lookup -- reusing the single
    source of truth so forecast vis is directly comparable to observed METAR vis.
    Rebuilds the token metar._parse_vis expects: 'P6SM' for SM, the bare meters
    string (with any >/M prefix) otherwise."""
    if v is None:
        return None, None, None
    token = f"{v.distance}SM" if str(v.unit).endswith("SM") else str(v.distance)
    return _parse_vis(token)


def _validity(v) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    """(from_day, from_hour, from_minute, to_day, to_hour). FM groups carry a
    single start instant (incl. minutes) and no end; BECMG/TEMPO/overall carry a
    start..end window."""
    if v is None:
        return (None, None, None, None, None)
    if hasattr(v, "end_hour"):                 # Validity — a start..end window
        return (v.start_day, v.start_hour, 0, v.end_day, v.end_hour)
    # FMValidity. Newer library spells it start_minutes; older releases had the
    # typo strart_minutes -- read the correct name first, fall back for old versions.
    mins = getattr(v, "start_minutes", getattr(v, "strart_minutes", 0))
    return (v.start_day, v.start_hour, mins, None, None)


def _period(src, change_type: str | None, probability: int | None, valid) -> TafGroup:
    """Build a TafGroup from a library TAF (prevailing) or TAFTrend (change group)
    — both expose the same wind/visibility/clouds/weather_conditions/cavok shape."""
    fd, fh, fm, td, th = valid
    w = src.wind
    sm, m, flag = _vis_numeric(src.visibility)
    return TafGroup(
        change_type=change_type,
        probability=probability,
        from_day=fd, from_hour=fh, from_minute=fm, to_day=td, to_hour=th,
        wind_dir_deg=w.degrees if w else None,
        wind_dir_card=w.direction if w else None,
        wind_speed=w.speed if w else None,
        wind_gust=w.gust if w else None,
        wind_unit=w.unit if w else None,
        visibility=_vis_str(src.visibility),
        vis_sm=sm, vis_m=m, vis_flag=flag,
        cavok=bool(src.cavok),
        weather=_weather(src.weather_conditions),
        clouds=_clouds(src.clouds),
    )


def _split_periods(raw: str) -> list[str]:
    """Slice raw into chunks aligned with [prevailing, *trends], so a dropped
    token can be re-attributed to the group it belongs to. The first chunk (up to
    the first change-group keyword) is the prevailing conditions."""
    bounds = [m.start() for m in _BOUNDARY.finditer(raw)]
    if not bounds:
        return [raw]
    chunks = [raw[: bounds[0]]]
    for i, b in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else len(raw)
        chunks.append(raw[b:end])
    return chunks


def _recover_showers(chunk: str, weather: list[str]) -> None:
    """Append any bare-showers group (VCSH / SH) the library silently drops -- it
    is standard present weather, common in TAFs. Mutates `weather` in place."""
    for tok in chunk.split():
        if _SH_ONLY.match(tok) and tok not in weather:
            weather.append(tok)


def _explicit_fields(chunk: str) -> set[str]:
    """Which forecast fields the raw chunk EXPLICITLY states (sec 5.1). Whitespace-
    tokenize + classify each token; the change keyword, validity (has a slash) and
    header tokens match nothing. KNOWN limitation: meters-vis detection can false-
    positive on a 4-digit REMARK numeral (AF remarks have no delimiter), so the
    scoring path strips remarks into parse_body before parsing human TAFs."""
    fields: set[str] = set()
    for tok in chunk.split():
        if _TOK_WIND.fullmatch(tok):
            fields.add("wind")
            if "G" in tok:
                fields.add("gust")
        elif tok == "CAVOK":                        # implies clear vis + sky + weather
            fields |= {"visibility", "sky", "weather"}
        elif tok.endswith("SM") or _TOK_METERS.fullmatch(tok):
            fields.add("visibility")
        elif _TOK_SKY.fullmatch(tok) or tok in _SKY_CLEAR or _VV.fullmatch(tok):
            fields.add("sky")
        elif _TOK_QNH.fullmatch(tok):
            fields.add("qnh")
        elif tok == "NSW" or _TOK_WX.fullmatch(tok):
            fields.add("weather")
    return fields


def _recover_hazards(chunk: str, g: TafGroup) -> None:
    """Re-attach the AF total-obscuration + hazard groups the library drops/ignores
    (VV, WS, VA, icing, turbulence). Mutates `g` in place. Heights are hundreds of feet."""
    if (m := _VV.search(chunk)) and g.vert_vis_ft is None:
        g.vert_vis_ft = int(m.group(1)) * 100
    if (m := _WS.search(chunk)) and g.wind_shear is None:
        g.wind_shear = WindShear(
            height_ft=int(m.group(1)) * 100, wind_dir=int(m.group(2)), wind_speed=int(m.group(3))
        )
    if (m := _VA_GRP.search(chunk)) and g.volcanic_ash is None:
        g.volcanic_ash = VolcanicAsh(base_ft=int(m.group(1)) * 100, top_ft=int(m.group(2)) * 100)
    g.icing = [
        IcingLayer(ic=int(code), base_ft=int(base) * 100, thickness_ft=int(thick) * 1000)
        for code, base, thick in _ICING.findall(chunk)
    ]
    g.turbulence = [
        TurbulenceLayer(b=int(code) if code.isdigit() else code,
                        base_ft=int(base) * 100, thickness_ft=int(thick) * 1000)
        for code, base, thick in _TURB.findall(chunk)
    ]


def parse(line: str) -> TafObs:
    raw = line.strip().rstrip("=").strip()
    t = _PARSER.parse(raw)                          # library handles the 'TAF [AMD|COR]' prefix

    qnh = int(q.group(1)) / 100 if (q := _QNH.search(raw)) else None
    vf = _validity(t.validity)
    mx, mn = t.max_temperature, t.min_temperature

    prevailing = _period(t, None, None, vf)
    groups = [
        _period(tr, tr.type.name, tr.probability, _validity(tr.validity))
        for tr in t.trends
    ]
    # Re-attach showers groups the library drops. Each raw chunk lines up with
    # [prevailing, *trends]; if the split doesn't align 1:1 we skip recovery (the
    # token still survives in the retained raw line either way).
    periods = [prevailing, *groups]
    chunks = _split_periods(raw)
    if len(chunks) == len(periods):
        for chunk, period in zip(chunks, periods):
            _recover_showers(chunk, period.weather)
            # NSW ('no significant weather' -- weather is forecast to end) is also
            # dropped by the library; re-attach it so the decoded view shows it
            # instead of a misleading blank.
            if "NSW" in chunk.split() and "NSW" not in period.weather:
                period.weather.append("NSW")
            if cq := _QNH.search(chunk):     # military TAFs carry a QNH per group
                period.qnh_inhg = int(cq.group(1)) / 100
            _recover_hazards(chunk, period)
            # Record which fields this chunk explicitly stated (sec 5.1). Only here,
            # inside the 1:1 alignment guard -- unaligned parses leave it empty/unknown.
            period.explicit_fields = _explicit_fields(chunk)

    return TafObs(
        station=t.station,
        issue_day=t.day,
        issue_time=t.time,
        valid_from_day=vf[0], valid_from_hour=vf[1],
        valid_to_day=vf[3], valid_to_hour=vf[4],
        amendment=bool(t.amendment),
        corrected=bool(t.corrected),
        nil=bool(t.nil),
        canceled=bool(t.canceled),
        qnh_inhg=qnh,
        prevailing=prevailing,
        groups=groups,
        max_temp=TafTemp(temp_c=mx.temperature, day=mx.day, hour=mx.hour) if mx else None,
        min_temp=TafTemp(temp_c=mn.temperature, day=mn.day, hour=mn.hour) if mn else None,
        raw=raw,
    )


def _wind(g: TafGroup) -> str:
    if g.wind_speed is None:
        return "—"
    if g.wind_speed == 0:
        return "calm"
    d = f"{g.wind_dir_deg:03d}" if g.wind_dir_deg is not None else (g.wind_dir_card or "VRB")
    gust = f"G{g.wind_gust}" if g.wind_gust else ""
    return f"{d}/{g.wind_speed}{gust}{g.wind_unit}"


def _sky(clouds: list[CloudLayer]) -> str:
    if not clouds:
        return "SKC"
    return " ".join(
        c.cover + (f"{c.height_ft}ft" if c.height_ft is not None else "") + (c.type or "")
        for c in clouds
    )


def _hazards(g: TafGroup) -> str:
    """Compact decoded view of the recovered hazard groups, or '' if none."""
    parts = []
    if g.wind_shear:
        ws = g.wind_shear
        parts.append(f"WS {ws.height_ft}ft {ws.wind_dir:03d}/{ws.wind_speed}KT")
    if g.volcanic_ash:
        va = g.volcanic_ash
        parts.append(f"VA {va.base_ft}-{va.top_ft}ft")
    parts += [f"ICE(Ic{ic.ic}) {ic.base_ft}ft +{ic.thickness_ft}ft" for ic in g.icing]
    parts += [f"TURB(B{tb.b}) {tb.base_ft}ft +{tb.thickness_ft}ft" for tb in g.turbulence]
    return "  ".join(parts)


def _vis_display(g: TafGroup) -> str:
    """Reported token, with the statute-mile equivalent appended for meters vis so
    the AF (SM) reader isn't left converting: '9999 (P6SM)', '4800 (3SM)'. SM
    reports and CAVOK stand alone."""
    if g.cavok:
        return "CAVOK"
    if g.visibility is None:
        return "—"
    if g.visibility.endswith("SM") or g.vis_sm is None:
        return g.visibility
    return f"{g.visibility} ({_fmt_sm(g.vis_sm, g.vis_flag)})"


def _group_line(g: TafGroup, show_qnh: bool) -> str:
    if g.change_type is None:
        head = "INIT"
    elif g.change_type == "FM":
        head = f"FM {g.from_day:02d}{g.from_hour:02d}{g.from_minute:02d}Z"
    else:
        window = f"{g.from_day:02d}{g.from_hour:02d}/{g.to_day:02d}{g.to_hour:02d}"
        if not g.probability:
            head = f"{g.change_type} {window}"
        elif g.change_type == "PROB":              # bare PROBxx group: PROB is the keyword
            head = f"PROB{g.probability} {window}"
        else:                                      # PROBxx qualifying a TEMPO/BECMG
            head = f"PROB{g.probability} {g.change_type} {window}"
    vis = _vis_display(g)
    wx = " ".join(g.weather)
    sky = f"VV{g.vert_vis_ft}ft" if g.vert_vis_ft is not None else _sky(g.clouds)
    tail = f"{wx}  {sky}" if wx else sky
    if hz := _hazards(g):
        tail = f"{tail}  [{hz}]"
    cols = f"  {head:<22} {_wind(g):<13} {vis:<12}"
    if show_qnh:                                  # only for military TAFs that carry QNH
        cols += f" {(f'{g.qnh_inhg:.2f}inHg' if g.qnh_inhg else ''):<10}"
    return f"{cols} {tail}"


def _reflow(raw: str) -> str:
    """Re-insert the conventional TAF line breaks the json feed strips: each change
    group (and the TX/TN group) onto its own indented continuation line."""
    return _REFLOW.sub(lambda m: "\n     " + m.group(0), raw)


def render(obs: TafObs) -> str:
    """Decoded view for the messages array, with the raw TAF underneath (line
    breaks restored) so nothing the decoder dropped (NSW, RMK) is lost to the model."""
    flags = " ".join(
        f for f, on in [("AMD", obs.amendment), ("COR", obs.corrected),
                        ("NIL", obs.nil), ("CNL", obs.canceled)] if on
    )
    valid = (
        f"{obs.valid_from_day:02d}{obs.valid_from_hour:02d}/"
        f"{obs.valid_to_day:02d}{obs.valid_to_hour:02d}"
        if obs.valid_from_day is not None else "—"
    )
    periods = [obs.prevailing, *obs.groups]
    show_qnh = any(g.qnh_inhg for g in periods)
    header = (
        f"  {'group':<22} {'wind':<13} {'vis':<12}"
        + (f" {'QNH':<10}" if show_qnh else "")
        + " present-wx + sky(ft AGL)"
    )
    out = [
        f"{obs.station} TAF {('[' + flags + '] ') if flags else ''}"
        f"issued {obs.issue_day:02d} {obs.issue_time.strftime('%H%MZ')}, valid {valid}",
        header,
    ]
    out += [_group_line(g, show_qnh) for g in periods]
    if obs.max_temp:
        line = f"  TX {obs.max_temp.temp_c}C @ {obs.max_temp.day:02d}{obs.max_temp.hour:02d}Z"
        if obs.min_temp:
            line += f"   TN {obs.min_temp.temp_c}C @ {obs.min_temp.day:02d}{obs.min_temp.hour:02d}Z"
        out.append(line)
    out += ["", "Raw:", f"  {_reflow(obs.raw)}"]
    return "\n".join(out)
