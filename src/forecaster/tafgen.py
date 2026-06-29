"""Typed TafProduct -> valid AF TAF text (the forecast OUTPUT seam).

Symmetric to metar.py on the input side: tafparse.py turns TAF text into a typed
TafObs (what we READ); this turns a typed TafProduct into canonical TAF text (what
we WRITE). The model never types TAF grammar -- it fills the fields of a TafProduct
(a pydantic model, so its JSON schema IS the tool the model calls) and render_taf()
emits correctly-formatted text deterministically. That isolates weather REASONING
(the thing we benchmark) from TAF TYPOGRAPHY (leading zeros, group order, spacing).

Rules below are AF-specific, from AFMAN 15-124 ch.1 (docs/TAF Coding.pdf). Where AF
practice diverges from the civil/international grammar tafparse.py accepts, this
module follows the AF manual:
  - 30-hour validity for every non-amended TAF                       (1.3.1.1)
  - visibility in METERS, 4-digit, rounded down to Table 1.1         (1.3.5)
  - QNH on the prevailing period + each FM/BECMG, never in TEMPO     (1.3.12)
  - NO PROB30/PROB40 -- the manual defines only BECMG/TEMPO/FM, so
    TafProductGroup deliberately has no `prob` field (civil-only).   (1.3.3)
  - TX/TN temperature group, max first/min last, first 24h           (1.3.13.1)
  - limited-duty remarks LAST NO AMDS / LIMITED METWATCH             (1.3.13.2)

Hazard groups, between the cloud group and QNH (we don't always use these, but they
are encoded here so the seam is complete):
  - non-convective low-level wind shear  WShxhxhx/dddfffKT  (1.3.9.2; not in TEMPO)
  - volcanic ash                         VAbbbttt           (1.3.9.1)
  - icing                                6IchihihitL        (1.3.10; repeatable)
  - turbulence                           5BhBhBhBtL         (1.3.11; repeatable)
The code figures come straight from Tables 1.4-1.7 (reproduced inline below).

roundtrip() closes the loop: render -> tafparse.parse -> compare, so the output and
input seams are held to agree on every shared field. tafparse now recovers the hazard
groups + VV from raw, so those round-trip too (not just wind/vis/wx/sky/QNH/temps).
"""

from typing import Literal

from forecaster.metar import CloudLayer, _VIS_TABLE
from forecaster.tafparse import (
    IcingLayer, TafTemp, TurbulenceLayer, VolcanicAsh, WindShear, parse as parse_taf,
)

from pydantic import BaseModel, Field, field_validator

# Reportable visibility values in meters (Table 1.1, second column). Reuse metar's
# Table 8.1 as the single source -- the meters column matches Table 1.1 -- plus
# 9999 ("7 SM and above" = unrestricted). AF rounds vis DOWN to one of these.
_VIS_METERS = sorted({m for _, m in _VIS_TABLE} | {9999})


def _end_plus_hours(day: int, hour: int, hours: int) -> tuple[int, int]:
    """(day, hour) + N hours, applying the AFMAN midnight convention: a period that
    ENDS at midnight UTC is encoded hour 24 of the previous day, not 00 (1.3.2.1.5).
    Month-blind (no month on a TAF) -- a span into the next month needs the real end
    day from the caller; same caveat as validate()."""
    end = day * 24 + hour + hours
    d, h = divmod(end, 24)
    return (d - 1, 24) if h == 0 else (d, h)


# ---- Forecast period + product ----------------------------------------------
# The hazard sub-models (WindShear/VolcanicAsh/IcingLayer/TurbulenceLayer) live in
# tafparse and are imported above, so generation and parsing share one shape.

class TafProductGroup(BaseModel):
    """One forecast period of an AF TAF. change=None is the PREVAILING period
    (its timing comes from the TAF's overall validity); otherwise FM/BECMG/TEMPO.
    No `prob` field on purpose -- AF TAFs do not use PROB30/PROB40 (1.3.3)."""

    change: Literal["FM", "BECMG", "TEMPO"] | None = None   # None=prevailing
    # Timing. FM is a single instant YYGGgg (minutes allowed, 1.3.3.3); BECMG/TEMPO
    # are a YYGG/YYGeGe window. The prevailing period leaves these unset.
    from_day: int | None = Field(None, ge=1, le=31)
    from_hour: int | None = Field(None, ge=0, le=24)
    from_minute: int = Field(0, ge=0, le=59)   # only FM uses minutes; 4-digit time is FM-only
    to_day: int | None = Field(None, ge=1, le=31)    # BECMG/TEMPO window end; None for FM/prevailing
    to_hour: int | None = Field(None, ge=0, le=24)
    # Wind (dddffGfmfmKT). wind_dir: degrees (rendered to nearest 10), 'VRB', or None+speed 0 = calm.
    wind_dir: int | str | None = None
    wind_speed: int = Field(0, ge=0)
    wind_gust: int | None = Field(None, ge=0)
    # Visibility in METERS (1.3.5). 9999 = unrestricted. Rounded down to Table 1.1.
    vis_m: int | None = Field(None, ge=0)
    weather: list[str] = []          # present-wx tokens, e.g. ['-SHSN', 'BLSN']; ['NSW'] to cancel
    clouds: list[CloudLayer] = []    # ascending bases; empty => SKC; type 'CB' appends CB
    vert_vis_ft: int | None = Field(None, ge=0)   # VVhshshs total obscuration; overrides clouds
    # Hazard groups (rendered between clouds and QNH).
    wind_shear: WindShear | None = None
    volcanic_ash: VolcanicAsh | None = None
    icing: list[IcingLayer] = []
    turbulence: list[TurbulenceLayer] = []
    qnh_inhg: float | None = None    # required on prevailing/FM/BECMG, forbidden on TEMPO
    remarks: list[str] = []          # per-group remarks (partial-obscuration phenom+layer, WND dddVddd)

    @field_validator("wind_dir")
    @classmethod
    def _check_wind_dir(cls, v: int | str | None) -> int | str | None:
        """Degrees 0-360, 'VRB', or None. (The 'nearest 10 deg' AFMAN rule is a
        validate() finding, not a construction error -- 275 is a real value.) The
        int|str union lets the model emit a quoted number ('240'); coerce it, since
        that is plainly a direction, not VRB."""
        if isinstance(v, str) and v.lstrip("-").isdigit():
            v = int(v)
        if v is None or v == "VRB" or (isinstance(v, int) and 0 <= v <= 360):
            return v
        raise ValueError("wind_dir must be 0-360, 'VRB', or None")

    @field_validator("clouds")
    @classmethod
    def _check_covers(cls, v: list[CloudLayer]) -> list[CloudLayer]:
        """A TAF cloud group uses FEW/SCT/BKN/OVC only; total obscuration is vert_vis_ft."""
        bad = [c.cover for c in v if c.cover not in {"FEW", "SCT", "BKN", "OVC"}]
        if bad:
            raise ValueError(f"cloud cover must be FEW/SCT/BKN/OVC; got {bad}")
        return v


class TafProduct(BaseModel):
    """One AF TAF to be emitted. The model authors these fields; render_taf() turns
    them into valid text and validate() checks them against the AFMAN rules."""

    station: str = Field(pattern=r"^[A-Z]{4}$")       # 4-letter ICAO (CCCC)
    issue_day: int = Field(ge=1, le=31)               # YYGGggZ
    issue_hour: int = Field(ge=0, le=23)
    issue_minute: int = Field(ge=0, le=59)
    valid_from_day: int = Field(ge=1, le=31)          # YYG1G1/YYG2G2 (30h for non-amended)
    valid_from_hour: int = Field(ge=0, le=24)
    valid_to_day: int = Field(ge=1, le=31)
    valid_to_hour: int = Field(ge=0, le=24)
    amendment: bool = False          # AMD (only one modifier at a time, 1.3.2.1.2)
    corrected: bool = False          # COR
    prevailing: TafProductGroup
    groups: list[TafProductGroup] = []   # change groups, in TAF order
    max_temp: TafTemp | None = None  # TX (first 24h)
    min_temp: TafTemp | None = None  # TN
    remarks: list[str] = []          # TAF-level remarks (LAST NO AMDS / LIMITED METWATCH), after TX/TN

    @classmethod
    def issue(cls, *, station: str, issue_day: int, issue_hour: int, issue_minute: int,
              valid_from_day: int, valid_from_hour: int, prevailing: TafProductGroup,
              groups: list[TafProductGroup] | None = None,
              max_temp: TafTemp | None = None, min_temp: TafTemp | None = None,
              remarks: list[str] | None = None, corrected: bool = False) -> "TafProduct":
        """Build a routine (or COR) TAF, computing the 30-hour valid_to with the
        midnight 00/24 convention (1.3.1.1, 1.3.2.1.5). You give the issue time and
        the whole-hour validity START; the end is start+30h, so the span is correct
        by construction -- you can never author a non-30h routine TAF this way."""
        vt_day, vt_hour = _end_plus_hours(valid_from_day, valid_from_hour, 30)
        return cls(
            station=station, issue_day=issue_day, issue_hour=issue_hour, issue_minute=issue_minute,
            valid_from_day=valid_from_day, valid_from_hour=valid_from_hour,
            valid_to_day=vt_day, valid_to_hour=vt_hour, corrected=corrected,
            prevailing=prevailing, groups=groups or [],
            max_temp=max_temp, min_temp=min_temp, remarks=remarks or [],
        )

    @classmethod
    def amend(cls, original: "TafProduct", *, at_day: int, at_hour: int, at_minute: int,
              prevailing: TafProductGroup | None = None) -> "TafProduct":
        """Build an amendment (AMD) of `original` at the given time (1.3.2.1.2.1): new
        header/issue time, validity clipped to [current whole hour .. original end],
        and change groups no longer valid (ending at/before the clip hour) dropped. An
        FM's effective end is the next group's start. The prevailing carries over
        unless you pass a new one describing the conditions now in effect."""
        clip = at_day * 24 + at_hour
        orig_end = original.valid_to_day * 24 + original.valid_to_hour
        kept: list[TafProductGroup] = []
        for i, g in enumerate(original.groups):
            if g.to_day is not None and g.to_hour is not None:
                end = g.to_day * 24 + g.to_hour
            else:                                 # FM: effective end = next group's start
                nxt = next((n for n in original.groups[i + 1:] if n.from_day is not None), None)
                end = (nxt.from_day * 24 + nxt.from_hour) if nxt else orig_end
            if end > clip:
                kept.append(g)
        return cls(
            station=original.station,
            issue_day=at_day, issue_hour=at_hour, issue_minute=at_minute,
            valid_from_day=at_day, valid_from_hour=at_hour,
            valid_to_day=original.valid_to_day, valid_to_hour=original.valid_to_hour,
            amendment=True, prevailing=prevailing or original.prevailing,
            groups=kept, max_temp=original.max_temp, min_temp=original.min_temp,
        )


# ---- Rendering --------------------------------------------------------------

def _round_down_vis(m: int) -> int:
    """Round meters DOWN to the nearest reportable Table 1.1 value (1.3.5)."""
    return max(v for v in _VIS_METERS if v <= m) if m >= _VIS_METERS[0] else _VIS_METERS[0]


def _wind(g: TafProductGroup) -> str:
    if g.wind_speed == 0 and g.wind_dir in (None, 0):
        return "00000KT"                                  # calm (1.3.4.1.1)
    spd = f"{g.wind_speed:03d}" if g.wind_speed >= 100 else f"{g.wind_speed:02d}"
    d = "VRB" if g.wind_dir == "VRB" else f"{int(g.wind_dir):03d}"
    gust = ""
    if g.wind_gust:
        gust = f"G{g.wind_gust:03d}" if g.wind_gust >= 100 else f"G{g.wind_gust:02d}"
    return f"{d}{spd}{gust}KT"


def _vis(g: TafProductGroup) -> str | None:
    if g.vis_m is None:
        return None
    return "9999" if g.vis_m >= 9999 else f"{_round_down_vis(g.vis_m):04d}"


def _sky(g: TafProductGroup) -> str:
    if g.vert_vis_ft is not None:                         # total obscuration (1.3.7.2/.4)
        return f"VV{g.vert_vis_ft // 100:03d}"
    if not g.clouds:
        return "SKC"
    return " ".join(
        f"{c.cover}{(c.height_ft or 0) // 100:03d}{'CB' if c.type == 'CB' else ''}"
        for c in g.clouds
    )


def _windshear(ws: WindShear) -> str:
    spd = f"{ws.wind_speed:03d}" if ws.wind_speed >= 100 else f"{ws.wind_speed:02d}"
    return f"WS{ws.height_ft // 100:03d}/{ws.wind_dir:03d}{spd}KT"


def _va(va: VolcanicAsh) -> str:
    return f"VA{va.base_ft // 100:03d}{va.top_ft // 100:03d}"


def _icing(ic: IcingLayer) -> str:
    return f"6{ic.ic}{ic.base_ft // 100:03d}{ic.thickness_ft // 1000}"


def _turb(tb: TurbulenceLayer) -> str:
    return f"5{tb.b}{tb.base_ft // 100:03d}{tb.thickness_ft // 1000}"


def _qnh(g: TafProductGroup) -> str | None:
    if g.qnh_inhg is None:
        return None
    return f"QNH{round(g.qnh_inhg * 100):04d}INS"          # 29.92 -> QNH2992INS


def _temp(t: TafTemp) -> str:
    """Value+time half of a TX/TN group: 17C@072100Z -> '17/0721Z', -1 -> 'M01/0212Z'."""
    v = f"M{abs(t.temp_c):02d}" if t.temp_c < 0 else f"{t.temp_c:02d}"
    return f"{v}/{t.day:02d}{t.hour:02d}Z"


def _hh(v: int | None) -> str:
    """Two-digit field, or '??' when the model omitted it -- so a malformed change
    group renders visibly instead of crashing; validate() reports the real finding."""
    return f"{v:02d}" if v is not None else "??"


def _head(g: TafProductGroup) -> str | None:
    """Change-group indicator, or None for the prevailing period."""
    if g.change is None:
        return None
    if g.change == "FM":
        return f"FM{_hh(g.from_day)}{_hh(g.from_hour)}{_hh(g.from_minute)}"
    return f"{g.change} {_hh(g.from_day)}{_hh(g.from_hour)}/{_hh(g.to_day)}{_hh(g.to_hour)}"


def _body(g: TafProductGroup) -> str:
    """Element sequence (Fig 1.1): wind, vis, weather, clouds, [WS, VA, icing,
    turbulence], QNH, remarks."""
    parts = [_wind(g)]
    if (v := _vis(g)) is not None:
        parts.append(v)
    if g.weather:
        parts.append(" ".join(g.weather))
    parts.append(_sky(g))
    if g.wind_shear:
        parts.append(_windshear(g.wind_shear))
    if g.volcanic_ash:
        parts.append(_va(g.volcanic_ash))
    parts += [_icing(ic) for ic in g.icing]
    parts += [_turb(tb) for tb in g.turbulence]
    if (q := _qnh(g)) is not None:
        parts.append(q)
    parts += g.remarks
    return " ".join(parts)


def _line(g: TafProductGroup) -> str:
    head = _head(g)
    body = _body(g)
    return f"{head} {body}" if head else body


def render_taf(p: TafProduct) -> str:
    """A TafProduct -> canonical AF TAF text. Each change group on its own line
    (1.3.3); TX/TN and TAF-level remarks close the forecast."""
    mod = " AMD" if p.amendment else " COR" if p.corrected else ""
    issue = f"{p.issue_day:02d}{p.issue_hour:02d}{p.issue_minute:02d}Z"
    valid = f"{p.valid_from_day:02d}{p.valid_from_hour:02d}/{p.valid_to_day:02d}{p.valid_to_hour:02d}"
    lines = [f"TAF{mod} {p.station} {issue} {valid} {_body(p.prevailing)}"]
    lines += [_line(g) for g in p.groups]

    tail: list[str] = []
    if p.max_temp:
        tail.append(f"TX{_temp(p.max_temp)}")
    if p.min_temp:
        tail.append(f"TN{_temp(p.min_temp)}")
    tail += p.remarks
    if tail:                                   # TX/TN + remarks close the LAST line (per AFMAN figures)
        lines[-1] = f"{lines[-1]} {' '.join(tail)}"
    return "\n".join(lines)


def last_no_amds(after_day: int, after_hour: int, next_day: int, next_hour: int) -> str:
    """Limited-duty remark: airfield closed, TAF no longer required (1.3.13.2.1)."""
    return f"LAST NO AMDS AFT {after_day:02d}{after_hour:02d} NEXT {next_day:02d}{next_hour:02d}"


def limited_metwatch(from_day: int, from_hour: int, to_day: int, to_hour: int) -> str:
    """Limited-duty remark: open, no weather personnel, no automated sensor (1.3.13.2.2)."""
    return f"LIMITED METWATCH {from_day:02d}{from_hour:02d} TIL {to_day:02d}{to_hour:02d}"


# ---- AFMAN 15-124 rule checker ----------------------------------------------

_COVER_RANK = {"FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}


def _check_group(g: TafProductGroup, tag: str, out: list[str]) -> None:
    """Element-level rules that apply to any period (prevailing or change group)."""
    # Wind (1.3.4)
    if isinstance(g.wind_dir, int):
        if not 0 <= g.wind_dir <= 360:
            out.append(f"{tag}: wind direction out of range 0-360 (1.3.4.1)")
        elif g.wind_dir % 10 != 0:
            out.append(f"{tag}: wind direction to nearest 10 deg (1.3.4.1)")
    if g.wind_dir == "VRB" and g.wind_speed > 6:
        out.append(f"{tag}: VRB only for wind <=6kt or air-mass TS (1.3.4.2)")
    if g.wind_gust is not None and g.wind_gust <= g.wind_speed:
        # The 10kt threshold (over mean OR lull) cannot be checked -- the lull is
        # not encoded -- but a gust must at least exceed the mean (1.3.4.2.2).
        out.append(f"{tag}: gust must exceed the mean wind speed (1.3.4.2.2)")

    # Visibility (1.3.5): a restriction below 9999 needs a weather/obscuration cause.
    sig_wx = [w for w in g.weather if w != "NSW"]
    if g.vis_m is not None and g.vis_m < 9999 and not sig_wx and g.vert_vis_ft is None:
        out.append(f"{tag}: vis <9999 needs a weather/obscuration cause (1.3.5)")

    # Clouds (1.3.7)
    heights = [c.height_ft for c in g.clouds if c.height_ft is not None]
    if heights != sorted(heights):
        out.append(f"{tag}: cloud layers must ascend by base (1.3.7)")
    ranks = [_COVER_RANK.get(c.cover, 0) for c in g.clouds]
    if any(ranks[i] < ranks[i - 1] for i in range(1, len(ranks))):
        out.append(f"{tag}: summation principle -- cover cannot decrease with height (1.3.7.1)")
    ovc = next((i for i, c in enumerate(g.clouds) if c.cover == "OVC"), None)
    if ovc is not None and ovc != len(g.clouds) - 1:
        out.append(f"{tag}: report layers only up to the first overcast (1.3.7)")
    if any("TS" in w for w in g.weather) and not any(c.type == "CB" for c in g.clouds):
        out.append(f"{tag}: TS forecast but no CB in cloud group (1.3.7.8)")

    # Hazard groups
    if g.volcanic_ash and g.volcanic_ash.base_ft == 0 and "VA" not in g.weather:
        out.append(f"{tag}: surface-based VA must also be present weather VA (1.3.9.1.3)")
    for layer in (*g.icing, *g.turbulence):
        if not 1000 <= layer.thickness_ft <= 9000:
            out.append(f"{tag}: icing/turbulence thickness must be 1000-9000ft (Table 1.6)")


def validate(p: TafProduct) -> list[str]:
    """Check a TafProduct against the AFMAN 15-124 rules. Returns a list of human-
    readable findings (empty = clean). A format CHECKER, not an exception raiser, so
    a caller (or the agent) can see every rule it broke at once.

    NOTE: span/within-24h checks work in absolute day*24+hour and so cannot see a
    month boundary (a TAF whose validity crosses into the next month) -- there is no
    month on a TAF. Flagged here; correct for any TAF that stays within one month.
    """
    out: list[str] = []

    # Validity (1.3.1.1 / 1.3.2.1.5)
    span = (p.valid_to_day * 24 + p.valid_to_hour) - (p.valid_from_day * 24 + p.valid_from_hour)
    if not p.amendment and span != 30:
        out.append(f"non-amended TAF must be valid 30h (1.3.1.1); got {span}h")
    if p.amendment and not (0 < span <= 30):
        out.append(f"amended TAF validity must be >0 and <=30h (1.3.2.1.5); got {span}h")
    if p.valid_to_hour == 0:
        out.append("validity END at midnight must be encoded hour 24, not 00 (1.3.2.1.5)")

    # Prevailing period + each change group
    _check_group(p.prevailing, "prevailing", out)
    if p.prevailing.qnh_inhg is None:
        out.append("prevailing: QNH expected on the initial period (1.3.12)")

    starts: list[tuple[int, int, int]] = []
    for i, g in enumerate(p.groups):
        tag = f"group {i} ({g.change})"
        _check_group(g, tag, out)

        if g.change in ("BECMG", "TEMPO") and (g.to_day is None or g.to_hour is None):
            out.append(f"{tag}: BECMG/TEMPO need a YYGG/YYGeGe window (1.3.3)")
        if g.change in ("BECMG", "TEMPO") and g.to_hour == 0:
            out.append(f"{tag}: window END at midnight must be hour 24, not 00 (1.3.2.1.5)")
        if g.change == "BECMG" and g.to_hour is not None and g.from_hour is not None:
            win = (g.to_day * 24 + g.to_hour) - (g.from_day * 24 + g.from_hour)
            if win > 2:
                out.append(f"{tag}: BECMG window never exceeds 2h (1.3.3.1); got {win}h")
        if g.change == "TEMPO":
            if g.qnh_inhg is not None:
                out.append(f"{tag}: QNH not allowed in TEMPO (1.3.12)")
            if g.wind_shear is not None:
                out.append(f"{tag}: wind shear not allowed in TEMPO (1.3.9.2.2)")
        if g.change in (None, "FM", "BECMG") and g.qnh_inhg is None:
            out.append(f"{tag}: QNH expected on FM/BECMG (1.3.12)")
        if g.change == "FM" and g.vis_m is None:
            out.append(f"{tag}: FM is self-contained -- must include visibility (1.3.3.3)")
        if g.from_day is not None and g.from_hour is not None:
            starts.append((g.from_day, g.from_hour, g.from_minute))

    if starts != sorted(starts):
        out.append("change groups must be in chronological order (1.3.3)")

    # Temperatures (1.3.13.1)
    if p.max_temp and p.min_temp and p.max_temp.temp_c < p.min_temp.temp_c:
        out.append("TX (max) should be >= TN (min) (1.3.13.1)")
    start = p.valid_from_day * 24 + p.valid_from_hour
    for t, name in ((p.max_temp, "TX"), (p.min_temp, "TN")):
        if t and not 0 <= (t.day * 24 + t.hour) - start <= 24:
            out.append(f"{name} time must fall in the first 24h of validity (1.3.13.1)")
    return out


# ---- Round-trip checker -----------------------------------------------------

def _norm_clouds(clouds: list[CloudLayer]) -> list[tuple[str, int | None, str | None]]:
    """Cloud layers as comparable tuples, bases rounded down to hundreds the way
    render does (650ft -> 600), so authoring a non-reportable base is not a false diff."""
    return [
        (c.cover, None if c.height_ft is None else c.height_ft // 100 * 100, c.type)
        for c in clouds
    ]


def _expected_vis_m(g: TafProductGroup) -> int | None:
    """The meters value render actually emits (rounded down to Table 1.1)."""
    if g.vis_m is None:
        return None
    return 9999 if g.vis_m >= 9999 else _round_down_vis(g.vis_m)


def _diff_group(label: str, pg: TafProductGroup, og, out: list[str]) -> None:
    """Compare an authored period against the parsed-back period, field by field."""
    if pg.change != og.change_type:
        out.append(f"{label}: change {pg.change!r} -> parsed {og.change_type!r}")
    if pg.change is not None:
        if (pg.from_day, pg.from_hour) != (og.from_day, og.from_hour):
            out.append(f"{label}: start-time -> parsed {og.from_day}/{og.from_hour}")
        if pg.change == "FM" and pg.from_minute != (og.from_minute or 0):
            out.append(f"{label}: FM minute {pg.from_minute} -> parsed {og.from_minute}")
        if pg.change in ("BECMG", "TEMPO") and (pg.to_day, pg.to_hour) != (og.to_day, og.to_hour):
            out.append(f"{label}: end-time -> parsed {og.to_day}/{og.to_hour}")

    if isinstance(pg.wind_dir, int):
        if pg.wind_dir != og.wind_dir_deg:
            out.append(f"{label}: wind dir {pg.wind_dir} -> parsed {og.wind_dir_deg}")
    elif pg.wind_dir == "VRB" and og.wind_dir_card != "VRB":
        out.append(f"{label}: wind VRB -> parsed {og.wind_dir_card or og.wind_dir_deg}")
    if pg.wind_speed != (og.wind_speed or 0):
        out.append(f"{label}: wind speed {pg.wind_speed} -> parsed {og.wind_speed}")
    if pg.wind_gust != og.wind_gust:
        out.append(f"{label}: gust {pg.wind_gust} -> parsed {og.wind_gust}")

    if _expected_vis_m(pg) != og.vis_m:
        out.append(f"{label}: vis {_expected_vis_m(pg)}m -> parsed {og.vis_m}m")
    if sorted(pg.weather) != sorted(og.weather):
        out.append(f"{label}: weather {pg.weather} -> parsed {og.weather}")
    if _norm_clouds(pg.clouds) != _norm_clouds(og.clouds):
        out.append(f"{label}: clouds {_norm_clouds(pg.clouds)} -> parsed {_norm_clouds(og.clouds)}")
    if pg.vert_vis_ft != og.vert_vis_ft:
        out.append(f"{label}: VV {pg.vert_vis_ft} -> parsed {og.vert_vis_ft}")
    if pg.qnh_inhg != og.qnh_inhg:
        out.append(f"{label}: QNH {pg.qnh_inhg} -> parsed {og.qnh_inhg}")

    # Hazards: same model classes on both sides, so pydantic value-equality holds.
    for name, a, b in (("wind shear", pg.wind_shear, og.wind_shear),
                       ("volcanic ash", pg.volcanic_ash, og.volcanic_ash),
                       ("icing", pg.icing, og.icing),
                       ("turbulence", pg.turbulence, og.turbulence)):
        if a != b:
            out.append(f"{label}: {name} {a} -> parsed {b}")


def roundtrip(p: TafProduct) -> list[str]:
    """Render p, parse the text back with tafparse, and report semantic differences
    (empty = the seams agree). Proves render_taf() emits text that decodes to the
    same forecast. Values render intentionally normalizes (vis + cloud bases to the
    reportable increments) are compared post-normalization, not flagged as drift;
    values tafparse derives but TafProduct never authors (vis_sm, raw) are skipped.

    Free-text REMARKS (partial-obscuration causes like 'FG FEW000', limited-duty
    notes like 'LAST NO AMDS ...') are EXCLUDED: AF TAF remarks carry no delimiter,
    so any parser folds them back into the forecast groups (or misreads a remark
    number as visibility). They cannot round-trip by construction -- render_taf's
    byte-exact output is what covers them -- so we parse a remark-stripped render and
    compare only the structured forecast body."""
    stripped = p.model_copy(deep=True)
    stripped.remarks = []
    for g in (stripped.prevailing, *stripped.groups):
        g.remarks = []
    obs = parse_taf(render_taf(stripped).replace("\n", " "))
    out: list[str] = []

    if p.station != obs.station:
        out.append(f"station {p.station} -> parsed {obs.station}")
    if (p.issue_day, p.issue_hour, p.issue_minute) != (
        obs.issue_day, obs.issue_time.hour, obs.issue_time.minute):
        out.append(f"issue {p.issue_day:02d}{p.issue_hour:02d}{p.issue_minute:02d}Z -> "
                   f"parsed {obs.issue_day:02d}{obs.issue_time.strftime('%H%M')}Z")
    if (p.valid_from_day, p.valid_from_hour, p.valid_to_day, p.valid_to_hour) != (
        obs.valid_from_day, obs.valid_from_hour, obs.valid_to_day, obs.valid_to_hour):
        out.append("validity differs after round-trip")
    if p.amendment != obs.amendment:
        out.append(f"AMD {p.amendment} -> parsed {obs.amendment}")
    if p.corrected != obs.corrected:
        out.append(f"COR {p.corrected} -> parsed {obs.corrected}")

    p_periods = [p.prevailing, *p.groups]
    o_periods = [obs.prevailing, *obs.groups]
    if len(p_periods) != len(o_periods):
        out.append(f"group count: authored {len(p_periods)} -> parsed {len(o_periods)}")
    else:
        for i, (pg, og) in enumerate(zip(p_periods, o_periods)):
            _diff_group("prevailing" if i == 0 else f"group {i - 1} ({pg.change})", pg, og, out)

    for t, ot, name in ((p.max_temp, obs.max_temp, "TX"), (p.min_temp, obs.min_temp, "TN")):
        if (t is None) != (ot is None):
            out.append(f"{name} presence differs after round-trip")
        elif t and ot and (t.temp_c, t.day, t.hour) != (ot.temp_c, ot.day, ot.hour):
            out.append(f"{name} {t.temp_c}@{t.day:02d}{t.hour:02d}Z -> "
                       f"parsed {ot.temp_c}@{ot.day:02d}{ot.hour:02d}Z")
    return out
