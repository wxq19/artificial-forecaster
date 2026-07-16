"""Shared scoring primitives (scoring-design sec 5). PURE: no duckdb/SQL, no
network, no matplotlib, no LLM. Imports the typed TAF objects from tafparse and
reads plain METAR dicts (the shape store.window / store.latest return).

Everything downstream of scoring keys off the objects here:

- State + availability statuses (5.2.4): a resolved set of weather elements where
  every nullable field carries an EXPLICIT status, so None is never ambiguous
  (known_numeric vs known_unlimited vs unknown, etc.).
- absolutize / opportunities / resolve_group_state / forecast_state (5.2): turn a
  TAF's relative day/hour groups into absolute naive-UTC intervals and resolve the
  fully-inherited state of any group at any instant.
- normalize_weather (5.6): verification-specific present-weather normalizer (atomic
  + class-level scoring events) -- NOT wxcodes' display/severity taxonomy.
- classifiers + StationProfile (5.3): TAFVER cig/vis categories and the DAF A4.1
  flight category (lower-of), each behind its own contract.
- build_truth (5.4/5.5): the intervalized two-view (conservative/union + per-field
  predominant) hourly truth builder + coverage manifest.

Naive UTC throughout (the repo seam contract). No new dependencies.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from pydantic import BaseModel

from forecaster.metar import CloudLayer, _parse_vis
from forecaster.tafparse import TafGroup, TafObs

# ---------------------------------------------------------------------------
# State + availability statuses (5.2.4)
# ---------------------------------------------------------------------------

# Availability status vocabularies. `unknown` always routes a scorer to
# status=unavailable with a reason -- never to a default value.
CEIL_KNOWN_NUMERIC = "known_numeric"
CEIL_KNOWN_UNLIMITED = "known_unlimited"
STATUS_UNKNOWN = "unknown"
VIS_KNOWN_NUMERIC = "known_numeric"
VIS_KNOWN_UNLIMITED = "known_unlimited"
DIR_NUMERIC = "numeric"
DIR_VRB = "vrb"
DIR_CALM = "calm"
SPD_NUMERIC = "numeric"
GUST_PRESENT = "present"
GUST_KNOWN_ABSENT = "known_absent"
GUST_INHERITED_ABSENT = "inherited_absent"
WX_KNOWN = "known"           # NSW / CAVOK resolve here with an empty set
QNH_KNOWN = "known"


class State(BaseModel):
    """A fully-resolved set of weather elements (forecast or observed). Every
    nullable field is paired with a status so `None` is never ambiguous."""

    wind_dir: int | None = None
    wind_dir_status: str = STATUS_UNKNOWN        # numeric | vrb | calm | unknown
    wind_speed: int | None = None
    wind_speed_status: str = STATUS_UNKNOWN      # numeric | unknown (calm = numeric 0)
    wind_gust: int | None = None
    gust_status: str = STATUS_UNKNOWN            # present | known_absent | inherited_absent | unknown
    vis_sm: float | None = None
    vis_m: int | None = None
    vis_flag: str | None = None
    vis_status: str = STATUS_UNKNOWN             # known_numeric | known_unlimited | unknown
    ceiling_ft: int | None = None
    ceiling_status: str = STATUS_UNKNOWN         # known_numeric | known_unlimited | unknown
    weather: list[str] = []                      # raw tokens (NSW dropped -> empty set, WX_KNOWN)
    weather_status: str = STATUS_UNKNOWN         # known | unknown
    qnh_inhg: float | None = None
    qnh_status: str = STATUS_UNKNOWN             # known | unknown


def _ceiling_from_clouds(clouds: list[CloudLayer], vert_vis_ft: int | None) -> tuple[int | None, str]:
    """Lowest BKN/OVC base (or a VV indefinite ceiling) -> (ft, status). No ceiling-
    bearing layer -> known_unlimited (clear/FEW/SCT). Mirrors metar's ceiling rule."""
    if vert_vis_ft is not None:
        return vert_vis_ft, CEIL_KNOWN_NUMERIC       # total obscuration = indefinite ceiling
    heights = [c.height_ft for c in clouds if c.cover in ("BKN", "OVC") and c.height_ft is not None]
    if heights:
        return min(heights), CEIL_KNOWN_NUMERIC
    return None, CEIL_KNOWN_UNLIMITED


def _vis_status(vis_sm: float | None, vis_flag: str | None, cavok: bool) -> str:
    """CAVOK or a P-flagged >=6SM value (P6SM / 9999m) is unlimited; a real number is
    numeric; nothing is unknown."""
    if cavok:
        return VIS_KNOWN_UNLIMITED
    if vis_sm is None:
        return STATUS_UNKNOWN
    if vis_flag == "P" and vis_sm >= 6:
        return VIS_KNOWN_UNLIMITED
    return VIS_KNOWN_NUMERIC


def _wind_from(dir_deg, dir_card, speed) -> tuple[int | None, str, int | None, str]:
    """(dir, dir_status, speed, speed_status) from raw wind fields, VRB/calm aware."""
    if speed is None and dir_deg is None and dir_card is None:
        return None, STATUS_UNKNOWN, None, STATUS_UNKNOWN
    spd = speed
    spd_status = SPD_NUMERIC if spd is not None else STATUS_UNKNOWN
    if dir_card == "VRB":
        return None, DIR_VRB, spd, spd_status
    if spd == 0:
        return None, DIR_CALM, 0, SPD_NUMERIC
    if dir_deg is not None:
        return dir_deg, DIR_NUMERIC, spd, spd_status
    return None, STATUS_UNKNOWN, spd, spd_status


def _forecast_state(g: TafGroup, base: State | None) -> State:
    """Resolve a group's full state. base is None for a self-contained group
    (prevailing / FM -- an omitted field is a KNOWN absence/clear/unlimited or, for
    wind/vis/qnh, unknown); a concrete base for BECMG/TEMPO overlays -- an omitted
    field INHERITS from base (gust becomes inherited_absent)."""
    ef = g.explicit_fields
    s = State()

    # Wind + gust
    if "wind" in ef:
        s.wind_dir, s.wind_dir_status, s.wind_speed, s.wind_speed_status = _wind_from(
            g.wind_dir_deg, g.wind_dir_card, g.wind_speed
        )
        if "gust" in ef:
            s.wind_gust, s.gust_status = g.wind_gust, GUST_PRESENT
        else:
            s.wind_gust, s.gust_status = None, GUST_KNOWN_ABSENT
    elif base is not None:
        s.wind_dir, s.wind_dir_status = base.wind_dir, base.wind_dir_status
        s.wind_speed, s.wind_speed_status = base.wind_speed, base.wind_speed_status
        s.wind_gust = base.wind_gust
        s.gust_status = base.gust_status if base.gust_status == GUST_PRESENT else GUST_INHERITED_ABSENT
    # else: self-contained group with no wind -> stays unknown (should not happen)

    # Visibility
    if "visibility" in ef or g.cavok:
        s.vis_sm, s.vis_m, s.vis_flag = g.vis_sm, g.vis_m, g.vis_flag
        s.vis_status = _vis_status(g.vis_sm, g.vis_flag, g.cavok)
    elif base is not None:
        s.vis_sm, s.vis_m, s.vis_flag, s.vis_status = base.vis_sm, base.vis_m, base.vis_flag, base.vis_status
    # else self-contained without vis -> unknown

    # Sky / ceiling
    if "sky" in ef or g.cavok:
        if g.cavok:
            s.ceiling_ft, s.ceiling_status = None, CEIL_KNOWN_UNLIMITED
        else:
            s.ceiling_ft, s.ceiling_status = _ceiling_from_clouds(g.clouds, g.vert_vis_ft)
    elif base is not None:
        s.ceiling_ft, s.ceiling_status = base.ceiling_ft, base.ceiling_status
    elif base is None:
        # self-contained group that stated no sky -> clear -> unlimited ceiling
        s.ceiling_ft, s.ceiling_status = None, CEIL_KNOWN_UNLIMITED

    # Weather (NSW / CAVOK -> known empty)
    if "weather" in ef or g.cavok:
        toks = [w for w in g.weather if w != "NSW"]
        s.weather, s.weather_status = toks, WX_KNOWN
    elif base is not None:
        s.weather, s.weather_status = list(base.weather), base.weather_status
    else:
        s.weather, s.weather_status = [], WX_KNOWN     # self-contained, no wx -> known clear

    # QNH
    if "qnh" in ef and g.qnh_inhg is not None:
        s.qnh_inhg, s.qnh_status = g.qnh_inhg, QNH_KNOWN
    elif base is not None:
        s.qnh_inhg, s.qnh_status = base.qnh_inhg, base.qnh_status
    # else self-contained without QNH (civil) -> unknown

    return s


def persistence_taf(ob: dict, valid_from: datetime, valid_to: datetime) -> TafObs:
    """The persistence BASELINE (sec 10): the given ob frozen as a single-prevailing-
    group TAF over [valid_from, valid_to). The model must beat it. explicit_fields are
    set for every element the ob supplies, so the resolver treats it like a real TAF."""
    ef = {"sky", "weather"}          # persistence always commits to observed sky + weather
    if ob.get("wind_speed") is not None:
        ef.add("wind")
    if ob.get("wind_gust") is not None:
        ef.add("gust")
    if ob.get("vis_sm") is not None:
        ef.add("visibility")
    if ob.get("altimeter_inhg") is not None:
        ef.add("qnh")
    g = TafGroup(
        change_type=None, probability=None,
        from_day=valid_from.day, from_hour=valid_from.hour, from_minute=0,
        to_day=valid_to.day, to_hour=valid_to.hour,
        wind_dir_deg=ob.get("wind_dir_deg"), wind_dir_card=ob.get("wind_dir_card"),
        wind_speed=ob.get("wind_speed"), wind_gust=ob.get("wind_gust"),
        wind_unit=ob.get("wind_unit") or "KT",
        visibility=ob.get("visibility"), vis_sm=ob.get("vis_sm"), vis_m=ob.get("vis_m"),
        vis_flag=ob.get("vis_flag"), cavok=bool(ob.get("cavok")),
        weather=list(ob.get("weather") or []),
        clouds=[CloudLayer(**c) for c in (ob.get("clouds") or [])],
        qnh_inhg=ob.get("altimeter_inhg"),
        vert_vis_ft=ob.get("vertical_visibility_ft"),
        explicit_fields=ef,
    )
    return TafObs(
        station=ob["station"], issue_day=valid_from.day,
        issue_time=time(valid_from.hour, valid_from.minute),
        valid_from_day=valid_from.day, valid_from_hour=valid_from.hour,
        valid_to_day=valid_to.day, valid_to_hour=valid_to.hour,
        amendment=False, corrected=False, nil=False, canceled=False,
        qnh_inhg=ob.get("altimeter_inhg"), prevailing=g, groups=[],
        max_temp=None, min_temp=None,
        raw=f"PERSISTENCE {ob['station']} from ob @ {ob.get('obs_time')}",
    )


def observed_state(ob: dict) -> State:
    """Build a State from a stored METAR dict (store.window shape)."""
    s = State()
    s.wind_dir, s.wind_dir_status, s.wind_speed, s.wind_speed_status = _wind_from(
        ob.get("wind_dir_deg"), ob.get("wind_dir_card"), ob.get("wind_speed")
    )
    if ob.get("wind_gust") is not None:
        s.wind_gust, s.gust_status = ob["wind_gust"], GUST_PRESENT
    elif s.wind_speed_status == SPD_NUMERIC:
        s.wind_gust, s.gust_status = None, GUST_KNOWN_ABSENT
    s.vis_sm, s.vis_m, s.vis_flag = ob.get("vis_sm"), ob.get("vis_m"), ob.get("vis_flag")
    s.vis_status = _vis_status(ob.get("vis_sm"), ob.get("vis_flag"), bool(ob.get("cavok")))
    if ob.get("ceiling_ft") is not None:
        s.ceiling_ft, s.ceiling_status = ob["ceiling_ft"], CEIL_KNOWN_NUMERIC
    elif ob.get("vertical_visibility_ft") is not None:
        s.ceiling_ft, s.ceiling_status = ob["vertical_visibility_ft"], CEIL_KNOWN_NUMERIC
    else:
        s.ceiling_status = CEIL_KNOWN_UNLIMITED
    s.weather, s.weather_status = list(ob.get("weather") or []), WX_KNOWN
    if ob.get("altimeter_inhg") is not None:
        s.qnh_inhg, s.qnh_status = ob["altimeter_inhg"], QNH_KNOWN
    return s


# ---------------------------------------------------------------------------
# Present-weather normalizer (5.6)
# ---------------------------------------------------------------------------

# Class buckets keyed by phenomenon (2-letter WMO code). DZ is LIQUID here (a
# forecast RA when DZ was observed is a `liquid` hit -- sec 5.6.4), even though
# wxcodes treats DZ as an obscuration for its SEVERITY taxonomy; this is a separate,
# verification-specific normalizer.
_LIQUID = {"RA", "DZ"}
_FROZEN = {"SN", "SG", "PL", "GR", "GS", "IC"}
_OBSCURATION = {"FG", "BR", "HZ", "FU", "VA", "DU", "SA", "PY", "PO"}
_OTHER = {"TS", "SQ", "FC", "SS", "DS", "UP"}
_DESCRIPTORS = {"MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ"}


def _pairs(code: str) -> list[str]:
    return [code[i:i + 2] for i in range(0, len(code) - 1, 2)]


def normalize_weather(tokens: list[str]) -> tuple[set[str], set[str]]:
    """(atomic_events, scoring_events). Atomic keys are `class:CODE`
    (`liquid:RA`, `freezing:RA`, `frozen:SN`, `obscuration:FG`, `other:TS`); a
    compound TSRA yields both `other:TS` and `liquid:RA`. scoring_events is the
    CLASS-level set. Intensity (+/-) and proximity (VC) are stripped for matching;
    FZ and TS are class-changing and preserved. NSW / empty -> empty sets."""
    atomic: set[str] = set()
    for raw in tokens:
        tok = raw
        if tok in ("NSW", ""):
            continue
        for pre in ("+", "-", "VC"):
            if tok.startswith(pre):
                tok = tok[len(pre):]
        codes = _pairs(tok)
        freezing = "FZ" in codes
        if "TS" in codes:
            atomic.add("other:TS")
        for c in codes:
            if c in _DESCRIPTORS:
                continue
            if c in _LIQUID:
                atomic.add(f"{'freezing' if freezing else 'liquid'}:{c}")
            elif c in _FROZEN:
                atomic.add(f"frozen:{c}")
            elif c in _OBSCURATION:
                atomic.add(f"obscuration:{c}")
            elif c in _OTHER:
                atomic.add(f"other:{c}")
    scoring = {a.split(":", 1)[0] for a in atomic}
    return atomic, scoring


# ---------------------------------------------------------------------------
# Absolutizer + opportunities + resolver (5.2)
# ---------------------------------------------------------------------------

_GTYPE = {None: "INITIAL", "FM": "FM", "BECMG": "BECMG", "TEMPO": "TEMPO", "PROB": "PROB"}


class AbsGroup(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    group_index: int             # 0 = prevailing, then change groups in TAF order
    group_type: str              # INITIAL | FM | BECMG | TEMPO | PROB
    probability: int | None
    start: datetime              # absolute naive UTC
    end: datetime | None         # absolute naive UTC; None for FM (open until next FM / valid_to)
    group: TafGroup


class Opportunity(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    group_index: int
    group_type: str
    probability: int | None
    bin_start: datetime
    bin_end: datetime
    interval_start: datetime     # active-interval intersection with the bin
    interval_end: datetime
    lead_hr: int                 # whole hours from valid_from (0-based)
    role: str = "baseline"       # baseline (prevailing timeline) | overlay (TEMPO/PROB)
    alternate_indices: list[int] = []   # BECMG-becoming groups scored BEST-OF with the baseline


def _abs(anchor: datetime, day: int, hour: int, minute: int = 0) -> datetime:
    """Relative day/hour(/min) -> absolute naive UTC against the window anchor. A day
    less than the anchor's wraps to the next month; hour 24 = next-day midnight. The
    window spans <=30 h, so at most one month/year rollover."""
    add = 0
    if hour == 24:
        hour, add = 0, 1
    year, month = anchor.year, anchor.month
    if day < anchor.day:
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return datetime(year, month, day, hour, minute) + timedelta(days=add)


def absolute_validity(taf: TafObs, issue_ref: datetime) -> tuple[datetime, datetime, datetime]:
    """(issue_time, valid_from, valid_to) as absolute naive UTC. `issue_ref` supplies
    the issue DATE (year/month/day) -- a TAF carries only day+time, so the caller
    anchors the calendar (AWC epoch, or a supplied --valid date). Month/year rollover
    and hour-24 (midnight end) are handled by _abs."""
    issue = datetime(issue_ref.year, issue_ref.month, taf.issue_day,
                     taf.issue_time.hour, taf.issue_time.minute)
    valid_from = _abs(issue, taf.valid_from_day, taf.valid_from_hour)
    valid_to = _abs(valid_from, taf.valid_to_day, taf.valid_to_hour)
    return issue, valid_from, valid_to


def absolutize(taf: TafObs, valid_from: datetime, valid_to: datetime) -> list[AbsGroup]:
    """Every group -> an AbsGroup with absolute naive-UTC start/end. Prevailing spans
    the whole window; FM starts at its exact minute (end None, resolved in
    opportunities); BECMG/TEMPO/PROB carry a start..end window."""
    out = [AbsGroup(group_index=0, group_type="INITIAL", probability=None,
                    start=valid_from, end=valid_to, group=taf.prevailing)]
    for i, g in enumerate(taf.groups, start=1):
        gt = _GTYPE.get(g.change_type, "TEMPO")
        if g.change_type == "FM":
            start = _abs(valid_from, g.from_day, g.from_hour, g.from_minute or 0)
            out.append(AbsGroup(group_index=i, group_type="FM", probability=g.probability,
                                start=start, end=None, group=g))
        else:
            start = _abs(valid_from, g.from_day, g.from_hour, 0)
            end = _abs(valid_from, g.to_day, g.to_hour, 0)
            out.append(AbsGroup(group_index=i, group_type=gt, probability=g.probability,
                                start=start, end=end, group=g))
    return out


def _baseline_segments(abs_groups: list[AbsGroup], valid_from: datetime, valid_to: datetime):
    """The EVOLVING prevailing timeline: [(start, end, governing AbsGroup), ...],
    contiguous over [valid_from, valid_to). The prevailing group evolves through change
    groups -- an FM replaces at its START, and a BECMG BECOMES the prevailing group at
    its END time (AFMAN 15-124: post-valid-time the BECMG conditions prevail). Each
    segment is LABELLED by the group that now governs it; the resolved STATE is the fully
    evolved baseline (`_baseline_state_at`). TEMPO/PROB are overlays, not here."""
    changepoints = [(valid_from, abs_groups[0])]                      # INITIAL prevailing
    for g in abs_groups:
        if g.group_type == "FM":
            changepoints.append((g.start, g))
        elif g.group_type == "BECMG" and g.end is not None:
            changepoints.append((g.end, g))                          # effective at its END
    changepoints.sort(key=lambda x: x[0])
    bounds = [max(t, valid_from) for t, _ in changepoints] + [valid_to]
    segs = []
    for i, (_, g) in enumerate(changepoints):
        a, b = max(bounds[i], valid_from), min(bounds[i + 1], valid_to)
        if b > a:
            segs.append((a, b, g))
    return segs


def opportunities(taf: TafObs, valid_from: datetime, valid_to: datetime) -> list[Opportunity]:
    """Every (group, bin-intersection) whose active interval touches a UTC clock-hour
    bin of [valid_from, valid_to). NOTHING is discarded: an FM at HH45 yields TWO
    opportunities in its bin (45-min predecessor + 15-min FM); overlapping
    BECMG/TEMPO add their own. Selection/weighting is each scorer's policy, not here."""
    abs_groups = absolutize(taf, valid_from, valid_to)
    segs = _baseline_segments(abs_groups, valid_from, valid_to)
    becmgs = [g for g in abs_groups if g.group_type == "BECMG" and g.end is not None]
    overlays = [g for g in abs_groups if g.group_type in ("TEMPO", "PROB")]

    out: list[Opportunity] = []
    h = valid_from
    while h < valid_to:
        nxt = h + timedelta(hours=1)
        lead = int((h - valid_from).total_seconds() // 3600)
        # baseline opportunities (the evolving prevailing timeline) intersecting this bin
        for a, b, g in segs:
            s, e = max(a, h), min(b, nxt)
            if e > s:
                # a BECMG whose TRANSITION window overlaps this interval is a best-of
                # alternate: during the "becoming" window either the old OR the new
                # conditions are acceptable (scored best-of, NOT double-counted).
                alts = [bg.group_index for bg in becmgs if bg.start < e and bg.end > s]
                out.append(Opportunity(group_index=g.group_index, group_type=g.group_type,
                                       probability=g.probability, bin_start=h, bin_end=nxt,
                                       interval_start=s, interval_end=e, lead_hr=lead,
                                       role="baseline", alternate_indices=alts))
        # overlay opportunities (TEMPO / PROB) -- their own rows, as before
        for g in overlays:
            if g.end is None:
                continue
            s, e = max(g.start, h), min(g.end, nxt)
            if e > s:
                out.append(Opportunity(group_index=g.group_index, group_type=g.group_type,
                                       probability=g.probability, bin_start=h, bin_end=nxt,
                                       interval_start=s, interval_end=e, lead_hr=lead,
                                       role="overlay"))
        h = nxt
    return out


def _baseline_state_at(abs_groups: list[AbsGroup], t: datetime, exclude_index: int) -> State:
    """The prevailing baseline at instant t: prevailing, replaced by the latest FM
    with start <= t (self-contained), then overlaid by every BECMG COMPLETE by t
    (end <= t). Excludes the group being resolved so a BECMG/TEMPO doesn't overlay
    itself here."""
    state = _forecast_state(abs_groups[0].group, base=None)     # prevailing
    fms = [g for g in abs_groups
           if g.group_type == "FM" and g.start <= t and g.group_index != exclude_index]
    if fms:
        state = _forecast_state(max(fms, key=lambda g: g.start).group, base=None)
    becmgs = sorted(
        (g for g in abs_groups if g.group_type == "BECMG" and g.group_index != exclude_index
         and g.end is not None and g.end <= t),
        key=lambda g: g.end,
    )
    for b in becmgs:
        state = _forecast_state(b.group, base=state)
    return state


def resolve_group_state(taf: TafObs, group_index: int, valid_from: datetime,
                        valid_to: datetime) -> State:
    """The fully-inherited State of ONE group (sec 7 atomic unit). Prevailing/FM are
    self-contained; BECMG/TEMPO/PROB overlay their explicit fields on the concurrent
    baseline."""
    abs_groups = absolutize(taf, valid_from, valid_to)
    target = next(g for g in abs_groups if g.group_index == group_index)
    if target.group_type in ("INITIAL", "FM"):
        return _forecast_state(target.group, base=None)
    base = _baseline_state_at(abs_groups, target.start, exclude_index=group_index)
    return _forecast_state(target.group, base=base)


class ForecastState(BaseModel):
    """The set of acceptable states at an instant: the prevailing baseline plus any
    active TEMPO/PROB alternates (never mutating prevailing)."""
    prevailing: State
    alternates: list[State] = []
    groups_active: list[int] = []       # group_index of every group active at t


def forecast_state(taf: TafObs, t: datetime, *, valid_from: datetime,
                   valid_to: datetime) -> ForecastState:
    """Prevailing baseline at t + TEMPO/PROB alternates whose window contains t."""
    abs_groups = absolutize(taf, valid_from, valid_to)
    prevailing = _baseline_state_at(abs_groups, t, exclude_index=-1)
    active = []
    # baseline group active at t
    for a, b, g in _baseline_segments(abs_groups, valid_from, valid_to):
        if a <= t < b:
            active.append(g.group_index)
    alternates = []
    for g in abs_groups:
        if g.group_type in ("TEMPO", "PROB") and g.end is not None and g.start <= t < g.end:
            alternates.append(_forecast_state(g.group, base=prevailing))
            active.append(g.group_index)
    return ForecastState(prevailing=prevailing, alternates=alternates, groups_active=active)


# ---------------------------------------------------------------------------
# Category classifiers + station profile (5.3)
# ---------------------------------------------------------------------------

_CAT_LETTERS = "ABCDE"          # worst -> best (A=0 .. E=4)


class CategoryBand(BaseModel):
    """One ordered category band. [lo, hi) with None = open end. `lo`/`hi` are feet
    for ceiling bands, statute miles for visibility bands."""
    id: str
    lo: float | None             # inclusive lower edge (None = -inf)
    hi: float | None             # exclusive upper edge (None = +inf)


class StationProfile(BaseModel):
    """Versioned per-station scoring profile (5.3 / sec 6). SEPARATE sections for the
    two classifier contracts. v1 minima are the fixed provisional default (200 ft /
    1/2 SM ~ 800 m), overridable per station; `provisional` rides into run metadata."""
    station: str
    provisional: bool = True
    version: str = "v1-provisional"
    source_reference: str | None = None
    # DAF flight-category inputs (Attachment 4)
    landing_min_ceiling_ft: int = 200
    landing_min_vis_m: int = 800
    use_oconus_vis_substitutions: bool = False
    oconus_provenance: str | None = None
    # TAFVER installation category ladders (ordered worst -> best)
    tafver_ceiling_bands: list[CategoryBand] = []
    tafver_vis_bands: list[CategoryBand] = []


def default_profile(station: str) -> StationProfile:
    """A provisional profile whose TAFVER ladders mirror the DAF A4.1 boundaries at
    the fixed default minima -- enough to exercise all five categories. Real
    installation tables swap the band lists in later with no API change."""
    ceil = [
        CategoryBand(id="A", lo=None, hi=200),
        CategoryBand(id="B", lo=200, hi=700),
        CategoryBand(id="C", lo=700, hi=1000),
        CategoryBand(id="D", lo=1000, hi=2000),
        CategoryBand(id="E", lo=2000, hi=None),
    ]
    vis = [        # statute miles: 800 m ~ 0.5, 3200 m ~ 2, 4800 m ~ 3
        CategoryBand(id="A", lo=None, hi=0.5),
        CategoryBand(id="B", lo=0.5, hi=2.0),
        CategoryBand(id="D", lo=2.0, hi=3.0),
        CategoryBand(id="E", lo=3.0, hi=None),
    ]
    return StationProfile(station=station, tafver_ceiling_bands=ceil, tafver_vis_bands=vis)


def validate_profile(profile: StationProfile) -> list[str]:
    """Pre-run profile validation: each band ladder must be contiguous and exhaustive
    over its domain (first lo open, last hi open, each hi == the next lo). Returns
    findings (empty = valid)."""
    out: list[str] = []
    for name, bands in (("ceiling", profile.tafver_ceiling_bands), ("vis", profile.tafver_vis_bands)):
        if not bands:
            out.append(f"tafver {name} bands: empty")
            continue
        if bands[0].lo is not None:
            out.append(f"tafver {name} bands: first band lo must be open (-inf), got {bands[0].lo}")
        if bands[-1].hi is not None:
            out.append(f"tafver {name} bands: last band hi must be open (+inf), got {bands[-1].hi}")
        for a, b in zip(bands, bands[1:]):
            if a.hi != b.lo:
                out.append(f"tafver {name} bands: gap/overlap between {a.id} (hi={a.hi}) "
                           f"and {b.id} (lo={b.lo})")
    if profile.landing_min_ceiling_ft <= 0 or profile.landing_min_vis_m <= 0:
        out.append("landing minima must be positive")
    return out


def _band_id(value: float, bands: list[CategoryBand]) -> str | None:
    for band in bands:
        lo_ok = band.lo is None or value >= band.lo
        hi_ok = band.hi is None or value < band.hi
        if lo_ok and hi_ok:
            return band.id
    return None


def tafver_ceiling_category(ceiling_ft: int | None, ceiling_status: str,
                            profile: StationProfile) -> str | None:
    """TAFVER ceiling category id from the profile ladder. Unlimited -> the top
    (last) band; unknown -> None (never classified)."""
    if ceiling_status == CEIL_KNOWN_UNLIMITED:
        return profile.tafver_ceiling_bands[-1].id
    if ceiling_status != CEIL_KNOWN_NUMERIC or ceiling_ft is None:
        return None
    return _band_id(ceiling_ft, profile.tafver_ceiling_bands)


def tafver_visibility_category(vis_sm: float | None, vis_flag: str | None, vis_status: str,
                               profile: StationProfile) -> str | None:
    """TAFVER visibility category id from the profile ladder. Unlimited (P6SM /
    9999 m / CAVOK) -> the top band; unknown -> None."""
    if vis_status == VIS_KNOWN_UNLIMITED:
        return profile.tafver_vis_bands[-1].id
    if vis_status != VIS_KNOWN_NUMERIC or vis_sm is None:
        return None
    return _band_id(vis_sm, profile.tafver_vis_bands)


def _cig_index(ceiling_ft: int | None, ceiling_status: str, profile: StationProfile) -> int | None:
    if ceiling_status == CEIL_KNOWN_UNLIMITED:
        return 4
    if ceiling_status != CEIL_KNOWN_NUMERIC or ceiling_ft is None:
        return None
    lmc = profile.landing_min_ceiling_ft
    if ceiling_ft >= 2000:
        return 4
    if ceiling_ft >= 1000:
        return 3
    if ceiling_ft >= lmc + 500:
        return 2
    if ceiling_ft >= lmc:
        return 1
    return 0


def _vis_index(vis_m: int | None, vis_status: str, profile: StationProfile) -> int | None:
    if vis_status == VIS_KNOWN_UNLIMITED:
        return 4
    if vis_status != VIS_KNOWN_NUMERIC or vis_m is None:
        return None
    e_thr, d_thr = (5000, 3000) if profile.use_oconus_vis_substitutions else (4800, 3200)
    lmv = profile.landing_min_vis_m
    d_floor = max(d_thr, lmv)
    if vis_m >= e_thr:
        return 4
    if vis_m >= d_floor:
        return 3        # D/C share this vis band; C comes from the ceiling via lower-of
    if vis_m >= lmv:
        return 1        # A4.1 B row: < d_thr, >= vis min
    return 0


def daf_flight_category(ceiling_ft: int | None, ceiling_status: str, vis_m: int | None,
                        vis_status: str, *, profile: StationProfile) -> str | None:
    """DAF flight category (Attachment 4 Table A4.1): the LOWER (worse) of the
    ceiling-derived and vis-derived categories (3.4.2.7 Note 2). Any unknown input ->
    None (unavailable), never a default."""
    cig = _cig_index(ceiling_ft, ceiling_status, profile)
    vis = _vis_index(vis_m, vis_status, profile)
    if cig is None or vis is None:
        return None
    return _CAT_LETTERS[min(cig, vis)]


# ---------------------------------------------------------------------------
# Truth ingestion + intervalization + two-view hourly builder (5.4 / 5.5)
# ---------------------------------------------------------------------------

class TruthPolicy(BaseModel):
    """Versioned truth-policy constants (sec 5.5 / 6). Persisted with a hash per run."""
    name: str = "truth-v1"
    # v2: conservative view no longer reads a missing vis/ceiling as "unlimited" -- an
    # unknown field routes to unavailable (unscored). Changes scored output -> new hash.
    version: str = "2"
    max_hold_min: int = 90          # an ob's state holds this long past its time before a gap
    predominant_min_min: int = 30   # >= this many aggregate minutes to be predominant (3.4.2.8)


class _Interval(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    start: datetime
    end: datetime
    state: State | None             # None = coverage gap
    report_time: datetime | None
    report_type: str | None


def intervalize(obs: list[dict], valid_from: datetime, valid_to: datetime,
                policy: TruthPolicy) -> list[_Interval]:
    """Continuous-window intervalization (5.4). Each ob holds from its time until
    min(next ob, ob + max_hold); beyond max_hold a coverage GAP interval begins. The
    last ob at/before valid_from is carry-in; the first ob at/after valid_to only
    terminates the last in-window interval (never scored)."""
    rows = sorted(obs, key=lambda o: o["obs_time"])
    hold = timedelta(minutes=policy.max_hold_min)
    intervals: list[_Interval] = []
    for i, ob in enumerate(rows):
        t = ob["obs_time"]
        nxt = rows[i + 1]["obs_time"] if i + 1 < len(rows) else valid_to
        state_end = min(nxt, t + hold)
        if state_end > t:
            intervals.append(_Interval(start=t, end=state_end, state=observed_state(ob),
                                       report_time=t, report_type=ob.get("report_type")))
        if nxt > t + hold:      # a gap opens after this ob's hold expires
            intervals.append(_Interval(start=t + hold, end=nxt, state=None,
                                        report_time=None, report_type=None))
    # clip everything to the scored window
    clipped: list[_Interval] = []
    for iv in intervals:
        a, b = max(iv.start, valid_from), min(iv.end, valid_to)
        if b > a:
            clipped.append(iv.model_copy(update={"start": a, "end": b}))
    return clipped


class FieldVal(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    value: object = None
    status: str = STATUS_UNKNOWN
    minutes: float = 0.0
    sources: list[datetime] = []


class HourTruth(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    hour: datetime
    lead_hr: int
    status: str                     # available | unavailable
    reason: str | None = None
    coverage_minutes: float = 0.0
    gap_minutes: float = 0.0
    cons: dict[str, FieldVal] = {}          # conservative/union view, per field
    pred: dict[str, FieldVal] = {}          # per-field predominant view
    temporaries: dict[str, list] = {}       # transient values that never became predominant


class CoverageManifest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    expected_hours: int
    observed_hours: int
    missing_hours: int
    gap_minutes: float
    first_ob: datetime | None
    last_ob: datetime | None
    n_routine: int
    n_speci: int
    sources: list[str]


_CONS_FIELDS = ("ceiling", "vis", "altimeter", "wind_speed", "wind_dir", "wind_gust", "weather")


def _minutes(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60.0


def _conservative(pieces: list[tuple[_Interval, float]]) -> dict[str, FieldVal]:
    """Pessimistic union: lowest cig/vis/altimeter, max sustained wind (+ its dir),
    max gust, union of weather. `pieces` = (interval, minutes) for obs intervals."""
    cons: dict[str, FieldVal] = {}

    # ceiling: lowest numeric; unlimited ONLY if EVERY covering ob is unlimited; if any
    # covering ob's ceiling is unknown (and none numeric), the field is unknown -- never a
    # default. Absence-of-a-ceiling from a partial ob must not read as clear skies.
    numeric = [(iv.state.ceiling_ft, iv) for iv, _ in pieces if iv.state.ceiling_status == CEIL_KNOWN_NUMERIC]
    if numeric:
        low = min(numeric, key=lambda x: x[0])
        cons["ceiling"] = FieldVal(value=low[0], status=CEIL_KNOWN_NUMERIC,
                                   sources=[low[1].report_time])
    elif all(iv.state.ceiling_status == CEIL_KNOWN_UNLIMITED for iv, _ in pieces):
        cons["ceiling"] = FieldVal(value=None, status=CEIL_KNOWN_UNLIMITED,
                                   sources=[iv.report_time for iv, _ in pieces])
    else:
        cons["ceiling"] = FieldVal(status=STATUS_UNKNOWN)

    # visibility: lowest numeric meters; unlimited ONLY if every covering ob is unlimited;
    # else (some covering ob never reported vis) the field is unknown, not "unrestricted".
    vnum = [(iv.state.vis_m, iv) for iv, _ in pieces
            if iv.state.vis_status == VIS_KNOWN_NUMERIC and iv.state.vis_m is not None]
    if vnum:
        low = min(vnum, key=lambda x: x[0])
        cons["vis"] = FieldVal(value=low[0], status=VIS_KNOWN_NUMERIC, sources=[low[1].report_time])
    elif all(iv.state.vis_status == VIS_KNOWN_UNLIMITED for iv, _ in pieces):
        cons["vis"] = FieldVal(value=None, status=VIS_KNOWN_UNLIMITED,
                               sources=[iv.report_time for iv, _ in pieces])
    else:
        cons["vis"] = FieldVal(status=STATUS_UNKNOWN)

    # altimeter: lowest observed
    alt = [(iv.state.qnh_inhg, iv) for iv, _ in pieces if iv.state.qnh_status == QNH_KNOWN]
    if alt:
        low = min(alt, key=lambda x: x[0])
        cons["altimeter"] = FieldVal(value=low[0], status=QNH_KNOWN, sources=[low[1].report_time])
    else:
        cons["altimeter"] = FieldVal(status=STATUS_UNKNOWN)

    # wind: max sustained (ties -> earliest ob), dir from that ob; max gust
    spd = [(iv.state.wind_speed, iv) for iv, _ in pieces if iv.state.wind_speed_status == SPD_NUMERIC]
    if spd:
        top = max(spd, key=lambda x: (x[0], -x[1].report_time.timestamp()))
        cons["wind_speed"] = FieldVal(value=top[0], status=SPD_NUMERIC, sources=[top[1].report_time])
        cons["wind_dir"] = FieldVal(value=top[1].state.wind_dir, status=top[1].state.wind_dir_status,
                                    sources=[top[1].report_time])
    else:
        cons["wind_speed"] = FieldVal(status=STATUS_UNKNOWN)
        cons["wind_dir"] = FieldVal(status=STATUS_UNKNOWN)
    gusts = [(iv.state.wind_gust, iv) for iv, _ in pieces if iv.state.gust_status == GUST_PRESENT]
    if gusts:
        top = max(gusts, key=lambda x: x[0])
        cons["wind_gust"] = FieldVal(value=top[0], status=GUST_PRESENT, sources=[top[1].report_time])
    else:
        cons["wind_gust"] = FieldVal(value=None, status=GUST_KNOWN_ABSENT)

    # weather: union of all observed tokens
    wx: set[str] = set()
    for iv, _ in pieces:
        wx |= set(iv.state.weather)
    cons["weather"] = FieldVal(value=sorted(wx), status=WX_KNOWN,
                               sources=[iv.report_time for iv, _ in pieces])
    return cons


def _predominant(pieces: list[tuple[_Interval, float]], key, policy: TruthPolicy) -> tuple[FieldVal, list]:
    """Greatest-aggregate-minutes value for one field (3.4.2.8 extended per-field).
    `key(state) -> hashable value or None`. Predominant if it holds >= the policy
    minute floor; else the field is unavailable and its values are temporaries."""
    agg: dict = {}
    times: dict = {}
    for iv, mins in pieces:
        v = key(iv.state)
        if v is None:
            continue
        agg[v] = agg.get(v, 0.0) + mins
        times.setdefault(v, []).append(iv.report_time)
    if not agg:
        return FieldVal(status=STATUS_UNKNOWN), []
    best = max(agg, key=lambda v: agg[v])
    if agg[best] >= policy.predominant_min_min:
        temporaries = [v for v in agg if v != best]
        return FieldVal(value=best, status="known", minutes=agg[best], sources=times[best]), temporaries
    return FieldVal(status="unavailable"), list(agg.keys())


def build_truth(obs: list[dict], valid_from: datetime, valid_to: datetime,
                *, policy: TruthPolicy | None = None) -> tuple[list[HourTruth], CoverageManifest]:
    """The ONE intervalized truth builder, TWO views (5.5). Returns per-hour truth
    over [valid_from, valid_to) + a coverage manifest. An hour with no obs coverage
    is status=unavailable (no_obs / coverage_gap), never a zero-point miss."""
    policy = policy or TruthPolicy()
    intervals = intervalize(obs, valid_from, valid_to, policy)

    hours: list[HourTruth] = []
    h = valid_from
    while h < valid_to:
        nxt = h + timedelta(hours=1)
        lead = int((h - valid_from).total_seconds() // 3600)
        pieces: list[tuple[_Interval, float]] = []
        gap_min = 0.0
        for iv in intervals:
            a, b = max(iv.start, h), min(iv.end, nxt)
            if b <= a:
                continue
            mins = _minutes(a, b)
            if iv.state is None:
                gap_min += mins
            else:
                pieces.append((iv.model_copy(update={"start": a, "end": b}), mins))
        cover_min = sum(m for _, m in pieces)
        if cover_min <= 0:
            reason = "coverage_gap" if gap_min > 0 else "no_obs"
            hours.append(HourTruth(hour=h, lead_hr=lead, status="unavailable", reason=reason,
                                   coverage_minutes=0.0, gap_minutes=gap_min))
            h = nxt
            continue
        cons = _conservative(pieces)
        pred: dict[str, FieldVal] = {}
        temps: dict[str, list] = {}
        keymap = {
            "ceiling": lambda s: s.ceiling_ft if s.ceiling_status == CEIL_KNOWN_NUMERIC else (
                "unlimited" if s.ceiling_status == CEIL_KNOWN_UNLIMITED else None),
            "vis": lambda s: s.vis_m if s.vis_status == VIS_KNOWN_NUMERIC else (
                "unlimited" if s.vis_status == VIS_KNOWN_UNLIMITED else None),
            "altimeter": lambda s: s.qnh_inhg if s.qnh_status == QNH_KNOWN else None,
            "wind_speed": lambda s: s.wind_speed if s.wind_speed_status == SPD_NUMERIC else None,
            "wind_dir": lambda s: s.wind_dir if s.wind_dir_status == DIR_NUMERIC else None,
            "wind_gust": lambda s: s.wind_gust if s.gust_status == GUST_PRESENT else None,
            "weather": lambda s: tuple(sorted(s.weather)) if s.weather_status == WX_KNOWN else None,
        }
        for field, key in keymap.items():
            pv, tv = _predominant(pieces, key, policy)
            pred[field] = pv
            if tv:
                temps[field] = tv
        hours.append(HourTruth(hour=h, lead_hr=lead, status="available",
                               coverage_minutes=cover_min, gap_minutes=gap_min,
                               cons=cons, pred=pred, temporaries=temps))
        h = nxt

    in_window = [o for o in obs if valid_from <= o["obs_time"] < valid_to]
    manifest = CoverageManifest(
        expected_hours=len(hours),
        observed_hours=sum(1 for x in hours if x.status == "available"),
        missing_hours=sum(1 for x in hours if x.status == "unavailable"),
        gap_minutes=sum(x.gap_minutes for x in hours),
        first_ob=min((o["obs_time"] for o in in_window), default=None),
        last_ob=max((o["obs_time"] for o in in_window), default=None),
        n_routine=sum(1 for o in in_window if o.get("report_type") != "SPECI"),
        n_speci=sum(1 for o in in_window if o.get("report_type") == "SPECI"),
        sources=sorted({o.get("source") for o in in_window if o.get("source")}),
    )
    return hours, manifest


def _vis_from_m(vis_m) -> tuple[int | None, float | None, str]:
    """Truth-view vis is keyed in METERS; rebuild (vis_m, vis_sm, status). 9999 m
    (>= 10 km) is unlimited; a real distance converts to SM via the Table 8.1 seam so
    TAFVER's SM-band classifier has what it needs."""
    if vis_m is None:
        return None, None, STATUS_UNKNOWN
    if vis_m >= 9999:
        return None, None, VIS_KNOWN_UNLIMITED
    sm, _, _ = _parse_vis(str(vis_m))
    return vis_m, sm, VIS_KNOWN_NUMERIC


def predominant_state(h: HourTruth) -> State:
    """Reconstruct a State from an HourTruth's PREDOMINANT view (the amend/skill truth
    view). Predominant FieldVals carry status `known`/`unlimited`, so this maps them
    back to the State availability statuses."""
    s = State()
    c = h.pred.get("ceiling")
    if c and c.status == "known":
        if c.value == "unlimited":
            s.ceiling_status = CEIL_KNOWN_UNLIMITED
        else:
            s.ceiling_ft, s.ceiling_status = c.value, CEIL_KNOWN_NUMERIC
    v = h.pred.get("vis")
    if v and v.status == "known":
        if v.value == "unlimited":
            s.vis_status = VIS_KNOWN_UNLIMITED
        else:
            s.vis_m, s.vis_sm, s.vis_status = _vis_from_m(v.value)
    ws = h.pred.get("wind_speed")
    if ws and ws.status == "known":
        s.wind_speed, s.wind_speed_status = ws.value, SPD_NUMERIC
    wd = h.pred.get("wind_dir")
    if wd and wd.status == "known":
        s.wind_dir, s.wind_dir_status = wd.value, DIR_NUMERIC
    wg = h.pred.get("wind_gust")
    if wg and wg.status == "known":
        s.wind_gust, s.gust_status = wg.value, GUST_PRESENT
    elif ws and ws.status == "known":
        s.gust_status = GUST_KNOWN_ABSENT
    al = h.pred.get("altimeter")
    if al and al.status == "known":
        s.qnh_inhg, s.qnh_status = al.value, QNH_KNOWN
    wx = h.pred.get("weather")
    if wx and wx.status == "known":
        s.weather, s.weather_status = list(wx.value), WX_KNOWN
    return s


def conservative_state(h: HourTruth) -> State:
    """Reconstruct a State from an HourTruth's CONSERVATIVE/union view (the TAFVER truth
    view). Conservative FieldVals carry the real availability status constants."""
    s = State()
    c = h.cons.get("ceiling")
    if c:
        if c.status == CEIL_KNOWN_NUMERIC:
            s.ceiling_ft, s.ceiling_status = c.value, CEIL_KNOWN_NUMERIC
        elif c.status == CEIL_KNOWN_UNLIMITED:
            s.ceiling_status = CEIL_KNOWN_UNLIMITED
    v = h.cons.get("vis")
    if v:
        if v.status == VIS_KNOWN_NUMERIC:
            s.vis_m, s.vis_sm, s.vis_status = _vis_from_m(v.value)
        elif v.status == VIS_KNOWN_UNLIMITED:
            s.vis_status = VIS_KNOWN_UNLIMITED
    ws = h.cons.get("wind_speed")
    if ws and ws.status == SPD_NUMERIC:
        s.wind_speed, s.wind_speed_status = ws.value, SPD_NUMERIC
    wd = h.cons.get("wind_dir")
    if wd and wd.status == DIR_NUMERIC:
        s.wind_dir, s.wind_dir_status = wd.value, DIR_NUMERIC
    wg = h.cons.get("wind_gust")
    if wg and wg.status == GUST_PRESENT:
        s.wind_gust, s.gust_status = wg.value, GUST_PRESENT
    elif ws and ws.status == SPD_NUMERIC:
        s.gust_status = GUST_KNOWN_ABSENT
    al = h.cons.get("altimeter")
    if al and al.status == QNH_KNOWN:
        s.qnh_inhg, s.qnh_status = al.value, QNH_KNOWN
    wx = h.cons.get("weather")
    if wx and wx.status == WX_KNOWN:
        s.weather, s.weather_status = list(wx.value or []), WX_KNOWN
    return s
