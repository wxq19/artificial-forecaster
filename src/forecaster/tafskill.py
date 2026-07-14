"""Scorer 3 -- skill (scoring-design sec 9). PURE: no duckdb/SQL, no network, no
matplotlib, no LLM. Imports tafstate primitives (+ the predominant-view reconstruction
from tafamend) and reads METAR dicts.

Three orthogonal axes over the SAME resolved states and truth views:

- Axis 1 (sec 9.1) -- continuous element errors on the PREDOMINANT view: signed +
  absolute error per (hour, element) for wind/ceiling/vis, per-GROUP QNH, per-TAF
  TX/TN. Forecast side is the PREVAILING state only (TEMPO credit lives on axis 2).
- Axis 2 (sec 9.2) -- event contingency on the UNION/conservative view: a versioned
  event catalog; hit/miss/false-alarm/correct-negative cells -> POD/FAR/CSI/HSS +
  deterministic min-cost episode timing.
- Axis 3 (sec 9.3) -- ordinal category distance on the predominant view: DAF flight
  category series -> MACE + worst excursion + signed mean.

TAFVER awards 1/0 inside a tolerance and the amend scorer counts threshold crossings;
neither sees the MAGNITUDE of a miss, and percent-correct is inflated by benign
weather. This scorer fills both blind spots and uniquely covers TX/TN and QNH drift.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from pydantic import BaseModel

from forecaster.metar import _parse_vis
from forecaster.tafparse import TafObs
from forecaster.tafstate import (
    CEIL_KNOWN_NUMERIC, CEIL_KNOWN_UNLIMITED, DIR_NUMERIC, GUST_INHERITED_ABSENT,
    GUST_KNOWN_ABSENT, GUST_PRESENT, SPD_NUMERIC, VIS_KNOWN_NUMERIC,
    VIS_KNOWN_UNLIMITED, State, StationProfile, TruthPolicy, _abs, absolutize,
    build_truth, conservative_state as _conservative_state, daf_flight_category,
    default_profile, forecast_state, normalize_weather,
    predominant_state as _predominant_state,
)

_CAT_LETTERS = "ABCDE"          # worst -> best (A=0 .. E=4), mirrors tafstate
_LEAD_BINS = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30)]


class SkillPolicy(BaseModel):
    """Versioned skill constants (sec 6/9). Changing any constant is a new policy
    version/hash and therefore a new run; old statistics stay reproducible."""
    name: str = "skill-v1"
    version: str = "1"
    catalog_version: str = "events-v1"
    dir_speed_min_kt: int = 6            # dir error eligible only when BOTH speeds > this
    vis_cap_sm: float = 6.0             # P6SM / 9999 m are at-cap; never a small artefact
    temp_window_hours: int = 24        # TX/TN period anchored on original_cycle_start
    temp_coverage_floor_hours: int = 18  # < this many hours with a temp -> sparse_temp_coverage
    match_window_h: int = 3            # episode-match eligibility (sec 9.2)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ElementRow(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    grain: str                          # hour | group | taf
    hour: datetime | None = None
    lead_hr: int | None = None
    group_type: str | None = None
    element: str
    fcst_value: float | None = None
    obs_value: float | None = None
    signed_error: float | None = None
    abs_error: float | None = None
    status: str                         # scored | unavailable
    reason: str | None = None


class ElementStat(BaseModel):
    element: str
    bin: str                            # lead bin label or "overall"
    n: int
    bias: float | None = None
    mae: float | None = None


class EventHour(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    hour: datetime
    event: str
    fcst: bool
    obs: bool | None                    # None = obs not evaluable this hour (no cell)
    via_tempo: bool = False
    cell: str | None = None             # hit | miss | false_alarm | correct_negative


class Contingency(BaseModel):
    event: str
    a: int                              # hit
    b: int                              # false alarm
    c: int                              # miss
    d: int                              # correct negative
    pod: float | None = None
    far: float | None = None
    csi: float | None = None
    freq_bias: float | None = None
    hss: float | None = None


class Episode(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    event: str
    disposition: str                    # matched | missed | false_alarm
    obs_onset: datetime | None = None
    obs_end: datetime | None = None
    fcst_onset: datetime | None = None
    fcst_end: datetime | None = None
    onset_error_h: float | None = None  # fcst - obs; negative = early
    end_error_h: float | None = None


class TafSkillScore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    station: str
    valid_from: datetime
    valid_to: datetime
    catalog_version: str
    element_rows: list[ElementRow]
    element_stats: list[ElementStat]
    event_hours: list[EventHour]
    contingency: list[Contingency]
    episodes: list[Episode]
    category_series: list[list]         # [[hourZ, fcst_cat, obs_cat], ...]
    mace: float | None = None
    worst_excursion: dict | None = None  # {"hour":..., "delta":..., "fcst":..., "obs":...}
    signed_mace_mean: float | None = None
    hours_scored: int = 0
    hours_unavailable: int = 0


# ---------------------------------------------------------------------------
# Contingency math (pure -- scores one TAF and a pooled batch identically)
# ---------------------------------------------------------------------------

def contingency_scores(a: int, b: int, c: int, d: int) -> dict:
    """POD/FAR/CSI/freq_bias/HSS from a 2x2 table (sec 9.2). Zero denominators ->
    None, never 0/inf."""
    pod = a / (a + c) if (a + c) else None
    far = b / (a + b) if (a + b) else None
    csi = a / (a + b + c) if (a + b + c) else None
    freq_bias = (a + b) / (a + c) if (a + c) else None
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    hss = (2 * (a * d - b * c)) / denom if denom else None
    return {"pod": pod, "far": far, "csi": csi, "freq_bias": freq_bias, "hss": hss}


# ---------------------------------------------------------------------------
# Axis 1 -- continuous element errors (predominant view)
# ---------------------------------------------------------------------------

def _prevailing_at(taf: TafObs, hour: datetime, vf: datetime, vt: datetime) -> State:
    return forecast_state(taf, hour + timedelta(minutes=30), valid_from=vf, valid_to=vt).prevailing


def _signed_circular(f: int, o: int) -> int:
    """((f - o + 180) mod 360) - 180 in [-180, 180); negative = forecast CCW of obs."""
    return ((f - o + 180) % 360) - 180


def _capped_sm(state: State, cap: float) -> tuple[float | None, bool, bool]:
    """(capped_sm, at_cap, available). Unlimited -> (cap, True, True); numeric ->
    canonicalize via the Table 8.1 seam then cap; unknown -> (None, False, False)."""
    if state.vis_status == VIS_KNOWN_UNLIMITED:
        return cap, True, True
    if state.vis_status != VIS_KNOWN_NUMERIC:
        return None, False, False
    sm = state.vis_sm
    if sm is None and state.vis_m is not None:
        sm, _, _ = _parse_vis(str(state.vis_m))
    if sm is None:
        return None, False, False
    return min(sm, cap), sm >= cap, True


def _one_sided_unlimited(fc, oc) -> bool:
    """One side unlimited AND the other a real numeric (not unknown) -> one-sided."""
    return {fc, oc} == {CEIL_KNOWN_UNLIMITED, CEIL_KNOWN_NUMERIC}


def _row(element, hour, lead, gtype, f, o, err, status, reason=None, grain="hour") -> ElementRow:
    return ElementRow(grain=grain, hour=hour, lead_hr=lead, group_type=gtype, element=element,
                      fcst_value=f, obs_value=o, signed_error=err,
                      abs_error=None if err is None else abs(err), status=status, reason=reason)


def _axis1_hourly(taf, hours, vf, vt, policy) -> list[ElementRow]:
    rows: list[ElementRow] = []
    for h in hours:
        lead = h.lead_hr
        if h.status != "available":
            for el in ("wind_speed", "wind_dir", "wind_gust", "ceiling", "visibility"):
                rows.append(_row(el, h.hour, lead, None, None, None, None, "unavailable", h.reason))
            continue
        f = _prevailing_at(taf, h.hour, vf, vt)
        o = _predominant_state(h)
        gt = "PREVAILING"

        # wind speed (calm = numeric 0)
        if f.wind_speed_status == SPD_NUMERIC and o.wind_speed_status == SPD_NUMERIC:
            e = f.wind_speed - o.wind_speed
            rows.append(_row("wind_speed", h.hour, lead, gt, f.wind_speed, o.wind_speed, e, "scored"))
        else:
            rows.append(_row("wind_speed", h.hour, lead, gt, None, None, None, "unavailable",
                             "wind_speed_missing"))

        # wind direction: both numeric AND both speeds > floor
        both_num = f.wind_dir_status == DIR_NUMERIC and o.wind_dir_status == DIR_NUMERIC
        both_strong = ((f.wind_speed or 0) > policy.dir_speed_min_kt
                       and (o.wind_speed or 0) > policy.dir_speed_min_kt)
        if both_num and both_strong:
            e = _signed_circular(f.wind_dir, o.wind_dir)
            rows.append(_row("wind_dir", h.hour, lead, gt, f.wind_dir, o.wind_dir, e, "scored"))
        else:
            rows.append(_row("wind_dir", h.hour, lead, gt, None, None, None, "unavailable",
                             "light_or_vrb_or_missing"))

        # wind gust: both present, else occurrence mismatch / both-absent
        fg = f.gust_status == GUST_PRESENT
        og = o.gust_status == GUST_PRESENT
        if fg and og:
            e = f.wind_gust - o.wind_gust
            rows.append(_row("wind_gust", h.hour, lead, gt, f.wind_gust, o.wind_gust, e, "scored"))
        elif fg != og:
            rows.append(_row("wind_gust", h.hour, lead, gt, None, None, None, "unavailable",
                             "gust_occurrence_mismatch"))
        else:
            rows.append(_row("wind_gust", h.hour, lead, gt, None, None, None, "unavailable",
                             "no_gust_either_side"))

        # ceiling: both known_numeric
        fc, oc = f.ceiling_status, o.ceiling_status
        if fc == CEIL_KNOWN_NUMERIC and oc == CEIL_KNOWN_NUMERIC:
            e = f.ceiling_ft - o.ceiling_ft
            rows.append(_row("ceiling", h.hour, lead, gt, f.ceiling_ft, o.ceiling_ft, e, "scored"))
        elif fc == CEIL_KNOWN_UNLIMITED and oc == CEIL_KNOWN_UNLIMITED:
            rows.append(_row("ceiling", h.hour, lead, gt, None, None, None, "unavailable",
                             "both_unlimited"))
        elif _one_sided_unlimited(fc, oc):
            rows.append(_row("ceiling", h.hour, lead, gt, None, None, None, "unavailable",
                             "one_sided_unlimited"))
        else:
            rows.append(_row("ceiling", h.hour, lead, gt, None, None, None, "unavailable",
                             "ceiling_unknown"))

        # visibility: canonical SM, cap 6.0; both at cap -> zero information
        fsm, f_cap, f_ok = _capped_sm(f, policy.vis_cap_sm)
        osm, o_cap, o_ok = _capped_sm(o, policy.vis_cap_sm)
        if f_ok and o_ok and f_cap and o_cap:
            rows.append(_row("visibility", h.hour, lead, gt, fsm, osm, None, "unavailable",
                             "both_unrestricted"))
        elif f_ok and o_ok:
            rows.append(_row("visibility", h.hour, lead, gt, fsm, osm, fsm - osm, "scored"))
        else:
            rows.append(_row("visibility", h.hour, lead, gt, None, None, None, "unavailable",
                             "visibility_unknown"))
    return rows


def _group_governing_windows(taf, vf, vt):
    """Per non-TEMPO group with an explicit QNH: (group_type, qnh_inhg, start, end) --
    the span over which that group governs the prevailing QNH (sec 9.1)."""
    abs_groups = absolutize(taf, vf, vt)
    # ordered change points that re-set QNH: prevailing + FM + BECMG (not TEMPO/PROB)
    governing = [g for g in abs_groups if g.group_type in ("INITIAL", "FM", "BECMG")]
    starts = sorted(
        [(vf if g.group_type == "INITIAL" else (g.start if g.group_type == "FM" else g.end), g)
         for g in governing],
        key=lambda x: x[0],
    )
    out = []
    for i, (start, g) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else vt
        if "qnh" in g.group.explicit_fields and g.group.qnh_inhg is not None:
            out.append((g.group_type, g.group.qnh_inhg, max(start, vf), min(end, vt)))
    return out


def _axis1_qnh(taf, obs, vf, vt) -> list[ElementRow]:
    rows: list[ElementRow] = []
    for gtype, qnh, start, end in _group_governing_windows(taf, vf, vt):
        alts = [o["altimeter_inhg"] for o in obs
                if o.get("altimeter_inhg") is not None and start <= o["obs_time"] < end]
        if not alts:
            rows.append(_row("qnh", start, None, gtype, qnh, None, None, "unavailable",
                             "no_observed_altimeter", grain="group"))
            continue
        omin = min(alts)                # TAF QNH means "lowest expected during the period"
        e = round((qnh - omin) * 100) / 100
        rows.append(_row("qnh", start, None, gtype, qnh, omin, e, "scored", grain="group"))
    return rows


def _axis1_txtn(taf, obs, vf, vt, policy) -> list[ElementRow]:
    """TX/TN value + timing over the original_cycle_start 24 h temp window (sec 9.1)."""
    rows: list[ElementRow] = []
    tw_end = vf + timedelta(hours=policy.temp_window_hours)
    in_win = [o for o in obs if o.get("temp_c") is not None and vf <= o["obs_time"] < tw_end]
    covered = len({o["obs_time"].replace(minute=0, second=0, microsecond=0) for o in in_win})
    for element, tt, pick in (("temp_tx", taf.max_temp, max), ("temp_tn", taf.min_temp, min)):
        if tt is None:
            rows.append(_row(element, None, None, "TAF", None, None, None, "unavailable",
                             "no_forecast_extreme", grain="taf"))
            continue
        if covered < policy.temp_coverage_floor_hours or not in_win:
            rows.append(_row(element, None, None, "TAF", tt.temp_c, None, None, "unavailable",
                             "sparse_temp_coverage", grain="taf"))
            continue
        extreme = pick(o["temp_c"] for o in in_win)
        # earliest ob supplying the extreme (ties -> earliest)
        src = min((o for o in in_win if o["temp_c"] == extreme), key=lambda o: o["obs_time"])
        val_err = tt.temp_c - extreme
        fcst_time = _abs(vf, tt.day, tt.hour)
        timing_err = (fcst_time - src["obs_time"]).total_seconds() / 3600.0
        rows.append(_row(element, None, None, "TAF", tt.temp_c, extreme, val_err, "scored",
                         grain="taf"))
        rows.append(_row(f"{element}_timing", None, None, "TAF", tt.hour, None, timing_err,
                         "scored", grain="taf"))
    return rows


def _circular_mean(errs: list[float]) -> float | None:
    s = sum(math.sin(math.radians(e)) for e in errs)
    c = sum(math.cos(math.radians(e)) for e in errs)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return None
    return math.degrees(math.atan2(s, c))


def _bin_label(lead: int | None) -> str | None:
    if lead is None:
        return None
    for lo, hi in _LEAD_BINS:
        if lo <= lead < hi:
            return f"{lo}-{hi}"
    return None


def _element_stats(rows: list[ElementRow]) -> list[ElementStat]:
    """Pool signed/abs errors per element per lead bin AND overall; circular mean for
    direction. Never average per-row -- pool then recompute (the anti-averaging rule)."""
    scored = [r for r in rows if r.status == "scored" and r.signed_error is not None]
    stats: list[ElementStat] = []
    elements = sorted({r.element for r in scored})
    for el in elements:
        el_rows = [r for r in scored if r.element == el]
        buckets: dict[str, list[ElementRow]] = {"overall": el_rows}
        for r in el_rows:
            lab = _bin_label(r.lead_hr)
            if lab:
                buckets.setdefault(lab, []).append(r)
        for label, rs in buckets.items():
            errs = [r.signed_error for r in rs]
            abs_errs = [r.abs_error for r in rs]
            bias = _circular_mean(errs) if el == "wind_dir" else sum(errs) / len(errs)
            stats.append(ElementStat(element=el, bin=label, n=len(rs), bias=bias,
                                     mae=sum(abs_errs) / len(abs_errs)))
    return stats


# ---------------------------------------------------------------------------
# Axis 2 -- event contingency (union/conservative view)
# ---------------------------------------------------------------------------

def _cig_pred(thr):
    def f(s: State, prof):
        if s.ceiling_status == CEIL_KNOWN_NUMERIC:
            return s.ceiling_ft < thr
        if s.ceiling_status == CEIL_KNOWN_UNLIMITED:
            return False
        return None
    return f


def _vis_pred(thr_std, thr_oconus):
    def f(s: State, prof):
        thr = thr_oconus if prof.use_oconus_vis_substitutions else thr_std
        if s.vis_status == VIS_KNOWN_NUMERIC and s.vis_m is not None:
            return s.vis_m < thr
        if s.vis_status == VIS_KNOWN_UNLIMITED:
            return False
        return None
    return f


def _cig_landing(s: State, prof):
    if s.ceiling_status == CEIL_KNOWN_NUMERIC:
        return s.ceiling_ft < prof.landing_min_ceiling_ft
    if s.ceiling_status == CEIL_KNOWN_UNLIMITED:
        return False
    return None


def _vis_landing(s: State, prof):
    if s.vis_status == VIS_KNOWN_NUMERIC and s.vis_m is not None:
        return s.vis_m < prof.landing_min_vis_m
    if s.vis_status == VIS_KNOWN_UNLIMITED:
        return False
    return None


def _wx_pred(want: str):
    def f(s: State, prof):
        if s.weather_status != "known":
            return None
        atomic, _ = normalize_weather(s.weather)
        if want == "ts":
            return "other:TS" in atomic
        return any(a.split(":", 1)[0] == want for a in atomic)
    return f


def _gust_ge(thr):
    def f(s: State, prof):
        if s.gust_status == GUST_PRESENT and s.wind_gust is not None:
            return s.wind_gust >= thr
        if s.gust_status in (GUST_KNOWN_ABSENT, GUST_INHERITED_ABSENT):
            return False
        return None
    return f


def _wind_ge(thr):
    def f(s: State, prof):
        if s.wind_speed_status == SPD_NUMERIC:
            return s.wind_speed >= thr
        return None
    return f


# (name, provenance, predicate). Versioned as SkillPolicy.catalog_version.
EVENT_CATALOG_V1 = [
    ("cig_lt_2000", "DAF A4.1 E/D boundary", _cig_pred(2000)),
    ("cig_lt_1000", "DAF A4.1 D/C boundary", _cig_pred(1000)),
    ("cig_lt_3000", "project diagnostic (alternate-planning)", _cig_pred(3000)),
    ("cig_lt_landing_min", "station profile (provisional minima)", _cig_landing),
    ("vis_lt_3sm", "DAF A4.1 boundary", _vis_pred(4800, 5000)),
    ("vis_lt_2sm", "DAF A4.1 boundary", _vis_pred(3200, 3000)),
    ("vis_lt_1sm", "project diagnostic", _vis_pred(1600, 1600)),
    ("vis_lt_landing_min", "station profile (provisional minima)", _vis_landing),
    ("ts", "phenomenon diagnostic", _wx_pred("ts")),
    ("gust_ge_25", "project diagnostic", _gust_ge(25)),
    ("wind_ge_25", "project diagnostic", _wind_ge(25)),
    ("fzprecip", "phenomenon diagnostic", _wx_pred("freezing")),
    ("frozen_precip", "phenomenon diagnostic", _wx_pred("frozen")),
]


def _forecast_event_flag(taf, hour, vf, vt, pred, profile) -> tuple[bool, bool]:
    """(fcst_yes, via_tempo). Prevailing OR any active TEMPO/PROB alternate meets the
    predicate; unevaluable prevailing counts as no unless an alternate says yes."""
    fs = forecast_state(taf, hour + timedelta(minutes=30), valid_from=vf, valid_to=vt)
    if pred(fs.prevailing, profile) is True:
        return True, False
    for alt in fs.alternates:
        if pred(alt, profile) is True:
            return True, True
    return False, False


def _axis2_event_hours(taf, hours, vf, vt, profile) -> list[EventHour]:
    out: list[EventHour] = []
    for name, _prov, pred in EVENT_CATALOG_V1:
        for h in hours:
            fyes, via = _forecast_event_flag(taf, h.hour, vf, vt, pred, profile)
            if h.status != "available":
                out.append(EventHour(hour=h.hour, event=name, fcst=fyes, obs=None, via_tempo=via))
                continue
            oval = pred(_conservative_state(h), profile)
            if oval is None:
                out.append(EventHour(hour=h.hour, event=name, fcst=fyes, obs=None, via_tempo=via))
                continue
            cell = ("hit" if fyes and oval else "false_alarm" if fyes and not oval
                    else "miss" if oval else "correct_negative")
            out.append(EventHour(hour=h.hour, event=name, fcst=fyes, obs=oval,
                                 via_tempo=via, cell=cell))
    return out


def _contingency(event_hours: list[EventHour]) -> list[Contingency]:
    out: list[Contingency] = []
    for name, _prov, _pred in EVENT_CATALOG_V1:
        cells = [e.cell for e in event_hours if e.event == name and e.cell is not None]
        a = cells.count("hit")
        b = cells.count("false_alarm")
        c = cells.count("miss")
        d = cells.count("correct_negative")
        sc = contingency_scores(a, b, c, d)
        out.append(Contingency(event=name, a=a, b=b, c=c, d=d, **sc))
    return out


def _episode_hours(event_hours, name, side) -> list[list[datetime]]:
    """Consecutive same-flag hours -> episodes (lists of hour datetimes). `side` picks
    the forecast-yes or observed-yes flag."""
    rows = sorted((e for e in event_hours if e.event == name), key=lambda e: e.hour)
    eps: list[list[datetime]] = []
    cur: list[datetime] = []
    for e in rows:
        yes = e.fcst if side == "fcst" else (e.obs is True)
        if yes:
            cur.append(e.hour)
        elif cur:
            eps.append(cur)
            cur = []
    if cur:
        eps.append(cur)
    return eps


def _match_episodes(obs_eps, fcst_eps, window_h) -> list[tuple[int | None, int | None]]:
    """Deterministic minimum-cost one-to-one assignment (sec 9.2). Eligible pairs
    overlap or sit within window_h hours; among maximum-cardinality matchings, minimize
    total cost (cost rewards overlap then onset proximity); ties break on earlier
    observed then forecast onset."""
    win = timedelta(hours=window_h)

    def overlap(o, f) -> int:
        return len(set(o) & set(f))

    def eligible(o, f) -> bool:
        return overlap(o, f) > 0 or abs((o[0] - f[0]).total_seconds()) <= win.total_seconds()

    def pair_cost(o, f) -> float:
        return -100.0 * overlap(o, f) + abs((f[0] - o[0]).total_seconds()) / 3600.0

    best: dict = {"key": None, "match": []}

    def recurse(oi: int, used_f: set[int], match: list[tuple[int, int]]):
        if oi == len(obs_eps):
            cost = sum(pair_cost(obs_eps[o], fcst_eps[f]) for o, f in match)
            tie = tuple(obs_eps[o][0].timestamp() for o, f in match) + \
                tuple(fcst_eps[f][0].timestamp() for o, f in match)
            key = (-len(match), cost, tie)
            if best["key"] is None or key < best["key"]:
                best["key"] = key
                best["match"] = list(match)
            return
        recurse(oi + 1, used_f, match)      # leave this observed episode unmatched
        for fi in range(len(fcst_eps)):
            if fi not in used_f and eligible(obs_eps[oi], fcst_eps[fi]):
                recurse(oi + 1, used_f | {fi}, match + [(oi, fi)])

    recurse(0, set(), [])
    matched = best["match"]
    matched_o = {o for o, _ in matched}
    matched_f = {f for _, f in matched}
    result: list[tuple[int | None, int | None]] = list(matched)
    result += [(o, None) for o in range(len(obs_eps)) if o not in matched_o]
    result += [(None, f) for f in range(len(fcst_eps)) if f not in matched_f]
    return result


def _axis2_episodes(event_hours) -> list[Episode]:
    out: list[Episode] = []
    for name, _prov, _pred in EVENT_CATALOG_V1:
        obs_eps = _episode_hours(event_hours, name, "obs")
        fcst_eps = _episode_hours(event_hours, name, "fcst")
        if not obs_eps and not fcst_eps:
            continue
        for oi, fi in _match_episodes(obs_eps, fcst_eps, 3):
            if oi is not None and fi is not None:
                o, f = obs_eps[oi], fcst_eps[fi]
                out.append(Episode(
                    event=name, disposition="matched",
                    obs_onset=o[0], obs_end=o[-1], fcst_onset=f[0], fcst_end=f[-1],
                    onset_error_h=(f[0] - o[0]).total_seconds() / 3600.0,
                    end_error_h=(f[-1] - o[-1]).total_seconds() / 3600.0))
            elif oi is not None:
                o = obs_eps[oi]
                out.append(Episode(event=name, disposition="missed", obs_onset=o[0], obs_end=o[-1]))
            else:
                f = fcst_eps[fi]
                out.append(Episode(event=name, disposition="false_alarm",
                                   fcst_onset=f[0], fcst_end=f[-1]))
    return out


# ---------------------------------------------------------------------------
# Axis 3 -- ordinal category distance (predominant view)
# ---------------------------------------------------------------------------

def _cat(state: State, profile: StationProfile) -> str | None:
    return daf_flight_category(state.ceiling_ft, state.ceiling_status,
                               state.vis_m, state.vis_status, profile=profile)


def _axis3(taf, hours, vf, vt, profile) -> tuple[list, float | None, dict | None, float | None]:
    series = []
    deltas = []
    signed = []
    worst = None
    for h in hours:
        if h.status != "available":
            continue
        fc = _cat(_prevailing_at(taf, h.hour, vf, vt), profile)
        oc = _cat(_predominant_state(h), profile)
        if fc is None or oc is None:
            continue
        fi, oi = _CAT_LETTERS.index(fc), _CAT_LETTERS.index(oc)
        series.append([h.hour.strftime("%d%H%MZ"), fc, oc])
        deltas.append(abs(fi - oi))
        signed.append(fi - oi)
        if worst is None or abs(fi - oi) > worst["delta"]:
            worst = {"hour": h.hour.strftime("%d%H%MZ"), "delta": abs(fi - oi), "fcst": fc, "obs": oc}
    mace = sum(deltas) / len(deltas) if deltas else None
    signed_mean = sum(signed) / len(signed) if signed else None
    return series, mace, worst, signed_mean


# ---------------------------------------------------------------------------
# Top-level scorer + benchmark deltas
# ---------------------------------------------------------------------------

def score_skill(
    taf: TafObs,
    obs: list[dict],
    valid_from: datetime,
    valid_to: datetime,
    *,
    profile: StationProfile | None = None,
    policy: SkillPolicy | None = None,
    truth_policy: TruthPolicy | None = None,
) -> TafSkillScore:
    """Run all three skill axes over one archived TAF + its truth obs."""
    profile = profile or default_profile(taf.station)
    policy = policy or SkillPolicy()
    hours, _manifest = build_truth(obs, valid_from, valid_to, policy=truth_policy)

    rows = _axis1_hourly(taf, hours, valid_from, valid_to, policy)
    rows += _axis1_qnh(taf, obs, valid_from, valid_to)
    rows += _axis1_txtn(taf, obs, valid_from, valid_to, policy)
    stats = _element_stats(rows)

    event_hours = _axis2_event_hours(taf, hours, valid_from, valid_to, profile)
    contingency = _contingency(event_hours)
    episodes = _axis2_episodes(event_hours)

    series, mace, worst, signed_mean = _axis3(taf, hours, valid_from, valid_to, profile)

    return TafSkillScore(
        station=taf.station, valid_from=valid_from, valid_to=valid_to,
        catalog_version=policy.catalog_version,
        element_rows=rows, element_stats=stats, event_hours=event_hours,
        contingency=contingency, episodes=episodes, category_series=series,
        mace=mace, worst_excursion=worst, signed_mace_mean=signed_mean,
        hours_scored=sum(1 for h in hours if h.status == "available"),
        hours_unavailable=sum(1 for h in hours if h.status != "available"),
    )


def _mae_on(rows: list[ElementRow], element: str) -> tuple[dict[datetime, float], float | None]:
    """Map hour -> abs_error for scored rows of one element, and the pooled MAE."""
    scored = {r.hour: r.abs_error for r in rows
              if r.element == element and r.status == "scored" and r.abs_error is not None
              and r.hour is not None}
    mae = sum(scored.values()) / len(scored) if scored else None
    return scored, mae


def skill_deltas(subject: TafSkillScore, baseline: TafSkillScore) -> dict:
    """Benchmark deltas on MATCHED hours only (sec 10): an availability difference must
    not masquerade as skill. Zero-baseline ratios -> None + reason with raw values."""
    out: dict = {"elements": {}, "mace": None}
    elements = {r.element for r in subject.element_rows if r.grain == "hour"}
    for el in sorted(elements):
        s_map, _ = _mae_on(subject.element_rows, el)
        b_map, _ = _mae_on(baseline.element_rows, el)
        common = sorted(set(s_map) & set(b_map))
        if not common:
            out["elements"][el] = {"mae_skill": None, "reason": "no_matched_hours", "n": 0}
            continue
        s_mae = sum(s_map[h] for h in common) / len(common)
        b_mae = sum(b_map[h] for h in common) / len(common)
        if b_mae == 0:
            out["elements"][el] = {"mae_skill": None, "reason": "zero_baseline_mae",
                                   "subject_mae": s_mae, "baseline_mae": b_mae, "n": len(common)}
        else:
            out["elements"][el] = {"mae_skill": 1 - s_mae / b_mae, "subject_mae": s_mae,
                                   "baseline_mae": b_mae, "n": len(common)}

    # MACE on matched hours (both produced an ordinal point that hour)
    s_ord = {row[0]: row for row in subject.category_series}
    b_ord = {row[0]: row for row in baseline.category_series}
    common = sorted(set(s_ord) & set(b_ord))
    if common:
        def _mace(series_map):
            vals = [abs(_CAT_LETTERS.index(series_map[h][1]) - _CAT_LETTERS.index(series_map[h][2]))
                    for h in common]
            return sum(vals) / len(vals)
        s_mace, b_mace = _mace(s_ord), _mace(b_ord)
        if b_mace == 0:
            out["mace"] = {"mace_skill": None, "reason": "zero_baseline_mace",
                           "subject_mace": s_mace, "baseline_mace": b_mace, "n": len(common)}
        else:
            out["mace"] = {"mace_skill": 1 - s_mace / b_mace, "subject_mace": s_mace,
                           "baseline_mace": b_mace, "n": len(common)}
    return out
