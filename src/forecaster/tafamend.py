"""Scorer 2 -- amendment-implied busts (scoring-design sec 8). PURE: no duckdb/SQL,
no network, no matplotlib, no LLM. Imports tafstate primitives + reads METAR dicts.

The reframe: a TAF's amendment criteria ARE a verification metric. DAFI 15-129
3.4.2.6 requires amendment whenever observed conditions stop matching Attachment 2.
We replay the observed sequence over the archived validity window; every doctrine
threshold crossing relative to what the TAF said is a bust signal.

Truth view: predominant + temporaries (sec 5.5). An hour is IN-SPEC when it matches
the prevailing forecast OR an active TEMPO/PROB alternate (best fit) -- that is how a
correctly forecast TEMPO avoids scoring as a bust; Rule 8 separately busts when a
TEMPO condition instead becomes PREDOMINANT.

Rules that build now (sec 8.1): category (cig/vis lower-of), Rule 1 winds, Rule 5
altimeter, Rule 7 thunderstorms, Rule 8 TEMPO, Rule 9 BECMG/FM timing. Deferred
(sec 8.2) rules are reported in `rules_not_scored`, never silently passed.

Aggregation (sec 8.3): three layers -- hourly results -> rule episodes (consecutive
failing hours of one rule) -> amendment triggers (episodes sharing an onset hour
merge). Headline is trigger_count; after-amd-service failures are excluded from it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from pydantic import BaseModel

from forecaster.tafparse import TafObs
from forecaster.tafstate import (
    DIR_NUMERIC, GUST_PRESENT,
    QNH_KNOWN, SPD_NUMERIC, State, StationProfile,
    TruthPolicy, absolutize, build_truth, daf_flight_category, default_profile, forecast_state,
    normalize_weather, observed_state, predominant_state as _predominant_state,
    resolve_group_state,
)

# Bump on any scored-output-changing code fix, even with unchanged policy JSON
# (sec 11): a rerun with a new scorer_version is a NEW run, never a replace.
SCORER_VERSION = "1"

# Rules computable now, and the doctrine rules deferred for lack of data (sec 8.2).
BUILD_NOW_RULES = ["category", "wind", "altimeter", "thunderstorm", "tempo", "change_timing"]
DEFERRED_RULES = {
    "icing": "Rule 2 -- needs PIREP/model data",
    "turbulence": "Rule 3 -- needs PIREP/model data",
    "warning": "Rules 4/6 -- need WWA data",
    "representative": "Rule 10 -- human judgment, un-automatable",
    "no_longer_expected": "Rule 8/9 'no longer expected' clause -- needs forecast revisions",
}


class AmendPolicy(BaseModel):
    """Versioned amend rule set + constants (sec 6/8). Changing any constant is a new
    policy version/hash and therefore a new run; old statistics stay reproducible."""
    name: str = "amend-v1"
    version: str = "1"
    wind_speed_tol_kt: int = 10          # Rule 1: bust when |fcst-obs| > this
    gust_tol_kt: int = 10
    dir_tol_deg: int = 30
    dir_speed_min_kt: int = 15           # direction check eligible only when expected >= this
    ts_timing_tol_min: int = 30          # Rule 7
    change_timing_tol_min: int = 30      # Rule 9: complete within this after the specified time
    persist_min_min: int = 30            # Rule 9: an early change busts only if it persists >= this
    active_rules: list[str] = BUILD_NOW_RULES


class RuleHourResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    hour: datetime
    rule: str
    result: str                          # pass | fail | unavailable
    reason: str | None = None
    fcst: str | None = None
    obs: str | None = None
    detail: str | None = None
    after_amd_service: bool = False


class RuleEpisode(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    rule: str
    onset: datetime
    end: datetime                        # last failing hour (inclusive)
    hours: int
    worst_detail: str | None = None
    after_amd_service: bool = False


class AmendTrigger(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    onset: datetime
    rules: list[str]
    after_amd_service: bool = False


class TafAmendScore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    station: str
    valid_from: datetime
    valid_to: datetime
    hourly_results: list[RuleHourResult]
    rule_episodes: list[RuleEpisode]
    triggers: list[AmendTrigger]
    trigger_count: int
    per_rule_episodes: dict[str, int]
    hours_scored: int
    hours_in_spec: int
    in_spec_fraction: float | None
    hours_after_amd_service: int
    triggers_after_amd_service: int
    category_series: list[list]          # [[hourZ, fcst_cat, obs_cat], ...]
    rules_not_scored: dict[str, str]


# ---------------------------------------------------------------------------
# Per-hour observed/forecast state helpers
# ---------------------------------------------------------------------------

def _cat(state: State, profile: StationProfile) -> str | None:
    return daf_flight_category(state.ceiling_ft, state.ceiling_status,
                               state.vis_m, state.vis_status, profile=profile)


def _circular(a: int, b: int) -> int:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _wind_inspec(fcst: State, obs: State, policy: AmendPolicy) -> tuple[bool, str]:
    """Rule 1 in-spec vs one forecast candidate (sec 8.1). Speed, gust (with the
    presence-completion of DECIDED #14), and the eligibility-gated direction check."""
    if fcst.wind_speed_status == SPD_NUMERIC and obs.wind_speed_status == SPD_NUMERIC:
        if abs(fcst.wind_speed - obs.wind_speed) > policy.wind_speed_tol_kt:
            return False, f"speed {fcst.wind_speed} vs obs {obs.wind_speed}"
    fg = fcst.wind_gust if fcst.gust_status == GUST_PRESENT else None
    og = obs.wind_gust if obs.gust_status == GUST_PRESENT else None
    fm = fcst.wind_speed if fcst.wind_speed_status == SPD_NUMERIC else None
    om = obs.wind_speed if obs.wind_speed_status == SPD_NUMERIC else None
    if fg is not None and og is not None:
        if abs(fg - og) > policy.gust_tol_kt:
            return False, f"gust {fg} vs obs {og}"
    elif fg is not None and og is None and om is not None:
        if fg - om > policy.gust_tol_kt:
            return False, f"fcst gust {fg} vs obs mean {om}"
    elif og is not None and fg is None and fm is not None:
        if og - fm > policy.gust_tol_kt:
            return False, f"obs gust {og} vs fcst mean {fm}"
    expected_strong = max(fcst.wind_speed or 0, fcst.wind_gust or 0) >= policy.dir_speed_min_kt
    if (fcst.wind_dir_status == DIR_NUMERIC and obs.wind_dir_status == DIR_NUMERIC
            and expected_strong):
        d = _circular(fcst.wind_dir, obs.wind_dir)
        if d > policy.dir_tol_deg:
            return False, f"dir {fcst.wind_dir} vs obs {obs.wind_dir} ({d} deg)"
    return True, ""


def _state_matches(target: State, obs: State, fields: set[str], profile: StationProfile,
                   policy: AmendPolicy) -> bool:
    """Does `obs` meet `target` on the target's EXPLICIT fields (sec 8.1, Rule 8/9)?
    cig/vis -> same DAF category; weather -> obs classes superset target classes;
    wind -> within Rule 1 tolerance."""
    if "sky" in fields or "visibility" in fields:
        tc, oc = _cat(target, profile), _cat(obs, profile)
        if tc is None or oc is None or tc != oc:
            return False
    if "weather" in fields:
        _, tcl = normalize_weather(target.weather)
        _, ocl = normalize_weather(obs.weather)
        if not tcl <= ocl:
            return False
    if "wind" in fields:
        ok, _ = _wind_inspec(target, obs, policy)
        if not ok:
            return False
    return True


def _band_hundredths(v: float) -> str:
    h = round(v * 100)
    if h >= 3100:
        return "HIGH"        # >= 31.00 inHg (inclusive)
    if h < 2800:
        return "LOW"         # < 28.00 inHg (strict: 27.99 crosses, 28.00 does not)
    return "NORMAL"


# ---------------------------------------------------------------------------
# Rule checkers -- each a pure function returning per-hour results
# ---------------------------------------------------------------------------

class _Ctx(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    taf: TafObs
    valid_from: datetime
    valid_to: datetime
    obs: list[dict]
    hours: list                          # list[HourTruth]
    profile: StationProfile
    policy: AmendPolicy


def _candidates(ctx: _Ctx, hour: datetime) -> list[State]:
    fs = forecast_state(ctx.taf, hour + timedelta(minutes=30),
                        valid_from=ctx.valid_from, valid_to=ctx.valid_to)
    return [fs.prevailing, *fs.alternates]


def rule_category(ctx: _Ctx) -> list[RuleHourResult]:
    out = []
    for h in ctx.hours:
        if h.status != "available":
            out.append(RuleHourResult(hour=h.hour, rule="category", result="unavailable",
                                      reason=h.reason))
            continue
        obs = _predominant_state(h)
        oc = _cat(obs, ctx.profile)
        cats = [_cat(c, ctx.profile) for c in _candidates(ctx, h.hour)]
        prevailing_cat = cats[0]
        if oc is None or all(c is None for c in cats):
            out.append(RuleHourResult(hour=h.hour, rule="category", result="unavailable",
                                      reason="category_unresolved", fcst=prevailing_cat, obs=oc))
            continue
        ok = oc in cats
        out.append(RuleHourResult(hour=h.hour, rule="category",
                                  result="pass" if ok else "fail",
                                  fcst=prevailing_cat, obs=oc,
                                  detail=None if ok else f"fcst {prevailing_cat} -> obs {oc}"))
    return out


def rule_wind(ctx: _Ctx) -> list[RuleHourResult]:
    out = []
    for h in ctx.hours:
        if h.status != "available":
            out.append(RuleHourResult(hour=h.hour, rule="wind", result="unavailable",
                                      reason=h.reason))
            continue
        obs = _predominant_state(h)
        if obs.wind_speed_status != SPD_NUMERIC:
            out.append(RuleHourResult(hour=h.hour, rule="wind", result="unavailable",
                                      reason="no_predominant_wind"))
            continue
        best_reason = ""
        ok_any = False
        for c in _candidates(ctx, h.hour):
            ok, reason = _wind_inspec(c, obs, ctx.policy)
            if ok:
                ok_any = True
                break
            best_reason = best_reason or reason
        out.append(RuleHourResult(hour=h.hour, rule="wind",
                                  result="pass" if ok_any else "fail",
                                  detail=None if ok_any else best_reason))
    return out


def rule_altimeter(ctx: _Ctx) -> list[RuleHourResult]:
    out = []
    for h in ctx.hours:
        if h.status != "available":
            out.append(RuleHourResult(hour=h.hour, rule="altimeter", result="unavailable",
                                      reason=h.reason))
            continue
        obs = _predominant_state(h)
        cand = _candidates(ctx, h.hour)[0]      # prevailing; TEMPO excluded from altimeter
        if obs.qnh_status != QNH_KNOWN or cand.qnh_status != QNH_KNOWN:
            out.append(RuleHourResult(hour=h.hour, rule="altimeter", result="unavailable",
                                      reason="no_qnh"))
            continue
        fb, ob = _band_hundredths(cand.qnh_inhg), _band_hundredths(obs.qnh_inhg)
        ok = fb == ob
        out.append(RuleHourResult(hour=h.hour, rule="altimeter",
                                  result="pass" if ok else "fail",
                                  fcst=f"{cand.qnh_inhg:.2f}/{fb}", obs=f"{obs.qnh_inhg:.2f}/{ob}",
                                  detail=None if ok else f"{fb} -> {ob}"))
    return out


def _hour_of(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def _merge_intervals(flags: list[tuple[datetime, datetime, bool]]) -> list[tuple[datetime, datetime]]:
    """Merge consecutive True (start, end) spans."""
    out = []
    cur = None
    for start, end, on in flags:
        if on:
            cur = (cur[0], end) if cur else (start, end)
        elif cur:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def _observed_ts_intervals(ctx: _Ctx) -> list[tuple[datetime, datetime]]:
    rows = sorted(ctx.obs, key=lambda o: o["obs_time"])
    flags = []
    for i, ob in enumerate(rows):
        t = ob["obs_time"]
        nxt = rows[i + 1]["obs_time"] if i + 1 < len(rows) else ctx.valid_to
        has_ts = any("TS" in w for w in (ob.get("weather") or []))
        flags.append((t, nxt, has_ts))
    return _merge_intervals(flags)


def _forecast_ts_intervals(ctx: _Ctx) -> list[tuple[datetime, datetime]]:
    flags = []
    for h in ctx.hours:
        nxt = h.hour + timedelta(hours=1)
        ts = any(any("TS" in w for w in c.weather) for c in _candidates(ctx, h.hour))
        flags.append((h.hour, nxt, ts))
    return _merge_intervals(flags)


def rule_thunderstorm(ctx: _Ctx) -> list[RuleHourResult]:
    tol = timedelta(minutes=ctx.policy.ts_timing_tol_min)
    fcst = _forecast_ts_intervals(ctx)
    obs = _observed_ts_intervals(ctx)
    fails: dict[datetime, str] = {}
    # observed TS not covered by a forecast TS window (within tolerance) -> bust
    for os, oe in obs:
        if not any(fs - tol <= os <= fe + tol for fs, fe in fcst):
            fails[_hour_of(os)] = f"observed TS onset {os:%d%H%MZ} unforecast"
    # forecast TS window with no observed TS nearby -> over-forecast bust
    for fs, fe in fcst:
        if not any(os <= fe + tol and oe >= fs - tol for os, oe in obs):
            fails[_hour_of(fs)] = f"forecast TS {fs:%d%H%MZ} did not occur"
    out = []
    for h in ctx.hours:
        if h.status != "available":
            out.append(RuleHourResult(hour=h.hour, rule="thunderstorm", result="unavailable",
                                      reason=h.reason))
        elif h.hour in fails:
            out.append(RuleHourResult(hour=h.hour, rule="thunderstorm", result="fail",
                                      detail=fails[h.hour]))
        else:
            out.append(RuleHourResult(hour=h.hour, rule="thunderstorm", result="pass"))
    return out


def rule_tempo(ctx: _Ctx) -> list[RuleHourResult]:
    """Bust when a forecast TEMPO condition becomes PREDOMINANT (should have been a
    BECMG/prevailing), or never occurs during its window (sec 8.1 Rule 8)."""
    abs_groups = absolutize(ctx.taf, ctx.valid_from, ctx.valid_to)
    tempos = [g for g in abs_groups if g.group_type in ("TEMPO", "PROB")]
    fails: dict[datetime, str] = {}
    for tg in tempos:
        target = resolve_group_state(ctx.taf, tg.group_index, ctx.valid_from, ctx.valid_to)
        fields = tg.group.explicit_fields
        window_hours = [h for h in ctx.hours
                        if tg.start <= h.hour < tg.end and h.status == "available"]
        predominant_hit = None
        occurred = False
        for h in window_hours:
            obs_pred = _predominant_state(h)
            if _state_matches(target, obs_pred, fields, ctx.profile, ctx.policy):
                predominant_hit = h.hour if predominant_hit is None else predominant_hit
                occurred = True
        # brief occurrence (union weather / any ob) so a correctly-forecast brief
        # condition is not scored as "never occurred"
        if not occurred:
            for h in window_hours:
                union = State(weather=list(h.cons.get("weather").value if h.cons.get("weather") else []),
                              weather_status="known")
                if "weather" in fields and _state_matches(
                        State(weather=target.weather, weather_status="known"),
                        union, {"weather"}, ctx.profile, ctx.policy):
                    occurred = True
                    break
        if predominant_hit is not None:
            fails[predominant_hit] = "TEMPO condition became predominant"
        elif window_hours and not occurred:
            fails.setdefault(window_hours[0].hour, "forecast TEMPO never occurred")
    out = []
    for h in ctx.hours:
        if h.status != "available":
            out.append(RuleHourResult(hour=h.hour, rule="tempo", result="unavailable",
                                      reason=h.reason))
        elif h.hour in fails:
            out.append(RuleHourResult(hour=h.hour, rule="tempo", result="fail",
                                      detail=fails[h.hour]))
        else:
            out.append(RuleHourResult(hour=h.hour, rule="tempo", result="pass"))
    return out


def rule_change_timing(ctx: _Ctx) -> list[RuleHourResult]:
    """Rule 9: a BECMG/FM change must complete within the timing tolerance after its
    specified time (FM minute; BECMG window END), and must not occur early-and-persist.
    'occurred' = observed matches the change group's explicit fields."""
    abs_groups = absolutize(ctx.taf, ctx.valid_from, ctx.valid_to)
    changes = [g for g in abs_groups if g.group_type in ("FM", "BECMG")]
    tol = timedelta(minutes=ctx.policy.change_timing_tol_min)
    persist = timedelta(minutes=ctx.policy.persist_min_min)
    rows = sorted(ctx.obs, key=lambda o: o["obs_time"])
    fails: dict[datetime, str] = {}
    for cg in changes:
        change_time = cg.start if cg.group_type == "FM" else cg.end
        target = resolve_group_state(ctx.taf, cg.group_index, ctx.valid_from, ctx.valid_to)
        fields = cg.group.explicit_fields
        matched = [(ob["obs_time"], _state_matches(target, observed_state(ob), fields,
                                                   ctx.profile, ctx.policy)) for ob in rows]
        after = [t for t, m in matched if m and t >= change_time]
        first_after = min(after) if after else None
        if first_after is None or first_after > change_time + tol:
            hr = _hour_of(change_time if change_time < ctx.valid_to else ctx.valid_to - timedelta(hours=1))
            if ctx.valid_from <= hr < ctx.valid_to:
                fails[hr] = (f"{cg.group_type} change not established within "
                             f"{ctx.policy.change_timing_tol_min} min of {change_time:%d%H%MZ}")
        # early-and-persists: a run of matching obs ending before change_time, >= persist
        early = [(t, m) for t, m in matched if t < change_time]
        run_start = None
        for i, (t, m) in enumerate(early):
            if m and run_start is None:
                run_start = t
            if run_start is not None and (not m or i == len(early) - 1):
                run_end = t if m else early[i - 1][0]
                if run_end - run_start >= persist:
                    hr = _hour_of(run_start)
                    if ctx.valid_from <= hr < ctx.valid_to:
                        fails.setdefault(hr, f"{cg.group_type} change occurred early and persisted")
                run_start = None
    out = []
    for h in ctx.hours:
        if h.status != "available":
            out.append(RuleHourResult(hour=h.hour, rule="change_timing", result="unavailable",
                                      reason=h.reason))
        elif h.hour in fails:
            out.append(RuleHourResult(hour=h.hour, rule="change_timing", result="fail",
                                      detail=fails[h.hour]))
        else:
            out.append(RuleHourResult(hour=h.hour, rule="change_timing", result="pass"))
    return out


_RULES = {
    "category": rule_category,
    "wind": rule_wind,
    "altimeter": rule_altimeter,
    "thunderstorm": rule_thunderstorm,
    "tempo": rule_tempo,
    "change_timing": rule_change_timing,
}


# ---------------------------------------------------------------------------
# Amendment-service remarks (sec 8, DECIDED #13) + aggregation (sec 8.3)
# ---------------------------------------------------------------------------

_LAST_NO_AMDS = re.compile(r"LAST\s+NO\s+AMDS\s+AFT\s+(\d{2})(\d{2})(\d{2})")
_LIMITED = re.compile(r"LIMITED\s+METWATCH\s+(\d{2})(\d{2})(\d{2})\s+TIL\s+(\d{2})(\d{2})(\d{2})")


def _amd_service_gaps(taf: TafObs, valid_from: datetime, valid_to: datetime):
    """Intervals during which amendment service is OFF, from the bulletin's own
    remarks. Rule failures inside these are flagged after_amd_service and excluded
    from the headline trigger count (the bulletin said so)."""
    from forecaster.tafstate import _abs  # noqa: PLC0415  (internal helper reuse)
    gaps = []
    if m := _LAST_NO_AMDS.search(taf.raw):
        d, hh, mm = int(m[1]), int(m[2]), int(m[3])
        gaps.append((_abs(valid_from, d, hh, mm), valid_to))
    if m := _LIMITED.search(taf.raw):
        s = _abs(valid_from, int(m[1]), int(m[2]), int(m[3]))
        e = _abs(valid_from, int(m[4]), int(m[5]), int(m[6]))
        gaps.append((s, e))
    return gaps


def _in_gaps(hour: datetime, gaps) -> bool:
    return any(s <= hour < e for s, e in gaps)


def _episodes(results: list[RuleHourResult]) -> list[RuleEpisode]:
    """Layer 2: consecutive failing hours of ONE rule collapse into one episode."""
    episodes = []
    by_rule: dict[str, list[RuleHourResult]] = {}
    for r in results:
        by_rule.setdefault(r.rule, []).append(r)
    for rule, rows in by_rule.items():
        rows = sorted(rows, key=lambda r: r.hour)
        run = []
        for r in rows:
            if r.result == "fail":
                run.append(r)
            elif run:
                episodes.append(_make_episode(rule, run))
                run = []
        if run:
            episodes.append(_make_episode(rule, run))
    return sorted(episodes, key=lambda e: (e.onset, e.rule))


def _make_episode(rule: str, run: list[RuleHourResult]) -> RuleEpisode:
    return RuleEpisode(
        rule=rule, onset=run[0].hour, end=run[-1].hour, hours=len(run),
        worst_detail=next((r.detail for r in run if r.detail), None),
        after_amd_service=all(r.after_amd_service for r in run),
    )


def _triggers(episodes: list[RuleEpisode]) -> list[AmendTrigger]:
    """Layer 3: episodes whose ONSET falls in the same clock hour merge into one
    trigger (one amendment covers everything wrong at that moment)."""
    by_onset: dict[datetime, list[RuleEpisode]] = {}
    for e in episodes:
        by_onset.setdefault(e.onset, []).append(e)
    triggers = []
    for onset, eps in sorted(by_onset.items()):
        triggers.append(AmendTrigger(
            onset=onset, rules=sorted({e.rule for e in eps}),
            after_amd_service=all(e.after_amd_service for e in eps),
        ))
    return triggers


def score_amend(
    taf: TafObs,
    obs: list[dict],
    valid_from: datetime,
    valid_to: datetime,
    *,
    profile: StationProfile | None = None,
    policy: AmendPolicy | None = None,
    truth_policy: TruthPolicy | None = None,
) -> TafAmendScore:
    """Run the amendment-bust scorer over one archived TAF + its truth obs."""
    profile = profile or default_profile(taf.station)
    policy = policy or AmendPolicy()
    hours, _manifest = build_truth(obs, valid_from, valid_to, policy=truth_policy)
    ctx = _Ctx(taf=taf, valid_from=valid_from, valid_to=valid_to, obs=obs, hours=hours,
               profile=profile, policy=policy)

    gaps = _amd_service_gaps(taf, valid_from, valid_to)
    results: list[RuleHourResult] = []
    for name in policy.active_rules:
        for r in _RULES[name](ctx):
            r.after_amd_service = _in_gaps(r.hour, gaps)
            results.append(r)

    episodes = _episodes(results)
    triggers = _triggers([e for e in episodes if not e.after_amd_service])
    all_triggers = _triggers(episodes)

    avail_hours = [h for h in hours if h.status == "available"]
    fail_hours = {r.hour for r in results if r.result == "fail" and not r.after_amd_service}
    in_spec = sum(1 for h in avail_hours if h.hour not in fail_hours)
    per_rule = {}
    for e in episodes:
        per_rule[e.rule] = per_rule.get(e.rule, 0) + 1

    cat_series = []
    for r in results:
        if r.rule == "category" and r.result != "unavailable":
            cat_series.append([r.hour.strftime("%d%H%MZ"), r.fcst, r.obs])

    return TafAmendScore(
        station=taf.station, valid_from=valid_from, valid_to=valid_to,
        hourly_results=results, rule_episodes=episodes, triggers=triggers,
        trigger_count=len(triggers), per_rule_episodes=per_rule,
        hours_scored=len(avail_hours), hours_in_spec=in_spec,
        in_spec_fraction=(in_spec / len(avail_hours)) if avail_hours else None,
        hours_after_amd_service=sum(1 for h in avail_hours if _in_gaps(h.hour, gaps)),
        triggers_after_amd_service=len(all_triggers) - len(triggers),
        category_series=cat_series, rules_not_scored=DEFERRED_RULES,
    )
