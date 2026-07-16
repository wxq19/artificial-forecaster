"""Scorer 1 -- TAFVER (scoring-design sec 7). PURE: no duckdb/SQL, no network, no
matplotlib, no LLM. The official AF percent-correct measure, ACCI15-120 Attachment 7.

Atomic unit: one resolved forecast OPPORTUNITY (the fully inherited state of one
INITIAL/FM/BECMG/TEMPO group, sec 5.2) intersecting one UTC hour. If several groups
are eligible in an hour, each emits its own rows -- never blended. Truth is the
CONSERVATIVE/union view (sec 5.5).

Seven Table A7.1 elements score 0/1 (present weather scores a fractional CSI). The
aggregation is the ANTI-AVERAGING rule: sum points, then divide -- never average hourly
or per-element percentages (sec 7.2). The initial prevailing group reports in its own
INITIAL diagnostic bucket; the combined headline sums all points regardless of bucket.

PROVISIONAL: three cells are owner-DECIDED project policy where the ACCI is silent
(wind-direction keyed to OBSERVED speed, empty-set present-weather CSI = unavailable,
forecast-only gust = 0). Reports label them provisional; results are not "official
TAFVER" until the SME items + golden fixture (sec 15) clear.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from forecaster.tafparse import TafObs
from forecaster.tafstate import (
    DIR_NUMERIC, GUST_PRESENT, QNH_KNOWN, SPD_NUMERIC,
    State, StationProfile, TruthPolicy, build_truth, conservative_state, default_profile,
    forecast_state, normalize_weather, opportunities, resolve_group_state, stable_hash,
    tafver_ceiling_category, tafver_visibility_category,
)

# Bump on any scored-output-changing code fix, even with unchanged policy JSON
# (sec 11): a rerun with a new scorer_version is a NEW run, never a replace.
SCORER_VERSION = "1"

_ELEMENTS = ["ceiling", "visibility", "wind_speed", "wind_dir", "wind_gust",
             "present_weather", "altimeter"]
_PW_CLASSES = ["liquid", "freezing", "frozen", "obscuration", "other"]


class TafverPolicy(BaseModel):
    """Versioned TAFVER MOP constants (sec 6/7). Changing any constant is a new policy
    version/hash and therefore a new run; old statistics stay reproducible."""
    name: str = "tafver-v1-provisional"
    version: str = "1"
    wind_speed_min_kt: int = 6           # forecast sustained >= this to score speed
    wind_speed_tol_kt: int = 9           # correct when |fcst-obs| <= this
    dir_speed_min_kt: int = 6            # applicable OBSERVED speed > this to score dir
    dir_speed_threshold_kt: int = 15    # tolerance step at observed speed >= this
    dir_tol_lo_deg: int = 50            # tolerance when observed speed < threshold
    dir_tol_hi_deg: int = 30            # tolerance when observed speed >= threshold
    gust_tol_kt: int = 10               # both-present correct when |fcst-obs| <= this
    altimeter_tol_inhg: float = 0.05    # correct when observed_min >= forecast - this


class TafverRow(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    group_index: int
    group_type: str
    interval_start: datetime
    interval_end: datetime
    lead_hr: int
    element: str
    fcst_value: str | None = None
    obs_value: str | None = None
    fcst_cat: str | None = None
    obs_cat: str | None = None
    points_earned: float = 0.0
    points_available: int = 0
    status: str                          # scored | unavailable
    reason: str | None = None


class ElementSummary(BaseModel):
    element: str
    bucket: str                          # ALL | INITIAL | FM | BECMG | TEMPO | PROB
    earned: float
    available: int
    percent: float | None


class CategoryStat(BaseModel):
    element: str                         # ceiling | visibility
    category: str
    fcst_hours: int
    obs_hours: int
    accuracy: float | None               # earned/available among rows where obs_cat == c
    bias: float | None                   # fcst_hours / obs_hours


class TafverScore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    station: str
    valid_from: datetime
    valid_to: datetime
    provisional: bool
    rows: list[TafverRow]
    element_summaries: list[ElementSummary]   # per element (ALL) + per group-type bucket
    group_type_summaries: list[ElementSummary]  # combined per bucket (element="combined")
    combined_earned: float
    combined_available: int
    combined_percent: float | None
    category_stats: list[CategoryStat]
    pw_accuracy: float | None
    pw_event_bias: dict[str, float | None]
    policy_hash: str
    profile_hash: str
    obs_hash: str


# ---------------------------------------------------------------------------
# Provenance hashing (sec 6/11) -- stable, reproducible
# ---------------------------------------------------------------------------

_hash = stable_hash            # one shared implementation (tafstate.stable_hash)


def obs_hash(obs: list[dict]) -> str:
    """Canonical truth hash (sec 11): ordered by obs_time, fixed field order
    (station, obs_time, report_type, raw)."""
    rows = sorted(obs, key=lambda o: o["obs_time"])
    canon = [[o.get("station"), str(o.get("obs_time")), o.get("report_type"), o.get("raw")]
             for o in rows]
    return _hash(canon)


# ---------------------------------------------------------------------------
# Element MOPs (Table A7.1) -- each returns a partial TafverRow dict
# ---------------------------------------------------------------------------

def _r(status, earned=0.0, available=0, reason=None, **kw) -> dict:
    return dict(status=status, points_earned=earned, points_available=available,
                reason=reason, **kw)


def _score_ceiling(f: State, o: State, profile) -> dict:
    fc = tafver_ceiling_category(f.ceiling_ft, f.ceiling_status, profile)
    oc = tafver_ceiling_category(o.ceiling_ft, o.ceiling_status, profile)
    if fc is None or oc is None:
        return _r("unavailable", reason="ceiling_category_unresolved", fcst_cat=fc, obs_cat=oc)
    return _r("scored", earned=1.0 if fc == oc else 0.0, available=1, fcst_cat=fc, obs_cat=oc)


def _score_visibility(f: State, o: State, profile) -> dict:
    fc = tafver_visibility_category(f.vis_sm, f.vis_flag, f.vis_status, profile)
    oc = tafver_visibility_category(o.vis_sm, o.vis_flag, o.vis_status, profile)
    if fc is None or oc is None:
        return _r("unavailable", reason="vis_category_unresolved", fcst_cat=fc, obs_cat=oc)
    return _r("scored", earned=1.0 if fc == oc else 0.0, available=1, fcst_cat=fc, obs_cat=oc)


def _score_wind_speed(f: State, o: State, policy) -> dict:
    if f.wind_speed_status != SPD_NUMERIC or f.wind_speed < policy.wind_speed_min_kt:
        return _r("unavailable", reason="forecast_speed_below_threshold")
    if o.wind_speed_status != SPD_NUMERIC:
        return _r("unavailable", reason="observed_speed_missing")
    ok = abs(f.wind_speed - o.wind_speed) <= policy.wind_speed_tol_kt
    return _r("scored", earned=1.0 if ok else 0.0, available=1,
              fcst_value=str(f.wind_speed), obs_value=str(o.wind_speed))


def _score_wind_dir(f: State, o: State, policy) -> dict:
    if o.wind_speed_status != SPD_NUMERIC or o.wind_speed <= policy.dir_speed_min_kt:
        return _r("unavailable", reason="observed_speed_at_or_below_threshold")
    if f.wind_dir_status != DIR_NUMERIC or o.wind_dir_status != DIR_NUMERIC:
        return _r("unavailable", reason="direction_nonnumeric")
    d = abs(f.wind_dir - o.wind_dir) % 360
    d = min(d, 360 - d)
    tol = policy.dir_tol_hi_deg if o.wind_speed >= policy.dir_speed_threshold_kt else policy.dir_tol_lo_deg
    return _r("scored", earned=1.0 if d <= tol else 0.0, available=1,
              fcst_value=str(f.wind_dir), obs_value=f"{o.wind_dir}@{o.wind_speed}kt")


def _score_wind_gust(f: State, o: State, policy) -> dict:
    fg = f.gust_status == GUST_PRESENT
    og = o.gust_status == GUST_PRESENT
    if not fg and not og:
        return _r("scored", earned=1.0, available=1, fcst_value="none", obs_value="none")
    if fg and og:
        ok = abs(f.wind_gust - o.wind_gust) <= policy.gust_tol_kt
        return _r("scored", earned=1.0 if ok else 0.0, available=1,
                  fcst_value=f"G{f.wind_gust}", obs_value=f"G{o.wind_gust}")
    if og and not fg:               # observed-only: explicit 0 in the ACCI
        return _r("scored", earned=0.0, available=1, fcst_value="none",
                  obs_value=f"G{o.wind_gust}", reason="observed_only_gust")
    return _r("scored", earned=0.0, available=1, fcst_value=f"G{f.wind_gust}",
              obs_value="none", reason="forecast_only_gust")   # project policy, provisional


def _score_present_weather(f: State, o: State) -> dict:
    _, fcl = normalize_weather(f.weather)
    _, ocl = normalize_weather(o.weather)
    if not fcl and not ocl:
        return _r("unavailable", reason="both_weather_empty")
    hits = len(fcl & ocl)
    fa = len(fcl - ocl)
    miss = len(ocl - fcl)
    csi = hits / (hits + fa + miss) if (hits + fa + miss) else 0.0
    return _r("scored", earned=csi, available=1,
              fcst_value=",".join(sorted(fcl)) or "none",
              obs_value=",".join(sorted(ocl)) or "none")


def _score_altimeter(f: State, o: State, policy, group_type: str) -> dict:
    if group_type in ("TEMPO", "PROB"):
        return _r("unavailable", reason="tempo_altimeter_excluded")
    if f.qnh_status != QNH_KNOWN:
        return _r("unavailable", reason="forecast_qnh_missing")
    if o.qnh_status != QNH_KNOWN:
        return _r("unavailable", reason="observed_altimeter_missing")
    # integer hundredths; never round before the boundary test
    obs_h = round(o.qnh_inhg * 100)
    thr_h = round(f.qnh_inhg * 100) - round(policy.altimeter_tol_inhg * 100)
    ok = obs_h >= thr_h
    return _r("scored", earned=1.0 if ok else 0.0, available=1,
              fcst_value=f"{f.qnh_inhg:.2f}", obs_value=f"{o.qnh_inhg:.2f}")


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------

def score_tafver(
    taf: TafObs,
    obs: list[dict],
    valid_from: datetime,
    valid_to: datetime,
    *,
    profile: StationProfile | None = None,
    policy: TafverPolicy | None = None,
    truth_policy: TruthPolicy | None = None,
) -> TafverScore:
    """Run TAFVER over one archived TAF + its truth obs. Deterministic: identical
    inputs -> identical rows + combined (the run-immutability contract lands with
    persistence in the harness wire-in)."""
    profile = profile or default_profile(taf.station)
    policy = policy or TafverPolicy()
    hours, _manifest = build_truth(obs, valid_from, valid_to, policy=truth_policy)
    hour_map = {h.hour: h for h in hours}

    rows: list[TafverRow] = []
    for opp in opportunities(taf, valid_from, valid_to):
        # Acceptable forecast STATES for this opportunity. A baseline opportunity is the
        # EVOLVED prevailing at the interval midpoint (a BECMG that has COMPLETED is
        # already folded in); during a BECMG's own transition window the "becoming" state
        # is added as a BEST-OF alternate -- the hour is correct if the obs matches the
        # old OR the becoming conditions (never double-counted). Overlay (TEMPO/PROB)
        # opportunities resolve to their own fully-inherited state on their own rows.
        mid = opp.interval_start + (opp.interval_end - opp.interval_start) / 2
        if opp.role == "overlay":
            states = [resolve_group_state(taf, opp.group_index, valid_from, valid_to)]
        else:
            states = [forecast_state(taf, mid, valid_from=valid_from, valid_to=valid_to).prevailing]
            states += [resolve_group_state(taf, i, valid_from, valid_to)
                       for i in opp.alternate_indices]
        h = hour_map.get(opp.bin_start)
        base = dict(group_index=opp.group_index, group_type=opp.group_type,
                    interval_start=opp.interval_start, interval_end=opp.interval_end,
                    lead_hr=opp.lead_hr)
        if h is None or h.status != "available":
            reason = h.reason if h is not None else "no_hour"
            for el in _ELEMENTS:
                rows.append(TafverRow(**base, element=el,
                                      **_r("unavailable", reason=reason or "no_obs")))
            continue
        o = conservative_state(h)
        variants = [_score_all(f, o, profile, policy, opp.group_type) for f in states]
        for el in _ELEMENTS:
            rows.append(TafverRow(**base, element=el, **_best_of([v[el] for v in variants])))

    return _aggregate(taf, valid_from, valid_to, rows, obs, profile, policy)


def _score_all(f: State, o: State, profile, policy, group_type: str) -> dict:
    """Score all seven MOPs for one forecast state against the hourly truth."""
    return {
        "ceiling": _score_ceiling(f, o, profile),
        "visibility": _score_visibility(f, o, profile),
        "wind_speed": _score_wind_speed(f, o, policy),
        "wind_dir": _score_wind_dir(f, o, policy),
        "wind_gust": _score_wind_gust(f, o, policy),
        "present_weather": _score_present_weather(f, o),
        "altimeter": _score_altimeter(f, o, policy, group_type),
    }


def _best_of(variants: list[dict]) -> dict:
    """Best-of across acceptable forecast states for one element: if any state scored,
    take the highest-earning one (single point available -- never double-counted);
    otherwise the element is unavailable (first reason)."""
    scored = [v for v in variants if v["status"] == "scored"]
    if not scored:
        return variants[0]
    return max(scored, key=lambda v: v["points_earned"])


def _aggregate(taf, vf, vt, rows, obs, profile, policy) -> TafverScore:
    avail = [r for r in rows if r.status == "scored"]

    def summ(subset, element, bucket) -> ElementSummary:
        e = sum(r.points_earned for r in subset)
        a = sum(r.points_available for r in subset)
        return ElementSummary(element=element, bucket=bucket, earned=e, available=a,
                              percent=(100 * e / a) if a else None)

    element_summaries = [summ([r for r in avail if r.element == el], el, "ALL")
                         for el in _ELEMENTS]
    buckets = ["INITIAL", "FM", "BECMG", "TEMPO", "PROB"]
    for el in _ELEMENTS:
        for b in buckets:
            sub = [r for r in avail if r.element == el and r.group_type == b]
            if sub:
                element_summaries.append(summ(sub, el, b))

    group_type_summaries = []
    for b in buckets:
        sub = [r for r in avail if r.group_type == b]
        if sub:
            group_type_summaries.append(summ(sub, "combined", b))

    total_e = sum(r.points_earned for r in avail)
    total_a = sum(r.points_available for r in avail)

    # A7.2 category accuracy + bias (cig / vis)
    cat_stats: list[CategoryStat] = []
    for el in ("ceiling", "visibility"):
        el_rows = [r for r in avail if r.element == el]
        cats = sorted({c for r in el_rows for c in (r.fcst_cat, r.obs_cat) if c})
        for c in cats:
            fcst_hours = sum(1 for r in el_rows if r.fcst_cat == c)
            obs_hours = sum(1 for r in el_rows if r.obs_cat == c)
            when_obs = [r for r in el_rows if r.obs_cat == c]
            acc = (sum(r.points_earned for r in when_obs) / len(when_obs)) if when_obs else None
            cat_stats.append(CategoryStat(element=el, category=c, fcst_hours=fcst_hours,
                                          obs_hours=obs_hours, accuracy=acc,
                                          bias=(fcst_hours / obs_hours) if obs_hours else None))

    # present-weather accuracy + class-level event bias
    pw = [r for r in avail if r.element == "present_weather"]
    pw_acc = (sum(r.points_earned for r in pw) / len(pw)) if pw else None
    pw_bias: dict[str, float | None] = {}
    for cls in _PW_CLASSES:
        fh = sum(1 for r in pw if cls in (r.fcst_value or "").split(","))
        oh = sum(1 for r in pw if cls in (r.obs_value or "").split(","))
        pw_bias[cls] = (fh / oh) if oh else None

    return TafverScore(
        station=taf.station, valid_from=vf, valid_to=vt, provisional=profile.provisional,
        rows=rows, element_summaries=element_summaries,
        group_type_summaries=group_type_summaries,
        combined_earned=total_e, combined_available=total_a,
        combined_percent=(100 * total_e / total_a) if total_a else None,
        category_stats=cat_stats, pw_accuracy=pw_acc, pw_event_bias=pw_bias,
        policy_hash=_hash(policy.model_dump()), profile_hash=_hash(profile.model_dump()),
        obs_hash=obs_hash(obs),
    )


def fitl_value_added(fitl: TafverScore, model: TafverScore) -> dict:
    """The ACCI's paired FITL-minus-model(GALWEM) MOP difference (sec 7.2). This LABEL
    is reserved for exactly that pairing: REFUSED unless both scores share station,
    window, obs hash, profile hash, and policy hash. Our persistence/human/climo
    comparisons are BENCHMARK DELTAS (sec 10), never FITL value added."""
    for attr in ("station", "valid_from", "valid_to", "obs_hash", "profile_hash", "policy_hash"):
        if getattr(fitl, attr) != getattr(model, attr):
            return {"value_added": None, "reason": f"pairing_refused_{attr}_mismatch"}
    if fitl.combined_percent is None or model.combined_percent is None:
        return {"value_added": None, "reason": "null_combined"}
    return {"value_added": fitl.combined_percent - model.combined_percent,
            "fitl_percent": fitl.combined_percent, "model_percent": model.combined_percent}
