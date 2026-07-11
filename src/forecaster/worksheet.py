"""Typed TafWorksheet -> the pre-emit reasoning artifact (the WORKSHEET seam).

Sibling in spirit to tafgen.py: a pydantic product + a semantic validate() + a
model-facing guide(). Where tafgen owns the TAF OUTPUT (a typed TafProduct the
model fills, rendered to canonical text), this owns the pre-forecast WORKSHEET --
the structured record of the model's forecast REASONING that it fills and submits
through its own validation sink BEFORE emit_taf.

Why it exists (CLAUDE.md, the KLSV agent runs): two systematic, reproducible
misses -- a model read the observed DEWPOINT as the overnight TN, and another
converted the SAME MSLP to different inHg values at different points. The
worksheet makes those two cross-checks first-class (sanity_checks) and decomposes
the task so a weaker model converges (fill a timeline + a strategy) instead of
ruminating to the token cap.

Design (docs/taf_worksheet_design.md, Milestone 1):
  - ONE typed object, submitted as a single validated sink call (submit_taf_worksheet
    in tools.py), NOT filled section-by-section -- coherence + one unambiguous
    "commit now" moment (Locked decision 4).
  - Guardrails (pydantic) reject IMPOSSIBLE values at construction (bad enum, a
    non-ICAO station); validate() catches WELL-FORMED-BUT-INCOMPLETE worksheets and
    returns findings (a list of strings, never raises) -- the same feedback-not-crash
    contract as tafgen.validate(). Findings are SECTION-PREFIXED (e.g.
    'sanity_checks: ...') so a driver's required-mode gate can exclude the advisory
    'model_run_verification:' findings cleanly.
  - worksheet_guide() spells out the shape the JSON schema hides behind
    anyOf[$ref, null] (modeled on tafgen.emit_taf_guide()); it ships WITH the schema.

No SQL, no matplotlib, no network -- the model is filled here and persisted by the
store seam, exactly like a TafProduct.
"""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

# ---- Reusable coerced enums -------------------------------------------------
# Models emit enum values loosely (title-case, stray whitespace); a strict Literal
# would reject 'Moderate'/'HIGH' as a schema error and burn a turn. Coerce to the
# canonical lower-case token BEFORE validation (the emit-tool-arg-quirks lesson:
# an output schema must coerce, not just reject). A non-string passes through so the
# real "not a legal value" error still fires.

def _lower(v: object) -> object:
    return v.strip().lower() if isinstance(v, str) else v


Confidence = Annotated[Literal["low", "moderate", "high"], BeforeValidator(_lower)]
RiskLevel = Annotated[Literal["none", "low", "moderate", "high"], BeforeValidator(_lower)]
Quality = Annotated[Literal["good", "mixed", "poor", "unknown"], BeforeValidator(_lower)]
ReviewStatus = Annotated[
    Literal["reviewed", "attempted_unavailable", "skipped"], BeforeValidator(_lower)
]
SourceType = Annotated[Literal["tool", "manual_input", "document"], BeforeValidator(_lower)]
DriverType = Annotated[
    Literal["synoptic", "mesoscale", "local", "climo", "model_signal", "observed_trend"],
    BeforeValidator(_lower),
]
HazardKind = Annotated[
    Literal["ceiling", "visibility", "fog", "precip", "thunder", "wind", "wind_shear",
            "icing", "turbulence", "freezing_precip", "other"],
    BeforeValidator(_lower),
]
TafConstruct = Annotated[
    Literal["prevailing", "fm", "becmg", "tempo", "omit"], BeforeValidator(_lower)
]
TimelineStrategy = Annotated[
    Literal["event_based", "block_based", "hybrid"], BeforeValidator(_lower)
]
TafStyle = Annotated[Literal["af", "civil", "other"], BeforeValidator(_lower)]
AmendmentContext = Annotated[Literal["routine", "amd", "cor"], BeforeValidator(_lower)]

# Evidence-ref lists appear on every reasoning section; alias the shape once.
EvidenceRefs = list[str]


# ---- Section 1: meta (mostly system-filled) ---------------------------------

class Meta(BaseModel):
    """Identity + experiment controls. The SINK fills these from run context (mode,
    evidence_mode, model, station, timestamps); the model need not author them."""

    worksheet_version: str = "1.0"
    mode: str | None = None                # off | advisory | required (set by the sink)
    evidence_mode: str | None = None       # off | key_claims | strict (set by the sink)
    model: str | None = None
    created_at_utc: str | None = None
    completed_at_utc: str | None = None
    station: str | None = None
    source_session_id: str | None = None


# ---- Section 2: task --------------------------------------------------------

class Task(BaseModel):
    """What forecast is being produced. station is the only hard guardrail (ICAO)."""

    station: str = Field(pattern=r"^[A-Z]{4}$")
    forecast_type: str = ""                # e.g. "routine 30h AF TAF"
    valid_from_utc: str = ""               # ISO or DDHHMMZ; free text, cross-checked by the model
    valid_to_utc: str = ""
    taf_style: TafStyle = "af"
    amendment_context: AmendmentContext = "routine"
    user_constraints: str | None = None


# ---- Section 3: data_review -------------------------------------------------

class ReviewItem(BaseModel):
    """One product/tool the model consulted (or tried to). status records whether it
    was actually reviewed, so a gap (attempted_unavailable) is auditable, not silent."""

    source_type: SourceType = "tool"
    source_name: str = ""                  # e.g. "get_trend", "get_map(surface_prog)"
    purpose: str = ""
    status: ReviewStatus = "reviewed"
    takeaway: str = ""
    evidence_refs: EvidenceRefs = []


class DataReview(BaseModel):
    required_reviews: list[ReviewItem] = []
    optional_reviews: list[ReviewItem] = []
    missing_inputs: list[str] = []
    data_quality_notes: str | None = None


# ---- Section 4: current_state ----------------------------------------------

class CurrentState(BaseModel):
    """The operational STARTING point (short), not a narrative dump."""

    observed_regime_summary: str = ""
    flight_category_now: str | None = None     # VFR/MVFR/IFR/LIFR (free text; not gated)
    ceiling_visibility_state: str = ""
    wind_state: str = ""
    precipitation_state: str = ""
    convection_state: str = ""
    temperature_moisture_state: str = ""
    confidence: Confidence | None = None
    evidence_refs: EvidenceRefs = []


# ---- Section 5: forecast_drivers -------------------------------------------

class Driver(BaseModel):
    """One meteorological factor that actually drives the TAF."""

    name: str
    type: DriverType = "synoptic"
    why_it_matters: str = ""
    expected_effect: str = ""
    timing: str = ""
    confidence: Confidence | None = None
    evidence_refs: EvidenceRefs = []


class ForecastDrivers(BaseModel):
    primary_drivers: list[Driver] = []
    secondary_drivers: list[Driver] = []
    discarded_signals: list[str] = []
    driver_conflicts: list[str] = []


# ---- Section 6: hazards -----------------------------------------------------

class Hazard(BaseModel):
    """A hazard CONSIDERED -- records what was rejected (risk none/low), not only what
    is forecast, so the reasoning is auditable."""

    hazard: HazardKind
    risk_level: RiskLevel
    timing: str = ""
    rationale: str = ""
    taf_relevance: str = ""
    evidence_refs: EvidenceRefs = []


class Hazards(BaseModel):
    hazard_assessment: list[Hazard] = []


# ---- Section 7: forecast_timeline ------------------------------------------

class TimelinePeriod(BaseModel):
    """One forecast block. The SOURCE MATERIAL for FM/BECMG/TEMPO decisions -- a
    taf_strategy change group references a period by its label."""

    label: str
    start_utc: str = ""
    end_utc: str = ""
    expected_conditions: str = ""
    flight_category_expected: str | None = None
    dominant_driver_ids: list[str] = []    # references into forecast_drivers (by name)
    key_uncertainties: str | None = None
    evidence_refs: EvidenceRefs = []


class ForecastTimeline(BaseModel):
    timeline_strategy: TimelineStrategy = "event_based"
    periods: list[TimelinePeriod] = []


# ---- Section 8: sanity_checks (guards the two known misses) -----------------

class SanityChecks(BaseModel):
    """First-class BECAUSE these are the specific errors that justified the feature.
    The validator checks the fields are present and non-empty; the values are the
    model's explicit cross-checks."""

    # each forecast TX/TN vs the observed diurnal range (guards dewpoint-as-TN)
    tx_tn_vs_observed: str = ""
    # source hPa/inHg value + the SINGLE converted value used everywhere (guards the
    # inconsistent re-conversion slip)
    qnh_conversion_check: str = ""
    # diurnal wind/gust cycle per valid day vs obs (+ climo when built)
    wind_diurnal_check: str = ""
    # is any vis/ceiling restriction expected, and why/why not (so SKC/9999 is a decision)
    restriction_check: str = ""
    evidence_refs: EvidenceRefs = []


# ---- Section 9: taf_strategy (bridge to emit_taf) ---------------------------

class ChangeGroupStrategy(BaseModel):
    """How one timeline period becomes TAF structure. timeline_period_label MUST match
    a forecast_timeline period label (validate() checks the reference resolves)."""

    timeline_period_label: str
    expected_taf_construct: TafConstruct
    why: str = ""
    confidence: Confidence | None = None


class TafStrategy(BaseModel):
    """Where worksheet reasoning becomes TAF structure -- shaped so emit_taf is a
    near-mechanical transcription of these fields."""

    prevailing_strategy: str = ""
    change_group_strategy: list[ChangeGroupStrategy] = []
    amendment_watch_items: list[str] = []
    temperature_strategy: str = ""        # TX/TN values + timing, consistent w/ sanity_checks
    wind_strategy: str = ""
    visibility_ceiling_strategy: str = ""
    hazard_group_strategy: str | None = None   # WS/VA/icing/turbulence per the tafgen hazards
    coding_notes: str | None = None


# ---- Section 10: uncertainty ------------------------------------------------

class UncertaintyItem(BaseModel):
    issue: str
    impact_on_taf: str = ""
    most_likely_outcome: str = ""
    alternate_outcome: str = ""
    confidence: Confidence | None = None
    evidence_refs: EvidenceRefs = []


class UncertaintyAnalysis(BaseModel):
    main_uncertainties: list[UncertaintyItem] = []
    alternative_scenarios: list[str] = []
    decision_points: list[str] = []
    monitoring_recommendations: list[str] = []


# ---- Section 11: final_assessment -------------------------------------------

class FinalAssessment(BaseModel):
    forecast_summary: str = ""
    biggest_risk_to_accuracy: str = ""
    overall_confidence: Confidence | None = None
    # must NOT be true while any required section is missing/incomplete (validate() checks)
    ready_for_emit_taf: bool = False
    validation_notes: str | None = None


# ---- Section 12: model_run_verification (advisory, even in required mode) ----

class GuidanceEntry(BaseModel):
    source: str
    forecast_hour_or_window: str = ""
    what_was_checked: str = ""
    initialization_quality: Quality = "unknown"
    recent_verification_quality: Quality = "unknown"
    key_bias_or_issue: str | None = None
    how_it_affects_forecast_use: str | None = None
    evidence_refs: EvidenceRefs = []


class ModelRunVerification(BaseModel):
    """Advisory until get_model_run_verification exists (Milestone 2): the object must
    be PRESENT but fields may be 'unknown'; the model is not gated on it. The one
    supportable check today is a cold-start disagreement (freshest guidance vs latest ob)."""

    guidance_sources_reviewed: list[str] = []
    initialization_assessment: str | None = None     # the supportable cold-start check
    recent_verification_assessment: str | None = None  # usually 'unknown' in v1
    trusted_guidance: str | None = None
    discounted_guidance: str | None = None
    guidance_blend_strategy: str | None = None
    guidance_entries: list[GuidanceEntry] = []


# ---- Top-level worksheet ----------------------------------------------------
# Every SECTION is optional (default None): an omitted section becomes a clean
# semantic finding ("section required") rather than a hard pydantic rejection, which
# is the "well-formed-but-incomplete -> feedback, not crash" contract. Within a
# provided section, IMPOSSIBLE values (bad enum, missing item id like a Driver.name)
# still fail at construction -- named, correctable feedback like emit_taf's TafTemp.

class TafWorksheet(BaseModel):
    meta: Meta | None = None
    task: Task | None = None
    data_review: DataReview | None = None
    current_state: CurrentState | None = None
    forecast_drivers: ForecastDrivers | None = None
    hazards: Hazards | None = None
    forecast_timeline: ForecastTimeline | None = None
    sanity_checks: SanityChecks | None = None
    taf_strategy: TafStrategy | None = None
    uncertainty: UncertaintyAnalysis | None = None
    final_assessment: FinalAssessment | None = None
    model_run_verification: ModelRunVerification | None = None


# ---- Semantic validator -----------------------------------------------------
# Comprehensive completeness check for `required` mode; in `advisory` mode the SAME
# findings are produced but the driver does not block emit_taf on them (the plan's
# Validation policy). Findings are section-prefixed; 'model_run_verification:' findings
# stay advisory even in required mode (Locked decision 3), so a driver gate should
# exclude that prefix.

# Required data-review coverage for v1: each category needs a reviewed item whose
# source_name names one of these tools (all exist today).
_REQUIRED_REVIEWS: dict[str, set[str]] = {
    "latest/recent observations": {"get_latest_obs", "query_obs"},
    "recent trend (meteogram)": {"get_trend"},
    "a synoptic map": {"get_map"},
    "a forecast profile": {"get_fcst_sounding", "get_point_forecast"},
}

MODEL_RUN_VERIFICATION_PREFIX = "model_run_verification"


def _blank(s: str | None) -> bool:
    """True when a string field is missing or whitespace-only."""
    return not (s or "").strip()


def _covered(reviews: list[ReviewItem], tool_names: set[str]) -> bool:
    """True if any REVIEWED item's source_name mentions one of tool_names."""
    for r in reviews:
        if r.status != "reviewed":
            continue
        name = (r.source_name or "").lower()
        if any(t in name for t in tool_names):
            return True
    return False


def _check_evidence(section: str, refs: EvidenceRefs, known: set[str] | None,
                    evidence_mode: str, out: list[str]) -> None:
    """key_claims/strict: a material claim needs at least one evidence_ref, and (when
    the loop has threaded the ids, i.e. `known` is provided) each ref must resolve to a
    real evidence_id. With `known` None (self-test / no threading) only presence is
    checked -- resolution is agent-loop plumbing, not schema (design 'Scope note')."""
    if evidence_mode not in ("key_claims", "strict"):
        return
    if not refs:
        out.append(f"{section}: evidence_mode={evidence_mode} needs >=1 evidence_ref")
        return
    if known is not None:
        bad = [r for r in refs if r not in known]
        if bad:
            out.append(f"{section}: evidence_refs do not resolve to a tool call: {bad}")


def validate(
    ws: TafWorksheet,
    *,
    mode: str = "required",
    evidence_mode: str = "key_claims",
    known_evidence_ids: list[str] | None = None,
) -> list[str]:
    """Check a TafWorksheet for completeness/coherence. Returns section-prefixed
    findings (empty = complete). A CHECKER, not an exception raiser -- the sink hands
    findings back so the model re-submits a fix. `mode` is informational in v1 (the
    same checks run for advisory + required; the driver gates only in required);
    `known_evidence_ids` enables evidence-ref RESOLUTION when the loop has threaded
    ids (else presence-only)."""
    out: list[str] = []
    known = set(known_evidence_ids) if known_evidence_ids is not None else None

    # 2. task
    if ws.task is None:
        out.append("task: section is required (station, forecast_type, valid period)")
    else:
        if _blank(ws.task.forecast_type):
            out.append("task: forecast_type is empty")
        if _blank(ws.task.valid_from_utc) or _blank(ws.task.valid_to_utc):
            out.append("task: valid_from_utc/valid_to_utc must both be set")

    # 3. data_review (comprehensive: the four v1 required categories)
    if ws.data_review is None:
        out.append("data_review: section is required")
    else:
        for label, tools_for in _REQUIRED_REVIEWS.items():
            if not _covered(ws.data_review.required_reviews, tools_for):
                out.append(f"data_review: no reviewed item covers {label} "
                           f"({'/'.join(sorted(tools_for))})")
        for i, r in enumerate(ws.data_review.required_reviews):
            if r.status == "reviewed" and _blank(r.takeaway):
                out.append(f"data_review: required_reviews[{i}] ({r.source_name}) "
                           "has no takeaway")

    # 4. current_state
    if ws.current_state is None:
        out.append("current_state: section is required")
    elif _blank(ws.current_state.observed_regime_summary):
        out.append("current_state: observed_regime_summary is empty")
    elif evidence_mode in ("key_claims", "strict"):
        _check_evidence("current_state", ws.current_state.evidence_refs, known, evidence_mode, out)

    # 5. forecast_drivers (>=1 primary, each substantive)
    if ws.forecast_drivers is None:
        out.append("forecast_drivers: section is required")
    else:
        prim = ws.forecast_drivers.primary_drivers
        if not prim:
            out.append("forecast_drivers: at least one primary_driver is required")
        for i, d in enumerate(prim):
            if _blank(d.why_it_matters) or _blank(d.expected_effect):
                out.append(f"forecast_drivers: primary_drivers[{i}] ({d.name}) needs "
                           "why_it_matters and expected_effect")
            _check_evidence(f"forecast_drivers.primary[{i}]", d.evidence_refs, known,
                            evidence_mode, out)

    # 6. hazards (>=1 assessment)
    if ws.hazards is None:
        out.append("hazards: section is required")
    else:
        if not ws.hazards.hazard_assessment:
            out.append("hazards: at least one hazard_assessment is required")
        for i, h in enumerate(ws.hazards.hazard_assessment):
            if _blank(h.rationale):
                out.append(f"hazards: hazard_assessment[{i}] ({h.hazard}) needs a rationale")
            _check_evidence(f"hazards[{i}]", h.evidence_refs, known, evidence_mode, out)

    # 7. forecast_timeline (>=1 period); collect labels for the taf_strategy cross-ref
    timeline_labels: set[str] = set()
    if ws.forecast_timeline is None:
        out.append("forecast_timeline: section is required")
    else:
        periods = ws.forecast_timeline.periods
        if not periods:
            out.append("forecast_timeline: at least one period is required")
        for i, p in enumerate(periods):
            timeline_labels.add(p.label)
            if _blank(p.expected_conditions):
                out.append(f"forecast_timeline: period[{i}] ({p.label}) has no expected_conditions")
            _check_evidence(f"forecast_timeline[{i}]", p.evidence_refs, known, evidence_mode, out)

    # 8. sanity_checks (all four present + non-empty -- the whole point of the feature)
    if ws.sanity_checks is None:
        out.append("sanity_checks: section is required (guards the TX/TN and QNH slips)")
    else:
        sc = ws.sanity_checks
        for field_name, why in (
            ("tx_tn_vs_observed", "cross-check each TX/TN vs the observed diurnal range"),
            ("qnh_conversion_check", "state the source pressure + the single converted value"),
            ("wind_diurnal_check", "state the diurnal wind/gust cycle per valid day"),
            ("restriction_check", "state whether a vis/ceiling restriction is expected, and why"),
        ):
            if _blank(getattr(sc, field_name)):
                out.append(f"sanity_checks: {field_name} is empty -- {why}")
        _check_evidence("sanity_checks", sc.evidence_refs, known, evidence_mode, out)

    # 9. taf_strategy (bridge; change-group labels must reference real timeline periods)
    if ws.taf_strategy is None:
        out.append("taf_strategy: section is required")
    else:
        ts = ws.taf_strategy
        for field_name in ("prevailing_strategy", "temperature_strategy", "wind_strategy",
                           "visibility_ceiling_strategy"):
            if _blank(getattr(ts, field_name)):
                out.append(f"taf_strategy: {field_name} is empty")
        for i, cg in enumerate(ts.change_group_strategy):
            if cg.timeline_period_label not in timeline_labels:
                out.append(f"taf_strategy: change_group_strategy[{i}] references timeline period "
                           f"{cg.timeline_period_label!r}, which is not in forecast_timeline")

    # 10. uncertainty (>=1)
    if ws.uncertainty is None:
        out.append("uncertainty: section is required")
    elif not ws.uncertainty.main_uncertainties:
        out.append("uncertainty: at least one main_uncertainty is required")

    # 11. final_assessment
    required_incomplete = bool(out)      # any finding so far => required sections incomplete
    if ws.final_assessment is None:
        out.append("final_assessment: section is required")
    else:
        fa = ws.final_assessment
        if _blank(fa.forecast_summary):
            out.append("final_assessment: forecast_summary is empty")
        if _blank(fa.biggest_risk_to_accuracy):
            out.append("final_assessment: biggest_risk_to_accuracy is empty")
        if fa.ready_for_emit_taf and required_incomplete:
            out.append("final_assessment: ready_for_emit_taf is true but required sections "
                       "are still incomplete (see the findings above)")

    # 12. model_run_verification -- ADVISORY even in required mode. Object must be
    # present, but fields may be 'unknown'. Prefixed so a driver gate can exclude it.
    if ws.model_run_verification is None:
        out.append(f"{MODEL_RUN_VERIFICATION_PREFIX}: object should be present (fields may be "
                   "'unknown' until get_model_run_verification exists)")

    return out


def blocking_findings(findings: list[str]) -> list[str]:
    """The subset of validate() findings that GATE emit_taf in required mode: everything
    except the advisory model_run_verification ones (Locked decision 3). A driver uses
    this to decide whether to refuse emit_taf."""
    return [f for f in findings if not f.startswith(MODEL_RUN_VERIFICATION_PREFIX + ":")]


# ---- Model-facing reference guide -------------------------------------------
# Same rationale + pattern as tafgen.emit_taf_guide(): the JSON schema hides nested
# fields behind anyOf[$ref, null], so the model guesses the shape and ruminates. This
# spells it out. Generated FROM a live worksheet that the self-test also validate()s,
# so the guide physically cannot drift from the rules the sink enforces.


def _example_worksheet() -> TafWorksheet:
    """A worked, COMPLETE worksheet for the guide + self-test. A dry summer ridge at
    KLSV (the repo's target case): SKC/9999 held, a diurnal wind cycle, no restriction.
    validate() must come back clean for evidence_mode key_claims (asserted by the test)."""
    return TafWorksheet(
        task=Task(
            station="KLSV", forecast_type="routine 30h AF TAF",
            valid_from_utc="072300Z", valid_to_utc="090500Z",
        ),
        data_review=DataReview(
            required_reviews=[
                ReviewItem(source_name="get_latest_obs(KLSV)", purpose="current conditions",
                           takeaway="25017G22KT, SKC, 9999, 40C/-5C, A2975 at 2255Z",
                           evidence_refs=["ev_001"]),
                ReviewItem(source_name="get_trend(KLSV, 24h)", purpose="recent trend",
                           takeaway="diurnal wind cycle; sky clear all period; pressure steady-rising",
                           evidence_refs=["ev_002"]),
                ReviewItem(source_name="get_map(surface_analysis)", purpose="synoptic pattern",
                           takeaway="broad ridge over the Great Basin; no fronts in 30h",
                           evidence_refs=["ev_003"]),
                ReviewItem(source_name="get_point_forecast(KLAS, gfs)", purpose="model evolution",
                           takeaway="GFS holds dry; T2M 40C afternoon, 27C overnight; wind veers diurnally",
                           evidence_refs=["ev_004"]),
            ],
            optional_reviews=[
                ReviewItem(source_name="get_climo(KLSV, 7)", purpose="typical July",
                           takeaway="TX mean 40.5C; afternoon SW wind; TS 0.7%; fog 0%",
                           evidence_refs=["ev_005"]),
            ],
            missing_inputs=["no BUFKIT output at KLSV; used KLAS as a proxy"],
        ),
        current_state=CurrentState(
            observed_regime_summary="hot, dry, cloud-free summer ridge; gusty SW afternoon wind",
            flight_category_now="VFR",
            ceiling_visibility_state="unlimited ceiling, 9999 visibility",
            wind_state="250 at 17 gusting 22 kt (afternoon peak)",
            precipitation_state="none; PoP ~0",
            convection_state="no convection; CAPE negligible under the ridge",
            temperature_moisture_state="40C / -5C dewpoint; very dry boundary layer",
            confidence="high", evidence_refs=["ev_001", "ev_002"],
        ),
        forecast_drivers=ForecastDrivers(
            primary_drivers=[
                Driver(name="Great Basin ridge", type="synoptic",
                       why_it_matters="suppresses clouds and precip for the whole period",
                       expected_effect="SKC 9999 persists", timing="entire 30h",
                       confidence="high", evidence_refs=["ev_003"]),
                Driver(name="diurnal boundary-layer mixing", type="local",
                       why_it_matters="drives the daily wind speed/direction cycle",
                       expected_effect="gusty SW afternoons, light/variable overnight",
                       timing="peaks ~21-00Z, minimum ~09-12Z",
                       confidence="high", evidence_refs=["ev_002", "ev_004"]),
            ],
            discarded_signals=["a weak model PoP blip -- below TAF threshold, dry column"],
        ),
        hazards=Hazards(hazard_assessment=[
            Hazard(hazard="wind", risk_level="moderate", timing="afternoons (21-00Z)",
                   rationale="gusts to ~22 kt each afternoon with mixing",
                   taf_relevance="gust group on the prevailing + next-afternoon FM",
                   evidence_refs=["ev_001", "ev_004"]),
            Hazard(hazard="thunder", risk_level="none", timing="n/a",
                   rationale="dry column under a ridge; climo TS 0.7% in July",
                   taf_relevance="no TS/CB group", evidence_refs=["ev_005"]),
            Hazard(hazard="fog", risk_level="none", timing="n/a",
                   rationale="dewpoint -5C; climo fog 0%",
                   taf_relevance="no restriction", evidence_refs=["ev_005"]),
        ]),
        forecast_timeline=ForecastTimeline(
            timeline_strategy="block_based",
            periods=[
                TimelinePeriod(label="P1 afternoon peak", start_utc="072300Z", end_utc="080200Z",
                               expected_conditions="25017G22KT SKC 9999",
                               flight_category_expected="VFR",
                               dominant_driver_ids=["diurnal boundary-layer mixing"],
                               evidence_refs=["ev_001"]),
                TimelinePeriod(label="P2 overnight lull", start_utc="080200Z", end_utc="081500Z",
                               expected_conditions="VRB05KT SKC 9999, low ~27C",
                               flight_category_expected="VFR",
                               dominant_driver_ids=["diurnal boundary-layer mixing"],
                               evidence_refs=["ev_004"]),
                TimelinePeriod(label="P3 next afternoon", start_utc="081500Z", end_utc="090500Z",
                               expected_conditions="23018G25KT SKC 9999, high ~40C",
                               flight_category_expected="VFR",
                               dominant_driver_ids=["diurnal boundary-layer mixing"],
                               evidence_refs=["ev_004"]),
            ],
        ),
        sanity_checks=SanityChecks(
            tx_tn_vs_observed="TX 40C @0822Z, TN 27C @0812Z. Observed range 27-40C over the last "
            "24h; TN 27C is well above the -5C dewpoint (NOT the dewpoint) -- passes.",
            qnh_conversion_check="latest ob A2975 = 29.75 inHg; used 29.75 for the prevailing QNH "
            "and 2952 for the overnight FM (one conversion each, not re-derived).",
            wind_diurnal_check="obs + GFS agree: SW gusts ~22 kt peaking 21-00Z, light/variable "
            "09-12Z; climo confirms the afternoon SW maximum. Gust groups placed on the two afternoons.",
            restriction_check="no vis/ceiling restriction expected -- dry ridge, dewpoint -5C, "
            "climo fog 0%; SKC 9999 is a deliberate call, not a default.",
            evidence_refs=["ev_001", "ev_002", "ev_004", "ev_005"],
        ),
        taf_strategy=TafStrategy(
            prevailing_strategy="25017G22KT 9999 SKC QNH2975INS for the afternoon peak",
            change_group_strategy=[
                ChangeGroupStrategy(timeline_period_label="P2 overnight lull",
                                    expected_taf_construct="fm",
                                    why="wind eases and veers overnight", confidence="high"),
                ChangeGroupStrategy(timeline_period_label="P3 next afternoon",
                                    expected_taf_construct="fm",
                                    why="afternoon gusts rebuild", confidence="high"),
            ],
            amendment_watch_items=["unexpected gust > 30 kt", "any cirrus blow-off from distant convection"],
            temperature_strategy="TX 40/0822Z, TN 27/0812Z (consistent with sanity_checks)",
            wind_strategy="gusty SW afternoons, VRB05 overnight; gust groups on both afternoons",
            visibility_ceiling_strategy="9999 SKC throughout -- no restriction",
        ),
        uncertainty=UncertaintyAnalysis(
            main_uncertainties=[
                UncertaintyItem(issue="exact overnight wind minimum timing",
                                impact_on_taf="FM time for the lull",
                                most_likely_outcome="VRB/light by ~08Z",
                                alternate_outcome="stays 08-10 kt if mixing lingers",
                                confidence="moderate", evidence_refs=["ev_004"]),
            ],
            monitoring_recommendations=["watch the 06-09Z obs for the wind fall-off"],
        ),
        final_assessment=FinalAssessment(
            forecast_summary="hot dry ridge: SKC 9999 all period with a diurnal SW wind cycle "
            "(gusty afternoons, light overnight); no restrictions or convection.",
            biggest_risk_to_accuracy="overnight low + wind-minimum timing (TN and the lull FM)",
            overall_confidence="high", ready_for_emit_taf=True,
        ),
        model_run_verification=ModelRunVerification(
            guidance_sources_reviewed=["GFS point forecast (KLAS proxy)"],
            initialization_assessment="GFS f000 ~matches the latest KLAS ob (T, wind); init looks good",
            recent_verification_assessment="unknown -- get_model_run_verification not yet available",
            guidance_blend_strategy="lean on obs+climo for the diurnal cycle; GFS for the trend",
            guidance_entries=[
                GuidanceEntry(source="GFS (KLAS)", forecast_hour_or_window="f000-f030",
                              what_was_checked="surface T, wind, MSLP vs the latest ob",
                              initialization_quality="good", recent_verification_quality="unknown",
                              key_bias_or_issue="none evident at init",
                              how_it_affects_forecast_use="used for trend, not absolute values",
                              evidence_refs=["ev_004"]),
            ],
        ),
    )


def worksheet_guide(evidence_mode: str = "key_claims") -> str:
    """A model-facing reference for the submit_taf_worksheet shape: a worked, valid
    worksheet (as JSON) plus a flattened field guide naming the fields the JSON schema
    hides behind anyOf[$ref, null]. For the driver system prompt -- NOT a replacement
    for the pydantic schema, which stays the validator."""
    ws = _example_worksheet()
    example_json = json.dumps(ws.model_dump(exclude_defaults=True, exclude_none=True), indent=2)
    ev_line = ("Because evidence_mode=key_claims, each driver, timeline period, hazard, and the "
               "sanity_checks/current_state sections need >=1 evidence_ref (an ev_NNN id echoed "
               "in a tool receipt)." if evidence_mode in ("key_claims", "strict")
               else "evidence_refs are optional in this run (evidence_mode=off).")
    return "\n".join([
        "submit_taf_worksheet SHAPE -- fill this reasoning worksheet BEFORE emit_taf.",
        "Submit it ONCE as a single object (you reason across tool calls first, then commit).",
        "",
        "Worked example (a valid, complete worksheet for a dry summer ridge):",
        "```json",
        example_json,
        "```",
        "",
        "Section guide (all sections are required in `required` mode; keep each concise):",
        "  meta               -- system-filled; you may omit it.",
        "  task               -- station (4-letter ICAO), forecast_type, valid_from_utc/valid_to_utc,",
        "                        taf_style (af|civil|other), amendment_context (routine|amd|cor).",
        "  data_review        -- required_reviews: a list of {source_name, purpose, status, takeaway,",
        "                        evidence_refs}. Cover ALL four: latest/recent obs, the meteogram",
        "                        (get_trend), a synoptic map (get_map), and a forecast profile",
        "                        (get_fcst_sounding OR get_point_forecast). status is",
        "                        reviewed|attempted_unavailable|skipped.",
        "  current_state      -- observed_regime_summary + the ceiling/vis/wind/precip/convection/",
        "                        temperature-moisture states; short. confidence low|moderate|high.",
        "  forecast_drivers   -- primary_drivers (>=1): each {name, type, why_it_matters,",
        "                        expected_effect, timing, confidence, evidence_refs}. type is one of",
        "                        synoptic|mesoscale|local|climo|model_signal|observed_trend.",
        "  hazards            -- hazard_assessment (>=1): each {hazard, risk_level, timing, rationale,",
        "                        taf_relevance, evidence_refs}. Record hazards you REJECTED too",
        "                        (risk_level none/low). risk_level none|low|moderate|high.",
        "  forecast_timeline  -- timeline_strategy (event_based|block_based|hybrid) + periods (>=1):",
        "                        each {label, start_utc, end_utc, expected_conditions,",
        "                        flight_category_expected, dominant_driver_ids, evidence_refs}.",
        "                        A change group later references a period by its LABEL.",
        "  sanity_checks      -- ALL FOUR strings, non-empty (this is why the worksheet exists):",
        "                        tx_tn_vs_observed (cross-check each TX/TN vs the observed diurnal",
        "                        range -- a TN near the dewpoint is a RED FLAG), qnh_conversion_check",
        "                        (source hPa/inHg + the ONE converted value used everywhere),",
        "                        wind_diurnal_check, restriction_check.",
        "  taf_strategy       -- prevailing_strategy, change_group_strategy (each {timeline_period_label",
        "                        [must match a forecast_timeline label], expected_taf_construct",
        "                        [prevailing|fm|becmg|tempo|omit], why, confidence}),",
        "                        temperature_strategy, wind_strategy, visibility_ceiling_strategy.",
        "  uncertainty        -- main_uncertainties (>=1): each {issue, impact_on_taf,",
        "                        most_likely_outcome, alternate_outcome, confidence, evidence_refs}.",
        "  final_assessment   -- forecast_summary, biggest_risk_to_accuracy, overall_confidence,",
        "                        ready_for_emit_taf (only true when the worksheet is complete).",
        "  model_run_verification -- present but may be 'unknown' (advisory in v1): guidance init",
        "                        check vs the latest ob; guidance_entries with good|mixed|poor|unknown.",
        "",
        ev_line,
        "",
        "The reply is a completeness check, not data: fix any findings it returns and re-submit "
        "until clean, THEN emit_taf using the forecast_timeline + taf_strategy you wrote.",
    ])
