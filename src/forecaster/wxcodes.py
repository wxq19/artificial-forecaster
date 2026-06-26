"""Present-weather classification + a deterministic severity rule.

Grounded in the FMH-1 / METAR present-weather table (docs/Present Weather
Values.png). Two separable concerns:
  - FAMILY: the phenomenon group used to COLOR a chart (rain, snow, ...).
  - SEVERITY: a single 0-10 operational-hazard score to rank/select weather
    (e.g. "top-3 most severe families" on a meteogram) so we never guess.

Severity rule (the agreed manual rule):
  - Fixed-tier phenomena carry an explicit score; within a tier the left-most
    code is most severe (encoded as a decimal fraction).
  - Descriptor escalators dominate: TS -> thunder tier, FZ -> freezing tier,
    over whatever precip they attach to.
  - Intensity adjusts fixed precip: '+' heavier, '-' lighter; 'VC' (in the
    vicinity, not at the field) is halved.
  - Obscurations have no intensity, so their severity IS the associated
    visibility (lower vis = worse) via FAA-style flight-category buckets.
    Exceptions: VA (volcanic ash) is an engine hazard, not a vis hazard ->
    fixed high; DZ (drizzle) is treated as an obscuration per its vis impact.

Deferred (v1 does NOT yet model these descriptors): BL/DR (blowing/drifting
raise the vis hazard), MI/PR/BC (shallow/partial/patches lower it), SH/UP.
"""

from dataclasses import dataclass

DESCRIPTORS = {"MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ"}

# color families (decoupled from severity)
_FAMILY = {
    "DZ": "rain", "RA": "rain",
    "SN": "snow", "SG": "snow", "IC": "snow",
    "PL": "ice", "GR": "ice", "GS": "ice",
    "BR": "fog", "FG": "fog",
    "FU": "haze", "HZ": "haze", "PY": "haze", "VA": "haze",
    "DU": "dust", "SA": "dust", "PO": "dust", "DS": "dust", "SS": "dust",
    "SQ": "other", "FC": "other",
}

# fixed severities; decimal fraction = intra-tier (left-most-worse) order
_FIXED_SEV = {
    "FC": 10.3, "DS": 10.2, "SS": 10.1,
    "TS": 9.3, "SQ": 9.2, "VA": 9.1,
    "GR": 7.0,                       # FZ handled in _severity (FZRA > FZDZ)
    "PL": 6.2, "GS": 6.1,
    "SN": 5.3, "SG": 5.2, "IC": 5.1,
    "RA": 3.0,
    "PO": 1.0,
}

# vis-driven obscurations: per-type offset breaks ties within a vis bucket
_OBSC_TIEBREAK = {"DU": 0.7, "SA": 0.6, "FU": 0.5, "FG": 0.4,
                  "BR": 0.3, "HZ": 0.2, "PY": 0.1, "DZ": 0.0}

_INTENSITY = {"-": "light", "+": "heavy", "VC": "vicinity", "": "moderate"}


def _vis_bucket(vis_sm: float | None) -> float:
    """FAA-style flight-category severity from visibility (statute miles)."""
    if vis_sm is None:
        return 2.0          # unknown vis: low-but-present fallback
    if vis_sm < 0.5:
        return 7.0
    if vis_sm < 1:
        return 6.0
    if vis_sm < 3:
        return 5.0
    if vis_sm <= 5:
        return 3.0
    return 1.0


@dataclass(frozen=True)
class WxGroup:
    raw: str                # original group, e.g. "+TSRA"
    family: str             # color family
    intensity: str          # light | moderate | heavy | vicinity
    severity: float         # 0-10 operational-hazard score


def _split(group: str) -> tuple[str, list[str]]:
    """Return (intensity, [2-letter codes]); intensity is '-', '+', 'VC', or ''."""
    g, intensity = group, ""
    if g.startswith(("+", "-")):
        intensity, g = g[0], g[1:]
    elif g.startswith("VC"):
        intensity, g = "VC", g[2:]
    return intensity, [g[i:i + 2] for i in range(0, len(g), 2)]


def _severity(intensity: str, codes: list[str], vis_sm: float | None) -> float:
    if "TS" in codes:
        base = 9.3
    elif "FZ" in codes:
        base = 8.2 if "RA" in codes else 8.1 if "DZ" in codes else 8.0
    else:
        fixed = [_FIXED_SEV[c] for c in codes if c in _FIXED_SEV]
        obsc = [c for c in codes if c in _OBSC_TIEBREAK]
        if fixed:
            base = max(fixed)
        elif obsc:
            base = _vis_bucket(vis_sm) + max(_OBSC_TIEBREAK[c] for c in obsc)
        else:
            base = 0.0
        if fixed:                       # intensity only adjusts fixed precip
            base += 1.0 if intensity == "+" else -1.0 if intensity == "-" else 0.0
    if intensity == "VC":
        base *= 0.5
    return round(base, 2)


def classify(group: str, vis_sm: float | None = None) -> WxGroup:
    """Classify one present-weather group into family + intensity + severity."""
    intensity, codes = _split(group)
    fam = next((_FAMILY[c] for c in codes if c in _FAMILY), "other")
    if "FZ" in codes:
        fam = "freezing"
    if "TS" in codes:
        fam = "thunder"
    return WxGroup(group, fam, _INTENSITY[intensity], _severity(intensity, codes, vis_sm))


def classify_ob(weather: list[str], vis_sm: float | None) -> list[WxGroup]:
    """Classify every present-weather group in one ob (shares the ob's vis)."""
    return [classify(g, vis_sm) for g in weather]
