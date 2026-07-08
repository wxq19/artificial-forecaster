"""Agent-facing tools.

The model can only emit a tool CALL (structured JSON like
{"name": "query_obs", "station": "KORD", ...}); it never sees SQL or a
connection. This module validates the call and runs it against a READ-ONLY
DuckDB connection, so a hallucinated tool call physically cannot write or delete.
Only read tools are registered here — that's the menu the model is limited to.
Results come back as compact text the VLM can reason over.
"""

import base64
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from forecaster import charts, fcstsounding, soundings, store, tafgen, wxmaps
from forecaster.tafgen import TafProduct


@dataclass
class ToolResult:
    """What a tool hands back to the loop: a REQUIRED text receipt (a tool reply
    must be text in the OpenAI format) plus any rendered PNGs. Charts reach the
    model as images via tool_messages(), which wraps each PNG in a follow-up user
    message. `images` is a list so one call can return several charts (v2)."""

    text: str
    images: list[bytes] = field(default_factory=list)
    window: tuple | None = None   # (start, end) for time-bounded tools (Fix 3 guard)
    taf: TafProduct | None = None   # emit_taf hands back the captured forecast object

QUERY_OBS = {
    "type": "function",
    "function": {
        "name": "query_obs",
        "description": (
            "Retrieve surface weather observations (METARs) for an airport, oldest "
            "first. Two ways to set the window: for RECENT/trend data give `hours` "
            "(look-back from the most recent observation) — this anchors on the "
            "latest ob SERVER-SIDE, the same anchor get_trend uses, so windows stay "
            "aligned; do NOT compute dates yourself. For a specific historical range "
            "give absolute `start` and `end` (ISO UTC). Each row gives time, wind, "
            "visibility (statute miles), ceiling (ft AGL), present weather, "
            "temperature/dewpoint (C), and altimeter. Do not invent observations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "hours": {
                    "type": "integer",
                    "description": "Relative look-back in hours from the latest ob "
                    "(use for recent/trend questions; aligns with get_trend)",
                },
                "start": {
                    "type": "string",
                    "description": "Absolute UTC start, ISO 8601 (use with end for a "
                    "specific historical range), e.g. 2024-01-12T00:00",
                },
                "end": {
                    "type": "string",
                    "description": "Absolute UTC end, ISO 8601, e.g. 2024-01-13T00:00",
                },
            },
            "required": ["station"],
        },
    },
}

GET_LATEST = {
    "type": "function",
    "function": {
        "name": "get_latest_obs",
        "description": (
            "Most recent observation(s) for an airport, newest first. Use this "
            "when asked about current conditions or 'right now' and NO explicit "
            "time range is given; use query_obs when a date/time range is given."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "n": {
                    "type": "integer",
                    "description": "How many recent obs to return (default 1)",
                },
            },
            "required": ["station"],
        },
    },
}

GET_TREND = {
    "type": "function",
    "function": {
        "name": "get_trend",
        "description": (
            "Render a meteogram (PNG) of how conditions have CHANGED over the last "
            "N hours at an airport, anchored on the most recent observation. The "
            "image stacks temperature/dewpoint, wind, visibility, ceiling, pressure, "
            "and a colored present-weather band over a shared UTC time axis. Use it "
            "to judge whether conditions are improving, deteriorating, or steady "
            "(e.g. for a persistence forecast). Use query_obs for a specific "
            "date/time range; get_latest_obs for a single current ob."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "hours": {
                    "type": "integer",
                    "description": "Look-back window in hours from the latest ob "
                    "(default 24, max 48)",
                },
            },
            "required": ["station"],
        },
    },
}

GET_SOUNDING = {
    "type": "function",
    "function": {
        "name": "get_sounding",
        "description": (
            "Fetch an observed upper-air skew-T sounding (radiosonde) as an image to "
            "judge vertical structure: stability/CAPE, inversions, moisture layers, "
            "freezing level, and wind shear with height. Soundings are launched only "
            "at 00Z and 12Z from upper-air sites (NOT every airport); you get the most "
            "recent synoptic run at or before now. `site` is an upper-air station id "
            "(e.g. OUN, MPX), which may differ from the nearest airport's ICAO."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": "Upper-air sounding site id, e.g. OUN or MPX",
                },
                "source": {
                    "type": "string",
                    "enum": ["spc", "wyoming"],
                    "description": "Provider: spc (default, richer analysis) or wyoming",
                },
            },
            "required": ["site"],
        },
    },
}

# Menu string generated from the catalog so the tool contract can't drift from wxmaps.
_MAP_MENU = "; ".join(f"{n} ({s.label})" for n, s in wxmaps.CATALOG.items())
GET_MAP = {
    "type": "function",
    "function": {
        "name": "get_map",
        "description": (
            "Fetch a surface or upper-air weather chart as an image for synoptic "
            "situational awareness: fronts and pressure systems, jet stream, steering "
            "flow, moisture, and how the pattern is forecast to evolve. Analysis charts "
            "(surface_*, ocean_*, meso_*) show CURRENT conditions; gfs_* are GFS "
            "FORECAST panels -- for those, pass `fhr`, the forecast hour (a multiple of "
            "6, e.g. 0, 6, 12, 24, 36). Charts: " + _MAP_MENU + "."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart": {
                    "type": "string",
                    "enum": list(wxmaps.CATALOG),
                    "description": "Which chart to fetch (see the list in the description)",
                },
                "fhr": {
                    "type": "integer",
                    "description": "GFS forecast hour, multiple of 6 (0-384); only used "
                    "by the gfs_* forecast charts, ignored otherwise",
                },
            },
            "required": ["chart"],
        },
    },
}

GET_FCST_SOUNDING = {
    "type": "function",
    "function": {
        "name": "get_fcst_sounding",
        "description": (
            "Fetch a MODEL FORECAST sounding (skew-T image) for an airport at a chosen "
            "forecast hour -- the PREDICTED vertical structure (stability/CAPE, inversions, "
            "moisture, wind shear) at a future valid time. Unlike get_sounding, which is an "
            "OBSERVED sounding at 00/12Z, this projects the atmosphere forward. `station` is "
            "a 4-letter ICAO; `model` defaults to gfs (the only model with coverage outside "
            "North America); `fhr` is the forecast hour (0 = analysis; hourly early, then "
            "3-hourly). Coverage is dense over North America and sparse OCONUS -- an "
            "unavailable station is reported back so you can pick another."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(fcstsounding.MODELS),
                          "description": "forecast model (default gfs)"},
                "fhr": {"type": "integer",
                        "description": "forecast hour (0=analysis; e.g. 6, 12, 24, 36)"},
            },
            "required": ["station"],
        },
    },
}

GET_POINT_FORECAST = {
    "type": "function",
    "function": {
        "name": "get_point_forecast",
        "description": (
            "Hourly MODEL point forecast TABLE for an airport: surface conditions over time "
            "-- temperature, dewpoint, wind, MSL pressure, low/mid/high cloud, and hourly "
            "precipitation at each forecast hour, from the model's BUFKIT output. Use it to "
            "see how conditions EVOLVE hour by hour at a site (complements get_fcst_sounding, "
            "which is the vertical profile at one hour). Each row is one forecast hour; read a "
            "column downward for a variable's trend. `station` 4-letter ICAO; `model` defaults "
            "to gfs (only gfs has OCONUS coverage); `hours` limits the horizon (default 48). "
            "Values are raw model surface fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(fcstsounding.MODELS),
                          "description": "forecast model (default gfs)"},
                "hours": {"type": "integer",
                          "description": "forecast hours to include from the run (default 48)"},
            },
            "required": ["station"],
        },
    },
}

TOOLS = [QUERY_OBS, GET_LATEST, GET_TREND, GET_SOUNDING, GET_MAP, GET_FCST_SOUNDING,
         GET_POINT_FORECAST]

# The OUTPUT tool: the model emits its forecast as the fields of a TafProduct, and
# our code renders + checks it. The parameter schema IS the pydantic model's JSON
# schema, so the one class is both the tool contract and the validator. Unlike the
# read tools, emit_taf is a SINK -- its result is the AFMAN check, not data to
# reason over -- so the loop can feed validate() findings back for a re-emit.
EMIT_TAF = {
    "type": "function",
    "function": {
        "name": "emit_taf",
        "description": (
            "Emit a complete Air Force terminal aerodrome forecast (TAF) as "
            "structured fields. Fill the prevailing period and any FM/BECMG/TEMPO "
            "change groups. Rules: a routine TAF is valid 30 hours; visibility is in "
            "METERS (9999 = unrestricted, >=7SM); wind direction is degrees to the "
            "nearest 10 as an INTEGER (or 'VRB'); QNH is the altimeter in inches of "
            "mercury (e.g. 29.92); include CB cloud type whenever a thunderstorm (TS) "
            "is forecast; do not put QNH in a TEMPO group. Base the forecast only on "
            "the observations and trend provided."
        ),
        "parameters": TafProduct.model_json_schema(),
    },
}


def _fmt(rows: list[dict], order: str = "oldest first") -> str:
    """Per ob: a decoded summary line (our normalized vis_sm/ceiling_ft) followed
    by the RAW METAR beneath it, so nothing the decoder skips — RMK, RVR, exact
    pressure, peak wind — is lost to the model. The raw line is the ground truth;
    the decoded line is a scannable convenience. `order` only labels the header to
    match how the caller sorted the rows (range reads run oldest-first; a 'latest'
    read stays newest-first — the sort order carries intent, so we don't flatten it)."""
    if not rows:
        return "(no observations in range)"
    out = [
        f"{len(rows)} observations (UTC, {order}). Each ob: decoded summary, "
        "then the raw METAR/SPECI beneath. A SPECI means weather forced an "
        "off-cycle report — treat it as a significance signal.",
        "decoded cols: UTC time (ISO) | type | wind | vis | ceiling | present-wx | T/Td(C)",
    ]
    for r in rows:
        wind = "—"
        if r["wind_speed"] is not None:
            d = (
                f"{r['wind_dir_deg']:03d}"
                if r["wind_dir_deg"] is not None
                else (r["wind_dir_card"] or "VRB")
            )
            g = f"G{r['wind_gust']}" if r["wind_gust"] else ""
            wind = f"{d}/{r['wind_speed']}{g}"
        vis = "—" if r["vis_sm"] is None else f"{(r['vis_flag'] or '')}{r['vis_sm']:g}SM"
        ceil = "unlim" if r["ceiling_ft"] is None else f"{r['ceiling_ft']}ft"
        wx = " ".join(r["weather"]) or "-"
        t = "—" if r["temp_c"] is None else f"{r['temp_c']}"
        dp = "—" if r["dewpoint_c"] is None else f"{r['dewpoint_c']}"
        td = f"{t}/{dp}"
        kind = r["report_type"] or "—"
        out.append(
            f"  {r['obs_time']:%Y-%m-%dT%H:%MZ} {kind:<5} {wind:<11} {vis:<7} {ceil:<7} {wx:<14} {td}"
        )
        out.append(f"    {r['raw']}")
    return "\n".join(out)


def _window_line(start, end) -> str:
    """Canonical window echo so every time-bounded result states its exact span in
    one comparable line (Fix 2)."""
    return f"window: {start:%Y-%m-%dT%H:%MZ} .. {end:%Y-%m-%dT%H:%MZ}"


def _resolve_window(con, station, args):
    """Resolve a query window. Relative mode (preferred for recent/trend): `hours`
    anchors on the latest ob — IDENTICAL to get_trend, so windows align across
    tools. Absolute mode: explicit ISO start+end. The seam owns the arithmetic;
    the model never computes timestamps. Returns (start, end, None) on success, or
    (None, None, reason) — the reason DISTINGUISHES 'no obs for this station' from
    'no window arguments given' so the model gets accurate feedback (#9)."""
    if args.get("hours") is not None:
        anchor = store.latest(con, station, 1)
        if not anchor:
            return None, None, f"no observations stored for {station} to anchor a relative window"
        end = anchor[0]["obs_time"]
        return end - timedelta(hours=_int_arg(args["hours"], 24, lo=1)), end, None
    if args.get("start") and args.get("end"):
        # Normalize to NAIVE UTC: fromisoformat('...Z') yields a tz-AWARE datetime,
        # which would compare unequal to get_trend's naive obs_time window and
        # false-trip the guard. The seam owns the naive-UTC contract (see store).
        return _naive_utc(args["start"]), _naive_utc(args["end"]), None
    return None, None, ("give either hours (relative to the latest ob) or both start "
                        "and end (ISO UTC)")


def _naive_utc(iso: str) -> datetime:
    """Parse an ISO datetime to naive UTC (drop any 'Z'/offset)."""
    dt = datetime.fromisoformat(iso)
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _int_arg(v, default: int, *, lo: int, hi: int | None = None) -> int:
    """Coerce a model-supplied count/duration to an int and clamp. Models emit numbers
    as strings; a value that will not parse raises ValueError, which run_tool turns into
    tool feedback rather than a dead loop."""
    n = default if v is None else int(v)
    n = max(lo, n)
    return n if hi is None else min(n, hi)


def _emit_taf(args: dict) -> ToolResult:
    """Capture the model's structured forecast: build a TafProduct (guardrails fire
    here), render it, and run the AFMAN rule check + round-trip. The receipt is that
    check, phrased so the model can re-emit a fix; the built product rides back on
    ToolResult.taf. A schema/guardrail failure is reported as text, not raised, so a
    malformed call becomes correctable feedback rather than a crashed loop."""
    try:
        product = TafProduct(**args)
    except ValidationError as e:
        errs = "\n".join(f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                         for err in e.errors())
        return ToolResult(f"emit_taf rejected ({e.error_count()} schema error(s)); fix and "
                          f"re-emit:\n{errs}")
    findings = tafgen.validate(product)
    try:
        text = tafgen.render_taf(product)
    except Exception as e:  # noqa: BLE001 -- a group missing required timing; report, don't crash
        return ToolResult(
            f"emit_taf built but could not render ({type(e).__name__}: {e}); a change group "
            "is likely missing its day/hour fields. Fix and re-emit.", taf=product)
    lines = ["TAF emitted:", "", text, ""]
    if findings:
        lines.append(f"AFMAN check found {len(findings)} issue(s) -- correct them and re-emit:")
        lines += [f"  - {f}" for f in findings]
        return ToolResult("\n".join(lines), taf=product)   # skip round-trip on a known-bad TAF
    try:
        diffs = tafgen.roundtrip(product)
    except Exception as e:  # noqa: BLE001 -- a group that renders but won't re-parse is feedback, not a crash
        return ToolResult(
            f"emit_taf passed the AFMAN check but its render could not be parsed back "
            f"({type(e).__name__}: {e}); a change group is likely missing timing fields. "
            "Fix and re-emit.", taf=product)
    lines.append("AFMAN check: clean.")
    if diffs:
        lines.append("round-trip differences: " + "; ".join(diffs))
    return ToolResult("\n".join(lines), taf=product)


def _get_sounding(args: dict) -> ToolResult:
    """Fetch an observed skew-T image from a public provider (network, no DB) and
    hand it back for the model to read. Site ids live in the provider's namespace,
    so a bad id/date surfaces as a fetch error the model can correct -- not a crash.
    The receipt cites the exact synoptic time + source URL (provenance)."""
    site = args.get("site")
    if not site:
        return ToolResult('error: get_sounding needs a "site" upper-air id, e.g. "site": "OUN"')
    source = str(args.get("source") or "spc").lower()
    if source not in ("spc", "wyoming"):
        return ToolResult(f'error: unknown source {source!r}; use "spc" or "wyoming"')
    try:
        t = soundings.synoptic_time()
        url = soundings.skewt_url(site, t, source=source)
        img = soundings.fetch_skewt(site, t, source=source)
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        return ToolResult(
            f"error: could not fetch {source} sounding for {str(site).upper()} "
            f"({type(e).__name__}: {e}); check the site id for this provider"
        )
    receipt = (
        f"Observed skew-T for {str(site).upper()} at {t:%Y-%m-%dT%H:%MZ} "
        f"(source: {source}, {url}); image follows."
    )
    return ToolResult(receipt, images=[img])


def _get_map(args: dict) -> ToolResult:
    """Fetch a catalogued surface/upper-air chart image (network, no DB). A forecast
    chart gets its GFS forecast hour snapped to the 6h grid; an unknown chart name or a
    fetch failure comes back as feedback, not a crash. Receipt cites the source URL."""
    name = args.get("chart")
    if not name or name not in wxmaps.CATALOG:
        return ToolResult(
            'error: get_map needs a valid "chart"; choose from: ' + ", ".join(wxmaps.CATALOG)
        )
    spec = wxmaps.CATALOG[name]
    fhr = 0
    if spec.source == "tt":
        fhr = _int_arg(args.get("fhr"), 0, lo=0, hi=wxmaps.GFS_MAX_FHR)
        fhr -= fhr % wxmaps.GFS_STEP_H          # snap down to the 6h GFS grid
    try:
        url = wxmaps.map_url(name, fhr=fhr)
        img = wxmaps.fetch_map(name, fhr=fhr)
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not fetch chart {name} ({type(e).__name__}: {e})")
    lead = f", GFS f{fhr:03d}" if spec.source == "tt" else ""
    return ToolResult(
        f"{spec.label} [{name}]{lead} (source: {spec.source}, {url}); image follows.",
        images=[img],
    )


def _get_fcst_sounding(args: dict) -> ToolResult:
    """Fetch + render a model forecast sounding (network, no DB). A missing station or
    forecast hour comes back as feedback -- fcstsounding raises ValueError with the reason
    (404 / available hours) rather than crashing the loop. Receipt cites the source URL."""
    station = args.get("station")
    if not station:
        return ToolResult('error: get_fcst_sounding needs a "station" ICAO, e.g. "station": "KMSP"')
    model = str(args.get("model") or "gfs").lower()
    if model not in fcstsounding.MODELS:
        return ToolResult(f"error: unknown model {model!r}; choose from {', '.join(fcstsounding.MODELS)}")
    fhr = _int_arg(args.get("fhr"), 12, lo=0, hi=384)
    try:
        prof = fcstsounding.fetch_profile(station, model=model, fhr=fhr)
        png = charts.skewt(prof)
    except Exception as e:  # noqa: BLE001 -- fetch/parse failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not build forecast sounding for {str(station).upper()} "
                          f"{model} f{fhr:03d} ({type(e).__name__}: {e})")
    receipt = (f"{model.upper()} forecast skew-T for {prof.station}, f{fhr:03d} valid "
               f"{prof.valid} (run {prof.run:%Y-%m-%dT%H:%MZ}, {prof.url}); image follows.")
    return ToolResult(receipt, images=[png])


def _uv_to_dirspd(u: float, v: float) -> tuple[int, int]:
    """Wind (u, v in m/s) -> (direction deg to nearest 10, speed kt). A presentation of the
    raw vector; the stored point-forecast data keeps the u/v components."""
    spd = round(math.hypot(u, v) * 1.94384)
    d = int(round((270.0 - math.degrees(math.atan2(v, u))) % 360.0 / 10.0) * 10) % 360
    return d, spd


def _fmt_point(pf, n: int) -> str:
    """Format a PointForecast as a text table: one row per forecast hour, columns are the
    raw surface variables (wind shown as dir/speed). Read a column down for a trend."""
    rows = pf.rows[:n]
    out = [
        f"{pf.model.upper()} point forecast for {pf.station} -- run {pf.run:%Y-%m-%dT%H:%MZ}, "
        f"{len(rows)} hourly steps (source: {pf.url}). Raw model surface fields; each row is "
        "one forecast hour -- read a column down to see a variable's trend.",
        (f"{'Valid (UTC)':<18}{'T C':>5}{'Td C':>6}{'Wind kt':>10}{'MSLP':>7}"
         f"{'Cld L/M/H %':>14}{'P01 mm':>8}"),
    ]
    for r in rows:
        wd, ws = _uv_to_dirspd(r["uwnd_ms"], r["vwnd_ms"])
        vt = f"{r['valid']:%Y-%m-%dT%H:%MZ}"
        cloud = f"{r['lcld']:.0f}/{r['mcld']:.0f}/{r['hcld']:.0f}"
        out.append(
            f"{vt:<18}{r['t2m_c']:>5.0f}{r['td2m_c']:>6.0f}{f'{wd:03d}/{ws}':>10}"
            f"{r['mslp_hpa']:>7.0f}{cloud:>14}{r['p01_mm']:>8.1f}"
        )
    return "\n".join(out)


def _get_point_forecast(args: dict) -> ToolResult:
    """Fetch + format a model point forecast table (network, no DB). A missing station (404)
    comes back as feedback via fcstsounding's ValueError, not a crash."""
    station = args.get("station")
    if not station:
        return ToolResult('error: get_point_forecast needs a "station" ICAO, e.g. "station": "KMSP"')
    model = str(args.get("model") or "gfs").lower()
    if model not in fcstsounding.MODELS:
        return ToolResult(f"error: unknown model {model!r}; choose from {', '.join(fcstsounding.MODELS)}")
    hours = _int_arg(args.get("hours"), 48, lo=1, hi=384)
    try:
        pf = fcstsounding.fetch_point(station, model=model)
    except Exception as e:  # noqa: BLE001 -- fetch/parse failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not fetch point forecast for {str(station).upper()} "
                          f"{model} ({type(e).__name__}: {e})")
    return ToolResult(_fmt_point(pf, hours))


def run_tool(name: str, args: dict, *, db_path: str | None = None) -> ToolResult:
    """Execute a model-issued tool call. The read tools run against a READ-ONLY
    connection; emit_taf (output sink) and the network fetches (get_sounding, get_map,
    get_fcst_sounding, get_point_forecast) need no DB and are handled first. Returns a
    ToolResult: text receipt + images/TAF."""
    if name == "emit_taf":
        return _emit_taf(args)
    if name == "get_sounding":
        return _get_sounding(args)
    if name == "get_map":
        return _get_map(args)
    if name == "get_fcst_sounding":
        return _get_fcst_sounding(args)
    if name == "get_point_forecast":
        return _get_point_forecast(args)
    con = (
        store.connect(db_path, read_only=True)
        if db_path
        else store.connect(read_only=True)
    )
    try:
        station = args.get("station")
        if not station:
            return ToolResult('error: this tool needs a "station" ICAO id, e.g. "station": "KBLV"')
        station = str(station).upper()
        if name == "query_obs":
            start, end, err = _resolve_window(con, station, args)
            if err:
                return ToolResult(f"error: {err}")
            rows = store.window(con, station, start, end)
            return ToolResult(
                _window_line(start, end) + "\n" + _fmt(rows, "oldest first"),
                window=(start, end),
            )
        if name == "get_latest_obs":
            n = _int_arg(args.get("n"), 1, lo=1, hi=200)
            rows = store.latest(con, station, n)
            return ToolResult(_fmt(rows, "newest first"))
        if name == "get_trend":
            hours = _int_arg(args.get("hours"), 24, lo=1, hi=48)  # coerce + clamp the look-back
            anchor = store.latest(con, station, 1)
            if not anchor:
                return ToolResult(f"(no observations for {station})")
            end = anchor[0]["obs_time"]
            start = end - timedelta(hours=hours)
            rows = store.window(con, station, start, end)
            if not rows:
                return ToolResult(
                    f"{_window_line(start, end)}\n"
                    f"(no observations for {station} in the last {hours}h)"
                )
            png = charts.meteogram(rows, station=station, hours=hours)
            receipt = (
                f"{_window_line(start, end)}\n"
                f"Meteogram for {station}, last {hours}h ({len(rows)} obs); image follows."
            )
            return ToolResult(receipt, images=[png], window=(start, end))
        return ToolResult(f"error: unknown tool {name!r}")
    except Exception as e:  # noqa: BLE001 -- any read-tool failure becomes feedback, not a dead loop
        return ToolResult(f"error: {name} failed ({type(e).__name__}: {e})")
    finally:
        con.close()


# NOTE: tool_messages and final_answer are agent-loop plumbing, not tool
# definitions. They live here for now only because every driver imports `tools`;
# move them to a dedicated `agent.py` once that module exists (see CLAUDE.md).
def _image_mime(data: bytes) -> str:
    """Content type from magic bytes. A meteogram is PNG, but a fetched skew-T can be
    a GIF (SPC) or PNG (Wyoming), and a vision model rejects an image whose data URL
    lies about its type -- so label each image by what it actually is."""
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def tool_messages(call_id: str, result: ToolResult) -> list[dict]:
    """Turn a ToolResult into the messages to append after a tool call: the
    required text receipt (role 'tool'), plus — if the tool returned images — a
    follow-up 'user' message carrying each image as a base64 image_url, since a tool
    reply can't hold an image in the OpenAI format. Returns 1 or 2 messages."""
    msgs: list[dict] = [
        {"role": "tool", "tool_call_id": call_id, "content": result.text}
    ]
    if result.images:
        content: list[dict] = [
            {"type": "text", "text": "Image(s) from the tool call:"}
        ]
        for img in result.images:
            b64 = base64.b64encode(img).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_image_mime(img)};base64,{b64}"},
            })
        msgs.append({"role": "user", "content": content})
    return msgs


def final_answer(msg, finish_reason: str | None) -> tuple[str, str | None]:
    """Pull the model's answer out of a completed (no-tool-call) message.

    A reasoning model can leave `content` EMPTY while spilling the whole answer
    into `reasoning`, and still stop cleanly (finish_reason 'stop', NOT 'length').
    Reading content alone then logs a CORRECT answer as blank — a silent scoring
    bug. Guard it: on empty content + clean stop, fall back to reasoning and
    return a flag so the caller can mark the run instead of recording a miss.
    Returns (answer_text, flag); flag is None when content was present as normal.
    """
    content = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning", None) or "").strip()
    if content:
        return content, None
    if reasoning and finish_reason == "stop":
        return reasoning, "recovered from reasoning field (content empty, clean stop)"
    if finish_reason == "length":
        return (
            "_(empty — ran out of tokens; raise MAX_TOKENS)_",
            "content empty: finish_reason=length",
        )
    return "_(empty — no content and no reasoning)_", "content empty: no reasoning either"


def window_conflict(windows: list) -> str | None:
    """windows: list of (tool_label, (start, end)) gathered across the WHOLE
    conversation. If more than one DISTINCT window is present, return an advisory
    note listing each — non-blocking; the model decides if it's intentional. Pure;
    the caller dedupes before injecting. (Loop plumbing -> agent.py later.)"""
    distinct: dict = {}
    for label, win in windows:
        distinct.setdefault(win, []).append(label)
    if len(distinct) <= 1:
        return None
    lines = [
        "Heads up: your tool calls are not all looking at the same time period. "
        "If that is intentional, carry on; otherwise re-query so the windows align:"
    ]
    for (start, end), labels in distinct.items():
        lines.append(
            f"  {', '.join(labels)}: {start:%Y-%m-%dT%H:%MZ} .. {end:%Y-%m-%dT%H:%MZ}"
        )
    return "\n".join(lines)
