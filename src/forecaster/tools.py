"""Agent-facing tools.

The model can only emit a tool CALL (structured JSON like
{"name": "query_obs", "station": "KORD", ...}); it never sees SQL or a
connection. This module validates the call and runs it against a READ-ONLY
DuckDB connection, so a hallucinated tool call physically cannot write or delete.
Only read tools are registered here — that's the menu the model is limited to.
Results come back as compact text the VLM can reason over.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from forecaster import charts, store, tafgen
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

TOOLS = [QUERY_OBS, GET_LATEST, GET_TREND]

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


def run_tool(name: str, args: dict, *, db_path: str | None = None) -> ToolResult:
    """Execute a model-issued tool call. The read tools run against a READ-ONLY
    connection; emit_taf is an output sink (no DB) handled first. Returns a
    ToolResult: a text receipt + any rendered PNGs / captured TAF."""
    if name == "emit_taf":
        return _emit_taf(args)
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
def tool_messages(call_id: str, result: ToolResult) -> list[dict]:
    """Turn a ToolResult into the messages to append after a tool call: the
    required text receipt (role 'tool'), plus — if the tool rendered images — a
    follow-up 'user' message carrying each PNG as a base64 image_url, since a tool
    reply can't hold an image in the OpenAI format. Returns 1 or 2 messages."""
    msgs: list[dict] = [
        {"role": "tool", "tool_call_id": call_id, "content": result.text}
    ]
    if result.images:
        content: list[dict] = [
            {"type": "text", "text": "Rendered chart(s) from get_trend:"}
        ]
        for png in result.images:
            b64 = base64.b64encode(png).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
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
