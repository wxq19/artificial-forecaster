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

from forecaster import charts, store


@dataclass
class ToolResult:
    """What a tool hands back to the loop: a REQUIRED text receipt (a tool reply
    must be text in the OpenAI format) plus any rendered PNGs. Charts reach the
    model as images via tool_messages(), which wraps each PNG in a follow-up user
    message. `images` is a list so one call can return several charts (v2)."""

    text: str
    images: list[bytes] = field(default_factory=list)
    window: tuple | None = None   # (start, end) for time-bounded tools (Fix 3 guard)

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
        td = f"{r['temp_c']}/{r['dewpoint_c']}"
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
    the model never computes timestamps. Returns (start, end) or (None, None)."""
    if args.get("hours") is not None:
        anchor = store.latest(con, station, 1)
        if not anchor:
            return None, None
        end = anchor[0]["obs_time"]
        return end - timedelta(hours=int(args["hours"])), end
    if args.get("start") and args.get("end"):
        # Normalize to NAIVE UTC: fromisoformat('...Z') yields a tz-AWARE datetime,
        # which would compare unequal to get_trend's naive obs_time window and
        # false-trip the guard. The seam owns the naive-UTC contract (see store).
        return _naive_utc(args["start"]), _naive_utc(args["end"])
    return None, None


def _naive_utc(iso: str) -> datetime:
    """Parse an ISO datetime to naive UTC (drop any 'Z'/offset)."""
    dt = datetime.fromisoformat(iso)
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def run_tool(name: str, args: dict, *, db_path: str | None = None) -> ToolResult:
    """Execute a model-issued tool call against a READ-ONLY connection. Returns a
    ToolResult: a text receipt + any rendered PNGs (charts go to the model as
    images via tool_messages, since a tool reply is text-only)."""
    con = (
        store.connect(db_path, read_only=True)
        if db_path
        else store.connect(read_only=True)
    )
    try:
        station = args["station"].upper()
        if name == "query_obs":
            start, end = _resolve_window(con, station, args)
            if start is None:
                return ToolResult(
                    "error: give either hours (relative to the latest ob) or both "
                    "start and end (ISO UTC)"
                )
            rows = store.window(con, station, start, end)
            return ToolResult(
                _window_line(start, end) + "\n" + _fmt(rows, "oldest first"),
                window=(start, end),
            )
        if name == "get_latest_obs":
            rows = store.latest(con, station, args.get("n", 1))
            return ToolResult(_fmt(rows, "newest first"))
        if name == "get_trend":
            hours = min(int(args.get("hours", 24)), 48)  # cap the look-back
            anchor = store.latest(con, station, 1)
            if not anchor:
                return ToolResult(f"(no observations for {station})")
            end = anchor[0]["obs_time"]
            start = end - timedelta(hours=hours)
            rows = store.window(con, station, start, end)
            png = charts.meteogram(rows, station=station, hours=hours)
            receipt = (
                f"{_window_line(start, end)}\n"
                f"Meteogram for {station}, last {hours}h ({len(rows)} obs); image follows."
            )
            return ToolResult(receipt, images=[png], window=(start, end))
        return ToolResult(f"error: unknown tool {name!r}")
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
