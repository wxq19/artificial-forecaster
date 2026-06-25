"""Agent-facing tools.

The model can only emit a tool CALL (structured JSON like
{"name": "query_obs", "station": "KORD", ...}); it never sees SQL or a
connection. This module validates the call and runs it against a READ-ONLY
DuckDB connection, so a hallucinated tool call physically cannot write or delete.
Only read tools are registered here — that's the menu the model is limited to.
Results come back as compact text the VLM can reason over.
"""

from datetime import datetime

from forecaster import store

QUERY_OBS = {
    "type": "function",
    "function": {
        "name": "query_obs",
        "description": (
            "Retrieve surface weather observations (METARs) for an airport over a "
            "UTC time range, oldest first. Each row gives time, wind, visibility in "
            "statute miles, ceiling in feet AGL, present weather, temperature and "
            "dewpoint in C, and altimeter. Call this to inspect actual conditions "
            "before answering; do not invent observations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "start": {
                    "type": "string",
                    "description": "UTC range start, ISO 8601, e.g. 2024-01-12T00:00",
                },
                "end": {
                    "type": "string",
                    "description": "UTC range end, ISO 8601, e.g. 2024-01-13T00:00",
                },
            },
            "required": ["station", "start", "end"],
        },
    },
}

TOOLS = [QUERY_OBS]


def _fmt(rows: list[dict]) -> str:
    """Per ob: a decoded summary line (our normalized vis_sm/ceiling_ft) followed
    by the RAW METAR beneath it, so nothing the decoder skips — RMK, RVR, exact
    pressure, peak wind — is lost to the model. The raw line is the ground truth;
    the decoded line is a scannable convenience."""
    if not rows:
        return "(no observations in range)"
    out = [
        f"{len(rows)} observations (UTC, oldest first). Each ob: decoded summary, "
        "then the raw METAR/SPECI beneath. A SPECI means weather forced an "
        "off-cycle report — treat it as a significance signal.",
        "decoded cols: DDHHMMz | type | wind | vis | ceiling | present-wx | T/Td(C)",
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
            f"  {r['obs_time']:%d%H%MZ} {kind:<5} {wind:<11} {vis:<7} {ceil:<7} {wx:<14} {td}"
        )
        out.append(f"    {r['raw']}")
    return "\n".join(out)


def run_tool(name: str, args: dict, *, db_path: str | None = None) -> str:
    """Execute a model-issued tool call against a READ-ONLY connection."""
    if name != "query_obs":
        return f"error: unknown tool {name!r}"
    con = (
        store.connect(db_path, read_only=True)
        if db_path
        else store.connect(read_only=True)
    )
    try:
        rows = store.window(
            con,
            args["station"].upper(),
            datetime.fromisoformat(args["start"]),
            datetime.fromisoformat(args["end"]),
        )
    finally:
        con.close()
    return _fmt(rows)
