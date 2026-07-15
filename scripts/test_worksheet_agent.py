"""Full-agent WORKSHEET test: the agent fills a TAF worksheet, then emits a TAF from it.

The worksheet counterpart to test_taf_agent.py. Same point-in-time obs store and tool
suite, but the model must now SUBMIT a structured pre-forecast worksheet
(submit_taf_worksheet) that passes its completeness check BEFORE (advisory) or as a gate
FOR (required) emit_taf. The agent loop itself lives in forecaster.agent (run_agent) --
this driver exercises the worksheet-specific config it supports:

  - EVIDENCE THREADING (cfg.evidence): every data/read tool call is tagged with a generated
    evidence_id (ev_001, ...), echoed at the top of that tool's receipt so the model can
    cite it, and the id set is passed to submit_taf_worksheet so its evidence_refs RESOLVE.
  - THE MODE GATE (cfg.worksheet_mode): `advisory` (default) validates + surfaces findings
    but never blocks emit_taf; `required` refuses emit_taf until a worksheet has passed;
    `off` skips the worksheet entirely.
  - PERSISTENCE (config.persist_worksheets): the final accepted worksheet + evidence +
    emitted TAF + findings are written to the store.

Like test_taf_agent, the DB-backed tools read a THROWAWAY point-in-time store holding only
obs BEFORE the valid start, so they cannot peek past the forecast start.
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import awc, store, tafgen
from forecaster import worksheet as wksht
from forecaster.agent import AgentConfig, RunResult, run_agent
from forecaster.config import settings
from forecaster.tools import EMIT_TAF, SUBMIT_WORKSHEET, TOOLS

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KLSV")
_ap.add_argument("--valid", default="2026-07-07T23:00", help="TAF valid start (naive UTC)")
_ap.add_argument("--max-steps", type=int, default=14, help="max model turns")
_ap.add_argument("--max-tokens", type=int, default=12000, help="completion budget per turn")
_ap.add_argument("--ingest-hours", type=int, default=48, help="hours of obs to load")
_ap.add_argument("--mode", default=settings.worksheet_mode,
                 choices=["off", "advisory", "required"], help="worksheet_mode for this run")
_ap.add_argument("--model", help="run ONE model by id (default: the small converging set)")
_args = _ap.parse_args()

# Apply the run's mode to the config so the sink (settings.worksheet_mode) agrees with the gate.
settings.worksheet_mode = _args.mode

STATION = _args.station.upper()
VALID = datetime.fromisoformat(_args.valid.rstrip("Z"))
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2
MODE = _args.mode
# Toolset: read/data tools + the worksheet sink + emit_taf. In `off` mode drop the sink.
TOOLSET = TOOLS + ([SUBMIT_WORKSHEET] if MODE != "off" else []) + [EMIT_TAF]

TOOL_CAPS = {"get_map": 8, "get_sounding": 8, "get_fcst_sounding": 8, "get_point_forecast": 8}

MODELS = [("Gemma", "google/gemma-4-31B-it"), ("MiniMax", "MiniMaxAI/MiniMax-M3")]
if _args.model:
    MODELS = [(_args.model.split("/")[-1], _args.model)]

DB_PATH = str(Path(tempfile.mkdtemp(prefix="ws_agent_")) / "obs.duckdb")
load_summary = awc.load_metar(STATION, hours=_args.ingest_hours, db_path=DB_PATH, before=VALID)
con = store.connect(DB_PATH, read_only=True)
try:
    n_obs = store.count(con, STATION)
    latest_obs = store.latest(con, STATION, 1)
finally:
    con.close()
LATEST_OBS_TIME = latest_obs[0]["obs_time"] if latest_obs else None

_GATE_LINE = {
    "off": "Do NOT use a worksheet; reason, then call emit_taf.",
    "advisory": "Fill and submit a worksheet (submit_taf_worksheet) BEFORE emit_taf. Its "
                "findings are advisory -- address them, but you may emit once your reasoning is sound.",
    "required": "You MUST submit a worksheet (submit_taf_worksheet) that passes its completeness "
                "check BEFORE emit_taf is accepted. If emit_taf is refused, fix the worksheet and "
                "re-submit, then emit.",
}[MODE]

SYSTEM = (
    "You are a USAF weather forecaster issuing terminal aerodrome forecasts under AFMAN "
    "15-124. Tools: query_obs/get_latest_obs (stored METARs), get_trend (meteogram), "
    "get_sounding/get_fcst_sounding (skew-Ts), get_map (synoptic charts), get_point_forecast "
    "(hourly model point forecast), get_climo (typical conditions), get_imagery (sat/radar), "
    "get_current_taf (the current official TAF), check_taf (AFMAN dry-run), and emit_taf "
    "(submit the forecast). Each data-tool receipt begins with an [evidence_id: ev_NNN] you can "
    "cite in the worksheet's evidence_refs. " + _GATE_LINE + " Think step by step, gather what "
    "you need, and base the forecast only on tool data. "
    f"You have at most {MAX_STEPS} tool-calling turns."
    + ("\n\n" + wksht.worksheet_guide(settings.evidence_mode) if MODE != "off" else "")
    + "\n\n" + tafgen.emit_taf_guide()
)
TASK = (
    f"Produce a 30-hour Air Force TAF for {STATION} (Nellis AFB, Las Vegas NV), valid from "
    f"{VALID:%d%H%M}Z ({VALID:%Y-%m-%d %H:%MZ}).\n"
    f"NOTE: {STATION} has surface observations but NO model BUFKIT output -- use nearby KLAS "
    "(~10 mi SW) for get_fcst_sounding and get_point_forecast. Begin by checking current and "
    "recent conditions."
)


def run_model(label: str, model: str) -> tuple[str, RunResult]:
    """Drive the full worksheet+TAF loop for one model via the shared agent loop."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    cfg = AgentConfig(
        model=model, toolset=TOOLSET, max_steps=MAX_STEPS, max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE, tool_caps=TOOL_CAPS, worksheet_mode=MODE, db_path=DB_PATH,
    )
    return label, run_agent(messages, cfg)


def _persist(label: str, res: RunResult) -> str | None:
    """Persist the accepted worksheet + evidence + emitted TAF to the store. Returns the
    worksheet_id, or None if nothing to persist / persistence is off."""
    ws = res.worksheet
    if not settings.persist_worksheets or ws is None:
        return None
    taf = res.final_taf or res.last_taf
    taf_text = tafgen.render_taf(taf) if taf is not None else None
    wid = f"{STATION}_{VALID:%Y%m%dT%H%M}_{label}"
    con = store.connect(DB_PATH)
    try:
        store.init_worksheet_schema(con)
        store.insert_worksheet(
            con, worksheet_id=wid, worksheet_json=ws.model_dump_json(), station=STATION,
            forecast_type=(ws.task.forecast_type if ws.task else None),
            valid_from_utc=(ws.task.valid_from_utc if ws.task else None),
            valid_to_utc=(ws.task.valid_to_utc if ws.task else None),
            mode=MODE, evidence_mode=settings.evidence_mode, model=res.model,
            final_taf_text=taf_text,
            taf_product_json=(taf.model_dump_json() if taf is not None else None),
            checker_findings_json=json.dumps(res.worksheet_findings),
            status=("accepted" if not wksht.blocking_findings(res.worksheet_findings) else "advisory"),
            evidence=res.evidence)
    finally:
        con.close()
    return wid


results = [run_model(label, model) for label, model in MODELS]
persisted = {label: _persist(label, res) for label, res in results}


def build_markdown() -> str:
    md = [
        f"# Full-agent worksheet test -- {STATION} valid {VALID:%d%H%M}Z (mode: {MODE})",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **worksheet_mode:** {MODE}  **evidence_mode:** {settings.evidence_mode}",
        f"- **Toolset:** {', '.join(t['function']['name'] for t in TOOLSET)}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}, max_steps={MAX_STEPS}",
        f"- **Obs (pre-cutoff feed):** {n_obs} {STATION} rows"
        + (f", latest {LATEST_OBS_TIME:%Y-%m-%dT%H:%MZ}" if LATEST_OBS_TIME else "")
        + f"; load {load_summary}",
        "",
        "## Summary",
        "",
        "| Model | Steps | Worksheet | TAF outcome | Evidence | Persisted | Tokens (p+c) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, res in results:
        if res.fatal:
            ws_state, taf_state = "-", f"FATAL: {res.fatal[:40]}"
        else:
            blk = wksht.blocking_findings(res.worksheet_findings) if res.worksheet_findings else []
            ws_state = ("clean" if res.worksheet is not None and not res.worksheet_findings
                        else f"{len(blk)} blocking" if res.worksheet is None
                        and res.worksheet_findings else "accepted" if res.worksheet is not None
                        else "none")
            taf_state = ("clean TAF" if res.final_taf is not None
                         else f"TAF w/ {len(tafgen.validate(res.last_taf))} findings"
                         if res.last_taf is not None else "no TAF")
        md.append(f"| {label} | {len(res.steps)} | {ws_state} | {taf_state} | "
                  f"{len(res.evidence)} ids | {persisted.get(label) or '-'} | "
                  f"{res.prompt_tokens}+{res.completion_tokens} |")

    for label, res in results:
        md += ["", "---", "", f"## {label} (`{res.model}`)"]
        if res.fatal:
            md += ["", f"**FATAL:** `{res.fatal}`"]
        for s in res.steps:
            md += ["", f"### Step {s.n} (finish: `{s.finish_reason}`) -- "
                   f"{s.prompt_tokens} ptok / {s.completion_tokens} ctok"]
            if s.reasoning:
                md += ["", "**Reasoning:**", "", "```text", s.reasoning, "```"]
            if s.content:
                md += ["", "**Reply:**", "", "```text", s.content, "```"]
            for c in s.calls:
                img = f" [{c['n_images']} img]" if c.get("n_images") else ""
                md += [f"- `{c['name']}({c['args']})`{img} -> {c['result']}"]
            if s.answer:
                md += ["", "**Final answer:**", "", s.answer]
                if s.recovery:
                    md += ["", f"> harness note: {s.recovery}"]
        if res.worksheet_findings:
            md += ["", "**Final worksheet findings:**"] + [f"- {f}" for f in res.worksheet_findings]
        taf = res.final_taf or res.last_taf
        if taf is not None:
            md += ["", "**Emitted TAF:**", "", "```text"]
            try:
                md += [tafgen.render_taf(taf)]
            except Exception as e:  # noqa: BLE001
                md += [f"(render failed: {e})"]
            md += ["```", f"- AFMAN findings: {tafgen.validate(taf) or 'clean'}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"worksheet_agent_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== Full-agent worksheet test -- {STATION} valid {VALID:%d%H%M}Z (mode {MODE}, {n_obs} obs) ===")
for label, res in results:
    ws_state = ("clean" if res.worksheet is not None and not res.worksheet_findings
                else "accepted" if res.worksheet is not None else "none")
    taf_state = ("clean TAF" if res.final_taf else f"{len(tafgen.validate(res.last_taf))} findings"
                 if res.last_taf else "no TAF")
    print(f"  {label:<8} {len(res.steps)} steps | ws: {ws_state:<8} | {taf_state:<12} | "
          f"ev {len(res.evidence)} | {res.prompt_tokens}p+{res.completion_tokens}c")
print(f"\nFull transcript -> {log_path}")
