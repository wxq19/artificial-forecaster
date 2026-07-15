"""KINS (Creech AFB) full-agent WORKSHEET + TAF test.

A thin adaptation of test_worksheet_agent.py for KINS (Creech AFB, Indian Springs NV),
so we can generate our TAF and compare it to the real Creech TAF issued for the same
cycle. Same incremental-fetch agent loop as the base driver (forecaster.agent.run_agent;
the model gathers tools itself -- NO giant pre-seeded packet, which is what pushed the KOFF
runs past Together's serverless worker limit and gave a ruminating model a huge surface to
spin on).

Run ONE model per process (pass --model) so MiniMax and Kimi can run truly in parallel;
each writes its own model-stamped log. KINS has surface obs + issues a military TAF but
has NO BUFKIT output -- use nearby KLAS (~45 mi SE) for the model point forecast / sounding,
same proxy as KLSV/Nellis.
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import awc, climo, store, tafgen
from forecaster import worksheet as wksht
from forecaster.agent import AgentConfig, RunResult, run_agent
from forecaster.config import settings
from forecaster.tools import EMIT_TAF, SUBMIT_WORKSHEET, TOOLS

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KINS")
_ap.add_argument("--valid", default="2026-07-10T21:00", help="TAF valid start (naive UTC)")
_ap.add_argument("--max-steps", type=int, default=14, help="max model turns")
_ap.add_argument("--max-tokens", type=int, default=16000, help="completion budget per turn")
_ap.add_argument("--ingest-hours", type=int, default=48, help="hours of obs to load")
_ap.add_argument("--mode", default="advisory",
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
# This run deliberately WITHHOLDS get_current_taf: we want to see whether the model builds
# a forecast from data or just parrots the official TAF. Drop it from the read tools.
_READ_TOOLS = [t for t in TOOLS if t["function"]["name"] != "get_current_taf"]
# Toolset: read/data tools + the worksheet sink + emit_taf. In `off` mode drop the sink.
TOOLSET = _READ_TOOLS + ([SUBMIT_WORKSHEET] if MODE != "off" else []) + [EMIT_TAF]

TOOL_CAPS = {"get_map": 8, "get_sounding": 8, "get_fcst_sounding": 8, "get_point_forecast": 8}

MODELS = [("Kimi", "moonshotai/Kimi-K2.7-Code")]
if _args.model:
    MODELS = [(_args.model.split("/")[-1], _args.model)]

DB_PATH = str(Path(tempfile.mkdtemp(prefix="kins_ws_")) / "obs.duckdb")
load_summary = awc.load_metar(STATION, hours=_args.ingest_hours, db_path=DB_PATH, before=VALID)
con = store.connect(DB_PATH, read_only=True)
try:
    n_obs = store.count(con, STATION)
    latest_obs = store.latest(con, STATION, 1)
finally:
    con.close()
LATEST_OBS_TIME = latest_obs[0]["obs_time"] if latest_obs else None

# Build station climatology INTO the same throwaway DB so get_climo works in-harness. The
# multi-year build history is ingested to a scratch DB and thrown away (leakage guard); only
# the climo_* product rows land in DB_PATH -- the point-in-time obs feed is untouched.
try:
    climo_summary = climo.build(STATION, [VALID.month], db_path=DB_PATH)
except Exception as e:  # noqa: BLE001 -- climo is a nice-to-have; don't abort the run
    climo_summary = f"climo build FAILED: {type(e).__name__}: {e}"

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
    "check_taf (AFMAN dry-run), and emit_taf "
    "(submit the forecast). Each data-tool receipt begins with an [evidence_id: ev_NNN] you can "
    "cite in the worksheet's evidence_refs. " + _GATE_LINE + " Think step by step, gather what "
    "you need, and base the forecast only on tool data. "
    f"You have at most {MAX_STEPS} tool-calling turns."
    + ("\n\n" + wksht.worksheet_guide(settings.evidence_mode) if MODE != "off" else "")
    + "\n\n" + tafgen.emit_taf_guide()
)
TASK = (
    f"Produce a 30-hour Air Force TAF for {STATION} (Creech AFB, Indian Springs NV -- high "
    f"desert ~45 mi NW of Las Vegas), valid from {VALID:%d%H%M}Z ({VALID:%Y-%m-%d %H:%MZ}).\n"
    f"NOTE: {STATION} has surface observations but NO model BUFKIT output -- use nearby KLAS "
    "(Las Vegas, ~45 mi SE) for get_fcst_sounding and get_point_forecast. Begin by checking "
    "current and recent conditions."
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
        f"# KINS worksheet+TAF test -- {STATION} valid {VALID:%d%H%M}Z (mode: {MODE})",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **worksheet_mode:** {MODE}  **evidence_mode:** {settings.evidence_mode}",
        f"- **Toolset:** {', '.join(t['function']['name'] for t in TOOLSET)}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}, max_steps={MAX_STEPS}",
        f"- **Obs (pre-cutoff feed):** {n_obs} {STATION} rows"
        + (f", latest {LATEST_OBS_TIME:%Y-%m-%dT%H:%MZ}" if LATEST_OBS_TIME else "")
        + f"; load {load_summary}",
        f"- **Climo build:** {climo_summary}",
        "- **get_current_taf WITHHELD this run** (testing whether the model parrots the official TAF).",
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
                if c.get("full_args"):
                    md += ["", f"  Full `{c['name']}` arguments (what the model submitted):",
                           "", "```json", c["full_args"], "```"]
                if c.get("receipt"):
                    md += ["", "  Full tool receipt (the data the model saw):",
                           "", "```text", c["receipt"], "```"]
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
model_tag = MODELS[0][0].lower() if len(MODELS) == 1 else "both"
log_path = log_dir / f"kins_worksheet_agent_{model_tag}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== KINS worksheet+TAF test -- {STATION} valid {VALID:%d%H%M}Z (mode {MODE}, {n_obs} obs) ===")
for label, res in results:
    ws_state = ("clean" if res.worksheet is not None and not res.worksheet_findings
                else "accepted" if res.worksheet is not None else "none")
    taf_state = ("clean TAF" if res.final_taf else f"{len(tafgen.validate(res.last_taf))} findings"
                 if res.last_taf else "no TAF")
    print(f"  {label:<8} {len(res.steps)} steps | ws: {ws_state:<8} | {taf_state:<12} | "
          f"ev {len(res.evidence)} | {res.prompt_tokens}p+{res.completion_tokens}c")
print(f"\nFull transcript -> {log_path}")
