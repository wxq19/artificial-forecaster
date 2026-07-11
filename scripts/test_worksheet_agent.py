"""Full-agent WORKSHEET test: the agent fills a TAF worksheet, then emits a TAF from it.

The worksheet counterpart to test_taf_agent.py. Same point-in-time obs store and tool
suite, but the model must now SUBMIT a structured pre-forecast worksheet
(submit_taf_worksheet) that passes its completeness check BEFORE (advisory) or as a gate
FOR (required) emit_taf. This is the Milestone-1 "bring it together" driver: it exercises
the agent-loop plumbing the design assigns to the driver (migrating to a future agent.py):

  - EVIDENCE THREADING: every data/read tool call is tagged with a generated evidence_id
    (ev_001, ...), the id is echoed at the top of that tool's receipt so the model can
    cite it, and the id set is passed to submit_taf_worksheet so its evidence_refs RESOLVE
    (not just presence-check).
  - THE MODE GATE (config.worksheet_mode): `advisory` (default) validates + surfaces
    findings but never blocks emit_taf; `required` refuses emit_taf until a worksheet has
    passed (no blocking findings); `off` skips the worksheet entirely.
  - PERSISTENCE: the final accepted worksheet + evidence + emitted TAF + findings are
    written to the store (config.persist_worksheets).

Like test_taf_agent, the DB-backed tools read a THROWAWAY point-in-time store holding only
obs BEFORE the valid start, so they cannot peek past the forecast start.
"""

import argparse
import base64
import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from forecaster import awc, store, tafgen, tools
from forecaster import worksheet as wksht
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import EMIT_TAF, SUBMIT_WORKSHEET, TOOLS, final_answer, run_tool

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

# The sinks + pure validators do NOT produce evidence (they are outputs/dry-runs, not data).
_NON_EVIDENCE = {"emit_taf", "submit_taf_worksheet", "check_taf"}

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


def run_model(label: str, model: str) -> dict:
    """Drive the full worksheet+TAF loop for one model. Returns a record for the log."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    steps: list[dict] = []
    used: Counter = Counter()
    evidence: list[dict] = []          # threaded evidence rows (id, tool, args, receipt)
    ev_ids: list[str] = []
    final_taf = last_taf = None
    accepted_ws = None                 # last worksheet with no blocking findings
    ws_findings: list[str] = []
    worksheet_ok = False
    ptok = ctok = 0
    for n in range(1, MAX_STEPS + 1):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLSET, tool_choice="auto",
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
        except Exception as e:  # noqa: BLE001 -- a model that rejects the toolset is a finding
            steps.append({"n": n, "error": f"{type(e).__name__}: {e}"})
            return _record(label, model, steps, used, final_taf, last_taf, accepted_ws,
                           ws_findings, evidence, ptok, ctok, fatal=f"{type(e).__name__}: {e}")
        ptok += r.usage.prompt_tokens
        ctok += r.usage.completion_tokens
        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        rec = {"n": n, "finish": r.choices[0].finish_reason,
               "ptok": r.usage.prompt_tokens, "ctok": r.usage.completion_tokens,
               "content": (msg.content or "").strip(),
               "reasoning": (getattr(msg, "reasoning", None) or "").strip(), "calls": []}

        if not tcs:
            answer, recovery = final_answer(msg, r.choices[0].finish_reason)
            rec["answer"], rec["recovery"] = answer, recovery
            steps.append(rec)
            break

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs]})
        images: list[tuple[str, bytes]] = []
        for tc in tcs:
            name = tc.function.name
            used[name] += 1
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": f"error: unparseable arguments: {e}"})
                rec["calls"].append({"name": name, "args": tc.function.arguments[:120],
                                     "result": f"unparseable args: {e}"})
                continue
            cap = TOOL_CAPS.get(name)
            if cap is not None and used[name] > cap:
                capped = (f"cap reached: {name} may be called at most {cap} times per run; "
                          "you have enough data -- move on.")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": capped})
                rec["calls"].append({"name": name, "args": "(capped)", "result": capped[:160]})
                continue

            # THE GATE: in required mode, refuse emit_taf until a worksheet has passed.
            if name == "emit_taf" and MODE == "required" and not worksheet_ok:
                refuse = ("emit_taf refused: worksheet_mode=required. Submit a "
                          "submit_taf_worksheet that passes its completeness check first, "
                          "then emit.")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": refuse})
                rec["calls"].append({"name": name, "args": "(gated)", "result": refuse[:160]})
                continue

            res = run_tool(name, args, db_path=DB_PATH, evidence_ids=ev_ids or None)

            # EVIDENCE THREADING: tag a data-tool receipt with a fresh id the model can cite.
            receipt = res.text
            if name not in _NON_EVIDENCE and not receipt.startswith("error:"):
                ev_id = f"ev_{len(evidence) + 1:03d}"
                ev_ids.append(ev_id)
                evidence.append({"evidence_id": ev_id, "tool_name": name,
                                 "tool_args_json": json.dumps(args),
                                 "receipt_text": receipt.splitlines()[0][:200]})
                receipt = f"[evidence_id: {ev_id}]\n{receipt}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": receipt})

            label_line = res.text.splitlines()[0] if res.text else name
            images += [(label_line, im) for im in res.images]
            if name == "submit_taf_worksheet" and res.worksheet is not None:
                ws_findings = res.findings
                if not wksht.blocking_findings(res.findings):
                    accepted_ws, worksheet_ok = res.worksheet, True
            if name == "emit_taf" and res.taf is not None:
                last_taf = res.taf
                if not tafgen.validate(res.taf):
                    final_taf = res.taf
            rec["calls"].append({"name": name, "args": json.dumps(args)[:160],
                                 "result": res.text.splitlines()[0][:160],
                                 "n_images": len(res.images)})
        if images:
            content = [{"type": "text", "text": "Images from the tool calls above, each "
                        "preceded by its tool's receipt line:"}]
            for label_line, im in images:
                content.append({"type": "text", "text": f"[image for: {label_line}]"})
                b64 = base64.b64encode(im).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{tools._image_mime(im)};base64,{b64}"}})
            messages.append({"role": "user", "content": content})
        steps.append(rec)
        if final_taf is not None:
            break
    return _record(label, model, steps, used, final_taf, last_taf, accepted_ws,
                   ws_findings, evidence, ptok, ctok)


def _record(label, model, steps, used, final_taf, last_taf, accepted_ws, ws_findings,
            evidence, ptok, ctok, fatal=None) -> dict:
    return {"label": label, "model": model, "steps": steps, "used": used,
            "final_taf": final_taf, "last_taf": last_taf, "accepted_ws": accepted_ws,
            "ws_findings": ws_findings, "evidence": evidence, "ptok": ptok, "ctok": ctok,
            "fatal": fatal}


def _persist(res: dict) -> str | None:
    """Persist the accepted worksheet + evidence + emitted TAF to the store. Returns the
    worksheet_id, or None if nothing to persist / persistence is off."""
    ws = res["accepted_ws"]
    if not settings.persist_worksheets or ws is None:
        return None
    taf = res["final_taf"] or res["last_taf"]
    taf_text = tafgen.render_taf(taf) if taf is not None else None
    wid = f"{STATION}_{VALID:%Y%m%dT%H%M}_{res['label']}"
    con = store.connect(DB_PATH)
    try:
        store.init_worksheet_schema(con)
        store.insert_worksheet(
            con, worksheet_id=wid, worksheet_json=ws.model_dump_json(), station=STATION,
            forecast_type=(ws.task.forecast_type if ws.task else None),
            valid_from_utc=(ws.task.valid_from_utc if ws.task else None),
            valid_to_utc=(ws.task.valid_to_utc if ws.task else None),
            mode=MODE, evidence_mode=settings.evidence_mode, model=res["model"],
            final_taf_text=taf_text,
            taf_product_json=(taf.model_dump_json() if taf is not None else None),
            checker_findings_json=json.dumps(res["ws_findings"]),
            status=("accepted" if not wksht.blocking_findings(res["ws_findings"]) else "advisory"),
            evidence=res["evidence"])
    finally:
        con.close()
    return wid


results = [run_model(label, model) for label, model in MODELS]
persisted = {res["label"]: _persist(res) for res in results}


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
    for res in results:
        if res["fatal"]:
            ws_state, taf_state = "-", f"FATAL: {res['fatal'][:40]}"
        else:
            blk = wksht.blocking_findings(res["ws_findings"]) if res["ws_findings"] else []
            ws_state = ("clean" if res["accepted_ws"] is not None and not res["ws_findings"]
                        else f"{len(blk)} blocking" if res["accepted_ws"] is None
                        and res["ws_findings"] else "accepted" if res["accepted_ws"] is not None
                        else "none")
            taf_state = ("clean TAF" if res["final_taf"] is not None
                         else f"TAF w/ {len(tafgen.validate(res['last_taf']))} findings"
                         if res["last_taf"] is not None else "no TAF")
        md.append(f"| {res['label']} | {len(res['steps'])} | {ws_state} | {taf_state} | "
                  f"{len(res['evidence'])} ids | {persisted.get(res['label']) or '-'} | "
                  f"{res['ptok']}+{res['ctok']} |")

    for res in results:
        md += ["", "---", "", f"## {res['label']} (`{res['model']}`)"]
        if res["fatal"]:
            md += ["", f"**FATAL:** `{res['fatal']}`"]
        for s in res["steps"]:
            tok = f" -- {s['ptok']} ptok / {s['ctok']} ctok" if "ptok" in s else ""
            md += ["", f"### Step {s['n']} (finish: `{s.get('finish', '?')}`){tok}"]
            if s.get("error"):
                md += [f"- error: `{s['error']}`"]
                continue
            if s["reasoning"]:
                md += ["", "**Reasoning:**", "", "```text", s["reasoning"], "```"]
            if s["content"]:
                md += ["", "**Reply:**", "", "```text", s["content"], "```"]
            for c in s["calls"]:
                img = f" [{c['n_images']} img]" if c.get("n_images") else ""
                md += [f"- `{c['name']}({c['args']})`{img} -> {c['result']}"]
            if s.get("answer"):
                md += ["", "**Final answer:**", "", s["answer"]]
                if s.get("recovery"):
                    md += ["", f"> harness note: {s['recovery']}"]
        if res["ws_findings"]:
            md += ["", "**Final worksheet findings:**"] + [f"- {f}" for f in res["ws_findings"]]
        taf = res["final_taf"] or res["last_taf"]
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
for res in results:
    ws_state = ("clean" if res["accepted_ws"] is not None and not res["ws_findings"]
                else "accepted" if res["accepted_ws"] is not None else "none")
    taf_state = ("clean TAF" if res["final_taf"] else f"{len(tafgen.validate(res['last_taf']))} findings"
                 if res["last_taf"] else "no TAF")
    print(f"  {res['label']:<8} {len(res['steps'])} steps | ws: {ws_state:<8} | {taf_state:<12} | "
          f"ev {len(res['evidence'])} | {res['ptok']}p+{res['ctok']}c")
print(f"\nFull transcript -> {log_path}")
