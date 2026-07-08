"""Full-agent TAF test: each model uses the ENTIRE tool suite to build a KLSV TAF.

Hands every model (Gemma, Qwen, Kimi K2.7, MiniMax M3) the complete toolset -- METAR
queries (query_obs/get_latest_obs), the meteogram (get_trend), observed + model-forecast
soundings (get_sounding/get_fcst_sounding), synoptic charts (get_map), an hourly point
forecast (get_point_forecast), and the emit_taf OUTPUT tool -- and asks each to produce a
30-hour AF TAF for KLSV valid 2300Z. The model drives: it decides which tools to call,
reasons over text + image results, then emits a TAF and corrects AFMAN findings. This is
the first end-to-end exercise of the whole agent loop, so we can see how each model
orchestrates the tools. The full transcript (per model: reasoning, every tool call, the
emitted TAFs + findings, the final TAF, tools-used tally, tokens) -> a markdown log.

KLSV (Nellis AFB) has surface obs but NO BUFKIT model output, so the prompt points the
model at nearby KLAS for the model-forecast tools (a realistic proxy).
"""

import argparse
import base64
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from forecaster import awc, store, tafgen, tools
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import EMIT_TAF, TOOLS, final_answer, run_tool

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KLSV")
_ap.add_argument("--valid", default="2026-07-07T23:00", help="TAF valid start (naive UTC)")
_ap.add_argument("--max-steps", type=int, default=12, help="max model turns (default: 12)")
_ap.add_argument("--max-tokens", type=int, default=30000, help="completion budget per turn")
_ap.add_argument("--ingest-hours", type=int, default=48, help="hours of KLSV obs to load")
_args = _ap.parse_args()

STATION = _args.station.upper()
VALID = datetime.fromisoformat(_args.valid.rstrip("Z"))
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2
TOOLSET = TOOLS + [EMIT_TAF]      # all read/data tools + the emit_taf output tool

MODELS = [
    ("Gemma", "google/gemma-4-31B-it"),
    ("Qwen", "Qwen/Qwen3.5-9B"),
    ("Kimi", "moonshotai/Kimi-K2.7-Code"),
    ("MiniMax", "MiniMaxAI/MiniMax-M3"),
]

# --- ingest KLSV obs so the DB-backed tools have data ---
load_summary = awc.load_metar(STATION, hours=_args.ingest_hours)
con = store.connect(read_only=True)
try:
    n_obs = store.count(con, STATION)
finally:
    con.close()

SYSTEM = (
    "You are a USAF weather forecaster issuing terminal aerodrome forecasts under AFMAN "
    "15-124. You have tools to gather data: query_obs/get_latest_obs (stored METARs), "
    "get_trend (a meteogram of recent trends), get_sounding (OBSERVED skew-T), "
    "get_fcst_sounding (MODEL forecast skew-T), get_map (synoptic surface/upper-air "
    "charts), get_point_forecast (hourly model point forecast), and emit_taf (submit your "
    "forecast). Think step by step in your replies. Gather what you need, reason about how "
    "wind, visibility, ceiling, and weather will evolve over the 30-hour period, THEN call "
    "emit_taf. If emit_taf returns AFMAN findings, correct them and re-emit until clean. "
    "Base the forecast only on data returned by the tools."
)
TASK = (
    f"Produce a 30-hour Air Force TAF for {STATION} (Nellis AFB, Las Vegas NV), valid from "
    f"{VALID:%d%H%M}Z ({VALID:%Y-%m-%d %H:%MZ}).\n"
    f"NOTE: {STATION} has surface observations but NO model BUFKIT output -- use nearby "
    "KLAS (Las Vegas Harry Reid, ~10 mi SW) for get_fcst_sounding and get_point_forecast. "
    "Observed radiosonde soundings exist only at upper-air sites, which may be far from "
    f"{STATION}. Begin by checking current and recent conditions."
)


def run_model(label: str, model: str) -> dict:
    """Drive the full tool loop for one model. Returns a record for the log."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TASK}]
    steps: list[dict] = []
    used: Counter = Counter()
    final_taf = last_taf = None
    ptok = ctok = 0
    for n in range(1, MAX_STEPS + 1):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLSET, tool_choice="auto",
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
        except Exception as e:  # noqa: BLE001 -- a model that rejects the toolset is a finding
            steps.append({"n": n, "error": f"{type(e).__name__}: {e}"})
            return {"label": label, "model": model, "steps": steps, "used": used,
                    "final_taf": final_taf, "last_taf": last_taf, "ptok": ptok, "ctok": ctok,
                    "fatal": f"{type(e).__name__}: {e}"}
        ptok += r.usage.prompt_tokens
        ctok += r.usage.completion_tokens
        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        rec = {"n": n, "finish": r.choices[0].finish_reason,
               "content": (msg.content or "").strip(),
               "reasoning": (getattr(msg, "reasoning", None) or "").strip(), "calls": []}

        if not tcs:
            answer, recovery = final_answer(msg, r.choices[0].finish_reason)
            rec["answer"] = answer
            rec["recovery"] = recovery
            steps.append(rec)
            break

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs]})
        images: list[bytes] = []
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
            res = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.text})
            images += res.images
            if name == "emit_taf" and res.taf is not None:
                last_taf = res.taf
                if not tafgen.validate(res.taf):
                    final_taf = res.taf
            rec["calls"].append({"name": name, "args": json.dumps(args)[:160],
                                 "result": res.text.splitlines()[0][:160],
                                 "n_images": len(res.images)})
        if images:
            content = [{"type": "text", "text": "Images from the tool calls above:"}]
            for im in images:
                b64 = base64.b64encode(im).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{tools._image_mime(im)};base64,{b64}"}})
            messages.append({"role": "user", "content": content})
        steps.append(rec)
        if final_taf is not None:
            break
    return {"label": label, "model": model, "steps": steps, "used": used,
            "final_taf": final_taf, "last_taf": last_taf, "ptok": ptok, "ctok": ctok,
            "fatal": None}


results = [run_model(label, model) for label, model in MODELS]


def build_markdown() -> str:
    md = [
        f"# Full-agent TAF test -- {STATION} valid {VALID:%d%H%M}Z",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **Toolset:** {', '.join(t['function']['name'] for t in TOOLSET)}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}, max_steps={MAX_STEPS}",
        f"- **Obs:** {n_obs} KLSV rows in DB; load {load_summary}",
        "",
        "## Summary",
        "",
        "| Model | Steps | Tools used | Outcome | Tokens (p+c) |",
        "|---|---|---|---|---|",
    ]
    for res in results:
        used = ", ".join(f"{k}x{v}" for k, v in res["used"].items()) or "(none)"
        if res["fatal"]:
            outcome = f"FATAL: {res['fatal'][:60]}"
        elif res["final_taf"] is not None:
            outcome = "clean TAF"
        elif res["last_taf"] is not None:
            outcome = f"TAF w/ {len(tafgen.validate(res['last_taf']))} findings"
        else:
            outcome = "no TAF emitted"
        md.append(f"| {res['label']} | {len(res['steps'])} | {used} | {outcome} | "
                  f"{res['ptok']}+{res['ctok']} |")

    for res in results:
        md += ["", "---", "", f"## {res['label']} (`{res['model']}`)"]
        if res["fatal"]:
            md += ["", f"**FATAL:** `{res['fatal']}`"]
        for s in res["steps"]:
            md += ["", f"### Step {s['n']} (finish: `{s.get('finish', '?')}`)"]
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
        taf = res["final_taf"] or res["last_taf"]
        if taf is not None:
            md += ["", "**Emitted TAF:**", "", "```text"]
            try:
                md += [tafgen.render_taf(taf)]
            except Exception as e:  # noqa: BLE001
                md += [f"(render failed: {e})"]
            md += ["```"]
            findings = tafgen.validate(taf)
            md += [f"- AFMAN findings: {findings or 'clean'}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"taf_agent_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== Full-agent TAF test -- {STATION} valid {VALID:%d%H%M}Z ({n_obs} obs) ===")
for res in results:
    used = ", ".join(f"{k}x{v}" for k, v in res["used"].items()) or "(none)"
    outcome = ("FATAL" if res["fatal"] else "clean TAF" if res["final_taf"]
               else f"{len(tafgen.validate(res['last_taf']))} findings" if res["last_taf"]
               else "no TAF")
    print(f"  {res['label']:<8} {len(res['steps'])} steps | {outcome:<12} | "
          f"{res['ctok']} ctok | tools: {used}")
print(f"\nFull transcript -> {log_path}")
