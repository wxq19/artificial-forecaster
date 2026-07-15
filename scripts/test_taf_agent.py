"""Full-agent TAF test: each model uses the ENTIRE tool suite to build a KLSV TAF.

Hands every model (Gemma, Qwen, Kimi K2.7, MiniMax M3) the complete toolset -- METAR
queries (query_obs/get_latest_obs), the meteogram (get_trend), observed + model-forecast
soundings (get_sounding/get_fcst_sounding), synoptic charts (get_map), an hourly point
forecast (get_point_forecast), and the emit_taf OUTPUT tool -- and asks each to produce a
30-hour AF TAF for KLSV valid 2300Z. The model drives: it decides which tools to call,
reasons over text + image results, then emits a TAF and corrects AFMAN findings. The loop
itself now lives in forecaster.agent (run_agent); this driver just builds the prompt +
config, runs it per model, and renders the transcript (per model: reasoning, every tool
call, the emitted TAFs + findings, the final TAF, tools-used tally, tokens) to a markdown log.

KLSV (Nellis AFB) has surface obs but NO BUFKIT model output, so the prompt points the
model at nearby KLAS for the model-forecast tools (a realistic proxy).
"""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import awc, store, tafgen
from forecaster.agent import AgentConfig, RunResult, run_agent
from forecaster.config import settings
from forecaster.tools import EMIT_TAF, TOOLS

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KLSV")
_ap.add_argument("--valid", default="2026-07-07T23:00", help="TAF valid start (naive UTC)")
_ap.add_argument("--max-steps", type=int, default=12, help="max model turns (default: 12)")
_ap.add_argument("--max-tokens", type=int, default=8000, help="completion budget per turn")
_ap.add_argument("--ingest-hours", type=int, default=48, help="hours of KLSV obs to load")
_args = _ap.parse_args()

STATION = _args.station.upper()
VALID = datetime.fromisoformat(_args.valid.rstrip("Z"))
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2
TOOLSET = TOOLS + [EMIT_TAF]      # all read/data tools + the emit_taf output tool

# Per-tool call caps (network/image tools): a call PAST the cap returns feedback instead
# of executing, so a model can't burn the whole run gathering (the Kimi get_map x22 case).
# The cheap DB/text tools stay uncapped.
TOOL_CAPS = {"get_map": 8, "get_sounding": 8, "get_fcst_sounding": 8, "get_point_forecast": 8}

MODELS = [
    ("Gemma", "google/gemma-4-31B-it"),
    ("Qwen", "Qwen/Qwen3.5-9B"),
    ("Kimi", "moonshotai/Kimi-K2.7-Code"),
    ("MiniMax", "MiniMaxAI/MiniMax-M3"),
]

# --- POINT-IN-TIME obs store: a THROWAWAY DB holding only obs BEFORE the valid start,
# so the DB-backed tools (get_latest_obs/query_obs/get_trend) cannot peek past the
# forecast start. Without this, re-running a PAST valid time leaks obs from inside the
# 30h window (the model would "observe" what it is meant to forecast). NOTE: the network
# model tools (get_point_forecast/get_fcst_sounding/get_map) still fetch the LATEST run,
# so for a past valid time some model guidance is still post-start -- a residual leak. ---
DB_PATH = str(Path(tempfile.mkdtemp(prefix="taf_agent_")) / "obs.duckdb")
load_summary = awc.load_metar(STATION, hours=_args.ingest_hours, db_path=DB_PATH, before=VALID)
con = store.connect(DB_PATH, read_only=True)
try:
    n_obs = store.count(con, STATION)
    latest_obs = store.latest(con, STATION, 1)
finally:
    con.close()
LATEST_OBS_TIME = latest_obs[0]["obs_time"] if latest_obs else None

SYSTEM = (
    "You are a USAF weather forecaster issuing terminal aerodrome forecasts under AFMAN "
    "15-124. You have tools to gather data: query_obs/get_latest_obs (stored METARs), "
    "get_trend (a meteogram of recent trends), get_sounding (OBSERVED skew-T), "
    "get_fcst_sounding (MODEL forecast skew-T), get_map (synoptic surface/upper-air "
    "charts), get_point_forecast (hourly model point forecast), and emit_taf (submit your "
    "forecast). Think step by step in your replies. Gather what you need, reason about how "
    "wind, visibility, ceiling, and weather will evolve over the 30-hour period, THEN call "
    "emit_taf. If emit_taf returns AFMAN findings, correct them and re-emit until clean. "
    "Base the forecast only on data returned by the tools. "
    f"You have at most {MAX_STEPS} tool-calling turns; gather efficiently and call emit_taf "
    f"by turn {MAX_STEPS - 2}, leaving turns to correct any AFMAN findings."
    "\n\n" + tafgen.emit_taf_guide()
)
TASK = (
    f"Produce a 30-hour Air Force TAF for {STATION} (Nellis AFB, Las Vegas NV), valid from "
    f"{VALID:%d%H%M}Z ({VALID:%Y-%m-%d %H:%MZ}).\n"
    f"NOTE: {STATION} has surface observations but NO model BUFKIT output -- use nearby "
    "KLAS (Las Vegas Harry Reid, ~10 mi SW) for get_fcst_sounding and get_point_forecast. "
    "Observed radiosonde soundings exist only at upper-air sites, which may be far from "
    f"{STATION}. Begin by checking current and recent conditions."
)


def run_model(label: str, model: str) -> tuple[str, RunResult]:
    """Drive the full tool loop for one model via the shared agent loop."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    cfg = AgentConfig(
        model=model, toolset=TOOLSET, max_steps=MAX_STEPS, max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE, tool_caps=TOOL_CAPS, evidence=False,
        step_budget_nudge=True, db_path=DB_PATH,
    )
    return label, run_agent(messages, cfg)


results = [run_model(label, model) for label, model in MODELS]


def _outcome(res: RunResult) -> str:
    if res.fatal:
        return f"FATAL: {res.fatal[:60]}"
    if res.final_taf is not None:
        return "clean TAF"
    if res.last_taf is not None:
        return f"TAF w/ {len(tafgen.validate(res.last_taf))} findings"
    return "no TAF emitted"


def build_markdown() -> str:
    md = [
        f"# Full-agent TAF test -- {STATION} valid {VALID:%d%H%M}Z",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **Toolset:** {', '.join(t['function']['name'] for t in TOOLSET)}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}, max_steps={MAX_STEPS}, "
        f"tool caps={TOOL_CAPS}",
        f"- **Obs (pre-cutoff feed):** {n_obs} {STATION} rows, all BEFORE the {VALID:%d%H%M}Z valid start "
        f"(latest ob {LATEST_OBS_TIME:%Y-%m-%dT%H:%MZ}); load {load_summary}"
        if LATEST_OBS_TIME else f"- **Obs:** {n_obs} rows; load {load_summary}",
        "",
        "## Summary",
        "",
        "| Model | Steps | Converge | Tools used | Outcome | End ctx (ptok) | Tokens (p+c) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, res in results:
        used = ", ".join(f"{k}x{v}" for k, v in res.used.items()) or "(none)"
        conv = "fatal" if res.fatal else res.convergence
        end_ctx = res.steps[-1].prompt_tokens if res.steps else "-"
        md.append(f"| {label} | {len(res.steps)} | {conv} | {used} | {_outcome(res)} | "
                  f"{end_ctx} | {res.prompt_tokens}+{res.completion_tokens} |")
    md += ["",
           "> Cost note: every turn re-sends the ENTIRE conversation -- including every image "
           "returned so far -- as fresh PROMPT tokens, so prompt tokens dominate and grow each "
           "step. 'End ctx' is the final turn's prompt size (the conversation's peak). A wide "
           "gather loop (huge prompt total, small completion) can cost more than a ruminator "
           "(huge completion); read both columns. Convergence: unprompted (emitted before the "
           "nudge) / nudged (only after the step-{n-2} nudge) / never."]

    for label, res in results:
        md += ["", "---", "", f"## {label} (`{res.model}`)"]
        if res.fatal:
            md += ["", f"**FATAL:** `{res.fatal}`"]
        for s in res.steps:
            md += ["", f"### Step {s.n} (finish: `{s.finish_reason}`) -- "
                   f"{s.prompt_tokens} ptok / {s.completion_tokens} ctok"]
            if res.nudge_step == s.n:
                md += ["", "> harness: convergence nudge injected (budget nearly spent, no emit)."]
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
        taf = res.final_taf or res.last_taf
        if taf is not None:
            md += ["", "**Emitted TAF:**", "", "```text"]
            try:
                md += [tafgen.render_taf(taf)]
            except Exception as e:  # noqa: BLE001
                md += [f"(render failed: {e})"]
            md += ["```"]
            md += [f"- AFMAN findings: {tafgen.validate(taf) or 'clean'}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"taf_agent_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== Full-agent TAF test -- {STATION} valid {VALID:%d%H%M}Z ({n_obs} obs) ===")
for label, res in results:
    used = ", ".join(f"{k}x{v}" for k, v in res.used.items()) or "(none)"
    conv = "fatal" if res.fatal else res.convergence
    print(f"  {label:<8} {len(res.steps)} steps | {conv:<10} | {_outcome(res):<20} | "
          f"{res.prompt_tokens}p+{res.completion_tokens}c | tools: {used}")
print(f"\nFull transcript -> {log_path}")
