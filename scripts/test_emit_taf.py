"""End-to-end: the agent GENERATES a TAF via the emit_taf tool (roadmap step 6).

Hands the model the decoded observations available AT A CUTOFF plus a meteogram, and
asks it to reason about the trend and then issue a 30-hour AF TAF by calling emit_taf.
The model fills the fields of a TafProduct (the tool's parameter schema IS that
pydantic model); our code builds + renders + AFMAN-checks it. If validate() finds
violations the findings go back as the tool result and the model RE-EMITS a corrected
TAF -- so the rule checker is a live corrective signal, not just a gate.

The cutoff matters for a HONEST forecast: VALID_FROM is both the TAF's valid start AND
the data cutoff -- the model sees only obs strictly BEFORE that time, never the period
it is forecasting. tool_choice is 'auto' (not forced) and the prompt asks the model to
think first, so its reasoning lands in the message content and is captured to the log.
The whole exchange -- obs, every step's reasoning, the AFMAN findings, the final TAF --
is written to a self-contained markdown log under logs/.
"""

import argparse
import base64
import json
from datetime import datetime, timedelta
from pathlib import Path

from forecaster import awc, charts, store, tafgen, tools
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import EMIT_TAF, run_tool

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KBLV", help="ICAO station (default: KBLV reference case)")
_ap.add_argument("--valid-from", default="2026-06-29T16:00",
                 help="TAF valid start AND obs cutoff, naive-UTC 'YYYY-MM-DDTHH:MM' (default: KBLV 291600Z)")
_ap.add_argument("--lookback", type=int, default=24,
                 help="hours of pre-cutoff obs + meteogram handed to the model (default: 24)")
_ap.add_argument("--model", default=settings.llm_model,
                 help="model id override (default: settings.llm_model from .env)")
_ap.add_argument("--max-steps", type=int, default=8,
                 help="max model turns: reason -> emit -> (re-emit on findings) (default: 8)")
_ap.add_argument("--max-tokens", type=int, default=32768,
                 help="completion token budget per turn -- ample room to reason AND emit (default: 32768)")
_ap.add_argument("--diurnal-nudge", action="store_true",
                 help="append a general diurnal-recurrence guidance line to the system prompt (A/B the gust-handling finding)")
_args = _ap.parse_args()

STATION = _args.station.upper()
# Naive UTC (the store's window contract); tolerate a trailing Z so 15:00Z parses.
VALID_FROM = datetime.fromisoformat(_args.valid_from.rstrip("Z"))   # valid start AND obs cutoff
LOOKBACK = _args.lookback   # hours of obs (before the cutoff) + meteogram handed to the model
MODEL = _args.model
DIURNAL_NUDGE = _args.diurnal_nudge
MAX_STEPS = _args.max_steps  # model turns: reason -> emit -> (re-emit on findings)
TEMPERATURE = 0.2
MAX_TOKENS = _args.max_tokens


# --- 1. Ingest, then read ONLY obs strictly before the cutoff ---
# Pull a generous live window, then clip to [cutoff-LOOKBACK, cutoff) so nothing at or
# after VALID_FROM can leak in (the period the model must forecast).
load_summary = awc.load_metar(STATION, hours=LOOKBACK + 12)
con = store.connect(read_only=True)
try:
    rows = store.window(con, STATION, VALID_FROM - timedelta(hours=LOOKBACK), VALID_FROM)
finally:
    con.close()
rows = [r for r in rows if r["obs_time"] < VALID_FROM]   # STRICT cutoff
if not rows:
    raise SystemExit(f"no observations for {STATION} before {VALID_FROM:%d%H%MZ}; aborting")

obs_block = tools._fmt(rows, "oldest first")
meteogram = charts.meteogram(rows, station=STATION, hours=LOOKBACK)
issue_dt = max(r["obs_time"] for r in rows)              # issue at the latest ob we actually have


# --- 2. Opening messages: obs text + meteogram image + reason-then-emit instruction ---
b64 = base64.b64encode(meteogram).decode()
system_content = (
    "You are a USAF weather forecaster issuing terminal aerodrome forecasts under "
    "AFMAN 15-124. Base the forecast ONLY on the observations and meteogram given -- "
    "you have NO data at or after the valid time, so this is a genuine forecast. "
    "Before calling emit_taf, think step by step IN YOUR REPLY: describe the current "
    "trend (improving, deteriorating, or steady), how you expect wind, visibility, "
    "ceiling, and weather to evolve over the 30-hour period, and any convective or "
    "restriction risks. THEN call emit_taf with your forecast."
)
if DIURNAL_NUDGE:
    # General principle only (NOT "add gusts") -- a fair A/B of the diurnal-recurrence miss.
    system_content += (
        " Distinguish DIURNAL features from one-off synoptic events: anything tied to the "
        "daily heating cycle -- afternoon wind and gusts, convective cloud, the temperature "
        "swing -- RECURS on every day the TAF is valid, even when the forecast opens during a "
        "calm part of the cycle. Do not dismiss a recent wind or gust episode as a passed "
        "front without ruling out a diurnal cause; if it gusted in a past afternoon, expect "
        "gusts again in the valid afternoons unless the pattern is clearly changing."
    )
system_content += "\n\n" + tafgen.emit_taf_guide()
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": [
        {"type": "text", "text": (
            f"Station {STATION}. All observations available as of {VALID_FROM:%d%H%MZ} "
            f"(the last {LOOKBACK}h, decoded with raw beneath each):\n\n{obs_block}\n\n"
            "A meteogram of the same period follows. Reason about the trend, then issue a "
            f"30-hour TAF for {STATION}, issued day {issue_dt.day:02d} at "
            f"{issue_dt.hour:02d}{issue_dt.minute:02d}Z, valid from day "
            f"{VALID_FROM.day:02d} {VALID_FROM.hour:02d}00Z. Call emit_taf when ready."
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]},
]


# --- 3. Reason -> emit -> re-emit loop (tool_choice 'auto' so reasoning is visible) ---
steps: list[dict] = []
final_taf = None
prompt_tok = completion_tok = 0

for n in range(1, MAX_STEPS + 1):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[EMIT_TAF],
        tool_choice="auto", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    prompt_tok += r.usage.prompt_tokens
    completion_tok += r.usage.completion_tokens
    msg = r.choices[0].message
    content = (msg.content or "").strip()
    reasoning_field = (getattr(msg, "reasoning", None) or "").strip()
    tcs = msg.tool_calls or []
    rec: dict = {"n": n, "content": content, "reasoning": reasoning_field,
                 "finish": r.choices[0].finish_reason}

    if not tcs:
        # Model reasoned but did not emit yet -- keep its text and nudge it to call.
        rec["note"] = "no emit_taf call this turn; nudged"
        steps.append(rec)
        messages.append({"role": "assistant", "content": msg.content})
        messages.append({"role": "user", "content":
                         "Now call emit_taf with the forecast you described."})
        continue

    tc = tcs[0]
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError as e:
        rec["error"] = f"unparseable tool arguments: {e}"
        steps.append(rec)
        break
    res = run_tool("emit_taf", args)
    rec["findings"] = tafgen.validate(res.taf) if res.taf else None
    rec["receipt"] = res.text
    rec["taf"] = res.taf
    steps.append(rec)

    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
        {"id": tc.id, "type": "function",
         "function": {"name": "emit_taf", "arguments": tc.function.arguments}}]})
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.text})
    if res.taf and not rec["findings"]:
        final_taf = res.taf
        break
    messages.append({"role": "user", "content":
                     "Re-emit the TAF with those issues fixed; change nothing else."})


# --- 4. Markdown log (with the model's reasoning) ---
def build_markdown() -> str:
    md = [
        f"# Agent TAF Generation (emit_taf) — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{MODEL}` @ {settings.llm_base_url}",
        f"- **Prompt variant:** {'diurnal-nudge' if DIURNAL_NUDGE else 'baseline'}",
        f"- **Cutoff / valid from:** {VALID_FROM:%d%H%MZ} (obs strictly before this only)",
        f"- **Obs:** {len(rows)} rows over the {LOOKBACK}h before cutoff "
        f"(latest {issue_dt:%d%H%MZ}); load: {load_summary}",
        f"- **Outcome:** {'clean TAF' if final_taf else 'no clean TAF'} after {len(steps)} "
        f"step(s); tokens {prompt_tok}+{completion_tok}",
        "",
        "## Observations given to the model (pre-cutoff)",
        "", "```text", obs_block, "```",
        "_(a meteogram image of the same period was also provided)_",
        "",
        "## Model steps",
    ]
    for s in steps:
        md += ["", f"### Step {s['n']} (finish: `{s['finish']}`)"]
        if s["reasoning"]:
            md += ["", "**Reasoning field:**", "", "```text", s["reasoning"], "```"]
        if s["content"]:
            md += ["", "**Reply / reasoning:**", "", "```text", s["content"], "```"]
        if s.get("note"):
            md += ["", f"_{s['note']}_"]
        if s.get("error"):
            md += ["", f"- error: {s['error']}"]
        if s.get("receipt"):
            md += ["", "**emit_taf result:**", "", "```text", s["receipt"], "```"]
            if s["findings"]:
                md += ["", f"AFMAN findings ({len(s['findings'])}):"] + [f"- {f}" for f in s["findings"]]
            elif s.get("taf"):
                md += ["", "AFMAN check: clean."]
    if final_taf:
        md += ["", "## Final TAF (AFMAN-clean)", "", "```text", tafgen.render_taf(final_taf), "```",
               "", f"- round-trip: {tafgen.roundtrip(final_taf) or 'clean'}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")   # keep Gemma vs Qwen logs distinguishable
nudge_slug = "_nudge" if DIURNAL_NUDGE else ""        # keep A/B (baseline vs nudged) logs distinguishable
log_path = log_dir / f"emit_taf_{STATION}_{model_slug}{nudge_slug}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== emit_taf — {STATION} (valid {VALID_FROM:%d%H%MZ}, {len(rows)} pre-cutoff obs) ===")
for s in steps:
    tag = ("clean" if s.get("taf") and not s["findings"]
           else f"{len(s['findings'])} finding(s)" if s.get("findings")
           else s.get("note") or s.get("error") or "reasoned")
    print(f"  step {s['n']}: {tag}")
if final_taf:
    print("\n=== FINAL TAF ===")
    print(tafgen.render_taf(final_taf))
else:
    print("\n(no AFMAN-clean TAF produced)")
print(f"\nReasoning + full exchange written to {log_path}")
