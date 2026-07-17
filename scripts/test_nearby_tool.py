"""End-to-end: the agent reasons over get_nearby_obs (the mesoscale picture).

Seeds a THROWAWAY obs DB with the home station plus its nearest-neighbor airfields (live
AWC ingest -- the same path collect.py uses), then drives the get_nearby_obs tool loop: the
model asks for the surrounding observations and reasons about whether a restriction (fog,
low ceiling, gusts) is REGIONAL or purely LOCAL, and what is upwind/upslope. Exercises the
neighbors roster, the leakage-safe DB read (run_tool with db_path), and the spatial
annotation (distance / bearing / elevation delta).

The DB is a tempfile so this never touches the benchmark DB; obs are live (not cut off) --
this is a tool exercise, not a scored collection run.
"""

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import awc, neighbors, store, tools
from forecaster.agent import final_answer
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import GET_NEARBY_OBS, run_tool

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KWRI", help="home airport ICAO (default: KWRI)")
_ap.add_argument("--model", default=settings.llm_model, help="model id override (default: .env)")
_ap.add_argument("--max-steps", type=int, default=4, help="max model turns (default: 4)")
_ap.add_argument("--max-tokens", type=int, default=8192, help="completion budget per turn")
_args = _ap.parse_args()

STATION = _args.station.upper()
MODEL = _args.model
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2

# --- seed a throwaway DB with the home station + its neighbors (live AWC, like collect.py) ---
tmp = Path(tempfile.mkdtemp(prefix="nearby_"))
db = str(tmp / "obs.duckdb")
con = store.connect(db)
store.init_schema(con)
con.close()
ingest_log: list[str] = []
for icao in [STATION] + [n[0] for n in neighbors.neighbors_of(STATION)]:
    try:
        res = awc.load_metar(icao, hours=3, db_path=db)
        ingest_log.append(f"{icao}: {res.get('inserted', 0)} obs")
    except Exception as e:  # noqa: BLE001 -- a dud neighbor is skipped, as in the collector
        ingest_log.append(f"{icao}: skipped ({type(e).__name__})")
print("Ingest:", "; ".join(ingest_log))

messages = [
    {"role": "system", "content": (
        "You are a weather forecaster building a TAF. You have a get_nearby_obs tool that "
        "returns the latest observation from the nearest airfields around your station, each "
        "labeled with distance, compass bearing FROM your station, and elevation difference. "
        "Call it, then reason about the MESOSCALE picture: is any restriction (haze, fog, low "
        "ceiling, gusts) regional or purely local, and what is upwind of your field? State "
        "your answer once, then stop."
    )},
    {"role": "user", "content": (
        f"For {STATION}, use the surrounding airfield observations to judge whether current "
        "conditions (visibility restrictions, ceilings, wind) are a REGIONAL pattern or LOCAL "
        f"to {STATION}, and note anything upwind that could move toward the field."
    )},
]

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

steps: list[dict] = []
answer = recovery = None
prompt_tok = completion_tok = 0

try:
    for n in range(1, MAX_STEPS + 1):
        r = client.chat.completions.create(
            model=MODEL, messages=messages, tools=[GET_NEARBY_OBS],
            tool_choice="auto", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        prompt_tok += r.usage.prompt_tokens
        completion_tok += r.usage.completion_tokens
        msg = r.choices[0].message
        finish = r.choices[0].finish_reason
        tcs = msg.tool_calls or []
        rec: dict = {"n": n, "finish": finish, "content": (msg.content or "").strip(),
                     "reasoning": (getattr(msg, "reasoning", None) or "").strip(), "calls": []}
        if not tcs:
            answer, recovery = final_answer(msg, finish)
            rec["answer"] = answer
            steps.append(rec)
            break
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs]})
        for tc in tcs:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                res = tools.ToolResult(f"error: unparseable tool arguments: {e}")
            else:
                res = run_tool(tc.function.name, args, db_path=db)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.text})
            rec["calls"].append({"name": tc.function.name, "args": tc.function.arguments,
                                 "receipt": res.text})
        steps.append(rec)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


def build_markdown() -> str:
    md = [
        f"# Agent Nearby-Obs Reasoning (get_nearby_obs) -- {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{MODEL}` @ {settings.llm_base_url}",
        f"- **Station:** {STATION}",
        f"- **Ingest:** {'; '.join(ingest_log)}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}; "
        f"tokens {prompt_tok}+{completion_tok}",
        "",
        "## Messages sent (initial)",
    ]
    for m in messages[:2]:
        md += ["", f"### role: {m['role']}", "", "```text", m["content"], "```"]
    md += ["", "## Model steps"]
    for s in steps:
        md += ["", f"### Step {s['n']} (finish: `{s['finish']}`)"]
        if s["reasoning"]:
            md += ["", "**Reasoning field:**", "", "```text", s["reasoning"], "```"]
        if s["content"]:
            md += ["", "**Reply:**", "", "```text", s["content"], "```"]
        for c in s["calls"]:
            md += ["", f"**Tool call:** `{c['name']}({c['args']})`",
                   "", "```text", c["receipt"], "```"]
        if s.get("answer"):
            md += ["", "**Answer:**", "", s["answer"]]
    if recovery:
        md += ["", f"> harness note: {recovery}"]
    return "\n".join(md)


log_path = log_dir / f"nearby_{STATION}_{model_slug}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== get_nearby_obs reasoning -- {STATION} ({MODEL}) ===")
for s in steps:
    tag = "answered" if s.get("answer") else f"{len(s['calls'])} tool call(s)"
    print(f"  step {s['n']} [{s['finish']}]: {tag}")
if answer:
    print("\n=== MODEL ANSWER ===")
    print(answer)
    if recovery:
        print(f"\n[harness note: {recovery}]")
print(f"\nLog written to {log_path}")
