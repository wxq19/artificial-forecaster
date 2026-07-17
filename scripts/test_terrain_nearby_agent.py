"""End-to-end: the TWO-STEP spatial loop -- get_terrain (orient on the map) then
get_nearby_obs (fetch the chosen neighbors).

This is the driver for the workflow the pair of tools was built for: the model first calls
get_terrain to see the relief map with the nearby airfields plotted (blue = obs-available,
violet = context), reads the fetchable index, then DECIDES which neighbor observations to
pull -- passing get_nearby_obs a `stations` subset (e.g. the fields upwind or toward a coast)
rather than blindly taking all of them -- and finally reasons about whether current conditions
are a REGIONAL pattern or purely LOCAL. Exercises both tools in one loop, the image-return
path (get_terrain), and the selective DB read (get_nearby_obs with db_path).

The obs DB is a throwaway tempfile seeded with the home station + its neighbors via live AWC
ingest (the same path collect.py uses); obs are live (not cut off) -- this is a tool exercise,
not a scored collection run. Images the model saw are saved beside the markdown log.
"""

import argparse
import base64
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import awc, neighbors, store, tools
from forecaster.agent import final_answer
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import GET_NEARBY_OBS, GET_TERRAIN, run_tool

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KVBG", help="home airport ICAO (default: KVBG)")
_ap.add_argument("--model", default=settings.llm_model, help="model id override (default: .env)")
_ap.add_argument("--max-steps", type=int, default=5, help="max model turns (default: 5)")
_ap.add_argument("--max-tokens", type=int, default=8192, help="completion budget per turn")
_args = _ap.parse_args()

STATION = _args.station.upper()
MODEL = _args.model
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2

# --- seed a throwaway DB with the home station + its neighbors (live AWC, like collect.py) ---
tmp = Path(tempfile.mkdtemp(prefix="terrain_nearby_"))
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
        "You are a weather forecaster building a TAF for an airfield. You have two spatial "
        "tools. Use them in ORDER:\n"
        "1. get_terrain -- returns a relief map with the nearby airfields plotted (blue dots "
        "with labels are stations you CAN pull observations for; violet dots are context only) "
        "plus a text list of those fetchable airfields with distance/bearing/elevation. Call "
        "this FIRST and read both the map and the list.\n"
        "2. get_nearby_obs -- pass it a `stations` list of the specific fetchable airfields you "
        "want the latest observation from. CHOOSE deliberately: the fields upwind of you, or "
        "toward a coast or terrain feature that matters, not all of them blindly.\n"
        "Then judge whether the current visibility/ceiling/wind conditions are a REGIONAL "
        "pattern or purely LOCAL, and note anything upwind that could move toward the field. "
        "State your answer once, then stop."
    )},
    {"role": "user", "content": (
        f"Build the spatial picture for {STATION}: look at its terrain and surrounding "
        "airfields, decide which neighbor observations are worth pulling and why, then use "
        "them to judge whether conditions are regional or local."
    )},
]

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")
img_dir = Path("logs") / f"terrain_nearby_{STATION}_{model_slug}_{stamp}"
img_dir.mkdir(parents=True, exist_ok=True)

steps: list[dict] = []
saved: list[tuple[str, str]] = []
answer = recovery = None
prompt_tok = completion_tok = 0

try:
    for n in range(1, MAX_STEPS + 1):
        r = client.chat.completions.create(
            model=MODEL, messages=messages, tools=[GET_TERRAIN, GET_NEARBY_OBS],
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
        images: list[tuple[str, bytes]] = []
        for tc in tcs:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                res = tools.ToolResult(f"error: unparseable tool arguments: {e}")
            else:
                res = run_tool(tc.function.name, args, db_path=db)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.text})
            label = res.text.splitlines()[0] if res.text else tc.function.name
            for im in res.images:
                ext = "jpg" if im[:3] == b"\xff\xd8\xff" else "png"
                fname = f"step{n}_{len(saved)}.{ext}"
                (img_dir / fname).write_bytes(im)
                saved.append((fname, label))
                images.append((label, im))
            rec["calls"].append({"name": tc.function.name, "args": tc.function.arguments,
                                 "receipt": res.text, "n_images": len(res.images)})
        if images:
            content: list[dict] = [{"type": "text", "text": "Images from the tool calls above, "
                                    "each preceded by its receipt line:"}]
            for label, im in images:
                content.append({"type": "text", "text": f"[image for: {label}]"})
                b64 = base64.b64encode(im).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{tools._image_mime(im)};base64,{b64}"}})
            messages.append({"role": "user", "content": content})
        steps.append(rec)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


def build_markdown() -> str:
    md = [
        f"# Agent Spatial Loop (get_terrain -> get_nearby_obs) -- {STATION}",
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
    if saved:
        md += ["", "## Images the model saw"]
        for fname, label in saved:
            md += ["", f"**{label}**", "", f"![{fname}]({fname})"]
    md += ["", "## Model steps"]
    for s in steps:
        md += ["", f"### Step {s['n']} (finish: `{s['finish']}`)"]
        if s["reasoning"]:
            md += ["", "**Reasoning field:**", "", "```text", s["reasoning"], "```"]
        if s["content"]:
            md += ["", "**Reply:**", "", "```text", s["content"], "```"]
        for c in s["calls"]:
            md += ["", f"**Tool call:** `{c['name']}({c['args']})` -> {c.get('n_images', 0)} "
                   "image(s)", "", "```text", c["receipt"], "```"]
        if s.get("answer"):
            md += ["", "**Answer:**", "", s["answer"]]
    if recovery:
        md += ["", f"> harness note: {recovery}"]
    return "\n".join(md)


log_path = img_dir / "log.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== spatial loop -- {STATION} ({MODEL}) ===")
for s in steps:
    if s.get("answer"):
        tag = "answered"
    else:
        tag = ", ".join(f"{c['name']}({c['args']})" for c in s["calls"]) or "no calls"
    print(f"  step {s['n']} [{s['finish']}]: {tag}")
if answer:
    print("\n=== MODEL ANSWER ===")
    print(answer)
    if recovery:
        print(f"\n[harness note: {recovery}]")
print(f"\nImages + log written to {img_dir}")
