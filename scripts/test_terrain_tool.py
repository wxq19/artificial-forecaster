"""End-to-end: the agent reasons over get_terrain (text rose + shaded-relief map).

Drives the get_terrain tool loop: a natural-language question -> the model calls get_terrain
for a station -> our code classifies the terrain from DEM elevation (Open-Meteo), checks the
coast (global-land-mask), and stitches an Esri shaded-relief map (station marker, nearby
airfields, range rings) -> the text rose + map image are fed back to the VLM -> the model
reads them and reasons about terrain-driven weather (upslope fog, downslope drying, cold-air
pooling, sea breeze). Exercises terrain.py (relief_map + descriptors) and the image-return
path. No DB (terrain is static, network-only).

Images the model saw are saved beside the markdown log so the run is reviewable.
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

from forecaster import tools
from forecaster.agent import final_answer
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import GET_TERRAIN, run_tool

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KVBG", help="airport ICAO (default: KVBG, coastal + terrain)")
_ap.add_argument("--model", default=settings.llm_model, help="model id override (default: .env)")
_ap.add_argument("--max-steps", type=int, default=4, help="max model turns (default: 4)")
_ap.add_argument("--max-tokens", type=int, default=8192, help="completion budget per turn")
_args = _ap.parse_args()

STATION = _args.station.upper()
MODEL = _args.model
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2

messages = [
    {"role": "system", "content": (
        "You are a weather forecaster. You have a get_terrain tool that returns the STATIC "
        "terrain and coastline around an airport as a text summary plus a shaded-relief map. "
        "Call it, then READ the summary AND the image and reason about terrain-driven weather "
        "for this field -- upslope fog/precipitation, downslope drying/warming, valley cold-air "
        "pooling, and sea-breeze or advection fog if it is coastal. State your answer once, "
        "then stop."
    )},
    {"role": "user", "content": (
        f"Describe the terrain and coastline around {STATION} and explain the main "
        "terrain-driven weather effects a TAF forecaster there should watch for. "
        "Tie each effect to a specific direction or feature you see."
    )},
]

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")
img_dir = Path("logs") / f"terrain_{STATION}_{model_slug}_{stamp}"
img_dir.mkdir(parents=True, exist_ok=True)

steps: list[dict] = []
saved: list[tuple[str, str]] = []
answer = recovery = None
prompt_tok = completion_tok = 0

for n in range(1, MAX_STEPS + 1):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[GET_TERRAIN],
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
            res = run_tool(tc.function.name, args)
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


def build_markdown() -> str:
    md = [
        f"# Agent Terrain Reasoning (get_terrain) -- {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{MODEL}` @ {settings.llm_base_url}",
        f"- **Station:** {STATION}",
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
            md += ["", f"**Tool call:** `{c['name']}({c['args']})` -> {c['n_images']} image(s)",
                   "", "```text", c["receipt"], "```"]
        if s.get("answer"):
            md += ["", "**Answer:**", "", s["answer"]]
    if recovery:
        md += ["", f"> harness note: {recovery}"]
    return "\n".join(md)


log_path = img_dir / "log.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== get_terrain reasoning -- {STATION} ({MODEL}) ===")
for s in steps:
    tag = "answered" if s.get("answer") else f"{len(s['calls'])} tool call(s)"
    print(f"  step {s['n']} [{s['finish']}]: {tag}")
if answer:
    print("\n=== MODEL ANSWER ===")
    print(answer)
    if recovery:
        print(f"\n[harness note: {recovery}]")
print(f"\nImages + log written to {img_dir}")
