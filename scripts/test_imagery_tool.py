"""End-to-end: the agent reasons over BOTH satellite and radar imagery.

Drives the get_imagery tool loop: a natural-language question -> the model calls
get_imagery for a SATELLITE view and for RADAR at a station -> our code fetches the
pre-rendered, composited images (STAR CDN GOES JPEG; IEM radmap.php PNG) from the live
providers -> the images are fed back to the VLM -> the model reads them and answers.
Exercises the imagery.py seam, the get_imagery tool + station-aware radar cascade, and
the JPEG/PNG image-return path end to end.

Multiple tool calls in one turn are handled the OpenAI-safe way (all tool receipts first,
then the images batched into ONE follow-up user message). The images the model actually
saw are saved beside the markdown log so the run is reviewable.
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

from forecaster import tools
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import GET_IMAGERY, final_answer, run_tool

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KLSV", help="airport ICAO for the radar view (default: KLSV)")
_ap.add_argument("--model", default=settings.llm_model, help="model id override (default: .env = Gemma)")
_ap.add_argument("--max-steps", type=int, default=5, help="max model turns (default: 5)")
_ap.add_argument("--max-tokens", type=int, default=8192, help="completion budget per turn (default: 8192)")
_args = _ap.parse_args()

STATION = _args.station.upper()
MODEL = _args.model
MAX_STEPS = _args.max_steps
MAX_TOKENS = _args.max_tokens
TEMPERATURE = 0.2

messages = [
    {"role": "system", "content": (
        "You are a weather forecaster. You have a get_imagery tool that returns OBSERVED "
        "satellite and radar images. Call it to SEE current conditions, then READ the "
        "returned images and answer from what they show -- do not guess. Use satellite for "
        "the cloud field and convective (cold cloud-top) structure, and radar for "
        "precipitation echoes. State your answer once, then stop."
    )},
    {"role": "user", "content": (
        f"Assess the current convective and precipitation situation near {STATION}.\n"
        f"1. Retrieve a SATELLITE view for {STATION} and describe the cloud field and any "
        "convective (cold cloud-top) structure you can see.\n"
        f"2. Retrieve RADAR for {STATION} and describe any precipitation echoes near the field.\n"
        "3. Synthesize: is there active or approaching convection/precipitation near "
        f"{STATION}, and do the satellite and radar pictures agree?"
    )},
]

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")
img_dir = Path("logs") / f"imagery_{STATION}_{model_slug}_{stamp}"
img_dir.mkdir(parents=True, exist_ok=True)


def _ext(data: bytes) -> str:
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    return "png"


steps: list[dict] = []
saved: list[tuple[str, str]] = []       # (filename, receipt label) for the log
answer = recovery = None
prompt_tok = completion_tok = 0

for n in range(1, MAX_STEPS + 1):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[GET_IMAGERY],
        tool_choice="auto", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    prompt_tok += r.usage.prompt_tokens
    completion_tok += r.usage.completion_tokens
    msg = r.choices[0].message
    finish = r.choices[0].finish_reason
    tcs = msg.tool_calls or []
    rec: dict = {"n": n, "finish": finish, "content": (msg.content or "").strip(),
                 "reasoning": (getattr(msg, "reasoning", None) or "").strip(), "calls": []}

    if not tcs:                                        # model answered
        answer, recovery = final_answer(msg, finish)
        rec["answer"] = answer
        steps.append(rec)
        break

    # Echo the assistant tool_calls, append all receipts, THEN batch images (OpenAI-safe).
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
            fname = f"step{n}_{len(saved)}.{_ext(im)}"
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
        f"# Agent Imagery Reasoning (get_imagery) -- {STATION}",
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

print(f"=== get_imagery reasoning -- {STATION} ({MODEL}) ===")
for s in steps:
    tag = "answered" if s.get("answer") else f"{len(s['calls'])} tool call(s): " + \
        ", ".join(f"{c['name']}({json.loads(c['args']).get('kind','?')})" for c in s["calls"])
    print(f"  step {s['n']} [{s['finish']}]: {tag}")
if answer:
    print("\n=== MODEL ANSWER ===")
    print(answer)
    if recovery:
        print(f"\n[harness note: {recovery}]")
print(f"\nImages + log written to {img_dir}")
