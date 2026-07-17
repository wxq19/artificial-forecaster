"""End-to-end: the agent reads a satellite LOOP (filmstrip + optional video).

Drives the get_loop tool for one station: our code fetches N time-stamped frames and
composes a labeled filmstrip (universal, any vision model) PLUS a short mp4. The filmstrip
rides the existing image_url path; the mp4 is delivered as a video_url content part ONLY
when the model supports video (llm.supports_video -- e.g. MiniMax M3 on Together, a provider
extension of the OpenAI format). The model then reads the sequence and describes the
MOTION/TREND. Verifies the loop tool, the filmstrip, the mp4, and the gated video delivery.

The images/video the model actually saw are saved beside the markdown log for review.
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

from forecaster import tools
from forecaster.config import settings
from forecaster.llm import client, supports_video
from forecaster.tools import GET_LOOP, run_tool

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--station", default="KWRI", help="airport ICAO (default: KWRI)")
_ap.add_argument("--model", default=settings.llm_model, help="model id (default: .env)")
_ap.add_argument("--frames", type=int, default=6, help="loop frames (default: 6)")
_ap.add_argument("--step-min", type=int, default=30, help="minutes between frames (default: 30)")
_ap.add_argument("--product", default="geocolor", help="geocolor/visible/infrared/water_vapor")
_ap.add_argument("--max-steps", type=int, default=4)
_ap.add_argument("--max-tokens", type=int, default=6144)
_args = _ap.parse_args()

STATION, MODEL = _args.station.upper(), _args.model
VIDEO = supports_video(MODEL)   # deliver the mp4 to video-capable models; else filmstrip only

messages = [
    {"role": "system", "content": (
        "You are a weather forecaster. You have a get_loop tool that returns a short SATELLITE "
        "LOOP -- a labeled filmstrip (oldest to newest)" + (" and a short video" if VIDEO else "") +
        ". Call it, then READ the frames in time order and describe MOTION and TREND: which way "
        "clouds are moving, and whether cloud/convection is growing, eroding, or steady. State "
        "your answer once, then stop."
    )},
    {"role": "user", "content": (
        f"Retrieve a satellite loop for {STATION} and tell me how the cloud field is EVOLVING "
        "over the loop -- direction of movement and whether it is building or clearing."
    )},
]

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = Path("logs") / f"loop_{STATION}_{MODEL.split('/')[-1].replace(':', '-')}_{stamp}"
out_dir.mkdir(parents=True, exist_ok=True)


def _ext(data: bytes) -> str:
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    return "png"


steps: list[dict] = []
saved: list[str] = []
answer = None
prompt_tok = completion_tok = 0

for n in range(1, _args.max_steps + 1):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[GET_LOOP],
        tool_choice="auto", temperature=0.2, max_tokens=_args.max_tokens,
    )
    prompt_tok += r.usage.prompt_tokens
    completion_tok += r.usage.completion_tokens
    msg = r.choices[0].message
    finish = r.choices[0].finish_reason
    tcs = msg.tool_calls or []
    rec = {"n": n, "finish": finish, "content": (msg.content or "").strip(), "calls": []}

    if not tcs:
        answer = (msg.content or "").strip() or (getattr(msg, "reasoning", "") or "").strip()
        rec["answer"] = answer
        steps.append(rec)
        break

    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in tcs]})
    content: list[dict] = [{"type": "text", "text": "Loop output(s):"}]
    for tc in tcs:
        try:
            a = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            res = tools.ToolResult(f"error: unparseable tool arguments: {e}")
        else:
            res = run_tool(tc.function.name, a)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.text})
        rec["calls"].append({"args": tc.function.arguments, "receipt": res.text,
                             "n_images": len(res.images), "n_videos": len(res.videos)})
        for im in res.images:
            fn = f"step{n}_film_{len(saved)}.{_ext(im)}"
            (out_dir / fn).write_bytes(im)
            saved.append(fn)
            content.append({"type": "text", "text": "[filmstrip: oldest->newest]"})
            b64 = base64.b64encode(im).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{tools._image_mime(im)};base64,{b64}"}})
        if VIDEO:
            for vid in res.videos:
                fn = f"step{n}_loop_{len(saved)}.mp4"
                (out_dir / fn).write_bytes(vid)
                saved.append(fn)
                content.append({"type": "text", "text": "[video loop: oldest->newest]"})
                b64 = base64.b64encode(vid).decode()
                content.append({"type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{b64}"}})
    messages.append({"role": "user", "content": content})
    steps.append(rec)

md = [f"# Agent Satellite Loop (get_loop) -- {STATION}", f"_{datetime.now():%Y-%m-%d %H:%M:%S}_", "",
      f"- **Model:** `{MODEL}` @ {settings.llm_base_url}",
      f"- **Video delivery:** {'ON (video_url)' if VIDEO else 'OFF (filmstrip only)'}",
      f"- **Params:** frames={_args.frames}, step_min={_args.step_min}, product={_args.product}; "
      f"tokens {prompt_tok}+{completion_tok}", ""]
for f in saved:
    md += [f"**{f}**", "", (f"![{f}]({f})" if not f.endswith(".mp4") else f"[{f}]({f})"), ""]
md += ["## Steps"]
for s in steps:
    md += ["", f"### Step {s['n']} (finish: `{s['finish']}`)"]
    for c in s["calls"]:
        md += ["", f"**get_loop({c['args']})** -> {c['n_images']} img, {c['n_videos']} video",
               "", "```text", c["receipt"], "```"]
    if s.get("answer"):
        md += ["", "**Answer:**", "", s["answer"]]
(out_dir / "log.md").write_text("\n".join(md), encoding="utf-8")

print(f"=== get_loop reasoning -- {STATION} ({MODEL}); video={'ON' if VIDEO else 'OFF'} ===")
for s in steps:
    tag = "answered" if s.get("answer") else f"{len(s['calls'])} call(s)"
    print(f"  step {s['n']} [{s['finish']}]: {tag}")
if answer:
    print("\n=== MODEL ANSWER ===\n" + answer)
print(f"\nSaved to {out_dir}")
