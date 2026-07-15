"""End-to-end: the agent fetches an observed skew-T and REASONS over the image.

Drives the get_sounding tool loop: a natural-language question -> the model calls
get_sounding -> our code fetches a pre-rendered skew-T (SPC GIF by default, or
Wyoming PNG) from the live provider -> the image is fed back to the VLM -> the model
reads the sounding and answers. This exercises the new soundings.py client seam, the
get_sounding tool, and the GIF/PNG image-return path end to end.

Run with --fetch-only FIRST to fetch + cache the exact image the model will read and
print its path (review it yourself), then run without the flag to send it to the VLM.
Because the published synoptic image is byte-stable, the live fetch in the loop is the
same image you reviewed. The full exchange -- the model's reasoning and its answer --
is written to a self-contained markdown log under logs/.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from forecaster import agent, soundings, tools
from forecaster.config import settings
from forecaster.llm import client
from forecaster.agent import final_answer
from forecaster.tools import GET_SOUNDING, run_tool

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--site", default="MPX", help="upper-air site id in the provider's namespace (default: MPX)")
_ap.add_argument("--source", default="spc", choices=["spc", "wyoming"],
                 help="sounding provider (default: spc -- richer analysis)")
_ap.add_argument("--fetch-only", action="store_true",
                 help="fetch + cache the image and print its path for review, then exit (no VLM)")
_ap.add_argument("--model", default=settings.llm_model, help="model id override (default: .env)")
_ap.add_argument("--max-steps", type=int, default=4, help="max model turns (default: 4)")
_ap.add_argument("--max-tokens", type=int, default=12288,
                 help="completion budget per turn -- Qwen ruminates; leave room (default: 12288)")
_args = _ap.parse_args()

SITE = _args.site.upper()
SOURCE = _args.source
MODEL = _args.model
MAX_STEPS = _args.max_steps
TEMPERATURE = 0.2
MAX_TOKENS = _args.max_tokens
T = soundings.synoptic_time()   # the run the loop will fetch (00Z/12Z at or before now)


# --- Review path: fetch + cache the exact image, print where it landed, stop ---
if _args.fetch_only:
    soundings.fetch_skewt(SITE, T, source=SOURCE, use_cache=True)
    path = soundings.cache_path(SITE, T, source=SOURCE)
    print(f"fetched {SOURCE} skew-T for {SITE} at {T:%Y-%m-%dT%H:%MZ}")
    print(f"URL:   {soundings.skewt_url(SITE, T, source=SOURCE)}")
    print(f"image: {path}")
    raise SystemExit(0)


# --- Reason-over-the-sounding tool loop ---
messages = [
    {"role": "system", "content": (
        "You are a weather forecaster. To see the vertical structure of the atmosphere, "
        "call get_sounding to retrieve an observed skew-T sounding, then read the image "
        "and answer. Base your answer only on what the sounding shows. State your answer "
        "once and then stop; do not repeat or re-derive it."
    )},
    {"role": "user", "content": (
        f"Retrieve the latest observed skew-T sounding for upper-air site {SITE} and "
        "analyze it.\n"
        "1. Describe the temperature and dewpoint profile: where is the air moist vs dry?\n"
        "2. Identify any inversions and the approximate freezing level.\n"
        "3. Assess stability: is the profile favorable for convection, and how does the "
        "wind change with height?"
    )},
]

steps: list[dict] = []
answer = recovery = None
prompt_tok = completion_tok = 0

for n in range(1, MAX_STEPS + 1):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[GET_SOUNDING],
        tool_choice="auto", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    prompt_tok += r.usage.prompt_tokens
    completion_tok += r.usage.completion_tokens
    msg = r.choices[0].message
    finish = r.choices[0].finish_reason
    tcs = msg.tool_calls or []
    rec: dict = {"n": n, "content": (msg.content or "").strip(),
                 "reasoning": (getattr(msg, "reasoning", None) or "").strip(), "finish": finish}

    if not tcs:                                   # model answered (no more tool calls)
        answer, recovery = final_answer(msg, finish)
        rec["answer"] = answer
        steps.append(rec)
        break

    # Echo the assistant's tool_calls, then answer each with a receipt (+ the image).
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in tcs]})
    rec["calls"] = []
    for tc in tcs:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            res = tools.ToolResult(f"error: unparseable tool arguments: {e}")
        else:
            res = run_tool(tc.function.name, args)
        rec["calls"].append({"name": tc.function.name, "args": tc.function.arguments,
                             "receipt": res.text, "n_images": len(res.images)})
        messages += agent.tool_messages(tc.id, res)
    steps.append(rec)


# --- Markdown log ---
def build_markdown() -> str:
    md = [
        f"# Agent Skew-T Reasoning (get_sounding) — {SITE}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{MODEL}` @ {settings.llm_base_url}",
        f"- **Sounding:** {SOURCE} {SITE} at {T:%Y-%m-%dT%H:%MZ}",
        f"- **Source URL:** {soundings.skewt_url(SITE, T, source=SOURCE)}",
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
        for c in s.get("calls", []):
            md += ["", f"**Tool call:** `{c['name']}({c['args']})` -> {c['n_images']} image(s)",
                   "", "```text", c["receipt"], "```"]
        if s.get("answer"):
            md += ["", "**Answer:**", "", s["answer"]]
    if recovery:
        md += ["", f"> harness note: {recovery}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")
log_path = log_dir / f"sounding_{SITE}_{model_slug}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print(f"=== get_sounding — {SITE} ({SOURCE}, {T:%Y-%m-%dT%H:%MZ}) ===")
for s in steps:
    tag = "answered" if s.get("answer") else f"{len(s.get('calls', []))} tool call(s)"
    print(f"  step {s['n']}: {tag}")
if answer:
    print("\n=== MODEL ANSWER ===")
    print(answer)
    if recovery:
        print(f"\n[harness note: {recovery}]")
print(f"\nFull exchange written to {log_path}")
