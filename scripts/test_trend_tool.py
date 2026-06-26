"""End-to-end test of the IMAGE-returning tool path: ask a trend question, the
model calls get_trend, our code renders a meteogram (PNG) and feeds it BACK into
the conversation as a base64 image (via tools.tool_messages), and the model
reasons over the chart it can now see. This is the plumbing every future chart
tool needs. Writes a markdown log under logs/ that embeds the rendered PNG (saved
to data/charts/temp/) so you can see exactly what the model saw.

Image ordering note: when an assistant turn has tool_calls, EVERY tool_call must
be answered by its role='tool' receipt before any other message. So we append all
receipts first and DEFER the image-bearing user message(s) until after the loop
over tool_calls — otherwise a parallel call would interleave a user message
between two receipts, which the API rejects.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from forecaster import iem, store, tools
from forecaster.config import settings
from forecaster.llm import client

STATION = "KORD"
START, END = datetime(2024, 1, 12), datetime(2024, 1, 14)
QUESTION = (
    f"Look at how conditions have evolved over the last 24 hours at {STATION}. "
    "Are things improving, deteriorating, or steady? Walk through temperature/"
    "dewpoint, wind, visibility, ceiling, pressure tendency, and present weather, "
    "then give a one-line persistence outlook for the next few hours."
)
TEMPERATURE = 0.2
MAX_TOKENS = 16384
MAX_TURNS = 5

# 1) ensure the station is loaded (idempotent; keeps the demo off IEM if present)
have = 0
if os.path.exists(settings.db_path):
    con = store.connect(settings.db_path, read_only=True)
    have = store.count(con, STATION)
    con.close()
if have:
    print(f"INGEST: skipped — {have} {STATION} obs already in DB")
    summary = {"station": STATION, "fetched": 0, "parsed": 0, "inserted": 0}
else:
    summary = iem.load(STATION, START, END)
    print("INGEST:", {k: summary[k] for k in ("station", "fetched", "parsed", "inserted")})

# 2) run the agent loop
messages = [
    {
        "role": "system",
        "content": (
            "You are a weather forecaster with tools over a surface-observation "
            "database. To assess a TREND over recent hours, call get_trend — it "
            "returns a meteogram image you should read directly. Reason only over "
            "what the tools return; do not invent observations. State your answer "
            "once and stop."
        ),
    },
    {"role": "user", "content": QUESTION},
]

chart_dir = Path("data/charts/temp")
chart_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

transcript = []          # (label, text) pairs for the log
tool_names = []          # which tools the model called — the thing under test
saved_images = []        # persisted PNG paths (for the markdown log)
prompt_tokens = completion_tokens = 0
finish_reason = None

for _turn in range(MAX_TURNS):
    r = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        tools=tools.TOOLS,
        tool_choice="auto",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    if r.usage:
        prompt_tokens += r.usage.prompt_tokens
        completion_tokens += r.usage.completion_tokens
    finish_reason = r.choices[0].finish_reason
    msg = r.choices[0].message
    if reasoning := getattr(msg, "reasoning", None):
        transcript.append(("model reasoning", reasoning))

    if not msg.tool_calls:
        answer, flag = tools.final_answer(msg, finish_reason)
        if flag:
            transcript.append(("harness note", flag))
        transcript.append(("model answer", answer))
        break

    messages.append(msg)                     # assistant turn (carries tool_calls)
    image_msgs = []                          # defer until AFTER all tool receipts
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        tool_names.append(tc.function.name)
        transcript.append(("tool call", f"{tc.function.name}({json.dumps(args)})"))
        result = tools.run_tool(tc.function.name, args)
        transcript.append(("tool result", result.text))
        out = tools.tool_messages(tc.id, result)
        messages.append(out[0])              # tool receipt first
        image_msgs.extend(out[1:])           # image-bearing user message deferred
        for png in result.images:            # persist + log what the model saw
            p = chart_dir / f"trend_{tc.function.name}_{stamp}_{len(saved_images)}.png"
            p.write_bytes(png)
            saved_images.append(p)
            transcript.append(("tool image", f"![chart](../{p})"))
    messages.extend(image_msgs)
else:
    transcript.append(("model answer", "_(hit MAX_TURNS without a final answer)_"))

used_trend = "get_trend" in tool_names


def build_markdown() -> str:
    md = [
        f"# Trend-Tool (image) Test — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{settings.llm_model}`  |  **Endpoint:** {settings.llm_base_url}",
        f"- **Tools called:** {tool_names or '(none)'}",
        f"- **Called get_trend:** {'PASS' if used_trend else 'FAIL'}",
        f"- **Images returned to model:** {len(saved_images)}",
        "",
        "## Question",
        "",
        QUESTION,
    ]
    for label, text in transcript:
        if label == "tool image":
            md += ["", f"## {label}", "", text]
        else:
            fence = "```text" if label in ("tool result", "tool call") else ""
            md += ["", f"## {label}", ""]
            md += [fence, text, "```"] if fence else [text]
    md += [
        "",
        "## Result",
        "",
        f"- get_trend used: {'PASS' if used_trend else 'FAIL'} "
        f"(called {tool_names or '(none)'})",
        f"- finish_reason: `{finish_reason}`",
        f"- tokens: prompt {prompt_tokens} + completion {completion_tokens}",
    ]
    return "\n".join(md)


log_path = log_dir / f"trend_tool_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(f"\n[get_trend {'PASS' if used_trend else 'FAIL'} — called {tool_names}]")
print(
    f"[images returned: {len(saved_images)}; finish_reason={finish_reason}; "
    f"completion_tokens={completion_tokens}]"
)
print(f"Full exchange written to {log_path}")
