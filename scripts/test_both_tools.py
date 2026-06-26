"""Scenario D: steer the model to use BOTH data forms on the same KORD trend
question — call get_trend for the meteogram IMAGE *and* the obs tools for the raw
METAR TEXT — then synthesize. Tests whether combining shape (chart) with exact
values (text) beats either alone (image missed wind direction + BLSN + exact
numbers; text missed the visual gestalt). Same question/model/temp as the A/B/C
comparison. Writes a markdown log (with the embedded PNG) under logs/.
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
MAX_TURNS = 6

SYSTEM = (
    "You are a weather forecaster with tools over a surface-observation database. "
    "For a trend question use BOTH: call get_trend to see the meteogram image, AND "
    "query the underlying observations (get_latest_obs / query_obs) to read exact "
    "values, wind direction, and present-weather details. Synthesize the visual "
    "trend with the precise numbers. Reason only over what the tools return; do "
    "not invent observations. State your answer once and stop."
)

# ensure the station is loaded (idempotent)
have = 0
if os.path.exists(settings.db_path):
    con = store.connect(settings.db_path, read_only=True)
    have = store.count(con, STATION)
    con.close()
if not have:
    iem.load(STATION, START, END)
print(f"INGEST: {have or 'loaded'} {STATION} obs in DB")

chart_dir = Path("data/charts/temp")
chart_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": QUESTION},
]
transcript, tool_names, saved_images = [], [], []
seen_windows, last_note = [], None       # Fix 3: conversation-wide window tracking
prompt_tokens = completion_tokens = 0
finish_reason = None

for _turn in range(MAX_TURNS):
    r = client.chat.completions.create(
        model=settings.llm_model, messages=messages, tools=tools.TOOLS,
        tool_choice="auto", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
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
    messages.append(msg)
    image_msgs = []
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        tool_names.append(tc.function.name)
        transcript.append(("tool call", f"{tc.function.name}({json.dumps(args)})"))
        result = tools.run_tool(tc.function.name, args)
        transcript.append(("tool result", result.text))
        if result.window:
            seen_windows.append((tc.function.name, result.window))
        out = tools.tool_messages(tc.id, result)
        messages.append(out[0])
        image_msgs.extend(out[1:])
        for png in result.images:
            p = chart_dir / f"both_{stamp}_{len(saved_images)}.png"
            p.write_bytes(png)
            saved_images.append(p)
            transcript.append(("tool image", f"![chart](../{p})"))
    messages.extend(image_msgs)
    note = tools.window_conflict(seen_windows)
    if note and note != last_note:
        messages.append({"role": "user", "content": note})
        transcript.append(("window check", note))
        last_note = note
else:
    transcript.append(("model answer", "_(hit MAX_TURNS)_"))

distinct = sorted(set(tool_names))
used_both = "get_trend" in tool_names and any(
    t in tool_names for t in ("query_obs", "get_latest_obs")
)

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(f"\n[tools called (in order): {tool_names}]")
print(f"[used BOTH chart+obs: {'YES' if used_both else 'NO'} | distinct={distinct}]")
print(
    f"[images={len(saved_images)} | finish={finish_reason} | "
    f"compl tok={completion_tokens}]"
)

md = [f"# Both-Tools (Scenario D) — {STATION}", f"_{datetime.now():%Y-%m-%d %H:%M:%S}_", ""]
md += [
    f"- **Model:** `{settings.llm_model}`",
    f"- **Tools called (in order):** {tool_names}",
    f"- **Used both chart + obs:** {'YES' if used_both else 'NO'}",
    f"- **Images returned:** {len(saved_images)} | finish: `{finish_reason}` | "
    f"compl tok: {completion_tokens}",
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
log_path = log_dir / f"both_tools_{STATION}_{stamp}.md"
log_path.write_text("\n".join(md), encoding="utf-8")
print(f"Full exchange -> {log_path}")
