"""Fix 3 demonstration: deliberately push the model toward ABSOLUTE query_obs
windows so its window desyncs from get_trend (which anchors on the latest ob),
then let the harness window_conflict note fire. We do NOT tell the model to fix
the mismatch — the point is to see whether the advisory note alone makes it
realign (e.g. re-query with hours=) or whether it just carries on. Writes a
markdown log (with embedded PNG) under logs/.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from forecaster import agent, iem, store, tools
from forecaster.config import settings
from forecaster.llm import client

STATION = "KORD"
START, END = datetime(2024, 1, 12), datetime(2024, 1, 14)
QUESTION = (
    f"Assess the last 24 hours at {STATION}: are conditions improving, "
    "deteriorating, or steady? Give a one-line persistence outlook."
)
TEMPERATURE = 0.2
MAX_TOKENS = 16384
MAX_TURNS = 6

SYSTEM = (
    "You are a weather forecaster with tools over a surface-observation database. "
    "To answer, call get_trend for the recent 24-hour meteogram, AND call query_obs "
    "with explicit absolute start='2024-01-12T00:00' and end='2024-01-12T12:00' to "
    "inspect the earlier part of the event (do NOT use the hours parameter for "
    "query_obs). Then synthesize both. Reason only over what the tools return; do "
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
note_fired = False
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
        answer, flag = agent.final_answer(msg, finish_reason)
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
        out = agent.tool_messages(tc.id, result)
        messages.append(out[0])
        image_msgs.extend(out[1:])
        for png in result.images:
            p = chart_dir / f"guard_{stamp}_{len(saved_images)}.png"
            p.write_bytes(png)
            saved_images.append(p)
            transcript.append(("tool image", f"![chart](../{p})"))
    messages.extend(image_msgs)
    note = agent.window_conflict(seen_windows)
    if note and note != last_note:
        note_fired = True
        messages.append({"role": "user", "content": note})
        transcript.append(("window check", note))
        last_note = note
else:
    transcript.append(("model answer", "_(hit MAX_TURNS)_"))

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(f"\n[tools called (in order): {tool_names}]")
print(f"[window note fired: {'YES' if note_fired else 'NO'}]")
print(f"[distinct windows seen: {len({w for _, w in seen_windows})}]")
print(f"[finish={finish_reason} | compl tok={completion_tokens}]")

md = [f"# Window-Guard (Fix 3) Test — {STATION}", f"_{datetime.now():%Y-%m-%d %H:%M:%S}_", ""]
md += [
    f"- **Model:** `{settings.llm_model}`",
    f"- **Tools called (in order):** {tool_names}",
    f"- **Window note fired:** {'YES' if note_fired else 'NO'}",
    f"- **Distinct windows seen:** {len({w for _, w in seen_windows})}",
    f"- **finish:** `{finish_reason}` | compl tok: {completion_tokens}",
    "",
    "## Question",
    "",
    QUESTION,
]
for label, text in transcript:
    if label == "tool image":
        md += ["", f"## {label}", "", text]
    else:
        fence = "```text" if label in ("tool result", "tool call", "window check") else ""
        md += ["", f"## {label}", ""]
        md += [fence, text, "```"] if fence else [text]
log_path = log_dir / f"window_guard_{STATION}_{stamp}.md"
log_path.write_text("\n".join(md), encoding="utf-8")
print(f"Full exchange -> {log_path}")
