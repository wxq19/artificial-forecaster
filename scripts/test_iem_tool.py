"""End-to-end agent-tool test: can the VLM answer a real forecasting question by
CALLING our DuckDB query tool, rather than being handed the data?

Flow: ingest a station/period from IEM (idempotent) -> ask the model a question
with the query_obs tool available -> the model emits a tool call -> we run it
against a READ-ONLY connection -> feed the result back -> the model answers. The
whole exchange (every tool call + result + the model's reasoning) is written to a
self-contained markdown log under logs/.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from forecaster import iem, store, tools
from forecaster.config import settings
from forecaster.llm import client

STATION = "KORD"
START, END = datetime(2024, 1, 12), datetime(2024, 1, 14)   # the Jan 2024 snowstorm
QUESTION = (
    f"What was the worst weather at {STATION} between 2024-01-12 and 2024-01-13 "
    "(UTC)? Identify the single worst period, give the conditions then (ceiling, "
    "visibility, wind, present weather), and explain what made it the worst. "
    "Tie-break rule: if two periods are close, rank by lowest ceiling, then lowest "
    "visibility, then highest wind. Pick ONE and commit — do not re-compare at length."
)
TEMPERATURE = 0.2
MAX_TOKENS = 16384         # reasoning model: thinking AND answer share this budget
MAX_TURNS = 5             # safety cap on the tool loop

# 1) Ensure the data is present locally. Only hit IEM if this station isn't
#    loaded yet — keeps the demo off the network (and off IEM's rate limiter) on
#    every run; the tool queries the local DB regardless of how it got there.
have = 0
if os.path.exists(settings.db_path):
    con = store.connect(settings.db_path, read_only=True)
    have = store.count(con, STATION)
    con.close()

if have:
    print(f"INGEST: skipped — {have} {STATION} obs already in DB")
    summary = {"station": STATION, "fetched": 0, "parsed": 0, "inserted": 0, "errors": []}
else:
    summary = iem.load(STATION, START, END)
    print("INGEST:", {k: summary[k] for k in ("station", "fetched", "parsed", "inserted")})

# 2) run the agent loop
messages = [
    {
        "role": "system",
        "content": (
            "You are a weather forecaster with access to a surface-observation "
            "database via the query_obs tool. Use the tool to retrieve real METARs "
            "for the airport and time range in question; reason only over what it "
            "returns. Do not invent observations."
        ),
    },
    {"role": "user", "content": QUESTION},
]

transcript = []          # (label, text) pairs for the log
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
    if r.usage:                              # accumulate across every turn in the loop
        prompt_tokens += r.usage.prompt_tokens
        completion_tokens += r.usage.completion_tokens
    finish_reason = r.choices[0].finish_reason   # the final turn's is what matters
    msg = r.choices[0].message
    if reasoning := getattr(msg, "reasoning", None):
        transcript.append(("model reasoning", reasoning))

    if not msg.tool_calls:
        answer = msg.content or "_(empty — check finish_reason)_"
        transcript.append(("model answer", answer))
        break

    messages.append(msg)                         # assistant turn (carries tool_calls)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        transcript.append(("tool call", f"{tc.function.name}({json.dumps(args)})"))
        result = tools.run_tool(tc.function.name, args)
        transcript.append(("tool result", result))
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )
else:
    answer = "_(hit MAX_TURNS without a final answer)_"
    transcript.append(("model answer", answer))


def build_markdown() -> str:
    md = [
        f"# Agent Tool Test — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{settings.llm_model}`  |  **Endpoint:** {settings.llm_base_url}",
        f"- **Ingest:** fetched {summary['fetched']}, parsed {summary['parsed']}, "
        f"newly inserted {summary['inserted']}",
        "",
        "## Question",
        "",
        QUESTION,
    ]
    for label, text in transcript:
        fence = "```text" if label in ("tool result", "tool call") else ""
        md += ["", f"## {label}", ""]
        md += [fence, text, "```"] if fence else [text]

    warn = (
        "  -- ran out of tokens mid-generation; reasoning ate the whole budget "
        "(raise MAX_TOKENS and/or tighten the prompt)"
        if finish_reason == "length"
        else ""
    )
    md += [
        "",
        "## Result",
        "",
        f"- finish_reason: `{finish_reason}`{warn}",
        f"- tokens: prompt {prompt_tokens} + completion {completion_tokens} "
        f"(per-call budget MAX_TOKENS={MAX_TOKENS})",
    ]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"iem_tool_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(
    f"\n[finish_reason={finish_reason}, "
    f"completion_tokens={completion_tokens}, MAX_TOKENS={MAX_TOKENS}]"
)
print(f"Full exchange written to {log_path}")
