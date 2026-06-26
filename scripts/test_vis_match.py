"""End-to-end MULTI-tool reasoning test: a question that can't be answered with a
single call, where the second call DEPENDS on the first.

"Over the previous 24 hours, which times had the same visibility as right now?"
forces a chain:
  1. get_latest_obs  -> learn the current visibility AND the current timestamp
  2. derive a 24h window ending at that timestamp
  3. query_obs over that window
  4. filter the returned obs down to those whose visibility matches step 1

We don't script the steps — we hand the model both tools and watch whether it
sequences them correctly and reasons over the result. Sibling to
test_latest_tool.py (single-tool selection) and test_iem_tool.py (range only).
The whole exchange is written to a markdown log under logs/.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from forecaster import iem, store, tools
from forecaster.config import settings
from forecaster.llm import client

STATION = "KORD"
START, END = datetime(2024, 1, 12), datetime(2024, 1, 14)   # same snowstorm data
QUESTION = (
    f"First find the current (most recent) observation at {STATION} and note its "
    "visibility. Then, looking only at the 24 hours BEFORE that observation, list "
    "every observation time whose visibility equals the current visibility. For "
    "each match give the time (UTC) and the visibility. End with a count of how "
    "many matches you found. Once you have the matches, state them ONCE and stop — "
    "do not re-derive, recount, or second-guess your list."
)
TEMPERATURE = 0.2
MAX_TOKENS = 16384         # reasoning model: thinking AND answer share this budget
MAX_TURNS = 6             # safety cap — this question needs at least two tool turns

# 1) Ensure the data is present locally. Only hit IEM if this station isn't
#    loaded yet — keeps the run off the network (and off IEM's rate limiter); the
#    tools query the local DB regardless of how it got there.
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
            "database. Two tools are available: get_latest_obs (the most recent "
            "observation, no time range) and query_obs (observations over an "
            "explicit UTC date/time range). Some questions need both: use "
            "get_latest_obs to anchor 'now', then query_obs for the surrounding "
            "history. Reason only over what the tools return; do not invent "
            "observations. Visibility is the vis column in statute miles (e.g. 1.5SM)."
        ),
    },
    {"role": "user", "content": QUESTION},
]

transcript = []          # (label, text) pairs for the log
tool_names = []          # ordered list of tools the model called — under test
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
        tool_names.append(tc.function.name)
        transcript.append(("tool call", f"{tc.function.name}({json.dumps(args)})"))
        result = tools.run_tool(tc.function.name, args)
        transcript.append(("tool result", result))
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )
else:
    answer = "_(hit MAX_TURNS without a final answer)_"
    transcript.append(("model answer", answer))

# Expected chain: get_latest_obs THEN query_obs. We check both were called and
# that the latest call came first (the query range depends on it). This is a
# heuristic, not a correctness oracle for the matches themselves.
used_both = "get_latest_obs" in tool_names and "query_obs" in tool_names
right_order = used_both and tool_names.index("get_latest_obs") < tool_names.index(
    "query_obs"
)


def build_markdown() -> str:
    md = [
        f"# Multi-Tool Reasoning Test — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{settings.llm_model}`  |  **Endpoint:** {settings.llm_base_url}",
        f"- **Ingest:** fetched {summary['fetched']}, parsed {summary['parsed']}, "
        f"newly inserted {summary['inserted']}",
        f"- **Tools called (in order):** {tool_names or '(none)'}",
        f"- **Used both tools:** {'yes' if used_both else 'no'}  |  "
        f"**get_latest before query_obs:** {'yes ✅' if right_order else 'no ❌'}",
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
        f"- tool chain: {tool_names or '(none)'}",
        f"- used both tools: {used_both}  |  correct order: {right_order}",
        f"- finish_reason: `{finish_reason}`{warn}",
        f"- tokens: prompt {prompt_tokens} + completion {completion_tokens} "
        f"(per-call budget MAX_TOKENS={MAX_TOKENS})",
    ]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"vis_match_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(f"\n[tool chain: {tool_names}]")
print(f"[used both: {used_both}, correct order: {right_order}]")
print(
    f"[finish_reason={finish_reason}, "
    f"completion_tokens={completion_tokens}, MAX_TOKENS={MAX_TOKENS}]"
)
print(f"Full exchange written to {log_path}")
