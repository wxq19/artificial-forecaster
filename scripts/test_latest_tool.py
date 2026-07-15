"""End-to-end tool-SELECTION test: with two read tools on the menu, does the VLM
pick the right one? The question asks for CURRENT conditions and gives NO time
range, so the correct move is get_latest_obs — not query_obs (which is for an
explicit date/time window). Sibling to test_iem_tool.py, which exercises the
query_obs/range path; keep both so each tool has its own end-to-end check.

Flow: ensure the station is loaded (idempotent IEM pull) -> ask a "what's it
doing now" question with BOTH tools available -> the model emits a tool call ->
we run it against a READ-ONLY connection -> feed the result back -> the model
answers. The whole exchange is written to a self-contained markdown log under
logs/. Watch the "tool call" lines: success = get_latest_obs, not query_obs.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from forecaster import agent, iem, store, tools
from forecaster.config import settings
from forecaster.llm import client

STATION = "KORD"
START, END = datetime(2024, 1, 12), datetime(2024, 1, 14)   # same data as the range test
QUESTION = (
    f"What are the latest reported conditions at {STATION} right now? Use the most "
    "recent observation available and summarize the ceiling, visibility, wind, and "
    "present weather. Do NOT pull a historical time range — I want current "
    "conditions only."
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
            "database. Two tools are available: query_obs (for an explicit date/time "
            "range) and get_latest_obs (for the most recent observation, no range). "
            "Choose the tool that fits the question; reason only over what it "
            "returns. Do not invent observations."
        ),
    },
    {"role": "user", "content": QUESTION},
]

transcript = []          # (label, text) pairs for the log
tool_names = []          # which tools the model actually called — the thing under test
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
        answer, flag = agent.final_answer(msg, finish_reason)
        if flag:
            transcript.append(("harness note", flag))
        transcript.append(("model answer", answer))
        break

    messages.append(msg)                         # assistant turn (carries tool_calls)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        tool_names.append(tc.function.name)
        transcript.append(("tool call", f"{tc.function.name}({json.dumps(args)})"))
        result = tools.run_tool(tc.function.name, args)
        transcript.append(("tool result", result.text))
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result.text}
        )
else:
    answer = "_(hit MAX_TURNS without a final answer)_"
    transcript.append(("model answer", answer))

# Did the model select the tool we expected? This is the assertion of the test.
selection_ok = "get_latest_obs" in tool_names and "query_obs" not in tool_names


def build_markdown() -> str:
    md = [
        f"# Tool-Selection Test — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{settings.llm_model}`  |  **Endpoint:** {settings.llm_base_url}",
        f"- **Ingest:** fetched {summary['fetched']}, parsed {summary['parsed']}, "
        f"newly inserted {summary['inserted']}",
        f"- **Tools called:** {tool_names or '(none)'}",
        f"- **Expected get_latest_obs (and not query_obs):** "
        f"{'PASS ✅' if selection_ok else 'FAIL ❌'}",
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
        f"- tool selection: {'PASS' if selection_ok else 'FAIL'} "
        f"(called {tool_names or '(none)'})",
        f"- finish_reason: `{finish_reason}`{warn}",
        f"- tokens: prompt {prompt_tokens} + completion {completion_tokens} "
        f"(per-call budget MAX_TOKENS={MAX_TOKENS})",
    ]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"latest_tool_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(
    f"\n[tool selection {'PASS' if selection_ok else 'FAIL'} — called {tool_names}]"
)
print(
    f"[finish_reason={finish_reason}, "
    f"completion_tokens={completion_tokens}, MAX_TOKENS={MAX_TOKENS}]"
)
print(f"Full exchange written to {log_path}")
