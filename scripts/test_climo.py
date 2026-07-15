"""Agent-tool test: does the VLM reach for get_climo (not query_obs) when asked what
is TYPICAL, and does it cite the climatology values correctly?

Prereq: the KLSV July climo must already be built --
    uv run python scripts/build_climo.py --station KLSV --months 7
Building is a multi-minute network job, so this driver does NOT build; it checks the
product is present and asks the model a "what is typical" question with the full tool
menu available. The whole exchange is written to a markdown log under logs/.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from forecaster import agent, store, tools
from forecaster.config import settings
from forecaster.llm import client

STATION = "KLSV"
MONTH = 7
QUESTION = (
    f"What is TYPICAL July weather at {STATION} (Nellis AFB)? Give the normal daily "
    "high and low with their percentile spread, the prevailing wind direction and how "
    "it shifts through the day, and the thunderstorm/fog risk and when it peaks. Use the "
    "climatology, not current conditions. State it once and stop."
)
TEMPERATURE = 0.2
MAX_TOKENS = 12288
MAX_TURNS = 5

# 1) confirm the product exists (this driver does not build -- see module docstring)
built = False
if os.path.exists(settings.db_path):
    con = store.connect(settings.db_path, read_only=True)
    try:
        built = store.climo_month(con, STATION, MONTH) is not None
    except Exception:                                       # noqa: BLE001 -- no climo tables yet
        built = False
    con.close()
if not built:
    raise SystemExit(
        f"No {STATION} month {MONTH} climatology in {settings.db_path}. Build it first:\n"
        f"  uv run python scripts/build_climo.py --station {STATION} --months {MONTH}"
    )

# 2) run the agent loop
messages = [
    {
        "role": "system",
        "content": (
            "You are a weather forecaster with tools for current observations and for "
            "CLIMATOLOGY (typical conditions). For questions about what is NORMAL or "
            "TYPICAL, call get_climo. Reason only over what the tools return; do not "
            "invent values."
        ),
    },
    {"role": "user", "content": QUESTION},
]

transcript = []
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
        answer, flag = agent.final_answer(msg, finish_reason)
        if flag:
            transcript.append(("note", flag))
        transcript.append(("model answer", answer))
        break
    messages.append(msg)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        transcript.append(("tool call", f"{tc.function.name}({json.dumps(args)})"))
        result = tools.run_tool(tc.function.name, args)
        transcript.append(("tool result", result.text))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.text})
else:
    transcript.append(("model answer", "_(hit MAX_TURNS without a final answer)_"))


def build_markdown() -> str:
    md = [
        f"# Climatology Tool Test -- {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{settings.llm_model}`  |  **Endpoint:** {settings.llm_base_url}",
        "",
        "## Question",
        "",
        QUESTION,
    ]
    for label, text in transcript:
        fence = "```text" if label in ("tool result", "tool call") else ""
        md += ["", f"## {label}", ""]
        md += [fence, text, "```"] if fence else [text]
    md += [
        "",
        "## Result",
        "",
        f"- finish_reason: `{finish_reason}`",
        f"- tokens: prompt {prompt_tokens} + completion {completion_tokens} "
        f"(MAX_TOKENS={MAX_TOKENS})",
    ]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"climo_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("\n=== MODEL ANSWER ===")
print(transcript[-1][1])
print(f"\n[finish_reason={finish_reason}, completion_tokens={completion_tokens}]")
print(f"Full exchange written to {log_path}")
