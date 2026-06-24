"""Quick test: does render() output actually drive useful reasoning from the model?

Renders a station's observations to the text view and feeds it into a single
chat call, asking the model to read the *trend*, not just echo one ob. The full
exchange — every message sent, the model's reasoning, and its answer — is written
to a self-contained markdown log so nothing has to be cross-referenced.
"""

from datetime import datetime
from pathlib import Path

from forecaster.config import settings
from forecaster.llm import client
from forecaster.metar import parse_file, render

STATION = "KMSN"  # convective day: clear -> showers -> thunderstorms -> clearing
TEMPERATURE = 0.2
MAX_TOKENS = 8192  # Qwen3.5 is a reasoning model; budget covers thinking AND answer

block = render(parse_file(f"data/metars/{STATION}.txt"))

messages = [
    {
        "role": "system",
        "content": (
            "You are a weather forecaster. You reason over surface "
            "observations (METARs) provided in the conversation. Use only the data "
            "given; do not invent observations."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Below are recent METAR observations for {STATION}, oldest first.\n\n"
            f"{block}\n\n"
            "1. Summarize how conditions evolved over this period (ceiling, "
            "visibility, wind, present weather).\n"
            "2. State the most recent conditions.\n"
            "3. Note the single most operationally significant event in the period."
        ),
    },
]

r = client.chat.completions.create(
    model=settings.llm_model,
    messages=messages,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)
msg = r.choices[0].message
# Reasoning models put chain-of-thought in a separate field; the answer is in content.
reasoning = getattr(msg, "reasoning", None)
answer = msg.content or "_(empty — check finish_reason)_"


def build_markdown() -> str:
    md = [
        f"# METAR Reasoning Test — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        "## Request",
        f"- **Model:** `{settings.llm_model}`",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        "",
        "## Messages sent",
    ]
    for m in messages:
        # Fenced so the monospaced METAR table keeps its alignment when rendered.
        md += ["", f"### role: {m['role']}", "", "```text", m["content"], "```"]
    if reasoning:
        md += ["", "## Model reasoning", "", reasoning]
    md += ["", "## Model answer", "", answer]
    md += [
        "",
        "## Result",
        f"- finish_reason: `{r.choices[0].finish_reason}`",
        f"- prompt_tokens: {r.usage.prompt_tokens}",
        f"- completion_tokens: {r.usage.completion_tokens}",
    ]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"metar_test_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

# Console: just the answer and a pointer; the full exchange is in the log.
print("=== MODEL ANSWER ===")
print(answer)
print(
    f"\n[finish_reason={r.choices[0].finish_reason}, "
    f"completion_tokens={r.usage.completion_tokens}]"
)
print(f"\nFull exchange written to {log_path}")
