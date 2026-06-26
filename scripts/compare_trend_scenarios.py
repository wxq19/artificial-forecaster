"""Compare three reasoning conditions on the SAME trend question at KORD:
  A  hinted prompt + full tools     -> expected: get_trend (meteogram image)
  B  neutral prompt + OBS-ONLY tools -> text obs only (no image possible)
  C  neutral prompt + full tools     -> does it CHOOSE get_trend unprompted?
Isolates (A vs B) image-vs-text reasoning and (C) whether the model picks the
chart tool on its own. Same question/model/temp across all three. Writes one
combined markdown log under logs/.
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

SYS_HINT = (
    "You are a weather forecaster with tools over a surface-observation database. "
    "To assess a TREND over recent hours, call get_trend — it returns a meteogram "
    "image you should read directly. Reason only over what the tools return; do "
    "not invent observations. State your answer once and stop."
)
SYS_NEUTRAL = (
    "You are a weather forecaster with tools over a surface-observation database. "
    "Use whatever tools you need to answer. Reason only over what the tools "
    "return; do not invent observations. State your answer once and stop."
)

# ensure the station is loaded (idempotent; keeps the demo off IEM if present)
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


def run_scenario(tag: str, system: str, toolset: list) -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": QUESTION},
    ]
    tool_names, saved, transcript = [], [], []
    finish, ptok, ctok = None, 0, 0
    for _turn in range(MAX_TURNS):
        r = client.chat.completions.create(
            model=settings.llm_model, messages=messages, tools=toolset,
            tool_choice="auto", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        if r.usage:
            ptok += r.usage.prompt_tokens
            ctok += r.usage.completion_tokens
        finish = r.choices[0].finish_reason
        msg = r.choices[0].message
        if reasoning := getattr(msg, "reasoning", None):
            transcript.append(("model reasoning", reasoning))
        if not msg.tool_calls:
            answer, flag = tools.final_answer(msg, finish)
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
            out = tools.tool_messages(tc.id, result)
            messages.append(out[0])
            image_msgs.extend(out[1:])
            for png in result.images:
                p = chart_dir / f"cmp_{tag}_{stamp}_{len(saved)}.png"
                p.write_bytes(png)
                saved.append(p)
                transcript.append(("tool image", f"![chart](../{p})"))
        messages.extend(image_msgs)
    else:
        transcript.append(("model answer", "_(hit MAX_TURNS)_"))
    return {
        "tag": tag, "answer": transcript[-1][1], "tools": tool_names,
        "finish": finish, "ptok": ptok, "ctok": ctok, "images": len(saved),
        "transcript": transcript,
    }


SCENARIOS = [
    ("A_hint_fulltools", SYS_HINT, tools.TOOLS),
    ("B_neutral_obsonly", SYS_NEUTRAL, [tools.QUERY_OBS, tools.GET_LATEST]),
    ("C_neutral_fulltools", SYS_NEUTRAL, tools.TOOLS),
]
results = [run_scenario(*s) for s in SCENARIOS]

for res in results:
    print(f"\n========== {res['tag']} ==========")
    print(
        f"tools={res['tools']} | images={res['images']} | "
        f"finish={res['finish']} | ctok={res['ctok']}"
    )
    print(res["answer"])

md = [f"# Trend-Scenario Comparison — {STATION}", f"_{datetime.now():%Y-%m-%d %H:%M:%S}_", ""]
md += [f"- **Model:** `{settings.llm_model}`", f"- **Question:** {QUESTION}", ""]
md += ["## Summary", "", "| scenario | tools called | images | finish | compl tok |",
       "|---|---|---|---|---|"]
for res in results:
    md.append(
        f"| {res['tag']} | {res['tools']} | {res['images']} | "
        f"`{res['finish']}` | {res['ctok']} |"
    )
for res in results:
    md += ["", f"## {res['tag']}", ""]
    for label, text in res["transcript"]:
        if label == "tool image":
            md += [f"### {label}", "", text, ""]
        else:
            fence = "```text" if label in ("tool result", "tool call") else ""
            md += [f"### {label}", ""]
            md += [fence, text, "```", ""] if fence else [text, ""]
log_path = log_dir / f"trend_compare_{STATION}_{stamp}.md"
log_path.write_text("\n".join(md), encoding="utf-8")
print(f"\nCombined log -> {log_path}")
