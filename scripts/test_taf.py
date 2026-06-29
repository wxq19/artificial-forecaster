"""End-to-end test: live AWC TAFs -> taf.parse -> render -> model reasoning.

Pulls CURRENT TAFs straight from aviationweather.gov for a military, a US
civilian, and an international station (exercising QNH/meters, statute miles, and
CAVOK/PROB in one run), decodes each through the taf seam, and feeds one rendered
TAF into a single chat call so we can see the model reason over a forecast. The
full exchange — every raw TAF, its decoded view, and the model's answer — is
written to a self-contained markdown log under logs/.
"""

from datetime import datetime
from pathlib import Path

from forecaster import awc, tafparse as taf
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import final_answer

STATIONS = ["KADW", "KORD", "EGLL"]  # military (QNH/m) | US civilian (SM/FM) | intl (CAVOK/PROB)
FOCUS = "KADW"                        # the TAF fed to the model
TEMPERATURE = 0.2
MAX_TOKENS = 12288                    # Qwen3.5 ruminates; budget covers thinking AND the answer


def fetch_decoded() -> dict[str, dict]:
    """Live-fetch + decode each station. Network/parse failures are recorded, not
    fatal, so one bad station doesn't sink the run."""
    out: dict[str, dict] = {}
    for st in STATIONS:
        try:
            issue, raw = awc.fetch_taf(st)[0]
            out[st] = {"issue": issue, "raw": raw, "rendered": taf.render(taf.parse(raw))}
        except Exception as e:  # noqa: BLE001 — record and continue
            out[st] = {"error": f"{type(e).__name__}: {e}"}
    return out


decoded = fetch_decoded()
focus = decoded[FOCUS]
block = focus.get("rendered", f"(fetch failed: {focus.get('error')})")

messages = [
    {
        "role": "system",
        "content": (
            "You are a weather forecaster. You reason over terminal aerodrome "
            "forecasts (TAFs) provided in the conversation. Use only the data "
            "given; do not invent groups or values. State your answer once and "
            "then stop; do not repeat or re-derive it."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Below is the current TAF for {FOCUS}, decoded, with the raw line beneath.\n\n"
            f"{block}\n\n"
            "1. Summarize how the forecast evolves (wind, visibility, ceiling, weather).\n"
            "2. Identify the period of worst expected flying conditions and why.\n"
            "3. State whether any thunderstorms are forecast, and in which periods."
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
reasoning = getattr(msg, "reasoning", None)
# Recover an answer stranded in the reasoning field / flag a token-cap miss.
answer, recovery = final_answer(msg, r.choices[0].finish_reason)


def build_markdown() -> str:
    md = [
        f"# TAF Reasoning Test — {FOCUS}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        "## Live TAFs (decoded)",
    ]
    for st in STATIONS:
        d = decoded[st]
        md += ["", f"### {st}"]
        if "error" in d:
            md += [f"- fetch/parse failed: `{d['error']}`"]
            continue
        # Fenced so the monospaced decoded table keeps its alignment when rendered.
        md += ["", "```text", d["rendered"], "```"]
    md += [
        "",
        "## Request",
        f"- **Model:** `{settings.llm_model}`",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        "",
        "## Messages sent",
    ]
    for m in messages:
        md += ["", f"### role: {m['role']}", "", "```text", m["content"], "```"]
    if reasoning:
        md += ["", "## Model reasoning", "", reasoning]
    md += ["", "## Model answer", "", answer]
    if recovery:
        md += ["", f"> harness note: {recovery}"]
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
log_path = log_dir / f"taf_test_{FOCUS}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("=== MODEL ANSWER ===")
print(answer)
if recovery:
    print(f"\n[harness note: {recovery}]")
print(
    f"\n[finish_reason={r.choices[0].finish_reason}, "
    f"completion_tokens={r.usage.completion_tokens}]"
)
print(f"\nFull exchange written to {log_path}")
