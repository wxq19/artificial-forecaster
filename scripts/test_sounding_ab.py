"""A/B: same station, same time, same question -- SPC vs Wyoming skew-T source.

Feeds the model each provider's rendered skew-T DIRECTLY (not via the tool's
source-guessing) so the ONLY variable is the image. SPC 'MPX' and Wyoming '72649'
are the SAME station (Chanhassen/Minneapolis) at the same 00/12Z synoptic run, so
any difference in the answer is attributable to the source image -- SPC prints the
derived indices (CAPE/CIN/FZL/EBWD/hodograph) ON the plot, while Wyoming shows only
the bare T/Td curves and wind barbs. The identical question is asked of both; the
two answers are written side by side to a self-contained markdown log under logs/.
"""

import argparse
import base64
from datetime import datetime
from pathlib import Path

from forecaster import soundings
from forecaster.config import settings
from forecaster.llm import client
from forecaster.agent import final_answer
from forecaster.tools import _image_mime

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--model", default=settings.llm_model, help="model id override (default: .env)")
_ap.add_argument("--max-tokens", type=int, default=12288, help="completion budget per call")
_args = _ap.parse_args()

MODEL = _args.model
TEMPERATURE = 0.2
MAX_TOKENS = _args.max_tokens
T = soundings.synoptic_time()

# Same physical station + time, one id per provider's namespace.
CASES = [
    {"label": "SPC", "site": "MPX", "source": "spc"},
    {"label": "Wyoming", "site": "72649", "source": "wyoming"},
]

SYSTEM = (
    "You are a weather forecaster. Analyze the attached observed skew-T sounding "
    "image. Base your answer only on what the sounding shows. State your answer once "
    "and then stop; do not repeat or re-derive it."
)
# IDENTICAL for both sources -- the image is the only thing that changes.
QUESTION = (
    "The attached image is an observed skew-T sounding. Analyze it:\n"
    "1. Describe the temperature and dewpoint profile: where is the air moist vs dry?\n"
    "2. Identify any inversions and the approximate freezing level.\n"
    "3. Assess stability: is the profile favorable for convection, and how does the "
    "wind change with height?"
)


def ask(case: dict) -> dict:
    """Feed one provider's image + the shared question; return the answer + metadata."""
    img = soundings.fetch_skewt(case["site"], T, source=case["source"], use_cache=True)
    b64 = base64.b64encode(img).decode()
    url = soundings.skewt_url(case["site"], T, source=case["source"])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": QUESTION},
            {"type": "image_url",
             "image_url": {"url": f"data:{_image_mime(img)};base64,{b64}"}},
        ]},
    ]
    r = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    msg = r.choices[0].message
    answer, recovery = final_answer(msg, r.choices[0].finish_reason)
    return {
        **case, "url": url, "answer": answer, "recovery": recovery,
        "finish": r.choices[0].finish_reason,
        "prompt_tok": r.usage.prompt_tokens, "completion_tok": r.usage.completion_tokens,
    }


results = [ask(c) for c in CASES]


def build_markdown() -> str:
    md = [
        "# Skew-T Source A/B (SPC vs Wyoming) — MPX / 72649",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Model:** `{MODEL}` @ {settings.llm_base_url}",
        f"- **Station/time:** MPX (72649), {T:%Y-%m-%dT%H:%MZ} — identical sounding, both sources",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        "",
        "## Question (identical for both)",
        "", "```text", QUESTION, "```",
    ]
    for res in results:
        md += [
            "", f"## {res['label']} ({res['source']})",
            f"- **Source URL:** {res['url']}",
            f"- finish: `{res['finish']}`; tokens {res['prompt_tok']}+{res['completion_tok']}",
            "", "**Answer:**", "", res["answer"],
        ]
        if res["recovery"]:
            md += ["", f"> harness note: {res['recovery']}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_slug = MODEL.split("/")[-1].replace(":", "-")
log_path = log_dir / f"sounding_ab_MPX_{model_slug}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

for res in results:
    print(f"\n=== {res['label']} ({res['source']}) — finish={res['finish']}, "
          f"completion_tok={res['completion_tok']} ===")
    print(res["answer"])
    if res["recovery"]:
        print(f"[harness note: {res['recovery']}]")
print(f"\nSide-by-side written to {log_path}")
