"""Multi-model comparison: reasoning over synoptic maps (wxmaps get_map products).

Feeds every model the SAME four-chart packet -- current surface analysis, current
500 mb analysis, and the GFS 500 mb + MSLP/precip forecasts valid +24h -- and the
SAME question, which forces integration ACROSS the charts (locate the pattern, project
its 24h evolution, then say where the active weather goes and why). Runs Gemma, Qwen,
Kimi K2.7, and MiniMax M3 through the identical prompt so their synoptic reasoning is
directly comparable. Each answer + metadata is written side by side to a markdown log.

Charts are fetched via the wxmaps seam (self-contained; cites provenance). Per-model
failures are recorded, not fatal, so one bad model doesn't sink the comparison.
"""

import base64
from datetime import datetime
from pathlib import Path

from forecaster import wxmaps
from forecaster.config import settings
from forecaster.llm import client
from forecaster.agent import final_answer
from forecaster.tools import _image_mime

TEMPERATURE = 0.2
MAX_TOKENS = 12288          # reasoning models spend tokens thinking; leave room for the answer

MODELS = [
    ("Gemma", "google/gemma-4-31B-it"),
    ("Qwen", "Qwen/Qwen3.5-9B"),
    ("Kimi", "moonshotai/Kimi-K2.7-Code"),
    ("MiniMax", "MiniMaxAI/MiniMax-M3"),
]

# (caption, catalog name, forecast hour or None) -- a current->forecast synoptic packet.
CHARTS = [
    ("Chart 1 -- CURRENT surface analysis (fronts, isobars, pressure systems)", "surface_analysis", None),
    ("Chart 2 -- CURRENT 500 mb analysis (heights, vorticity, wind)", "meso_500mb", None),
    ("Chart 3 -- GFS 500 mb FORECAST valid +24h (heights, vorticity, wind)", "gfs_500mb", 24),
    ("Chart 4 -- GFS MSLP + precipitation FORECAST valid +24h", "gfs_mslp_precip", 24),
]

SYSTEM = (
    "You are a weather forecaster analyzing synoptic charts. Base your answer only on "
    "the charts provided; do not invent features. State your answer once and then stop; "
    "do not repeat or re-derive it."
)
QUESTION = (
    "Using all four charts above:\n"
    "1. Describe the current large-scale pattern: locate the main 500 mb troughs and "
    "ridges, and the surface fronts and pressure systems.\n"
    "2. How does the 500 mb pattern change over the next 24 hours (progression of the "
    "troughs/ridges)?\n"
    "3. Integrating all four charts, where across the CONUS is the most active or "
    "significant weather expected in the next 24 hours, and why?"
)

RUN = wxmaps.latest_gfs_run()


def build_content() -> tuple[list[dict], list[str]]:
    """Interleave each caption with its image so the model maps chart->description,
    then append the question. Returns (content, provenance URLs)."""
    content: list[dict] = []
    urls: list[str] = []
    for caption, name, fhr in CHARTS:
        img = wxmaps.fetch_map(name, fhr=fhr or 0, run=RUN if fhr is not None else None)
        b64 = base64.b64encode(img).decode()
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{_image_mime(img)};base64,{b64}"}})
        urls.append(wxmaps.map_url(name, fhr=fhr or 0, run=RUN if fhr is not None else None))
    content.append({"type": "text", "text": QUESTION})
    return content, urls


content, urls = build_content()
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

results = []
for label, model in MODELS:
    rec = {"label": label, "model": model}
    try:
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        msg = r.choices[0].message
        answer, recovery = final_answer(msg, r.choices[0].finish_reason)
        rec.update(answer=answer, recovery=recovery, finish=r.choices[0].finish_reason,
                   ptok=r.usage.prompt_tokens, ctok=r.usage.completion_tokens)
    except Exception as e:  # noqa: BLE001 -- record and continue
        rec["error"] = f"{type(e).__name__}: {e}"
    results.append(rec)
    print(f"\n=== {label} ({model}) ===")
    if "error" in rec:
        print(f"  ERROR: {rec['error']}")
    else:
        print(f"  [finish={rec['finish']}, ctok={rec['ctok']}]")
        print(rec["answer"])
        if rec["recovery"]:
            print(f"  [harness note: {rec['recovery']}]")


def build_markdown() -> str:
    md = [
        "# Multi-model synoptic-map reasoning comparison",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        f"- **GFS run:** {RUN:%Y-%m-%dT%H:%MZ}",
        "",
        "## Charts provided (identical for every model)",
    ]
    for (caption, _, _), url in zip(CHARTS, urls):
        md += [f"- {caption} -- {url}"]
    md += ["", "## Question (identical for every model)", "", "```text", QUESTION, "```",
           "", "## Answers"]
    for rec in results:
        md += ["", f"### {rec['label']} (`{rec['model']}`)"]
        if "error" in rec:
            md += ["", f"- ERROR: `{rec['error']}`"]
            continue
        md += [f"- finish: `{rec['finish']}`; tokens {rec['ptok']}+{rec['ctok']}",
               "", rec["answer"]]
        if rec["recovery"]:
            md += ["", f"> harness note: {rec['recovery']}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"maps_compare_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")
print(f"\nSide-by-side written to {log_path}")
