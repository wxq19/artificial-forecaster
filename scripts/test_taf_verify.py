"""TAF verification test: how is a live TAF verifying against what was observed?

Exercises the full AWC -> DuckDB -> model path the loader just unlocked:
  1. awc.load_metar(KIND, hours=24)  -> ingest the last 24h of obs (source='awc')
  2. store.window(...)               -> read those obs back from the DB
  3. awc.fetch_taf(KIND) + taf.parse -> the current TAF
  4. feed BOTH to the model, asking it to score the ELAPSED portion of the TAF's
     validity (the part that overlaps the observed window) -- a manual TAFVER.

A TAF forecasts forward, so only the validity that has already run can be scored;
the obs from the TAF's valid_from to the latest report are that overlap. The full
exchange is written to a self-contained markdown log under logs/.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import awc, metar, store, taf
from forecaster.config import settings
from forecaster.llm import client
from forecaster.tools import final_answer

STATION = "KIND"
HOURS = 24
TEMPERATURE = 0.2
MAX_TOKENS = 12288


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)   # naive UTC (the store's contract)


# 1. Ingest the last 24h of observations (idempotent; re-runs add 0).
summary = awc.load_metar(STATION, hours=HOURS)

# 2. Read them back from the DB and render (re-parse the retained raw line).
now = _utc_now()
con = store.connect(read_only=True)
try:
    rows = store.window(con, STATION, now - timedelta(hours=HOURS), now + timedelta(hours=1))
finally:
    con.close()
obs = [metar.parse(r["raw"]) for r in rows]
obs_block = metar.render(obs) if obs else "(no observations in window)"

# 3. The current TAF.
issue, raw_taf = awc.fetch_taf(STATION)[0]
taf_obs = taf.parse(raw_taf)
taf_block = taf.render(taf_obs)
valid = (
    f"{taf_obs.valid_from_day:02d}{taf_obs.valid_from_hour:02d}/"
    f"{taf_obs.valid_to_day:02d}{taf_obs.valid_to_hour:02d}"
)

# 4. Ask the model to score the elapsed validity.
messages = [
    {
        "role": "system",
        "content": (
            "You are a weather forecaster performing TAF verification. You compare "
            "a TAF's forecast against the surface observations (METARs) that occurred "
            "during its validity. Use only the data given; do not invent observations. "
            "State your assessment once and then stop."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Current TAF for {STATION} (valid {valid}):\n\n{taf_block}\n\n"
            f"Observed METARs for {STATION}, last {HOURS}h, oldest first:\n\n{obs_block}\n\n"
            "Only the part of the TAF validity that has already elapsed can be scored "
            "(from valid start up to the most recent observation). For that elapsed "
            "window:\n"
            "1. Ceiling and visibility: did observed conditions stay within the forecast, "
            "or bust (observed worse or better than forecast)?\n"
            "2. Were forecast weather phenomena (showers, thunderstorms, etc.) observed "
            "when and as predicted?\n"
            "3. Overall, is this TAF verifying well or poorly so far, and what was the "
            "single biggest miss?"
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
answer, recovery = final_answer(msg, r.choices[0].finish_reason)


def build_markdown() -> str:
    md = [
        f"# TAF Verification Test — {STATION}",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        "## Ingest summary (awc.load_metar)",
        f"- fetched={summary['fetched']} parsed={summary['parsed']} "
        f"inserted={summary['inserted']} errors={len(summary['errors'])}",
        f"- observations read back from DB for scoring: {len(obs)}",
        "",
        "## TAF under test",
        "",
        "```text",
        taf_block,
        "```",
        "",
        "## Observed METARs (last 24h)",
        "",
        "```text",
        obs_block,
        "```",
        "",
        "## Request",
        f"- **Model:** `{settings.llm_model}`",
        f"- **Endpoint:** {settings.llm_base_url}",
        f"- **Params:** temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
    ]
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
log_path = log_dir / f"taf_verify_{STATION}_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("=== INGEST ===")
print(f"  {summary['fetched']} fetched, {summary['inserted']} inserted, {len(obs)} read back")
print("\n=== MODEL ANSWER ===")
print(answer)
if recovery:
    print(f"\n[harness note: {recovery}]")
print(
    f"\n[finish_reason={r.choices[0].finish_reason}, "
    f"completion_tokens={r.usage.completion_tokens}]"
)
print(f"\nFull exchange written to {log_path}")
