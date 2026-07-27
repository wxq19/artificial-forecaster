"""Screen candidate models for TAF forecasting + chart reading, at a measured cost per forecast.

WHY THIS EXISTS. Round 1 proved that a provider price list does not predict what a forecast
costs. Kimi K2.7 listed at 2.4x MiniMax's input price but cost 20x per USABLE TAF ($1.29 vs
$0.065), because two things a price page cannot show dominate: how many tokens a model burns
on the same task (round-1 median summed input: Gemma 97k, MiniMax 142k, Kimi 480k) and how
often it converges at all (clean-emit 69%/72%/40%). So this screen measures the end number --
dollars per AFMAN-clean, TAFVER-scored TAF -- not dollars per token.

METHOD. Every model replays a FROZEN round-1 transcript, truncated just before the original
model's first check_taf/emit_taf turn. All candidates therefore see byte-identical evidence
(same tool results, same charts), so the model is the only variable. Nothing is fetched from a
weather provider, the run is reproducible, and it is leakage-safe by construction: the
transcript was frozen at issue time, before the verifying obs existed.

The replay deliberately holds tool SELECTION constant, so it does not measure orchestration --
round 1's real discriminator. That is stage 2's job (`--stage2`), a live agent loop for
finalists only, so full agent cost is paid only for models that survive the cheap screen.

TWO SCORED AXES.
  Test B (forecast) -- the model must emit a TAF. emit_taf + check_taf are the only tools, and
    it may iterate on findings, so this measures convergence-to-AFMAN-clean as well as skill.
    Scored with the real `tafver` against banked obs, directly comparable to round 1's pooled
    83.51 (model) / 86.43 (human) / 81.73 (persistence).
  Test A (chart reading) -- the same charts, no tools, factual questions whose answers are
    checkable against the banked METARs. Deliberately narrow: the meteogram plots observations
    we hold exactly, so those answers are objective. Forecast soundings and prog charts are NOT
    scored, because nothing in the DB can adjudicate them.

COST. Every call requests OpenRouter usage accounting, so spend is the endpoint's exact figure
(usage.cost), never a reconstruction from a price table. A running total hard-stops the run at
--budget. reasoning_tokens is captured per turn, which measures rumination directly rather than
inferring it from a completion-token total.

  uv run python scripts/screen_models.py --dry-run
  uv run python scripts/screen_models.py --budget 8
  uv run python scripts/screen_models.py --stage2 --models minimax/minimax-m3,...
"""

import argparse
import gzip
import json
import re
import concurrent.futures as cf
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecaster import store, tafgen, tools  # noqa: E402
from forecaster.agent import AgentConfig, run_agent  # noqa: E402
from forecaster.config import settings  # noqa: E402
from forecaster.tafparse import parse as parse_taf  # noqa: E402
from forecaster.tafstate import TruthPolicy, absolute_validity, default_profile  # noqa: E402
from forecaster.tafver import TafverPolicy, score_tafver  # noqa: E402

BENCH_DB = "data/benchmark/forecaster.duckdb"
RUNS_DIR = Path("data/benchmark/runs")


# ---------------------------------------------------------------- fixtures

@dataclass(frozen=True)
class Fixture:
    """One frozen forecast problem. `run_id` names the transcript that supplies the evidence;
    a worksheet-OFF run is used so the system prompt is already emit-only."""

    station: str
    valid_from: str
    run_id: str
    note: str


# Three round-1 evaluations chosen for DISCRIMINATION: persistence scores 53-62% on all of
# them, meaning the weather actually changed, so a model cannot do well by restating the
# current observation. Each already carries scored human + persistence + model baselines.
FIXTURES = [
    Fixture("KFTK", "2026-07-20 11:00:00", "KFTK_20260720T1100_MiniMax-M3_off_t0.0_taf",
            "CONUS convective; persistence 53.8, human 88.6"),
    Fixture("RJTY", "2026-07-17 05:00:00", "RJTY_20260717T0500_MiniMax-M3_off_t0.0_taf",
            "OCONUS Japan; persistence 58.0, human 79.1"),
    Fixture("KWRI", "2026-07-18 10:00:00", "KWRI_20260718T1000_MiniMax-M3_off_t0.0_taf",
            "CONUS mid; persistence 61.7, human 71.9"),
]


# ---------------------------------------------------------------- model matrix

@dataclass(frozen=True)
class Arm:
    """One (model, reasoning-setting) cell."""

    model: str
    label: str
    extra: dict = field(default_factory=dict)   # merged into the request body


# One reputable provider per model, so the screen measures the MODEL, not OpenRouter's default
# routing across flaky backends. Learned the hard way: unpinned, Gemma routed to SiliconFlow,
# whose serving template leaked special tokens INTO the tool-call JSON ("cover": "<|\"SCT<|") --
# provider-quality noise masquerading as model failure. Each pick is the lab's own endpoint or a
# well-known host (DeepInfra/Together), tools+vision confirmed via the /endpoints API 2026-07-24.
# Round 2 will pin a provider anyway, so this is also the deployment-faithful test.
# Each pick EMPIRICALLY QUALIFIED 2026-07-24 (scratchpad qualify.py): a real chart image + a tool
# call must return clean choices with no special-token leakage. This weeds out (a) tools-only
# endpoints that reject image input (DeepInfra's Gemma 404s on any image) and (b) corrupting
# serving templates (SiliconFlow's Gemma leaks '<|' into the tool-call JSON). max_completion on
# every chosen endpoint is >= MAX_TOKENS (lowest is DeepInfra qwen-235b at 16384).
# Providers that produced a CLEAN emit on the REAL task (parallel qual_emit.py, 2026-07-24),
# excluding the community-flagged bad hosts. 7 are confirmed clean; grok/mimo/mistral get one shot
# each on their best remaining host (with PNG-normalized images) -- the guard records a failure
# without crashing, so including uncertain models is free. GLM dropped: only Novita emitted clean
# and Novita is excluded.
PROVIDERS: dict[str, str] = {
    "google/gemma-4-31b-it": "Together",            # CLEAN
    "minimax/minimax-m3": "Together",               # CLEAN (Novita mangled the clouds array)
    "xiaomi/mimo-v2.5": "GMICloud",                 # one shot: lab+DO+Parasail failed; GMICloud untested
    "moonshotai/kimi-k3": "Moonshot AI",            # CLEAN
    "thinkingmachines/inkling": "Together",         # CLEAN (cheaper/faster than BaseTen)
    "x-ai/grok-4.5": "xAI",                         # one shot: failed on GIF; PNG normalization may fix
    "google/gemini-3.1-flash-lite": "Google",       # CLEAN
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral",   # one shot: accepted 7 imgs but did not emit
    "qwen/qwen3-vl-32b-instruct": "Alibaba",        # CLEAN
    "qwen/qwen3-vl-235b-a22b-instruct": "Alibaba",  # CLEAN
}

# Served quantization at each chosen (single) endpoint, from the /endpoints API 2026-07-24.
# Recorded because on a ROUTER precision varies across providers and is NOT strictly comparable
# (CLAUDE.md); the ranking must be read quant-aware. int4 (Kimi) is the most aggressive -- a poor
# score there could be the quant, not the model. "closed"/"undisclosed" = lab does not publish it
# (single endpoint, so still reproducible). Every pick is a single endpoint, so the served quant
# is deterministic; the report prints it beside each model.
QUANT: dict[str, str] = {
    "google/gemma-4-31b-it": "undisclosed",        # Together 2-endpoint; scores were stable (86.0/79.9) so effectively one serving
    "minimax/minimax-m3": "fp8",
    "xiaomi/mimo-v2.5": "fp8",
    "moonshotai/kimi-k3": "int4",
    "thinkingmachines/inkling": "fp8",
    "x-ai/grok-4.5": "closed",
    "google/gemini-3.1-flash-lite": "closed",
    "mistralai/mistral-small-3.2-24b-instruct": "undisclosed",
    "qwen/qwen3-vl-32b-instruct": "undisclosed",
    "qwen/qwen3-vl-235b-a22b-instruct": "fp8",
}


# Providers to never route to. novita/groq/cerebras: community-flagged low quality and confirmed
# here -- Novita mangled MiniMax's nested tool-call array into XML and errored Gemma at 16k.
# siliconflow: leaked chat-template special tokens INTO Gemma's tool-call JSON. Enforced per
# request (a hard order-pin overrides account-level presets, so the exclusion must be explicit).
IGNORE_PROVIDERS = ["novita", "groq", "cerebras", "siliconflow"]


def provider_block(model: str) -> dict:
    """The OpenRouter provider pin for one model: force the chosen backend, no fallback, and
    hard-exclude the known-bad providers. Omit require_parameters (unlike llm.routing) -- these
    endpoints are hand-verified tools+vision, and require_parameters would 404 a reasoning-ablation
    arm on a host that does not advertise the reasoning param."""
    prov = PROVIDERS.get(model)
    if not prov:
        return {}
    return {"provider": {"order": [prov], "allow_fallbacks": False, "ignore": IGNORE_PROVIDERS}}


def _arms(model: str, kind: str) -> list[Arm]:
    """Reasoning ablations. `effort` models take reasoning_effort low|high; `toggle` models
    expose only an on/off reasoning object; `none` has no reasoning channel at all."""
    if kind == "effort":
        return [Arm(model, "reason-low", {"reasoning": {"effort": "low"}}),
                Arm(model, "reason-high", {"reasoning": {"effort": "high"}})]
    if kind == "toggle":
        return [Arm(model, "reason-off", {"reasoning": {"enabled": False}}),
                Arm(model, "reason-on", {"reasoning": {"enabled": True}})]
    return [Arm(model, "base", {})]


# Controls first: Gemma and MiniMax are the round-1 anchors. Their score HERE is the baseline
# the candidates are read against -- not their round-1 score, which was earned on a different
# evidence set (each model chose its own tools).
MATRIX: list[Arm] = [
    *_arms("google/gemma-4-31b-it", "none"),                   # control; single arm -- Together serves reason-off cleanly but errors when reasoning is ENABLED at 16k
    *_arms("minimax/minimax-m3", "toggle"),                    # control
    *_arms("xiaomi/mimo-v2.5", "toggle"),
    *_arms("moonshotai/kimi-k3", "effort"),
    *_arms("thinkingmachines/inkling", "effort"),
    *_arms("x-ai/grok-4.5", "effort"),
    *_arms("google/gemini-3.1-flash-lite", "effort"),
    # z-ai/glm-4.6v DROPPED: only Novita emitted clean, and Novita is excluded (bad-provider list).
    *_arms("mistralai/mistral-small-3.2-24b-instruct", "none"),
    *_arms("qwen/qwen3-vl-32b-instruct", "none"),
    *_arms("qwen/qwen3-vl-235b-a22b-instruct", "none"),
]

# Round-1 production value (collect.py): 2x the cap that broke MiniMax, ample for CoT (round-1
# models converged well under it), and -- critically -- known-good on the TIGHTEST pinned
# provider. A hard provider pin with no fallback 404s if max_tokens exceeds that host's
# completion cap (DeepInfra Gemma caps below 24k), so this must clear the lowest, not the
# highest. The budget guard bounds SPEND; REQUEST_TIMEOUT bounds a single stuck/ruminating call.
MAX_TOKENS = 16000
MAX_STEPS = 8               # emit -> read findings -> re-emit, with room to spare
# A call slower than this is stuck or generating absurd reasoning -- either way that cell has
# failed to converge, which is a RESULT (recorded as fatal), not something to wait out. At
# ~50s/call for a normal turn, 300s is 6x headroom.
REQUEST_TIMEOUT = 300.0


# ---------------------------------------------------------------- transcript replay

_STOP_TOOLS = {"emit_taf", "check_taf", "submit_taf_worksheet"}


def _to_png(data_url: str) -> str:
    """Transcode any evidence image to a PNG data URL. Some providers reject formats the charts
    ship in -- xAI accepts JPG/PNG/WebP/ICO but NOT GIF (the WPC maps are GIF), and Z.AI mis-parses
    mixed formats. Normalizing to one format the round-1 charts already mostly use removes that as a
    variable so an image-format quirk cannot masquerade as a model failure."""
    import base64
    import io

    from PIL import Image
    try:
        head, b64 = data_url.split(",", 1)
        if "image/png" in head:
            return data_url
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 -- a transcode failure leaves the original; do not drop evidence
        return data_url


def _normalize_images(msgs: list[dict]) -> list[dict]:
    """Return a copy with every image part transcoded to PNG (content lists are rebuilt, originals
    untouched)."""
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            m = dict(m)
            m["content"] = [
                {**p, "image_url": {**p["image_url"], "url": _to_png(p["image_url"]["url"])}}
                if p.get("type") == "image_url" else p
                for p in c]
        out.append(m)
    return out


def load_evidence(fx: Fixture) -> list[dict]:
    """The frozen messages up to (not including) the first turn where the original model began
    producing a TAF. Everything before that is pure evidence: the task, the tool results, the
    charts. Images are normalized to PNG (see _to_png). Raises rather than silently truncating."""
    path = RUNS_DIR / fx.run_id / "messages.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        msgs = json.load(f)
    for i, m in enumerate(msgs):
        names = {t["function"]["name"] for t in (m.get("tool_calls") or [])}
        if m["role"] == "assistant" and (names & _STOP_TOOLS):
            return _normalize_images(msgs[:i])
    raise ValueError(f"{fx.run_id}: no emit/check turn found; cannot locate a truncation point")


def count_images(msgs: list[dict]) -> int:
    n = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if p.get("type") == "image_url")
    return n


_EMIT_NUDGE = (
    "You have all the evidence you are going to get; no further data tools are available.\n"
    "Produce the 30-hour Air Force TAF now by calling emit_taf. If the AFMAN validator returns "
    "findings, read them and call emit_taf again with the corrections. Use check_taf first if "
    "you want a dry run. Do not ask for more data."
)


# ---------------------------------------------------------------- scoring

def score_taf_product(product, fx: Fixture, con) -> tuple[float | None, str]:
    """Real TAFVER for an emitted TafProduct against the banked obs for this window.

    The product is rendered to text and re-parsed, because the scorer consumes the same parsed
    shape as an archived human TAF -- scoring the model through the identical path keeps the
    comparison honest rather than giving generated TAFs a private route."""
    try:
        raw = tafgen.render_taf(product)
        taf = parse_taf(raw)
        # TafProduct carries only day-of-month/hour/minute (no year/month), so anchor the
        # issue reference on the fixture's KNOWN calendar month. These fixtures are same-day
        # issues, so vf's year/month are the issue's year/month.
        fvf = datetime.fromisoformat(fx.valid_from)
        issue_ref = datetime(fvf.year, fvf.month, product.issue_day,
                             product.issue_hour, product.issue_minute)
        _, vf, vt = absolute_validity(taf, issue_ref)
        obs = store.scoring_window(con, fx.station, vf, vt)
        if not obs:
            return None, "no obs in window"
        sc = score_tafver(taf, obs, vf, vt, profile=default_profile(fx.station),
                          policy=TafverPolicy(), truth_policy=TruthPolicy())
        return sc.combined_percent, ""
    except Exception as e:  # noqa: BLE001 -- an unscoreable TAF is a result, not a crash
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- vision test

@dataclass
class VisionQ:
    key: str
    question: str
    answer: float | None      # from the DB
    tol: float
    unit: str


def build_vision_questions(fx: Fixture, con) -> list[VisionQ]:
    """Questions answerable from the METEOGRAM, whose underlying observations we hold exactly.

    Scope is deliberately narrow. The fixture also carries forecast soundings and prog charts,
    but nothing in the DB can adjudicate those, so they are shown and not scored -- a wrong
    answer there would be unfalsifiable, not a measurement."""
    vf = datetime.fromisoformat(fx.valid_from)
    # get_trend's default look-back is 24h ending at the latest pre-cutoff ob: the exact span
    # the meteogram in this transcript depicts.
    rows = store.window(con, fx.station, vf.replace(tzinfo=None) - _DAY, vf.replace(tzinfo=None))
    if not rows:
        return []
    temps = [r["temp_c"] for r in rows if r.get("temp_c") is not None]
    # wind_unit is KT for every roster field; guard anyway so a non-KT station is skipped
    # rather than graded against the wrong scale.
    winds = [r["wind_speed"] for r in rows
             if r.get("wind_speed") is not None and (r.get("wind_unit") or "KT") == "KT"]
    if not temps:
        return []
    return [
        VisionQ("temp_max", "the HIGHEST temperature in degrees C shown on the temperature panel",
                max(temps), 1.5, "C"),
        VisionQ("temp_min", "the LOWEST temperature in degrees C shown on the temperature panel",
                min(temps), 1.5, "C"),
        VisionQ("wind_max", "the HIGHEST sustained wind speed in knots shown on the wind panel",
                max(winds) if winds else None, 4.0, "kt"),
    ]


_DAY = __import__("datetime").timedelta(hours=24)

_VISION_PROMPT = (
    "Read the attached weather charts. Answer ONLY from what is plotted -- do not estimate from "
    "climatology or reasoning about the season.\n\n"
    "Report your answers as a JSON object on the final line, of the form "
    '{{"temp_max": <number>, "temp_min": <number>, "wind_max": <number>}}.\n\n'
    "{qs}"
)


def _extract_json(text: str) -> dict:
    """Last JSON object in the reply. Models wrap answers in prose or fences; the contract is
    only that the object appears, so take the last one rather than demanding a bare reply."""
    best = {}
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                best = obj
        except json.JSONDecodeError:
            continue
    return best


# ---------------------------------------------------------------- runner

@dataclass
class Result:
    fixture: str
    model: str
    arm: str
    test: str
    cost: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    steps: int = 0
    provider: str | None = None
    taf_clean: bool = False
    tafver: float | None = None
    vision_hits: int = 0
    vision_total: int = 0
    stop_reason: str = ""
    error: str = ""
    seconds: float = 0.0


class Budget:
    """Hard spend stop. Checked BEFORE each call using the worst observed cost so far, so the
    ceiling cannot be blown through by one expensive model mid-matrix."""

    def __init__(self, cap: float):
        self.cap, self.spent, self.worst = cap, 0.0, 0.0
        self._lock = threading.Lock()   # cells run in parallel threads; spend is shared state

    def add(self, cost: float | None) -> None:
        if cost:
            with self._lock:
                self.spent += cost
                self.worst = max(self.worst, cost)

    def would_exceed(self) -> bool:
        with self._lock:
            return self.spent + self.worst > self.cap


def run_forecast(fx: Fixture, arm: Arm, evidence: list[dict], con, budget: Budget) -> Result:
    """Test B: replay the evidence, require an emitted TAF, score it."""
    r = Result(fx.station, arm.model, arm.label, "forecast")
    msgs = [dict(m) for m in evidence] + [{"role": "user", "content": _EMIT_NUDGE}]
    cfg = AgentConfig(
        model=arm.model, toolset=[tools.EMIT_TAF, tools.CHECK_TAF],
        max_steps=MAX_STEPS, max_tokens=MAX_TOKENS, temperature=0.0,
        worksheet_mode="off", evidence=False, stop_on_clean_taf=True,
        step_budget_nudge=True, request_timeout=REQUEST_TIMEOUT,
        extra_body={**arm.extra, **provider_block(arm.model), "usage": {"include": True}},
    )
    t0 = time.monotonic()
    res = run_agent(msgs, cfg)
    r.seconds = time.monotonic() - t0
    r.cost, r.prompt_tokens, r.completion_tokens = res.cost, res.prompt_tokens, res.completion_tokens
    r.reasoning_tokens, r.steps, r.stop_reason = res.reasoning_tokens, len(res.steps), res.stop_reason
    r.provider = " | ".join(res.providers) or None
    r.error = res.fatal or ""
    budget.add(res.cost)
    product = res.final_taf or res.last_taf
    r.taf_clean = res.final_taf is not None
    if product is not None:
        r.tafver, err = score_taf_product(product, fx, con)
        if err and not r.error:
            r.error = err
    elif not r.error:
        r.error = "no TAF emitted"
    return r


def run_vision(fx: Fixture, arm: Arm, evidence: list[dict], qs: list[VisionQ],
               budget: Budget) -> Result:
    """Test A: the same charts, no tools, objectively gradeable questions."""
    from forecaster.llm import client, routing

    r = Result(fx.station, arm.model, arm.label, "vision")
    r.vision_total = len([q for q in qs if q.answer is not None])
    # Reuse the frozen image parts; drop the tool-call scaffolding so the model sees charts
    # plus a question, not a half-finished agent conversation.
    images = []
    for m in evidence:
        c = m.get("content")
        if isinstance(c, list):
            images += [p for p in c if p.get("type") == "image_url"]
    qtext = "\n".join(f"- {q.key}: {q.question}" for q in qs if q.answer is not None)
    content = [{"type": "text", "text": _VISION_PROMPT.format(qs=qtext)}] + images
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=arm.model, messages=[{"role": "user", "content": content}],
            max_tokens=MAX_TOKENS, temperature=0.0, timeout=REQUEST_TIMEOUT,
            extra_body={**arm.extra, **provider_block(arm.model),
                        "usage": {"include": True}, **routing()},
        )
    except Exception as e:  # noqa: BLE001 -- a refusing endpoint is a result
        r.error = f"{type(e).__name__}: {e}"
        r.seconds = time.monotonic() - t0
        return r
    r.seconds = time.monotonic() - t0
    u = resp.usage
    r.cost = getattr(u, "cost", None) if u else None
    r.prompt_tokens = u.prompt_tokens if u else 0
    r.completion_tokens = u.completion_tokens if u else 0
    det = getattr(u, "completion_tokens_details", None) if u else None
    r.reasoning_tokens = getattr(det, "reasoning_tokens", None) or 0
    r.provider = getattr(resp, "provider", None)
    budget.add(r.cost)
    got = _extract_json(resp.choices[0].message.content or "")
    for q in qs:
        if q.answer is None:
            continue
        try:
            if abs(float(got[q.key]) - q.answer) <= q.tol:
                r.vision_hits += 1
        except (KeyError, TypeError, ValueError):
            pass
    if not got:
        r.error = "no JSON answer in reply"
    return r


# ---------------------------------------------------------------- report

def write_report(results: list[Result], budget: Budget, path: Path, dry: bool) -> None:
    by_model: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        by_model.setdefault((r.model, r.arm), []).append(r)

    lines = [
        "# Model screen -- TAF forecasting + chart reading",
        "",
        f"- Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}",
        f"- Endpoint: {settings.llm_base_url}  |  provider pin: "
        f"{settings.llm_provider_order or '(none)'}",
        f"- Fixtures: {len(FIXTURES)}  |  arms: {len(MATRIX)}  |  max_tokens: {MAX_TOKENS}",
        f"- **Measured spend: ${budget.spent:.4f}** (cap ${budget.cap:.2f})",
        "",
        "Costs are the endpoint's own `usage.cost`, not a price-table reconstruction.",
        "TAFVER is the real scorer against banked obs; round-1 pooled reference: "
        "human 86.43, model 83.51, persistence 81.73.",
        "",
        "## Ranking -- cost per AFMAN-clean TAF",
        "",
        "Quant is the served precision (single pinned endpoint per model). Scores are NOT "
        "iso-precision: bf16 > fp8 > int4. Read the ranking as best model AS SERVED, not in a "
        "lab. Kimi runs int4 (most aggressive) -- a weak score there may be the quant.",
        "",
        "| model | quant | arm | clean | TAFVER | vision | $/run | **$/clean TAF** | reason tok | steps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for (model, arm), rs in by_model.items():
        fc = [r for r in rs if r.test == "forecast"]
        vs = [r for r in rs if r.test == "vision"]
        n = len(fc) or 1
        clean = sum(1 for r in fc if r.taf_clean)
        scored = [r.tafver for r in fc if r.tafver is not None]
        cost = sum((r.cost or 0) for r in rs)
        percl = cost / clean if clean else None
        vh = sum(r.vision_hits for r in vs)
        vt = sum(r.vision_total for r in vs)
        rows.append((percl if percl is not None else 9e9, model, arm, clean, n,
                     (sum(scored) / len(scored)) if scored else None,
                     vh, vt, cost / max(len(fc), 1), percl,
                     sum(r.reasoning_tokens for r in rs),
                     sum(r.steps for r in fc) / max(len(fc), 1)))
    for _, model, arm, clean, n, tv, vh, vt, perrun, percl, rtok, steps in sorted(rows):
        lines.append(
            f"| `{model}` | {QUANT.get(model, '?')} | {arm} | {clean}/{n} | "
            f"{f'{tv:.1f}' if tv is not None else '--'} | "
            f"{vh}/{vt} | ${perrun:.4f} | "
            f"{f'${percl:.4f}' if percl is not None else 'NO CLEAN TAF'} | "
            f"{rtok:,} | {steps:.1f} |")

    lines += ["", "## Per run", "",
              "| fixture | model | arm | test | clean | TAFVER | vision | $ | in tok | out tok "
              "| reason tok | steps | stop | provider | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.fixture} | `{r.model}` | {r.arm} | {r.test} | "
            f"{'yes' if r.taf_clean else ('--' if r.test == 'vision' else 'NO')} | "
            f"{f'{r.tafver:.1f}' if r.tafver is not None else '--'} | "
            f"{f'{r.vision_hits}/{r.vision_total}' if r.test == 'vision' else '--'} | "
            f"{f'${r.cost:.5f}' if r.cost is not None else '--'} | "
            f"{r.prompt_tokens:,} | {r.completion_tokens:,} | {r.reasoning_tokens:,} | "
            f"{r.steps} | {r.stop_reason} | {r.provider or '--'} | {r.error[:70]} |")

    lines += ["", "## Fixtures", ""]
    for fx in FIXTURES:
        lines.append(f"- **{fx.station}** {fx.valid_from} -- {fx.note} (`{fx.run_id}`)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- parallel worker

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    """Thread-safe single-line log so interleaved cells don't garble each other."""
    with _PRINT_LOCK:
        print(msg, flush=True)


def run_cell(fx: Fixture, arm: Arm, ev: list[dict], qs: list[VisionQ], budget: Budget,
             db_path: str, skip_vision: bool) -> list[Result]:
    """One matrix cell (forecast [+ vision]) with its OWN DuckDB connection -- a connection is
    not safe to share across threads, so each worker opens and closes its own read-only handle.
    Skips (cheaply) if the budget is already exhausted when the worker starts, which bounds
    overshoot to at most max_workers in-flight cells."""
    if budget.would_exceed():
        return []
    out: list[Result] = []
    con = store.connect(db_path, read_only=True)
    try:
        r = run_forecast(fx, arm, ev, con, budget)
        out.append(r)
        _log(f"  {datetime.now(timezone.utc):%H:%M:%S} [{fx.station}] {arm.model} {arm.label:<12} "
             f"forecast clean={'Y' if r.taf_clean else 'N'} "
             f"tafver={f'{r.tafver:.1f}' if r.tafver is not None else '--':>5} "
             f"${(r.cost or 0):.4f} rtok={r.reasoning_tokens:,} {r.seconds:.0f}s "
             f"{r.stop_reason} {r.error[:40]}  [total ${budget.spent:.3f}]")
        if not skip_vision:
            rv = run_vision(fx, arm, ev, qs, budget)
            out.append(rv)
            _log(f"  {datetime.now(timezone.utc):%H:%M:%S} [{fx.station}] {arm.model} "
                 f"{arm.label:<12} vision   {rv.vision_hits}/{rv.vision_total} "
                 f"${(rv.cost or 0):.4f} rtok={rv.reasoning_tokens:,} {rv.seconds:.0f}s "
                 f"{rv.error[:40]}")
    finally:
        con.close()
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=8.0, help="hard spend cap in USD")
    ap.add_argument("--dry-run", action="store_true", help="show the matrix, spend nothing")
    ap.add_argument("--models", help="comma-separated subset of model slugs")
    ap.add_argument("--fixtures", help="comma-separated subset of stations")
    ap.add_argument("--db", default=BENCH_DB)
    ap.add_argument("--skip-vision", action="store_true")
    # A screen spans models from many labs, and no single backend serves them all -- a pin
    # that is right for a benchmark ROUND would 404 most of this matrix (allow_fallbacks is
    # false by design). So the screen runs unpinned by default and RECORDS the provider that
    # actually served each call; pin later, per model, once the winners are known.
    ap.add_argument("--pin", default="",
                    help="provider pin for this run (default: none -- let OpenRouter route)")
    ap.add_argument("--out", help="report path (default logs/model_screen_<ts>.md)")
    ap.add_argument("--parallel", type=int, default=6,
                    help="cells to run concurrently (I/O-bound API calls; arms hit different "
                         "providers so per-provider load stays low). 1 = serial.")
    args = ap.parse_args()
    settings.llm_provider_order = args.pin      # see --pin help: unpinned across labs

    matrix = MATRIX
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        matrix = [a for a in matrix if a.model in want]
    fixtures = FIXTURES
    if args.fixtures:
        want = {f.strip().upper() for f in args.fixtures.split(",")}
        fixtures = [f for f in fixtures if f.station in want]
    if not matrix or not fixtures:
        print("nothing selected")
        return 1

    con = store.connect(args.db, read_only=True)
    ev_by_fx, qs_by_fx = {}, {}
    for fx in fixtures:
        ev_by_fx[fx.station] = load_evidence(fx)
        qs_by_fx[fx.station] = build_vision_questions(fx, con)
    con.close()   # workers open their own per-thread connections

    n_calls = len(matrix) * len(fixtures) * (1 if args.skip_vision else 2)
    print(f"Endpoint : {settings.llm_base_url}")
    print(f"Pin      : {settings.llm_provider_order or '(none)'}")
    print(f"Matrix   : {len(matrix)} arms x {len(fixtures)} fixtures "
          f"x {1 if args.skip_vision else 2} tests = {n_calls} calls")
    print(f"Parallel : {args.parallel} concurrent cells")
    print(f"Budget   : ${args.budget:.2f} hard cap  |  max_tokens {MAX_TOKENS}\n")
    for fx in fixtures:
        ev = ev_by_fx[fx.station]
        qs = [q for q in qs_by_fx[fx.station] if q.answer is not None]
        print(f"  {fx.station} {fx.valid_from}: {len(ev)} msgs, {count_images(ev)} images, "
              f"{len(qs)} vision Qs -> " +
              ", ".join(f"{q.key}={q.answer:g}{q.unit}" for q in qs))
    print()
    for a in matrix:
        print(f"  {a.model:<44} {a.label}")
    if args.dry_run:
        print("\n--dry-run: nothing spent.")
        return 0

    budget = Budget(args.budget)
    results: list[Result] = []
    cells = [(fx, arm) for fx in fixtures for arm in matrix]
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(run_cell, fx, arm, ev_by_fx[fx.station],
                          [q for q in qs_by_fx[fx.station] if q.answer is not None],
                          budget, args.db, args.skip_vision): (fx, arm)
                for fx, arm in cells}
        for f in cf.as_completed(futs):
            try:
                results.extend(f.result())
            except Exception as e:  # noqa: BLE001 -- one cell's crash must not sink the batch
                fx, arm = futs[f]
                _log(f"  CELL CRASH [{fx.station}] {arm.model} {arm.label}: {type(e).__name__}: {e}")
    if budget.would_exceed():
        _log(f"\n!! BUDGET REACHED ${budget.spent:.4f} (cap ${budget.cap:.2f}); later cells skipped")

    out = Path(args.out) if args.out else Path("logs") / (
        f"model_screen_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, budget, out, args.dry_run)
    print(f"\nTOTAL MEASURED SPEND ${budget.spent:.4f} of ${args.budget:.2f}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
