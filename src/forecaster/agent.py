"""The agent loop -- the ONLY file that drives the model's tool-calling turns.

Everything the model needs to KNOW (system prompt, task) and everything a run
VARIES (model, toolset, caps, worksheet mode) comes in through `messages` +
AgentConfig; the loop body has no per-driver branches. Everything a run PRODUCES
-- the full frozen `messages` array (prompt + reasoning + every tool call and the
data it returned + the images the model saw), per-turn token/finish records, the
emitted TAF/worksheet, and a stop_reason -- comes back on one RunResult, which is
both what a driver renders to markdown AND what the collector persists as the
provenance of the run.

Seam boundaries: the loop does NOT build the client (that stays in llm.py) and
does NOT dispatch tools (that stays in tools.run_tool, the tool registry). It sits
on top of both. The loop-plumbing helpers final_answer/tool_messages/window_conflict
live here; _image_mime is imported from tools (a tool-output format helper tools.py
also uses for get_imagery).
"""

import base64
import json
from collections import Counter
from dataclasses import dataclass, field

from forecaster import tafgen
from forecaster import worksheet as wksht
from forecaster.config import settings
from forecaster.llm import client as _default_client
from forecaster.tafgen import TafProduct
from forecaster.tools import ToolResult, _image_mime, run_tool
from forecaster.worksheet import TafWorksheet

# The sinks + pure validators are OUTPUTS/dry-runs, not data -- they earn no evidence id.
_NON_EVIDENCE = {"emit_taf", "submit_taf_worksheet", "check_taf"}

# One-time nudge when the step budget is nearly spent and no TAF has been attempted.
# Injected as role 'user' (Together 400s on a mid-conversation system message).
_BUDGET_NUDGE = (
    "You have used most of your turns and have not emitted a TAF yet. You have enough "
    "data -- stop gathering, reason briefly, and call emit_taf now."
)

# Recovery nudge when a turn hits the token cap BEFORE calling any tool: the turn was
# truncated mid-thought, not finished, so we re-prompt to continue instead of ending.
_LENGTH_NUDGE = (
    "Your previous turn was cut off at the token limit before you finished. Be concise: "
    "state your conclusion briefly and call the next tool (or emit_taf) now."
)


@dataclass
class StepRecord:
    """One model turn: what it said, what it cost, and what it called."""

    n: int
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    content: str
    reasoning: str
    calls: list[dict] = field(default_factory=list)   # per call: name, args, result-line, n_images
    answer: str | None = None       # set only on the terminal no-tool-call turn
    recovery: str | None = None     # final_answer's reasoning-field recovery flag, if any
    served_model: str | None = None         # the model id the API RETURNED for this turn
    system_fingerprint: str | None = None   # provider build id; often None on Together


@dataclass
class AgentConfig:
    """Everything that varies per run. Prompt/task ride in `messages`, not here."""

    model: str
    toolset: list[dict]
    max_steps: int = 14
    max_tokens: int = 12000
    temperature: float = 0.2
    # One fixed seed for the whole matrix (config.llm_seed); a driver may still override.
    # Sent to the API only when non-None.
    seed: int | None = field(default_factory=lambda: settings.llm_seed)
    tool_caps: dict[str, int] | None = None
    worksheet_mode: str = "advisory"    # off|advisory|required -- governs the emit_taf GATE only
    evidence: bool = True               # thread [evidence_id: ev_NNN] onto data-tool receipts
    stop_on_clean_taf: bool = True      # stop as soon as emit_taf returns a validate()-clean TAF
    step_budget_nudge: bool = False     # at turn max_steps-2, nudge to emit if none attempted yet
    db_path: str | None = None


@dataclass
class RunResult:
    """The full provenance of one agent run -- rendered by drivers, persisted by the collector."""

    model: str
    messages: list[dict]                # the FULL frozen array = the transcript
    steps: list[StepRecord] = field(default_factory=list)
    used: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    final_taf: TafProduct | None = None             # first validate()-clean emit
    last_taf: TafProduct | None = None              # most recent emit, clean or not
    worksheet: TafWorksheet | None = None           # last worksheet with no blocking findings
    worksheet_findings: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    stop_reason: str = "max_steps"      # emitted_clean | no_tool_call | max_steps | fatal
    fatal: str | None = None
    first_emit_step: int | None = None  # first turn emit_taf was attempted (convergence)
    nudge_step: int | None = None       # turn the step-budget nudge fired, if any
    # Generation-side provenance, stamped by run_agent from cfg + the API responses. None
    # of it is reconstructable later, so it is captured at run time and persisted per run.
    base_url: str | None = None         # which endpoint served the run (local/Together/vLLM)
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    toolset_names: list[str] = field(default_factory=list)   # tools OFFERED this run (order kept)

    @property
    def convergence(self) -> str:
        """unprompted (emitted before any nudge) | nudged (only after) | never (no emit)."""
        if self.first_emit_step is None:
            return "never"
        return "nudged" if self.nudge_step is not None else "unprompted"

    @property
    def served_models(self) -> list[str]:
        """Distinct model ids the API RETURNED across turns. More than one = a provider
        reroute/requantize mid-run: a data-quality flag, not a value to average over."""
        return sorted({s.served_model for s in self.steps if s.served_model})

    @property
    def system_fingerprints(self) -> list[str]:
        """Distinct provider build ids seen across turns (often empty on Together)."""
        return sorted({s.system_fingerprint for s in self.steps if s.system_fingerprint})


def run_agent(messages: list[dict], cfg: AgentConfig, *, client=None) -> RunResult:
    """Drive the tool-calling loop for one (model, config) against a seeded `messages`
    array. `messages` is mutated in place as the transcript grows and is the same list
    returned on RunResult.messages. Never raises on a model/tool error -- a failed turn
    is captured on RunResult.fatal and ends the loop, because a run is data, not a crash.
    Pass `client` to inject a stub for offline testing; defaults to the llm.py seam."""
    client = client or _default_client
    caps = cfg.tool_caps or {}
    res = RunResult(model=cfg.model, messages=messages)
    res.base_url = str(getattr(client, "base_url", "") or "") or None
    res.temperature, res.max_tokens, res.seed = cfg.temperature, cfg.max_tokens, cfg.seed
    res.toolset_names = [t["function"]["name"] for t in cfg.toolset]
    ev_ids: list[str] = []
    worksheet_ok = False

    for n in range(1, cfg.max_steps + 1):
        # seed is sent only when set: the OpenAI client serializes an explicit None as
        # JSON null, which some OpenAI-compatible servers reject.
        kwargs = dict(model=cfg.model, messages=messages, tools=cfg.toolset,
                      tool_choice="auto", temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        try:
            r = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 -- a model that rejects the toolset is a finding, not a crash
            res.fatal = f"{type(e).__name__}: {e}"
            res.stop_reason = "fatal"
            return res

        res.prompt_tokens += r.usage.prompt_tokens
        res.completion_tokens += r.usage.completion_tokens
        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        rec = StepRecord(
            n=n, finish_reason=r.choices[0].finish_reason,
            prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
            content=(msg.content or "").strip(),
            reasoning=(getattr(msg, "reasoning", None) or "").strip(),
            served_model=getattr(r, "model", None),
            system_fingerprint=getattr(r, "system_fingerprint", None),
        )

        if not tcs:                     # no tool calls this turn
            fr = r.choices[0].finish_reason
            # Truncated at the token cap before any tool call = cut off mid-thought, NOT a
            # final answer. Keep any partial content, nudge to wrap up, and continue (still
            # bounded by max_steps) so a long reasoning turn isn't mistaken for "done".
            if fr == "length" and n < cfg.max_steps:
                if (msg.content or "").strip():
                    messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "user", "content": _LENGTH_NUDGE})
                rec.recovery = "truncated (finish_reason=length); nudged to continue"
                res.steps.append(rec)
                continue
            rec.answer, rec.recovery = final_answer(msg, fr)
            res.steps.append(rec)
            res.stop_reason = "no_tool_call"
            break

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs]})

        images: list[tuple[str, bytes]] = []
        for tc in tcs:
            name = tc.function.name
            res.used[name] += 1
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": f"error: unparseable arguments: {e}"})
                rec.calls.append({"name": name, "args": tc.function.arguments[:120],
                                  "result": f"unparseable args: {e}"})
                continue

            cap = caps.get(name)
            if cap is not None and res.used[name] > cap:
                capped = (f"cap reached: {name} may be called at most {cap} times per run; "
                          "you have enough data -- move on.")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": capped})
                rec.calls.append({"name": name, "args": "(capped)", "result": capped[:160]})
                continue

            # THE GATE: in required mode, refuse emit_taf until a worksheet has passed.
            if name == "emit_taf" and cfg.worksheet_mode == "required" and not worksheet_ok:
                refuse = ("emit_taf refused: worksheet_mode=required. Submit a "
                          "submit_taf_worksheet that passes its completeness check first, "
                          "then emit.")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": refuse})
                rec.calls.append({"name": name, "args": "(gated)", "result": refuse[:160]})
                continue

            if name == "emit_taf" and res.first_emit_step is None:
                res.first_emit_step = n     # first convergence attempt, clean or not

            result = run_tool(name, args, db_path=cfg.db_path, evidence_ids=ev_ids or None)

            # EVIDENCE THREADING: tag a data-tool receipt with a fresh id the model can cite.
            receipt = result.text
            if cfg.evidence and name not in _NON_EVIDENCE and not receipt.startswith("error:"):
                ev_id = f"ev_{len(res.evidence) + 1:03d}"
                ev_ids.append(ev_id)
                res.evidence.append({"evidence_id": ev_id, "tool_name": name,
                                     "tool_args_json": json.dumps(args),
                                     "receipt_text": receipt.splitlines()[0][:200]})
                receipt = f"[evidence_id: {ev_id}]\n{receipt}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": receipt})

            label_line = result.text.splitlines()[0] if result.text else name
            images += [(label_line, im) for im in result.images]

            if name == "submit_taf_worksheet" and result.worksheet is not None:
                res.worksheet_findings = result.findings
                if not wksht.blocking_findings(result.findings):
                    res.worksheet, worksheet_ok = result.worksheet, True
            if name == "emit_taf" and result.taf is not None:
                res.last_taf = result.taf
                if not tafgen.validate(result.taf):
                    res.final_taf = result.taf
            rec.calls.append({"name": name, "args": json.dumps(args)[:160],
                              "result": result.text.splitlines()[0][:160],
                              "n_images": len(result.images),
                              "receipt": result.text,   # full receipt (the data the model saw)
                              "full_args": (json.dumps(args, indent=2)
                                            if name in ("submit_taf_worksheet", "emit_taf")
                                            else None)})

        # A tool reply is text-only in the OpenAI format, so images ride in a follow-up
        # user message (batched: one message for all of this turn's charts).
        if images:
            content = [{"type": "text", "text": "Images from the tool calls above, each "
                        "preceded by its tool's receipt line:"}]
            for label_line, im in images:
                content.append({"type": "text", "text": f"[image for: {label_line}]"})
                b64 = base64.b64encode(im).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{_image_mime(im)};base64,{b64}"}})
            messages.append({"role": "user", "content": content})

        # One-time convergence nudge: budget nearly spent and no emit attempt yet.
        if (cfg.step_budget_nudge and n == cfg.max_steps - 2
                and res.first_emit_step is None and res.nudge_step is None):
            messages.append({"role": "user", "content": _BUDGET_NUDGE})
            res.nudge_step = n

        res.steps.append(rec)
        if cfg.stop_on_clean_taf and res.final_taf is not None:
            res.stop_reason = "emitted_clean"
            break

    return res


def final_answer(msg, finish_reason: str | None) -> tuple[str, str | None]:
    """Pull the model's answer out of a completed (no-tool-call) message.

    A reasoning model can leave `content` EMPTY while spilling the whole answer
    into `reasoning`, and still stop cleanly (finish_reason 'stop', NOT 'length').
    Reading content alone then logs a CORRECT answer as blank — a silent scoring
    bug. Guard it: on empty content + clean stop, fall back to reasoning and
    return a flag so the caller can mark the run instead of recording a miss.
    Returns (answer_text, flag); flag is None when content was present as normal.
    """
    content = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning", None) or "").strip()
    if content:
        return content, None
    if reasoning and finish_reason == "stop":
        return reasoning, "recovered from reasoning field (content empty, clean stop)"
    if finish_reason == "length":
        return (
            "_(empty — ran out of tokens; raise MAX_TOKENS)_",
            "content empty: finish_reason=length",
        )
    return "_(empty — no content and no reasoning)_", "content empty: no reasoning either"


def tool_messages(call_id: str, result: ToolResult) -> list[dict]:
    """Turn a ToolResult into the messages to append after a tool call: the
    required text receipt (role 'tool'), plus — if the tool returned images — a
    follow-up 'user' message carrying each image as a base64 image_url, since a tool
    reply can't hold an image in the OpenAI format. Returns 1 or 2 messages. (Used by
    the simpler single-tool drivers; run_agent builds these messages inline so it can
    also thread evidence ids and batch multi-image turns.)"""
    msgs: list[dict] = [
        {"role": "tool", "tool_call_id": call_id, "content": result.text}
    ]
    if result.images:
        content: list[dict] = [
            {"type": "text", "text": "Image(s) from the tool call:"}
        ]
        for img in result.images:
            b64 = base64.b64encode(img).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_image_mime(img)};base64,{b64}"},
            })
        msgs.append({"role": "user", "content": content})
    return msgs


def window_conflict(windows: list) -> str | None:
    """windows: list of (tool_label, (start, end)) gathered across the WHOLE
    conversation. If more than one DISTINCT window is present, return an advisory
    note listing each — non-blocking; the model decides if it's intentional. Pure;
    the caller dedupes before injecting."""
    distinct: dict = {}
    for label, win in windows:
        distinct.setdefault(win, []).append(label)
    if len(distinct) <= 1:
        return None
    lines = [
        "Heads up: your tool calls are not all looking at the same time period. "
        "If that is intentional, carry on; otherwise re-query so the windows align:"
    ]
    for (start, end), labels in distinct.items():
        lines.append(
            f"  {', '.join(labels)}: {start:%Y-%m-%dT%H:%MZ} .. {end:%Y-%m-%dT%H:%MZ}"
        )
    return "\n".join(lines)
