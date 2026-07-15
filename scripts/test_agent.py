"""Agent-loop self-test: pins run_agent's control flow with NO model and NO network.

The loop in agent.py is deterministic given (a) the model's turns and (b) each tool's
result -- so this stubs BOTH: a scripted `client` supplies the tool_calls the model would
emit, and a scripted `run_tool` (monkeypatched onto agent.run_tool) supplies the
ToolResults the tools would return. The loop's actual decision inputs -- tafgen.validate()
(clean vs dirty emit) and worksheet.blocking_findings() (accepted vs blocking worksheet) --
are the REAL functions, so this proves the loop wires them correctly, not that they work
(their own self-tests do that).

It checks, in order:
  - happy path: data tool -> evidence-tagged receipt + batched image (correct mime) ->
    clean emit -> stop_reason emitted_clean, final_taf set, token totals summed;
  - a dirty emit does NOT stop the loop; the next clean emit does;
  - unparseable tool arguments become a tool-error receipt without calling run_tool;
  - a per-tool cap returns feedback (and does NOT call run_tool) once exceeded;
  - required mode REFUSES emit_taf until a worksheet passes, then allows it;
  - a worksheet with blocking findings is not accepted (advisory mode);
  - final_answer recovery: content present -> that; content empty + reasoning + clean stop
    -> reasoning, with a recovery flag;
  - max_steps exhaustion and a client exception each set the right stop_reason.

A self-contained markdown report lands under logs/.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import forecaster.agent as agent
from forecaster.agent import AgentConfig, run_agent
from forecaster.metar import CloudLayer
from forecaster.tafgen import TafProduct, TafProductGroup
from forecaster.tools import ToolResult


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


checks: list[Check] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append(Check(name, passed, detail))


def C(cover: str, ft: int) -> CloudLayer:
    return CloudLayer(cover=cover, height_ft=ft)


# --- stub model client -------------------------------------------------------
def _tc(call_id: str, name: str, arguments: str):
    """A scripted tool_call, shaped like the OpenAI object the loop reads."""
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


def _resp(*, content=None, reasoning=None, tool_calls=None, finish_reason="tool_calls",
          ptok=100, ctok=50):
    """A scripted chat.completions response (one choice)."""
    msg = SimpleNamespace(content=content, reasoning=reasoning, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice],
                           usage=SimpleNamespace(prompt_tokens=ptok, completion_tokens=ctok))


class StubClient:
    """Replays scripted responses; a scripted BaseException is raised (the fatal path)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self._responses.pop(0)   # KeyError->IndexError here means a scenario under-scripted
        if isinstance(r, BaseException):
            raise r
        return r


# --- stub tool dispatch ------------------------------------------------------
def _install_run_tool(mapping: dict):
    """Point agent.run_tool at a scripted dispatch; returns it so .calls can be asserted."""
    calls = []

    def _rt(name, args, *, db_path=None, evidence_ids=None):
        calls.append((name, args))
        rt = mapping[name]
        return rt(args) if callable(rt) else rt

    _rt.calls = calls
    agent.run_tool = _rt
    return _rt


_REAL_RUN_TOOL = agent.run_tool
PNG = b"\x89PNG\r\n\x1a\n_fake"     # magic bytes -> image/png
GIF = b"GIF89a_fake"               # magic bytes -> image/gif


def _seed():
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def _cfg(**kw):
    return AgentConfig(model="stub/model", toolset=[], **kw)


# Real products so the REAL tafgen.validate() decides clean vs dirty.
CLEAN = TafProduct(
    station="KXYZ", issue_day=10, issue_hour=15, issue_minute=55,
    valid_from_day=10, valid_from_hour=16, valid_to_day=11, valid_to_hour=22,
    prevailing=TafProductGroup(wind_dir=240, wind_speed=10, vis_m=9999, clouds=[C("FEW", 10000)]),
    military=False,
)
BROKEN = TafProduct(
    station="KXXX", issue_day=1, issue_hour=12, issue_minute=0,
    valid_from_day=1, valid_from_hour=12, valid_to_day=2, valid_to_hour=10,   # 22h, not 30
    prevailing=TafProductGroup(wind_dir=275, wind_speed=10, wind_gust=8, vis_m=4800,
                               clouds=[C("BKN", 2000), C("SCT", 3000)]),
)


def _image_urls(messages: list[dict]) -> list[str]:
    """Every image data-URL across the transcript's follow-up user messages."""
    urls = []
    for m in messages:
        if m["role"] == "user" and isinstance(m.get("content"), list):
            urls += [p["image_url"]["url"] for p in m["content"] if p.get("type") == "image_url"]
    return urls


try:
    from forecaster import tafgen

    # 0. Fixture sanity: the products really are clean / dirty per validate().
    check("fixture: CLEAN validates clean", tafgen.validate(CLEAN) == [])
    check("fixture: BROKEN validates dirty", tafgen.validate(BROKEN) != [])

    # 1. Happy path: get_trend (image + evidence) -> clean emit.
    rt = _install_run_tool({
        "get_trend": ToolResult("Meteogram for KXYZ; image follows.", images=[PNG, GIF]),
        "emit_taf": ToolResult("AFMAN check: no findings", taf=CLEAN),
    })
    client = StubClient([
        _resp(tool_calls=[_tc("c1", "get_trend", '{"station": "KXYZ", "hours": 24}')],
              ptok=100, ctok=50),
        _resp(tool_calls=[_tc("c2", "emit_taf", '{"station": "KXYZ"}')], ptok=200, ctok=80),
    ])
    msgs = _seed()
    res = run_agent(msgs, _cfg(), client=client)
    check("happy: stop_reason emitted_clean", res.stop_reason == "emitted_clean", res.stop_reason)
    check("happy: final_taf is the clean product", res.final_taf is CLEAN)
    check("happy: token totals summed", res.prompt_tokens == 300 and res.completion_tokens == 130,
          f"{res.prompt_tokens}/{res.completion_tokens}")
    check("happy: one evidence row (data tool only)",
          len(res.evidence) == 1 and res.evidence[0]["tool_name"] == "get_trend",
          str(res.evidence))
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    check("happy: data receipt is evidence-tagged",
          any(m["content"].startswith("[evidence_id: ev_001]") for m in tool_msgs))
    check("happy: emit receipt is NOT evidence-tagged",
          any(m["content"].startswith("AFMAN check") for m in tool_msgs))
    urls = _image_urls(msgs)
    check("happy: both images batched with correct mime",
          len(urls) == 2 and urls[0].startswith("data:image/png")
          and urls[1].startswith("data:image/gif"), str([u[:15] for u in urls]))
    check("happy: assistant tool_calls message recorded",
          any(m["role"] == "assistant" and m.get("tool_calls") for m in msgs))

    # 2. A dirty emit continues the loop; the next clean emit stops it.
    _install_run_tool({"emit_taf": lambda a: ToolResult(
        "check", taf=(BROKEN if a.get("v") == "bad" else CLEAN))})
    client = StubClient([
        _resp(tool_calls=[_tc("d1", "emit_taf", '{"v": "bad"}')]),
        _resp(tool_calls=[_tc("d2", "emit_taf", '{"v": "ok"}')]),
    ])
    res = run_agent(_seed(), _cfg(), client=client)
    check("dirty-then-clean: emitted twice, stopped clean",
          res.used["emit_taf"] == 2 and res.stop_reason == "emitted_clean"
          and res.final_taf is CLEAN, f"{res.used['emit_taf']} {res.stop_reason}")

    # 3. Unparseable arguments -> tool-error receipt, run_tool NOT called.
    rt = _install_run_tool({"get_trend": ToolResult("unused")})
    client = StubClient([
        _resp(tool_calls=[_tc("u1", "get_trend", "{not json")]),
        _resp(content="done", tool_calls=None, finish_reason="stop"),
    ])
    msgs = _seed()
    res = run_agent(msgs, _cfg(), client=client)
    check("bad-args: run_tool not called", rt.calls == [], str(rt.calls))
    check("bad-args: tool-error receipt present",
          any(m["role"] == "tool" and m["content"].startswith("error: unparseable arguments")
              for m in msgs))
    check("bad-args: ends on no_tool_call", res.stop_reason == "no_tool_call", res.stop_reason)

    # 4. Per-tool cap: second get_map returns feedback, run_tool called only once.
    rt = _install_run_tool({"get_map": ToolResult("chart ok")})
    client = StubClient([
        _resp(tool_calls=[_tc("m1", "get_map", '{"name": "wpc_sfc"}')]),
        _resp(tool_calls=[_tc("m2", "get_map", '{"name": "wpc_sfc"}')]),
        _resp(content="done", finish_reason="stop"),
    ])
    msgs = _seed()
    res = run_agent(msgs, _cfg(tool_caps={"get_map": 1}), client=client)
    check("cap: run_tool called only under the cap", len(rt.calls) == 1, str(rt.calls))
    check("cap: over-cap receipt is feedback",
          any(m["role"] == "tool" and "cap reached" in m["content"] for m in msgs))
    check("cap: attempts still counted", res.used["get_map"] == 2, str(res.used))

    # 5. Required mode: emit refused until a worksheet passes, then allowed.
    ws_obj = object()   # loop only needs `is not None`; blocking_findings runs on findings
    rt = _install_run_tool({
        "emit_taf": ToolResult("AFMAN check: no findings", taf=CLEAN),
        "submit_taf_worksheet": ToolResult("worksheet accepted", worksheet=ws_obj, findings=[]),
    })
    client = StubClient([
        _resp(tool_calls=[_tc("g1", "emit_taf", "{}")]),                  # refused (no worksheet yet)
        _resp(tool_calls=[_tc("g2", "submit_taf_worksheet", "{}")]),      # passes
        _resp(tool_calls=[_tc("g3", "emit_taf", "{}")]),                  # now allowed
    ])
    msgs = _seed()
    res = run_agent(msgs, _cfg(worksheet_mode="required"), client=client)
    check("required: first emit refused before worksheet",
          any(m["role"] == "tool" and "emit_taf refused" in m["content"] for m in msgs))
    check("required: run_tool got submit + one emit (not the refused emit)",
          [n for n, _ in rt.calls] == ["submit_taf_worksheet", "emit_taf"], str(rt.calls))
    check("required: worksheet accepted + clean emit",
          res.worksheet is ws_obj and res.stop_reason == "emitted_clean")

    # 6. Advisory mode: a blocking worksheet finding is NOT accepted.
    _install_run_tool({"submit_taf_worksheet": ToolResult(
        "findings", worksheet=object(), findings=["current_state: implausible temp"])})
    client = StubClient([
        _resp(tool_calls=[_tc("b1", "submit_taf_worksheet", "{}")]),
        _resp(content="done", finish_reason="stop"),
    ])
    res = run_agent(_seed(), _cfg(worksheet_mode="advisory"), client=client)
    check("advisory: blocking worksheet not accepted",
          res.worksheet is None and res.worksheet_findings == ["current_state: implausible temp"],
          str(res.worksheet_findings))

    # 7. final_answer recovery: content present, then stranded-reasoning.
    client = StubClient([_resp(content="the answer", finish_reason="stop")])
    res = run_agent(_seed(), _cfg(), client=client)
    check("answer: content used directly",
          res.steps[-1].answer == "the answer" and res.steps[-1].recovery is None)
    client = StubClient([_resp(content="", reasoning="stranded here", finish_reason="stop")])
    res = run_agent(_seed(), _cfg(), client=client)
    check("answer: recovered from reasoning field",
          res.steps[-1].answer == "stranded here" and res.steps[-1].recovery is not None,
          res.steps[-1].recovery or "")

    # 8. max_steps exhaustion (never emits).
    _install_run_tool({"get_trend": ToolResult("ok")})
    client = StubClient([
        _resp(tool_calls=[_tc("s1", "get_trend", "{}")]),
        _resp(tool_calls=[_tc("s2", "get_trend", "{}")]),
    ])
    res = run_agent(_seed(), _cfg(max_steps=2), client=client)
    check("max_steps: stop_reason max_steps", res.stop_reason == "max_steps", res.stop_reason)
    check("max_steps: ran exactly the budget", len(res.steps) == 2, str(len(res.steps)))

    # 9. Fatal: the client raises.
    client = StubClient([RuntimeError("endpoint down")])
    res = run_agent(_seed(), _cfg(), client=client)
    check("fatal: stop_reason fatal + message captured, no steps",
          res.stop_reason == "fatal" and "endpoint down" in (res.fatal or "")
          and res.steps == [], res.fatal or "")

    # 10. Step-budget nudge + convergence classification.
    # never: nudge fires (n == max_steps-2 == 1), model still never emits.
    _install_run_tool({"get_trend": ToolResult("ok")})
    client = StubClient([_resp(tool_calls=[_tc(f"n{i}", "get_trend", "{}")]) for i in range(3)])
    msgs = _seed()
    res = run_agent(msgs, _cfg(max_steps=3, step_budget_nudge=True), client=client)
    check("nudge: fired at turn max_steps-2",
          res.nudge_step == 1 and any(m["role"] == "user" and m.get("content") == agent._BUDGET_NUDGE
                                      for m in msgs), str(res.nudge_step))
    check("convergence: never (no emit)", res.convergence == "never", res.convergence)

    # unprompted: emits clean before any nudge could fire.
    _install_run_tool({"emit_taf": ToolResult("clean", taf=CLEAN)})
    client = StubClient([_resp(tool_calls=[_tc("e1", "emit_taf", "{}")])])
    res = run_agent(_seed(), _cfg(max_steps=5, step_budget_nudge=True), client=client)
    check("convergence: unprompted (emit before nudge)",
          res.convergence == "unprompted" and res.nudge_step is None
          and res.first_emit_step == 1, res.convergence)

    # nudged: gathers past the nudge, then emits.
    _install_run_tool({"get_trend": ToolResult("ok"), "emit_taf": ToolResult("clean", taf=CLEAN)})
    client = StubClient([
        _resp(tool_calls=[_tc("g1", "get_trend", "{}")]),   # n=1: nudge fires (max_steps-2)
        _resp(tool_calls=[_tc("g2", "emit_taf", "{}")]),    # n=2: emits after the nudge
    ])
    res = run_agent(_seed(), _cfg(max_steps=3, step_budget_nudge=True), client=client)
    check("convergence: nudged (emit after nudge)",
          res.convergence == "nudged" and res.nudge_step == 1 and res.first_emit_step == 2,
          f"{res.convergence} nudge={res.nudge_step} emit={res.first_emit_step}")
finally:
    agent.run_tool = _REAL_RUN_TOOL   # restore the real dispatch


# --- report ------------------------------------------------------------------
npass = sum(c.passed for c in checks)
md = [
    "# Agent-loop self-test",
    f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
    "",
    f"**{npass}/{len(checks)} checks passed.** No model or network -- a scripted client + "
    "scripted run_tool exercise agent.run_agent's control flow directly.",
    "",
    "| Check | Result | Detail |",
    "| --- | --- | --- |",
]
for c in checks:
    md.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"agent_selftest_{datetime.now():%Y%m%d-%H%M%S}.md"
log_path.write_text("\n".join(md) + "\n")

print("=== AGENT-LOOP SELF-TEST ===")
for c in checks:
    print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  -- {c.detail}" if not c.passed else ""))
print(f"\n{npass}/{len(checks)} passed. Report: {log_path}")
