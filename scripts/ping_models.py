"""One-shot endpoint ping: confirm each matrix model RESOLVES and CAN CALL A TOOL.

A model string the provider does not recognize fails the whole cell at run time (recorded
as fatal, not a crash). This pings every DISTINCT model in schedule.MATRIX so a bad id is
caught BEFORE a cycle fires.

The ping sends a real toolset, because on a ROUTER a resolving model id is not enough. One
provider can expose several endpoints for the same model, and the cheaper quantization often
has NO tool support -- routing there strips the toolset and the agent silently degrades into
a model that never calls a tool (an infra fault that reads as a model failure). A text-only
ping cannot see that; a tool-calling ping fails loudly instead.

Resolve = the API returned without error. Tools = the model answered with a tool_call. A
model that resolves but will not call the tool is reported and counts as a FAIL, since every
cell in the matrix is a tool-calling run.

  uv run python scripts/ping_models.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # import the sibling schedule.py

from schedule import MATRIX  # noqa: E402

from forecaster.config import settings  # noqa: E402
from forecaster.llm import client, routing  # noqa: E402


# A minimal but REAL tool: the point is to exercise the tool-call path end to end, not to
# mirror the production toolset. Deliberately trivial so any tool-capable model calls it.
_PING_TOOL = [{
    "type": "function",
    "function": {
        "name": "report_ok",
        "description": "Report that the endpoint is reachable. Call this to answer.",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string", "description": "Always 'ok'."}},
            "required": ["status"],
        },
    },
}]


def main() -> int:
    models = sorted({c.model for c in MATRIX})
    print(f"Pinging {len(models)} model(s) on {settings.llm_base_url}")
    # On a router, a resolving model id is only half the check: confirm the pin took, since a
    # pin that is silently ignored is worse than none (it looks controlled and is not).
    pin = routing()
    print(f"provider pin: {settings.llm_provider_order or '(none -- router chooses)'}"
          f"  fallbacks={settings.llm_provider_allow_fallbacks}\n" if pin else
          "provider pin: (none -- router chooses, or endpoint does not route)\n")
    ok = 0
    for m in models:
        t0 = time.monotonic()
        try:
            r = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Call report_ok with status 'ok'."}],
                tools=_PING_TOOL, tool_choice="auto",
                max_tokens=64, temperature=0, **({"extra_body": pin} if pin else {}),
            )
            dt = time.monotonic() - t0
            served = getattr(r, "model", "?")
            prov = getattr(r, "provider", None)
            msg = r.choices[0].message
            tcs = msg.tool_calls or []
            fr = r.choices[0].finish_reason
            content = (msg.content or "").strip()[:40]
            called = [tc.function.name for tc in tcs]
            tag = " OK " if tcs else "TOOL"
            print(f"  [{tag}] {m}")
            print(f"         served={served} finish={fr} {dt:.1f}s "
                  f"tools={called or 'NONE CALLED'} content={content!r}")
            if prov:
                print(f"         provider={prov}")
            if tcs:
                ok += 1
            else:
                # Resolves but will not tool-call: on a router this is the tools=False
                # endpoint case, and every matrix cell would degrade silently.
                print("         ^ model resolved but did NOT call the tool -- check the "
                      "route's tool support before running a billed cell")
        except Exception as e:  # noqa: BLE001 -- a bad id / endpoint error is the thing we are testing for
            print(f"  [FAIL] {m}")
            print(f"         {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(models)} resolved AND tool-calling")
    return 0 if ok == len(models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
