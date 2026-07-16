"""One-shot endpoint ping: confirm each matrix model RESOLVES on the configured endpoint.

A model string the provider does not recognize fails the whole cell at run time (recorded
as fatal, not a crash). This pings every DISTINCT model in schedule.MATRIX with a tiny call
so a bad id is caught BEFORE a cycle fires. A resolve = the API returns without error; empty
content from a reasoning model still counts (we only care that the id is served).

  uv run python scripts/ping_models.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # import the sibling schedule.py

from schedule import MATRIX  # noqa: E402

from forecaster.config import settings  # noqa: E402
from forecaster.llm import client  # noqa: E402


def main() -> int:
    models = sorted({c.model for c in MATRIX})
    print(f"Pinging {len(models)} model(s) on {settings.llm_base_url}\n")
    ok = 0
    for m in models:
        t0 = time.monotonic()
        try:
            r = client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=16, temperature=0,
            )
            dt = time.monotonic() - t0
            served = getattr(r, "model", "?")
            fr = r.choices[0].finish_reason
            content = (r.choices[0].message.content or "").strip()[:40]
            print(f"  [ OK ] {m}")
            print(f"         served={served} finish={fr} {dt:.1f}s content={content!r}")
            ok += 1
        except Exception as e:  # noqa: BLE001 -- a bad id / endpoint error is the thing we are testing for
            print(f"  [FAIL] {m}")
            print(f"         {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(models)} resolved")
    return 0 if ok == len(models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
