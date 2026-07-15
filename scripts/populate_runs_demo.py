"""Populate a DEMO runs table with one synthetic agent run -- no model, no network.

Builds a plausible RunResult by hand (a two-step run: meteogram -> clean emit, with a
worksheet) and persists it via runlog.persist_run into a throwaway demo DuckDB under
data/. Used by test_runlog.py (imports example_result) and by notebooks/runs_explorer.ipynb
(imports populate) so the notebook always has a row to show. Idempotent by run_id.

  uv run python scripts/populate_runs_demo.py            # -> data/runs_demo.duckdb
"""

import argparse
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from forecaster import worksheet as w
from forecaster.agent import RunResult, StepRecord
from forecaster.config import settings
from forecaster.metar import CloudLayer
from forecaster.runlog import persist_run
from forecaster.tafgen import TafProduct, TafProductGroup

DEMO_DB = str(Path(settings.db_path).parent / "runs_demo.duckdb")
DEMO_ARTIFACTS = str(Path(settings.db_path).parent / "runs_demo_artifacts")

# A 1x1 PNG so the transcript carries a real base64 image data URL (as a live run would).
_PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

_STATION = "KBLV"
_MODEL = "MiniMaxAI/MiniMax-M3"
_ISSUE = datetime(2026, 7, 15, 15, 55)
_VALID_FROM = datetime(2026, 7, 15, 16, 0)
_VALID_TO = datetime(2026, 7, 16, 22, 0)


def _clean_taf() -> TafProduct:
    """A validate()-clean 30h civil TAF whose calendar matches _ISSUE."""
    return TafProduct(
        station=_STATION, issue_day=15, issue_hour=15, issue_minute=55,
        valid_from_day=15, valid_from_hour=16, valid_to_day=16, valid_to_hour=22,
        prevailing=TafProductGroup(wind_dir=240, wind_speed=10, vis_m=9999,
                                   clouds=[CloudLayer(cover="FEW", height_ft=10000)]),
        military=False,
    )


def example_result() -> RunResult:
    """A hand-built RunResult: 2 steps (get_trend image -> clean emit) + a worksheet."""
    messages = [
        {"role": "system", "content": "You are a USAF weather forecaster (AFMAN 15-124)."},
        {"role": "user", "content": f"Produce a 30h TAF for {_STATION} valid 151600Z."},
        {"role": "assistant", "content": "Checking the recent trend first.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_trend",
                                      "arguments": '{"station": "KBLV", "hours": 24}'}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "[evidence_id: ev_001]\nMeteogram for KBLV, last 24h; image follows."},
        {"role": "user", "content": [
            {"type": "text", "text": "[image for: Meteogram for KBLV, last 24h]"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}}]},
        {"role": "assistant", "content": "Diurnal wind cycle, dry. Emitting.",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "emit_taf", "arguments": '{"station": "KBLV"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "AFMAN check: no findings"},
    ]
    steps = [
        StepRecord(n=1, finish_reason="tool_calls", prompt_tokens=1200, completion_tokens=90,
                   content="Checking the recent trend first.", reasoning="",
                   calls=[{"name": "get_trend", "args": '{"station": "KBLV"}',
                           "result": "Meteogram for KBLV, last 24h", "n_images": 1}]),
        StepRecord(n=2, finish_reason="tool_calls", prompt_tokens=2600, completion_tokens=140,
                   content="Diurnal wind cycle, dry. Emitting.", reasoning="",
                   calls=[{"name": "emit_taf", "args": '{"station": "KBLV"}',
                           "result": "AFMAN check: no findings", "n_images": 0,
                           "receipt": "AFMAN check: no findings", "full_args": '{"station": "KBLV"}'}]),
    ]
    res = RunResult(model=_MODEL, messages=messages)
    res.steps = steps
    res.used = Counter({"get_trend": 1, "emit_taf": 1})
    res.prompt_tokens = 3800
    res.completion_tokens = 230
    res.final_taf = res.last_taf = _clean_taf()
    res.worksheet = w._example_worksheet()
    res.worksheet_findings = []
    res.evidence = [{"evidence_id": "ev_001", "tool_name": "get_trend",
                     "tool_args_json": '{"station": "KBLV"}', "receipt_text": "Meteogram for KBLV"}]
    res.stop_reason = "emitted_clean"
    res.first_emit_step = 2
    return res


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def populate(db_path: str = DEMO_DB, artifacts_dir: str = DEMO_ARTIFACTS) -> dict:
    """Persist the example run into the demo DB. Idempotent (fixed run_id)."""
    run_id = f"{_STATION}_{_VALID_FROM:%Y%m%dT%H%M}_{_MODEL.split('/')[-1]}"
    return persist_run(
        example_result(), run_id=run_id, station=_STATION, issue_time=_ISSUE,
        valid_from=_VALID_FROM, valid_to=_VALID_TO, worksheet_mode="advisory",
        experiment_id=f"{_STATION}_{_VALID_FROM:%Y%m%dT%H%M}", harness_git_sha=_git_sha(),
        evidence_mode="key_claims", db_path=db_path, artifacts_dir=artifacts_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate a demo runs table.")
    ap.add_argument("--db", default=DEMO_DB)
    ap.add_argument("--artifacts", default=DEMO_ARTIFACTS)
    args = ap.parse_args()
    summary = populate(args.db, args.artifacts)
    print("Persisted demo run:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nDemo DB: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
