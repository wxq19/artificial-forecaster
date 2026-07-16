"""Persistence-layer self-test: runlog.persist_run + store runs schema. No model/network.

Persists a synthetic RunResult (from populate_runs_demo.example_result) into a THROWAWAY
DuckDB + temp artifacts dir, then asserts the whole provenance write path:
  - the transcript blob is on disk and round-trips the messages array (images included);
  - the runs row records model/tokens/steps/convergence + references (taf_id, worksheet_id,
    transcript_path);
  - the emitted TAF is archived in `tafs` (producer_kind artificial, linked by run_id);
  - the worksheet is persisted; a PENDING evaluation exists for the emitted TAF;
  - idempotency: persisting twice replaces, never duplicates;
  - a NO-EMIT run still writes a runs row (taf_id NULL) + transcript, and NO evaluation.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

from populate_runs_demo import (
    _ISSUE, _STATION, _VALID_FROM, _VALID_TO, example_result,
)

from forecaster import store
from forecaster.agent import RunResult
from forecaster.runlog import persist_run, read_transcript


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


checks: list[Check] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append(Check(name, passed, detail))


def _persist(res: RunResult, run_id: str, tmp: Path) -> dict:
    return persist_run(
        res, run_id=run_id, station=_STATION, issue_time=_ISSUE,
        valid_from=_VALID_FROM, valid_to=_VALID_TO, worksheet_mode="advisory",
        db_path=str(tmp / "demo.duckdb"), artifacts_dir=str(tmp / "runs"))


tmp = Path(tempfile.mkdtemp(prefix="runlog_test_"))
DB = str(tmp / "demo.duckdb")

# 1. Persist a clean, worksheet-backed run.
res = example_result()
summary = _persist(res, "RUN_A", tmp)

# transcript blob on disk (gzipped) + round-trips.
tp = Path(summary["transcript_path"])
check("transcript file written (gzipped)", tp.exists() and tp.name == "messages.json.gz", str(tp))
if tp.exists():
    loaded = read_transcript(tp)
    has_img = any(isinstance(m.get("content"), list)
                  and any(p.get("type") == "image_url" for p in m["content"]) for m in loaded)
    check("transcript round-trips the messages (image included)",
          loaded == res.messages and has_img)

con = store.connect(DB, read_only=True)
try:
    row = store.run(con, "RUN_A")
    check("runs row present", row is not None)
    if row:
        check("runs row: model + tokens + steps",
              row["model"] == res.model and row["prompt_tokens"] == 3800
              and row["completion_tokens"] == 230 and row["n_steps"] == 2,
              f"{row['model']} {row['prompt_tokens']}/{row['completion_tokens']} n={row['n_steps']}")
        check("runs row: convergence + stop_reason + tool count",
              row["convergence"] == "unprompted" and row["stop_reason"] == "emitted_clean"
              and row["n_tool_calls"] == 2, f"{row['convergence']} {row['stop_reason']}")
        check("runs row: references set (taf_id, worksheet_id, transcript_path, clean)",
              row["taf_id"] == summary["taf_id"] and row["worksheet_id"] == summary["worksheet_id"]
              and row["transcript_path"] == str(tp) and row["taf_clean"] is True)

    taf = store.taf(con, summary["taf_id"])
    check("emitted TAF archived (producer_kind artificial, linked by run_id)",
          taf is not None and taf["producer_kind"] == "artificial" and taf["run_id"] == "RUN_A"
          and taf["station"] == _STATION, str(taf and taf.get("producer_kind")))

    ws = store.worksheet(con, summary["worksheet_id"])
    check("worksheet persisted", ws is not None and ws["station"] == _STATION)

    ev = store.evaluation(con, summary["evaluation_id"])
    check("pending evaluation written", ev is not None and ev["status"] == "pending"
          and ev["valid_from"] == _VALID_FROM, str(ev and ev.get("status")))
finally:
    con.close()

# 2. Idempotency: persist RUN_A again -> replace, not duplicate.
_persist(example_result(), "RUN_A", tmp)
con = store.connect(DB, read_only=True)
try:
    n_runs = con.execute("SELECT count(*) FROM runs WHERE run_id = 'RUN_A'").fetchone()[0]
    n_tafs = con.execute("SELECT count(*) FROM tafs WHERE run_id = 'RUN_A'").fetchone()[0]
    check("idempotent: one run row + one taf after re-persist", n_runs == 1 and n_tafs == 1,
          f"runs={n_runs} tafs={n_tafs}")
finally:
    con.close()

# 3. A no-emit run: transcript + runs row (taf_id NULL), no evaluation.
noemit = example_result()
noemit.final_taf = noemit.last_taf = None
noemit.worksheet = None
noemit.stop_reason = "no_tool_call"
noemit.first_emit_step = None
s2 = _persist(noemit, "RUN_B", tmp)
check("no-emit: taf_id + evaluation_id are None", s2["taf_id"] is None and s2["evaluation_id"] is None)
check("no-emit: transcript still written", Path(s2["transcript_path"]).exists())
con = store.connect(DB, read_only=True)
try:
    row = store.run(con, "RUN_B")
    check("no-emit: runs row exists with NULL taf_id + convergence 'never'",
          row is not None and row["taf_id"] is None and row["convergence"] == "never")
finally:
    con.close()

npass = sum(c.passed for c in checks)
print("=== PERSISTENCE-LAYER SELF-TEST (runlog) ===")
for c in checks:
    print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  -- {c.detail}" if not c.passed else ""))
print(f"\n{npass}/{len(checks)} passed. Temp DB: {DB}")
