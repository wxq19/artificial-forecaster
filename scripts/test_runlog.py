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

from forecaster.metar import CloudLayer
from forecaster.tafgen import TafProduct, TafProductGroup

from forecaster import store
from forecaster.agent import RunResult, StepRecord
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
# portability: the stored path is RELATIVE and resolves against the DB dir (DB + runs/ move together).
check("transcript_rel is relative", not Path(summary["transcript_rel"]).is_absolute(),
      summary["transcript_rel"])
check("relative transcript resolves via db_path",
      read_transcript(summary["transcript_rel"], db_path=DB) == res.messages)

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
              and row["transcript_path"] == summary["transcript_rel"] and row["taf_clean"] is True)

    taf = store.taf(con, summary["taf_id"])
    check("emitted TAF archived (producer_kind artificial, linked by run_id)",
          taf is not None and taf["producer_kind"] == "artificial" and taf["run_id"] == "RUN_A"
          and taf["station"] == _STATION, str(taf and taf.get("producer_kind")))

    ws = store.worksheet(con, summary["worksheet_id"])
    check("worksheet persisted", ws is not None and ws["station"] == _STATION)

    ev = store.evaluation(con, summary["evaluation_id"])
    check("pending evaluation written", ev is not None and ev["status"] == "pending"
          and ev["valid_from"] == _VALID_FROM, str(ev and ev.get("status")))
    check("clean run: tool_errors_json is '{}' (no failed calls)",
          row["tool_errors_json"] == "{}", str(row and row.get("tool_errors_json")))
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

# 4. Window tripwire: a TAF emitted with a validity one day off the requested window.
#    The TAF is still archived (taf_id set, tafs row present) but NO evaluation is created,
#    and the runs row records window_mismatch.
offset = example_result()
offset.final_taf = offset.last_taf = TafProduct(
    station=_STATION, issue_day=15, issue_hour=15, issue_minute=55,
    valid_from_day=16, valid_from_hour=16, valid_to_day=17, valid_to_hour=22,  # one day off
    prevailing=TafProductGroup(wind_dir=240, wind_speed=10, vis_m=9999,
                               clouds=[CloudLayer(cover="FEW", height_ft=10000)]),
    military=False,
)
offset.worksheet = None
s3 = _persist(offset, "RUN_C", tmp)
check("mismatch: taf_id set but evaluation_id None",
      s3["taf_id"] is not None and s3["evaluation_id"] is None,
      f"taf_id={s3['taf_id']} eval={s3['evaluation_id']}")
con = store.connect(DB, read_only=True)
try:
    row = store.run(con, "RUN_C")
    check("mismatch: runs row records window_mismatch + taf_id",
          row is not None and row["window_mismatch"] is not None and row["taf_id"] == s3["taf_id"],
          str(row and row.get("window_mismatch")))
    check("mismatch: tafs row exists, NO evaluations row",
          store.taf(con, s3["taf_id"]) is not None
          and con.execute("SELECT count(*) FROM evaluations WHERE evaluation_id = 'RUN_C'").fetchone()[0] == 0)
    # matching-window case (RUN_A) left window_mismatch NULL.
    a = store.run(con, "RUN_A")
    check("match: window_mismatch NULL for a matching-window run", a["window_mismatch"] is None)
finally:
    con.close()

# 5. Salt: byte-identical TAF text under a DIFFERENT run_id -> a DISTINCT taf_id + its own
#    tafs row (RUN_A and RUN_D emit the same _clean_taf() text).
s4 = _persist(example_result(), "RUN_D", tmp)
check("salt: identical TAF text, distinct run_id -> distinct taf_id",
      s4["taf_id"] is not None and s4["taf_id"] != summary["taf_id"],
      f"{summary['taf_id']} vs {s4['taf_id']}")
con = store.connect(DB, read_only=True)
try:
    tD = store.taf(con, s4["taf_id"])
    check("salt: RUN_D has its own tafs row + lineage",
          tD is not None and tD["run_id"] == "RUN_D")
finally:
    con.close()

# 6. Duration: a started_at anchor -> a plausible runs.duration_s (> 0, < timeout).
from datetime import datetime as _dt, timedelta as _td, timezone as _tz  # noqa: E402

s5 = persist_run(
    example_result(), run_id="RUN_E", station=_STATION, issue_time=_ISSUE,
    valid_from=_VALID_FROM, valid_to=_VALID_TO, worksheet_mode="advisory",
    db_path=DB, artifacts_dir=str(tmp / "runs"),
    started_at=_dt.now(_tz.utc).replace(tzinfo=None) - _td(seconds=42))
con = store.connect(DB, read_only=True)
try:
    row = store.run(con, "RUN_E")
    check("duration: runs.duration_s plausible (> 0, < timeout)",
          row is not None and row["duration_s"] is not None
          and 40 <= row["duration_s"] < 1500, str(row and row.get("duration_s")))
    # no started_at -> NULL duration (RUN_A path)
    check("duration: NULL when started_at omitted", store.run(con, "RUN_A")["duration_s"] is None)
finally:
    con.close()

# 7. Tool-error tracking: a step whose call receipt starts with 'error:' -> counted per tool.
failing = example_result()
failing.steps.insert(0, StepRecord(
    n=0, finish_reason="tool_calls", prompt_tokens=100, completion_tokens=10, content="", reasoning="",
    calls=[{"name": "get_map", "args": "{}", "result": "error: could not fetch chart", "n_images": 0}]))
s6 = persist_run(
    failing, run_id="RUN_F", station=_STATION, issue_time=_ISSUE,
    valid_from=_VALID_FROM, valid_to=_VALID_TO, worksheet_mode="advisory",
    db_path=DB, artifacts_dir=str(tmp / "runs"))
con = store.connect(DB, read_only=True)
try:
    row = store.run(con, "RUN_F")
    check("tool-error: failed call counted per tool", row["tool_errors_json"] == '{"get_map": 1}',
          str(row and row.get("tool_errors_json")))
finally:
    con.close()

npass = sum(c.passed for c in checks)
print("=== PERSISTENCE-LAYER SELF-TEST (runlog) ===")
for c in checks:
    print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  -- {c.detail}" if not c.passed else ""))
print(f"\n{npass}/{len(checks)} passed. Temp DB: {DB}")
