"""Worksheet-seam self-test: guardrails, semantic validate(), guide, and the sink.

Like test_tafgen.py this calls NO model and hits NO network -- the worksheet seam
(worksheet.py + the submit_taf_worksheet/check_taf sinks) is deterministic, so this
is a fast, free, repeatable correctness gate that doubles as documentation of what a
complete TAF worksheet looks like.

It checks, in order:
  - the worked example validate()s clean (required mode, key_claims) and resolves its
    evidence_refs against the ids it cites;
  - guardrails reject IMPOSSIBLE values (non-ICAO station, illegal enum) and COERCE the
    loose values a model actually emits (title-case enums);
  - semantic validate() catches an EMPTY worksheet, splits the advisory
    model_run_verification finding from the blocking ones, and flags the specific
    coherence errors the design calls out (ready_for_emit_taf while incomplete; a change
    group referencing a non-existent timeline period; missing/misresolved evidence);
  - worksheet_guide() is byte-stable and names the required data reviews;
  - the submit_taf_worksheet + check_taf SINKS accept the worked artifacts end-to-end
    and the worksheet survives a store round-trip.

A self-contained markdown report lands under logs/.
"""

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from forecaster import store, tools
from forecaster import worksheet as w


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _example_args() -> dict:
    """The worked worksheet as submit_taf_worksheet arguments."""
    return w._example_worksheet().model_dump(exclude_none=True)


_EVIDENCE = ["ev_001", "ev_002", "ev_003", "ev_004", "ev_005"]

checks: list[Check] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append(Check(name, passed, detail))


# 1. Worked example validates clean (required, key_claims) + evidence resolves.
ws = w._example_worksheet()
f_present = w.validate(ws, mode="required", evidence_mode="key_claims")
check("worked example: clean (presence)", f_present == [], "; ".join(f_present))
f_resolved = w.validate(ws, mode="required", evidence_mode="key_claims",
                        known_evidence_ids=_EVIDENCE)
check("worked example: evidence_refs resolve", f_resolved == [], "; ".join(f_resolved))

# 2. Guardrails: impossible values rejected.
def _rejects(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


check("guardrail: non-ICAO station rejected", _rejects(lambda: w.Task(station="klsv")))
check("guardrail: illegal hazard enum rejected",
      _rejects(lambda: w.Hazard(hazard="volcano", risk_level="low")))
check("guardrail: illegal risk_level rejected",
      _rejects(lambda: w.Hazard(hazard="wind", risk_level="extreme")))

# 3. Coercion: loose values a model emits are canonicalized, not rejected.
d = w.Driver(name="x", type="Synoptic", confidence=" HIGH ")
check("coercion: title-case enum -> canonical", d.type == "synoptic" and d.confidence == "high",
      f"type={d.type} confidence={d.confidence}")

# 4. Empty worksheet -> every required section flagged; MRV split from blocking.
empty = w.TafWorksheet()
f_empty = w.validate(empty)
required_sections = ["task", "data_review", "current_state", "forecast_drivers", "hazards",
                     "forecast_timeline", "sanity_checks", "taf_strategy", "uncertainty",
                     "final_assessment"]
missing = [s for s in required_sections if not any(x.startswith(s + ":") for x in f_empty)]
check("empty worksheet: all required sections flagged", missing == [], f"missing flags: {missing}")
mrv = [x for x in f_empty if x.startswith("model_run_verification:")]
blocking = w.blocking_findings(f_empty)
check("empty worksheet: MRV finding is advisory (excluded from blocking)",
      len(mrv) == 1 and all(not x.startswith("model_run_verification:") for x in blocking),
      f"mrv={len(mrv)} blocking={len(blocking)}")

# 5. ready_for_emit_taf true while incomplete is flagged.
ws_ready = w.TafWorksheet(final_assessment=w.FinalAssessment(
    forecast_summary="s", biggest_risk_to_accuracy="r", ready_for_emit_taf=True))
check("coherence: ready_for_emit_taf while incomplete flagged",
      any("ready_for_emit_taf is true" in x for x in w.validate(ws_ready)))

# 6. change_group referencing a non-existent timeline period is flagged.
ws_badref = w._example_worksheet()
ws_badref.taf_strategy.change_group_strategy[0].timeline_period_label = "NOPE"
check("coherence: dangling change_group -> timeline reference flagged",
      any("is not in forecast_timeline" in x for x in w.validate(ws_badref)))

# 7. Evidence: key_claims presence + misresolution.
ws_noev = w._example_worksheet()
for drv in ws_noev.forecast_drivers.primary_drivers:
    drv.evidence_refs = []
check("evidence: missing driver evidence_ref flagged (key_claims)",
      any("needs >=1 evidence_ref" in x for x in w.validate(ws_noev, evidence_mode="key_claims")))
check("evidence: off mode does not demand refs",
      not any("evidence_ref" in x for x in w.validate(ws_noev, evidence_mode="off")))
check("evidence: unresolvable ref flagged when ids are threaded",
      any("do not resolve" in x for x in w.validate(ws, known_evidence_ids=["ev_001"])))

# 8. Guide: byte-stable + names the required reviews.
g1, g2 = w.worksheet_guide(), w.worksheet_guide()
check("guide: byte-stable across calls", g1 == g2)
check("guide: names the four required reviews",
      all(t in g1 for t in ("get_trend", "get_map", "get_fcst_sounding", "get_point_forecast")))

# 9. Sink round-trip: submit_taf_worksheet accepts the worked artifact end-to-end.
r_ok = tools.run_tool("submit_taf_worksheet", _example_args(), evidence_ids=_EVIDENCE)
check("sink: submit_taf_worksheet accepts worked example",
      r_ok.worksheet is not None and r_ok.findings == [],
      r_ok.text.splitlines()[0])
r_bad = tools.run_tool("submit_taf_worksheet", {"task": {"station": "KLSV"}})
check("sink: incomplete worksheet returns findings, not a crash",
      bool(r_bad.findings) and "re-submit" in r_bad.text,
      f"{len(r_bad.findings)} findings")

# 10. check_taf sink: dry-run validate without emitting.
r_check = tools.run_tool("check_taf", {
    "station": "KBLV", "issue_day": 15, "issue_hour": 17, "issue_minute": 0,
    "valid_from_day": 15, "valid_from_hour": 18, "valid_to_day": 16, "valid_to_hour": 24,
    "prevailing": {"wind_dir": 240, "wind_speed": 9, "vis_m": 9999, "clouds": [], "qnh_inhg": 29.92},
    "max_temp": {"temp_c": 34, "day": 15, "hour": 22},
    "min_temp": {"temp_c": 21, "day": 16, "hour": 12}})
check("sink: check_taf dry-runs clean, sets no .taf",
      r_check.findings == [] and r_check.taf is None and "not emitted" in r_check.text)

# 11. Store round-trip: persist + read back + JSON re-parses to a TafWorksheet.
db = str(Path(tempfile.mkdtemp(prefix="ws_selftest_")) / "ws.duckdb")
con = store.connect(db)
store.init_worksheet_schema(con)
store.insert_worksheet(
    con, worksheet_id="wt1", worksheet_json=ws.model_dump_json(), station="KLSV",
    mode="advisory", evidence_mode="key_claims", status="accepted",
    checker_findings_json=json.dumps([]),
    evidence=[{"evidence_id": "ev_001", "tool_name": "get_latest_obs",
               "tool_args_json": json.dumps({"station": "KLSV"}), "receipt_text": "25017G22KT"}])
con.close()
con = store.connect(db, read_only=True)
row = store.worksheet(con, "wt1")
ev = store.worksheet_evidence(con, "wt1")
con.close()
reparsed = w.TafWorksheet(**json.loads(row["worksheet_json"])) if row else None
check("store: worksheet + evidence persist and re-parse",
      row is not None and len(ev) == 1 and reparsed is not None
      and reparsed.task.station == "KLSV")


def build_markdown() -> str:
    npass = sum(c.passed for c in checks)
    md = [
        "# Worksheet-Seam Self-Test (worksheet.py + sinks)",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"**{npass}/{len(checks)} checks passed.** No model or network -- pure "
        "guardrail/validate/guide/sink/store.",
        "",
        "| Check | Result | Detail |",
        "|-------|--------|--------|",
    ]
    for c in checks:
        md.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    md += ["", "## Worked example worksheet (renders clean)", "", "```json",
           json.dumps(w._example_worksheet().model_dump(exclude_defaults=True, exclude_none=True),
                      indent=2),
           "```", "", "## worksheet_guide() (injected into the driver system prompt)", "",
           "```text", w.worksheet_guide(), "```"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"worksheet_selftest_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("=== WORKSHEET-SEAM SELF-TEST ===")
for c in checks:
    print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  -- {c.detail}" if not c.passed else ""))
npass = sum(c.passed for c in checks)
print(f"\n{npass}/{len(checks)} passed. Report: {log_path}")
