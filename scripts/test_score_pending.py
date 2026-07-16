"""Pending-scoring self-test (M4 step 3). No model, no network: a temp benchmark DB
seeded with synthetic obs + an archived subject TAF + a paired human TAF + a pending
evaluation, then the score_taf --pending machinery driven directly.

Covers: the full scored path (status flip, provenance hashes, coverage manifest,
result tables populated for subject/human/persistence with headline<->tall-row
consistency), append-only idempotency on re-persist, the required-coverage gate
(skip stays pending) vs --allow-partial (status=partial), taf_id fallback through
the runs row, the unelapsed-window filter, and the batch aggregators.

Run: uv run python scripts/test_score_pending.py
"""

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # import the sibling score_taf.py

from score_taf import _coverage, cmd_pending, persist_scores, run as score_run  # noqa: E402

from forecaster import store  # noqa: E402
from forecaster.metar import parse as metar_parse  # noqa: E402
from forecaster.tafarchive import build_taf_row  # noqa: E402

TMP = tempfile.mkdtemp(prefix="score_pending_test_")
DB = str(Path(TMP) / "bench.duckdb")
ISSUE = datetime(2026, 7, 10, 9)
VF, VT = datetime(2026, 7, 10, 9), datetime(2026, 7, 10, 15)

SUBJECT_RAW = "TAF KXXX 100900Z 1009/1015 24010KT 9999 SKC"
HUMAN_RAW = "TAF KXXX 100900Z 1009/1015 20015G25KT 9999 SKC"

checks: list[tuple[str, bool, str]] = []


def check(label, cond, detail=""):
    checks.append((label, bool(cond), "" if cond else f"      {detail}"))


def mk_args(**over):
    a = argparse.Namespace(db=DB, backfill=None, allow_partial=False,
                           grace_hours=0.0, min_coverage=0.9,
                           scorers_list=["tafver", "amend", "skill"],
                           baselines_list=["persistence", "human"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def seed_obs(con, station, hours):
    """One parseable ob per given hour offset from VF (float offsets allowed), plus
    the 08:55 carry-in that anchors the persistence baseline."""
    raws = [f"{station} 100855Z 24008KT 10SM SKC 24/12 A2990"]
    for h in hours:
        hh, mm = VF.hour + int(h), int((h % 1) * 60)
        raws.append(f"{station} 10{hh:02d}{mm:02d}Z 24012KT 10SM SKC 26/12 A2992")
    batch = [metar_parse(r) for r in raws]
    store.insert_obs(con, batch, year=2026, month=7, source="test")


def table_count(con, table, sid=None):
    if sid is None:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return con.execute(f"SELECT count(*) FROM {table} WHERE scorer_run_id = ?",
                       [sid]).fetchone()[0]


def main() -> int:
    con = store.connect(DB)
    store.init_schema(con)
    store.init_scoring_schema(con)
    store.init_runs_schema(con)
    store.init_results_schema(con)

    # --- seed: full-coverage station KXXX (subject + human + pending evaluation) ---
    seed_obs(con, "KXXX", range(6))
    subj = build_taf_row(SUBJECT_RAW, issue_ref=ISSUE, producer_kind="artificial",
                         producer_name="test-model", source="agent_run", canonical=True)
    human = build_taf_row(HUMAN_RAW, issue_ref=ISSUE, producer_kind="human",
                          producer_name="test-unit", source="awc_poll", canonical=True)
    store.insert_taf(con, subj)
    store.insert_taf(con, human)
    store.insert_evaluation(con, {
        "evaluation_id": "ev_full", "station": "KXXX", "taf_id": subj["taf_id"],
        "valid_from": VF, "valid_to": VT, "status": "pending", "created_at": ISSUE})

    # taf_id-fallback evaluation: no taf_id on the spine row, resolved via runs
    store.insert_evaluation(con, {
        "evaluation_id": "ev_viaruns", "station": "KXXX",
        "valid_from": VF, "valid_to": VT, "status": "pending", "created_at": ISSUE})
    store.insert_run(con, {"run_id": "ev_viaruns", "station": "KXXX",
                           "taf_id": subj["taf_id"], "stop_reason": "emitted_clean"})

    # sparse-coverage station KYYY (2 of 6 hours) -> gate then partial
    seed_obs(con, "KYYY", [0, 1])
    sparse = build_taf_row(SUBJECT_RAW.replace("KXXX", "KYYY"), issue_ref=ISSUE,
                           producer_kind="artificial", source="agent_run", canonical=True)
    store.insert_taf(con, sparse)
    store.insert_evaluation(con, {
        "evaluation_id": "ev_sparse", "station": "KYYY", "taf_id": sparse["taf_id"],
        "valid_from": VF, "valid_to": VT, "status": "pending", "created_at": ISSUE})

    # unelapsed window -> must never be selected
    store.insert_evaluation(con, {
        "evaluation_id": "ev_future", "station": "KXXX", "taf_id": subj["taf_id"],
        "valid_from": datetime(2999, 1, 1), "valid_to": datetime(2999, 1, 2, 6),
        "status": "pending", "created_at": ISSUE})
    con.close()

    # --- coverage helper sanity ---
    cov = _coverage([{"obs_time": VF}, {"obs_time": VF.replace(hour=11)}], VF, VT)
    check("coverage: 2 of 6 hours -> 4 missing",
          cov["hours_total"] == 6 and cov["hours_with_obs"] == 2
          and len(cov["missing_hours"]) == 4,
          json.dumps(cov))

    # --- the pending pass (gate ON: sparse must be skipped, future must be ignored) ---
    rc = cmd_pending(mk_args())
    check("pending pass exits 0", rc == 0, f"rc={rc}")

    con = store.connect(DB, read_only=True)
    ev = store.evaluation(con, "ev_full")
    check("ev_full flipped to scored with provenance",
          ev["status"] == "scored" and ev["obs_hash"] and ev["truth_policy_hash"]
          and ev["profile_hash"] and ev["scored_at"] is not None,
          json.dumps({k: str(v)[:20] for k, v in ev.items()}))
    cov_m = json.loads(ev["coverage_manifest_json"])
    check("ev_full coverage manifest complete", cov_m["fraction"] == 1.0
          and cov_m["hours_with_obs"] == 6, json.dumps(cov_m))
    check("ev_viaruns scored via runs-row taf_id fallback",
          store.evaluation(con, "ev_viaruns")["status"] == "scored")
    check("ev_sparse still pending (failed required coverage != partial)",
          store.evaluation(con, "ev_sparse")["status"] == "pending")
    check("ev_future untouched",
          store.evaluation(con, "ev_future")["status"] == "pending")

    # --- result tables: subject + human + persistence rows for ev_full ---
    runs = con.execute("SELECT scorer_run_id, subject, taf_id, combined_earned, "
                       "combined_available, policy_hash FROM tafver_runs "
                       "WHERE evaluation_id = 'ev_full' ORDER BY subject").fetchall()
    subjects = sorted(r[1] for r in runs)
    check("tafver rows for human/persistence/subject",
          subjects == ["human", "persistence", "subject"], str(subjects))
    hum = next(r for r in runs if r[1] == "human")
    check("human row carries the human taf_id", hum[2] == human["taf_id"], str(hum[2]))
    for sid, subject, _tid, earned, avail, _ph in runs:
        got = con.execute("SELECT COALESCE(SUM(points_earned),0), "
                          "COALESCE(SUM(points_available),0) FROM tafver_hourly "
                          "WHERE scorer_run_id = ? AND status = 'scored'", [sid]).fetchone()
        check(f"tafver headline == SUM(tall rows) [{subject}]",
              abs(got[0] - earned) < 1e-9 and got[1] == avail,
              f"run {earned}/{avail} vs tall {got[0]}/{got[1]}")
    n_amend = con.execute("SELECT count(*) FROM tafamend_runs "
                          "WHERE evaluation_id = 'ev_full'").fetchone()[0]
    n_skill = con.execute("SELECT count(*) FROM tafskill_runs "
                          "WHERE evaluation_id = 'ev_full'").fetchone()[0]
    check("amend + skill runs persisted for all three producers",
          n_amend == 3 and n_skill == 3, f"amend={n_amend} skill={n_skill}")
    deltas = con.execute("SELECT deltas_json FROM tafskill_runs "
                         "WHERE evaluation_id = 'ev_full' AND subject = 'subject'"
                         ).fetchone()[0]
    check("subject skill row carries deltas vs persistence", deltas is not None)
    hourly_before = table_count(con, "tafver_hourly")

    # --- aggregators ---
    pts = store.tafver_points(con, subject="subject")
    check("tafver_points pools per element", len(pts) > 0
          and all("earned" in p and "available" in p for p in pts), str(pts)[:120])
    check("skill_errors + skill_cells readers run",
          isinstance(store.skill_errors(con), list)
          and isinstance(store.skill_cells(con), list))
    con.close()

    # --- idempotency: re-persist identical inputs -> zero new rows ---
    con = store.connect(DB)
    out = score_run(con, taf_id=subj["taf_id"],
                    scorers=["tafver", "amend", "skill"],
                    baselines=["persistence", "human"])
    created = persist_scores(con, "ev_full", out)
    check("re-persist creates zero scorer runs (append-only no-op)",
          created == {"tafver": 0, "amend": 0, "skill": 0}, str(created))
    check("re-persist adds zero tall rows",
          table_count(con, "tafver_hourly") == hourly_before)
    con.close()

    # --- allow-partial: the sparse evaluation scores as status=partial ---
    rc = cmd_pending(mk_args(allow_partial=True))
    con = store.connect(DB, read_only=True)
    evs = store.evaluation(con, "ev_sparse")
    check("allow-partial scores the sparse evaluation as partial",
          rc == 0 and evs["status"] == "partial",
          f"rc={rc} status={evs['status']}")
    check("partial coverage manifest records the gap",
          json.loads(evs["coverage_manifest_json"])["fraction"] < 0.9)
    check("ev_future still pending after both passes",
          store.evaluation(con, "ev_future")["status"] == "pending")
    con.close()

    # --- report ---
    print(f"\ntemp DB: {DB}")
    passed = sum(1 for _, ok, _ in checks if ok)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail:
            print(detail)
    print(f"\n{passed}/{len(checks)} passed.")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
