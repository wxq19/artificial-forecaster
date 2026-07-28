"""Self-test for the cross-evaluation readers + results_report aggregation.

No model, no network. Seeds a temp benchmark DB with TWO evaluations of DIFFERENT
window lengths (so their opportunity counts differ -- the condition under which pooling
and averaging disagree), scores them through the real score_taf path, then adds the two
contaminants the readers exist to exclude:

  - a SECOND scorer_version for the same evaluation (append-only rescore), and
  - an archive-difficulty scorer run under a synthetic `archdiff:` evaluation_id,
    which carries subject='subject' but is a HUMAN TAF with no evaluations-spine row.

Both have already produced wrong numbers in ad-hoc analysis, which is why they are
regression-tested rather than merely documented.

Run: uv run python scripts/test_results_report.py
"""

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts

import results_report as rr  # noqa: E402
from score_taf import persist_scores, run as score_run  # noqa: E402

from forecaster import store, tafver  # noqa: E402
from forecaster.metar import parse as metar_parse  # noqa: E402
from forecaster.tafarchive import build_taf_row  # noqa: E402

TMP = tempfile.mkdtemp(prefix="results_report_test_")
DB = str(Path(TMP) / "bench.duckdb")
ISSUE = datetime(2026, 7, 10, 9)

# Deliberately unequal windows: 12h and 4h. Pooling weights the long one more; averaging
# weights them equally. Everything about the pooling checks depends on this asymmetry.
LONG_VF, LONG_VT = datetime(2026, 7, 10, 9), datetime(2026, 7, 10, 21)
SHORT_VF, SHORT_VT = datetime(2026, 7, 10, 9), datetime(2026, 7, 10, 13)

# The short window's TAF is deliberately WRONG on wind (obs are 240/12) so the two
# evaluations score differently; equal scores would make the pooling test vacuous.
LONG_RAW = "TAF KLNG 100900Z 1009/1021 24012KT 9999 SKC"
SHORT_RAW = "TAF KSHT 100900Z 1009/1013 09025KT 0800 FG OVC002"
HUMAN_LONG = "TAF KLNG 100900Z 1009/1021 24010KT 9999 SKC"

checks: list[tuple[str, bool, str]] = []


def check(label, cond, detail=""):
    checks.append((label, bool(cond), "" if cond else f"      {detail}"))


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def mk_args(**over):
    a = argparse.Namespace(db=DB, backfill=None, allow_partial=False, grace_hours=0.0,
                           min_coverage=0.9, rescore=False,
                           scorers_list=["tafver", "amend", "skill"],
                           baselines_list=["persistence", "human"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def seed_obs(con, station, vf, n_hours):
    """One ob per hour across the window, plus the pre-window carry-in that anchors
    persistence. Steady 240/12 so the long TAF verifies well and the short one badly."""
    raws = [f"{station} 100855Z 24008KT 10SM SKC 24/12 A2990"]
    for h in range(n_hours):
        raws.append(f"{station} 10{vf.hour + h:02d}00Z 24012KT 10SM SKC 26/12 A2992")
    store.insert_obs(con, [metar_parse(r) for r in raws], year=2026, month=7, source="test")


def seed(con):
    seed_obs(con, "KLNG", LONG_VF, 13)
    seed_obs(con, "KSHT", SHORT_VF, 5)

    long_taf = build_taf_row(LONG_RAW, issue_ref=ISSUE, producer_kind="artificial",
                             producer_name="test-model", source="agent_run", canonical=True)
    short_taf = build_taf_row(SHORT_RAW, issue_ref=ISSUE, producer_kind="artificial",
                              producer_name="test-model", source="agent_run", canonical=True)
    human_taf = build_taf_row(HUMAN_LONG, issue_ref=ISSUE, producer_kind="human",
                              producer_name="test-unit", source="awc_poll", canonical=True)
    for row in (long_taf, short_taf, human_taf):
        store.insert_taf(con, row)

    store.insert_evaluation(con, {
        "evaluation_id": "ev_long", "station": "KLNG", "taf_id": long_taf["taf_id"],
        "valid_from": LONG_VF, "valid_to": LONG_VT, "status": "pending", "created_at": ISSUE})
    store.insert_evaluation(con, {
        "evaluation_id": "ev_short", "station": "KSHT", "taf_id": short_taf["taf_id"],
        "valid_from": SHORT_VF, "valid_to": SHORT_VT, "status": "pending", "created_at": ISSUE})

    # Orchestration provenance: run_evaluations joins on taf_id, and the cell slope
    # section needs a config_id.
    store.insert_run(con, {"run_id": "run_long", "station": "KLNG", "model": "test/model-a",
                           "config_id": "cfg_test", "taf_id": long_taf["taf_id"],
                           "n_steps": 5, "n_tool_calls": 11, "taf_clean": True,
                           "tools_used_json": '{"get_trend": 1, "emit_taf": 1}',
                           "stop_reason": "emitted_clean", "convergence": "unprompted",
                           "prompt_tokens": 1000, "completion_tokens": 200,
                           "producer_kind": "artificial"})
    store.insert_run(con, {"run_id": "run_short", "station": "KSHT", "model": "test/model-a",
                           "config_id": "cfg_test", "taf_id": short_taf["taf_id"],
                           "n_steps": 4, "n_tool_calls": 8, "taf_clean": True,
                           "tools_used_json": '{"get_trend": 1, "emit_taf": 1}',
                           "stop_reason": "emitted_clean", "convergence": "unprompted",
                           "prompt_tokens": 900, "completion_tokens": 150,
                           "producer_kind": "artificial"})
    return long_taf, short_taf, human_taf


def score_one(evaluation_id, taf_id, args):
    """Score + persist one evaluation, mirroring the --pending path."""
    with store.write_lock(DB):
        con = store.connect(DB)
        try:
            store.init_results_schema(con)
            out = score_run(con, taf_id=taf_id, scorers=args.scorers_list,
                            baselines=args.baselines_list)
            persist_scores(con, evaluation_id, out)
            store.finalize_evaluation(con, evaluation_id, status="scored")
        finally:
            con.close()
    return out


def main() -> int:
    con = store.connect(DB)
    store.init_schema(con)
    store.init_scoring_schema(con)
    store.init_runs_schema(con)
    store.init_results_schema(con)
    long_taf, short_taf, human_taf = seed(con)
    con.close()

    args = mk_args()
    base_version = tafver.SCORER_VERSION
    score_one("ev_long", long_taf["taf_id"], args)
    score_one("ev_short", short_taf["taf_id"], args)

    # --- contaminant 1: an archive-difficulty run (human TAF, subject slot, no spine) ---
    arch_id = store.archive_evaluation_id(human_taf["taf_id"])
    check("archive_evaluation_id uses the separable 'archdiff:' prefix",
          arch_id.startswith("archdiff:"), arch_id)
    with store.write_lock(DB):
        con = store.connect(DB)
        try:
            out = score_run(con, taf_id=human_taf["taf_id"], scorers=["tafver", "skill"],
                            baselines=["persistence"])
            persist_scores(con, arch_id, out)
        finally:
            con.close()

    con = store.connect(DB, read_only=True)

    # ---------------- readers: version filter ----------------
    versions = store.scorer_versions(con)
    check("scorer_versions returns the version present", versions == [base_version],
          str(versions))

    pts = store.evaluation_points(con, scorer_version=base_version)
    eids = {r["evaluation_id"] for r in pts}
    check("evaluation_points EXCLUDES the archdiff row (inner join to evaluations)",
          arch_id not in eids, str(sorted(eids)))
    check("evaluation_points returns exactly the two real evaluations",
          eids == {"ev_long", "ev_short"}, str(sorted(eids)))
    check("archdiff rows DO exist in the raw table (so exclusion is the reader's doing)",
          con.execute("SELECT count(*) FROM tafver_runs WHERE evaluation_id = ?",
                      [arch_id]).fetchone()[0] > 0)

    # --- the MIRROR of the exclusions above: the one reader that must INCLUDE archdiff ---
    # Everything else here joins `evaluations` and so drops these rows by construction;
    # the difficulty ranking joins `tafs` instead. If this ever returns empty while the
    # raw-table check above passes, the ranking is silently blind to its whole population.
    diff = store.archive_difficulty_points(con, scorer_version=base_version)
    check("archive_difficulty_points INCLUDES the archdiff row",
          len(diff) > 0, str(diff))
    check("archive_difficulty_points resolves the station via tafs (not evaluations)",
          all(r["station"] == human_taf["station"] for r in diff),
          str([r["station"] for r in diff]))
    check("archive_difficulty_points carries the graded TAF under subject='subject'",
          any(r["subject"] == "subject" for r in diff),
          str(sorted({r["subject"] for r in diff})))
    # Regression: a persistence baseline row has taf_id NULL, so resolving the station
    # via r.taf_id inner-joins every baseline away and the persistence column renders
    # empty -- losing the stable-regime vs hard-regime comparison the ranking exists for.
    check("archive_difficulty_points INCLUDES the NULL-taf_id persistence baseline",
          any(r["subject"] == "persistence" for r in diff),
          str(sorted({r["subject"] for r in diff})))
    check("archive_difficulty_points excludes REAL evaluations (no spine rows leak in)",
          all(r["n_tafs"] >= 1 for r in diff)
          and con.execute("SELECT count(*) FROM tafver_runs WHERE evaluation_id "
                          "NOT LIKE 'archdiff:%'").fetchone()[0] > 0
          and sum(r["n_tafs"] for r in diff if r["subject"] == "subject") == 1,
          str(diff))
    check("archive_difficulty_points is scorer_version-keyed",
          store.archive_difficulty_points(con, scorer_version="no-such-version") == [])
    check("archive_difficulty_points station filter narrows",
          store.archive_difficulty_points(con, scorer_version=base_version,
                                          station="ZZZZ") == [])

    dsec = "\n".join(rr.section_difficulty(diff))
    check("section_difficulty renders the station, not the archdiff id",
          human_taf["station"] in dsec and "archdiff" not in dsec, dsec)
    check("section_difficulty degrades cleanly with no rows",
          "No archive-difficulty rows" in "\n".join(rr.section_difficulty([])))

    subj = [r for r in pts if r["subject"] == "subject"]
    check("both evaluations produced subject rows", len(subj) == 2, str(len(subj)))

    # ---------------- the pooling identity ----------------
    earned = sum(r["earned"] for r in subj)
    available = sum(r["available"] for r in subj)
    pooled = rr.pct(earned, available)
    per_eval = [100.0 * r["earned"] / r["available"] for r in subj if r["available"]]
    mean_of = sum(per_eval) / len(per_eval)
    check("pct() implements sum(earned)/sum(available)",
          close(pooled, 100.0 * earned / available), f"{pooled}")
    check("the two evaluations have UNEQUAL opportunity counts (test is meaningful)",
          len({r["available"] for r in subj}) == 2,
          str(sorted(r["available"] for r in subj)))
    check("the two evaluations score DIFFERENTLY (test is meaningful)",
          len({round(p, 6) for p in per_eval}) == 2, str(per_eval))
    check("pooled != mean-of-percentages when weights differ (the historical bug)",
          not close(pooled, mean_of, 1e-6), f"pooled={pooled} mean={mean_of}")
    long_row = next(r for r in subj if r["evaluation_id"] == "ev_long")
    check("pooling weights toward the longer window",
          abs(pooled - 100.0 * long_row["earned"] / long_row["available"])
          < abs(mean_of - 100.0 * long_row["earned"] / long_row["available"]),
          f"pooled={pooled:.3f} mean={mean_of:.3f}")
    check("pct() returns None (not 0.0) for an empty denominator", rr.pct(0, 0) is None)

    # ---------------- reader cross-consistency ----------------
    els = store.element_points(con, scorer_version=base_version)
    el_subj = [r for r in els if r["subject"] == "subject"]
    check("element_points totals reconcile with evaluation_points",
          close(sum(r["earned"] for r in el_subj), earned)
          and sum(r["available"] for r in el_subj) == available,
          f"{sum(r['earned'] for r in el_subj)} vs {earned}")
    check("element_points excludes archdiff too",
          all(r["station"] in {"KLNG", "KSHT"} for r in els))

    lead = store.lead_points(con, scorer_version=base_version)
    lead_subj = [r for r in lead if r["subject"] == "subject"]
    check("lead_points totals reconcile with evaluation_points",
          close(sum(r["earned"] for r in lead_subj), earned)
          and sum(r["available"] for r in lead_subj) == available,
          f"{sum(r['available'] for r in lead_subj)} vs {available}")
    check("lead_points carries evaluation_id (so --paired can subset)",
          {r["evaluation_id"] for r in lead_subj} == {"ev_long", "ev_short"})
    check("lead_points carries obs_hour in 0..23 (the aliasing guard's input)",
          all(0 <= r["obs_hour"] <= 23 for r in lead_subj))
    check("lead_points carries the producing run's config_id",
          {r["config_id"] for r in lead_subj} == {"cfg_test"},
          str({r["config_id"] for r in lead_subj}))
    check("lead_hr is non-negative and within the longest window",
          all(0 <= r["lead_hr"] <= 12 for r in lead_subj))

    errs = store.element_errors(con, scorer_version=base_version)
    check("element_errors splits scored vs unavailable",
          all(r["n_scored"] is not None and r["n_unavailable"] is not None for r in errs)
          and any(r["n_unavailable"] > 0 for r in errs))
    check("element_errors reports no bias where nothing scored",
          all(r["bias"] is None for r in errs if not r["n_scored"]))

    lead_errs = store.lead_errors(con, scorer_version=base_version)
    check("lead_errors returns only scored rows with a lead",
          all(r["n_scored"] > 0 and r["lead_hr"] is not None for r in lead_errs))

    revs = store.run_evaluations(con)
    check("run_evaluations joins runs via taf_id for both evaluations",
          {r["evaluation_id"] for r in revs} == {"ev_long", "ev_short"},
          str({r["evaluation_id"] for r in revs}))
    check("run_evaluations carries orchestration provenance",
          all(r["n_tool_calls"] is not None and r["config_id"] == "cfg_test" for r in revs))
    check("run_evaluations returns only SCORED evaluations",
          all(r["status"] == "scored" for r in revs))
    con.close()

    # ---------------- contaminant 2: a second scorer_version ----------------
    bumped = str(int(base_version) + 1)
    tafver.SCORER_VERSION = bumped
    try:
        score_one("ev_long", long_taf["taf_id"], mk_args(rescore=True))
    finally:
        tafver.SCORER_VERSION = base_version

    con = store.connect(DB, read_only=True)
    check("scorer_versions now reports both, ascending",
          store.scorer_versions(con) == [base_version, bumped],
          str(store.scorer_versions(con)))

    old = [r for r in store.evaluation_points(con, scorer_version=base_version)
           if r["subject"] == "subject"]
    new = [r for r in store.evaluation_points(con, scorer_version=bumped)
           if r["subject"] == "subject"]
    unfiltered = [r for r in store.evaluation_points(con) if r["subject"] == "subject"]
    check("version filter isolates the original rows", len(old) == 2, str(len(old)))
    check("version filter isolates the rescored row", len(new) == 1, str(len(new)))
    # The unfiltered pathology is NOT an extra row -- the readers group by evaluation,
    # not by version, so a rescored evaluation's two versions collapse into ONE row whose
    # totals are the SUM of both. The row count looks right and the number is inflated,
    # which is exactly why this cannot be left to the caller to notice.
    unf_long = next(r for r in unfiltered if r["evaluation_id"] == "ev_long")
    old_long = next(r for r in old if r["evaluation_id"] == "ev_long")
    new_long = next(r for r in new if r["evaluation_id"] == "ev_long")
    check("UNFILTERED read keeps the row COUNT unchanged (the failure is silent)",
          len(unfiltered) == len(old) == 2, str(len(unfiltered)))
    check("UNFILTERED read DOUBLE-COUNTS a rescored evaluation in place",
          unf_long["available"] == old_long["available"] + new_long["available"],
          f"{unf_long['available']} != {old_long['available']}+{new_long['available']}")
    check("lead_points honours the version filter too",
          {r["evaluation_id"] for r in store.lead_points(con, scorer_version=bumped)
           if r["subject"] == "subject"} == {"ev_long"})

    # ---------------- the report builds end to end ----------------
    md = rr.build(con, scorer_version=base_version, paired=False)
    for heading in ("Headline TAFVER", "Lead-time degradation", "Orchestration",
                    "Tool use vs accuracy", "Directional bias per element"):
        check(f"report renders section: {heading}", heading in md)
    check("report states the scorer_version it used", f"scorer_version: **{base_version}**" in md)
    check("report headline carries the pooled figure", f"{pooled:.2f}" in md,
          f"expected {pooled:.2f}")
    check("report never mentions the archdiff evaluation", "archdiff" not in md)

    paired_md = rr.build(con, scorer_version=base_version, paired=True)
    check("--paired restricts to the human-paired evaluation",
          "human-paired subset" in paired_md and "KSHT" not in paired_md,
          "KSHT (no human TAF) should drop out of the paired view")
    con.close()

    passed = sum(1 for _, ok, _ in checks if ok)
    for label, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if detail:
            print(detail)
    print(f"\n{passed}/{len(checks)} passed.")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
