"""Scoring driver (scoring-design sec 12). Loads truth once, runs the requested
scorers over a subject TAF plus baselines, writes one combined markdown report.

Two modes:
  - ad-hoc (report only): --taf-id / --taf-text / --taf-file scores one TAF and
    writes the markdown report; nothing is persisted.
  - --pending (M4 step 3, the post-validity pass): selects pending evaluations whose
    windows have elapsed, checks truth coverage (optionally backfilling the missing
    obs from IEM with --backfill iem), scores subject + baselines (persistence and the
    paired HUMAN routine TAF), persists the per-scorer result tables, and flips the
    evaluation to scored (or partial with --allow-partial). All writes run under the
    single-writer lock. An evaluation that fails required coverage stays PENDING --
    partial success and failed-required-coverage are distinct outcomes.

This is application-side evaluation code: no LLM, never in the agent's messages array.

Usage:
  uv run python scripts/score_taf.py --taf-id <archived-id> --scorers amend
  uv run python scripts/score_taf.py --taf-text 'TAF KBLV ...' --issue-date 2026-07-09 \\
      --scorers amend --baselines persistence
  uv run python scripts/score_taf.py --pending --backfill iem
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import iem, stations, store, tafamend, tafskill, tafver
from forecaster.config import settings
from forecaster.tafparse import parse
from forecaster.tafamend import AmendPolicy, TafAmendScore, score_amend
from forecaster.tafskill import SkillPolicy, TafSkillScore, score_skill, skill_deltas
from forecaster.tafver import TafverPolicy, TafverScore, obs_hash, score_tafver
from forecaster.tafstate import (
    TruthPolicy, absolute_validity, default_profile, persistence_taf, stable_hash,
)

PERSISTENCE_MAX_AGE_MIN = 90            # baseline lookback (sec 10 versioned default)
_UNBUILT = {}


def _persistence_anchor(obs: list[dict], valid_from: datetime) -> dict | None:
    """Last ob strictly BEFORE valid_from, no older than the lookback (sec 10) -- never
    the first in-window ob (that would be a post-start self-leak)."""
    prior = [o for o in obs if o["obs_time"] < valid_from]
    if not prior:
        return None
    anchor = max(prior, key=lambda o: o["obs_time"])
    if valid_from - anchor["obs_time"] > timedelta(minutes=PERSISTENCE_MAX_AGE_MIN):
        return None
    return anchor


def run(
    con,
    *,
    taf_id: str | None = None,
    raw: str | None = None,
    issue_ref: datetime | None = None,
    scorers: list[str] | None = None,
    baselines: list[str] | None = None,
    profile=None,
) -> dict:
    """Resolve the subject TAF + its truth window, run the requested scorers over the
    subject and each baseline, return a result dict (+ markdown report)."""
    scorers = scorers or ["amend"]
    baselines = baselines or ["persistence"]
    for s in scorers:
        if s in _UNBUILT:
            raise ValueError(f"scorer '{s}' is not built yet (arrives in {_UNBUILT[s]})")

    subject_row = None
    if taf_id:
        subject_row = store.taf(con, taf_id)
        if not subject_row:
            raise ValueError(f"taf_id not found: {taf_id}")
        taf = parse(subject_row.get("parse_body") or subject_row["raw_taf"])
        station = subject_row["station"]
        vf, vt = subject_row["valid_from_utc"], subject_row["valid_to_utc"]
        canonical = subject_row.get("canonical")
    else:
        taf = parse(raw)
        _, vf, vt = absolute_validity(taf, issue_ref)
        station, canonical = taf.station, False

    profile = profile or default_profile(station)
    # ONE policy object per scorer + the shared truth policy, constructed HERE so the
    # provenance hashes persisted later are the hashes of what actually scored.
    policies = {"truth": TruthPolicy(), "tafver": TafverPolicy(),
                "amend": AmendPolicy(), "skill": SkillPolicy()}
    obs = store.scoring_window(con, station, vf, vt)
    want_amend, want_skill = "amend" in scorers, "skill" in scorers
    want_tafver = "tafver" in scorers

    def score_one(name, ftaf, anchor=None, tid=None):
        kw = dict(profile=profile, truth_policy=policies["truth"])
        return {"name": name, "anchor": anchor, "taf_id": tid,
                "amend": (score_amend(ftaf, obs, vf, vt, policy=policies["amend"], **kw)
                          if want_amend else None),
                "skill": (score_skill(ftaf, obs, vf, vt, policy=policies["skill"], **kw)
                          if want_skill else None),
                "tafver": (score_tafver(ftaf, obs, vf, vt, policy=policies["tafver"], **kw)
                           if want_tafver else None)}

    unavailable = {"anchor": None, "taf_id": None, "amend": None, "skill": None, "tafver": None}
    results = [score_one("subject", taf, tid=taf_id)]
    if "human" in baselines:
        hrow = store.human_taf_for_window(con, station, vf)
        if hrow is not None and hrow["taf_id"] != taf_id:
            try:
                htaf = parse(hrow.get("parse_body") or hrow["raw_taf"])
                results.append(score_one("human", htaf, tid=hrow["taf_id"]))
            except Exception as e:  # noqa: BLE001 -- an unparseable human TAF is reported, not fatal
                results.append({"name": "human", **unavailable,
                                "error": f"{type(e).__name__}: {e}"})
        else:
            results.append({"name": "human", **unavailable})
    if "persistence" in baselines:
        anchor = _persistence_anchor(obs, vf)
        if anchor is not None:
            results.append(score_one("persistence", persistence_taf(anchor, vf, vt),
                                     anchor=anchor["obs_time"]))
        else:
            results.append({"name": "persistence", **unavailable})

    report = _markdown(station, vf, vt, canonical, obs, results)
    return {"station": station, "valid_from": vf, "valid_to": vt, "canonical": canonical,
            "subject_row": subject_row, "results": results, "report": report,
            "obs": obs, "profile": profile, "policies": policies}


def _amend_line(name: str, s: TafAmendScore | None) -> str:
    if s is None:
        return f"| {name} | baseline unavailable | | |"
    frac = "n/a" if s.in_spec_fraction is None else f"{s.in_spec_fraction:.2f}"
    return f"| {name} | {s.trigger_count} | {frac} | {s.hours_scored} |"


def _mae(stats, element: str) -> str:
    st = next((s for s in stats if s.element == element and s.bin == "overall"), None)
    return "n/a" if st is None else f"{st.mae:.1f} (n={st.n})"


def _skill_line(name: str, s: TafSkillScore | None) -> str:
    if s is None:
        return f"| {name} | baseline unavailable | | | |"
    mace = "n/a" if s.mace is None else f"{s.mace:.2f}"
    return (f"| {name} | {mace} | {_mae(s.element_stats, 'wind_speed')} | "
            f"{_mae(s.element_stats, 'ceiling')} | {_mae(s.element_stats, 'visibility')} |")


def _markdown(station, vf, vt, canonical, obs, results) -> str:
    md = [f"# TAF score -- {station}", "",
          f"- valid: {vf:%Y-%m-%dT%H:%MZ} .. {vt:%Y-%m-%dT%H:%MZ}",
          f"- canonical: {canonical}",
          f"- truth obs in reader: {len(obs)}", ""]

    if results[0]["amend"] is not None:
        md += ["## Comparison (amend)", "",
               "| producer | triggers | in_spec | hours_scored |",
               "|---|---|---|---|"]
        md += [_amend_line(r["name"], r["amend"]) for r in results]
        md.append("")
        md += _amend_detail(results[0]["amend"])

    if results[0]["tafver"] is not None:
        md += ["## Comparison (TAFVER)", "",
               "| producer | combined % | earned/available | provisional |",
               "|---|---|---|---|"]
        md += [_tafver_line(r["name"], r["tafver"]) for r in results]
        md.append("")
        md += _tafver_detail(results[0]["tafver"])

    if results[0]["skill"] is not None:
        md += ["## Comparison (skill)", "",
               "| producer | MACE | wind_speed MAE | ceiling MAE | vis MAE |",
               "|---|---|---|---|---|"]
        md += [_skill_line(r["name"], r["skill"]) for r in results]
        md.append("")
        md += _skill_detail(results)

    return "\n".join(md)


def _tafver_line(name: str, s: TafverScore | None) -> str:
    if s is None:
        return f"| {name} | baseline unavailable | | |"
    pct = "n/a" if s.combined_percent is None else f"{s.combined_percent:.1f}"
    return f"| {name} | {pct} | {s.combined_earned:.1f}/{s.combined_available} | {s.provisional} |"


def _tafver_detail(s: TafverScore) -> list[str]:
    md = ["## TAFVER detail (subject)", "",
          f"- combined: **{'n/a' if s.combined_percent is None else f'{s.combined_percent:.1f}%'}** "
          f"({s.combined_earned:.1f}/{s.combined_available} points)"
          f"{'  [PROVISIONAL policy]' if s.provisional else ''}",
          f"- obs_hash {s.obs_hash[:12]}  policy_hash {s.policy_hash[:12]}  "
          f"profile_hash {s.profile_hash[:12]}", ""]
    md += ["Per-element MOP (available rows):", "",
           "| element | earned | available | percent |", "|---|---|---|---|"]
    for es in s.element_summaries:
        if es.bucket == "ALL":
            pct = "n/a" if es.percent is None else f"{es.percent:.1f}"
            md.append(f"| {es.element} | {es.earned:.1f} | {es.available} | {pct} |")
    md.append("")
    if s.group_type_summaries:
        md += ["Group-type buckets (diagnostic; combined sums all):", "",
               "| bucket | earned | available | percent |", "|---|---|---|---|"]
        for g in s.group_type_summaries:
            pct = "n/a" if g.percent is None else f"{g.percent:.1f}"
            md.append(f"| {g.bucket} | {g.earned:.1f} | {g.available} | {pct} |")
        md.append("")
    return md


def _amend_detail(subj: TafAmendScore) -> list[str]:
    md = ["## Amend detail (subject)", "",
          f"- potential amendment triggers: **{subj.trigger_count}** "
          f"(after-amd-service excluded: {subj.triggers_after_amd_service})",
          f"- in-spec fraction: {subj.in_spec_fraction}",
          f"- per-rule episodes: {subj.per_rule_episodes or '{}'}", ""]
    if subj.triggers:
        md += ["Triggers:", ""]
        md += [f"- {t.onset:%d%H%MZ}: {', '.join(t.rules)}" for t in subj.triggers]
        md.append("")
    if subj.rule_episodes:
        md += ["Rule episodes:", ""]
        for e in subj.rule_episodes:
            flag = " (after-amd-service)" if e.after_amd_service else ""
            md.append(f"- {e.rule}: {e.onset:%d%H%MZ}..{e.end:%d%H%MZ} "
                      f"({e.hours}h){flag} -- {e.worst_detail or ''}")
        md.append("")
    md += ["## Not scored (honesty appendix)", ""]
    md += [f"- {rule}: {why}" for rule, why in subj.rules_not_scored.items()]
    md += ["", "_bulletin-only v1: a superseded bulletin accrues potential busts over hours it "
           "no longer covered; not directly comparable to a unit's real amendment count._", ""]
    return md


def _skill_detail(results) -> list[str]:
    subj = results[0]["skill"]
    md = ["## Skill detail (subject)", "",
          f"- MACE {subj.mace}  worst excursion {subj.worst_excursion}",
          f"- signed ordinal mean {subj.signed_mace_mean} "
          "(positive = forecast better than observed = the dangerous direction)",
          f"- hours scored {subj.hours_scored}, unavailable {subj.hours_unavailable}", ""]
    md += ["Element bias/MAE (overall):", "",
           "| element | n | bias | MAE |", "|---|---|---|---|"]
    for st in subj.element_stats:
        if st.bin == "overall":
            md.append(f"| {st.element} | {st.n} | "
                      f"{'' if st.bias is None else f'{st.bias:.2f}'} | {st.mae:.2f} |")
    md.append("")
    events = [c for c in subj.contingency if c.a + c.b + c.c > 0]
    if events:
        md += ["Event cells (only events that occurred or were forecast):", "",
               "| event | hit | miss | false_alarm | POD | FAR | HSS |",
               "|---|---|---|---|---|---|---|"]
        for c in events:
            md.append(f"| {c.event} | {c.a} | {c.c} | {c.b} | "
                      f"{'' if c.pod is None else f'{c.pod:.2f}'} | "
                      f"{'' if c.far is None else f'{c.far:.2f}'} | "
                      f"{'' if c.hss is None else f'{c.hss:.2f}'} |")
        md.append("")
    # benchmark deltas vs the persistence baseline, on matched hours
    base = next((r["skill"] for r in results[1:] if r["skill"] is not None), None)
    if base is not None:
        d = skill_deltas(subj, base)
        md += ["Benchmark deltas vs persistence (matched hours only):", ""]
        for el, dd in d["elements"].items():
            sk = dd.get("mae_skill")
            md.append(f"- {el}: mae_skill {'n/a (' + dd.get('reason', '') + ')' if sk is None else f'{sk:+.2f}'} "
                      f"(n={dd['n']})")
        if d["mace"] is not None:
            mk_sk = d["mace"].get("mace_skill")
            md.append(f"- MACE: mace_skill {'n/a' if mk_sk is None else f'{mk_sk:+.2f}'} "
                      f"(n={d['mace']['n']})")
        md.append("")
    return md


# ---------------------------------------------------------------------------
# Persistence + the --pending post-validity pass (M4 step 3)
# ---------------------------------------------------------------------------

def _coverage(obs: list[dict], vf: datetime, vt: datetime) -> dict:
    """Truth-coverage manifest over the half-open window: which whole hours have at
    least one in-window ob. Drives the required-coverage gate and is persisted on the
    evaluation row (sec 11)."""
    hours_total = max(int((vt - vf).total_seconds() // 3600), 0)
    have = {int((o["obs_time"] - vf).total_seconds() // 3600)
            for o in obs if vf <= o["obs_time"] < vt}
    missing = [h for h in range(hours_total) if h not in have]
    return {"hours_total": hours_total, "hours_with_obs": hours_total - len(missing),
            "fraction": round((hours_total - len(missing)) / hours_total, 4) if hours_total else 0.0,
            "missing_hours": [(vf + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ")
                              for h in missing]}


def _producer_meta(con, r: dict) -> dict:
    """producer_kind/name for one result entry: read from the archived TAF row when the
    entry has one; the synthetic persistence baseline has neither."""
    if r["name"] == "persistence":
        return {"producer_kind": "baseline",
                "producer_name": f"persistence-{PERSISTENCE_MAX_AGE_MIN}min"}
    if r.get("taf_id"):
        row = store.taf(con, r["taf_id"])
        if row:
            return {"producer_kind": row.get("producer_kind"),
                    "producer_name": row.get("producer_name")}
    return {"producer_kind": None, "producer_name": None}


def persist_scores(con, evaluation_id: str, out: dict) -> dict:
    """Write the per-scorer result rows (sec 11) for every scored producer in `out`
    (a run() result). Idempotent: identical inputs find the existing scorer run and
    skip -- append-only history. Returns {scorer: rows_created} counts."""
    pol = out["policies"]
    created = {"tafver": 0, "amend": 0, "skill": 0}
    subj_skill = out["results"][0].get("skill")
    pers_skill = next((r["skill"] for r in out["results"]
                       if r["name"] == "persistence" and r.get("skill") is not None), None)

    def meta_for(r, policy, scorer_version):
        pd = policy.model_dump()
        return {"evaluation_id": evaluation_id, "taf_id": r.get("taf_id"),
                "subject": r["name"], **_producer_meta(con, r),
                "policy_name": policy.name, "policy_version": policy.version,
                "policy": pd, "policy_hash": stable_hash(pd),
                "scorer_version": scorer_version}

    for r in out["results"]:
        if r.get("tafver") is not None:
            _, new = store.insert_tafver_result(
                con, meta_for(r, pol["tafver"], tafver.SCORER_VERSION),
                r["tafver"].model_dump())
            created["tafver"] += new
        if r.get("amend") is not None:
            _, new = store.insert_tafamend_result(
                con, meta_for(r, pol["amend"], tafamend.SCORER_VERSION),
                r["amend"].model_dump())
            created["amend"] += new
        if r.get("skill") is not None:
            deltas = (skill_deltas(subj_skill, pers_skill)
                      if r["name"] == "subject" and pers_skill is not None
                      and subj_skill is not None else None)
            _, new = store.insert_tafskill_result(
                con, meta_for(r, pol["skill"], tafskill.SCORER_VERSION),
                r["skill"].model_dump(), deltas=deltas)
            created["skill"] += new
    return created


def _score_pending_one(ev: dict, args) -> str:
    """Score one elapsed pending evaluation end-to-end. Returns the outcome:
    'scored' | 'partial' | 'skipped (<why>)'. Raises on unexpected errors (the
    caller records them and moves on).

    Runs entirely under ONE write_lock hold: DuckDB is single-writer across
    processes, and mixing read-only and RW opens mid-pass would race the poller /
    collector on the Pi. Scoring one evaluation is seconds of work (the IEM
    backfill is the slow path -- run --pending at a quiet time), so briefly
    queueing the other crons is the simple, correct trade."""
    ev_id, station = ev["evaluation_id"], ev["station"]
    vf, vt = ev["valid_from"], ev["valid_to"]

    with store.write_lock(args.db):
        con = store.connect(args.db)
        try:
            taf_id = ev.get("taf_id")
            if not taf_id:                  # older rows predate the taf_id column
                run_row = store.run(con, ev_id)
                taf_id = run_row.get("taf_id") if run_row else None
            if not taf_id:
                return "skipped (no taf_id resolvable; evaluation has no scorable TAF)"

            obs = store.scoring_window(con, station, vf, vt)
            cov = _coverage(obs, vf, vt)
            if cov["fraction"] < args.min_coverage and args.backfill == "iem":
                # Backfill truth gaps from the IEM archive (military fields ARE served
                # for METARs; TAFs are what IEM lacks). Widened by the carry-in /
                # terminator margins. iem.load opens its own conn to the same DB --
                # same process, same RW mode, so the instance is shared cleanly.
                iem.load(station, vf - timedelta(hours=2), vt + timedelta(hours=1),
                         db_path=args.db)
                obs = store.scoring_window(con, station, vf, vt)
                cov = _coverage(obs, vf, vt)

            complete = cov["fraction"] >= args.min_coverage
            if not complete and not args.allow_partial:
                return (f"skipped (coverage {cov['fraction']:.0%} < "
                        f"{args.min_coverage:.0%}; {len(cov['missing_hours'])} hour(s) "
                        "missing -- backfill or --allow-partial)")

            out = run(con, taf_id=taf_id, scorers=args.scorers_list,
                      baselines=args.baselines_list)
            status = "scored" if complete else "partial"
            store.init_results_schema(con)
            counts = persist_scores(con, ev_id, out)
            store.finalize_evaluation(
                con, ev_id, status=status, obs_hash=obs_hash(out["obs"]),
                truth_policy_json=out["policies"]["truth"].model_dump_json(),
                truth_policy_hash=stable_hash(out["policies"]["truth"].model_dump()),
                profile_snapshot_json=out["profile"].model_dump_json(),
                profile_hash=stable_hash(out["profile"].model_dump()),
                coverage_manifest_json=json.dumps(cov))
        finally:
            con.close()

    Path("logs").mkdir(exist_ok=True)
    path = Path("logs") / f"tafscore_{station}_{ev_id}.md"
    path.write_text(out["report"], encoding="utf-8")

    r0 = out["results"][0]
    head = []
    if r0["tafver"] is not None:
        head.append(f"TAFVER={r0['tafver'].combined_percent}")
    if r0["amend"] is not None:
        head.append(f"triggers={r0['amend'].trigger_count}")
    if r0["skill"] is not None:
        head.append(f"MACE={r0['skill'].mace}")
    print(f"  {ev_id}: {status} coverage={cov['fraction']:.0%} {' '.join(head)} "
          f"(new rows {counts}) -> {path}")
    return status


def cmd_pending(args) -> int:
    """The post-validity pass: score every pending evaluation whose window has been
    fully elapsed for at least --grace-hours (lets the trailing obs settle)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with store.write_lock(args.db):
        con = store.connect(args.db)
        try:
            store.init_scoring_schema(con)
            store.init_results_schema(con)
            pend = store.pending_evaluations(con, before=now - timedelta(hours=args.grace_hours))
        finally:
            con.close()
    print(f"[{now:%Y-%m-%dT%H:%MZ}] {len(pend)} pending evaluation(s) with elapsed windows")
    outcomes: dict[str, int] = {}
    failed = 0
    for ev in pend:
        try:
            outcome = _score_pending_one(ev, args)
        except Exception as e:  # noqa: BLE001 -- one bad evaluation must not kill the pass
            print(f"  {ev['evaluation_id']}: ERROR {type(e).__name__}: {e}")
            failed += 1
            continue
        key = outcome.split(" ")[0]
        outcomes[key] = outcomes.get(key, 0) + 1
        if key == "skipped":
            print(f"  {ev['evaluation_id']}: {outcome}")
    print(f"done: {outcomes or '{}'}" + (f", {failed} ERROR(s)" if failed else ""))
    return 1 if failed else 0


def _score_archive_one(row: dict, args) -> str:
    """Difficulty-score one elapsed archived HUMAN TAF standalone (no evaluation spine): score
    it against obs and persist TAFVER (+ requested scorers) under a synthetic evaluation_id.
    Returns 'scored' | 'skipped (<why>)'. Runs under one write_lock hold like _score_pending_one
    -- DuckDB is single-writer, so a difficulty pass briefly queues the poller/collector."""
    taf_id, station = row["taf_id"], row["station"]
    vf, vt = row["valid_from_utc"], row["valid_to_utc"]
    ev_id = store.archive_evaluation_id(taf_id)

    with store.write_lock(args.db):
        con = store.connect(args.db)
        try:
            store.init_results_schema(con)
            if not args.rescore and store.archive_difficulty_scored(con, taf_id):
                return "skipped (already scored; --rescore to force)"

            obs = store.scoring_window(con, station, vf, vt)
            cov = _coverage(obs, vf, vt)
            if cov["fraction"] < args.min_coverage and args.backfill == "iem":
                # archive-only sites have NO collect.py dual-write, so truth comes from IEM
                # (which serves military METARs). Same carry-in/terminator margins as --pending.
                iem.load(station, vf - timedelta(hours=2), vt + timedelta(hours=1),
                         db_path=args.db)
                obs = store.scoring_window(con, station, vf, vt)
                cov = _coverage(obs, vf, vt)
            if cov["fraction"] < args.min_coverage and not args.allow_partial:
                return (f"skipped (coverage {cov['fraction']:.0%} < {args.min_coverage:.0%}; "
                        "backfill or --allow-partial)")

            out = run(con, taf_id=taf_id, scorers=args.scorers_list,
                      baselines=args.baselines_list)
            counts = persist_scores(con, ev_id, out)
        finally:
            con.close()

    regime = (stations.ARCHIVE_BY_ICAO[station].regime
              if station in stations.ARCHIVE_BY_ICAO else "roster")
    r0 = out["results"][0]
    head = []
    if r0["tafver"] is not None:
        head.append(f"TAFVER={r0['tafver'].combined_percent}")
    if r0["amend"] is not None:
        head.append(f"triggers={r0['amend'].trigger_count}")
    print(f"  {station} {vf:%Y-%m-%dT%H:%MZ} [{regime}] coverage={cov['fraction']:.0%} "
          f"{' '.join(head)} (new rows {counts})")
    return "scored"


def cmd_archive_difficulty(args) -> int:
    """Standalone TAFVER difficulty pass over the wide archive net: score every elapsed
    archived HUMAN routine TAF against obs (no model run), building the per-site/per-hour
    difficulty map used to pick the hard subset for the model matrix."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    station = args.station.upper() if args.station else None
    with store.write_lock(args.db):
        con = store.connect(args.db)
        try:
            store.init_scoring_schema(con)
            store.init_results_schema(con)
            tafs = store.archived_human_tafs(
                con, before=now - timedelta(hours=args.grace_hours), station=station)
        finally:
            con.close()
    print(f"[{now:%Y-%m-%dT%H:%MZ}] {len(tafs)} elapsed archived human TAF(s) to difficulty-score"
          + (f" (station {station})" if station else ""))
    outcomes: dict[str, int] = {}
    failed = 0
    for row in tafs:
        try:
            outcome = _score_archive_one(row, args)
        except Exception as e:  # noqa: BLE001 -- one bad TAF must not kill the pass
            print(f"  {row['station']} {row.get('taf_id')}: ERROR {type(e).__name__}: {e}")
            failed += 1
            continue
        key = outcome.split(" ")[0]
        outcomes[key] = outcomes.get(key, 0) + 1
        if key == "skipped":
            print(f"  {row['station']} {row['valid_from_utc']:%Y-%m-%dT%H:%MZ}: {outcome}")
    print(f"done: {outcomes or '{}'}" + (f", {failed} ERROR(s)" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a TAF against observed truth.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--taf-id")
    src.add_argument("--taf-text")
    src.add_argument("--taf-file")
    src.add_argument("--pending", action="store_true",
                     help="score all pending evaluations with elapsed windows (persists)")
    src.add_argument("--archive-difficulty", action="store_true",
                     help="difficulty-score every elapsed archived human TAF standalone (persists)")
    ap.add_argument("--station", help="--archive-difficulty: limit to one ICAO")
    ap.add_argument("--rescore", action="store_true",
                    help="--archive-difficulty: re-score TAFs that already have a result")
    ap.add_argument("--issue-date", help="issue DATE (YYYY-MM-DD) for --taf-text/--taf-file")
    ap.add_argument("--scorers", default=None,
                    help="default: amend (ad-hoc) / tafver,amend,skill (--pending)")
    ap.add_argument("--baselines", default=None,
                    help="default: persistence (ad-hoc) / persistence,human (--pending)")
    ap.add_argument("--backfill", choices=["iem"],
                    help="--pending: backfill missing truth obs from this source")
    ap.add_argument("--allow-partial", action="store_true",
                    help="--pending: score below --min-coverage as status=partial")
    ap.add_argument("--grace-hours", type=float, default=1.0,
                    help="--pending: wait this long past valid_to before scoring")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="--pending: required fraction of window hours with an ob")
    ap.add_argument("--db", default=settings.db_path)
    args = ap.parse_args()

    if args.pending:
        default_scorers, default_baselines = "tafver,amend,skill", "persistence,human"
    elif args.archive_difficulty:
        default_scorers, default_baselines = "tafver,amend", "persistence"
    else:
        default_scorers, default_baselines = "amend", "persistence"
    args.scorers_list = (args.scorers or default_scorers).split(",")
    args.baselines_list = (args.baselines or default_baselines).split(",")
    if args.pending:
        return cmd_pending(args)
    if args.archive_difficulty:
        return cmd_archive_difficulty(args)

    raw = None
    if args.taf_text:
        raw = args.taf_text
    elif args.taf_file:
        raw = open(args.taf_file, encoding="utf-8").read()
    issue_ref = datetime.strptime(args.issue_date, "%Y-%m-%d") if args.issue_date else None
    if raw is not None and issue_ref is None:
        ap.error("--taf-text/--taf-file requires --issue-date")

    con = store.connect(args.db, read_only=True)
    out = run(con, taf_id=args.taf_id, raw=raw, issue_ref=issue_ref,
              scorers=args.scorers_list, baselines=args.baselines_list)
    con.close()

    Path("logs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("logs") / f"tafscore_{out['station']}_{stamp}.md"
    path.write_text(out["report"], encoding="utf-8")
    r0 = out["results"][0]
    head = f"{out['station']} {out['valid_from']:%Y-%m-%dT%H:%MZ}..{out['valid_to']:%H:%MZ}"
    if r0["tafver"] is not None:
        head += f"  TAFVER={r0['tafver'].combined_percent}"
    if r0["amend"] is not None:
        head += f"  triggers={r0['amend'].trigger_count}  in_spec={r0['amend'].in_spec_fraction}"
    if r0["skill"] is not None:
        head += f"  MACE={r0['skill'].mace}"
    print(head)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
