"""Scoring driver (scoring-design sec 12). Loads truth once, runs the requested
scorers over a subject TAF plus baselines, writes one combined markdown report.

M1 wires up `--scorers amend` + the persistence baseline; tafver/skill land in later
milestones (a requested-but-unbuilt scorer errors, never silently passes). This is
application-side evaluation code: no LLM, never in the agent's messages array.

Usage:
  uv run python scripts/score_taf.py --taf-id <archived-id> --scorers amend
  uv run python scripts/score_taf.py --taf-text 'TAF KBLV ...' --issue-date 2026-07-09 \\
      --scorers amend --baselines persistence
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from forecaster import store
from forecaster.config import settings
from forecaster.tafparse import parse
from forecaster.tafamend import TafAmendScore, score_amend
from forecaster.tafskill import TafSkillScore, score_skill, skill_deltas
from forecaster.tafver import TafverScore, score_tafver
from forecaster.tafstate import absolute_validity, default_profile, persistence_taf

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

    if taf_id:
        row = store.taf(con, taf_id)
        if not row:
            raise ValueError(f"taf_id not found: {taf_id}")
        taf = parse(row["raw_taf"])
        station, vf, vt = row["station"], row["valid_from_utc"], row["valid_to_utc"]
        canonical = row.get("canonical")
    else:
        taf = parse(raw)
        _, vf, vt = absolute_validity(taf, issue_ref)
        station, canonical = taf.station, False

    profile = profile or default_profile(station)
    obs = store.scoring_window(con, station, vf, vt)
    want_amend, want_skill = "amend" in scorers, "skill" in scorers
    want_tafver = "tafver" in scorers

    def score_one(name, ftaf, anchor=None):
        return {"name": name, "anchor": anchor,
                "amend": score_amend(ftaf, obs, vf, vt, profile=profile) if want_amend else None,
                "skill": score_skill(ftaf, obs, vf, vt, profile=profile) if want_skill else None,
                "tafver": score_tafver(ftaf, obs, vf, vt, profile=profile) if want_tafver else None}

    results = [score_one("subject", taf)]
    if "persistence" in baselines:
        anchor = _persistence_anchor(obs, vf)
        if anchor is not None:
            results.append(score_one("persistence", persistence_taf(anchor, vf, vt),
                                     anchor=anchor["obs_time"]))
        else:
            results.append({"name": "persistence", "anchor": None,
                            "amend": None, "skill": None, "tafver": None})

    report = _markdown(station, vf, vt, canonical, obs, results)
    return {"station": station, "valid_from": vf, "valid_to": vt,
            "results": results, "report": report}


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a TAF against observed truth.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--taf-id")
    src.add_argument("--taf-text")
    src.add_argument("--taf-file")
    ap.add_argument("--issue-date", help="issue DATE (YYYY-MM-DD) for --taf-text/--taf-file")
    ap.add_argument("--scorers", default="amend")
    ap.add_argument("--baselines", default="persistence")
    ap.add_argument("--db", default=settings.db_path)
    args = ap.parse_args()

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
              scorers=args.scorers.split(","), baselines=args.baselines.split(","))
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
