"""Amendment-bust scorer self-test (scoring-design sec 8/13). No model, no network:
synthetic typed TAFs + observation dicts (store.window shape) with KNOWN outcomes.
Each doctrine rule is exercised in isolation (active_rules subset), then the three-
layer aggregation (episodes -> triggers) and amendment-service exclusion.

Run: uv run python scripts/test_tafamend.py
"""

import sys
from datetime import datetime
from pathlib import Path

from forecaster.tafparse import parse
from forecaster.tafstate import absolute_validity, default_profile, persistence_taf
from forecaster.tafamend import AmendPolicy, score_amend

REF = datetime(2026, 7, 9)


def mk(raw):
    taf = parse(raw)
    _, vf, vt = absolute_validity(taf, REF)
    return taf, vf, vt


def ob(t, **kw):
    base = dict(station="KXXX", obs_time=t, report_type="METAR", auto=False, cavok=False,
                wind_dir_deg=240, wind_dir_card=None, wind_speed=10, wind_gust=None,
                wind_unit="KT", visibility=None, vis_sm=6.0, vis_m=9999, vis_flag="P",
                ceiling_ft=None, vertical_visibility_ft=None, temp_c=None, dewpoint_c=None,
                altimeter_inhg=29.92, altimeter_hpa=None, weather=[], clouds=[], remarks=None,
                raw="", source="test", corrected=False)
    base.update(kw)
    return base


def hourly(vf, vt, **kw):
    """One ob at HH:00 for each window hour (each covers its full hour)."""
    return [ob(datetime(vf.year, vf.month, vf.day, h), **kw) for h in range(vf.hour, vt.hour)]


def check(label, cond, detail=""):
    return (label, bool(cond), "" if cond else f"      {detail}")


def only(*rules):
    return AmendPolicy(active_rules=list(rules))


# --------------------------------------------------------------------------- rules

def run_rules():
    r = []

    # clean TAF -> 0 triggers, in_spec 1.0
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC")
    s = score_amend(taf, hourly(vf, vt), vf, vt)
    r.append(check("clean TAF -> 0 triggers, in_spec 1.0",
                   s.trigger_count == 0 and s.in_spec_fraction == 1.0 and s.hours_scored == 6,
                   f"trig={s.trigger_count} in_spec={s.in_spec_fraction}"))

    # Rule 1 speed: 10 kt diff (exact) does NOT bust; 11 does (strict > 10)
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    s10 = score_amend(taf, hourly(vf, vt, wind_speed=20), vf, vt, policy=only("wind"))
    s11 = score_amend(taf, hourly(vf, vt, wind_speed=21), vf, vt, policy=only("wind"))
    r.append(check("Rule 1 speed: 10 kt in-spec, 11 kt busts",
                   s10.trigger_count == 0 and s11.trigger_count >= 1,
                   f"d10={s10.trigger_count} d11={s11.trigger_count}"))

    # Rule 1 direction: > 30 deg busts only when expected >= 15 kt
    strong, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24018KT 9999 SKC")
    weak, wvf, wvt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    sd_strong = score_amend(strong, hourly(vf, vt, wind_dir_deg=300, wind_speed=18), vf, vt,
                            policy=only("wind"))
    sd_weak = score_amend(weak, hourly(wvf, wvt, wind_dir_deg=300, wind_speed=10), wvf, wvt,
                          policy=only("wind"))
    r.append(check("Rule 1 dir: 60 deg busts at 18 kt, ignored at 10 kt",
                   sd_strong.trigger_count >= 1 and sd_weak.trigger_count == 0,
                   f"strong={sd_strong.trigger_count} weak={sd_weak.trigger_count}"))

    # Rule 1 gust-presence completion: obs gust 12 over fcst mean 10 -> fail; 9 over -> pass
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    g_fail = score_amend(taf, hourly(vf, vt, wind_gust=22), vf, vt, policy=only("wind"))
    g_pass = score_amend(taf, hourly(vf, vt, wind_gust=19), vf, vt, policy=only("wind"))
    r.append(check("Rule 1 gust completion: +12 busts, +9 in-spec",
                   g_fail.trigger_count >= 1 and g_pass.trigger_count == 0,
                   f"g22={g_fail.trigger_count} g19={g_pass.trigger_count}"))

    # VRB fixtures: no direction check when either side is non-numeric
    vrbf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 VRB18KT 9999 SKC")   # fcst VRB
    s_vrbf = score_amend(vrbf, hourly(vf, vt, wind_dir_deg=90, wind_speed=18), vf, vt,
                         policy=only("wind"))
    numf, nvf, nvt = mk("TAF KXXX 090900Z 0909/0912 24018KT 9999 SKC")  # obs VRB
    s_vrbo = score_amend(numf, hourly(nvf, nvt, wind_dir_deg=None, wind_dir_card="VRB",
                                      wind_speed=18), nvf, nvt, policy=only("wind"))
    r.append(check("Rule 1 VRB gate: forecast-VRB and observed-VRB never dir-scored",
                   s_vrbf.trigger_count == 0 and s_vrbo.trigger_count == 0,
                   f"vrbF={s_vrbf.trigger_count} vrbO={s_vrbo.trigger_count}"))

    # Category: E -> D busts; lower-of (good ceiling + sub-3SM vis) busts
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    s_ed = score_amend(taf, hourly(vf, vt, ceiling_ft=1500), vf, vt, policy=only("category"))
    s_lo = score_amend(taf, hourly(vf, vt, vis_m=3000, vis_sm=1.86, vis_flag=None), vf, vt,
                       policy=only("category"))
    r.append(check("category: E->D busts; lower-of sub-3SM vis busts",
                   s_ed.trigger_count >= 1 and s_lo.trigger_count >= 1,
                   f"ed={s_ed.trigger_count} lo={s_lo.trigger_count}"))

    # OCONUS substitution flips a 4900 m vis from in-spec (conus) to bust (oconus)
    pc = default_profile("KXXX")
    po = default_profile("KXXX")
    po.use_oconus_vis_substitutions = True
    s_c = score_amend(taf, hourly(vf, vt, vis_m=4900, vis_sm=3.04, vis_flag=None), vf, vt,
                      profile=pc, policy=only("category"))
    s_o = score_amend(taf, hourly(vf, vt, vis_m=4900, vis_sm=3.04, vis_flag=None), vf, vt,
                      profile=po, policy=only("category"))
    r.append(check("category OCONUS sub: 4900 m in-spec conus, busts oconus",
                   s_c.trigger_count == 0 and s_o.trigger_count >= 1,
                   f"conus={s_c.trigger_count} oconus={s_o.trigger_count}"))

    # Rule 5 altimeter: side-of-threshold + exact boundaries
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC QNH2992INS")
    def alt(v):
        return score_amend(taf, hourly(vf, vt, altimeter_inhg=v), vf, vt,
                           policy=only("altimeter")).trigger_count
    r.append(check("Rule 5 altimeter: 31.00 inclusive HIGH; 30.99 NORMAL",
                   alt(31.00) >= 1 and alt(30.99) == 0, f"3100={alt(31.00)} 3099={alt(30.99)}"))
    r.append(check("Rule 5 altimeter: 27.99 LOW busts; 28.00/28.01 NORMAL in-spec",
                   alt(27.99) >= 1 and alt(28.00) == 0 and alt(28.01) == 0,
                   f"2799={alt(27.99)} 2800={alt(28.00)} 2801={alt(28.01)}"))

    return r


def run_timing_and_tempo():
    r = []

    # Rule 7 TS timing: onset 25 min early in-spec@tol30, 35 min busts; tol=0 busts both
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC "
                     "TEMPO 0910/0912 -TSRA")
    def ts_run(onset_min, tol=30):
        base = [ob(datetime(2026, 7, 9, h)) for h in (9, 12, 13, 14)]
        onset = datetime(2026, 7, 9, 9, onset_min)
        ts = [ob(onset, weather=["TSRA"]),
              ob(datetime(2026, 7, 9, 10), weather=["TSRA"]),
              ob(datetime(2026, 7, 9, 11), weather=["TSRA"])]
        return score_amend(taf, base + ts, vf, vt,
                           policy=AmendPolicy(active_rules=["thunderstorm"],
                                              ts_timing_tol_min=tol)).trigger_count
    r.append(check("Rule 7 TS: onset 25 min early in-spec, 35 min busts",
                   ts_run(35) == 0 and ts_run(25) >= 1,
                   f"early25(=09:35)={ts_run(35)} early35(=09:25)={ts_run(25)}"))
    r.append(check("Rule 7 TS: ts_timing_tol_min=0 busts the 25-min case too",
                   ts_run(35, tol=0) >= 1, f"tol0={ts_run(35, tol=0)}"))

    # Rule 8 TEMPO: becomes-predominant busts; correctly-forecast brief is in-spec
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC "
                     "TEMPO 0910/0912 2000 TSRA")
    pred = [ob(datetime(2026, 7, 9, h)) for h in (9, 12, 13, 14)] + [
        ob(datetime(2026, 7, 9, 10), vis_m=2000, vis_sm=1.24, vis_flag=None, weather=["TSRA"]),
        ob(datetime(2026, 7, 9, 11), vis_m=2000, vis_sm=1.24, vis_flag=None, weather=["TSRA"])]
    s_pred = score_amend(taf, pred, vf, vt, policy=AmendPolicy(active_rules=["tempo"]))
    brief = [ob(datetime(2026, 7, 9, h)) for h in (9, 10, 11, 12, 13, 14)] + [
        ob(datetime(2026, 7, 9, 10, 20), report_type="SPECI", vis_m=2000, vis_sm=1.24,
           vis_flag=None, weather=["TSRA"]),
        ob(datetime(2026, 7, 9, 10, 40))]     # back to clear after 20 min
    s_brief = score_amend(taf, brief, vf, vt, policy=AmendPolicy(active_rules=["tempo"]))
    r.append(check("Rule 8 TEMPO: predominant busts, brief occurrence in-spec",
                   s_pred.trigger_count >= 1 and s_brief.trigger_count == 0,
                   f"pred={s_pred.trigger_count} brief={s_brief.trigger_count}"))

    # Rule 9 BECMG clocked from window END: complete 20 min after in-spec, 40 min busts
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC "
                     "BECMG 0910/0911 30025KT")
    def becmg(first_match):
        base = [ob(datetime(2026, 7, 9, h)) for h in (9, 10)]
        base.append(ob(datetime(2026, 7, 9, 11)))               # 11:00 still old wind
        base.append(ob(first_match, wind_dir_deg=300, wind_speed=25))
        base += [ob(datetime(2026, 7, 9, h), wind_dir_deg=300, wind_speed=25)
                 for h in (12, 13, 14)]
        return score_amend(taf, base, vf, vt,
                           policy=AmendPolicy(active_rules=["change_timing"])).trigger_count
    r.append(check("Rule 9 BECMG: complete end+20 in-spec, end+40 busts",
                   becmg(datetime(2026, 7, 9, 11, 20)) == 0
                   and becmg(datetime(2026, 7, 9, 11, 40)) >= 1,
                   f"end20={becmg(datetime(2026,7,9,11,20))} "
                   f"end40={becmg(datetime(2026,7,9,11,40))}"))

    # Rule 9 FM: early change busts only when it persists >= 30 min
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC "
                     "FM091000 30025KT 9999 SKC")
    persist = [ob(datetime(2026, 7, 9, 9))]
    persist += [ob(datetime(2026, 7, 9, 9, m), wind_dir_deg=300, wind_speed=25) for m in (20, 55)]
    persist += [ob(datetime(2026, 7, 9, h), wind_dir_deg=300, wind_speed=25) for h in range(10, 15)]
    s_persist = score_amend(taf, persist, vf, vt, policy=AmendPolicy(active_rules=["change_timing"]))
    brief_early = [ob(datetime(2026, 7, 9, 9)),
                   ob(datetime(2026, 7, 9, 9, 20), wind_dir_deg=300, wind_speed=25),
                   ob(datetime(2026, 7, 9, 9, 30))]     # reverts -> early blip < 30 min
    brief_early += [ob(datetime(2026, 7, 9, h), wind_dir_deg=300, wind_speed=25)
                    for h in range(10, 15)]
    s_brief = score_amend(taf, brief_early, vf, vt, policy=AmendPolicy(active_rules=["change_timing"]))
    r.append(check("Rule 9 FM: early-and-persists busts, brief early blip does not",
                   s_persist.trigger_count >= 1 and s_brief.trigger_count == 0,
                   f"persist={s_persist.trigger_count} blip={s_brief.trigger_count}"))
    return r


def run_aggregation():
    r = []

    # 10-hour persistent category miss = 10 failed hours, 1 episode, 1 trigger
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0919 24010KT 9999 SKC")
    s = score_amend(taf, hourly(vf, vt, ceiling_ft=1500), vf, vt, policy=only("category"))
    cat_hours = sum(1 for x in s.hourly_results if x.rule == "category" and x.result == "fail")
    r.append(check("dedup: 10 failed category hours -> 1 episode, 1 trigger",
                   cat_hours == 10 and s.per_rule_episodes.get("category") == 1
                   and s.trigger_count == 1,
                   f"fails={cat_hours} eps={s.per_rule_episodes} trig={s.trigger_count}"))

    # two rules failing at the same onset hour merge into ONE trigger
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    s2 = score_amend(taf, hourly(vf, vt, ceiling_ft=1500, wind_speed=25), vf, vt,
                     policy=only("category", "wind"))
    r.append(check("merge: category+wind same onset -> 1 trigger, both rules",
                   s2.trigger_count == 1 and s2.triggers[0].rules == ["category", "wind"],
                   f"trig={s2.trigger_count} rules={s2.triggers[0].rules if s2.triggers else None}"))

    # LAST NO AMDS AFT -> later failures flagged, excluded from trigger_count
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC LAST NO AMDS AFT 091200")
    obs = [ob(datetime(2026, 7, 9, h)) for h in (9, 10, 11)]              # in-spec (E)
    obs += [ob(datetime(2026, 7, 9, h), ceiling_ft=1500) for h in (12, 13, 14)]   # D, after cutoff
    s3 = score_amend(taf, obs, vf, vt, policy=only("category"))
    r.append(check("after_amd_service: post-cutoff busts excluded from trigger_count",
                   s3.trigger_count == 0 and s3.triggers_after_amd_service >= 1
                   and s3.hours_after_amd_service == 3,
                   f"trig={s3.trigger_count} after={s3.triggers_after_amd_service} "
                   f"hrs_after={s3.hours_after_amd_service}"))

    # persistence baseline: the anchor ob replayed against itself is in-spec
    anchor = ob(datetime(2026, 7, 9, 8, 55))
    ptaf = persistence_taf(anchor, vf, vt)
    sp = score_amend(ptaf, hourly(vf, vt), vf, vt)
    r.append(check("persistence baseline scores sane (self-consistent -> 0 triggers)",
                   sp.trigger_count == 0 and sp.hours_scored > 0,
                   f"trig={sp.trigger_count} scored={sp.hours_scored}"))

    # deferred rules are reported, never silently passed
    r.append(check("deferred rules land in rules_not_scored",
                   "icing" in s.rules_not_scored and "turbulence" in s.rules_not_scored))

    # unavailable hour (no obs) never counts as a bust
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    thin = [ob(datetime(2026, 7, 9, 9))]     # only hour 09 covered
    st = score_amend(taf, thin, vf, vt, policy=only("category"))
    r.append(check("thin obs: uncovered hours unavailable, not busts",
                   st.trigger_count == 0 and st.hours_scored < 3,
                   f"trig={st.trigger_count} scored={st.hours_scored}"))
    return r


SECTIONS = [
    ("1. per-rule busts (wind / category / altimeter)", run_rules),
    ("2. timing + TEMPO rules (TS / TEMPO / BECMG / FM)", run_timing_and_tempo),
    ("3. aggregation (episodes / triggers / amd-service)", run_aggregation),
]


def main() -> int:
    print("=== TAFAMEND (amendment-bust) SELF-TEST ===")
    all_results, md = [], ["# tafamend self-test", ""]
    for title, fn in SECTIONS:
        print(f"\n[Section {title}]")
        md += [f"## Section {title}", ""]
        for label, ok, detail in fn():
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            if detail:
                print(detail)
            md.append(f"- [{'PASS' if ok else 'FAIL'}] {label}")
            all_results.append(ok)
        md.append("")
    passed, total = sum(all_results), len(all_results)
    md.append(f"**{passed}/{total} passed.**")
    Path("logs").mkdir(exist_ok=True)
    (Path("logs") / "tafamend_selftest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{passed}/{total} passed. Report: logs/tafamend_selftest.md")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
