"""TAFVER self-test (scoring-design sec 7/13). No model, no network: synthetic typed
TAFs + observation dicts (store.window shape) with KNOWN outcomes; assert exact points.
Four sections: wind MOPs, category/present-weather/altimeter MOPs, aggregation (anti-
averaging + buckets + category bias), and provenance (determinism + FITL refusal).

Run: uv run python scripts/test_tafver.py
"""

import sys
from datetime import datetime
from pathlib import Path

from forecaster.tafparse import parse
from forecaster.tafstate import absolute_validity, default_profile
from forecaster.tafver import TafverPolicy, fitl_value_added, score_tafver

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


def score_one(taf_raw, **obkw):
    taf, vf, vt = mk(taf_raw)
    return score_tafver(taf, [ob(datetime(2026, 7, 9, 9), **obkw)], vf, vt)


def scored(s, element):
    return next((r for r in s.rows if r.element == element and r.status == "scored"), None)


def rowof(s, element):
    return next((r for r in s.rows if r.element == element), None)


def check(label, cond, detail=""):
    return (label, bool(cond), "" if cond else f"      {detail}")


# ------------------------------------------------------------------- wind MOPs

def run_wind():
    r = []

    # wind speed: forecast 15 kt; |15-6|=9 correct, |15-5|=10 incorrect
    s9 = score_one("TAF KXXX 090900Z 0909/0910 24015KT 9999 SKC", wind_speed=6)
    s10 = score_one("TAF KXXX 090900Z 0909/0910 24015KT 9999 SKC", wind_speed=5)
    r.append(check("wind speed: 9 kt error correct, 10 kt error incorrect",
                   scored(s9, "wind_speed").points_earned == 1.0
                   and scored(s10, "wind_speed").points_earned == 0.0,
                   f"e9={scored(s9,'wind_speed').points_earned} "
                   f"e10={scored(s10,'wind_speed').points_earned}"))

    # direction tolerance KEYED TO OBSERVED SPEED: same 40 deg error flips at 15 kt
    slo = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=280, wind_speed=14)
    shi = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=280, wind_speed=16)
    r.append(check("dir: 40 deg error correct at obs 14 kt (tol 50), incorrect at 16 kt (tol 30)",
                   scored(slo, "wind_dir").points_earned == 1.0
                   and scored(shi, "wind_dir").points_earned == 0.0,
                   f"lo={scored(slo,'wind_dir').points_earned} hi={scored(shi,'wind_dir').points_earned}"))

    # exact boundaries: 30/31 at >=15 kt; 50/51 at <15 kt
    d30 = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=270, wind_speed=16)
    d31 = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=271, wind_speed=16)
    d50 = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=290, wind_speed=10)
    d51 = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=291, wind_speed=10)
    r.append(check("dir boundaries: 30 ok / 31 no (>=15 kt); 50 ok / 51 no (<15 kt)",
                   scored(d30, "wind_dir").points_earned == 1.0
                   and scored(d31, "wind_dir").points_earned == 0.0
                   and scored(d50, "wind_dir").points_earned == 1.0
                   and scored(d51, "wind_dir").points_earned == 0.0))

    # circular: forecast 350 vs observed 010 -> d=20 (not 340)
    sc = score_one("TAF KXXX 090900Z 0909/0910 35020KT 9999 SKC", wind_dir_deg=10, wind_speed=20)
    r.append(check("dir circular 350 vs 010 -> 20 deg, correct at 20 kt",
                   scored(sc, "wind_dir").points_earned == 1.0))

    # calm / VRB -> unavailable, never numeric zero
    calm = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_dir_deg=None,
                     wind_dir_card=None, wind_speed=0)
    vrb = score_one("TAF KXXX 090900Z 0909/0910 VRB15KT 9999 SKC", wind_dir_deg=240, wind_speed=10)
    r.append(check("dir calm(obs)/VRB(fcst) -> unavailable, not 0",
                   rowof(calm, "wind_dir").status == "unavailable"
                   and rowof(vrb, "wind_dir").status == "unavailable",
                   f"calm={rowof(calm,'wind_dir').reason} vrb={rowof(vrb,'wind_dir').reason}"))

    # gust four cases + 10/11 boundary
    both_absent = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC")
    obs_only = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC", wind_gust=25)
    fcst_only = score_one("TAF KXXX 090900Z 0909/0910 24010G30KT 9999 SKC")
    g10 = score_one("TAF KXXX 090900Z 0909/0910 24010G30KT 9999 SKC", wind_gust=20)
    g11 = score_one("TAF KXXX 090900Z 0909/0910 24010G30KT 9999 SKC", wind_gust=19)
    r.append(check("gust: both-absent 1, obs-only 0, fcst-only 0, |10| 1, |11| 0",
                   scored(both_absent, "wind_gust").points_earned == 1.0
                   and scored(obs_only, "wind_gust").points_earned == 0.0
                   and scored(fcst_only, "wind_gust").points_earned == 0.0
                   and scored(g10, "wind_gust").points_earned == 1.0
                   and scored(g11, "wind_gust").points_earned == 0.0,
                   f"ba={scored(both_absent,'wind_gust').points_earned} "
                   f"oo={scored(obs_only,'wind_gust').points_earned} "
                   f"fo={scored(fcst_only,'wind_gust').points_earned} "
                   f"g10={scored(g10,'wind_gust').points_earned} g11={scored(g11,'wind_gust').points_earned}"))
    return r


# --------------------------------------------- category / present wx / altimeter

def run_category_pw_alt():
    r = []

    # ceiling category match / mismatch
    match = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 BKN015", ceiling_ft=1500)  # D=D
    miss = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 BKN015", ceiling_ft=2500)   # D vs E
    r.append(check("ceiling category: match 1, mismatch 0",
                   scored(match, "ceiling").points_earned == 1.0
                   and scored(miss, "ceiling").points_earned == 0.0,
                   f"m={scored(match,'ceiling').points_earned} x={scored(miss,'ceiling').points_earned}"))

    # visibility category from a NUMERIC-METERS truth: obs 10SM (vis_m=9999, no P flag)
    # must resolve to the top band, not vis_category_unresolved; a restricted vis matches too
    vtop = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC",
                     visibility="10SM", vis_sm=10.0, vis_m=9999, vis_flag=None)
    vlow = score_one("TAF KXXX 090900Z 0909/0910 24010KT 3000 BKN015",
                     visibility="3000", vis_sm=1.86, vis_m=3000, vis_flag=None,
                     ceiling_ft=1500)
    r.append(check("visibility category: obs 10SM -> top band correct; 3000 m obs matches fcst 3000",
                   scored(vtop, "visibility").points_earned == 1.0
                   and scored(vlow, "visibility").points_earned == 1.0,
                   f"top={rowof(vtop,'visibility').status}/{scored(vtop,'visibility') and scored(vtop,'visibility').points_earned} "
                   f"low={rowof(vlow,'visibility').status}"))

    # present weather CSI: class-level identity + partial + empty
    ident = score_one("TAF KXXX 090900Z 0909/0910 24010KT 5000 RA SCT020", weather=["DZ"])   # liquid==liquid
    mixed = score_one("TAF KXXX 090900Z 0909/0910 24010KT 5000 RA SCT020", weather=["SN"])   # liquid vs frozen
    tsra = score_one("TAF KXXX 090900Z 0909/0910 24010KT 5000 -TSRA SCT020", weather=["RA"])  # {other,liquid} vs {liquid}
    empty = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC")
    r.append(check("present-wx CSI: RA/DZ liquid=1.0, RA/SN=0.0, TSRA/RA=0.5, empty->unavailable",
                   scored(ident, "present_weather").points_earned == 1.0
                   and scored(mixed, "present_weather").points_earned == 0.0
                   and abs(scored(tsra, "present_weather").points_earned - 0.5) < 1e-9
                   and rowof(empty, "present_weather").status == "unavailable",
                   f"id={scored(ident,'present_weather').points_earned} "
                   f"mx={scored(mixed,'present_weather').points_earned} "
                   f"ts={scored(tsra,'present_weather').points_earned} "
                   f"em={rowof(empty,'present_weather').reason}"))

    # altimeter: one-sided 0.05 boundary; observed-above-forecast correct
    a05 = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC QNH2992INS", altimeter_inhg=29.87)
    a06 = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC QNH2992INS", altimeter_inhg=29.86)
    above = score_one("TAF KXXX 090900Z 0909/0910 24010KT 9999 SKC QNH2992INS", altimeter_inhg=29.99)
    r.append(check("altimeter: -0.05 correct, -0.06 incorrect, observed-above correct",
                   scored(a05, "altimeter").points_earned == 1.0
                   and scored(a06, "altimeter").points_earned == 0.0
                   and scored(above, "altimeter").points_earned == 1.0,
                   f"a05={scored(a05,'altimeter').points_earned} "
                   f"a06={scored(a06,'altimeter').points_earned} up={scored(above,'altimeter').points_earned}"))

    # altimeter excluded on TEMPO groups
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC QNH2992INS "
                     "TEMPO 0909/0912 4800 -RA")
    st = score_tafver(taf, [ob(datetime(2026, 7, 9, h)) for h in (9, 10, 11)], vf, vt)
    tempo_alt = [x for x in st.rows if x.element == "altimeter" and x.group_type == "TEMPO"]
    r.append(check("altimeter excluded on TEMPO (distinct reason)",
                   tempo_alt and all(x.status == "unavailable"
                                     and x.reason == "tempo_altimeter_excluded" for x in tempo_alt),
                   f"n={len(tempo_alt)}"))
    return r


# ------------------------------------------------------------ aggregation

def run_aggregation():
    r = []

    # combined = pooled points, NOT the mean of per-element percentages.
    # Calm-wind hours make wind_dir availability differ from ceiling's -> the two diverge.
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC QNH2992INS")
    obs = [ob(datetime(2026, 7, 9, 9)),                                   # all correct
           ob(datetime(2026, 7, 9, 10), wind_dir_deg=None, wind_speed=0),  # calm -> dir unavail
           ob(datetime(2026, 7, 9, 11), wind_dir_deg=60, wind_speed=30)]   # speed+dir wrong
    s = score_tafver(taf, obs, vf, vt)
    all_pcts = [es.percent for es in s.element_summaries if es.bucket == "ALL" and es.percent is not None]
    naive_mean = sum(all_pcts) / len(all_pcts)
    pooled = 100 * s.combined_earned / s.combined_available
    r.append(check("combined = pooled points, not mean-of-element-percentages",
                   abs(s.combined_percent - pooled) < 1e-9
                   and abs(s.combined_percent - naive_mean) > 1e-6,
                   f"combined={s.combined_percent:.2f} naive={naive_mean:.2f}"))

    # INITIAL bucket is separate from FM (diagnostic), combined sums both
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC "
                     "FM091200 27015KT 9999 SKC")
    s2 = score_tafver(taf, [ob(datetime(2026, 7, 9, h)) for h in range(9, 15)], vf, vt)
    buckets = {g.bucket for g in s2.group_type_summaries}
    r.append(check("INITIAL and FM buckets reported separately",
                   "INITIAL" in buckets and "FM" in buckets, f"buckets={buckets}"))

    # category bias zero-denominator -> null (forecast category never observed)
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 OVC005")   # ceiling 500 = cat B
    s3 = score_tafver(taf, [ob(datetime(2026, 7, 9, h)) for h in (9, 10, 11)], vf, vt)  # obs SKC = E
    catB = next((c for c in s3.category_stats if c.element == "ceiling" and c.category == "B"), None)
    r.append(check("category bias zero-denominator -> null (fcst B, obs never B)",
                   catB and catB.fcst_hours > 0 and catB.obs_hours == 0 and catB.bias is None,
                   f"catB={catB}"))

    # BECMG post-completion: the baseline is ATTRIBUTED to the BECMG group (not INITIAL)
    # and scores against the NEW state (AFMAN: post-valid-time the BECMG prevails).
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC BECMG 0910/0911 30020KT 9999 SKC")
    obs = [ob(datetime(2026, 7, 9, 9)), ob(datetime(2026, 7, 9, 10))] + \
          [ob(datetime(2026, 7, 9, h), wind_dir_deg=300, wind_speed=20) for h in range(11, 15)]
    sb = score_tafver(taf, obs, vf, vt)
    post = [r for r in sb.rows if r.element == "wind_dir" and r.status == "scored"
            and r.interval_start.hour >= 12]
    r.append(check("BECMG post-completion: attributed to BECMG + scores the NEW state",
                   post and all(x.points_earned == 1.0 and x.group_type == "BECMG" for x in post),
                   f"n={len(post)} grp={set(x.group_type for x in post)}"))

    # BECMG transition hour: BEST-OF (obs matches the becoming state only) -> one row,
    # correct; NOT a double-counted old+becoming pair.
    obs2 = [ob(datetime(2026, 7, 9, 9))] + \
           [ob(datetime(2026, 7, 9, hr), wind_dir_deg=300, wind_speed=20) for hr in range(10, 15)]
    sb2 = score_tafver(taf, obs2, vf, vt)
    tr = [x for x in sb2.rows if x.element == "wind_dir" and x.interval_start.hour == 10]
    r.append(check("BECMG transition: best-of matches becoming, single row (no double-count)",
                   len(tr) == 1 and tr[0].points_earned == 1.0 and tr[0].group_type == "INITIAL",
                   f"n={len(tr)} earned={tr[0].points_earned if tr else None}"))

    # missing hour (no obs) vs missing element (civil TAF, no QNH) -> DISTINCT reasons
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    thin = score_tafver(taf, [ob(datetime(2026, 7, 9, 9))], vf, vt)      # later hours uncovered
    civ_row = [x for x in thin.rows if x.element == "altimeter" and x.status == "unavailable"]
    hour_gap = [x for x in thin.rows if x.status == "unavailable"
                and x.reason in ("no_obs", "coverage_gap")]
    r.append(check("missing hour (coverage) vs missing element (forecast_qnh_missing) distinct",
                   any(x.reason == "forecast_qnh_missing" for x in civ_row) and len(hour_gap) > 0,
                   f"qnh_missing={[x.reason for x in civ_row][:1]} hour_gap_rows={len(hour_gap)}"))
    return r


# ------------------------------------------------------------ provenance

def run_provenance():
    r = []

    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC QNH2992INS")
    obs = [ob(datetime(2026, 7, 9, h), ceiling_ft=1500, raw=f"KXXX 09{h}00Z A") for h in (9, 10, 11)]
    s1 = score_tafver(taf, obs, vf, vt)
    s2 = score_tafver(taf, obs, vf, vt)
    r.append(check("determinism: identical inputs -> identical combined + obs_hash + rows",
                   s1.combined_percent == s2.combined_percent and s1.obs_hash == s2.obs_hash
                   and len(s1.rows) == len(s2.rows)))

    # FITL value added: refused unless station/window/hashes all match (obs_hash keys on
    # the raw report text, so distinct raw -> distinct hash even if derived fields match)
    obs_diff = [ob(datetime(2026, 7, 9, h), ceiling_ft=800, raw=f"KXXX 09{h}00Z B") for h in (9, 10, 11)]
    s_diff = score_tafver(taf, obs_diff, vf, vt)
    refused = fitl_value_added(s1, s_diff)
    r.append(check("FITL pairing refused on obs_hash mismatch",
                   refused["value_added"] is None and "obs_hash" in refused["reason"],
                   f"{refused}"))

    # FITL value added: computed when all provenance matches (same obs, diff policy? no --
    # must share policy hash too; here identical scores -> value_added 0)
    paired = fitl_value_added(s1, s2)
    r.append(check("FITL value added computed when provenance matches (identical -> 0)",
                   paired["value_added"] == 0.0, f"{paired}"))

    # a different policy hash also refuses (policy is part of provenance)
    s_pol = score_tafver(taf, obs, vf, vt, policy=TafverPolicy(wind_speed_tol_kt=5))
    refused_pol = fitl_value_added(s1, s_pol)
    r.append(check("FITL pairing refused on policy_hash mismatch",
                   refused_pol["value_added"] is None and "policy_hash" in refused_pol["reason"],
                   f"{refused_pol}"))

    # provisional flag rides through from the profile
    r.append(check("provisional flag surfaced from profile",
                   s1.provisional == default_profile("KXXX").provisional))
    return r


SECTIONS = [
    ("1. wind MOPs (speed / direction / gust)", run_wind),
    ("2. category / present-weather / altimeter MOPs", run_category_pw_alt),
    ("3. aggregation (anti-averaging / buckets / bias)", run_aggregation),
    ("4. provenance (determinism / FITL refusal)", run_provenance),
]


def main() -> int:
    print("=== TAFVER SELF-TEST ===")
    all_results, md = [], ["# tafver self-test", ""]
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
    (Path("logs") / "tafver_selftest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{passed}/{total} passed. Report: logs/tafver_selftest.md")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
