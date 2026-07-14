"""Skill scorer self-test (scoring-design sec 9/13). No model, no network: synthetic
typed TAFs + observation dicts (store.window shape) with KNOWN outcomes; assert exact
numbers. Four sections: axis 1 continuous errors, axis 1 QNH + TX/TN, axis 2 event
contingency, axis 2 timing + axis 3 ordinal + benchmark deltas + batch pooling.

Run: uv run python scripts/test_tafskill.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from forecaster.tafparse import parse
from forecaster.tafstate import absolute_validity
from forecaster.tafskill import (
    EVENT_CATALOG_V1, contingency_scores, score_skill, skill_deltas, _match_episodes,
)

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
    return [ob(datetime(vf.year, vf.month, vf.day, h), **kw) for h in range(vf.hour, vt.hour)]


def span(vf, n, temps=None, **kw):
    """n hourly obs from vf via timedelta (crosses midnight cleanly)."""
    out = []
    for i in range(n):
        extra = {}
        if temps is not None:
            extra["temp_c"] = temps(i)
        out.append(ob(vf + timedelta(hours=i), **{**kw, **extra}))
    return out


def find(rows, element, status="scored"):
    return next((r for r in rows if r.element == element and r.status == status), None)


def rowof(rows, element):
    return next((r for r in rows if r.element == element), None)


def check(label, cond, detail=""):
    return (label, bool(cond), "" if cond else f"      {detail}")


def t(h):
    return datetime(2026, 7, 9, 9) + timedelta(hours=h)


# --------------------------------------------------------------- axis 1 continuous

def run_axis1():
    r = []

    # perfect TAF -> every scored error 0, MACE 0
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    s = score_skill(taf, hourly(vf, vt), vf, vt)
    scored = [x for x in s.element_rows if x.grain == "hour" and x.status == "scored"]
    r.append(check("perfect TAF -> all scored errors 0, MACE 0",
                   scored and all(x.abs_error == 0 for x in scored) and s.mace == 0,
                   f"nscored={len(scored)} mace={s.mace}"))

    # signed error convention: forecast 20 kt over observed 10 -> +10
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24020KT 9999 SKC")
    s = score_skill(taf, hourly(vf, vt, wind_speed=10), vf, vt)
    ws = find(s.element_rows, "wind_speed")
    r.append(check("signed error = forecast - observed (over-forecast positive)",
                   ws and ws.signed_error == 10, f"signed={ws.signed_error if ws else None}"))

    # circular signed error: forecast 350 vs observed 010 -> -20
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 35020KT 9999 SKC")
    s = score_skill(taf, hourly(vf, vt, wind_dir_deg=10, wind_speed=20), vf, vt)
    wd = find(s.element_rows, "wind_dir")
    r.append(check("circular signed dir error 350 vs 010 -> -20",
                   wd and wd.signed_error == -20, f"signed={wd.signed_error if wd else None}"))

    # gust occurrence mismatch -> unavailable; the gust event cell stays intact
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010G25KT 9999 SKC")
    s = score_skill(taf, hourly(vf, vt), vf, vt)
    wg = rowof(s.element_rows, "wind_gust")
    gust_evt = next(c for c in s.contingency if c.event == "gust_ge_25")
    r.append(check("gust occurrence mismatch unavailable; gust_ge_25 false-alarm cell intact",
                   wg and wg.status == "unavailable" and wg.reason == "gust_occurrence_mismatch"
                   and gust_evt.b >= 1,
                   f"wg={wg.reason if wg else None} fa={gust_evt.b}"))

    # one-sided unlimited ceiling -> axis1 unavailable, but event + ordinal still fire
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    s = score_skill(taf, hourly(vf, vt, ceiling_ft=1500), vf, vt)
    cig = rowof(s.element_rows, "ceiling")
    cig_evt = next(c for c in s.contingency if c.event == "cig_lt_2000")
    r.append(check("one-sided unlimited ceiling routes to event+ordinal, not axis1",
                   cig and cig.reason == "one_sided_unlimited" and cig_evt.c >= 1 and s.mace > 0,
                   f"cig={cig.reason if cig else None} miss={cig_evt.c} mace={s.mace}"))

    # visibility both-at-cap -> both_unrestricted, zero error contribution
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT P6SM SKC")
    s = score_skill(taf, hourly(vf, vt), vf, vt)
    vis = rowof(s.element_rows, "visibility")
    r.append(check("vis forecast P6SM vs observed 9999 m -> both_unrestricted (no error)",
                   vis and vis.status == "unavailable" and vis.reason == "both_unrestricted",
                   f"vis={vis.reason if vis else None}"))
    return r


# --------------------------------------------------------------- axis 1 qnh + txtn

def run_qnh_txtn():
    r = []

    # QNH group minimum: forecast 29.92 vs observed min 29.90 -> +0.02 (hundredths)
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC QNH2992INS "
                     "TEMPO 0910/0911 4800 -RA")
    obs = [ob(datetime(2026, 7, 9, 9), altimeter_inhg=29.95),
           ob(datetime(2026, 7, 9, 10), altimeter_inhg=29.92),
           ob(datetime(2026, 7, 9, 11), altimeter_inhg=29.90)]
    s = score_skill(taf, obs, vf, vt)
    qnh_rows = [x for x in s.element_rows if x.element == "qnh"]
    qrow = qnh_rows[0] if qnh_rows else None
    r.append(check("QNH group-min 29.92 vs 29.90 -> +0.02; exactly one row (no TEMPO QNH)",
                   len(qnh_rows) == 1 and qrow.signed_error == 0.02 and qrow.grain == "group",
                   f"nrows={len(qnh_rows)} err={qrow.signed_error if qrow else None}"))

    # TX/TN value + timing over the 24 h temp window, anchored on cycle start
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/1015 24010KT 9999 SKC QNH2992INS "
                     "TX35/0918Z TN22/1005Z")
    temps = span(vf, 24, temps=lambda i: 33 if i == 9 else 18 if i == 20 else 25)
    s = score_skill(taf, temps, vf, vt)
    tx = find(s.element_rows, "temp_tx")
    tn = find(s.element_rows, "temp_tn")
    tx_t = find(s.element_rows, "temp_tx_timing")
    tn_t = find(s.element_rows, "temp_tn_timing")
    r.append(check("TX value 35-33=+2, TN value 22-18=+4",
                   tx and tx.signed_error == 2 and tn and tn.signed_error == 4,
                   f"tx={tx.signed_error if tx else None} tn={tn.signed_error if tn else None}"))
    r.append(check("TX/TN timing 0 h (forecast hour == extreme ob hour)",
                   tx_t and tx_t.signed_error == 0 and tn_t and tn_t.signed_error == 0,
                   f"txt={tx_t.signed_error if tx_t else None} tnt={tn_t.signed_error if tn_t else None}"))

    # sparse temp coverage -> unavailable (< 18 h with a temp)
    sparse = span(vf, 10, temps=lambda i: 25)
    s2 = score_skill(taf, sparse, vf, vt)
    txr = rowof(s2.element_rows, "temp_tx")
    r.append(check("sparse temp coverage (10 h) -> TX unavailable sparse_temp_coverage",
                   txr and txr.status == "unavailable" and txr.reason == "sparse_temp_coverage",
                   f"tx={txr.reason if txr else None}"))
    return r


# --------------------------------------------------------------- axis 2 events

def run_events():
    r = []

    # contingency math exact
    sc = contingency_scores(3, 1, 2, 4)
    r.append(check("contingency exact: POD .6 FAR .25 CSI .5 freq_bias .8 HSS .4",
                   abs(sc["pod"] - 0.6) < 1e-9 and abs(sc["far"] - 0.25) < 1e-9
                   and abs(sc["csi"] - 0.5) < 1e-9 and abs(sc["freq_bias"] - 0.8) < 1e-9
                   and abs(sc["hss"] - 0.4) < 1e-9, f"{sc}"))

    # zero denominators -> None, never 0/inf
    z = contingency_scores(0, 0, 0, 5)
    r.append(check("zero-denominator cells -> None (pod/far/csi/hss)",
                   z["pod"] is None and z["far"] is None and z["csi"] is None and z["hss"] is None,
                   f"{z}"))

    # via_tempo: prevailing misses, an active TEMPO alternate scores the hit
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC TEMPO 0909/0912 3000 -RA")
    s = score_skill(taf, hourly(vf, vt, vis_m=3000, vis_sm=1.86, vis_flag=None), vf, vt)
    hit_via = any(e.event == "vis_lt_3sm" and e.via_tempo and e.cell == "hit"
                  for e in s.event_hours)
    r.append(check("via_tempo hit: prevailing-only would miss, TEMPO alternate scores the hit",
                   hit_via))

    # observed TS unforecast -> event miss + a missed episode
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    obs = [ob(datetime(2026, 7, 9, 9)),
           ob(datetime(2026, 7, 9, 10), weather=["TSRA"]),
           ob(datetime(2026, 7, 9, 11), weather=["TSRA"])]
    s = score_skill(taf, obs, vf, vt)
    ts = next(c for c in s.contingency if c.event == "ts")
    missed = [e for e in s.episodes if e.event == "ts" and e.disposition == "missed"]
    r.append(check("unforecast TS -> ts miss cells + one missed episode",
                   ts.c >= 2 and len(missed) == 1,
                   f"miss={ts.c} episodes={[e.disposition for e in s.episodes if e.event=='ts']}"))

    # catalog is versioned + non-empty
    r.append(check("event catalog present (13 events)", len(EVENT_CATALOG_V1) == 13,
                   f"n={len(EVENT_CATALOG_V1)}"))
    return r


# --------------------------------------- axis 2 timing + axis 3 ordinal + deltas + batch

def run_timing_ordinal_deltas():
    r = []

    # assignment beats greedy: O1@2 O2@0 vs F1@4 F2@1, window 3.
    # greedy-nearest matches O1-F2 and strands O2 (F1 out of window); min-cost matches both.
    obs_eps = [[t(2)], [t(0)]]
    fcst_eps = [[t(4)], [t(1)]]
    m = _match_episodes(obs_eps, fcst_eps, 3)
    matched = {(o, f) for o, f in m if o is not None and f is not None}
    r.append(check("min-cost assignment beats greedy (2 matched: O1-F1, O2-F2)",
                   matched == {(0, 0), (1, 1)}, f"matched={sorted(matched)}"))

    # episodes respect MATCH_WINDOW_H: onset 5 h apart, no overlap -> not matched
    m2 = _match_episodes([[t(0)]], [[t(5)]], 3)
    r.append(check("episode match respects window (5 h apart -> missed + false_alarm)",
                   (0, None) in m2 and (None, 0) in m2, f"m={m2}"))

    # MACE on a known series + worst excursion: fcst E; obs E, D, A -> deltas 0,1,4
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0912 24010KT 9999 SKC")
    obs = [ob(datetime(2026, 7, 9, 9)),
           ob(datetime(2026, 7, 9, 10), ceiling_ft=1500),
           ob(datetime(2026, 7, 9, 11), ceiling_ft=100)]
    s = score_skill(taf, obs, vf, vt)
    r.append(check("MACE (0+1+4)/3 = 1.667, worst excursion 4",
                   abs(s.mace - 5 / 3) < 1e-9 and s.worst_excursion["delta"] == 4,
                   f"mace={s.mace} worst={s.worst_excursion}"))

    # matched-hours deltas: baseline scored on fewer hours -> both pools shrink to the
    # baseline's (an availability difference must not masquerade as skill)
    taf, vf, vt = mk("TAF KXXX 090900Z 0909/0915 24010KT 9999 SKC")
    subj = score_skill(taf, hourly(vf, vt, wind_speed=15), vf, vt)            # 6 scored hours
    base = score_skill(taf, hourly(vf, vt, wind_speed=15)[:3], vf, vt)        # fewer hours
    subj_n = sum(1 for x in subj.element_rows if x.element == "wind_speed" and x.status == "scored")
    base_n = sum(1 for x in base.element_rows if x.element == "wind_speed" and x.status == "scored")
    d = skill_deltas(subj, base)
    ws = d["elements"]["wind_speed"]
    r.append(check("matched-hours delta: n = smaller (baseline) pool, strict subset, mae_skill 0",
                   ws["n"] == base_n and base_n < subj_n and abs(ws["mae_skill"]) < 1e-9,
                   f"n={ws['n']} base_n={base_n} subj_n={subj_n} skill={ws['mae_skill']}"))

    # zero-baseline delta -> None + raw values retained
    perfect_base = score_skill(taf, hourly(vf, vt, wind_speed=10), vf, vt)    # matches fcst -> mae 0
    d2 = skill_deltas(subj, perfect_base)
    ws2 = d2["elements"]["wind_speed"]
    r.append(check("zero-baseline MAE -> mae_skill None + raw subject_mae kept",
                   ws2["mae_skill"] is None and ws2["reason"] == "zero_baseline_mae"
                   and ws2["subject_mae"] == 5,
                   f"{ws2}"))

    # batch pooling: summed-cells HSS differs from the mean of per-run HSSs
    h1 = contingency_scores(2, 0, 0, 1)["hss"]
    h2 = contingency_scores(0, 2, 1, 3)["hss"]
    hs_summed = contingency_scores(2, 2, 1, 4)["hss"]
    r.append(check("batch pooling: summed-cells HSS != mean of per-run HSSs",
                   abs(hs_summed - (h1 + h2) / 2) > 1e-6,
                   f"summed={hs_summed:.4f} mean={(h1 + h2) / 2:.4f}"))
    return r


SECTIONS = [
    ("1. axis 1 -- continuous element errors", run_axis1),
    ("2. axis 1 -- QNH group-min + TX/TN", run_qnh_txtn),
    ("3. axis 2 -- event contingency", run_events),
    ("4. axis 2 timing + axis 3 ordinal + deltas + batch", run_timing_ordinal_deltas),
]


def main() -> int:
    print("=== TAFSKILL (skill) SELF-TEST ===")
    all_results, md = [], ["# tafskill self-test", ""]
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
    (Path("logs") / "tafskill_selftest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{passed}/{total} passed. Report: logs/tafskill_selftest.md")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
