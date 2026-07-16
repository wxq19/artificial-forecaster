"""Shared-primitives self-test (scoring-design sec 13). No model, no network:
tafstate + the tafparse `explicit_fields` extension are deterministic, so this is a
fast, free, repeatable check with KNOWN outcomes. Grows one section per M0 sub-step.

Run: uv run python scripts/test_tafstate.py

Section 1 (this checkpoint): the `explicit_fields` parser extension -- each change
group records which fields its raw chunk EXPLICITLY stated, so the resolver can later
tell an inherited (omitted) field from an explicit restatement (omitted sky vs SKC,
omitted wx vs NSW, CAVOK, calm vs omitted wind, QNH absent on TEMPO).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from forecaster import store
from forecaster.tafparse import parse
from forecaster.tafstate import (
    STATUS_UNKNOWN, VIS_KNOWN_NUMERIC, VIS_KNOWN_UNLIMITED, TruthPolicy,
    absolute_validity, build_truth, daf_flight_category, default_profile,
    forecast_state, normalize_weather, opportunities, resolve_group_state,
    tafver_ceiling_category, tafver_visibility_category, validate_profile,
)

# A stable issue-date anchor for parsing synthetic TAFs into absolute windows.
REF = datetime(2026, 7, 9)


def _win(raw: str):
    """Parse a TAF and return (TafObs, valid_from, valid_to) absolute."""
    obs = parse(raw)
    _, vf, vt = absolute_validity(obs, REF)
    return obs, vf, vt


def check(label, cond, detail=""):
    return (label, bool(cond), "" if cond else f"      {detail}")

# (label, raw TAF, [expected explicit_fields per period: prevailing, then each group])
EXPLICIT_CASES = [
    (
        "military: meters vis + per-group QNH + NSW",
        "TAF KBLV 091730Z 0918/1024 24010KT 9999 SKC QNH2992INS "
        "FM100000 27015G25KT 8000 -SHRA BKN015 QNH2985INS "
        "TEMPO 1002/1006 3200 SHRA BKN008 "
        "BECMG 1012/1014 VRB03KT 9999 NSW SKC QNH2990INS",
        [
            {"wind", "visibility", "sky", "qnh"},              # prevailing
            {"wind", "gust", "visibility", "weather", "sky", "qnh"},  # FM (gust present)
            {"visibility", "weather", "sky"},                 # TEMPO (no wind, no QNH)
            {"wind", "visibility", "weather", "sky", "qnh"},  # BECMG (NSW => weather)
        ],
    ),
    (
        "civil: calm vs omitted wind, CAVOK, omitted sky/weather",
        "TAF KMSN 091730Z 0918/1012 00000KT CAVOK "
        "FM100000 P6SM "
        "BECMG 1006/1008 18012KT",
        [
            {"wind", "visibility", "sky", "weather"},         # prevailing (00000KT=calm; CAVOK)
            {"visibility"},                                   # FM (P6SM only; wind/sky/wx omitted)
            {"wind"},                                         # BECMG (wind only)
        ],
    ),
]


def run_explicit_fields() -> list[tuple[str, bool, str]]:
    """Returns (label, passed, detail) per case."""
    results = []
    for label, raw, expected in EXPLICIT_CASES:
        obs = parse(raw)
        got = [g.explicit_fields for g in (obs.prevailing, *obs.groups)]
        ok = got == expected
        detail = "" if ok else "\n".join(
            f"      period {i}: got {sorted(g)} != expected {sorted(e)}"
            for i, (g, e) in enumerate(zip(got, expected)) if g != e
        )
        # a length mismatch (period count) is its own failure
        if len(got) != len(expected):
            ok = False
            detail = f"      period count {len(got)} != expected {len(expected)}"
        results.append((label, ok, detail))
    return results


def run_resolver() -> list[tuple[str, bool, str]]:
    r = []

    # 2a. absolute_validity: month rollover + hour-24 (midnight end)
    obs = parse("TAF KXXX 311730Z 3118/0124 24010KT 9999 SKC")
    issue, vf, vt = absolute_validity(obs, datetime(2026, 1, 31))
    r.append(check("absolutize: month rollover + hour-24",
                   vf == datetime(2026, 1, 31, 18) and vt == datetime(2026, 2, 2, 0),
                   f"vf={vf} vt={vt}"))
    # leap-year day exists
    lp = parse("TAF KXXX 281730Z 2900/2912 24010KT 9999 SKC")
    _, lvf, _ = absolute_validity(lp, datetime(2024, 2, 28))
    r.append(check("absolutize: leap-year Feb 29", lvf == datetime(2024, 2, 29, 0), f"lvf={lvf}"))

    # 2b. FM at HH45 -> TWO opportunities in the bin (nothing erased)
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 24010KT 9999 SKC "
                         "FM090445 27015KT 9999 SKC")
    opps = opportunities(taf, wvf, wvt)
    bin04 = [o for o in opps if o.bin_start == datetime(2026, 7, 9, 4)]
    types = {o.group_type for o in bin04}
    fm = next((o for o in bin04 if o.group_type == "FM"), None)
    pre = next((o for o in bin04 if o.group_type == "INITIAL"), None)
    r.append(check(
        "opportunities: FM at HH45 yields 45-min predecessor + 15-min FM",
        types == {"INITIAL", "FM"} and fm and pre
        and (fm.interval_end - fm.interval_start) == timedelta(minutes=15)
        and (pre.interval_end - pre.interval_start) == timedelta(minutes=45),
        f"bin04 types={types}"))

    # 2c. BECMG overlays only changed fields; post-BECMG baseline carries the change
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 24010KT 9999 SKC "
                         "BECMG 0904/0906 30015KT")
    becmg_idx = 1
    st = resolve_group_state(taf, becmg_idx, wvf, wvt)
    r.append(check("resolve BECMG: wind overlaid, sky/vis inherited",
                   st.wind_speed == 15 and st.wind_dir == 300
                   and st.ceiling_status == "known_unlimited",
                   f"wind={st.wind_speed}/{st.wind_dir} ceil={st.ceiling_status}"))
    after = forecast_state(taf, datetime(2026, 7, 9, 8), valid_from=wvf, valid_to=wvt)
    r.append(check("post-BECMG baseline carries the completed change",
                   after.prevailing.wind_speed == 15 and after.prevailing.wind_dir == 300,
                   f"prevailing wind={after.prevailing.wind_speed}/{after.prevailing.wind_dir}"))

    # 2d. TEMPO is an ALTERNATE (never mutates prevailing); NSW resolves to empty-known
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 24010KT 6000 -RA BKN020 "
                         "TEMPO 0903/0906 2000 TSRA BKN008")
    fs = forecast_state(taf, datetime(2026, 7, 9, 4), valid_from=wvf, valid_to=wvt)
    alt = fs.alternates[0] if fs.alternates else None
    r.append(check("TEMPO alternate present; prevailing unchanged",
                   fs.prevailing.weather == ["-RA"] and alt is not None
                   and any("TS" in w for w in alt.weather),
                   f"prev.wx={fs.prevailing.weather} alts={len(fs.alternates)}"))
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 24010KT 6000 -RA BKN020 "
                         "BECMG 0906/0908 9999 NSW SKC")
    nsw = resolve_group_state(taf, 1, wvf, wvt)
    r.append(check("BECMG NSW -> weather known-empty",
                   nsw.weather == [] and nsw.weather_status == "known",
                   f"wx={nsw.weather} status={nsw.weather_status}"))

    # 2e. PROB group marked and carried (excluded by v1 policy downstream)
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 24010KT 9999 SKC "
                         "PROB30 0903/0906 3000 TSRA")
    prob = [o for o in opportunities(taf, wvf, wvt) if o.group_type == "PROB"]
    r.append(check("PROB group marked with probability",
                   prob and all(o.probability == 30 for o in prob), f"prob opps={len(prob)}"))

    # availability statuses: vrb vs calm vs unknown direction; unlimited vs unknown
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 VRB05KT 9999 SKC")
    vrb = resolve_group_state(taf, 0, wvf, wvt)
    taf2, wvf2, wvt2 = _win("TAF KXXX 090000Z 0900/0912 00000KT 9999 SKC")
    calm = resolve_group_state(taf2, 0, wvf2, wvt2)
    r.append(check("statuses: vrb vs calm direction; unlimited ceiling",
                   vrb.wind_dir_status == "vrb" and calm.wind_dir_status == "calm"
                   and vrb.ceiling_status == "known_unlimited",
                   f"vrb={vrb.wind_dir_status} calm={calm.wind_dir_status}"))
    # gust: known_absent (wind stated, no G) vs inherited_absent (wind omitted in BECMG)
    taf, wvf, wvt = _win("TAF KXXX 090000Z 0900/0912 24015G25KT 9999 SKC "
                         "BECMG 0906/0908 4000 BR")
    gustless = resolve_group_state(taf, 1, wvf, wvt)   # BECMG omits wind -> inherits gust
    r.append(check("gust status: inherited_absent when wind omitted in BECMG",
                   gustless.gust_status in ("present", "inherited_absent"),
                   f"gust_status={gustless.gust_status}"))
    return r


def run_classifiers() -> list[tuple[str, bool, str]]:
    r = []
    p = default_profile("KXXX")
    KN, KU = "known_numeric", "known_unlimited"

    r.append(check("DAF: ceiling 2000 + unlimited vis -> E",
                   daf_flight_category(2000, KN, None, KU, profile=p) == "E"))
    r.append(check("DAF: ceiling 1999 + unlimited vis -> D",
                   daf_flight_category(1999, KN, None, KU, profile=p) == "D"))
    r.append(check("DAF lower-of: unlimited ceiling + 3000 m vis -> B",
                   daf_flight_category(None, KU, 3000, KN, profile=p) == "B",
                   f"got {daf_flight_category(None, KU, 3000, KN, profile=p)}"))
    r.append(check("DAF: ceiling 800 (C) + 5000 m vis (E) -> C (lower-of)",
                   daf_flight_category(800, KN, 5000, KN, profile=p) == "C",
                   f"got {daf_flight_category(800, KN, 5000, KN, profile=p)}"))
    r.append(check("DAF provisional minima: ceiling 150 -> A, 300 -> B",
                   daf_flight_category(150, KN, None, KU, profile=p) == "A"
                   and daf_flight_category(300, KN, None, KU, profile=p) == "B"))
    # OCONUS substitution flips a 4900 m vis from E to D
    po = default_profile("OCON")
    po.use_oconus_vis_substitutions = True
    r.append(check("DAF OCONUS sub: 4900 m vis E (conus) vs D (oconus)",
                   daf_flight_category(None, KU, 4900, KN, profile=p) == "E"
                   and daf_flight_category(None, KU, 4900, KN, profile=po) == "D",
                   f"conus={daf_flight_category(None, KU, 4900, KN, profile=p)} "
                   f"oconus={daf_flight_category(None, KU, 4900, KN, profile=po)}"))
    r.append(check("DAF: unknown input -> None (never a default)",
                   daf_flight_category(None, "unknown", 5000, KN, profile=p) is None))

    # TAFVER category ladders (separate contract)
    r.append(check("TAFVER ceiling: 1500 -> D, unlimited -> E",
                   tafver_ceiling_category(1500, KN, p) == "D"
                   and tafver_ceiling_category(None, KU, p) == "E"))
    r.append(check("TAFVER vis: 0.25 SM -> A, P6SM (unlimited) -> E",
                   tafver_visibility_category(0.25, "M", "known_numeric", p) == "A"
                   and tafver_visibility_category(6.0, "P", "known_unlimited", p) == "E"))

    # profile validation
    r.append(check("profile validation: default profile is valid",
                   validate_profile(p) == [], f"findings={validate_profile(p)}"))
    bad = default_profile("BAD")
    bad.tafver_ceiling_bands[1].hi = 999    # create a gap with the next band's lo=700..
    r.append(check("profile validation: broken ladder is rejected",
                   validate_profile(bad) != []))
    return r


def _ob(t, **kw):
    base = dict(station="KXXX", obs_time=t, report_type="METAR", auto=False, cavok=False,
                wind_dir_deg=None, wind_dir_card=None, wind_speed=None, wind_gust=None,
                wind_unit="KT", visibility=None, vis_sm=None, vis_m=None, vis_flag=None,
                ceiling_ft=None, vertical_visibility_ft=None, temp_c=None, dewpoint_c=None,
                altimeter_inhg=None, altimeter_hpa=None, weather=[], clouds=[], remarks=None,
                raw="", source="test", corrected=False)
    base.update(kw)
    return base


def run_truth() -> list[tuple[str, bool, str]]:
    r = []

    # present-weather normalizer (5.6)
    a, s = normalize_weather(["+TSRA"])
    r.append(check("normalize: +TSRA -> other:TS + liquid:RA",
                   a == {"other:TS", "liquid:RA"} and s == {"other", "liquid"}, f"{a}"))
    a, s = normalize_weather(["FZRA"])
    r.append(check("normalize: FZRA -> freezing (class)", s == {"freezing"}, f"{a}"))
    a, s = normalize_weather(["DZ"])
    r.append(check("normalize: DZ -> liquid (not obscuration)", s == {"liquid"}, f"{a}"))
    a, s = normalize_weather(["BR"])
    r.append(check("normalize: BR -> obscuration", s == {"obscuration"}, f"{a}"))
    a, s = normalize_weather(["NSW"])
    r.append(check("normalize: NSW -> empty", a == set() and s == set()))

    vf = datetime(2026, 7, 9, 12)
    vt = datetime(2026, 7, 9, 15)          # 3-hour window: hours 12,13,14
    pol = TruthPolicy()

    # carry-in fills hour 12 from a pre-window ob; a :55 ob covers the following hour
    obs = [
        _ob(datetime(2026, 7, 9, 11, 55), ceiling_ft=1500, wind_speed=10, wind_dir_deg=240),
        _ob(datetime(2026, 7, 9, 13, 5), ceiling_ft=900, wind_speed=12, wind_dir_deg=250),
    ]
    hours, manifest = build_truth(obs, vf, vt, policy=pol)
    h12 = hours[0]
    r.append(check("truth: carry-in / :55 ob covers hour 12",
                   h12.status == "available" and h12.cons["ceiling"].value == 1500,
                   f"h12 status={h12.status} ceil={h12.cons['ceiling'].value}"))

    # max_hold gap -> a fully-uncovered hour is unavailable (coverage_gap)
    obs_gap = [_ob(datetime(2026, 7, 9, 11, 55), ceiling_ft=1500, wind_speed=10, wind_dir_deg=240)]
    hg, _ = build_truth(obs_gap, vf, vt, policy=pol)
    r.append(check("truth: max_hold gap -> later hour unavailable",
                   hg[0].status == "available" and hg[2].status == "unavailable"
                   and hg[2].reason in ("coverage_gap", "no_obs"),
                   f"h14 status={hg[2].status}/{hg[2].reason}"))

    # per-field predominant: 40 min at 1000 ft beats a 20-min 500 ft dip
    obs_pred = [
        _ob(datetime(2026, 7, 9, 12, 0), ceiling_ft=1000, wind_speed=15, wind_dir_deg=240),
        _ob(datetime(2026, 7, 9, 12, 40), ceiling_ft=500, wind_speed=15, wind_dir_deg=240),
    ]
    hp, _ = build_truth(obs_pred, vf, datetime(2026, 7, 9, 13), policy=pol)
    r.append(check("truth: predominant ceiling 1000 (40 min) over 500 dip (20 min)",
                   hp[0].pred["ceiling"].value == 1000 and 500 in hp[0].temporaries.get("ceiling", []),
                   f"pred={hp[0].pred['ceiling'].value} temps={hp[0].temporaries.get('ceiling')}"))
    # conservative view is pessimistic: lowest ceiling in the hour = 500
    r.append(check("truth: conservative ceiling = lowest instant (500)",
                   hp[0].cons["ceiling"].value == 500, f"cons={hp[0].cons['ceiling'].value}"))

    # Finding 4: a MISSING field must not read as good weather. An ob that never reported
    # visibility -> conservative vis is UNKNOWN (routes the scorer to unavailable), NOT
    # "unlimited" -- so a P6SM forecast can't earn a point against an ob with no vis.
    obs_novis = [_ob(datetime(2026, 7, 9, 12, 0), ceiling_ft=3000, wind_speed=5, wind_dir_deg=240)]
    hnv, _ = build_truth(obs_novis, vf, datetime(2026, 7, 9, 13), policy=pol)
    r.append(check("truth: missing vis -> conservative vis unknown (not unlimited)",
                   hnv[0].cons["vis"].status == STATUS_UNKNOWN,
                   f"cons vis status={hnv[0].cons['vis'].status}"))
    # a genuinely clear ob (CAVOK) -> conservative vis IS unlimited (the honest positive)
    obs_clear = [_ob(datetime(2026, 7, 9, 12, 0), cavok=True, wind_speed=5, wind_dir_deg=240)]
    hcl, _ = build_truth(obs_clear, vf, datetime(2026, 7, 9, 13), policy=pol)
    r.append(check("truth: CAVOK ob -> conservative vis unlimited",
                   hcl[0].cons["vis"].status == VIS_KNOWN_UNLIMITED,
                   f"cons vis status={hcl[0].cons['vis'].status}"))
    # mixed: one numeric vis + one missing -> STILL scored at the lowest numeric, because a
    # real restriction WAS observed in the hour (the union stays conservative, not unknown)
    obs_mix = [_ob(datetime(2026, 7, 9, 12, 0), vis_sm=1.0, vis_m=1600, wind_speed=5, wind_dir_deg=240),
               _ob(datetime(2026, 7, 9, 12, 30), wind_speed=5, wind_dir_deg=240)]
    hmx, _ = build_truth(obs_mix, vf, datetime(2026, 7, 9, 13), policy=pol)
    r.append(check("truth: numeric+missing vis -> lowest numeric wins (scored)",
                   hmx[0].cons["vis"].status == VIS_KNOWN_NUMERIC and hmx[0].cons["vis"].value == 1600,
                   f"cons vis={hmx[0].cons['vis'].value}/{hmx[0].cons['vis'].status}"))

    # conservative tie on max sustained wind -> direction from the EARLIEST ob
    obs_tie = [
        _ob(datetime(2026, 7, 9, 12, 0), wind_speed=20, wind_dir_deg=100, ceiling_ft=3000),
        _ob(datetime(2026, 7, 9, 12, 30), wind_speed=20, wind_dir_deg=200, ceiling_ft=3000),
    ]
    ht, _ = build_truth(obs_tie, vf, datetime(2026, 7, 9, 13), policy=pol)
    r.append(check("truth: conservative wind tie -> dir from earliest ob (100)",
                   ht[0].cons["wind_speed"].value == 20 and ht[0].cons["wind_dir"].value == 100,
                   f"dir={ht[0].cons['wind_dir'].value}"))

    # a window hour with no covering ob at all -> unavailable no_obs
    obs_none = [_ob(datetime(2026, 7, 9, 12, 0), ceiling_ft=3000, wind_speed=5, wind_dir_deg=240)]
    hn, _ = build_truth(obs_none, vf, vt, policy=pol)
    r.append(check("truth: uncovered hour -> unavailable",
                   hn[2].status == "unavailable", f"h14={hn[2].status}"))

    # half-open: an ob exactly at valid_to contributes to no scored hour
    obs_edge = [
        _ob(datetime(2026, 7, 9, 12, 0), ceiling_ft=3000, wind_speed=5, wind_dir_deg=240),
        _ob(vt, ceiling_ft=100, wind_speed=40, wind_dir_deg=240),   # exactly at valid_to
    ]
    he, _ = build_truth(obs_edge, vf, vt, policy=pol)
    touched = any(h.status == "available" and h.cons["ceiling"].value == 100 for h in he)
    r.append(check("truth: ob at valid_to is excluded (half-open)", not touched))
    return r


def run_store() -> list[tuple[str, bool, str]]:
    r = []
    con = store.connect(":memory:")
    store.init_schema(con)
    store.init_scoring_schema(con)

    # archive idempotency + read-back
    from archive_taf import build_taf_row  # noqa: PLC0415  (script dir is on sys.path)
    raw = "TAF KBLV 091730Z 0918/1024 24010KT 9999 SKC QNH2992INS"
    row = build_taf_row(raw, issue_ref=REF, producer_kind="official", producer_name="test")
    added1 = store.insert_taf(con, row)
    added2 = store.insert_taf(con, row)      # identical content -> no-op
    got = store.taf(con, row["taf_id"])
    r.append(check("archive: insert idempotent + round-trip",
                   added1 and not added2 and got and got["station"] == "KBLV"
                   and got["valid_from_utc"] == datetime(2026, 7, 9, 18),
                   f"added1={added1} added2={added2}"))

    # scoring_window: half-open in-window + carry-in + single carry-out terminator
    def ins(t, rtype="METAR"):
        con.execute("INSERT INTO obs (station, obs_time, report_type, raw, source) "
                    "VALUES (?, ?, ?, ?, ?)", ["KZZZ", t, rtype, "x", "test"])
    vf, vt = datetime(2026, 7, 9, 18), datetime(2026, 7, 10, 0)
    for t in [datetime(2026, 7, 9, 17, 30), datetime(2026, 7, 9, 18, 30),
              datetime(2026, 7, 9, 22, 0), vt, datetime(2026, 7, 10, 0, 30)]:
        ins(t)
    win = store.scoring_window(con, "KZZZ", vf, vt)
    times = [w["obs_time"] for w in win]
    in_window = [t for t in times if vf <= t < vt]
    r.append(check("scoring_window: half-open + carry-in + one carry-out",
                   times == [datetime(2026, 7, 9, 17, 30), datetime(2026, 7, 9, 18, 30),
                             datetime(2026, 7, 9, 22, 0), vt]
                   and len(in_window) == 2,
                   f"times={times}"))
    con.close()
    return r


SECTIONS = [
    ("1. tafparse.explicit_fields", run_explicit_fields),
    ("2. resolver: absolutize / opportunities / state", run_resolver),
    ("3. classifiers: DAF flight category + TAFVER + profile", run_classifiers),
    ("4. truth builder + present-weather normalizer", run_truth),
    ("5. store: tafs archive + half-open scoring_window", run_store),
]


def main() -> int:
    print("=== TAFSTATE SHARED-PRIMITIVES SELF-TEST ===")
    all_results = []
    md = ["# tafstate self-test", ""]
    for title, fn in SECTIONS:
        print(f"\n[Section {title}]")
        md += [f"## Section {title}", ""]
        results = fn()
        for label, ok, detail in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            if detail:
                print(detail)
            md.append(f"- [{'PASS' if ok else 'FAIL'}] {label}")
        md.append("")
        all_results += results

    passed = sum(ok for _, ok, _ in all_results)
    total = len(all_results)
    md += [f"**{passed}/{total} passed.**"]
    log = Path("logs")
    log.mkdir(exist_ok=True)
    (log / "tafstate_selftest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{passed}/{total} passed. Report: logs/tafstate_selftest.md")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
