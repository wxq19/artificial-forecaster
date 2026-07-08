"""Output-seam self-test: build TafProducts -> render -> validate -> round-trip.

Unlike the other test_*.py drivers, this one calls NO model and hits NO network --
the TAF OUTPUT seam (tafgen) is deterministic, so this is a fast, free, repeatable
check that doubles as documentation of what a valid AF TAF looks like.

For each case it: renders the TafProduct to text, runs validate() (the AFMAN 15-124
rule checker), and runs roundtrip() (render -> tafparse.parse -> compare). Four cases
reproduce the worked examples from AFMAN 15-124 ch.1 (docs/TAF Coding.pdf) and assert
the rendered text matches the manual BYTE-FOR-BYTE -- the strongest correctness gate
we have. One case is deliberately invalid to show validate() catching violations.
A self-contained markdown report is written under logs/.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from forecaster.metar import CloudLayer
from forecaster.tafgen import (
    TafProduct, TafProductGroup, _example_product, last_no_amds, render_taf, roundtrip, validate,
)
from forecaster.tafparse import IcingLayer, TafTemp, TurbulenceLayer, VolcanicAsh, WindShear


def C(cover: str, ft: int, cb: bool = False) -> CloudLayer:
    return CloudLayer(cover=cover, height_ft=ft, type="CB" if cb else None)


@dataclass
class Case:
    name: str
    desc: str
    product: TafProduct
    expected: str | None = None      # byte-exact AFMAN text, when this reproduces a figure
    expect_invalid: bool = False     # True => validate() SHOULD report findings
    render_only: bool = False        # partial doc example (e.g. Fig 1.6): check render+roundtrip, not validate-clean


# --- AFMAN 15-124 Figure 1.3: KBAD winter TAF (icing, VV, partial obscuration) ---
fig13 = TafProduct(
    station="KBAD", issue_day=1, issue_hour=15, issue_minute=55,
    valid_from_day=1, valid_from_hour=16, valid_to_day=2, valid_to_hour=22,
    prevailing=TafProductGroup(
        wind_dir=30, wind_speed=8, vis_m=800, weather=["PRFG"],
        clouds=[C("FEW", 0), C("BKN", 500), C("BKN", 1200)],
        qnh_inhg=30.01, remarks=["FG FEW000"]),
    groups=[
        TafProductGroup(change="TEMPO", from_day=1, from_hour=18, to_day=1, to_hour=21,
            wind_dir=140, wind_speed=12, wind_gust=18, vis_m=3200, weather=["-SHSN", "BLSN"],
            clouds=[C("FEW", 0), C("OVC", 600)],
            icing=[IcingLayer(ic=2, base_ft=600, thickness_ft=5000)], remarks=["BLSN FEW000"]),
        TafProductGroup(change="FM", from_day=1, from_hour=21, from_minute=45,
            wind_dir=150, wind_speed=12, wind_gust=20, vis_m=9999, weather=["NSW"],
            clouds=[C("OVC", 3000)], qnh_inhg=29.92),
        TafProductGroup(change="BECMG", from_day=1, from_hour=23, to_day=1, to_hour=24,
            wind_dir=150, wind_speed=12, wind_gust=20, vis_m=3200, weather=["-SN", "BLSN"],
            clouds=[C("FEW", 0), C("OVC", 400)],
            icing=[IcingLayer(ic=2, base_ft=400, thickness_ft=6000)],
            qnh_inhg=29.83, remarks=["BLSN FEW000"]),
        TafProductGroup(change="TEMPO", from_day=2, from_hour=1, to_day=2, to_hour=3,
            wind_dir=130, wind_speed=15, wind_gust=25, vis_m=200, weather=["-FZDZ", "FG"],
            vert_vis_ft=100,
            icing=[IcingLayer(ic=6, base_ft=0, thickness_ft=1000),
                   IcingLayer(ic=5, base_ft=1000, thickness_ft=9000)]),
    ],
    max_temp=TafTemp(temp_c=0, day=1, hour=21), min_temp=TafTemp(temp_c=-1, day=2, hour=12),
)
fig13_txt = (
    "TAF KBAD 011555Z 0116/0222 03008KT 0800 PRFG FEW000 BKN005 BKN012 QNH3001INS FG FEW000\n"
    "TEMPO 0118/0121 14012G18KT 3200 -SHSN BLSN FEW000 OVC006 620065 BLSN FEW000\n"
    "FM012145 15012G20KT 9999 NSW OVC030 QNH2992INS\n"
    "BECMG 0123/0124 15012G20KT 3200 -SN BLSN FEW000 OVC004 620046 QNH2983INS BLSN FEW000\n"
    "TEMPO 0201/0203 13015G25KT 0200 -FZDZ FG VV001 660001 650109 TX00/0121Z TNM01/0212Z"
)

# --- AFMAN Figure 1.4: ETAR corrected TAF (icing + turbulence) ---
fig14 = TafProduct(
    station="ETAR", issue_day=1, issue_hour=16, issue_minute=15,
    valid_from_day=1, valid_from_hour=16, valid_to_day=2, valid_to_hour=22, corrected=True,
    prevailing=TafProductGroup(
        wind_dir=280, wind_speed=12, wind_gust=25, vis_m=8000, weather=["-RASN"],
        clouds=[C("SCT", 600), C("BKN", 1500), C("OVC", 2000)],
        icing=[IcingLayer(ic=2, base_ft=1500, thickness_ft=8000)],
        turbulence=[TurbulenceLayer(b=4, base_ft=0, thickness_ft=9000)],
        qnh_inhg=29.60),
    groups=[TafProductGroup(change="BECMG", from_day=1, from_hour=18, to_day=1, to_hour=19,
        wind_dir=270, wind_speed=12, vis_m=9999, weather=["NSW"],
        clouds=[C("SCT", 1500), C("BKN", 2000)], qnh_inhg=29.65)],
    max_temp=TafTemp(temp_c=15, day=1, hour=20), min_temp=TafTemp(temp_c=4, day=2, hour=11),
)
fig14_txt = (
    "TAF COR ETAR 011615Z 0116/0222 28012G25KT 8000 -RASN SCT006 BKN015 OVC020 620158 540009 QNH2960INS\n"
    "BECMG 0118/0119 27012KT 9999 NSW SCT015 BKN020 QNH2965INS TX15/0120Z TN04/0211Z"
)

# --- AFMAN Figure 1.6: volcanic-ash plume aloft ---
fig16 = TafProduct(
    station="CCCC", issue_day=10, issue_hour=15, issue_minute=55,
    valid_from_day=10, valid_from_hour=16, valid_to_day=11, valid_to_hour=22,
    prevailing=TafProductGroup(wind_dir=240, wind_speed=10, vis_m=9999,
        clouds=[C("FEW", 10000)], volcanic_ash=VolcanicAsh(base_ft=10000, top_ft=20000),
        qnh_inhg=29.92),
)
fig16_txt = "TAF CCCC 101555Z 1016/1122 24010KT 9999 FEW100 VA100200 QNH2992INS"

# --- AFMAN Figure 1.7: non-convective low-level wind shear (+ FM minutes) ---
fig17 = TafProduct(
    station="CCCC", issue_day=1, issue_hour=15, issue_minute=55,
    valid_from_day=1, valid_from_hour=16, valid_to_day=2, valid_to_hour=22,
    prevailing=TafProductGroup(wind_dir=30, wind_speed=8, vis_m=800, weather=["PRFG"],
        clouds=[C("FEW", 0), C("BKN", 500), C("BKN", 1200)],
        wind_shear=WindShear(height_ft=1500, wind_dir=120, wind_speed=38),
        qnh_inhg=30.01, remarks=["FG FEW000"]),
    groups=[
        TafProductGroup(change="TEMPO", from_day=1, from_hour=18, to_day=1, to_hour=20,
            wind_dir=140, wind_speed=12, wind_gust=18, vis_m=3200, weather=["-SN", "BLSN"],
            clouds=[C("FEW", 0), C("OVC", 600)],
            icing=[IcingLayer(ic=2, base_ft=600, thickness_ft=5000)], remarks=["SN FEW000"]),
        TafProductGroup(change="FM", from_day=1, from_hour=21, from_minute=30,
            wind_dir=150, wind_speed=12, wind_gust=20, vis_m=9999, weather=["NSW"],
            clouds=[C("SCT", 3000)], qnh_inhg=29.92),
        TafProductGroup(change="BECMG", from_day=1, from_hour=23, to_day=1, to_hour=24,
            wind_dir=150, wind_speed=12, wind_gust=20, vis_m=3200, weather=["-SN", "BLSN"],
            clouds=[C("FEW", 0), C("OVC", 400)],
            icing=[IcingLayer(ic=2, base_ft=400, thickness_ft=6000)],
            qnh_inhg=29.83, remarks=["SN FEW000"]),
    ],
    max_temp=TafTemp(temp_c=8, day=1, hour=19), min_temp=TafTemp(temp_c=-4, day=2, hour=11),
)
fig17_txt = (
    "TAF CCCC 011555Z 0116/0222 03008KT 0800 PRFG FEW000 BKN005 BKN012 WS015/12038KT QNH3001INS FG FEW000\n"
    "TEMPO 0118/0120 14012G18KT 3200 -SN BLSN FEW000 OVC006 620065 SN FEW000\n"
    "FM012130 15012G20KT 9999 NSW SCT030 QNH2992INS\n"
    "BECMG 0123/0124 15012G20KT 3200 -SN BLSN FEW000 OVC004 620046 QNH2983INS SN FEW000 TX08/0119Z TNM04/0211Z"
)

# --- Amended TAF: clipped to the remaining validity (1.3.2.1.2.1) ---
amended = TafProduct(
    station="KADW", issue_day=1, issue_hour=18, issue_minute=47,
    valid_from_day=1, valid_from_hour=18, valid_to_day=2, valid_to_hour=22, amendment=True,
    prevailing=TafProductGroup(wind_dir=200, wind_speed=10, vis_m=9999,
        clouds=[C("SCT", 4000)], qnh_inhg=29.95),
    groups=[TafProductGroup(change="BECMG", from_day=1, from_hour=20, to_day=1, to_hour=21,
        wind_dir=240, wind_speed=12, vis_m=8000, weather=["-RA"],
        clouds=[C("BKN", 2000)], qnh_inhg=29.92)],
    max_temp=TafTemp(temp_c=17, day=2, hour=15), min_temp=TafTemp(temp_c=9, day=2, hour=10),
)

# --- Limited-duty TAF: LAST NO AMDS remark (1.3.13.2.1) ---
limited = TafProduct(
    station="KBLV", issue_day=1, issue_hour=12, issue_minute=0,
    valid_from_day=1, valid_from_hour=12, valid_to_day=2, valid_to_hour=18,
    prevailing=TafProductGroup(wind_dir=270, wind_speed=8, vis_m=9999,
        clouds=[C("SCT", 5000)], qnh_inhg=30.00),
    max_temp=TafTemp(temp_c=20, day=1, hour=20), min_temp=TafTemp(temp_c=10, day=2, hour=11),
    remarks=[last_no_amds(2, 6, 2, 12)],
)

# --- Deliberately invalid: validate() should light up ---
broken = TafProduct(
    station="KXXX", issue_day=1, issue_hour=12, issue_minute=0,
    valid_from_day=1, valid_from_hour=12, valid_to_day=2, valid_to_hour=10,   # 22h, not 30
    prevailing=TafProductGroup(wind_dir=275, wind_speed=10, wind_gust=8,      # dir not /10, gust < mean
        vis_m=4800, clouds=[C("BKN", 2000), C("SCT", 3000)]),                 # vis<9999 no wx; summation; no QNH
    groups=[TafProductGroup(change="TEMPO", from_day=1, from_hour=14, to_day=1, to_hour=16,
        wind_dir=300, wind_speed=15, vis_m=9999, qnh_inhg=29.9)],             # QNH in a TEMPO
)

# --- Amendment via amend(): TX/TN carry into the remaining first-24h temp window ---
_orig = TafProduct.issue(station="KADW", issue_day=1, issue_hour=12, issue_minute=0,
    valid_from_day=1, valid_from_hour=12,
    prevailing=TafProductGroup(wind_dir=200, wind_speed=10, vis_m=9999,
        clouds=[C("SCT", 4000)], qnh_inhg=29.95),
    max_temp=TafTemp(temp_c=18, day=1, hour=20), min_temp=TafTemp(temp_c=8, day=2, hour=10))
amended_ctor = TafProduct.amend(_orig, at_day=1, at_hour=16, at_minute=30,
    prevailing=TafProductGroup(wind_dir=240, wind_speed=12, vis_m=8000,
        weather=["-RA"], clouds=[C("BKN", 2000)], qnh_inhg=29.92))

CASES = [
    Case("Fig 1.3 — KBAD winter", "Icing, VV total obscuration, partial-obscuration remarks, FM minutes.",
         fig13, expected=fig13_txt),
    Case("Fig 1.4 — ETAR COR", "Corrected TAF with an icing AND a turbulence group.", fig14, expected=fig14_txt),
    Case("Fig 1.6 — volcanic ash", "VA plume aloft (a partial example LINE, not a full TAF -- no TX/TN).",
         fig16, expected=fig16_txt, render_only=True),
    Case("Fig 1.7 — wind shear", "Non-convective low-level wind shear; FM at :30.", fig17, expected=fig17_txt),
    Case("Amended TAF", "AMD clipped to the remaining validity (<30h is correct for an amendment).", amended),
    Case("Amend() carry-forward", "amend() keeps TX/TN; both still land in the remaining first-24h window.",
         amended_ctor),
    Case("Limited-duty", "LAST NO AMDS limited-METWATCH remark.", limited),
    Case("emit_taf guide example", "The worked example shipped in emit_taf_guide() -- must stay valid + round-trippable.",
         _example_product()),
    Case("Invalid (negative test)", "Bad span, off-10 wind, gust<mean, uncaused vis, summation, missing/misplaced QNH.",
         broken, expect_invalid=True),
]


@dataclass
class Result:
    case: Case
    rendered: str
    findings: list[str]
    diffs: list[str]
    byte_ok: bool
    validate_ok: bool
    rt_ok: bool = field(default=False)

    @property
    def passed(self) -> bool:
        return self.byte_ok and self.validate_ok and self.rt_ok


def run(case: Case) -> Result:
    rendered = render_taf(case.product)
    findings = validate(case.product)
    diffs = roundtrip(case.product)
    byte_ok = case.expected is None or rendered == case.expected
    if case.render_only:
        validate_ok = True                       # partial doc example: presence rules N/A
    else:
        validate_ok = bool(findings) if case.expect_invalid else not findings
    # An intentionally-invalid TAF need not round-trip; everything else must.
    rt_ok = True if case.expect_invalid else not diffs
    return Result(case, rendered, findings, diffs, byte_ok, validate_ok, rt_ok)


results = [run(c) for c in CASES]


def build_markdown() -> str:
    npass = sum(r.passed for r in results)
    md = [
        "# TAF Output-Seam Self-Test (tafgen)",
        f"_{datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"**{npass}/{len(results)} cases passed.** No model or network -- pure render/validate/round-trip.",
        "",
        "| Case | Byte-exact | validate() | round-trip | Result |",
        "|------|------------|------------|------------|--------|",
    ]
    for r in results:
        byte = "PASS" if r.byte_ok else "FAIL"
        if r.case.expected is None:
            byte = "n/a"
        val = ("caught" if r.findings else "MISSED") if r.case.expect_invalid else (
            "clean" if not r.findings else "FAIL")
        rt = "n/a" if r.case.expect_invalid else ("clean" if not r.diffs else "FAIL")
        md.append(f"| {r.case.name} | {byte} | {val} | {rt} | {'PASS' if r.passed else 'FAIL'} |")

    for r in results:
        md += ["", f"## {r.case.name}", "", f"_{r.case.desc}_", "", "```text", r.rendered, "```"]
        if r.case.expected is not None and not r.byte_ok:
            md += ["", "Expected (AFMAN):", "```text", r.case.expected, "```"]
        if r.case.expect_invalid:
            md += ["", "validate() findings (expected):"] + [f"- {f}" for f in r.findings]
        else:
            md += ["", f"- validate(): {'clean' if not r.findings else 'FAIL — ' + '; '.join(r.findings)}"]
            md += [f"- round-trip: {'clean' if not r.diffs else 'FAIL — ' + '; '.join(r.diffs)}"]
    return "\n".join(md)


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = log_dir / f"tafgen_selftest_{stamp}.md"
log_path.write_text(build_markdown(), encoding="utf-8")

print("=== TAF OUTPUT-SEAM SELF-TEST ===")
for r in results:
    print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.case.name}")
    if not r.passed:
        for f in r.findings if not r.case.expect_invalid else []:
            print(f"        validate: {f}")
        for d in r.diffs if not r.case.expect_invalid else []:
            print(f"        round-trip: {d}")
        if r.case.expected is not None and not r.byte_ok:
            print("        byte-exact: render != AFMAN text")
print(f"\n{sum(r.passed for r in results)}/{len(results)} passed. Report: {log_path}")
