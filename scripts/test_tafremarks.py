"""AF TAF remark splitter + WND-AFT overlay self-test (T5). No model/network.

Covers tafparse.strip_remarks, tafstate.parse_wind_after, and the resolver overlay
(forecast_state prevailing wind switches to the override value at/after its time). Fixtures
include THREE real remark-bearing bulletins pulled from the round-1 archive.
"""

from datetime import datetime

from forecaster.tafparse import parse, strip_remarks
from forecaster.tafstate import forecast_state, parse_wind_after

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))


# --- real bulletins from data/pi_status.duckdb (producer_kind='human') ---
KFTK = ("TAF KFTK 161900Z 1619/1801 24006KT 9999 VCSH SCT040 BKN080 QNH3000INS TEMPO "
        "1619/1702 8000 -SHRA VCTS OVC035CB BECMG 1702/1703 23006KT 9999 NSW FEW040 SCT080 "
        "QNH3003INS TEMPO 1707/1712 6000 BR BECMG 1716/1717 24006KT 9999 VCSH SCT030 BKN200 "
        "QNH2998INS TEMPO 1718/1801 6000 -SHRA VCTS OVC030CB TX33/1620Z TN22/1711Z "
        "LAST NO AMDS AFT 1621 NEXT 1711")
PABI = ("TAF PABI 162200Z 1622/1804 VRB06KT 9999 BKN030 BKN080 630803 QNH2993INS BECMG "
        "1701/1702 17015G25KT 9999 VCSH FEW050 BKN080 640803 520105 QNH2985INS BECMG "
        "1708/1709 18025G35KT 9999 VCSH BKN070 640703 520005 QNH2981INS BECMG 1711/1712 "
        "16010G20KT 9999 VCSH FEW050 SCT065 OVC090 640903 QNH2980INS BECMG 1718/1719 "
        "16012KT 6000 -RA BKN050 620703 QNH2974INS TX17/1721Z TN10/1713Z "
        "LAST NO AMDS AFT 1705 NEXT 1714")
RJTY = ("TAF AMD RJTY 170220Z 1702/1803 14009KT 9999 SCT015 BKN030 BKN080 QNH2969INS TEMPO "
        "1703/1710 VRB12G18KT 6000 -SHRA VCTS SCT010CB BKN020 BKN030 BECMG 1711/1712 "
        "16009KT 9999 BKN015 OVC025 QNH2964INS WND 17012KT AFT 1715 BECMG 1723/1724 18012KT "
        "9999 SCT010 BKN035 QNH2972INS BECMG 1802/1803 16012G18KT 9999 SCT030 BKN080 "
        "QNH2974INS TX31/1705Z TN25/1721Z")

# 1. no-remark civil TAF: body unchanged, remarks empty.
raw0 = "TAF KXYZ 171100Z 1712/1818 24010KT 9999 FEW250"
b0, r0 = strip_remarks(raw0)
check("no-remark: body unchanged, remarks empty", b0 == raw0 and r0 == "", f"{b0!r} {r0!r}")

# 2. LAST NO AMDS stripped (real KFTK), body ends at TX/TN, WND absent.
b1, r1 = strip_remarks(KFTK)
check("KFTK: LAST NO AMDS stripped to remarks",
      "LAST NO AMDS" not in b1 and "LAST NO AMDS AFT 1621 NEXT 1711" in r1
      and b1.rstrip().endswith("TN22/1711Z"), b1[-25:])
check("KFTK: body still parses, no wind override", parse(b1) is not None
      and parse_wind_after(r1, datetime(2026, 7, 16, 19, 0)) == [])

# 3. PABI (second real LAST NO AMDS) strips cleanly.
b2, r2 = strip_remarks(PABI)
check("PABI: LAST NO AMDS stripped, body parses",
      "LAST NO AMDS" not in b2 and "AFT 1705 NEXT 1714" in r2 and parse(b2) is not None)

# 4. Synthetic WND AFTR override: parsed with correct absolute datetime + fields.
raw3 = ("TAF KWRI 171100Z 1712/1818 20008KT 9999 SCT040 QNH2992INS "
        "WND 27015G25KT AFTR 0806 TX33/1620Z TN22/1711Z")
b3, r3 = strip_remarks(raw3)
ov = parse_wind_after(r3, datetime(2026, 8, 6, 0, 0))   # AFTR 0806 = day 08, hour 06
check("WND AFTR: extracted from body, parsed to override",
      "WND" not in b3 and len(ov) == 1 and ov[0].after == datetime(2026, 8, 8, 6, 0)
      and ov[0].wind_dir == 270 and ov[0].wind_speed == 15 and ov[0].wind_gust == 25,
      str(ov))

# 5. Resolver overlay (real RJTY, mid-body WND 17012KT AFT 1715 = day17 hour15 = 15:00Z):
#    hour before 15Z scores the group wind; hour at/after scores 170/12.
bR, rR = strip_remarks(RJTY)
check("RJTY: mid-body WND stripped from body", "WND 17012KT" not in bR)
taf = parse(bR)
taf.wind_overrides = parse_wind_after(rR, datetime(2026, 7, 17, 2, 20))
vf, vt = datetime(2026, 7, 17, 2, 0), datetime(2026, 7, 18, 3, 0)
before = forecast_state(taf, datetime(2026, 7, 17, 13, 0), valid_from=vf, valid_to=vt).prevailing
after = forecast_state(taf, datetime(2026, 7, 17, 16, 0), valid_from=vf, valid_to=vt).prevailing
check("RJTY overlay: before 15Z -> BECMG group wind 160/09",
      before.wind_dir == 160 and before.wind_speed == 9,
      f"{before.wind_dir}/{before.wind_speed}")
check("RJTY overlay: at/after 15Z -> override 170/12",
      after.wind_dir == 170 and after.wind_speed == 12,
      f"{after.wind_dir}/{after.wind_speed}")
# the override changes WIND ONLY -- sky/vis unchanged across the boundary.
check("RJTY overlay: wind-only (ceiling unchanged across the boundary)",
      before.ceiling_ft == after.ceiling_ft, f"{before.ceiling_ft} vs {after.ceiling_ft}")

# 6. Stacked remarks: WND ... AFTR ... then LAST NO AMDS -> both recognized.
raw4 = ("TAF KABC 171100Z 1712/1818 20008KT 9999 SCT040 QNH2992INS TX33/1620Z TN22/1711Z "
        "WND 30012KT AFTR 1720 LAST NO AMDS AFT 1712")
b4, r4 = strip_remarks(raw4)
ov4 = parse_wind_after(r4, datetime(2026, 7, 17, 11, 0))
check("stacked: both WND and LAST NO AMDS stripped, WND override parsed",
      "WND" not in b4 and "LAST NO AMDS" not in b4 and len(ov4) == 1
      and ov4[0].wind_dir == 300 and ov4[0].after == datetime(2026, 7, 17, 20, 0),
      f"body={b4[-25:]!r} ov={ov4}")

# 7. Unrecognized trailing text stays in the body (never guessed away).
raw5 = "TAF KDEF 171100Z 1712/1818 24010KT 9999 FEW250 SOMEGIBBERISH XYZ"
b5, r5 = strip_remarks(raw5)
check("unrecognized: trailing text stays in body", b5 == raw5 and r5 == "", f"{b5!r}")

npass = sum(p for _, p, _ in checks)
print("=== TAF REMARK / WND-AFT SELF-TEST (T5) ===")
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if not passed else ""))
print(f"\n{npass}/{len(checks)} passed.")
