"""human_composite subject self-test (T9). No model/network.

Covers tafstate.composite_taf (pure-FM synthesis before/after an amendment),
store.human_bulletins_for_window, and score_taf.run() emitting a `human_composite` result
row when >=1 amendment exists (and NONE when it doesn't).
"""

import tempfile
from datetime import datetime
from pathlib import Path

import score_taf
from forecaster import store
from forecaster.metar import parse as metar_parse
from forecaster.tafarchive import build_taf_row
from forecaster.tafparse import parse
from forecaster.tafstate import composite_taf, forecast_state

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))


VF = datetime(2026, 7, 17, 12, 0)
VT = datetime(2026, 7, 17, 18, 0)

# --- 1. composite_taf unit: routine 240/10 -> AMD (eff 15Z) 310/20G30 ---
routine = parse("TAF KTST 171100Z 1712/1718 24010KT 9999 FEW250")
amd = parse("TAF AMD KTST 171500Z 1715/1718 31020G30KT 9999 BKN030")
bulls = [
    {"taf": routine, "issue_time": datetime(2026, 7, 17, 11, 0),
     "valid_from": VF, "valid_to": VT, "taf_id": "R"},
    {"taf": amd, "issue_time": datetime(2026, 7, 17, 15, 0),
     "valid_from": datetime(2026, 7, 17, 15, 0), "valid_to": VT, "taf_id": "A"},
]
ctaf, construction = composite_taf(bulls, VF, VT)
before = forecast_state(ctaf, datetime(2026, 7, 17, 14, 0), valid_from=VF, valid_to=VT).prevailing
after = forecast_state(ctaf, datetime(2026, 7, 17, 16, 0), valid_from=VF, valid_to=VT).prevailing
check("composite: before AMD effect -> routine wind 240/10",
      before.wind_dir == 240 and before.wind_speed == 10 and before.wind_gust is None,
      f"{before.wind_dir}/{before.wind_speed}")
check("composite: at/after AMD effect -> amended wind 310/20G30 + BKN030",
      after.wind_dir == 310 and after.wind_speed == 20 and after.wind_gust == 30
      and after.ceiling_ft == 3000, f"{after.wind_dir}/{after.wind_speed} ceil={after.ceiling_ft}")
check("composite: no routine BECMG/vis bleeds past the boundary (ceiling switched)",
      before.ceiling_ft is None and after.ceiling_ft == 3000)
check("composite: construction records both sources + 2 segments",
      construction["routine_taf_id"] == "R" and construction["amendment_taf_ids"] == ["A"]
      and len(construction["segments"]) == 2
      and [s["producer"] for s in construction["segments"]] == ["routine", "amendment"],
      str(construction["segments"]))

# --- 2. store.human_bulletins_for_window + score_taf.run end-to-end ---
TMP = Path(tempfile.mkdtemp(prefix="composite_test_"))
DB = str(TMP / "bench.duckdb")


def seed_obs(con, station):
    raws = [f"{station} 171155Z 24008KT 10SM SKC 24/12 A2990"]
    for h in range(7):
        raws.append(f"{station} 17{12 + h:02d}00Z 28012KT 10SM BKN035 26/12 A2992")
    store.insert_obs(con, [metar_parse(r) for r in raws], year=2026, month=7, source="test")


con = store.connect(DB)
store.init_schema(con)
store.init_scoring_schema(con)

# Station WITH an amendment.
seed_obs(con, "KTST")
issue = datetime(2026, 7, 17, 11, 0)
rout = build_taf_row("TAF KTST 171100Z 1712/1718 24010KT 9999 FEW250", issue_ref=issue,
                     producer_kind="human", producer_name="unit", source="awc_poll", canonical=True)
amdr = build_taf_row("TAF AMD KTST 171500Z 1715/1718 31020G30KT 9999 BKN030", issue_ref=issue,
                     producer_kind="human", producer_name="unit", source="awc_poll", canonical=True)
subj = build_taf_row("TAF KTST 171200Z 1712/1718 26010KT 9999 SCT040", issue_ref=issue,
                     producer_kind="artificial", producer_name="model", source="agent_run",
                     canonical=True)
for row in (rout, amdr, subj):
    store.insert_taf(con, row)

check("human_bulletins_for_window: routine first + amendment",
      [b["bulletin_type"] for b in store.human_bulletins_for_window(con, "KTST", VF, VT)]
      == ["routine", "amendment"])

out = score_taf.run(con, taf_id=subj["taf_id"], scorers=["tafver", "amend", "skill"],
                    baselines=["human"])
names = [r["name"] for r in out["results"]]
comp = next((r for r in out["results"] if r["name"] == "human_composite"), None)
check("run: human_composite row present when an amendment exists", comp is not None, str(names))
check("run: composite scored by all three scorers",
      comp is not None and comp["amend"] is not None and comp["skill"] is not None
      and comp["tafver"] is not None)
check("run: composite carries construction provenance",
      comp is not None and comp.get("construction", {}).get("amendment_taf_ids") == [amdr["taf_id"]])
check("run: report shows the human_composite producer row",
      "human_composite" in out["report"])

# Station WITHOUT an amendment -> NO composite row.
seed_obs(con, "KNOA")
routN = build_taf_row("TAF KNOA 171100Z 1712/1718 24010KT 9999 FEW250", issue_ref=issue,
                      producer_kind="human", producer_name="unit", source="awc_poll", canonical=True)
subjN = build_taf_row("TAF KNOA 171200Z 1712/1718 26010KT 9999 SCT040", issue_ref=issue,
                      producer_kind="artificial", producer_name="model", source="agent_run",
                      canonical=True)
store.insert_taf(con, routN)
store.insert_taf(con, subjN)
outN = score_taf.run(con, taf_id=subjN["taf_id"], scorers=["tafver"], baselines=["human"])
check("run: NO composite row when no amendment exists",
      not any(r["name"] == "human_composite" for r in outN["results"]),
      str([r["name"] for r in outN["results"]]))

con.close()

npass = sum(p for _, p, _ in checks)
print("=== HUMAN_COMPOSITE SELF-TEST (T9) ===")
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if not passed else ""))
print(f"\n{npass}/{len(checks)} passed. Temp DB: {DB}")
