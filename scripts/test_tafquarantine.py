"""Malformed-bulletin handling self-test. No model/network.

Covers the two-layer response to a human coding error in a TAF:
  1. tafparse.repair_validity -- fix an over-long validity token when the grammar admits
     exactly ONE reading (observed live: KFAF 'BECMG 2100/21001'), and refuse otherwise.
  2. tafarchive quarantine -- archive the bulletin anyway when it cannot be repaired, so
     a fat-fingered human TAF is never silently dropped (it is both a scoring baseline
     and evidence of a human failure mode the generated TAFs do not have).
"""

import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import store
from forecaster.tafarchive import build_taf_row
from forecaster.tafparse import parse, repair_validity

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))


ISSUE = datetime(2026, 7, 20, 10, 0)

# The real bulletin that broke the poller on 2026-07-20 (KFAF, Felker AAF).
KFAF = ("TAF KFAF 201000Z 2010/2116 VRB06KT 9999 VCTS BKN050 QNH2988INS "
        "BECMG 2012/2013 VRB06KT 9999 NSW BKN025 QNH2991INS "
        "BECMG 2016/2017 VRB06KT 9999 BKN035 QNH2989INS "
        "BECMG 2100/21001 VRB06KT 9999 BKN045 QNH2984INS TX28/2018Z TN25/2010Z")
CLEAN = ("TAF KDMA 181100Z 1811/1917 14009KT 9999 VCSH SCT060 BKN100 QNH3007INS "
         "BECMG 1816/1817 31009KT 9999 NSW SCT060 QNH3006INS TX32/1822Z TN24/1812Z")

# --- 1. repair: the unique legal reading is taken ---
fixed, repairs = repair_validity(KFAF)
check("repair: 2100/21001 -> 2100/2101", repairs == [("2100/21001", "2100/2101")], str(repairs))
check("repair: rewrites only that token",
      fixed == KFAF.replace("2100/21001", "2100/2101"), fixed)

# --- 2. parse() retries with the repair and keeps the raw byte-exact ---
obs = parse(KFAF)
g = obs.groups[-1]
check("parse: recovers all 3 change groups", len(obs.groups) == 3, str(len(obs.groups)))
check("parse: repaired group window is 2100/2101",
      (g.from_day, g.from_hour, g.to_day, g.to_hour) == (21, 0, 21, 1),
      f"{g.from_day}{g.from_hour:02d}/{g.to_day}{g.to_hour:02d}")
check("parse: TafObs.raw stays the transmitted text", obs.raw == KFAF)
check("parse: repair is recorded on the TafObs", obs.repairs == [("2100/21001", "2100/2101")],
      str(obs.repairs))

# --- 3. a sound bulletin is never touched ---
check("clean TAF: no repair attempted", repair_validity(CLEAN)[1] == [])
check("clean TAF: parse records no repairs", parse(CLEAN).repairs == [])

# --- 4. refuse to guess: ambiguity and month-wrap are left alone ---
check("refuses unrepairable garbage",
      repair_validity("TAF KXXX 201000Z 2010/2116 BECMG 999999/2101 12005KT")[1] == [])
check("refuses when the header itself wraps the month",
      repair_validity("TAF KXXX 312300Z 3123/0105 BECMG 0100/01001 12005KT")[1] == [])
check("header token is never rewritten",
      repair_validity("TAF KXXX 201000Z 20100/2116 12005KT")[1] == [])

# --- 5. quarantine: unparseable body, readable header ---
BAD = "TAF AMD KVBG 172215Z 1722/1904 29010KT 9999 BKN010 QNH2998INS BECMG @@@@/@@@@ 31020KT"
row = build_taf_row(BAD, issue_ref=datetime(2026, 7, 17, 22, 15), producer_kind="human",
                    canonical=True, on_parse_error="quarantine")
check("quarantine: parse_status='failed'", row["parse_status"] == "failed", row["parse_status"])
check("quarantine: raw kept byte-exact", row["raw_taf"] == BAD)
check("quarantine: station from header", row["station"] == "KVBG", row["station"])
check("quarantine: window from header",
      (row["valid_from_utc"], row["valid_to_utc"])
      == (datetime(2026, 7, 17, 22), datetime(2026, 7, 19, 4)),
      f"{row['valid_from_utc']} .. {row['valid_to_utc']}")
check("quarantine: bulletin_type read from AMD", row["bulletin_type"] == "amendment",
      row["bulletin_type"])
check("quarantine: records the error", bool(row["parse_error"]), str(row["parse_error"]))

# --- 6. the strict contract still holds where it must ---
try:
    build_taf_row(BAD, issue_ref=ISSUE)
    check("default on_parse_error='raise' still raises", False, "no exception")
except Exception as e:
    check("default on_parse_error='raise' still raises", True, type(e).__name__)
try:
    build_taf_row("not a taf at all", issue_ref=ISSUE, on_parse_error="quarantine")
    check("quarantine refuses a bulletin with no readable header", False, "no exception")
except ValueError:
    check("quarantine refuses a bulletin with no readable header", True)

# --- 7. a repaired bulletin archives with the FIXED text as parse_body ---
rep = build_taf_row(KFAF, issue_ref=ISSUE, producer_kind="human", canonical=True,
                    on_parse_error="quarantine")
check("repaired row: parse_status='repaired'", rep["parse_status"] == "repaired",
      rep["parse_status"])
check("repaired row: raw_taf is the transmitted text", rep["raw_taf"] == KFAF)
check("repaired row: parse_body carries the fix",
      rep["parse_body"] is not None and "2100/2101" in rep["parse_body"],
      str(rep["parse_body"]))
check("repaired row: parse_body re-parses downstream",
      len(parse(rep["parse_body"]).groups) == 3)
check("repaired row: repairs_json recorded", "21001" in (rep["repairs_json"] or ""),
      str(rep["repairs_json"]))
check("repaired row: window is real, not header-guessed",
      rep["valid_from_utc"] == datetime(2026, 7, 20, 10), str(rep["valid_from_utc"]))

# --- 8. store round-trip, incl. migrating a DB created before these columns ---
TMP = Path(tempfile.mkdtemp(prefix="quarantine_test_"))

# An OLD DB: the shipped tafs schema with the three new columns stripped back out, which
# is exactly what the live Pi archive looks like before this change.
old = store.connect(str(TMP / "old.duckdb"))
old.execute("\n".join(ln for ln in store._TAFS_DDL.splitlines()
                      if not ln.strip().startswith(("parse_status", "parse_error",
                                                    "repairs_json"))))
store.init_scoring_schema(old)             # must migrate it forward, not fail
cols = {r[0] for r in old.execute("DESCRIBE tafs").fetchall()}
check("migration: adds columns to a pre-existing tafs table",
      {"parse_status", "parse_error", "repairs_json"} <= cols, str(sorted(cols)))
check("migration: pre-existing columns survive", "raw_taf" in cols and "parse_body" in cols)
store.insert_taf(old, row)
check("migration: a quarantined row inserts into the migrated table",
      old.execute("SELECT parse_status FROM tafs").fetchone()[0] == "failed")
old.close()

DB = str(TMP / "bench.duckdb")
con = store.connect(DB)
store.init_scoring_schema(con)
for r in (row, rep):
    store.insert_taf(con, r)
got = {r[0]: r for r in con.execute(
    "SELECT station, parse_status, parse_error, repairs_json FROM tafs").fetchall()}
check("store: quarantined row persists as failed", got["KVBG"][1] == "failed", str(got.get("KVBG")))
check("store: repaired row persists as repaired", got["KFAF"][1] == "repaired", str(got.get("KFAF")))
check("store: idempotent re-archive is a no-op", store.insert_taf(con, row) is False)

# The point of all this: malformed HUMAN bulletins stay queryable as a finding.
n_bad = con.execute("SELECT count(*) FROM tafs WHERE parse_status <> 'ok'").fetchone()[0]
check("store: malformed bulletins are countable for reporting", n_bad == 2, str(n_bad))
con.close()

npass = sum(p for _, p, _ in checks)
print("=== MALFORMED-BULLETIN SELF-TEST ===")
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if not passed else ""))
print(f"\n{npass}/{len(checks)} passed. Temp DB: {DB}")
