"""Poller self-test (T2): fetches happen OFF the single-writer lock. No network.

Monkeypatches awc.fetch_taf with canned bulletins and wraps store.write_lock with a timing
flag; then runs poll_tafs.main against a throwaway DB and asserts (a) the bulletins land in
`tafs`, (b) idempotent re-poll adds 0 rows, and (c) NO fetch_taf call happened while the
write lock was held.
"""

import contextlib
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import poll_tafs
from forecaster import awc, store

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))


tmp = Path(tempfile.mkdtemp(prefix="poll_test_"))
DB = str(tmp / "bench.duckdb")

# Canned bulletins for two stations (issue, raw). Parseable by build_taf_row.
CANNED = {
    "KWRI": [(datetime(2026, 7, 17, 11, 0), "TAF KWRI 171100Z 1712/1818 24010KT 9999 FEW250")],
    "KMIB": [(datetime(2026, 7, 17, 11, 0), "TAF KMIB 171100Z 1712/1818 31012KT 9999 SCT200")],
}

_locked = {"held": False}
_fetch_during_lock = {"violated": False}
_real_lock = store.write_lock
_real_fetch = awc.fetch_taf


@contextlib.contextmanager
def _timed_lock(db_path=None):
    with _real_lock(db_path):
        _locked["held"] = True
        try:
            yield
        finally:
            _locked["held"] = False


def _stub_fetch(station):
    if _locked["held"]:
        _fetch_during_lock["violated"] = True     # a fetch while the lock is held = the bug
    return list(CANNED.get(str(station).upper(), []))


def _run():
    sys.argv = ["poll_tafs.py", "--stations", "KWRI", "KMIB", "--db", DB]
    return poll_tafs.main()


try:
    store.write_lock = _timed_lock
    awc.fetch_taf = _stub_fetch
    # poll_tafs imported store/awc by name; patch the names it actually calls.
    poll_tafs.store.write_lock = _timed_lock
    poll_tafs.awc.fetch_taf = _stub_fetch

    rc = _run()
    check("poll returns 0", rc == 0)

    con = store.connect(DB, read_only=True)
    try:
        n = con.execute("SELECT count(*) FROM tafs").fetchone()[0]
        stations = {r[0] for r in con.execute("SELECT DISTINCT station FROM tafs").fetchall()}
        check("both canned bulletins archived", n == 2 and stations == {"KWRI", "KMIB"},
              f"n={n} stations={stations}")
    finally:
        con.close()

    check("NO fetch happened while the write lock was held",
          not _fetch_during_lock["violated"])

    # Idempotent re-poll: 0 new rows.
    _run()
    con = store.connect(DB, read_only=True)
    try:
        n2 = con.execute("SELECT count(*) FROM tafs").fetchone()[0]
        check("re-poll adds 0 rows (idempotent)", n2 == 2, f"n2={n2}")
    finally:
        con.close()
finally:
    store.write_lock = _real_lock
    awc.fetch_taf = _real_fetch
    poll_tafs.store.write_lock = _real_lock
    poll_tafs.awc.fetch_taf = _real_fetch

npass = sum(p for _, p, _ in checks)
print("=== POLLER SELF-TEST (T2) ===")
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if not passed else ""))
print(f"\n{npass}/{len(checks)} passed. Temp DB: {DB}")
