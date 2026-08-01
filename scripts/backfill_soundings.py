"""Replace archived STAGED-RELEASE soundings with the complete ascent.

    uv run python scripts/backfill_soundings.py                # report only
    uv run python scripts/backfill_soundings.py --apply        # rewrite the truncated ones

WHY THIS EXISTS. The Wyoming feed publishes an ascent in stages, most often truncated at
100 hPa, and the archiver froze whichever stage was posted when the sweep ran (contract rule
5: `ON CONFLICT DO NOTHING`, first capture wins). Measured 2026-07-31 across five sampled
captures: three held the 100 hPa version -- one of them fetched 129 minutes after launch --
so the loss is not a race the capture clock can win. Each truncated copy is missing the upper
troposphere and stratosphere, which is the tropopause and jet level a TAF's turbulence and
icing reasoning reads.

WHY A REWRITE IS LEGITIMATE HERE, when rule 5 normally forbids one. Rule 5 protects "serve a
replay what the model actually saw". Nothing has been served yet -- Phase 3.2 is unbuilt and
no round has run against this archive -- so the rule is not protecting anything, and freezing
a known-defective input for the whole round is the larger harm. Unlike imagery, an ascent IS
recoverable: Wyoming still serves past launches (verified at 45 h). This script must NOT be
run once a round has been served from the archive.

WHAT IT DOES NOT TOUCH: an ascent that is already complete, and one whose provider still has
only the staged version (it stays truncated and is reported, so a later pass can retry).
"""

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from forecaster import artifacts, charts, soundings, store

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "archive"


def connect_index(path: Path, *, read_only: bool):
    """Open the index, waiting out the hourly sweep's short lock windows."""
    import time
    for _ in range(60):
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            time.sleep(1)
    raise SystemExit(f"could not open {path}: still locked after 60 s")


def archived_soundings(con) -> list[dict]:
    rows = con.execute(
        """SELECT k.kind, k.identity, k.requested_utc, k.sha256, k.source_url,
                  k.provider, k.served_utc, k.fetched_utc, a.mime
           FROM artifact_keys k LEFT JOIN artifacts a ON a.sha256 = k.sha256
           WHERE k.kind = 'sounding'
           ORDER BY k.requested_utc"""
    ).fetchall()
    cols = ["kind", "identity", "requested_utc", "sha256", "source_url",
            "provider", "served_utc", "fetched_utc", "mime"]
    return [dict(zip(cols, r)) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite truncated ascents; without it, only report")
    ap.add_argument("--index", default=str(ARCHIVE / "index.duckdb"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N rewrites (0 = all)")
    a = ap.parse_args()

    # THREE PHASES, and the split is not cosmetic. DuckDB takes an exclusive FILE lock, so a
    # connection held across the network+render work would block the hourly sweep for the
    # whole 20-30 minute pass -- which is exactly how a long-running reader destroyed 67
    # stations of imagery on 2026-07-29. Read briefly, work with NO connection open, then
    # reopen for a short write. Blobs are plain files and need no lock, so phase 2 can put
    # them on disk and carry only the small index updates forward.
    con = connect_index(Path(a.index), read_only=True)
    try:
        rows = archived_soundings(con)
    finally:
        con.close()
    print(f"{len(rows)} archived sounding(s) in the index\n")
    complete = truncated = fixed = still_partial = failed = 0
    updates: list[tuple] = []
    if True:                                  # phase 2: network + render, NO index held
        for r in rows:
            ident, when = r["identity"], r["requested_utc"]
            # identity is "<wmo>/<src>" -- the feed matters, the two carry different launches.
            wmo, _, src = ident.partition("/")
            try:
                fresh = soundings.fetch_profile(wmo, when, src=(src or "bufr").upper())
            except Exception as e:  # noqa: BLE001 -- one dead site must not stop the pass
                print(f"  FETCH FAILED {ident} {when:%m-%d %H:%MZ} ({type(e).__name__}: {e})", flush=True)
                failed += 1
                continue
            if (note := soundings.ascent_stage_note(fresh)) is not None:
                # The provider still holds only the staged version. Leave the archive alone,
                # and do NOT render -- there is nothing to compare and nothing to write.
                print(f"  STILL PARTIAL {ident} {when:%m-%d %H:%MZ} -- {note}", flush=True)
                still_partial += 1
                continue
            if not a.apply:
                # REPORT MODE DOES NOT RENDER. Deciding "is the stored copy already this
                # ascent" needs a skew-T to compare addresses against, and that render is by
                # far the slowest step -- 205 of them overran a 50-minute budget on the Pi.
                # The actionable number is how many launches the provider can now serve
                # complete; the exact already-identical split only matters when writing.
                truncated += 1
                print(f"  CANDIDATE {ident} {when:%m-%d %H:%MZ} "
                      f"-> {fresh.n_raw} raw levels, top {min(fresh.pres):.0f} hPa", flush=True)
                continue
            # The archived bytes are a PNG, so completeness cannot be read back out of them.
            # Render the fresh profile and compare ADDRESSES instead: an identical sha means
            # the stored copy already IS this complete ascent.
            png = charts.skewt(fresh)
            sha = hashlib.sha256(png).hexdigest()
            if sha == r["sha256"]:
                complete += 1
                continue
            truncated += 1
            # The blob goes to disk NOW (a plain file write, no lock), and only the index
            # update is carried to phase 3. A crash here leaves an orphan blob, which is
            # harmless and is re-derived on a re-run.
            sha, _mime, _n, _new = artifacts.put(png, root=ARCHIVE)
            updates.append((sha, fresh.url, len(png), ident, when))
            fixed += 1
            print(f"  REWROTE {ident} {when:%m-%d %H:%MZ} -> {fresh.n_raw} raw levels, "
                  f"top {min(fresh.pres):.0f} hPa", flush=True)
            if a.limit and fixed >= a.limit:
                print(f"  (stopping at --limit {a.limit})")
                break

    if updates:
        seen = datetime.now(timezone.utc).replace(tzinfo=None)
        stamp = f"{seen:%Y-%m-%dT%H:%MZ}"
        wcon = connect_index(Path(a.index), read_only=False)
        try:
            for sha, url, n_bytes, ident, when in updates:
                store.insert_artifact(wcon, sha256=sha, kind="sounding",
                                      mime="image/png", n_bytes=n_bytes,
                                      first_seen_utc=seen)
                wcon.execute(
                    "UPDATE artifact_keys SET sha256 = ?, source_url = ?, note = ? "
                    "WHERE kind = 'sounding' AND identity = ? AND requested_utc = ?",
                    [sha, url,
                     f"backfilled {stamp}: the first capture was a staged release; "
                     f"this is the complete ascent", ident, when])
        finally:
            wcon.close()
        print(f"\nindex updated in one short write: {len(updates)} row(s)")

    if a.apply:
        print(f"\nrewritten: {fixed}"
              f"\nalready held the complete ascent: {complete}"
              f"\nstill only staged at the provider: {still_partial}"
              f"\nfetch failed: {failed}")
    else:
        # Report mode cannot split "candidate" into rewrite-vs-already-correct without
        # rendering each one, so it does not pretend to. A candidate is a launch the
        # provider can now serve COMPLETE; --apply then skips any whose bytes match.
        print(f"\ncandidates (provider now has the complete ascent): {truncated}"
              f"\nstill only staged at the provider: {still_partial}"
              f"\nfetch failed: {failed}"
              f"\n\nreport only -- nothing was written. Pass --apply to rewrite; it "
              f"re-checks each candidate and skips those already correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
