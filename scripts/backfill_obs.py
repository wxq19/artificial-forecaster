"""Bulk-backfill truth METARs from IEM for the whole polled station set.

WHY THIS EXISTS: the scorers (score_taf.py --pending / --archive-difficulty) each carry
a PER-TAF `--backfill iem` fallback that pulls one ~33h window at a time. That is the
wrong shape for difficulty mining: 63 stations x ~2 routine TAFs/day means the nightly
pass fires ~2 IEM requests per TAF (report_type 3 and 4), and IEM 429s after ~26 requests
at the 2s throttle -- so coverage decayed through the run (43% at 11Z down to 27% at 16Z
on 2026-07-25), everything under --min-coverage was skipped, and the same TAFs 429'd
again the next night. The backlog grew instead of draining.

One pass here pulls each station ONCE over the whole range, so the cost is ~2 requests
per station per chunk instead of ~2 per TAF -- roughly 130 requests for a week of all 63
stations, against ~2000 the per-TAF path would have made. Run it BEFORE the scorers and
their fallback never fires.

Network I/O happens OFF the single-writer lock (iem.collect), and the lock is taken only
for each station's fast insert (iem.persist) -- the poll_tafs.py two-phase pattern, so a
long backfill does not queue the poller behind minutes of HTTP.

  uv run python scripts/backfill_obs.py                      # polled set, last 7 days
  uv run python scripts/backfill_obs.py --days 14
  uv run python scripts/backfill_obs.py --stations KWRI KMIB --days 3
  uv run python scripts/backfill_obs.py --start 2026-07-20 --end 2026-07-27
  uv run python scripts/backfill_obs.py --dry-run           # show the plan, fetch nothing
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

from forecaster import iem, stations, store
from forecaster.config import settings

# Extra gap between stations, on top of iem's own 2s per-request throttle. Bulk passes
# are the case IEM's rate limiter exists for; the retry/backoff in iem._get handles a
# 429 when we still trip one, but spacing makes that the exception.
_STATION_GAP_S = 3.0


def _parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk-backfill IEM METARs for the polled station set.")
    ap.add_argument("--stations", nargs="*",
                    help="ICAO subset (default: model roster + archive-only sites)")
    ap.add_argument("--days", type=int, default=7,
                    help="look-back window ending now (default 7; ignored with --start)")
    ap.add_argument("--start", type=_parse_day, help="explicit UTC start date YYYY-MM-DD")
    ap.add_argument("--end", type=_parse_day,
                    help="explicit UTC end date YYYY-MM-DD (default: today)")
    ap.add_argument("--chunk-days", type=int, default=14,
                    help="split longer ranges into chunks of this many days (default 14)")
    ap.add_argument("--db", default=settings.db_path, help="benchmark DB path")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and each station's current latest ob; fetch nothing")
    args = ap.parse_args()

    icaos = [s.upper() for s in args.stations] if args.stations else stations.poll_icaos()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    end = args.end or now
    start = args.start or (end - timedelta(days=args.days))
    if start >= end:
        print(f"ERROR: start {start:%Y-%m-%d} is not before end {end:%Y-%m-%d}")
        return 2

    # IEM's range is day-granular, so chunk on whole days. One chunk = 2 requests.
    chunks: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=args.chunk_days), end)
        chunks.append((cur, nxt))
        cur = nxt

    print(f"[{now:%Y-%m-%dT%H:%MZ}] backfill {len(icaos)} station(s) "
          f"{start:%Y-%m-%d} .. {end:%Y-%m-%d} in {len(chunks)} chunk(s) "
          f"= ~{len(icaos) * len(chunks) * 2} IEM request(s)", flush=True)

    if args.dry_run:
        con = store.connect(args.db, read_only=True)
        try:
            for icao in icaos:
                rows = store.latest(con, icao, limit=1)
                have = f"{rows[0]['obs_time']:%Y-%m-%dT%H:%MZ}" if rows else "none"
                print(f"  {icao}: latest ob {have}")
        finally:
            con.close()
        print("dry run: nothing fetched")
        return 0

    totals = {"fetched": 0, "parsed": 0, "inserted": 0}
    failures: list[tuple[str, str]] = []
    parse_errors = 0

    for i, icao in enumerate(icaos):
        st_inserted = st_fetched = 0
        for c_start, c_end in chunks:
            try:
                # Phase 1: network + parse, NO lock held.
                collected = iem.collect(icao, c_start, c_end)
                # Phase 2: lock only for the insert.
                with store.write_lock(args.db):
                    inserted = iem.persist(collected, db_path=args.db)
            except Exception as e:                   # noqa: BLE001 — one station must not kill the pass
                failures.append((icao, f"{type(e).__name__}: {e}"))
                continue
            st_fetched += collected["fetched"]
            st_inserted += inserted
            parse_errors += len(collected["errors"])
            totals["fetched"] += collected["fetched"]
            totals["parsed"] += collected["parsed"]
            totals["inserted"] += inserted
        # flush: this runs for many minutes under cron with stdout redirected to a log
        # (block-buffered), and a silent log for the whole run is useless for watching it.
        print(f"  {icao}: fetched {st_fetched}, inserted {st_inserted}", flush=True)
        if i < len(icaos) - 1:
            time.sleep(_STATION_GAP_S)

    print(f"done: fetched {totals['fetched']}, parsed {totals['parsed']}, "
          f"inserted {totals['inserted']} new ob(s)"
          + (f", {parse_errors} parse error(s)" if parse_errors else "")
          + (f", {len(failures)} station failure(s)" if failures else ""))
    for icao, err in failures:
        print(f"  FAILED {icao}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
