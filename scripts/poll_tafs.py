"""Poll AWC for new official TAFs and archive them (the human-forecast collection cron).

One pass over the station roster: fetch each station's current TAF and archive it into the
`tafs` table. Idempotent by content hash, so running this every few minutes only ever adds
a genuinely NEW bulletin (a fresh cycle or an amendment) -- exactly the "archive it the
moment it is posted" behavior, with the cron handling repetition. Archiving at issue time,
before the validity window elapses, is what makes the benchmark leakage-proof: the human
forecast is frozen before its own truth exists.

Two phases so the throttled (1 req/s) AWC fetches happen OFF the single-writer lock: phase
1 fetches every station's TAF rows with NO lock held; phase 2 takes the lock ONCE and
inserts all rows quickly. The old design held the DB lock for a minute-plus of pure network
I/O every 5 minutes, queueing the collector's persists and the scorer behind it (T2).

  uv run python scripts/poll_tafs.py                 # roster + archive-only sites
  uv run python scripts/poll_tafs.py --stations KWRI KMIB
"""

import argparse
from datetime import datetime, timezone

from forecaster import awc, stations, store


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive current official TAFs for the roster.")
    ap.add_argument("--stations", nargs="*",
                    help="ICAO subset (default: model roster + archive-only sites)")
    ap.add_argument("--db", default=None, help="benchmark DB path (default: settings.db_path)")
    args = ap.parse_args()

    icaos = [s.upper() for s in args.stations] if args.stations else stations.poll_icaos()
    now = datetime.now(timezone.utc)
    print(f"[{now:%Y-%m-%dT%H:%MZ}] polling {len(icaos)} station(s) for new TAFs")

    # Phase 1 (NO lock): throttled network fetches for every station.
    fetched: list[tuple[str, list[dict], list[tuple[str, str]]]] = []
    for icao in icaos:
        try:
            rows, errors = awc.fetch_taf_rows(icao)
        except Exception as e:  # noqa: BLE001 -- a per-station fetch failure is not fatal to the pass
            print(f"  {icao}: ERROR {type(e).__name__}: {e}")
            continue
        fetched.append((icao, rows, errors))

    # Phase 2 (ONE lock hold): insert everything quickly, then release.
    total_new = total_seen = 0
    with store.write_lock(args.db):                 # single-writer: archive vs collect vs score
        con = store.connect(args.db) if args.db else store.connect()
        try:
            store.init_scoring_schema(con)
            for icao, rows, errors in fetched:
                new_bulletins = []
                for row in rows:
                    total_seen += 1
                    if store.insert_taf(con, row):
                        new_bulletins.append((row["taf_id"], row["bulletin_type"]))
                total_new += len(new_bulletins)
                if new_bulletins:
                    detail = ", ".join(f"{bt} {tid.split('-')[1]}Z" for tid, bt in new_bulletins)
                    print(f"  {icao}: NEW x{len(new_bulletins)} ({detail})")
                else:
                    print(f"  {icao}: no change ({len(rows)} current)")
                for raw, err in errors:
                    # full bulletin, not a prefix: the malformation is usually in a LATE
                    # change group, and a truncated log line cannot be diagnosed later.
                    print(f"    ! {err}\n      {raw.strip()}")
        finally:
            con.close()

    print(f"done: {total_new} newly archived across {len(icaos)} station(s) "
          f"({total_seen} current bulletins seen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
