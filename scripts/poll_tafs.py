"""Poll AWC for new official TAFs and archive them (the human-forecast collection cron).

One pass over the station roster: fetch each station's current TAF and archive it into the
`tafs` table. Idempotent by content hash, so running this every few minutes only ever adds
a genuinely NEW bulletin (a fresh cycle or an amendment) -- exactly the "archive it the
moment it is posted" behavior, with the cron handling repetition. Archiving at issue time,
before the validity window elapses, is what makes the benchmark leakage-proof: the human
forecast is frozen before its own truth exists.

The whole pass runs under the single-writer lock so it never collides with the collector
or the scorer writing the same .duckdb.

  uv run python scripts/poll_tafs.py                 # all roster stations
  uv run python scripts/poll_tafs.py --stations KWRI KMIB
"""

import argparse
from datetime import datetime, timezone

from forecaster import awc, stations, store


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive current official TAFs for the roster.")
    ap.add_argument("--stations", nargs="*", help="ICAO subset (default: the whole roster)")
    ap.add_argument("--db", default=None, help="benchmark DB path (default: settings.db_path)")
    args = ap.parse_args()

    icaos = [s.upper() for s in args.stations] if args.stations else stations.icaos()
    now = datetime.now(timezone.utc)
    print(f"[{now:%Y-%m-%dT%H:%MZ}] polling {len(icaos)} station(s) for new TAFs")

    total_new = total_seen = 0
    with store.write_lock(args.db):                 # single-writer: archive vs collect vs score
        for icao in icaos:
            try:
                summ = awc.load_taf(icao, db_path=args.db)
            except Exception as e:  # noqa: BLE001 -- a per-station fetch failure is not fatal to the pass
                print(f"  {icao}: ERROR {type(e).__name__}: {e}")
                continue
            total_seen += summ["archived"]
            total_new += len(summ["new"])
            tag = f"NEW x{len(summ['new'])}" if summ["new"] else "no change"
            detail = f" {summ['new']}" if summ["new"] else ""
            print(f"  {icao}: {tag} ({summ['archived']} current){detail}")
            for raw, err in summ["errors"]:
                print(f"    ! parse error: {err} :: {raw[:60]}")

    print(f"done: {total_new} newly archived across {len(icaos)} station(s) "
          f"({total_seen} current bulletins seen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
