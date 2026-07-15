"""Snapshot a TAF into the immutable `tafs` archive (scoring-design sec 4 / 5.1).

Non-LLM utility: a TAF must be frozen BEFORE its validity elapses so scoring has a
time-aligned forecast (a freshly fetched TAF barely overlaps past obs). This captures
the exact bulletin byte-for-byte, computes its absolute UTC window, and inserts an
idempotent row (content-hashed taf_id, so re-archiving identical text is a no-op).

Usage:
  uv run python scripts/archive_taf.py --taf-text 'TAF KBLV 091730Z 0918/1024 ...' \\
      --issue-date 2026-07-09 --producer-kind official --producer-name "KBLV 61 OSS"
  uv run python scripts/archive_taf.py --taf-file path/to/taf.txt --issue-date 2026-07-09
"""

import argparse
from datetime import datetime

from forecaster import store
from forecaster.tafarchive import build_taf_row  # noqa: F401  (re-exported for callers)


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive a TAF into the tafs table.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--taf-text")
    src.add_argument("--taf-file")
    ap.add_argument("--issue-date", required=True, help="issue DATE (YYYY-MM-DD) anchoring the calendar")
    ap.add_argument("--producer-kind", default="official")
    ap.add_argument("--producer-name")
    ap.add_argument("--source", default="import")
    ap.add_argument("--canonical", action="store_true",
                    help="mark as frozen-before-truth (default False for post-hoc imports)")
    args = ap.parse_args()

    raw = args.taf_text if args.taf_text else open(args.taf_file, encoding="utf-8").read()
    issue_ref = datetime.strptime(args.issue_date, "%Y-%m-%d")
    row = build_taf_row(raw, issue_ref=issue_ref, producer_kind=args.producer_kind,
                        producer_name=args.producer_name, source=args.source,
                        canonical=args.canonical)

    con = store.connect()
    store.init_scoring_schema(con)
    added = store.insert_taf(con, row)
    con.close()
    print(f"{'INSERTED' if added else 'EXISTS (no-op)'}  taf_id={row['taf_id']}")
    print(f"  {row['station']}  valid {row['valid_from_utc']:%Y-%m-%dT%H:%MZ} .. "
          f"{row['valid_to_utc']:%Y-%m-%dT%H:%MZ}  ({row['bulletin_type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
