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
import hashlib
from datetime import datetime, timezone

from forecaster import store
from forecaster.tafparse import parse
from forecaster.tafstate import absolute_validity


def content_sha256(raw: str) -> str:
    return hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()


def build_taf_row(
    raw: str,
    *,
    issue_ref: datetime,
    producer_kind: str = "official",
    producer_name: str | None = None,
    source: str = "import",
    canonical: bool = False,
    bulletin_type: str | None = None,
    parse_body: str | None = None,
) -> dict:
    """Parse a raw TAF and build a `tafs` row dict with an absolute UTC window and a
    content-derived taf_id. Pure (no DB); tested deterministically."""
    obs = parse(raw)
    issue, valid_from, valid_to = absolute_validity(obs, issue_ref)
    if bulletin_type is None:
        bulletin_type = ("correction" if obs.corrected else
                         "cancellation" if obs.canceled else
                         "amendment" if obs.amendment else "routine")
    sha = content_sha256(raw)
    return {
        "taf_id": f"{obs.station}-{issue:%Y%m%d%H%M}-{sha[:12]}",
        "station": obs.station,
        "issue_time_utc": issue,
        "valid_from_utc": valid_from,
        "valid_to_utc": valid_to,
        "original_cycle_start_utc": valid_from,     # routine: == valid_from (amend inherits parent)
        "bulletin_type": bulletin_type,
        "producer_kind": producer_kind,
        "producer_name": producer_name,
        "source": source,
        "canonical": canonical,
        "raw_taf": raw.strip(),
        "parse_body": parse_body,
        "content_sha256": sha,
        "archived_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }


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
