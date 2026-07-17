"""Turn a raw TAF bulletin into an immutable `tafs`-archive row (pure; no DB).

Shared by scripts/archive_taf.py (archiving human/official TAFs) and runlog.persist_run
(archiving the agent's emitted TAF), so BOTH use one taf_id scheme + window computation --
essential for the scoring join to line human vs artificial forecasts up by station+window.
"""

import hashlib
from datetime import datetime, timezone

from forecaster.tafparse import parse, strip_remarks
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
    salt: str | None = None,
) -> dict:
    """Parse a raw TAF and build a `tafs` row dict with an absolute UTC window and a
    content-derived taf_id. Pure (no DB); tested deterministically. Callers may add the
    lineage columns (run_id, experiment_id, worksheet_id, taf_product_json) afterward.

    `salt` (e.g. a run_id) disambiguates the taf_id ONLY: two matrix cells emitting
    byte-identical TAF text must not collide on taf_id (each needs its own lineage row),
    while the human/poller path passes no salt so content-dedup is preserved. The
    `content_sha256` COLUMN always hashes the raw alone (tamper evidence, salt-independent)."""
    # Populate parse_body with the remark-stripped body when remarks were actually present
    # (AF remarks have no delimiter and corrupt the parse). NULL when nothing was stripped,
    # so a clean TAF's column stays NULL (the raw stays byte-exact in raw_taf either way).
    if parse_body is None:
        stripped, _rmk = strip_remarks(raw)
        if stripped != raw.strip().rstrip("=").strip():
            parse_body = stripped
    obs = parse(parse_body if parse_body is not None else raw)
    issue, valid_from, valid_to = absolute_validity(obs, issue_ref)
    if bulletin_type is None:
        bulletin_type = ("correction" if obs.corrected else
                         "cancellation" if obs.canceled else
                         "amendment" if obs.amendment else "routine")
    sha = content_sha256(raw)
    id_sha = (hashlib.sha256((raw.strip() + "|" + salt).encode("utf-8")).hexdigest()
              if salt else sha)
    return {
        "taf_id": f"{obs.station}-{issue:%Y%m%d%H%M}-{id_sha[:12]}",
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
