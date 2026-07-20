"""Turn a raw TAF bulletin into an immutable `tafs`-archive row (pure; no DB).

Shared by scripts/archive_taf.py (archiving human/official TAFs) and runlog.persist_run
(archiving the agent's emitted TAF), so BOTH use one taf_id scheme + window computation --
essential for the scoring join to line human vs artificial forecasts up by station+window.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

from forecaster.tafparse import parse, repair_validity, strip_remarks
from forecaster.tafstate import absolute_validity


def content_sha256(raw: str) -> str:
    return hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()


# Enough of the bulletin header to place an UNPARSEABLE TAF on the timeline: the
# malformation we see live is always in a later change group, so the header survives.
_HEADER = re.compile(
    r"\bTAF\b(?:\s+(?:AMD|COR|RTD))*\s+([A-Z][A-Z0-9]{3})\s+\d{6}Z\s+"
    r"(\d{2})(\d{2})/(\d{2})(\d{2})"
)


def _anchor(day: int, hour: int, ref: datetime) -> datetime:
    """A DDHH token as absolute UTC, anchored on `ref`: rolls into the next month when
    the day number sits well behind the reference, and normalizes hour 24 to 00 of the
    following day. Approximate by construction -- used only for quarantined bulletins,
    where an indexable window beats no row at all."""
    base = ref.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    year, month = base.year, base.month
    if day < base.day - 15:                 # day number wrapped past the month end
        month += 1
        if month == 13:
            year, month = year + 1, 1
    roll = 0
    if hour == 24:
        hour, roll = 0, 1
    return datetime(year, month, day, hour) + timedelta(days=roll)


def _quarantine_row(raw: str, *, issue_ref: datetime, error: str, producer_kind: str,
                    producer_name: str | None, source: str, canonical: bool,
                    bulletin_type: str | None, salt: str | None) -> dict:
    """Archive row for a bulletin we could not parse OR repair. Station and window come
    from the header regex, so the TAF still lands on the timeline and can be re-parsed
    later if the seam learns its malformation. Everything group-derived stays NULL.

    A malformed human bulletin is itself a finding -- syntax errors are a human failure
    mode the generated TAFs do not have -- so it is recorded, never dropped."""
    raw = raw.strip()
    m = _HEADER.search(raw)
    if m is None:
        raise ValueError(f"unparseable TAF with no readable header: {raw[:80]!r}")
    station, fd, fh, td, th = m.group(1), *(int(g) for g in m.groups()[1:])
    issue = issue_ref.replace(tzinfo=None) if issue_ref.tzinfo else issue_ref
    valid_from = _anchor(fd, fh, issue)
    valid_to = _anchor(td, th, valid_from)
    if bulletin_type is None:
        head = raw.split(station)[0]
        bulletin_type = ("correction" if "COR" in head else
                         "amendment" if "AMD" in head else "routine")
    sha = content_sha256(raw)
    id_sha = (hashlib.sha256((raw + "|" + salt).encode("utf-8")).hexdigest()
              if salt else sha)
    return {
        "taf_id": f"{station}-{issue:%Y%m%d%H%M}-{id_sha[:12]}",
        "station": station,
        "issue_time_utc": issue,
        "valid_from_utc": valid_from,
        "valid_to_utc": valid_to,
        "original_cycle_start_utc": valid_from,
        "bulletin_type": bulletin_type,
        "producer_kind": producer_kind,
        "producer_name": producer_name,
        "source": source,
        "canonical": canonical,
        "raw_taf": raw,
        "parse_body": None,
        "parse_status": "failed",
        "parse_error": error,
        "repairs_json": None,
        "content_sha256": sha,
        "archived_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }


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
    on_parse_error: str = "raise",
) -> dict:
    """Parse a raw TAF and build a `tafs` row dict with an absolute UTC window and a
    content-derived taf_id. Pure (no DB); tested deterministically. Callers may add the
    lineage columns (run_id, experiment_id, worksheet_id, taf_product_json) afterward.

    `salt` (e.g. a run_id) disambiguates the taf_id ONLY: two matrix cells emitting
    byte-identical TAF text must not collide on taf_id (each needs its own lineage row),
    while the human/poller path passes no salt so content-dedup is preserved. The
    `content_sha256` COLUMN always hashes the raw alone (tamper evidence, salt-independent).

    `on_parse_error='quarantine'` archives a bulletin that will not parse even after
    repair, keeping the raw text and a header-derived window (parse_status='failed').
    The default 'raise' preserves the strict contract for the agent's own emitted TAFs,
    which come from tafgen and must never be malformed."""
    # Populate parse_body with the remark-stripped body when remarks were actually present
    # (AF remarks have no delimiter and corrupt the parse). NULL when nothing was stripped,
    # so a clean TAF's column stays NULL (the raw stays byte-exact in raw_taf either way).
    if parse_body is None:
        stripped, _rmk = strip_remarks(raw)
        if stripped != raw.strip().rstrip("=").strip():
            parse_body = stripped
    body = parse_body if parse_body is not None else raw
    try:
        obs = parse(body)
    except Exception as e:  # noqa: BLE001 -- caller decides: strict contract or quarantine
        if on_parse_error != "quarantine":
            raise
        return _quarantine_row(raw, issue_ref=issue_ref, error=f"{type(e).__name__}: {e}",
                               producer_kind=producer_kind, producer_name=producer_name,
                               source=source, canonical=canonical,
                               bulletin_type=bulletin_type, salt=salt)
    # A repaired bulletin must hand DOWNSTREAM re-parsers (scoring, tafstate) the fixed
    # text, so persist it as the parse_body; raw_taf still holds what was transmitted.
    parse_status, repairs = "ok", None
    if obs.repairs:
        parse_status, repairs = "repaired", json.dumps(obs.repairs)
        parse_body = repair_validity(body)[0]
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
        "parse_status": parse_status,
        "parse_error": None,
        "repairs_json": repairs,
        "content_sha256": sha,
        "archived_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
