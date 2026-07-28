"""IEM (Iowa Environmental Mesonet) ASOS archive loader.

Pulls historical METARs that arrive WITH an authoritative UTC timestamp, so the
real year/month come from the source — no inference. Each line is parsed to a
MetarObs and persisted via store.insert_obs, grouped by (year, month) so each
insert gets a single clean period (which also makes month rollover correct).

This is an ingestion orchestrator: it uses the metar + store seams and owns no
SQL and no DuckDB import of its own.
"""

import csv
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta

from forecaster import store
from forecaster.metar import MetarObs, parse

_IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# IEM report_type code -> our report_type tag. Fetched separately so each ob's
# type is certain (IEM strips the METAR/SPECI keyword from the raw line).
_REPORT_TYPES = {"3": "METAR", "4": "SPECI"}

# Be polite to IEM's free service: enforce a minimum gap between requests so no
# caller (this loader's two fetches, or a future bulk loop over months/stations)
# can fire back-to-back and trip the rate limiter. State is module-level on
# purpose — it throttles every fetch() regardless of who calls it.
_MIN_REQUEST_INTERVAL_S = 2.0
_last_request = 0.0

# IEM rate-limits bursts: at the 2s throttle above it 429s after ~26 requests. Any
# bulk pass (the roster backfill, a climo build) WILL be rate-limited partway through,
# so a 429 is a normal step in the protocol, not a failure -- back off and continue.
# 5xx are IEM transiently overloaded and retry identically. Backoff is climo.py's
# measured-safe 15s * attempt. Descriptive UA so IEM can see who we are.
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"
_MAX_HTTP_RETRIES = 5
_RETRY_HTTP_CODES = (429, 500, 502, 503, 504)


def _get(url: str) -> str:
    """GET with the module-level throttle plus retry/backoff on rate limits and
    transient failures. The throttle is re-applied before EVERY attempt, so a retry
    can never fire back-to-back with the request that was just rejected."""
    global _last_request
    last_error: Exception | None = None
    for attempt in range(1, _MAX_HTTP_RETRIES + 1):
        if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
            time.sleep(wait)        # space requests; no penalty on an isolated first call
        _last_request = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code not in _RETRY_HTTP_CODES:
                raise
        except urllib.error.URLError as e:               # transient connection drop
            last_error = e
        if attempt < _MAX_HTTP_RETRIES:
            time.sleep(15 * attempt)
    raise last_error                                     # exhausted retries


def fetch(
    station: str,
    start: datetime,
    end: datetime,
    *,
    report_type: str = "3,4",
) -> list[tuple[datetime, str]]:
    """GET raw METARs from IEM for the date range. Returns (valid_utc, raw_line)
    pairs in the order IEM serves them (chronological).

    The range is INCLUSIVE of the end date. IEM's own day2 is EXCLUSIVE, which is a
    sharp edge callers kept getting wrong -- score_taf.py's per-TAF backfill passed
    `valid_to + 1h` and so silently dropped the LAST calendar day of every scoring
    window it fetched, leaving permanent coverage holes that read as missing obs. The
    seam normalizes it here, once, so no caller has to remember.

    report_type filters AT THE SOURCE: '3,4' = routine METARs + SPECIs (the set a
    forecaster actually sees on AWC/Skyvector — our default); '1' = the 5-minute
    MADIS high-frequency stream (not used in the AF workflow, but available if we
    ever want denser data)."""
    day2 = end.date() + timedelta(days=1)        # exclusive at the source -> inclusive here
    params = {
        "station": station,
        "data": "metar",
        "report_type": report_type,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": day2.year, "month2": day2.month, "day2": day2.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
    }
    url = f"{_IEM_URL}?{urllib.parse.urlencode(params)}"
    text = _get(url)

    out: list[tuple[datetime, str]] = []
    for row in csv.DictReader(io.StringIO(text)):
        raw = (row.get("metar") or "").strip()
        valid = (row.get("valid") or "").strip()
        if not raw or not valid or raw == "M":
            continue
        out.append((datetime.strptime(valid, "%Y-%m-%d %H:%M"), raw))
    return out


def collect(station: str, start: datetime, end: datetime) -> dict:
    """Fetch routine METARs and SPECIs SEPARATELY (report_type 3 and 4) so each ob's
    type is known with certainty, tag them, and parse. NETWORK + PARSE ONLY -- no DB
    work at all, so a bulk caller can do its slow network I/O OFF the single-writer
    lock and take the lock only for the fast insert (poll_tafs.py's two-phase pattern).
    Returns {'station','fetched','parsed','errors','by_month'}; feed it to persist()."""
    by_month: dict[tuple[int, int], list[MetarObs]] = defaultdict(list)
    errors: list[tuple[str, str]] = []
    fetched = 0
    for code, kind in _REPORT_TYPES.items():
        for ts, raw in fetch(station, start, end, report_type=code):
            fetched += 1
            try:
                obs = parse(raw)
            except Exception as e:                   # noqa: BLE001 — log & skip a bad line
                errors.append((raw, str(e)))
                continue
            obs.report_type = kind                   # IEM stripped the keyword; tag it here
            by_month[(ts.year, ts.month)].append(obs)

    return {
        "station": station,
        "fetched": fetched,
        "parsed": sum(len(b) for b in by_month.values()),
        "errors": errors,
        "by_month": by_month,
    }


def persist(collected: dict, *, db_path: str | None = None) -> int:
    """Persist a collect() result (source='iem'), grouped by (year, month) so each
    insert gets a single clean period. DB-ONLY: the caller holds the write lock around
    this, not around collect(). Idempotent -- re-runs insert 0. Returns rows inserted."""
    con = store.connect(db_path) if db_path else store.connect()
    try:
        store.init_schema(con)
        return sum(
            store.insert_obs(con, batch, year=y, month=m, source="iem")
            for (y, m), batch in sorted(collected["by_month"].items())
        )
    finally:
        con.close()


def load(
    station: str,
    start: datetime,
    end: datetime,
    *,
    db_path: str | None = None,
) -> dict:
    """Fetch, parse and persist in one call — collect() + persist(). Returns a summary:
    rows fetched, parsed, newly inserted (idempotent re-runs add 0), and any parse
    errors. Callers already holding the write lock use this; a bulk pass should use
    collect()/persist() separately to keep the network off the lock."""
    collected = collect(station, start, end)
    inserted = persist(collected, db_path=db_path)
    return {
        "station": station,
        "fetched": collected["fetched"],
        "parsed": collected["parsed"],
        "inserted": inserted,
        "errors": collected["errors"],
    }
