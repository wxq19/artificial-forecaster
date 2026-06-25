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
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime

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


def fetch(
    station: str,
    start: datetime,
    end: datetime,
    *,
    report_type: str = "3,4",
) -> list[tuple[datetime, str]]:
    """GET raw METARs from IEM for the date range. Returns (valid_utc, raw_line)
    pairs in the order IEM serves them (chronological).

    report_type filters AT THE SOURCE: '3,4' = routine METARs + SPECIs (the set a
    forecaster actually sees on AWC/Skyvector — our default); '1' = the 5-minute
    MADIS high-frequency stream (not used in the AF workflow, but available if we
    ever want denser data)."""
    params = {
        "station": station,
        "data": "metar",
        "report_type": report_type,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
    }
    url = f"{_IEM_URL}?{urllib.parse.urlencode(params)}"

    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)            # space requests; no penalty on an isolated first call
    _last_request = time.monotonic()

    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode()

    out: list[tuple[datetime, str]] = []
    for row in csv.DictReader(io.StringIO(text)):
        raw = (row.get("metar") or "").strip()
        valid = (row.get("valid") or "").strip()
        if not raw or not valid or raw == "M":
            continue
        out.append((datetime.strptime(valid, "%Y-%m-%d %H:%M"), raw))
    return out


def load(
    station: str,
    start: datetime,
    end: datetime,
    *,
    db_path: str | None = None,
) -> dict:
    """Fetch routine METARs and SPECIs SEPARATELY (report_type 3 and 4) so each
    ob's type is known with certainty, tag them, parse, and persist. Returns a
    summary: rows fetched, parsed, newly inserted (idempotent re-runs add 0), and
    any parse errors. Persists with source='iem'."""
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

    con = store.connect(db_path) if db_path else store.connect()
    try:
        store.init_schema(con)
        inserted = sum(
            store.insert_obs(con, batch, year=y, month=m, source="iem")
            for (y, m), batch in sorted(by_month.items())
        )
    finally:
        con.close()

    parsed = sum(len(b) for b in by_month.values())
    return {
        "station": station,
        "fetched": fetched,
        "parsed": parsed,
        "inserted": inserted,
        "errors": errors,
    }
