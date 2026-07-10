"""Climatology builder -- the orchestrator that turns multi-year IEM history into
the persistent climo_* product tables.

Design: the raw history is THROWN AWAY. We ingest many years
of a station-month into a SCRATCH DuckDB, aggregate it into the climo_* tables in the
persistent DB (store.rebuild_climo, ATTACHing the scratch read-only), then discard the
scratch. Only the product persists, so the runtime read tools never anchor on stale
history. This module is a sibling to iem.py: it uses the iem + store seams and owns no
SQL and no duckdb import of its own.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from forecaster import iem, store
from forecaster.config import settings

_META_URL = "https://mesonet.agron.iastate.edu/api/1/station"
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"

# IEM rate-limits bursts: at the repo's 2s per-request throttle it 429s after ~26
# requests (Phase A Q2). Each iem.load fires two requests, so we add an extra gap
# between (year, month) loads AND back off on a 429.
_MIN_LOAD_GAP_S = 3.0
_MAX_429_RETRIES = 5


def _iem_sid(station: str) -> str:
    """Station id for the IEM METADATA endpoint. US 'K'+3 ICAOs must drop the K
    (KLSV -> LSV); the K-form 404s or silently resolves to a different station
    (KBLV -> the RAOB site). Non-US ICAOs are queried as-is (Phase A Q4)."""
    s = station.upper()
    return s[1:] if (len(s) == 4 and s.startswith("K")) else s


def _fixed_std_offset(tzname: str) -> float:
    """Fixed STANDARD-time UTC offset in hours from a tz name, no DST. DST always
    shifts the offset UP, so the minimum of a January and a July sample is standard
    time in BOTH hemispheres (Jan alone is wrong south of the equator; Phase A #6)."""
    tz = ZoneInfo(tzname)
    jan = datetime(2025, 1, 15, 12, tzinfo=tz).utcoffset()
    jul = datetime(2025, 7, 15, 12, tzinfo=tz).utcoffset()
    return min(jan, jul).total_seconds() / 3600.0


def station_meta(station: str) -> dict:
    """Fetch lat/lon/tzname for an ICAO from IEM and derive the fixed standard-time
    offset used for local-day TX/TN bucketing. Returns a dict with lat, lon, tzname,
    utc_offset_hours, and source (provenance). Raises ValueError if the id can't be
    resolved to an ASOS station (a wrong id would silently mis-bucket temperatures)."""
    sid = _iem_sid(station)
    url = f"{_META_URL}/{urllib.parse.quote(sid)}.json"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    rows = data.get("data") or []
    if not rows:
        raise ValueError(f"IEM has no station metadata for {station} (sid {sid!r})")
    row = rows[0]
    network = row.get("network") or ""
    if "ASOS" not in network:
        # The K-form can resolve to a non-ASOS station (e.g. a RAOB site); reject it.
        raise ValueError(
            f"IEM sid {sid!r} resolved to network {network!r}, not an ASOS station; "
            f"check the id for {station}"
        )
    lat, lon, tzname = row.get("latitude"), row.get("longitude"), row.get("tzname")
    source = "iem-api"
    offset: float | None = None
    if tzname:
        try:
            offset = _fixed_std_offset(tzname)
        except Exception:                                # noqa: BLE001 -- bad tz -> lon fallback
            offset = None
    if offset is None and lon is not None:
        offset = round(lon / 15.0)                       # last-resort meridian offset
        source = "lon15-fallback"
    return {
        "station": station.upper(),
        "lat": lat,
        "lon": lon,
        "tzname": tzname,
        "utc_offset_hours": offset,
        "source": source,
    }


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    """The ONE-day-buffered [start, end] for a (year, month) IEM request. IEM's day2
    is EXCLUSIVE (Phase A Q1), so end = first-of-next-month + 1 day yields exactly a
    one-day trailing buffer; start = first-of-month - 1 day is the leading buffer.
    The buffer lets local-day TX/TN at the month edges reclassify correctly."""
    start = datetime(year, month, 1) - timedelta(days=1)
    nxt = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    end = nxt + timedelta(days=1)
    return start, end


def ingest_history(
    station: str,
    months: list[int],
    *,
    start_year: int,
    end_year: int,
    scratch_db_path: str,
) -> dict:
    """Load every (year, month) in the POR into the SCRATCH DuckDB via iem.load, with
    the one-day edge buffer and 429 backoff. Returns a summary with per-year obs counts
    so a thin/absent year (KLSV 2006; OCONUS holes) is visible. Nothing here persists to
    the runtime DB -- the caller aggregates the scratch, then discards it."""
    per_year: dict[int, int] = {}
    errors: list[tuple[str, str]] = []
    total_inserted = 0
    for year in range(start_year, end_year + 1):
        for month in months:
            start, end = _month_bounds(year, month)
            for attempt in range(1, _MAX_429_RETRIES + 1):
                try:
                    summ = iem.load(station, start, end, db_path=scratch_db_path)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < _MAX_429_RETRIES:
                        time.sleep(15 * attempt)         # measured-safe backoff (Phase A Q2)
                        continue
                    raise
            per_year[year] = per_year.get(year, 0) + summ["inserted"]
            total_inserted += summ["inserted"]
            errors.extend(summ["errors"])
            time.sleep(_MIN_LOAD_GAP_S)                  # extra gap between loads
    return {
        "station": station.upper(),
        "start_year": start_year,
        "end_year": end_year,
        "months": months,
        "inserted": total_inserted,
        "per_year": per_year,
        "errors": errors,
    }


def _build_into(
    scratch: str, station: str, months: list[int], meta: dict,
    *, start_year: int, end_year: int, db_path: str | None,
) -> dict:
    """Ingest the POR into `scratch`, aggregate into the persistent climo_* tables."""
    ingest = ingest_history(
        station, months, start_year=start_year, end_year=end_year, scratch_db_path=scratch
    )
    con = store.connect(db_path) if db_path else store.connect()
    try:
        store.init_schema(con)
        store.init_climo_schema(con)
        rebuilt = store.rebuild_climo(
            con, scratch, station, months,
            utc_offset_hours=meta["utc_offset_hours"],
            lat=meta["lat"], lon=meta["lon"], tzname=meta["tzname"],
            source=meta["source"],
        )
    finally:
        con.close()
    return {"station": station, "meta": meta, "ingest": ingest,
            "rebuilt": rebuilt["months"], "scratch_db": scratch}


def build(
    station: str,
    months: list[int],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    db_path: str | None = None,
    scratch_dir: str | None = None,
) -> dict:
    """One-call climo build: resolve station metadata, ingest the POR into a scratch
    DuckDB, aggregate into the persistent climo_* tables, discard the scratch. Returns
    a combined summary (metadata + ingest per-year counts + the rebuilt month rows).
    The raw history never touches the runtime obs table.

    scratch_dir: normally None -> a TemporaryDirectory is used and deleted on exit. Pass
    a directory to RETAIN the scratch obs DB (its path comes back as `scratch_db`) so a
    caller can recompute a verification from the same inputs (build_climo.py --check)."""
    station = station.upper()
    start_year = start_year if start_year is not None else settings.climo_start_year
    end_year = end_year if end_year is not None else settings.climo_end_year
    meta = station_meta(station)
    kw = {"start_year": start_year, "end_year": end_year, "db_path": db_path}

    if scratch_dir is None:
        with TemporaryDirectory(prefix="climo_scratch_") as tmp:
            result = _build_into(str(Path(tmp) / "obs_history.duckdb"), station, months, meta, **kw)
            result.pop("scratch_db", None)                       # discarded with the tempdir
            return result
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    return _build_into(str(Path(scratch_dir) / "obs_history.duckdb"), station, months, meta, **kw)
