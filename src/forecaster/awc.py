"""AWC (aviationweather.gov) live observation + forecast client.

Pulls CURRENT METARs and TAFs straight from the official Aviation Weather Center
data API -- crucially including the military aerodromes that IEM does NOT serve.

Uses format=json so every report arrives with an AUTHORITATIVE epoch/ISO
timestamp (and, for METARs, the METAR/SPECI type) -- the same (utc_time, raw)
contract iem.fetch produces, so the downstream metar/taf + store seams are
unchanged. The raw TAF also comes back as ONE line in json (the raw text format
wraps across several), ready to hand straight to taf.parse().

This is a data-source client (seam like iem.py): it owns no SQL and no DuckDB
import. Parsing and persistence stay in the metar/taf + store seams.
"""

import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

from forecaster import store
from forecaster.metar import MetarObs, parse

_AWC_URL = "https://aviationweather.gov/api/data/{product}"

# Be polite to a free public API: space requests so no caller (a multi-station
# loop, say) can fire back-to-back. Module-level on purpose, like iem.py.
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0


def _get(product: str, params: dict) -> list[dict]:
    """GET one AWC data product as a JSON list, spacing requests politely."""
    url = f"{_AWC_URL.format(product=product)}?{urllib.parse.urlencode(params)}"

    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    _last_request = time.monotonic()

    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _ids(stations: str | list[str]) -> str:
    return stations if isinstance(stations, str) else ",".join(stations)


def _from_epoch(seconds: float) -> datetime:
    """Epoch seconds -> naive UTC datetime (the store's tz contract)."""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def _from_iso(s: str) -> datetime:
    """ISO-8601 'Z' string -> naive UTC datetime (already UTC, just drop tzinfo)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def station_latlon(station: str) -> tuple[float, float]:
    """(lat, lon) for an ICAO from AWC's station-info product. Keyed on the EXACT id
    (no K-stripping), so it resolves major airports and OCONUS sites that IEM's ASOS
    metadata lookup can miss (e.g. KMSP collides with a TDWR sid there). Raises
    ValueError if the id is unknown."""
    icao = station.upper()
    for r in _get("stationinfo", {"ids": icao, "format": "json"}) or []:
        lat, lon = r.get("lat"), r.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    raise ValueError(f"AWC has no station info for {icao}")


def fetch_metar(
    stations: str | list[str],
    *,
    hours: float | None = None,
) -> list[tuple[datetime, str, str | None]]:
    """Live METAR(s) for one or more ICAO ids. Returns (obs_time_utc, raw,
    report_type) tuples. report_type is the API's metarType (METAR/SPECI) -- no
    separate fetch like IEM. `hours` optionally pulls the recent back-window
    instead of just the single latest ob."""
    params: dict = {"ids": _ids(stations), "format": "json"}
    if hours is not None:
        params["hours"] = hours

    out: list[tuple[datetime, str, str | None]] = []
    for r in _get("metar", params):
        raw = (r.get("rawOb") or "").strip()
        if not raw:
            continue
        out.append((_from_epoch(r["obsTime"]), raw, r.get("metarType")))
    return out


def fetch_taf(stations: str | list[str]) -> list[tuple[datetime, str]]:
    """Live TAF(s) for one or more ICAO ids. Returns (issue_time_utc, raw_taf)
    tuples; raw_taf is a single line, ready for taf.parse()."""
    out: list[tuple[datetime, str]] = []
    for r in _get("taf", {"ids": _ids(stations), "format": "json"}):
        raw = (r.get("rawTAF") or "").strip()
        if not raw:
            continue
        out.append((_from_iso(r["issueTime"]), raw))
    return out


def load_metar(
    station: str,
    *,
    hours: float | None = None,
    db_path: str | None = None,
    before: datetime | None = None,
) -> dict:
    """Fetch live METAR(s) from AWC, parse, tag report_type, and persist via the
    store seam (source='awc'). The orchestrator half that fetch_metar lacks -- it
    owns no SQL of its own, mirroring iem.load. Groups by (year, month) so each
    insert lands in one clean period (the obs carries day+time; the authoritative
    year/month come from the API timestamp). Idempotent: re-runs, and overlap with
    IEM-loaded data, add 0 rows via the (station, obs_time) primary key.

    `hours` widens the pull to a recent back-window; omit it for just the latest
    ob. `before` (a UTC datetime) drops any ob at or after that time -- a point-in-time
    snapshot for forecast benchmarking, so a store built with it holds only what was
    observed BEFORE the cutoff (no peeking past the valid time). Returns a summary."""
    by_month: dict[tuple[int, int], list[MetarObs]] = defaultdict(list)
    errors: list[tuple[str, str]] = []
    fetched = skipped = 0
    before_naive = before.replace(tzinfo=None) if before and before.tzinfo else before
    for ts, raw, rtype in fetch_metar(station, hours=hours):
        if before_naive is not None and ts.replace(tzinfo=None) >= before_naive:
            skipped += 1
            continue
        fetched += 1
        try:
            obs = parse(raw)
        except Exception as e:                       # noqa: BLE001 -- log & skip a bad line
            errors.append((raw, str(e)))
            continue
        # AWC's rawOb keeps the METAR/SPECI keyword, so parse() usually sets this;
        # fall back to the API's metarType if a line ever omits it.
        obs.report_type = obs.report_type or rtype
        by_month[(ts.year, ts.month)].append(obs)

    con = store.connect(db_path) if db_path else store.connect()
    try:
        store.init_schema(con)
        inserted = sum(
            store.insert_obs(con, batch, year=y, month=m, source="awc")
            for (y, m), batch in sorted(by_month.items())
        )
    finally:
        con.close()

    parsed = sum(len(b) for b in by_month.values())
    return {
        "station": station,
        "fetched": fetched,
        "skipped_after_cutoff": skipped,
        "parsed": parsed,
        "inserted": inserted,
        "errors": errors,
    }
