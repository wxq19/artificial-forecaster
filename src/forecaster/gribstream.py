"""GRIBStream point-forecast client -- live/archived model time-series fetch seam.

Sibling to fcstsounding.py / soundings.py / awc.py: a network data-source client. Fetches a
model forecast time series for a point (or points) from the GRIBStream API and returns plain
rows. No matplotlib, no SQL. Two things set it apart from the BUFKIT seam (fcstsounding.py):
  - ARBITRARY lat/lon -- it extracts from the underlying GRIB, not a fixed ~2100-station
    list, so an off-BUFKIT site (e.g. KBAB) needs no proxy station.
  - Every row carries `forecasted_at` -- the model RUN reference time -- so the agent can
    reason about WHICH cycle a value came from, and `asOf` reproduces a leakage-safe
    point-in-time view (only forecasts issued at/before the cutoff).

The API key + base URL flow through config.py (the seam, like llm.py); nothing here
hardcodes either. Auth is a Bearer token. Credits bill as
    returned_valid_times * variables * ceil(coordinates / 500)
so a smoke test of one time * one variable * one point costs 1 credit -- keep it that way
while the free pool is small (see scripts/test_gribstream.py).
"""

import csv
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import settings

# Models exposed by the API (per gribstream.com/docs). Each is a path segment:
# POST {base_url}/{model}/timeseries. gfs+hrrr overlap the BUFKIT set; nbm is the govt
# multi-model BLEND (a consensus BASELINE, not raw-model spread). ifsoper (ECMWF IFS) is
# available too -- deferred to keep credits down; add it here when we wire IFS in.
MODELS = ("gfs", "hrrr", "nbm")

# Local pull ARCHIVE: every response is cached to disk keyed by a stable hash of the
# request, so a repeat pull of the same slice (e.g. the 6 matrix cells at one station/
# cycle, or a re-run) is FREE -- GRIBStream bills per returned row, and this is how we
# avoid paying twice for identical data. Shared across agents via the filesystem.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "gribstream"
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"

# Response columns the API always adds; everything else is a requested variable. `member`
# is the ensemble id (present even for a deterministic run) -- NOT billed as a variable.
_TS_COLS = ("forecasted_at", "forecasted_time")
_META_COLS = ("lat", "lon", "name", "member")


@dataclass
class Var:
    """One requested weather parameter: a GRIB `name` at a `level`, with an optional
    `alias` that becomes its response column header. Mirrors the API variable object."""
    name: str
    level: str
    alias: str = ""

    def as_dict(self) -> dict:
        d = {"name": self.name, "level": self.level}
        if self.alias:
            d["alias"] = self.alias
        return d


@dataclass
class TimeSeries:
    """A parsed GRIBStream time series. `rows` are dicts keyed by column: the two
    timestamps as naive-UTC datetimes, lat/lon as floats, name as str, and each variable
    (by alias or GRIB name) as a float. `runs` is the distinct set of model run reference
    times (`forecasted_at`) the response drew from -- the provenance the agent reasons over.
    """
    model: str
    url: str
    columns: list[str]
    rows: list[dict]
    runs: list[datetime] = field(default_factory=list)
    cached: bool = False           # served from the local archive (no API charge)

    @property
    def credits(self) -> int:
        """Best-effort credit estimate for what came back: valid_times * variables * 1
        (single-point). Charged on RETURNED rows, so a 0-row window costs nothing."""
        n_vars = len(self.columns) - len(_TS_COLS) - len(_META_COLS)
        return len(self.rows) * max(n_vars, 0)

    @property
    def charged(self) -> int:
        """Credits actually billed: 0 on an archive hit, else the gross estimate."""
        return 0 if self.cached else self.credits


def endpoint_url(model: str) -> str:
    """The timeseries endpoint for a model (provenance without fetching)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    return f"{settings.gribstream_base_url}/{model}/timeseries"


def _iso(t: datetime) -> str:
    """A naive-or-aware datetime -> the API's ISO-8601 Z form. Naive is treated as UTC
    (the naive-UTC seam contract used throughout the repo)."""
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(s: str) -> datetime:
    """API ISO timestamp (e.g. 2024-09-10T02:00:00Z) -> naive UTC datetime."""
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _cache_file(model: str, body: dict) -> Path:
    """Archive path for a request: a stable hash of (model, body). Sorted-key JSON so
    logically-identical requests map to the same file regardless of dict order."""
    blob = json.dumps({"model": model, "body": body}, sort_keys=True)
    return _CACHE_DIR / f"{model}_{hashlib.sha256(blob.encode()).hexdigest()[:16]}.csv"


def _cache_write(path: Path, text: str) -> None:
    """Atomic write (temp + rename) so a concurrent matrix cell never reads a half file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _post(model: str, body: dict) -> str:
    """POST a timeseries request; return the raw CSV text. Raises ValueError with the
    server's message on an HTTP error (401 bad key, 402 out of credits, 400 bad request)
    so the caller surfaces it as feedback, not a stack trace."""
    if not settings.gribstream_api_key:
        raise ValueError("GRIBSTREAM_API_KEY not set in .env")
    url = endpoint_url(model)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {settings.gribstream_api_key}",
            "Content-Type": "application/json",
            "User-Agent": _UA,
        },
    )
    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip()
        raise ValueError(f"GRIBStream {e.code}: {detail or e.reason}") from None


def _parse_csv(text: str) -> tuple[list[str], list[dict]]:
    """Parse the CSV response into (columns, rows). Timestamps become naive-UTC datetimes,
    lat/lon become floats, and each variable column is coerced to float (blank -> None)."""
    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    rows: list[dict] = []
    for raw in reader:
        row: dict = {}
        for col in columns:
            val = raw.get(col)
            if col in _TS_COLS:
                row[col] = _parse_dt(val) if val else None
            elif col in ("lat", "lon"):
                row[col] = float(val) if val else None
            elif col == "name":
                row[col] = val
            else:
                # A masked field (e.g. HGT at 'cloud ceiling' with no deck) comes back as
                # NaN -- normalize any non-finite value to None so callers do simple None
                # checks instead of tripping over float('nan').
                f = float(val) if val not in (None, "") else None
                row[col] = f if f is None or math.isfinite(f) else None
        rows.append(row)
    return columns, rows


def fetch_timeseries(
    model: str,
    lat: float,
    lon: float,
    variables: list[Var],
    *,
    from_time: datetime | None = None,
    until_time: datetime | None = None,
    times: list[datetime] | None = None,
    name: str = "",
    as_of: datetime | None = None,
    min_lead: str | None = None,
    max_lead: str | None = None,
    use_cache: bool = True,
) -> TimeSeries:
    """Fetch a single-point forecast time series.

    Give EITHER a window (`from_time`, `until_time` -- returns every model step inside it)
    OR an explicit `times` list (the API's timesList -- returns only those valid times, so a
    tool can subsample, e.g. 2-hourly across 30 h, without paying for every hourly step).

    `as_of` (a model run cutoff) yields a leakage-safe point-in-time view: only forecasts
    issued at/before it are considered -- pass the TAF issue time for historical replay.
    `min_lead`/`max_lead` are API duration strings ('1h', '48h'). Every returned row carries
    `forecasted_at` (the run it came from); `.runs` collects the distinct set.

    `use_cache` (default True) serves an identical prior request from the local archive for
    0 credits. Credits = returned_valid_times * len(variables) for a single point -- keep
    windows narrow while the free pool is small.
    """
    if not variables:
        raise ValueError("at least one variable is required")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    body: dict = {
        "coordinates": [{"lat": lat, "lon": lon, "name": name}],
        "variables": [v.as_dict() for v in variables],
    }
    if times is not None:
        body["timesList"] = [_iso(t) for t in times]
    elif from_time is not None and until_time is not None:
        body["fromTime"] = _iso(from_time)
        body["untilTime"] = _iso(until_time)
    else:
        raise ValueError("give either times=... or both from_time and until_time")
    if as_of is not None:
        body["asOf"] = _iso(as_of)
    if min_lead is not None:
        body["minLeadTime"] = min_lead
    if max_lead is not None:
        body["maxLeadTime"] = max_lead

    cache_file = _cache_file(model, body)
    cached = use_cache and cache_file.exists()
    if cached:
        text = cache_file.read_text("utf-8")
    else:
        text = _post(model, body)
        if use_cache:
            _cache_write(cache_file, text)

    columns, rows = _parse_csv(text)
    runs = sorted({r["forecasted_at"] for r in rows if r.get("forecasted_at")})
    return TimeSeries(model=model, url=endpoint_url(model), columns=columns,
                      rows=rows, runs=runs, cached=cached)
