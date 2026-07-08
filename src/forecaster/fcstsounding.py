"""Forecast sounding client -- model BUFKIT fetch + parse seam.

Sibling to soundings.py / wxmaps.py / awc.py: a network data-source client. Fetches a
model BUFKIT file (ISU mtarchive) and parses ONE forecast-hour profile plus the block's
stability indices into a plain FcstProfile. No matplotlib (charts.py renders the skew-T)
and no SQL. BUFKIT is TEXT, so unlike the observed soundings we render the plot
ourselves -- keeping the whole path in the uv/PyPI tier (MetPy is PyPI; no conda).

Coverage: dense over North America (US/Canada/Alaska/Hawaii/Mexico); OCONUS is sparse
and reachable only via GFS (the mesoscale models are North-America-only). A missing
station 404s -> a clean "not available" error the caller surfaces as feedback rather
than guessing a substitute.
"""

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ARCHIVE = ("https://mtarchive.geol.iastate.edu/{run:%Y/%m/%d}/bufkit/{run:%H}/"
            "{model}/{prefix}_{station}.buf")

# model -> (BUFKIT filename prefix, cycle hours, typical post lag hours). Only GFS has
# any OCONUS coverage; the others are North-America-only.
_MODELS: dict[str, tuple[str, int, int]] = {
    "gfs":    ("gfs3",   6, 5),
    "nam":    ("nam",    6, 4),
    "nam4km": ("nam4km", 6, 4),
    "rap":    ("rap",    1, 2),
    "hrrr":   ("hrrr",   1, 2),
}
MODELS = tuple(_MODELS)                       # public: the enum the tool advertises

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "fcstsoundings"
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"

# BUFKIT SNPARM profile order: PRES TMPC TMWC DWPC THTE DRCT SKNT OMEG HGHT (printed 8
# per line + HGHT on the next line -> 9 values per level).
_COLS = {"pres": 0, "tmpc": 1, "dwpc": 3, "drct": 5, "sknt": 6, "hght": 8}
# STNPRM stability/moisture indices we keep for the plot's corner box.
_IDX = ("CAPE", "CINS", "LIFT", "SHOW", "KINX", "TOTL", "PWAT", "LCLP")
_FLOAT = re.compile(r"-?\d+\.\d+")
_FILL = -9000.0                               # BUFKIT missing sentinel is -9999


@dataclass
class FcstProfile:
    """A parsed model forecast sounding for one valid time. Arrays are surface-first,
    fill rows removed. Plain lists/floats -- no units, no matplotlib (charts.py owns
    plotting)."""
    station: str
    model: str
    run: datetime                 # cycle, naive UTC
    fhr: int
    valid: str                    # BUFKIT valid string, e.g. '260708/1200'
    lat: float
    lon: float
    elev_m: float
    pres: list[float]             # hPa
    tmpc: list[float]             # C
    dwpc: list[float]             # C
    drct: list[float]             # deg
    sknt: list[float]             # kt
    hght: list[float]             # m
    indices: dict                 # CAPE/CINS/LIFT/... from the block
    url: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def latest_run(model: str = "gfs", now: datetime | None = None) -> datetime:
    """Freshest cycle old enough to have posted (naive UTC): back off the model's post
    lag, then snap down to its cycle grid."""
    _, cycle, lag = _MODELS[model]
    t = (now or _utcnow()) - timedelta(hours=lag)
    return t.replace(hour=(t.hour // cycle) * cycle, minute=0, second=0, microsecond=0)


def bufkit_url(station: str, model: str = "gfs", run: datetime | None = None) -> str:
    """The exact mtarchive URL for a station/model/run (provenance without fetching)."""
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    prefix = _MODELS[model][0]
    run = run or latest_run(model)
    return _ARCHIVE.format(run=run, model=model, prefix=prefix, station=station.lower())


def fetch_raw(station: str, model: str = "gfs", run: datetime | None = None,
              *, use_cache: bool = False) -> str:
    """Fetch a BUFKIT file as text. Raises ValueError('not available') on a 404 so an
    OCONUS gap reads as feedback, not a stack trace. Opt-in disk cache (air-gap/repro)."""
    run = run or latest_run(model)
    url = bufkit_url(station, model, run)
    cache_file = _CACHE_DIR / f"{model}_{station.lower()}_{run:%Y%m%d%H}.buf"
    if use_cache and cache_file.exists():
        return cache_file.read_text("latin-1")

    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("latin-1")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"{station.upper()} not available for model {model} at "
                f"{run:%Y-%m-%dT%H:%MZ} (BUFKIT 404); it is likely outside coverage"
            ) from None
        raise
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="latin-1")
    return text


def _num(pattern: str, blk: str, default: float = float("nan")) -> float:
    m = re.search(pattern, blk)
    return float(m.group(1)) if m else default


def parse(text: str, fhr: int, *, station: str, model: str, run: datetime, url: str) -> FcstProfile:
    """Parse the profile + indices for forecast hour `fhr`. Raises ValueError listing the
    available hours if `fhr` isn't in the file."""
    available: list[int] = []
    for blk in re.split(r"(?m)^STID = ", text)[1:]:
        m = re.search(r"STIM = (\d+)", blk)
        if not m:
            continue
        stim = int(m.group(1))
        available.append(stim)
        if stim != fhr:
            continue
        valid = re.search(r"TIME = (\S+)", blk).group(1)
        idx = {n: v for n in _IDX if (v := _num(rf"{n} = (-?\d+\.\d+)", blk)) == v}  # drop NaN
        data = blk[blk.find("HGHT") + 4:]
        vals: list[float] = []
        for ln in data.splitlines():
            toks = ln.split()
            if toks and all(_FLOAT.fullmatch(t) for t in toks):
                vals += [float(t) for t in toks]
            elif toks and vals:            # first non-numeric line after data -> profile ends
                break
        levels = [vals[i:i + 9] for i in range(0, len(vals) - 8, 9)]
        # drop fill rows (BUFKIT tops the profile with Td/T = -9999 in the near-vacuum levels)
        levels = [lv for lv in levels if lv[_COLS["dwpc"]] > _FILL and lv[_COLS["tmpc"]] > _FILL]
        cols = {name: [lv[i] for lv in levels] for name, i in _COLS.items()}
        return FcstProfile(
            station=station.upper(), model=model, run=run, fhr=fhr, valid=valid,
            lat=_num(r"SLAT = (-?\d+\.\d+)", blk), lon=_num(r"SLON = (-?\d+\.\d+)", blk),
            elev_m=_num(r"SELV = (-?\d+\.\d+)", blk), indices=idx, url=url, **cols,
        )
    hrs = sorted(set(available))
    raise ValueError(
        f"forecast hour f{fhr:03d} not in file; available f{hrs[0]:03d}..f{hrs[-1]:03d} "
        f"(e.g. {', '.join(f'{h}' for h in hrs[:12])}{'...' if len(hrs) > 12 else ''})"
    )


def fetch_profile(station: str, *, model: str = "gfs", fhr: int = 12,
                  run: datetime | None = None, use_cache: bool = False) -> FcstProfile:
    """Fetch + parse a model forecast sounding: one profile at forecast hour `fhr`.
    `model` defaults to GFS (the only model with OCONUS reach). Raises ValueError on a
    missing station (404) or a missing forecast hour -- both correctable feedback."""
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    run = run or latest_run(model)
    text = fetch_raw(station, model, run, use_cache=use_cache)
    return parse(text, fhr, station=station, model=model, run=run,
                 url=bufkit_url(station, model, run))


# --- Point forecast (BUFKIT SURFACE section) -----------------------------------------
# Our key -> BUFKIT surface column. RAW fields only (no derived quantities); wind stays
# as the u/v components here -- the presentation layer turns it into dir/speed.
_SFC = {"t2m_c": "T2MS", "td2m_c": "TD2M", "uwnd_ms": "UWND", "vwnd_ms": "VWND",
        "mslp_hpa": "PMSL", "lcld": "LCLD", "mcld": "MCLD", "hcld": "HCLD", "p01_mm": "P01M"}


@dataclass
class PointForecast:
    """A parsed model point forecast: the BUFKIT surface time series, one dict per
    forecast hour (valid datetime + RAW surface fields, see `_SFC`). No matplotlib, no
    derived quantities -- the tool layer formats + represents (e.g. wind dir/speed)."""
    station: str
    model: str
    run: datetime
    url: str
    rows: list[dict]


def _parse_surface(text: str) -> list[dict]:
    """Parse the BUFKIT surface block into per-hour rows. The header wraps over several
    lines; data begins at the first line whose first token is the numeric station id."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("STN YYMMDD"))
    cols: list[str] = []
    j = start
    while j < len(lines):
        toks = lines[j].split()
        if toks and toks[0].isdigit():        # first data row (station number) -> header done
            break
        cols += toks
        j += 1
    idx = {c: i for i, c in enumerate(cols)}
    nums: list[str] = []
    for ln in lines[j:]:
        nums += ln.split()
    n = len(cols)
    out: list[dict] = []
    for k in range(0, len(nums) - n + 1, n):
        r = nums[k:k + n]
        ymd, hm = r[idx["YYMMDD/HHMM"]].split("/")   # e.g. 260707/1800
        row = {key: float(r[idx[b]]) for key, b in _SFC.items()}
        row["valid"] = datetime(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]),
                                int(hm[:2]), int(hm[2:4]))
        out.append(row)
    return out


def fetch_point(station: str, *, model: str = "gfs", run: datetime | None = None,
                use_cache: bool = False) -> PointForecast:
    """Fetch + parse a model POINT forecast: the surface time series over all forecast
    hours. Same fetch as fetch_profile (one `.buf` carries both). Raises ValueError on a
    missing station (404). `model` defaults to GFS (the only one with OCONUS reach)."""
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    run = run or latest_run(model)
    text = fetch_raw(station, model, run, use_cache=use_cache)
    return PointForecast(station.upper(), model, run, bufkit_url(station, model, run),
                         _parse_surface(text))
