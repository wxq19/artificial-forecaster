"""Synoptic map image client -- live surface/upper-air chart fetch seam.

Sibling to soundings.py / awc.py: a network data-source client that fetches
PRE-RENDERED forecaster charts (surface analysis + progs, upper-air analysis, NWP
forecast panels) from public providers and returns raw bytes. No matplotlib
(charts.py stays the only matplotlib file), no SQL. Feeding the model the same maps
a human forecaster reads keeps the comparison honest.

Four provider families, one CATALOG entry per chart:
  - WPC (wpc.ncep.noaa.gov)      -- CONUS surface analysis + Day1/Day2 progs (GIF).
  - OPC (ocean.weather.gov)      -- Atlantic + Pacific oceanic surface analysis (PNG),
                                    for OCONUS/maritime coverage WPC's CONUS view lacks.
  - SPC mesoanalysis (spc.noaa.gov/exper/mesoanalysis) -- hourly RAP ANALYSIS at MSLP
                                    /850/700/500/300 mb, National sector s19 (GIF).
  - TropicalTidbits (tropicaltidbits.com) -- GFS FORECAST panels (PNG). Third-party and
                                    hotlink-gated: requires a Referer header, and the URL
                                    scheme can change (fragile -- watched, not trusted).

Analysis charts are "now" (no time arg). Forecast charts are the GFS run: TT samples
GFS 6-hourly to f384; frame = fhr//6 + 1, and `latest_gfs_run()` picks the freshest
posted cycle. Air-gap note (SuperCloud has no internet): fetch is cache-aware
(opt-in) so a pre-staged image replays offline; live-first while prototyping.
"""

import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Anchor the cache at the repo root (like config.py), not the cwd.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "maps"

# Space requests politely across ALL hosts (module-level, like iem/awc/soundings).
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0

_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"
# Providers that gate direct image access behind a same-site Referer (anti-hotlink).
_REFERER = {"tt": "https://www.tropicaltidbits.com/"}

# GFS posts ~3.5-5h after each 00/06/12/18Z cycle; wait this long before trusting one.
_GFS_POST_LAG_H = 5


@dataclass(frozen=True)
class ChartSpec:
    """One catalogued chart. `code` is the review-manifest id (A1..C4); `name` is the
    semantic key the tool/code use; `params` holds the source-specific URL bits."""
    code: str
    name: str
    label: str
    kind: str        # "analysis" | "forecast"
    source: str      # "wpc" | "opc" | "spc" | "tt"
    ext: str         # "gif" | "png"
    params: dict = field(default_factory=dict)


# The approved set (see CLAUDE.md review manifest). Keyed by semantic name.
CATALOG: dict[str, ChartSpec] = {
    c.name: c for c in [
        # --- A: surface (WPC current + Day1/2 progs; OPC oceanic analysis) ---
        ChartSpec("A1", "surface_analysis", "Surface analysis (fronts/isobars/pressure)",
                  "analysis", "wpc", "gif", {"path": "sfc/namussfcwbg.gif"}),
        ChartSpec("A2", "surface_fcst_day1", "Surface forecast -- Day 1 prog",
                  "forecast", "wpc", "gif", {"path": "basicwx/91fndfd.gif"}),
        ChartSpec("A3", "surface_fcst_day2", "Surface forecast -- Day 2 prog",
                  "forecast", "wpc", "gif", {"path": "basicwx/92fndfd.gif"}),
        ChartSpec("A4", "ocean_sfc_atlantic", "Oceanic surface analysis -- Atlantic",
                  "analysis", "opc", "png", {"path": "A_sfc_full_ocean_color.png"}),
        ChartSpec("A5", "ocean_sfc_pacific", "Oceanic surface analysis -- Pacific",
                  "analysis", "opc", "png", {"path": "P_sfc_full_ocean_color.png"}),
        # --- B: upper-air ANALYSIS (SPC mesoanalysis, National sector) ---
        ChartSpec("B1", "meso_mslp", "MSLP / surface mesoanalysis",
                  "analysis", "spc", "gif", {"prod": "pmsl"}),
        ChartSpec("B2", "meso_850mb", "850 mb -- low-level temp/moisture, LLJ",
                  "analysis", "spc", "gif", {"prod": "850mb"}),
        ChartSpec("B3", "meso_700mb", "700 mb -- mid-level moisture, vertical velocity",
                  "analysis", "spc", "gif", {"prod": "700mb"}),
        ChartSpec("B4", "meso_500mb", "500 mb -- steering flow, heights/vorticity",
                  "analysis", "spc", "gif", {"prod": "500mb"}),
        ChartSpec("B5", "meso_300mb", "300 mb -- jet stream / isotachs",
                  "analysis", "spc", "gif", {"prod": "300mb"}),
        # --- C: upper-air/synoptic FORECAST (TropicalTidbits GFS) ---
        ChartSpec("C1", "gfs_500mb", "500 mb height/vorticity forecast (GFS)",
                  "forecast", "tt", "png", {"field": "z500_vort"}),
        ChartSpec("C2", "gfs_250mb", "250 mb jet/wind forecast (GFS)",
                  "forecast", "tt", "png", {"field": "uv250"}),
        ChartSpec("C3", "gfs_mslp_precip", "MSLP + precipitation forecast (GFS)",
                  "forecast", "tt", "png", {"field": "mslp_pcpn_frzn", "f0": 6}),
        ChartSpec("C4", "gfs_850mb_temp", "850 mb temperature forecast (GFS)",
                  "forecast", "tt", "png", {"field": "T850"}),
    ]
}

# TT samples GFS every 6 hours; a forecast hour must land on that grid.
GFS_STEP_H = 6
GFS_MAX_FHR = 384


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def latest_gfs_run(now: datetime | None = None) -> datetime:
    """Freshest GFS cycle (00/06/12/18Z) old enough to have posted -- naive UTC. Back
    off _GFS_POST_LAG_H hours from now, then snap down to the 6-hourly cycle grid."""
    t = (now or _utcnow()) - timedelta(hours=_GFS_POST_LAG_H)
    return t.replace(hour=(t.hour // GFS_STEP_H) * GFS_STEP_H, minute=0, second=0, microsecond=0)


def _url(spec: ChartSpec, *, fhr: int, run: datetime, sector: str) -> str:
    if spec.source == "wpc":
        return f"https://www.wpc.ncep.noaa.gov/{spec.params['path']}"
    if spec.source == "opc":
        return f"https://ocean.weather.gov/{spec.params['path']}"
    if spec.source == "spc":
        prod = spec.params["prod"]
        return f"https://www.spc.noaa.gov/exper/mesoanalysis/{sector}/{prod}/{prod}.gif"
    if spec.source == "tt":
        # frame 1 is the field's FIRST forecast hour: f000 for instantaneous fields, but
        # f006 for the 6h-AVERAGED precip field (it has no f000 frame) -- carried per
        # chart as f0. Getting this wrong desyncs a forecast panel from the others by 6h.
        f0 = spec.params.get("f0", 0)
        frame = (fhr - f0) // GFS_STEP_H + 1
        return (f"https://www.tropicaltidbits.com/analysis/models/gfs/"
                f"{run:%Y%m%d%H}/gfs_{spec.params['field']}_us_{frame}.png")
    raise ValueError(f"unknown source {spec.source!r}")


def map_url(name: str, *, fhr: int = 0, run: datetime | None = None, sector: str = "s19") -> str:
    """The exact provider URL for a catalogued chart. Exposed so a caller can cite
    provenance without fetching. `fhr`/`run` apply only to forecast (TT) charts."""
    spec = CATALOG[name]
    if spec.source == "tt" and run is None:
        run = latest_gfs_run()
    return _url(spec, fhr=fhr, run=run, sector=sector)


def _get(url: str, *, referer: str | None = None) -> bytes:
    """GET raw bytes, spacing requests politely; send a Referer if the host needs it."""
    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    headers = {"User-Agent": _UA}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def cache_path(name: str, *, fhr: int = 0, run: datetime | None = None) -> Path:
    """Where fetch_map(..., use_cache=True) stores this image. Forecast charts key on
    (run, fhr) -- deterministic; analysis charts key on the fetch hour, since 'now'
    isn't in the URL."""
    spec = CATALOG[name]
    if spec.source == "tt":
        run = run or latest_gfs_run()
        tag = f"{run:%Y%m%d%H}_f{fhr:03d}"
    else:
        tag = f"{_utcnow():%Y%m%d%H}"
    return _CACHE_DIR / f"{spec.name}_{tag}.{spec.ext}"


def fetch_map(
    name: str,
    *,
    fhr: int = 0,
    run: datetime | None = None,
    sector: str = "s19",
    use_cache: bool = False,
) -> bytes:
    """Fetch one catalogued chart and return raw bytes. `name` is a CATALOG key (see
    CATALOG / the review manifest). For forecast (TT) charts, `fhr` is the GFS forecast
    hour (multiple of 6, 0..384) and `run` the cycle (default: latest posted); ignored
    for analysis charts. With use_cache, a hit replays from disk and a miss is saved."""
    spec = CATALOG[name]
    if spec.source == "tt":
        f0 = spec.params.get("f0", 0)
        if fhr % GFS_STEP_H or not f0 <= fhr <= GFS_MAX_FHR:
            raise ValueError(
                f"fhr must be a multiple of {GFS_STEP_H} in {f0}..{GFS_MAX_FHR} for {name}, got {fhr}")
        run = run or latest_gfs_run()
    url = _url(spec, fhr=fhr, run=run, sector=sector)
    cache_file = cache_path(name, fhr=fhr, run=run)

    if use_cache and cache_file.exists():
        return cache_file.read_bytes()
    data = _get(url, referer=_REFERER.get(spec.source))
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
    return data
