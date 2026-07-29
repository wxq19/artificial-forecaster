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

# Fallback when a TT (third-party, hotlink-gated, URL-fragile) forecast panel fails: the
# closest SPC mesoanalysis ANALYSIS chart at the same level. NOTE this is CURRENT analysis,
# not the forecast hour asked for -- the tool receipt must say so explicitly (T8).
TT_TO_SPC_MESO: dict[str, str] = {
    "gfs_500mb": "meso_500mb",          # 500mb hgt/vort -> SPC 500mb
    "gfs_250mb": "meso_300mb",          # 250mb jet -> SPC 300mb (nearest level)
    "gfs_mslp_precip": "meso_mslp",     # MSLP + precip -> SPC MSLP
    "gfs_850mb_temp": "meso_850mb",     # 850mb temp -> SPC 850mb
}

# TT samples GFS every 6 hours; a forecast hour must land on that grid.
GFS_STEP_H = 6
GFS_MAX_FHR = 384

# --- Which charts are HONEST for a given station -------------------------------------
# The A set (WPC surface analysis/progs) and the B set (SPC mesoanalysis) are US products.
# Serving them for a Patagonian or Japanese station is the RJTY-radar failure repeated: a
# confidently-labelled picture of the wrong continent, on the tool chosen in 67.9% of
# round-1 runs. A model cannot tell that a chart is of the wrong hemisphere -- it has no
# way to check -- so the chart must be WITHHELD, not left to fall back.
#
# The C set (TropicalTidbits GFS panels) is domain-parameterised, so it travels widely.
# TT advertises 36 domain codes; they are SHORT CODES, not words -- Europe is `eu`, not
# `europe`, and the Middle East is `me`. An earlier pass here guessed the long names, got
# 404s, and wrongly concluded Europe had no coverage. The codes are enumerated from the
# region menu on the model page (`?model=gfs&region=<code>&pkg=...`); do NOT guess them.
# Full list 2026-07-28: 06 07 12 ak asia atl aus cpac ea eatl epac eu eus global india io
# me nafr namer ncus neus nhem npac nwus safr samer scus secan sepac seus swpac swus us
# watl wpac wus.
#
# AVAILABILITY IS PER (DOMAIN, FIELD), not per domain -- verified live 2026-07-28 by
# fetching each catalogued field against each domain on the 12Z run, and by enumerating
# the package menu per region. A 404 on our field name does NOT mean the domain is
# sparse: wpac and cpac serve 23-24 packages, they simply NAME the equivalents
# differently (see _TT_FIELD_OVERRIDES). Read the menu before ruling anything out.
US_CHARTS = ("surface_analysis", "surface_fcst_day1", "surface_fcst_day2",
             "meso_mslp", "meso_850mb", "meso_700mb", "meso_500mb", "meso_300mb")
OCEAN_ATLANTIC, OCEAN_PACIFIC = "ocean_sfc_atlantic", "ocean_sfc_pacific"
TT_CHARTS = ("gfs_500mb", "gfs_250mb", "gfs_mslp_precip", "gfs_850mb_temp")
# The oceanic domains carry 23-24 packages -- they are NOT sparse, they NAME things
# differently: the jet panel is `uv200` (200 mb) not `uv250`, and the precip panel is
# `mslp_pcpn` (no frozen-precip legend, which a tropical domain has no use for) not
# `mslp_pcpn_frzn`. Enumerated from the package menu per region, then fetched to confirm.
# `T850` genuinely has no oceanic equivalent (`z850_vort`/`mslp_uv850` are height+wind,
# a DIFFERENT field), so that one chart is withheld rather than silently substituted.
TT_OCEANIC = TT_CHARTS      # all four, via the substitutions below


@dataclass(frozen=True)
class TtVariant:
    """A per-domain stand-in for a catalogued chart: the field code that domain serves,
    and what that panel ACTUALLY shows. The label matters as much as the field -- a
    substitute served under the catalogue's own label would tell the model it is looking
    at a temperature chart when it is looking at height and wind."""
    field: str
    label: str


# chart name -> {domain: variant}. Absent = the CATALOG field works on that domain.
# Policy (owner, 2026-07-28): where there is no 1-for-1 match, serve the BEST AVAILABLE
# equivalent rather than withholding -- do not handicap the agent when a usable panel is
# sitting right there. The honesty requirement is met by relabelling, not by refusing.
_TT_VARIANTS: dict[str, dict[str, TtVariant]] = {
    "gfs_250mb": {
        d: TtVariant("uv200", "200 mb wind/jet forecast (GFS)") for d in ("wpac", "cpac")},
    "gfs_mslp_precip": {
        d: TtVariant("mslp_pcpn", "MSLP + precipitation forecast (GFS)") for d in ("wpac", "cpac")},
    # No temperature field of ANY level exists on the oceanic domains -- T850/T850a/T925/
    # T700/T2m/Td2m all 404, checked individually rather than inferred from the package
    # menu (which is not exhaustive: T850 serves on `us` while being absent from its menu).
    # 850 mb height+wind is the closest usable product: it still gives the low-level flow
    # and LLJ signal an 850 mb panel is mostly read for, just not the thermal field.
    "gfs_850mb_temp": {
        d: TtVariant("mslp_uv850", "850 mb height + wind forecast (GFS) -- NOT temperature; "
                     "this domain publishes no temperature panel")
        for d in ("wpac", "cpac")},
}


def tt_variant(name: str, domain: str) -> TtVariant:
    """The field code and true label for this chart ON THIS DOMAIN. Verified 2026-07-28
    that the oceanic `mslp_pcpn` frame 1 is f006, the same anchor as `mslp_pcpn_frzn`
    (read off the panel's own 'Forecast Hour: [6]' label), so the f0 offset carries over
    unchanged and a substituted panel cannot desync by 6 h from its siblings."""
    sub = _TT_VARIANTS.get(name, {}).get(domain)
    return sub or TtVariant(CATALOG[name].params["field"], CATALOG[name].label)


def tt_field(name: str, domain: str) -> str:
    """Just the field code (URL building); see tt_variant for the label."""
    return tt_variant(name, domain).field

# (label, bbox W/S/E/N, TT domain, TT charts that domain serves, extra non-TT charts).
# ORDER MATTERS -- first bbox match wins, so tighter/likelier boxes come first (East Asia
# before Western Pacific, whose boxes overlap in longitude).
_MAP_REGIONS: tuple[tuple[str, tuple[float, float, float, float], str | None, tuple, tuple], ...] = (
    ("CONUS", (-125.0, 24.0, -66.0, 50.0), "us", TT_CHARTS,
     US_CHARTS + (OCEAN_ATLANTIC, OCEAN_PACIFIC)),
    ("Alaska", (-172.0, 51.0, -129.0, 72.0), "ak", TT_CHARTS, (OCEAN_PACIFIC,)),
    ("Hawaii", (-162.0, 18.0, -154.0, 23.0), "cpac", TT_OCEANIC, (OCEAN_PACIFIC,)),
    # Reaches down to 20N and west to 116E on purpose: at 30N/124E the box missed RODN
    # (Kadena, Okinawa, 26.3N 127.8E), which then matched nothing at all and got no charts.
    ("East Asia", (116.0, 20.0, 146.0, 46.0), "ea", TT_CHARTS, ()),
    ("Western Pacific", (130.0, 5.0, 175.0, 30.0), "wpac", TT_OCEANIC, ()),
    ("Southern South America", (-95.0, -58.0, -30.0, -15.0), "samer", TT_CHARTS, ()),
    ("Europe", (-11.0, 35.0, 32.0, 60.0), "eu", TT_CHARTS, ()),
    ("Middle East", (34.0, 12.0, 63.0, 40.0), "me", TT_CHARTS, ()),
)


def charts_for_latlon(lat: float, lon: float) -> tuple[tuple[str, ...], str | None, str]:
    """(allowed CATALOG names, TT domain, region label) for a station's position.

    Returns the charts that actually DEPICT this station's weather. An unknown position
    falls through to 'no charts' rather than to the US set -- silently serving CONUS is
    the exact failure mode being prevented, so the safe default is nothing."""
    for label, (w, s, e, n), domain, tt, extra in _MAP_REGIONS:
        if s <= lat <= n and w <= lon <= e:
            return tuple(extra) + tuple(tt), domain, label
    return (), None, "outside every mapped region"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def latest_gfs_run(now: datetime | None = None) -> datetime:
    """Freshest GFS cycle (00/06/12/18Z) old enough to have posted -- naive UTC. Back
    off _GFS_POST_LAG_H hours from now, then snap down to the 6-hourly cycle grid."""
    t = (now or _utcnow()) - timedelta(hours=_GFS_POST_LAG_H)
    return t.replace(hour=(t.hour // GFS_STEP_H) * GFS_STEP_H, minute=0, second=0, microsecond=0)


def _url(spec: ChartSpec, *, fhr: int, run: datetime, sector: str,
         domain: str = "us") -> str:
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
                f"{run:%Y%m%d%H}/gfs_{tt_field(spec.name, domain)}_{domain}_{frame}.png")
    raise ValueError(f"unknown source {spec.source!r}")


def map_url(name: str, *, fhr: int = 0, run: datetime | None = None, sector: str = "s19",
            domain: str = "us") -> str:
    """The exact provider URL for a catalogued chart. Exposed so a caller can cite
    provenance without fetching. `fhr`/`run` apply only to forecast (TT) charts."""
    spec = CATALOG[name]
    if spec.source == "tt" and run is None:
        run = latest_gfs_run()
    return _url(spec, fhr=fhr, run=run, sector=sector, domain=domain)


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


def cache_path(name: str, *, fhr: int = 0, run: datetime | None = None,
               domain: str = "us") -> Path:
    """Where fetch_map(..., use_cache=True) stores this image. Forecast charts key on
    (run, fhr) -- deterministic; analysis charts key on the fetch hour, since 'now'
    isn't in the URL. The TT DOMAIN is part of the key: the same field/run/fhr is a
    different picture per domain, so omitting it would serve a cached CONUS panel for a
    South American request (and vice versa) -- silent, and permanent once archived."""
    spec = CATALOG[name]
    if spec.source == "tt":
        run = run or latest_gfs_run()
        tag = f"{domain}_{run:%Y%m%d%H}_f{fhr:03d}"
    else:
        tag = f"{_utcnow():%Y%m%d%H}"
    return _CACHE_DIR / f"{spec.name}_{tag}.{spec.ext}"


def fetch_map(
    name: str,
    *,
    fhr: int = 0,
    run: datetime | None = None,
    sector: str = "s19",
    domain: str = "us",
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
    url = _url(spec, fhr=fhr, run=run, sector=sector, domain=domain)
    cache_file = cache_path(name, fhr=fhr, run=run, domain=domain)

    if use_cache and cache_file.exists():
        return cache_file.read_bytes()
    data = _get(url, referer=_REFERER.get(spec.source))
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
    return data
