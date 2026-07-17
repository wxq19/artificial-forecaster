"""Satellite + radar imagery client -- live observational-imagery fetch seam.

Sibling to wxmaps.py / soundings.py: a network data-source client that fetches
PRE-RENDERED, model-ready satellite and radar images from public providers and returns
raw bytes. No matplotlib (charts.py stays the only matplotlib file), no SQL. Feeding the
model the same imagery a human forecaster reads keeps the comparison honest.

Two families:
  - Satellite: three providers behind one region catalog, dispatched per region.provider.
    * goes_star: NOAA/NESDIS STAR CDN (cdn.star.nesdis.noaa.gov). Direct sized JPEGs by
      scope (CONUS / full disk / named sector) and product (geocolor / visible band 02 /
      clean-IR band 13 / mid-level water-vapor band 09). GOES-East = GOES19, GOES-West =
      GOES18 -- operator-updatable per bird (the retired GOES16/17 URLs redirect).
    * himawari_slider: RAMMB/CIRA SLIDER (Western Pacific / East Asia -- e.g. Japan). One
      zoom-0 tile IS the whole disk/sector, so a single fetch returns a full image with no
      stitching; timestamped, so the cache is reproducible.
    * meteosat_eumetsat_wms: EUMETSAT's own WMS (Europe / Africa / Middle East). One GetMap
      returns an already-colorized RGB (MTG geocolour) with a coastline/border overlay --
      NOT a bare raster. Used instead of SLIDER's Meteosat, whose feed is unreliable.
    All are research/informational imagery, NOT an operational decision source.
  - Radar: Iowa Environmental Mesonet (mesonet.agron.iastate.edu) -- already a repo
    dependency (iem.py). All three modes render through IEM's radmap.php, which returns a
    PRE-COMPOSITED, labeled PNG (state/county borders, dBZ legend, timestamp): national
    (sector=conus), a bbox-scoped regional map, and a bbox-scoped station-centered view.
    NWS RIDGE GIF (radar.weather.gov) is the low-bandwidth national fallback.

    NOTE: the raw IEM rasters (data/gis/images/4326/USCOMP, .../ridge/<SITE>) are BARE
    reflectivity with no geography or legend -- unusable to a VLM -- so we deliberately do
    NOT use them; radmap.php is the composited path.

Radar is station-aware: nearest_radar() finds the closest WSR-88D site (radarsites.py,
sourced from IEM's NEXRAD network) to a lat/lon, gated by a 150 km credibility guard so a
radar-sparse station degrades to a regional/national mosaic instead of a distant, falsely
precise local view.
"""

import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from forecaster.radarsites import RADARS

# Anchor the cache at the repo root (like config.py), not the cwd.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "imagery"

# Space requests politely across ALL hosts (module-level, like iem/awc/soundings/wxmaps).
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"


# --- Satellite: NOAA/NESDIS STAR CDN -------------------------------------------------
# Operator-updatable per bird: the GOES-East/West satellite numbers change at satellite
# transitions, and the retired-bird URLs redirect to a placeholder -- keep these current.
GOES_EAST = "19"
GOES_WEST = "18"
_STAR_CDN = "https://cdn.star.nesdis.noaa.gov"

# Canonical product vocabulary (the model-facing enum keys). For GOES the value is the STAR
# product/band token; the non-GOES providers map the SAME keys to their own tokens below.
# geocolor is the day/night default.
SAT_PRODUCTS: dict[str, str] = {
    "geocolor": "GEOCOLOR",   # true-color by day, IR-blended at night (default)
    "visible": "02",          # ABI band 2, 0.64 um red visible (daytime only)
    "infrared": "13",         # ABI band 13, 10.3 um clean-IR window (cloud-top temp)
    "water_vapor": "09",      # ABI band 9, 6.9 um mid-level water vapor
}

# Himawari-9 via RAMMB/CIRA SLIDER. A single zoom-0 tile = the whole disk/sector (no stitch).
# Third-party relative to STAR -- watched, not trusted (like TropicalTidbits in wxmaps).
_SLIDER_BASE = "https://rammb-slider.cira.colostate.edu"
_SLIDER_SAT = "himawari"     # the only SLIDER bird we use; Meteosat comes from EUMETSAT direct
# canonical product -> SLIDER Himawari-9 slug (band_03 = 0.64um vis, band_13 = 10.4um clean-IR,
# band_09 = 6.9um mid-WV; the AHI analogues of the GOES canonical bands).
_SLIDER_PRODUCTS: dict[str, str] = {
    "geocolor": "geocolor", "visible": "band_03",
    "infrared": "band_13", "water_vapor": "band_09",
}

# Meteosat via EUMETSAT's own WMS (SLIDER's Meteosat feed is unreliable). One GetMap returns a
# colorized RGB with a coastline/border overlay. canonical product -> WMS layer; MTG (Meteosat
# Third Gen, 0deg) geocolour is the day/night default and covers Europe/Africa AND the Middle
# East. Products without a mapping fall back to geocolour in v1.
_EUMETSAT_WMS = "https://view.eumetsat.int/geoserver/wms"
_EUMETSAT_BOUNDARIES = "osmgray:all_boundaries_light"
_EUMETSAT_LAYERS: dict[str, str] = {
    "geocolor": "mtg_fd:rgb_geocolour", "visible": "mtg_fd:rgb_truecolour",
}
_EUMETSAT_DEFAULT = "geocolor"

# Himawari-9 tight regional views via NOAA/OSPO (JMA relay): clean rectangular sector GIFs
# with coastlines, a color scale, and a printed timestamp. Day/night enhanced-IR ("rb"), plus
# daytime visible and water vapor -- NO geocolor, so geocolor falls back to enhanced IR (the
# tool relabels it 'infrared'). scope = the OSPO sector path (e.g. "jma/japan").
_OSPO_BASE = "https://www.ospo.noaa.gov"
_OSPO_PRODUCTS: dict[str, str] = {
    "geocolor": "rb", "infrared": "rb", "visible": "vis", "water_vapor": "wv",
}

# Human-readable source per provider, for receipts/provenance.
_PROVIDER_SOURCE: dict[str, str] = {
    "goes_star": "NESDIS/STAR (GOES)",
    "himawari_slider": "RAMMB/CIRA SLIDER (Himawari-9)",
    "himawari_ospo": "NOAA/OSPO (Himawari-9, enhanced IR)",
    "meteosat_eumetsat_wms": "EUMETSAT (Meteosat MTG, 0deg)",
}


@dataclass(frozen=True)
class SatRegion:
    """One satellite scope. Fields are interpreted per `provider`:
      - goes_star:             sat=bird ("18"/"19"), scope=STAR path (CONUS/FD/SECTOR/<code>),
                               size=sized-image stem ("1250x750").
      - himawari_slider:       scope=SLIDER sector slug ("full_disk"/"japan"); sat/size unused.
      - meteosat_eumetsat_wms: bbox=GetMap extent, size="WxH" request pixels; sat/scope unused.
    bbox (W,S,E,N) also routes a station to its covering region; every named sector and every
    non-GOES region carries one. CONUS/full-disk GOES stay bbox-less lat/lon fallbacks."""
    name: str
    sat: str        # "19" | "18" -- a field, not a hardcode, so a bird swap is one edit
    scope: str      # "CONUS" | "FD" | "SECTOR/pr" ...
    size: str       # "1250x750" etc.
    label: str
    bbox: tuple[float, float, float, float] | None = None   # (W, S, E, N); sectors only
    provider: str = "goes_star"


# Keyed by semantic region name. Sizes/sector codes were live-verified against the CDN.
# Sector bboxes are APPROXIMATE (sectors overlap generously and CONUS is the safety net),
# tunable, and used only to pick a default region from a station's coordinates.
SAT_REGIONS: dict[str, SatRegion] = {
    r.name: r for r in [
        SatRegion("conus_east", GOES_EAST, "CONUS", "1250x750", "CONUS (GOES-East)"),
        SatRegion("conus_west", GOES_WEST, "CONUS", "1250x750", "CONUS (GOES-West)"),
        SatRegion("full_disk_east", GOES_EAST, "FD", "1808x1808", "Full disk (GOES-East)"),
        SatRegion("full_disk_west", GOES_WEST, "FD", "1808x1808", "Full disk (GOES-West)"),
        SatRegion("alaska", GOES_WEST, "SECTOR/ak", "1000x1000", "Alaska",
                  (-170.0, 51.0, -129.0, 72.0)),
        SatRegion("hawaii", GOES_WEST, "SECTOR/hi", "1200x1200", "Hawaii",
                  (-161.0, 18.0, -154.0, 23.0)),
        SatRegion("pacific_northwest", GOES_WEST, "SECTOR/pnw", "1200x1200",
                  "Pacific Northwest", (-125.0, 41.0, -110.0, 50.0)),
        SatRegion("pacific_southwest", GOES_WEST, "SECTOR/psw", "1200x1200",
                  "Pacific Southwest", (-125.0, 30.0, -108.0, 43.0)),
        SatRegion("puerto_rico", GOES_EAST, "SECTOR/pr", "1200x1200", "Puerto Rico",
                  (-68.0, 17.0, -64.0, 19.5)),
        SatRegion("caribbean", GOES_EAST, "SECTOR/car", "1000x1000", "Caribbean",
                  (-85.0, 10.0, -60.0, 27.0)),
        SatRegion("northeast", GOES_EAST, "SECTOR/ne", "1200x1200", "Northeast",
                  (-85.0, 37.0, -66.0, 48.0)),
        SatRegion("southeast", GOES_EAST, "SECTOR/se", "1200x1200", "Southeast",
                  (-95.0, 24.0, -75.0, 37.0)),
        SatRegion("southern_plains", GOES_EAST, "SECTOR/sp", "1200x1200", "Southern Plains",
                  (-106.0, 26.0, -90.0, 39.0)),
        SatRegion("southern_rockies", GOES_EAST, "SECTOR/sr", "1200x1200", "Southern Rockies",
                  (-114.0, 31.0, -101.0, 42.0)),
        SatRegion("upper_mississippi", GOES_EAST, "SECTOR/umv", "1200x1200",
                  "Upper Mississippi Valley", (-98.0, 41.0, -86.0, 49.0)),
        SatRegion("great_lakes", GOES_EAST, "SECTOR/cgl", "1200x1200", "Central Great Lakes",
                  (-92.0, 38.0, -80.0, 46.0)),
        SatRegion("northern_rockies", GOES_EAST, "SECTOR/nr", "1200x1200", "Northern Rockies",
                  (-117.0, 42.0, -104.0, 49.5)),
        # Himawari-9: full_disk is the wide SLIDER geocolor view (clean single tile). japan is
        # the TIGHT local view via OSPO (clean rectangular enhanced-IR GIF, coastlines + scale +
        # timestamp) -- SLIDER's japan tile is padded/irregular, so OSPO serves the tight one.
        # A Japan-area station auto-routes to japan (its bbox is nearer than full_disk's).
        SatRegion("himawari_full_disk", "", "full_disk", "", "Himawari full disk (W Pacific/E Asia)",
                  (80.0, -60.0, 180.0, 60.0), provider="himawari_slider"),
        SatRegion("himawari_japan", "", "jma/japan", "", "Himawari -- Japan (enhanced IR)",
                  (122.0, 24.0, 150.0, 46.0), provider="himawari_ospo"),
        # Meteosat (EUMETSAT WMS, MTG 0deg geocolour): Europe / Africa / Middle East. bbox is
        # the GetMap extent; size is the request pixel WxH (roughly the bbox aspect).
        SatRegion("europe", "", "", "1000x760", "Europe (Meteosat)",
                  (-15.0, 34.0, 42.0, 62.0), provider="meteosat_eumetsat_wms"),
        SatRegion("middle_east", "", "", "900x740", "Middle East (Meteosat)",
                  (35.0, 12.0, 65.0, 40.0), provider="meteosat_eumetsat_wms"),
        SatRegion("africa", "", "", "820x1000", "Africa (Meteosat)",
                  (-20.0, -36.0, 52.0, 38.0), provider="meteosat_eumetsat_wms"),
    ]
}

# CONUS extent for the fallback when a station sits in no named sector; the GOES coverage
# bound rejects OCONUS points (no GOES East/West view -> Meteosat/Himawari, deferred).
_CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)
_GOES_LON = (-180.0, -8.0)     # Western Hemisphere incl. the Pacific; excludes EU/ME/Asia
_GOES_LAT = (-60.0, 70.0)
_EAST_WEST_LON = -100.0        # bird split for the CONUS/full-disk fallback


def satellite_source(region: str) -> str:
    """Human-readable provider/source label for a region (receipts/provenance)."""
    return _PROVIDER_SOURCE[SAT_REGIONS[region].provider]


def _eumetsat_getmap_url(r: SatRegion, product: str) -> str:
    """Exact EUMETSAT WMS GetMap URL: colorized RGB layer + a coastline/border overlay over
    the region bbox. TIME omitted -> the server returns the latest scan."""
    layer = _EUMETSAT_LAYERS.get(product, _EUMETSAT_LAYERS[_EUMETSAT_DEFAULT])
    width, height = r.size.split("x")
    w, s, e, n = r.bbox
    return (f"{_EUMETSAT_WMS}?service=WMS&version=1.1.1&request=GetMap"
            f"&layers={layer},{_EUMETSAT_BOUNDARIES}&srs=EPSG:4326"
            f"&bbox={w},{s},{e},{n}&width={width}&height={height}&format=image/png")


def _slider_tile_url(sector: str, prod: str, ts: str, *, sat: str = _SLIDER_SAT) -> str:
    """Zoom-0 single-tile URL -- the whole disk/sector as one image (no stitching)."""
    return (f"{_SLIDER_BASE}/data/imagery/{ts[0:4]}/{ts[4:6]}/{ts[6:8]}"
            f"/{sat}---{sector}/{prod}/{ts}/00/000_000.png")


def _slider_latest_ts(sector: str, prod: str, *, sat: str = _SLIDER_SAT) -> str:
    """Newest posted timestamp (YYYYMMDDHHMMSS) for a SLIDER sat/sector+product."""
    raw = _get(f"{_SLIDER_BASE}/data/json/{sat}/{sector}/{prod}/latest_times.json",
               referer=_SLIDER_BASE)
    return str(json.loads(raw)["timestamps_int"][0])


def satellite_url(region: str, product: str) -> str:
    """Provenance URL for a region+product (no network). GOES and EUMETSAT are exact and
    fetchable as-is; the SLIDER reference is the latest-times index (the tile URL needs the
    served timestamp, resolved in fetch_satellite)."""
    r = SAT_REGIONS[region]
    if r.provider == "goes_star":
        return f"{_STAR_CDN}/GOES{r.sat}/ABI/{r.scope}/{SAT_PRODUCTS[product]}/{r.size}.jpg"
    if r.provider == "himawari_slider":
        prod = _SLIDER_PRODUCTS.get(product, "geocolor")
        return f"{_SLIDER_BASE}/data/json/{_SLIDER_SAT}/{r.scope}/{prod}/latest_times.json"
    if r.provider == "himawari_ospo":
        return f"{_OSPO_BASE}/{r.scope}/{_OSPO_PRODUCTS.get(product, 'rb')}.gif"
    if r.provider == "meteosat_eumetsat_wms":
        return _eumetsat_getmap_url(r, product)
    raise ValueError(f"unknown satellite provider {r.provider!r}")


def _in_bbox(bbox: tuple[float, float, float, float], lat: float, lon: float) -> bool:
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def satellite_region_for_latlon(lat: float, lon: float) -> str | None:
    """Best default satellite region for a point, tightest useful view first: the named
    SECTOR whose bbox contains it (nearest sector center wins on overlap), else CONUS
    (bird by longitude), else full disk. None if the point is outside the GOES East/West
    footprint (OCONUS -> Meteosat/Himawari, a deferred fast-follow)."""
    hits = [r for r in SAT_REGIONS.values() if r.bbox and _in_bbox(r.bbox, lat, lon)]
    if hits:
        def _center_km(r: SatRegion) -> float:
            w, s, e, n = r.bbox
            return _haversine_km(lat, lon, (s + n) / 2, (w + e) / 2)
        return min(hits, key=_center_km).name
    if _in_bbox(_CONUS_BBOX, lat, lon):
        return "conus_west" if lon < _EAST_WEST_LON else "conus_east"
    if _GOES_LON[0] <= lon <= _GOES_LON[1] and _GOES_LAT[0] <= lat <= _GOES_LAT[1]:
        return "full_disk_west" if lon < _EAST_WEST_LON else "full_disk_east"
    return None


# --- Radar: Iowa Environmental Mesonet (pre-composited via radmap.php) ----------------
_IEM = "https://mesonet.agron.iastate.edu"
# Public so the tool can cite the ACTUAL source when national degrades from IEM to this GIF.
NWS_RIDGE_GIF_URL = "https://radar.weather.gov/ridge/standard/CONUS_0.gif"  # low-bw fallback

RADAR_PRODUCTS = ("station_reflectivity", "regional_mosaic", "national_mosaic")
RADAR_STATION_GUARD_KM = 150.0    # nearest radar must be within this to be "credible local"
_STATION_PAD_DEG = 2.4            # half-width of the station-centered radar bbox (~260 km)

# Curated regional fallbacks: bbox (W, S, E, N) in EPSG:4326 for radmap.php. "national" is
# special-cased (sector=conus). radmap renders a composited, geolocated map for a territory
# bbox too (verified for PR/HI), so hawaii/puerto_rico/alaska are best-effort territory
# fallbacks -- their nexrad ECHO coverage is best-effort (only confirmed the basemap renders;
# active-echo coverage over the territories is not yet verified on an active-weather frame).
RADAR_REGIONS: dict[str, tuple[tuple[float, float, float, float] | None, str]] = {
    "northwest": ((-125.0, 41.0, -110.0, 49.5), "Pacific Northwest"),
    "southwest": ((-125.0, 31.0, -108.0, 42.5), "Southwest"),
    "southern_plains": ((-106.0, 29.0, -90.0, 39.5), "Southern Plains"),
    "midwest": ((-97.0, 36.5, -80.5, 47.5), "Midwest"),
    "southeast": ((-92.0, 24.5, -75.0, 37.5), "Southeast"),
    "northeast": ((-82.0, 37.0, -66.5, 47.5), "Northeast"),
    "hawaii": ((-160.5, 18.5, -154.5, 22.5), "Hawaii"),
    "puerto_rico": ((-67.5, 17.5, -65.0, 18.7), "Puerto Rico"),
    "alaska": ((-170.0, 51.0, -129.0, 72.0), "Alaska"),
    "national": (None, "National (CONUS)"),
}

# radmap layer stacks: zoomed views get counties for context; national gets states only.
_RADMAP_LOCAL = "layers[]=nexrad&layers[]=uscounties&layers[]=states"
_RADMAP_NATIONAL = "sector=conus&layers[]=nexrad&layers[]=states"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(a))


def nearest_radar(lat: float, lon: float) -> tuple[dict, float] | None:
    """Closest WSR-88D site to a point: ({id,name,state,country,lat,lon}, dist_km), or
    None if the catalog is empty. Straight-line only -- a candidate, not a guarantee the
    radar is useful; the caller applies RADAR_STATION_GUARD_KM."""
    best: tuple[dict, float] | None = None
    for sid, name, state, country, rlat, rlon in RADARS:
        dkm = _haversine_km(lat, lon, rlat, rlon)
        if best is None or dkm < best[1]:
            best = ({"id": sid, "name": name, "state": state, "country": country,
                     "lat": rlat, "lon": rlon}, dkm)
    return best


def radar_region_for_latlon(lat: float, lon: float) -> str | None:
    """Curated region (CONUS or a best-effort territory: hawaii/puerto_rico/alaska) whose
    bbox contains the point, for a station with no credible local radar. None if outside
    every region -> caller uses the national mosaic."""
    for name, (bbox, _label) in RADAR_REGIONS.items():
        if bbox is None:
            continue
        w, s, e, n = bbox
        if w <= lon <= e and s <= lat <= n:
            return name
    return None


def _radmap_bbox_url(bbox: tuple[float, float, float, float], *, w: int, h: int) -> str:
    west, south, east, north = bbox
    return (f"{_IEM}/GIS/radmap.php?bbox={west:.2f},{south:.2f},{east:.2f},{north:.2f}"
            f"&{_RADMAP_LOCAL}&width={w}&height={h}")


def radar_url(mode: str, *, center: tuple[float, float] | None = None,
              region: str | None = None) -> str:
    """Exact radar image URL (provenance). mode: 'station' (needs center=(lat,lon)),
    'regional' (needs region), or 'national'."""
    if mode == "national":
        return f"{_IEM}/GIS/radmap.php?{_RADMAP_NATIONAL}&width=1000&height=600"
    if mode == "regional":
        bbox = RADAR_REGIONS[region][0]
        if bbox is None:                       # 'national' passed as a region
            return radar_url("national")
        return _radmap_bbox_url(bbox, w=900, h=700)
    if mode == "station":
        lat, lon = center
        bbox = (lon - _STATION_PAD_DEG, lat - _STATION_PAD_DEG,
                lon + _STATION_PAD_DEG, lat + _STATION_PAD_DEG)
        return _radmap_bbox_url(bbox, w=800, h=800)
    raise ValueError(f"unknown radar mode {mode!r}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _get(url: str, *, referer: str | None = None) -> bytes:
    """GET raw bytes, spacing requests politely across hosts (module-level throttle). A
    referer is sent for hotlink-gated third-party hosts (SLIDER)."""
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


def _cache_write(path: Path, data: bytes) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fetch_satellite(region: str, product: str, *, use_cache: bool = False) -> tuple[bytes, str]:
    """Fetch one satellite image; returns (image bytes, exact fetched URL). Analysis-style
    'now' imagery: GOES/EUMETSAT cache on the fetch hour (like wxmaps analysis charts); SLIDER
    caches on the served timestamp (reproducible). Dispatches on the region's provider."""
    r = SAT_REGIONS[region]
    if r.provider == "himawari_slider":
        prod = _SLIDER_PRODUCTS.get(product, "geocolor")
        ts = _slider_latest_ts(r.scope, prod)
        url = _slider_tile_url(r.scope, prod, ts)
        cache_file = _CACHE_DIR / f"sat_{region}_{prod}_{ts}.png"
        if use_cache and cache_file.exists():
            return cache_file.read_bytes(), url
        data = _get(url, referer=_SLIDER_BASE)
        if use_cache:
            _cache_write(cache_file, data)
        return data, url
    # goes_star / himawari_ospo / meteosat_eumetsat_wms: one deterministic GET (latest scan),
    # cache on the fetch hour.
    url = satellite_url(region, product)
    ext = {"goes_star": "jpg", "himawari_ospo": "gif"}.get(r.provider, "png")
    cache_file = _CACHE_DIR / f"sat_{region}_{product}_{_utcnow():%Y%m%d%H}.{ext}"
    if use_cache and cache_file.exists():
        return cache_file.read_bytes(), url
    data = _get(url)
    if use_cache:
        _cache_write(cache_file, data)
    return data, url


_METEOSAT_PAD_DEG = 7.0   # half-width of a station-centered Meteosat view (~1500 km across)


def meteosat_point_url(lat: float, lon: float, product: str,
                       *, pad: float = _METEOSAT_PAD_DEG) -> str:
    """EUMETSAT WMS GetMap centered on a point -- a tight, station-local Meteosat view. The
    WMS takes an arbitrary bbox, so no fixed sector is needed (the station-crop upgrade)."""
    layer = _EUMETSAT_LAYERS.get(product, _EUMETSAT_LAYERS[_EUMETSAT_DEFAULT])
    w, s, e, n = lon - pad, lat - pad, lon + pad, lat + pad
    return (f"{_EUMETSAT_WMS}?service=WMS&version=1.1.1&request=GetMap"
            f"&layers={layer},{_EUMETSAT_BOUNDARIES}&srs=EPSG:4326"
            f"&bbox={w:.3f},{s:.3f},{e:.3f},{n:.3f}&width=900&height=900&format=image/png")


def fetch_meteosat_point(lat: float, lon: float, product: str,
                         *, use_cache: bool = False) -> tuple[bytes, str]:
    """Fetch a station-centered Meteosat view; returns (png bytes, exact URL). Analysis-style
    'now' imagery, so the opt-in cache keys on the fetch hour + point."""
    url = meteosat_point_url(lat, lon, product)
    cache_file = _CACHE_DIR / f"sat_meteosat_{lat:.1f}_{lon:.1f}_{product}_{_utcnow():%Y%m%d%H}.png"
    if use_cache and cache_file.exists():
        return cache_file.read_bytes(), url
    data = _get(url)
    if use_cache:
        _cache_write(cache_file, data)
    return data, url


# --- Satellite LOOPS: N time-stamped frames for a filmstrip / short video -------------
# A VLM can't watch motion, so a "loop" is a filmstrip (one image, universal) or a short
# mp4 (video-capable models). This module fetches the FRAMES, STATION-CENTERED and zoomed
# via a bbox+TIME WMS so the field fills the frame (not a whole hemisphere): NASA GIBS for
# GOES (geocolor) and Himawari (clean-IR -- GIBS has no Himawari geocolor), EUMETSAT for
# Meteosat (MTG geocolor). All carry a coastline/border overlay. charts.py composes them.
LOOP_DEFAULT_FRAMES = 6
LOOP_MAX_FRAMES = 10
LOOP_DEFAULT_STEP_MIN = 30

# NASA GIBS WMS: colorized geostationary layers with a TIME dimension over an arbitrary bbox.
_GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
_GIBS_REF = "Reference_Features_15m"    # coastlines + political borders overlaid on the RGB
_GIBS_PAD_DEG = 4.0                     # half-width of a station-centered loop (~900 km across)
_GIBS_LAG_MIN = 40                      # near-real-time lag: newest frame is ~this many min old
_GIBS_BLANK_BYTES = 3000                # a GetMap smaller than this is an empty (no-data) frame


def _gibs_layer(coverage: str, product: str) -> str:
    """GIBS layer for a coverage family + canonical product. LOOPS default to clean-IR: it
    is produced every 10 min and is day/night stable, whereas GIBS geocolor NRT is
    intermittent (frequent no-data frames, worst at night) -- unusable for a smooth loop.
    Only an explicit `visible` request uses the daytime visible band."""
    if coverage == "himawari":
        return ("Himawari_AHI_Band3_Red_Visible_1km" if product == "visible"
                else "Himawari_AHI_Band13_Clean_Infrared")
    bird = "East" if coverage == "goes-east" else "West"
    return f"GOES-{bird}_ABI_Band13_Clean_Infrared"


def _gibs_time_url(layer: str, lat: float, lon: float, t: datetime, *, size: int = 900) -> str:
    w, s, e, n = lon - _GIBS_PAD_DEG, lat - _GIBS_PAD_DEG, lon + _GIBS_PAD_DEG, lat + _GIBS_PAD_DEG
    # WMS 1.3.0 + EPSG:4326 uses lat,lon axis order -> BBOX = south,west,north,east.
    return (f"{_GIBS_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer},{_GIBS_REF}"
            f"&CRS=EPSG:4326&BBOX={s},{w},{n},{e}&WIDTH={size}&HEIGHT={size}"
            f"&FORMAT=image/png&TIME={t:%Y-%m-%dT%H:%M:00Z}")


def _gibs_anchor_time(layer: str, lat: float, lon: float) -> datetime:
    """Newest available 10-min GIBS frame (NRT lag varies ~30-50 min): probe back from now
    with a tiny GetMap until one returns real (non-empty) pixels."""
    now = _utcnow().replace(second=0, microsecond=0)
    t = now - timedelta(minutes=now.minute % 10 + _GIBS_LAG_MIN)
    for _ in range(6):
        if len(_get(_gibs_time_url(layer, lat, lon, t, size=64))) > _GIBS_BLANK_BYTES:
            return t
        t -= timedelta(minutes=10)
    return t


def _gibs_loop_frames(layer: str, lat: float, lon: float, n: int, step_min: int
                      ) -> list[tuple[str, bytes]]:
    anchor = _gibs_anchor_time(layer, lat, lon)
    out: list[tuple[str, bytes]] = []
    for t in sorted(anchor - timedelta(minutes=step_min * k) for k in range(n)):
        data = _get(_gibs_time_url(layer, lat, lon, t))
        if len(data) > _GIBS_BLANK_BYTES:          # skip an empty frame rather than show black
            out.append((f"{t:%d %H:%MZ}", data))
    return out


def _eumetsat_loop_frames(lat: float, lon: float, product: str, n: int, step_min: int
                          ) -> list[tuple[str, bytes]]:
    # MTG posts every 10 min with a short latency; snap 'now' to the 10-min grid and back
    # off two slots so the newest requested frame is reliably available on the WMS.
    now = _utcnow().replace(second=0, microsecond=0)
    base = now - timedelta(minutes=now.minute % 10 + 20)
    targets = sorted(base - timedelta(minutes=step_min * k) for k in range(n))
    out: list[tuple[str, bytes]] = []
    for t in targets:
        url = meteosat_point_url(lat, lon, product) + f"&time={t:%Y-%m-%dT%H:%M:00.000Z}"
        out.append((f"{t:%d %H:%MZ}", _get(url)))
    return out


def satellite_loop(lat: float, lon: float, product: str, *, frames: int = LOOP_DEFAULT_FRAMES,
                   step_min: int = LOOP_DEFAULT_STEP_MIN
                   ) -> tuple[list[tuple[str, bytes]], str, str]:
    """Fetch a short, STATION-CENTERED loop of time-stamped frames for a point. Returns
    (frames oldest-first as (label, bytes), source_label, coverage_label). Provider by
    coverage: EUMETSAT WMS TIME for Meteosat; NASA GIBS WMS TIME for GOES (geocolor) and
    Himawari (clean-IR). Raises ValueError for a point with no geostationary coverage."""
    frames = max(2, min(int(frames), LOOP_MAX_FRAMES))
    region = satellite_region_for_latlon(lat, lon)
    if region is None:
        raise ValueError("no geostationary satellite coverage for this point")
    r = SAT_REGIONS[region]
    if r.provider == "meteosat_eumetsat_wms":
        fr = _eumetsat_loop_frames(lat, lon, product, frames, step_min)
        return fr, _PROVIDER_SOURCE["meteosat_eumetsat_wms"], "Meteosat geocolor, station-centered"
    mode = "visible" if product == "visible" else "clean-IR"
    if r.provider in ("himawari_slider", "himawari_ospo"):
        fr = _gibs_loop_frames(_gibs_layer("himawari", product), lat, lon, frames, step_min)
        return fr, "NASA GIBS (Himawari-9)", f"Himawari {mode}, station-centered"
    coverage = "goes-east" if r.sat == GOES_EAST else "goes-west"
    fr = _gibs_loop_frames(_gibs_layer(coverage, product), lat, lon, frames, step_min)
    return fr, f"NASA GIBS ({coverage.upper()})", f"{coverage.upper()} {mode}, station-centered"


def fetch_radar(mode: str, *, center: tuple[float, float] | None = None,
                region: str | None = None, use_cache: bool = False) -> bytes:
    """Fetch a composited radar image via radmap.php. On a national fetch failure, fall
    back to the official NWS low-bandwidth CONUS GIF (broad context). Opt-in cache keys
    on mode+target+hour."""
    url = radar_url(mode, center=center, region=region)
    if mode == "station":
        lat, lon = center
        key = f"station_{lat:.1f}_{lon:.1f}"
    else:
        key = mode if mode == "national" else f"regional_{region}"
    cache_file = _CACHE_DIR / f"radar_{key}_{_utcnow():%Y%m%d%H}.png"
    if use_cache and cache_file.exists():
        return cache_file.read_bytes()
    try:
        data = _get(url)
    except Exception:                          # noqa: BLE001 -- national degrades to the GIF
        if mode != "national":
            raise
        return _get(NWS_RIDGE_GIF_URL)         # gif bytes; caller's mime sniff handles it
    if use_cache:
        _cache_write(cache_file, data)
    return data
