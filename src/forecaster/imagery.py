"""Satellite + radar imagery client -- live observational-imagery fetch seam.

Sibling to wxmaps.py / soundings.py: a network data-source client that fetches
PRE-RENDERED, model-ready satellite and radar images from public providers and returns
raw bytes. No matplotlib (charts.py stays the only matplotlib file), no SQL. Feeding the
model the same imagery a human forecaster reads keeps the comparison honest.

Two families:
  - Satellite: NOAA/NESDIS STAR CDN (cdn.star.nesdis.noaa.gov). Direct sized JPEGs by
    scope (CONUS / full disk / named sector) and product (geocolor / visible band 02 /
    clean-IR band 13 / mid-level water-vapor band 09). GOES-East = GOES19, GOES-West =
    GOES18 -- operator-updatable per bird (the retired GOES16/17 URLs redirect). STAR is
    research/informational imagery, NOT an operational decision source.
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

import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
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

# product key -> STAR product/band token. geocolor is the day/night default.
SAT_PRODUCTS: dict[str, str] = {
    "geocolor": "GEOCOLOR",   # true-color by day, IR-blended at night (default)
    "visible": "02",          # ABI band 2, 0.64 um red visible (daytime only)
    "infrared": "13",         # ABI band 13, 10.3 um clean-IR window (cloud-top temp)
    "water_vapor": "09",      # ABI band 9, 6.9 um mid-level water vapor
}


@dataclass(frozen=True)
class SatRegion:
    """One satellite scope: which bird, the STAR scope path (CONUS / FD / SECTOR/<code>),
    the sized-image filename stem (varies by scope; verified per region), a label, and an
    approximate lat/lon bbox (W,S,E,N) used to route a station to its covering sector.
    Only the named SECTORs carry a bbox; CONUS/full-disk are handled as fallbacks."""
    name: str
    sat: str        # "19" | "18" -- a field, not a hardcode, so a bird swap is one edit
    scope: str      # "CONUS" | "FD" | "SECTOR/pr" ...
    size: str       # "1250x750" etc.
    label: str
    bbox: tuple[float, float, float, float] | None = None   # (W, S, E, N); sectors only


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
    ]
}

# CONUS extent for the fallback when a station sits in no named sector; the GOES coverage
# bound rejects OCONUS points (no GOES East/West view -> Meteosat/Himawari, deferred).
_CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)
_GOES_LON = (-180.0, -8.0)     # Western Hemisphere incl. the Pacific; excludes EU/ME/Asia
_GOES_LAT = (-60.0, 70.0)
_EAST_WEST_LON = -100.0        # bird split for the CONUS/full-disk fallback


def satellite_url(region: str, product: str) -> str:
    """Exact STAR CDN URL for a satellite region+product (provenance without fetching)."""
    r = SAT_REGIONS[region]
    return f"{_STAR_CDN}/GOES{r.sat}/ABI/{r.scope}/{SAT_PRODUCTS[product]}/{r.size}.jpg"


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


def _get(url: str) -> bytes:
    """GET raw bytes, spacing requests politely across hosts (module-level throttle)."""
    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _cache_write(path: Path, data: bytes) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fetch_satellite(region: str, product: str, *, use_cache: bool = False) -> bytes:
    """Fetch one STAR CDN satellite image. Analysis-style 'now' imagery, so the opt-in
    cache keys on the fetch hour (like wxmaps analysis charts)."""
    url = satellite_url(region, product)
    cache_file = _CACHE_DIR / f"sat_{region}_{product}_{_utcnow():%Y%m%d%H}.jpg"
    if use_cache and cache_file.exists():
        return cache_file.read_bytes()
    data = _get(url)
    if use_cache:
        _cache_write(cache_file, data)
    return data


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
