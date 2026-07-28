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
import re
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
        # NB GOES-West's "CONUS" product is PACUS -- it is centered on the eastern Pacific
        # and Hawaii and does NOT show the central or eastern US. Labeled for what it
        # actually is, because "CONUS (GOES-West)" read as a drop-in for conus_east and
        # sent two roster stations an image they were not in (see satellite_region_for_latlon).
        SatRegion("conus_west", GOES_WEST, "CONUS", "1250x750", "Pacific/West (GOES-West PACUS)"),
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
        # East edge widened -104 -> -99 (2026-07-28): the real nr sector reaches the upper
        # Mississippi (Iowa is visible at its right edge), but the old bbox stopped at the
        # Montana line, leaving the DAKOTAS in a gap between this and upper_mississippi.
        # KMIB and KRCA fell in that gap and dropped to the CONUS fallback. Sectors overlap
        # generously by design -- the nearest-center rule below arbitrates.
        SatRegion("northern_rockies", GOES_EAST, "SECTOR/nr", "1200x1200", "Northern Rockies",
                  (-117.0, 42.0, -99.0, 49.5)),
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

# CONUS extent for the fallback when a station sits in no named sector. These GOES bounds
# are only the LAST-RESORT fallback: the Himawari and Meteosat sectors in SAT_REGIONS are
# matched by bbox first, so an OCONUS point normally resolves before reaching here.
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
    (bird by longitude), else full disk. OCONUS IS COVERED: the Himawari (W Pacific/E Asia)
    and Meteosat (Europe/Middle East/Africa) sectors are matched by bbox in the same first
    pass as the GOES sectors, so Japan/Korea/Okinawa, Guam, Europe and the Gulf all resolve.
    None only for a point in no sector AND outside the GOES footprint -- i.e. a genuine gap
    (e.g. the Indian Ocean / central Asia), not an OCONUS-is-unsupported signal."""
    hits = [r for r in SAT_REGIONS.values() if r.bbox and _in_bbox(r.bbox, lat, lon)]
    if hits:
        def _center_km(r: SatRegion) -> float:
            w, s, e, n = r.bbox
            return _haversine_km(lat, lon, (s + n) / 2, (w + e) / 2)
        return min(hits, key=_center_km).name
    if _in_bbox(_CONUS_BBOX, lat, lon):
        # ALWAYS GOES-East here: its CONUS product covers the whole continental US, while
        # GOES-West's "CONUS" is PACUS (eastern Pacific + Hawaii) and would return an image
        # the station is not in. Splitting this fallback by longitude was the second half of
        # the KMIB/KRCA defect; the bird split below is still right for FULL DISKS, which
        # genuinely differ by hemisphere.
        return "conus_east"
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
# mp4 (video-capable models). This module fetches the FRAMES; charts.py composes them.
#
# REBUILT 2026-07-28: off NASA GIBS, onto each provider's OWN timestamped archive. GIBS had
# two defects that together made a loop unfreezable. It could only be probed BACKWARDS from
# now (fetch a tiny tile, check it isn't blank, step back), so no loop could be built for a
# past instant -- and an archive that can only capture "now" cannot be replayed. And its NRT
# geocolor is too intermittent to loop, so every product silently collapsed to clean-IR: the
# tool offered four products and served one, which is misinformation, not a limitation.
# Both providers used here PUBLISH the times they hold, so frames are now SELECTED from a
# real index rather than probed for, every canonical product is honoured, and `at` can name
# any instant still inside the provider's window.
#
# The trade: STAR serves the CDN's fixed scopes, so a GOES loop is SECTOR-wide (CONUS or
# full disk where no sector covers the point) rather than cropped to the station. Wider than
# the old GIBS crop, and the price of having an archive at all. Meteosat stays station-
# centered -- its WMS takes an arbitrary bbox AND a TIME, so it was already freezable.
LOOP_DEFAULT_FRAMES = 6
LOOP_MAX_FRAMES = 10
LOOP_DEFAULT_STEP_MIN = 30

# What each provider actually holds, for callers deciding when a loop must be captured:
# STAR keeps ~10 days at a 5-minute cadence; SLIDER's index is its last 100 scans (~17h at
# Himawari's 10-minute cadence). Past those windows a loop is not recoverable at all, so an
# archiver has to capture it during the round, not reconstruct it afterwards.
STAR_HISTORY_H = 240
SLIDER_HISTORY_H = 17

# Per-frame pixel target: take the smallest offered size that reaches it. charts.filmstrip
# renders each tile at ~341 px (3.1 in at 110 dpi), so 600 px is already oversampled and a
# sector's 2400x2400 would cost ~16x the bytes for detail no tile can show.
_LOOP_TARGET_PX = 500
# How far from a requested time we will accept a real frame. Half a default step, so a step
# finer than the provider's cadence yields fewer honestly-spaced frames, never repeats.
_LOOP_SNAP_TOL_MIN = 15

# STAR filenames: <YYYYDDDHHMM>_GOES19-ABI-<scope>-<PRODUCT>-<W>x<H>.jpg (day-of-year stamp).
_STAR_FRAME_RE = re.compile(r'href="((\d{11})_GOES\d+-ABI-[^"]*?-(\d+x\d+)\.jpg)"')


@dataclass(frozen=True)
class LoopFrame:
    """One fetched loop frame, carrying what an ARCHIVE needs to key on rather than only
    what a chart needs to draw.

    `time` is the frame's REAL valid time (naive UTC) and `url` is the exact byte source, so
    a stored frame is identified by what the provider actually returned -- not by what was
    requested. That distinction is the whole point: `label` is a display string with no year
    or month ('28 15:56Z'), fine on a filmstrip tile and useless as a storage key, and the
    requested time is not the served time once `_select_times` snaps to a real scan. Frames
    are shared across stations sitting in the same satellite sector, so `url` doubles as the
    natural de-duplication key."""
    time: datetime
    label: str
    url: str
    data: bytes


def _star_frame_index(sat: str, scope: str, product: str) -> dict[datetime, dict[str, str]]:
    """{valid time -> {size -> url}} from the CDN's directory index for one bird/scope/
    product. The index IS the archive listing, so every frame named here is a frame that
    exists -- the empty-NAM-directory failure mode cannot happen silently."""
    base = f"{_STAR_CDN}/GOES{sat}/ABI/{scope}/{SAT_PRODUCTS[product]}/"
    html = _get(base).decode("utf-8", "replace")
    out: dict[datetime, dict[str, str]] = {}
    for name, stamp, size in _STAR_FRAME_RE.findall(html):
        out.setdefault(datetime.strptime(stamp, "%Y%j%H%M"), {})[size] = base + name
    return out


def _pick_size(sizes: dict[str, str]) -> str:
    """Smallest offered size whose long edge reaches _LOOP_TARGET_PX, else the largest."""
    def long_edge(s: str) -> int:
        return max(int(x) for x in s.split("x"))
    ordered = sorted(sizes, key=long_edge)
    return next((s for s in ordered if long_edge(s) >= _LOOP_TARGET_PX), ordered[-1])


def _select_times(available: list[datetime], n: int, step_min: int,
                  at: datetime | None) -> list[datetime]:
    """Choose up to `n` frame times ending at/just before `at` (default: the newest frame
    held), spaced `step_min` apart, each snapped to the nearest REAL frame within
    _LOOP_SNAP_TOL_MIN. Returns ascending, de-duplicated: asking for a step finer than the
    provider's cadence gives fewer frames, not the same frame repeated."""
    times = sorted(t for t in available if at is None or t <= at)
    if not times:
        return []
    anchor = times[-1]
    picked: list[datetime] = []
    for k in range(n):
        target = anchor - timedelta(minutes=step_min * k)
        best = min(times, key=lambda t: abs((t - target).total_seconds()))
        if abs((best - target).total_seconds()) <= _LOOP_SNAP_TOL_MIN * 60 and best not in picked:
            picked.append(best)
    return sorted(picked)


def _frame_get(url: str, *, referer: str | None = None) -> bytes | None:
    """One loop frame, or None if it could not be fetched. Frames are INDEPENDENT: EUMETSAT's
    WMS in particular returns an intermittent 500 on perfectly valid times (measured
    2026-07-28 -- the same request succeeds on retry), and losing a whole station's loop to
    one flaky frame is a bad trade when the remaining frames still show the motion. One
    retry, then skip; the caller enforces the 2-frame floor below which there is no loop."""
    for attempt in range(2):
        try:
            return _get(url, referer=referer)
        except Exception:  # noqa: BLE001 -- any per-frame failure degrades to a skipped frame
            if attempt:
                return None
    return None


def _star_loop_frames(r: SatRegion, product: str, n: int, step_min: int,
                      at: datetime | None) -> list[LoopFrame]:
    idx = _star_frame_index(r.sat, r.scope, product)
    out: list[LoopFrame] = []
    for t in _select_times(list(idx), n, step_min, at):
        url = idx[t][_pick_size(idx[t])]
        if (data := _frame_get(url)) is not None:
            out.append(LoopFrame(t, f"{t:%d %H:%MZ}", url, data))
    return out


def _slider_times(sector: str, prod: str) -> list[datetime]:
    """The scan times SLIDER currently holds for a sector+product, oldest first."""
    raw = _get(f"{_SLIDER_BASE}/data/json/{_SLIDER_SAT}/{sector}/{prod}/latest_times.json",
               referer=_SLIDER_BASE)
    return sorted(datetime.strptime(str(ts), "%Y%m%d%H%M%S")
                  for ts in json.loads(raw)["timestamps_int"])


def _slider_loop_frames(sector: str, product: str, n: int, step_min: int,
                        at: datetime | None) -> list[LoopFrame]:
    prod = _SLIDER_PRODUCTS.get(product, _SLIDER_PRODUCTS["geocolor"])
    out: list[LoopFrame] = []
    for t in _select_times(_slider_times(sector, prod), n, step_min, at):
        url = _slider_tile_url(sector, prod, f"{t:%Y%m%d%H%M%S}")
        if (data := _frame_get(url, referer=_SLIDER_BASE)) is not None:
            out.append(LoopFrame(t, f"{t:%d %H:%MZ}", url, data))
    return out


def _eumetsat_loop_frames(lat: float, lon: float, product: str, n: int, step_min: int,
                          at: datetime | None) -> list[LoopFrame]:
    """MTG posts every 10 min with a short latency, and the WMS holds a rolling archive, so
    `at` just moves the anchor. Without one, back off two slots from now so the newest
    requested frame is reliably served."""
    base = (at or _utcnow()).replace(second=0, microsecond=0)
    base -= timedelta(minutes=base.minute % 10)
    if at is None:
        base -= timedelta(minutes=20)
    out: list[LoopFrame] = []
    for t in sorted(base - timedelta(minutes=step_min * k) for k in range(n)):
        url = meteosat_point_url(lat, lon, product) + f"&time={t:%Y-%m-%dT%H:%M:00.000Z}"
        if (data := _frame_get(url)) is not None:
            out.append(LoopFrame(t, f"{t:%d %H:%MZ}", url, data))
    return out


def satellite_loop(lat: float, lon: float, product: str, *, frames: int = LOOP_DEFAULT_FRAMES,
                   step_min: int = LOOP_DEFAULT_STEP_MIN, at: datetime | None = None
                   ) -> tuple[list[LoopFrame], str, str]:
    """Fetch a short loop of time-stamped frames for a point. Returns (frames oldest-first
    as LoopFrame, source_label, coverage_label).

    `at` anchors the loop at a past instant (naive UTC) instead of the newest frame held --
    this is what makes a loop archivable and replayable. Provider by coverage: NESDIS/STAR
    for GOES, RAMMB/CIRA SLIDER for Himawari, EUMETSAT WMS for Meteosat; every provider
    honours the canonical `product`. Raises ValueError for a point with no geostationary
    coverage.

    Returns the FRAMES, deliberately: the filmstrip and mp4 are composed downstream by
    charts.py, so an archiver stores these bytes and replay re-composes any (frames,
    step_min) the model asks for. Freezing the composite instead would bake in one
    parameter combination and one rendering."""
    frames = max(2, min(int(frames), LOOP_MAX_FRAMES))
    region = satellite_region_for_latlon(lat, lon)
    if region is None:
        raise ValueError("no geostationary satellite coverage for this point")
    r = SAT_REGIONS[region]
    if r.provider == "meteosat_eumetsat_wms":
        fr = _eumetsat_loop_frames(lat, lon, product, frames, step_min, at)
        return (fr, _PROVIDER_SOURCE["meteosat_eumetsat_wms"],
                f"Meteosat {product}, station-centered")
    if r.provider in ("himawari_slider", "himawari_ospo"):
        # OSPO serves one LATEST gif and keeps no archive, so Himawari loops always come
        # from SLIDER. An OSPO region borrows SLIDER's matching sector rather than falling
        # back to the whole disk, so a Japan station keeps its tight view.
        sector = "japan" if r.provider == "himawari_ospo" else r.scope
        fr = _slider_loop_frames(sector, product, frames, step_min, at)
        return (fr, _PROVIDER_SOURCE["himawari_slider"],
                f"Himawari {product}, {sector.replace('_', ' ')}")
    fr = _star_loop_frames(r, product, frames, step_min, at)
    bird = "GOES-East" if r.sat == GOES_EAST else "GOES-West"
    return fr, f"NESDIS/STAR ({bird})", f"{r.label}, {product}"


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
