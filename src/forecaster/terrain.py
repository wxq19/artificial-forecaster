"""Terrain + coastline awareness -- STATIC geography around a station.

Sibling of soundings.py / wxmaps.py: a network data-source client. Because the data is
static (elevation and coastline never change), there is no leakage concern and a cached
tile is PERMANENT -- the air-gap path (SuperCloud nodes have no internet) is trivial here.
It owns no matplotlib (charts.hillshade draws the relief image from the sampled grid) and
no SQL/DuckDB.

Three data sources, all PyPI-clean (NO conda / GEOS -- the geospatial C stack stays deferred):
  - OpenTopoMap tiles -- pre-rendered topographic relief map (shaded relief + elevation color
    + contours + place names), fetched + stitched into one image. This is the "picture" a
    forecaster reads; we fetch pixels, we do not draw terrain (charts.py stays matplotlib-only).
  - Open-Meteo elevation API -- batched 90 m SRTM/GLO elevation for a radial sampling grid,
    turned into the QUANTITATIVE text "terrain rose" (elevation, relief, upslope/downslope).
  - global-land-mask -- a bundled ~1 km land/OCEAN mask for coastal proximity. LIMITATION:
    the mask is land vs OCEAN, so large inland lakes (Great Lakes, Great Salt Lake) read as
    land -> lake-effect coastlines are NOT detected. Documented gap, not a silent wrong answer.

A forecaster reads the surrounding terrain to reason about upslope fog/precip, downslope
drying/warming, valley cold-air pooling, and sea-breeze / advection fog -- so we hand the
model the same picture (the relief map) plus the quantified text rose.
"""

import io
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from global_land_mask import globe
from PIL import Image, ImageDraw, ImageFont

from .geo import compass16, destination, haversine_km

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "terrain"

_ENDPOINT = "https://api.open-meteo.com/v1/elevation"
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0
_MAX_POINTS = 100                 # Open-Meteo caps locations per request

# 16-point radial grid for the elevation profile.
_AZIMUTHS: tuple[float, ...] = tuple(i * 22.5 for i in range(16))
_RANGES_KM: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0)

# A rise/fall this many meters (vs the station's DEM elevation) counts as sloped terrain,
# not noise. Coarse on purpose -- the model reads tendencies, not survey data.
_SLOPE_THRESH_M = 50.0

# Coastline search: denser rings than the elevation profile, out to the sea-breeze range.
_COAST_RANGES_KM: tuple[float, ...] = tuple(float(k) for k in range(10, 160, 10))

# Relief map (OpenTopoMap tiles). Static + a small volunteer server -> cache permanently and
# be polite (descriptive UA, throttle). 50-mile radius: a TAF forecaster cares about terrain
# within ~50 mi; beyond that is context, not a driver. Rings mark 25 and 50 mi.
_OTM_TILE = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"
_OTM_ATTRIB = "(c) OpenTopoMap (CC-BY-SA)"
_KM_PER_MI = 1.60934
_MAP_RADIUS_MI = 50.0
_MAP_RINGS_MI = (25, 50)
_MAP_MAX_TILES = 7          # per axis; pick the highest zoom (most detail) that fits this
_TILE_PX = 256


@dataclass
class TerrainProfile:
    """Static terrain + coastline picture around a point. `grid[i][j]` is the DEM elevation
    (m) at azimuth `azimuths[i]`, range `ranges_km[j]`; `center_elev_m` is the point itself."""
    lat: float
    lon: float
    center_elev_m: float
    azimuths: tuple[float, ...]
    ranges_km: tuple[float, ...]
    grid: list[list[float]]
    relief_m: float
    landform: str                              # valley/basin | ridge/exposed | sloped | flat
    upslope: list[str]                         # compass dirs the terrain rises toward
    downslope: list[str]                       # compass dirs the terrain falls toward
    max_rise: tuple[str, float, float] | None  # (bearing, delta_m, range_km)
    max_drop: tuple[str, float, float] | None
    coast: tuple[float, str] | None            # (dist_km, bearing) of nearest OCEAN, if <=150 km


def _get_json(url: str) -> dict:
    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    finally:
        _last_request = time.monotonic()


def _elevations(points: list[tuple[float, float]]) -> list[float]:
    """Batched DEM elevation (m) for each (lat, lon), order preserved. Chunks to the API cap."""
    out: list[float] = []
    for i in range(0, len(points), _MAX_POINTS):
        chunk = points[i:i + _MAX_POINTS]
        q = urllib.parse.urlencode({
            "latitude": ",".join(f"{la:.5f}" for la, _ in chunk),
            "longitude": ",".join(f"{lo:.5f}" for _, lo in chunk),
        })
        doc = _get_json(f"{_ENDPOINT}?{q}")
        out.extend(float(e) for e in doc["elevation"])
    return out


def _nearest_ocean(lat: float, lon: float) -> tuple[float, str] | None:
    """Nearest OCEAN point within 150 km as (dist_km, bearing), scanning rings outward.
    Returns None if all sampled points are land (inland station). Lakes read as land."""
    for rng in _COAST_RANGES_KM:
        hits = []
        for az in _AZIMUTHS:
            dlat, dlon = destination(lat, lon, az, rng)
            if bool(globe.is_ocean(dlat, dlon)):
                hits.append((haversine_km(lat, lon, dlat, dlon), compass16(az)))
        if hits:
            d, brg = min(hits, key=lambda t: t[0])
            return round(d, 0), brg
    return None


def _describe(center: float, grid: list[list[float]]) -> dict:
    """Derive relief / landform / upslope-downslope from the sampled grid."""
    flat = [center] + [e for row in grid for e in row]
    relief = max(flat) - min(flat)

    # Classify each azimuth ONCE by its dominant (largest-magnitude) signed delta across the
    # ranges, so upslope and downslope are disjoint -- a direction that both rises nearby and
    # falls farther out is called by whichever is stronger, not listed in both.
    rises: list[tuple[str, float, float]] = []   # (dir, delta, range)
    drops: list[tuple[str, float, float]] = []
    n_up = n_down = 0
    for i, az in enumerate(_AZIMUTHS):
        deltas = [(grid[i][j] - center, _RANGES_KM[j]) for j in range(len(_RANGES_KM))]
        dom = max(deltas, key=lambda t: abs(t[0]))       # dominant tendency for this azimuth
        if dom[0] >= _SLOPE_THRESH_M:
            rises.append((compass16(az), dom[0], dom[1]))
            n_up += 1
        elif dom[0] <= -_SLOPE_THRESH_M:
            drops.append((compass16(az), dom[0], dom[1]))
            n_down += 1

    if relief < _SLOPE_THRESH_M:
        landform = "flat"
    elif n_up >= 10 and n_up > n_down:
        landform = "valley/basin"
    elif n_down >= 10 and n_down > n_up:
        landform = "ridge/exposed"
    else:
        landform = "sloped"

    rises.sort(key=lambda t: -t[1])
    drops.sort(key=lambda t: t[1])
    max_rise = (rises[0][0], round(rises[0][1], 0), rises[0][2]) if rises else None
    max_drop = (drops[0][0], round(drops[0][1], 0), drops[0][2]) if drops else None
    return {
        "relief_m": round(relief, 0),
        "landform": landform,
        "upslope": [d for d, _dl, _r in rises],
        "downslope": [d for d, _dl, _r in drops],
        "max_rise": max_rise,
        "max_drop": max_drop,
    }


def _cache_file(lat: float, lon: float) -> Path:
    return _CACHE_DIR / f"terrain_{lat:.3f}_{lon:.3f}.json"


def sample(lat: float, lon: float, *, use_cache: bool = False) -> TerrainProfile:
    """Fetch + derive the terrain picture around (lat, lon). One batched elevation call
    (center + 16x4 grid); coastline is local (global-land-mask, no network). With
    use_cache, the elevation grid is read from / written to data/terrain/ (permanent --
    terrain never changes); the coast + descriptors are recomputed (cheap)."""
    cf = _cache_file(lat, lon)
    if use_cache and cf.exists():
        cached = json.loads(cf.read_text())
        center, grid = cached["center"], cached["grid"]
    else:
        points = [(lat, lon)]
        for az in _AZIMUTHS:
            for rng in _RANGES_KM:
                points.append(destination(lat, lon, az, rng))
        elev = _elevations(points)
        center = elev[0]
        it = iter(elev[1:])
        grid = [[next(it) for _ in _RANGES_KM] for _ in _AZIMUTHS]
        if use_cache:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({"center": center, "grid": grid}))

    desc = _describe(center, grid)
    return TerrainProfile(
        lat=lat, lon=lon, center_elev_m=round(center, 0),
        azimuths=_AZIMUTHS, ranges_km=_RANGES_KM, grid=grid,
        coast=_nearest_ocean(lat, lon), **desc,
    )


# --- Relief map (OpenTopoMap tiles) -------------------------------------------------------

def _tile_km(z: int, lat: float) -> float:
    """Ground width (km) of one Web-Mercator tile at zoom z and latitude lat."""
    return 40075.0 * math.cos(math.radians(lat)) / (2 ** z)


def _pick_zoom(box_km: float, lat: float) -> int:
    """Highest zoom (most detail) whose box fits within the per-axis tile budget."""
    for z in range(13, 5, -1):
        if math.ceil(box_km / _tile_km(z, lat)) + 1 <= _MAP_MAX_TILES:
            return z
    return 8


def _deg2tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Fractional (x, y) tile coordinates of a lat/lon at zoom z."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _fetch_tile(z: int, x: int, y: int) -> bytes:
    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    url = _OTM_TILE.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    finally:
        _last_request = time.monotonic()


def _tile_cache(z: int, x: int, y: int) -> Path:
    return _CACHE_DIR / "otm" / str(z) / str(x) / f"{y}.png"


def _tile_image(z: int, x: int, y: int, use_cache: bool) -> Image.Image:
    """One OTM tile as a PIL image, via the permanent disk cache when use_cache. A missing
    tile (edge of coverage / transient error) becomes a neutral gray block, not a crash."""
    cf = _tile_cache(z, x, y)
    if use_cache and cf.exists():
        return Image.open(io.BytesIO(cf.read_bytes())).convert("RGB")
    raw = _fetch_tile(z, x, y)
    if use_cache:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_bytes(raw)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def relief_map(lat: float, lon: float, *, use_cache: bool = True) -> bytes:
    """Stitched OpenTopoMap relief map centered on (lat, lon), cropped to a 100-mile box
    (50-mi radius), with a station marker and 25/50-mile range rings drawn on. Returns PNG
    bytes. Tiles are cached permanently under data/terrain/otm (terrain never changes, and the
    OTM server is a small volunteer project -- fetch once, be polite). Caller cites the
    attribution in its receipt."""
    box_km = 2 * _MAP_RADIUS_MI * _KM_PER_MI
    z = _pick_zoom(box_km, lat)
    px_per_km = _TILE_PX / _tile_km(z, lat)
    cx, cy = _deg2tile(lat, lon, z)
    need = int(math.ceil(box_km * px_per_km / _TILE_PX)) + 2      # tiles per axis (+pad)
    x0, y0 = int(cx) - need // 2, int(cy) - need // 2

    canvas = Image.new("RGB", (need * _TILE_PX, need * _TILE_PX))
    for dx in range(need):
        for dy in range(need):
            try:
                tile = _tile_image(z, x0 + dx, y0 + dy, use_cache)
            except Exception:  # noqa: BLE001 -- a missing tile is a gray block, not a crash
                tile = Image.new("RGB", (_TILE_PX, _TILE_PX), (205, 205, 205))
            canvas.paste(tile, (dx * _TILE_PX, dy * _TILE_PX))

    # crop to the exact box centered on the station
    spx, spy = (cx - x0) * _TILE_PX, (cy - y0) * _TILE_PX
    half = box_km * px_per_km / 2
    crop = canvas.crop((int(spx - half), int(spy - half), int(spx + half), int(spy + half)))

    # overlays: 25/50-mile range rings + station marker + labels
    d = ImageDraw.Draw(crop)
    c = crop.size[0] / 2
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:  # noqa: BLE001 -- default bitmap font if DejaVu is absent
        font = ImageFont.load_default()
    for mi in _MAP_RINGS_MI:
        rpx = mi * _KM_PER_MI * px_per_km
        d.ellipse([c - rpx, c - rpx, c + rpx, c + rpx], outline=(20, 20, 20), width=2)
        d.text((c + 3, c - rpx + 2), f"{mi} mi", fill=(20, 20, 20), font=font)
    d.line([(c - 12, c), (c + 12, c)], fill=(220, 0, 0), width=4)
    d.line([(c, c - 12), (c, c + 12)], fill=(220, 0, 0), width=4)
    d.text((6, 6), f"OpenTopoMap relief  {_OTM_ATTRIB}", fill=(0, 0, 0), font=font)

    # JPEG, not PNG: a topo raster is ~2 MB as PNG but ~300 KB as JPEG, and the VLM reads it
    # fine (same as the satellite imagery). Downscale very large crops to keep the payload lean.
    if crop.size[0] > 1100:
        crop = crop.resize((1100, 1100), Image.LANCZOS)
    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=85)
    return out.getvalue()
