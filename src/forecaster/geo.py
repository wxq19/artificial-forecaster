"""Shared spatial primitives -- great-circle distance, bearing, nearest-N ranking.

Pure geography, no dependencies beyond the stdlib. This is the ONE place "distance +
direction" lives; both the neighbor-observations roster (neighbors.py) and the terrain
tool (terrain.py) build on it, and imagery.nearest_radar is the same shape. The formula
matches imagery._haversine_km (WGS84 mean radius 6371.0088 km) so distances agree across
the codebase.
"""

from math import asin, atan2, cos, degrees, radians, sin, sqrt

__all__ = [
    "haversine_km", "bearing_deg", "compass16", "nearest_n", "destination",
]

_EARTH_KM = 6371.0088

# 16-point compass, clockwise from due north.
_COMPASS16 = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * _EARTH_KM * asin(sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing FROM point 1 TO point 2, degrees in [0, 360)."""
    p1, p2 = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(p2)
    y = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dlon)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


def compass16(deg: float) -> str:
    """Degrees -> nearest 16-point compass label (e.g. 315 -> 'NW')."""
    return _COMPASS16[int((deg % 360.0) / 22.5 + 0.5) % 16]


def destination(lat: float, lon: float, bearing: float, dist_km: float) -> tuple[float, float]:
    """The point `dist_km` from (lat, lon) along `bearing` degrees, as (lat, lon).
    Spherical forward geodesic -- the inverse of haversine_km + bearing_deg; used to lay
    out a radial sampling grid around a station (terrain.py)."""
    d = dist_km / _EARTH_KM
    br = radians(bearing)
    p1 = radians(lat)
    p2 = asin(sin(p1) * cos(d) + cos(p1) * sin(d) * cos(br))
    l2 = radians(lon) + atan2(sin(br) * sin(d) * cos(p1), cos(d) - sin(p1) * sin(p2))
    return degrees(p2), (degrees(l2) + 540.0) % 360.0 - 180.0


def nearest_n(
    lat: float,
    lon: float,
    catalog: list[tuple[str, float, float]],
    n: int,
    max_km: float,
) -> list[tuple[str, float, str]]:
    """The n closest catalog points to (lat, lon) within max_km, nearest first.

    `catalog` is [(id, lat, lon), ...]; a point whose id equals none is kept (the caller
    excludes the home id if needed). Returns [(id, dist_km, bearing_label), ...]. Straight-
    line only -- a candidate list, not a guarantee of relevance; max_km is the guard.
    """
    scored: list[tuple[float, str, str]] = []
    for cid, clat, clon in catalog:
        dkm = haversine_km(lat, lon, clat, clon)
        if dkm > max_km:
            continue
        brg = compass16(bearing_deg(lat, lon, clat, clon))
        scored.append((dkm, cid, brg))
    scored.sort(key=lambda t: t[0])
    return [(cid, round(dkm, 1), brg) for dkm, cid, brg in scored[:n]]
