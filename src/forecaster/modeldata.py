"""GRIBStream model-data ORCHESTRATOR -- the spatial pre-fetch into the coordinate-indexed
archive. Sibling to iem.py / climo.py: it uses the gribstream network client + the store
persistence seam and owns NEITHER (no SQL, no matplotlib, no urllib of its own).

What it does, once per station/cycle:
  1. Build the coordinate list for a site -- the station itself, its fetchable METAR
     neighbors (neighbors.py, so model data collocates with get_nearby_obs), and a coarse
     fixed upstream grid for advection where stations are sparse.
  2. Build the per-model variable bundle -- a surface set for ALL coordinates, and (GFS/HRRR
     only) a pressure-level HAZARD set for the site + grid used by the icing/turbulence scan.
  3. Pull each model with gribstream.fetch_points(as_of=issue_time) and flatten the rows into
     the model_data archive under a single write_lock hold.

Cost model (memorize): credits = valid_times * variables * ceil(coords/500). Coordinates
sit INSIDE ceil(/500), so <=500 points cost the SAME as one -- points are effectively free;
credits accrue on HOURS (subsample via a times grid) and VARIABLES/LEVELS. So we are generous
on points and disciplined on the time grid + hazard levels.

Leakage: a model FORECAST issued before the TAF issue time was legitimately available (the
human had it too), so the only guard is run <= issue_time -- enforced by pulling with
as_of = issue_time. The archive then needs no valid_time read-cutoff (see store.model_data).
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from forecaster import awc, gribstream, neighbors, store
from forecaster.config import settings

MODELS = gribstream.MODELS  # ("gfs", "hrrr", "nbm")

# GRIBStream's HRRR and NBM are CONUS-domain; an OCONUS site (Alaska/Japan) gets only all-null
# rows from them -- which still BILL (credits scale with returned valid_times, not values) and
# archive as a misleading all-`--` table. GFS is global. So drop the CONUS-only models when no
# coordinate falls in the contiguous-US box; verified live 2026-07-19 (PAED: GFS 171/171 non-
# null, HRRR + NBM 0/171).
_CONUS_BBOX = (24.0, 50.0, -125.0, -66.0)   # lat_min, lat_max, lon_min, lon_max
_CONUS_ONLY_MODELS = ("hrrr", "nbm")


def _in_conus(lat: float, lon: float) -> bool:
    la0, la1, lo0, lo1 = _CONUS_BBOX
    return la0 <= lat <= la1 and lo0 <= lon <= lo1


def _applicable_models(coords: list, models: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (kept, dropped): drop the CONUS-only models when NO coordinate is in the
    contiguous US (an OCONUS request), else keep everything. A mixed CONUS+OCONUS batch keeps
    all models (the CONUS coords still need them; the OCONUS nulls are the lesser evil)."""
    if any(_in_conus(c[0], c[1]) for c in coords):
        return models, ()
    dropped = tuple(m for m in models if m in _CONUS_ONLY_MODELS)
    return tuple(m for m in models if m not in _CONUS_ONLY_MODELS), dropped

# Response columns that are NOT requested variables (mirrors gribstream._TS_COLS+_META_COLS).
_SKIP_COLS = {"forecasted_at", "forecasted_time", "lat", "lon", "name", "member"}

# --- per-model surface variable bundle (names DIFFER across models) --------------------
# GFS + HRRR share u/v wind + GUST@surface; their MSLP name differs (GFS PRMSL vs HRRR
# MSLMA -- probed live 2026-07-17, see scripts/probe_hrrr_mslp.py). NBM is speed/dir wind,
# GUST@10m, and has NO sea-level-pressure field.
_MSLP = {
    "gfs": ("PRMSL", "mean sea level"),
    "hrrr": ("MSLMA", "mean sea level"),
    "nbm": None,
    "ifsoper": ("msl", "sfc"),                # ECMWF native shortname (see IFS notes below)
}

# ECMWF IFS uses NATIVE shortnames (2t/2d/10u/10v/msl/tcc @ level 'sfc'; pressure levels as
# 'pl 850'), NOT the GFS GRIB2 style, verified from the official model page
# (gribstream.com/models/ifsoper). ENABLED 2026-07-23 -- now in the default MODELS. Handled:
#   - IFS `tcc` (total cloud) is a FRACTION 0-1 vs GFS/NBM TCDC PERCENT 0-100 -> scaled to
#     percent at ingest in `_normalize`, so every reader sees the tcdc alias as percent.
#   - IFS is 3-HOURLY; off-grid valid times are not billed (gribstream.com pricing), so an
#     hourly request just returns the native 3-hourly subset -- no separate time grid needed.
# STILL a known gap (not blocking): IFS has NO surface visibility/gust/ceiling and NO CAPE/CIN/
# HLCY/CLMR, so it contributes no hazard bundle (_hazard_vars returns [] for it) -- icing/
# turbulence stays a GFS+HRRR product. A future IFS hazard bundle could add icing (t/r) + shear
# (u/v/w) at 'pl <hPa>' levels only.
# Global 0.25deg, runs 00/06/12/18Z, out to 360 h -> works OCONUS too (unlike HRRR/NBM).
_IFS_ENABLED = True


def _gefs_vars() -> list[gribstream.Var]:
    """GEFS ensemble surface fields -- the aviation set (verified available 2026-07-23):
    T/Td, u/v wind, GUST, MSLP, TCDC, VIS, and ceiling as HGT@cloud ceiling (same as GFS).
    Each field is fetched across members, so a reader can build a probability per hour."""
    V = gribstream.Var
    return [V("TMP", "2 m above ground", "t2m"), V("DPT", "2 m above ground", "td2m"),
            V("UGRD", "10 m above ground", "u10"), V("VGRD", "10 m above ground", "v10"),
            V("GUST", "surface", "gust"), V("PRMSL", "mean sea level", "mslp"),
            V("TCDC", "entire atmosphere", "tcdc"), V("VIS", "surface", "vis"),
            V("HGT", "cloud ceiling", "ceil")]


def _surface_vars(model: str) -> list[gribstream.Var]:
    V = gribstream.Var
    if model == "gefsatmos":
        return _gefs_vars()
    if model == "nbm":
        return [V("TMP", "2 m above ground", "t2m"), V("DPT", "2 m above ground", "td2m"),
                V("WIND", "10 m above ground", "wind"), V("WDIR", "10 m above ground", "wdir"),
                V("GUST", "10 m above ground", "gust"), V("TCDC", "surface", "tcdc"),
                V("VIS", "surface", "vis"), V("CEIL", "cloud ceiling", "ceil")]
    if model == "ifsoper":
        # Verified names (gribstream.com/models/ifsoper). NB tcc is a 0-1 FRACTION -- handle
        # before enabling (see notes above). No gust/vis/ceiling on IFS.
        return [V("2t", "sfc", "t2m"), V("2d", "sfc", "td2m"),
                V("10u", "sfc", "u10"), V("10v", "sfc", "v10"),
                V("msl", "sfc", "mslp"), V("tcc", "sfc", "tcdc")]
    vs = [V("TMP", "2 m above ground", "t2m"), V("DPT", "2 m above ground", "td2m"),
          V("UGRD", "10 m above ground", "u10"), V("VGRD", "10 m above ground", "v10"),
          V("GUST", "surface", "gust")]
    mslp = _MSLP.get(model)
    if mslp:
        vs.append(V(mslp[0], mslp[1], "mslp"))
    vs += [V("TCDC", "entire atmosphere", "tcdc"), V("VIS", "surface", "vis"),
           V("HGT", "cloud ceiling", "ceil")]
    return vs


# --- per-model hazard (pressure-level) bundle -- GFS/HRRR only -------------------------
# Icing needs T + RH per level (GFS adds CLMR cloud-liquid = supercooled-water confirmation);
# turbulence needs deep-layer wind (shear), omega (ascent), and CAPE/CIN (+ GFS helicity).
ICE_LEVELS = ("650 mb", "600 mb", "550 mb", "500 mb", "450 mb", "400 mb")
SHEAR_LEVELS = ("850 mb", "500 mb", "300 mb")
VVEL_LEVELS = ("700 mb", "500 mb", "300 mb")


def _lvl_key(hpa: int) -> str:
    """Alias suffix for a pressure level. Kept as the plain integer so the pre-existing
    hazard aliases (t650, u850, w500...) are unchanged -- the profile bundle reuses the SAME
    alias namespace, so one set of archived rows serves both readers. NB the old code sliced
    `"850 mb"[:3]`, which would have collided 1000 mb onto '100'; this does not."""
    return str(hpa)


def _pl(model: str, hpa: int) -> str:
    """Pressure-level string in the model's own dialect: GFS/HRRR take GRIB2 '<n> mb',
    ECMWF IFS takes native 'pl <n>' (verified live 2026-07-28)."""
    return f"pl {hpa}" if model == "ifsoper" else f"{hpa} mb"


# --- vertical profile bundle (model soundings) -----------------------------------------
# Replaces the BUFKIT forecast-sounding source (fcstsounding.py, removed 2026-07-28). BUFKIT
# was free but gave a fixed North-America station list, no IFS/NBM/ensemble, a third-party
# posting dependency, and a parser that silently corrupted 4 of its 5 models. GRIBStream
# gives the exact station lat/lon, the same asOf leakage guard as everything else here, and
# IFS -- the only model reaching our OCONUS sites with independent guidance.
#
# Levels PROBED LIVE 2026-07-28 (KWRI): GFS and HRRR both serve all 20 standard levels with
# TMP/RH/UGRD/VGRD/HGT (HRRR also has DPT; GFS does not, so dewpoint is derived from RH for
# both, keeping the two models directly comparable). IFS serves a 12-level SUBSET and is
# boundary-layer sparse -- 1000 -> 925 -> 850 with no 950/900, which is exactly where TAF
# ceilings and inversions live. Good aloft, weak where it matters most: say so in the receipt.
PROFILE_LEVELS = (1000, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
                  450, 400, 350, 300, 250, 200, 150, 100)
_IFS_PROFILE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100)
# NBM has no pressure levels at all and never will, so it is surface-only by construction.
PROFILE_MODELS = ("gfs", "hrrr", "ifsoper")


def profile_levels(model: str) -> tuple[int, ...]:
    """Pressure levels this model actually serves, highest pressure first (surface-first)."""
    if model == "ifsoper":
        return _IFS_PROFILE_LEVELS
    return PROFILE_LEVELS if model in PROFILE_MODELS else ()


def _profile_vars(model: str) -> list[gribstream.Var]:
    """T / RH / u / v / height at every level this model serves. Five variables per level is
    the whole cost driver: credits = valid_times * variables, so the level ladder and the
    time step are the two levers, never the point count."""
    levels = profile_levels(model)
    if not levels:
        return []
    V = gribstream.Var
    t, rh, u, v, hgt = (("t", "r", "u", "v", "gh") if model == "ifsoper"
                        else ("TMP", "RH", "UGRD", "VGRD", "HGT"))
    vs: list[gribstream.Var] = []
    for hpa in levels:
        lv, k = _pl(model, hpa), _lvl_key(hpa)
        vs += [V(t, lv, f"t{k}"), V(rh, lv, f"rh{k}"), V(u, lv, f"u{k}"),
               V(v, lv, f"v{k}"), V(hgt, lv, f"hgt{k}")]
    return vs


def _hazard_vars(model: str, *, profiles: bool = True) -> list[gribstream.Var]:
    """Icing/turbulence extras. NBM is surface-only; IFS carries no CAPE/CIN/HLCY/CLMR/VVEL,
    so icing/turbulence stays a GFS+HRRR product (IFS still contributes a PROFILE, above).

    When the profile bundle is on it already covers T/RH/u/v at every level, so this returns
    only what the profile does NOT: cloud liquid, vertical velocity, and the convective
    scalars. The aliases are identical either way, so `_fmt_hazard_scan` is unaffected."""
    if model in ("nbm", "ifsoper"):
        return []
    V = gribstream.Var
    vs: list[gribstream.Var] = []
    for lv in ICE_LEVELS:
        p = _lvl_key(int(lv[:3]))
        if not profiles:
            vs += [V("TMP", lv, f"t{p}"), V("RH", lv, f"rh{p}")]
        if model == "gfs":
            vs.append(V("CLMR", lv, f"clw{p}"))
    if not profiles:
        for lv in SHEAR_LEVELS:
            p = _lvl_key(int(lv[:3]))
            vs += [V("UGRD", lv, f"u{p}"), V("VGRD", lv, f"v{p}")]
    for lv in VVEL_LEVELS:
        vs.append(V("VVEL", lv, f"w{_lvl_key(int(lv[:3]))}"))
    if model == "gfs":
        vs += [V("CAPE", "surface", "cape"), V("CIN", "surface", "cin"),
               V("HLCY", "3000-0 m above ground", "hlcy")]
    else:  # hrrr
        vs += [V("CAPE", "180-0 mb above ground", "cape"),
               V("CIN", "180-0 mb above ground", "cin")]
    return vs


# --- coordinate builder ---------------------------------------------------------------
# A fixed ring grid around the site (points are free <=500, so this fills advection gaps
# where METAR neighbors are sparse). Bearings + radii are CONFIGURABLE; longitude offset is
# scaled by cos(lat) so a ring stays roughly circular in km. Default is deliberately DENSE
# (every 30 deg) since omnidirectional coverage costs nothing and the model reads its own
# wind to pick the relevant upwind direction from the gradient.
GRID_BEARINGS_DEG = tuple(range(0, 360, 30))   # 12 compass points (denser default)
GRID_RADII_DEG = (0.5, 1.0, 1.5)               # ~55/110/165 km at the equator
# Flow-relative UPSTREAM densification (opt-in): extra points reaching FARTHER out along the
# prevailing-wind sector, so what is advecting in is sampled at longer range. Placed at the
# flow-from bearing +/- the spread. Named u<brg>_<r> to distinguish from the base ring.
UPSTREAM_RADII_DEG = (2.0, 3.0)                # ~220/330 km at the equator
UPSTREAM_SPREAD_DEG = (-30, 0, 30)            # 3 bearings straddling the upwind direction

# 16-point compass sector -> degrees (the climo product's dir_mode_sector is a wind-FROM
# sector; upstream is TOWARD that bearing from the site).
_SECTOR_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def _offset_point(lat: float, lon: float, bearing_deg: float, radius_deg: float) -> tuple:
    coslat = max(math.cos(math.radians(lat)), 0.2)
    dlat = radius_deg * math.cos(math.radians(bearing_deg))
    dlon = radius_deg * math.sin(math.radians(bearing_deg)) / coslat
    return (round(lat + dlat, 4), round(lon + dlon, 4))


def _grid_points(lat: float, lon: float, *,
                 flow_from: float | None = None) -> list[tuple[float, float, str]]:
    """Ring grid around (lat, lon). If `flow_from` (wind-from bearing, deg) is given, ALSO add
    upstream points along that sector at extended radii (the flow-relative densification)."""
    out: list[tuple[float, float, str]] = []
    seen: set = set()
    for r in GRID_RADII_DEG:
        for b in GRID_BEARINGS_DEG:
            pt = _offset_point(lat, lon, b, r)
            if pt not in seen:
                seen.add(pt)
                out.append((pt[0], pt[1], f"g{b:03d}_{int(r * 10):02d}"))
    if flow_from is not None:
        for spread in UPSTREAM_SPREAD_DEG:
            b = (flow_from + spread) % 360
            for r in UPSTREAM_RADII_DEG:
                pt = _offset_point(lat, lon, b, r)
                if pt not in seen:
                    seen.add(pt)
                    out.append((pt[0], pt[1], f"u{int(round(b)) % 360:03d}_{int(r * 10):02d}"))
    return out


def site_coord(station: str) -> tuple[float, float, str]:
    """(lat, lon, ICAO) for a station via the AWC catalog (resolves CONUS + OCONUS)."""
    lat, lon = awc.station_latlon(station)
    return (round(lat, 4), round(lon, 4), station.upper())


def coords_for(station: str, *, include_grid: bool = True,
               flow_from: float | None = None) -> list[tuple[float, float, str]]:
    """The full coordinate set for a station: site + fetchable neighbors + ring grid (+ the
    flow-relative upstream points when `flow_from` is given), deduped. PURE GEOMETRY: the
    same inputs always yield the same list, so prefetch and collect.py's copy stay in lock-
    step (both go through station_coords, which resolves flow_from identically). Cap 500."""
    site = site_coord(station)
    coords: list[tuple[float, float, str]] = [site]
    seen = {(site[0], site[1])}
    for nb in neighbors.neighbors_of(station):
        icao, _, _, _, lat, lon = nb
        key = (round(lat, 4), round(lon, 4))
        if key not in seen:
            seen.add(key)
            coords.append((key[0], key[1], icao))
    if include_grid:
        for lat, lon, name in _grid_points(site[0], site[1], flow_from=flow_from):
            if (lat, lon) not in seen:
                seen.add((lat, lon))
                coords.append((lat, lon, name))
    return coords[:500]


def hazard_coords(station: str, *, flow_from: float | None = None,
                  include_grid: bool = False) -> list[tuple[float, float, str]]:
    """Coordinates the pressure-level bundle is pulled for. SITE COLUMN ONLY by default.

    CONFIG B, decided 2026-07-28. Points are free below 500 -- but only 500 IN TOTAL across
    the network, and the level bundle is the expensive tier (17 valid times x up to 112
    variables per model). At 71 stations the ring grid pushed level coordinates to 2,627 =
    6 chunks, and the level tier is ~84% of the bill: ~29,900 of 33,630 credits per pull.

    It bought nothing. NOTHING READS PRESSURE LEVELS OFF-SITE IN PRACTICE: `get_fcst_sounding`
    passes `None` and so always resolves to the station itself, and `get_nearby_model_data` --
    the only tool that reads every point -- reads one SURFACE alias at a time. Dropping the
    grid here takes the pull from 33,630 to 10,085 credits.

    ONE tool can still ASK off-site: `get_hazard_scan` forwards its `location` argument, so a
    model naming a neighbour resolves it (it is in the SURFACE set) and then finds no levels.
    That path is handled explicitly in `tools._fmt_hazard_scan`, which says the level bundle
    is site-only and sends the model back to the station -- a named limit, not a blank panel.

    The SURFACE grid is untouched (see `coords_for`): that is the one the agent actually
    sees, and it is what shows an advecting front before it reaches the neighbour obs.

    `include_grid=True` restores the old behaviour for a station where upstream structure
    ALOFT is genuinely wanted -- and levels stay recoverable later via `asOf`, so this is a
    reversible economy, unlike the live-fetched imagery."""
    site = site_coord(station)
    if not include_grid:
        return [site]
    return [site] + _grid_points(site[0], site[1], flow_from=flow_from)


# --- steering flow (the upwind orientation) -------------------------------------------
# We densify UPSTREAM along the STEERING FLOW -- the deep-layer mean wind that actually
# advects weather into the terminal -- NOT the surface wind, and from CURRENT model data,
# NOT climatology. Deep-layer mean = vector average of the u/v wind at these levels. GFS is
# the reference: global + full pressure levels, so it works CONUS and OCONUS (HRRR is
# CONUS-only, NBM has no pressure levels). Two-pass to break the chicken-and-egg (need the
# wind to place the samples, get the wind from a sample): pass 1 probes the SITE column,
# pass 2 fetches the oriented grid. Climo prevailing wind is the FALLBACK when no pressure
# data is available (e.g. an unbuilt archive), never the primary.
_STEER_LEVELS = (850, 700, 500)   # deep-layer mean; vector-averaged u/v across these


def _steering_bearing(u_by_lvl: dict, v_by_lvl: dict) -> float | None:
    """Vector-mean wind across _STEER_LEVELS -> the wind-FROM bearing (deg), or None if no
    level has data (or the mean is calm)."""
    us = [u_by_lvl[lv] for lv in _STEER_LEVELS if u_by_lvl.get(lv) is not None]
    vs = [v_by_lvl[lv] for lv in _STEER_LEVELS if v_by_lvl.get(lv) is not None]
    if not us:
        return None
    um, vm = sum(us) / len(us), sum(vs) / len(vs)
    if um == 0.0 and vm == 0.0:
        return None
    return (270.0 - math.degrees(math.atan2(vm, um))) % 360.0


def _steer_vars() -> list[gribstream.Var]:
    V = gribstream.Var
    out: list[gribstream.Var] = []
    for lv in _STEER_LEVELS:
        out += [V("UGRD", f"{lv} mb", f"u{lv}"), V("VGRD", f"{lv} mb", f"v{lv}")]
    return out


def _steering_probe(lat: float, lon: float, station: str, as_of: datetime, *,
                    use_cache: bool) -> tuple[float | None, list[dict], int]:
    """PASS 1 (live): fetch the SITE's GFS deep-layer winds at the issue anchor and derive the
    steering bearing. Returns (bearing, rows_to_archive, credits). The rows are archived under
    loc_id=station so collect.py's copy recomputes the IDENTICAL bearing offline. Leakage-safe
    (as_of pins the run). GFS-only -> works OCONUS. A fetch failure -> (None, [], 0)."""
    anchor = as_of.replace(minute=0, second=0, microsecond=0)
    try:
        ts = gribstream.fetch_points(
            "gfs", [(round(lat, 4), round(lon, 4), station.upper())], _steer_vars(),
            times=[anchor], as_of=as_of, use_cache=use_cache)
    except ValueError:
        return None, [], 0
    if not ts.rows:
        return None, [], ts.charged
    r = ts.rows[0]
    u = {lv: r.get(f"u{lv}") for lv in _STEER_LEVELS}
    v = {lv: r.get(f"v{lv}") for lv in _STEER_LEVELS}
    rows = _flatten("gfs", ts, as_of=as_of, fetched_at=_utcnow())
    return _steering_bearing(u, v), rows, ts.charged


def _steering_from_archive(lat: float, lon: float, as_of: datetime,
                           db_path: str | None) -> float | None:
    """Recompute the steering bearing from ARCHIVED site winds (no fetch) -- deterministic, so
    collect.py's copy reproduces exactly the bearing prefetch used. None if the winds aren't
    archived yet."""
    anchor = as_of.replace(minute=0, second=0, microsecond=0)
    aliases = [f"{c}{lv}" for lv in _STEER_LEVELS for c in ("u", "v")]
    try:
        con = store.connect(db_path or settings.db_path, read_only=True)
    except Exception:  # noqa: BLE001 -- no DB yet
        return None
    try:
        rows = store.model_data_series(con, "gfs", lat, lon, start=anchor, end=anchor,
                                       variables=aliases)
    except Exception:  # noqa: BLE001 -- model_data table absent
        return None
    finally:
        con.close()
    u = {int(r["variable"][1:]): r["value"] for r in rows if r["variable"].startswith("u")}
    v = {int(r["variable"][1:]): r["value"] for r in rows if r["variable"].startswith("v")}
    return _steering_bearing(u, v)


def _flow_from_climo(station: str, month: int, hour: int, db_path: str | None) -> float | None:
    """FALLBACK: the climatological prevailing wind-FROM bearing at (month, hour) from the
    climo product -- surface, and a long-term average, so only used when current steering data
    is unavailable. None if climo isn't built (-> no orientation)."""
    try:
        con = store.connect(db_path or settings.db_path, read_only=True)
    except Exception:  # noqa: BLE001 -- no DB yet -> no flow data
        return None
    try:
        hours = store.climo_hours(con, station, month)
    except Exception:  # noqa: BLE001 -- climo tables absent
        return None
    finally:
        con.close()
    if not hours:
        return None
    row = min(hours, key=lambda h: abs(((h["hour_utc"] - hour + 12) % 24) - 12))
    return _SECTOR_DEG.get((row.get("dir_mode_sector") or "").upper())


def _resolve_flow(station: str, lat: float, lon: float, as_of: datetime | None,
                  db_path: str | None, flow_relative: bool | None) -> float | None:
    """The upwind bearing for the COPY path (collect.py): steering from the ARCHIVED site
    winds, climo as fallback. Deterministic (no fetch), so it reproduces exactly what prefetch
    oriented on. None when flow-relative is off or nothing is available."""
    if flow_relative is None:
        flow_relative = settings.model_data_flow_relative
    if not flow_relative or as_of is None:
        return None
    b = _steering_from_archive(lat, lon, as_of, db_path)
    if b is None:
        b = _flow_from_climo(station, as_of.month, as_of.hour, db_path)
    return b


def station_coords(station: str, *, as_of: datetime | None = None, db_path: str | None = None,
                   flow_relative: bool | None = None) -> list[tuple[float, float, str]]:
    """The station's SURFACE coordinate set (site + neighbors + grid, incl. flow-relative
    upstream points). The COPY path: resolves the steering bearing from the ARCHIVE so
    collect.py gets exactly what prefetch fetched. Superset of the hazard set (hazard = site +
    grid only), so copying by this list captures both surface AND hazard rows."""
    site = site_coord(station)
    flow_from = _resolve_flow(station, site[0], site[1], as_of, db_path, flow_relative)
    return coords_for(station, flow_from=flow_from)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _time_grid(anchor: datetime, hours: int, step_h: int, back_h: int = 0) -> list[datetime]:
    """Valid-time subsample grid from anchor-back_h to anchor+hours (inclusive), stepping by
    step_h. The pre-anchor tail lets get_model_verification compare the archived forecast
    against the pre-issue obs already in the DB (no live fetch, leakage-safe)."""
    anchor = anchor.replace(minute=0, second=0, microsecond=0)
    start = anchor - timedelta(hours=back_h)
    n = (back_h + hours) // step_h
    return [start + timedelta(hours=step_h * k) for k in range(n + 1)]


def _normalize(model: str, alias: str, value):
    """Bring a raw model value onto the archive's per-alias unit convention, so a downstream
    reader can treat an alias identically across models. Only IFS needs it today: its `tcc`
    (total cloud) is a 0-1 FRACTION where GFS/NBM TCDC is a 0-100 PERCENT, and the tcdc alias
    is documented as percent. Scaling here keeps that contract at the one seam that writes the
    archive, instead of every formatter special-casing IFS."""
    if value is not None and model == "ifsoper" and alias == "tcdc":
        return value * 100.0
    return value


def _flatten(model: str, ts: gribstream.TimeSeries, *, as_of, fetched_at) -> list[dict]:
    """A TimeSeries -> model_data row dicts (one per row x variable column)."""
    var_cols = [c for c in ts.columns if c not in _SKIP_COLS]
    rows: list[dict] = []
    for r in ts.rows:
        run, valid = r.get("forecasted_at"), r.get("forecasted_time")
        lat, lon, loc = r.get("lat"), r.get("lon"), r.get("name")
        if run is None or valid is None or lat is None or lon is None:
            continue
        member = int(r.get("member") or 0)
        for v in var_cols:
            rows.append({
                "model": model, "run": run, "valid_time": valid,
                "lat": lat, "lon": lon, "loc_id": loc, "variable": v,
                "value": _normalize(model, v, r.get(v)), "member": member,
                "as_of": as_of, "fetched_at": fetched_at,
            })
    return rows


def _snapped_grid(anchor: datetime, hours: int, step_h: int) -> list[datetime]:
    """Valid-time grid SNAPPED to a 00Z-anchored multiple of `step_h`, extended so it still
    reaches anchor+hours.

    Coarse-cadence models exist ONLY at 00Z-anchored multiples -- IFS at 00/03/06...Z, GEFS
    likewise -- so a grid anchored on an odd issue hour (02Z -> 02/05/08...) matches nothing.
    The request then returns zero rows, zero credits and NO ERROR: a silently empty archive
    that looks like a successful pull. That failure has now happened twice, once for the IFS
    level bundle and once for GEFS in the consensus experiment, which is why both callers go
    through this one helper instead of each snapping its own grid.

    The extension matters as much as the snap: without rounding the span UP to a whole step,
    `_time_grid`'s floor division drops the partial step and the grid ends short of the
    horizon that was asked for."""
    a = anchor.replace(minute=0, second=0, microsecond=0)
    snapped = a.replace(hour=(a.hour // step_h) * step_h)
    span = math.ceil((hours + (a.hour - snapped.hour)) / step_h) * step_h
    return _time_grid(snapped, span, step_h)


def _chunk(coords: list, size: int = 500):
    """Split a coordinate list into <=size batches (the free-coordinate ceiling per request)."""
    for i in range(0, len(coords), size):
        yield coords[i:i + size]


# NATIVE surface cadence, in hours, for the 0-48h range we archive. Measured off the live
# archive 2026-07-29 by looking at the valid-time gaps each model actually RETURNED:
#   gfs [1,2]  hrrr [1,2]  nbm [2]  ifsoper [3]
# GFS/HRRR/NBM serve every hour asked for; IFS serves 3-hourly and NOTHING ELSE, so asking it
# for 28 two-hourly times bought 17 rows and paid for 28. A per-model step is what stops that.
# The value is the FINEST the model serves -- ask for less and we lose detail we are entitled
# to; ask for more and we pay for hours that never arrive.
NATIVE_STEP_H = {"gfs": 1, "hrrr": 1, "nbm": 1, "ifsoper": 3, "gefsatmos": 3}


def archive_cycle(as_of: datetime | None = None) -> datetime:
    """The 00/06/12/18Z run this pull is archiving.

    Distinct from `archive_run_and_as_of`, which returns None for GFS and IFS -- that function
    answers "must I pin as_of to make this model name a run", and for an already-6-hourly model
    the answer is no. But the RUN is knowable for every model, and it is the right anchor for a
    valid-time grid, so it gets its own helper rather than being inferred at each call site.

    `as_of=None` means now, matching prefetch_many/estimate_prefetch_many -- callers pass the
    raw CLI value through, which is None until those functions default it."""
    t = ((as_of or _utcnow()) - timedelta(hours=_ARCHIVE_POST_LAG_H)).replace(
        minute=0, second=0, microsecond=0)
    return t - timedelta(hours=t.hour % _ARCHIVE_CYCLE_H)


def _model_times(model: str, anchor: datetime, hours: int, step_h: int, hazard_step_h: int,
                 back_hours: int, *, run: datetime | None) -> tuple[list, list]:
    """(surface times, level times) for one model, anchored on its RUN where we know it.

    WHY THE RUN AND NOT THE CLOCK. The grid used to be built once from `as_of` -- the cron
    firing time -- before any model was considered. A 17:02Z cron gave a 17Z anchor, so we
    asked for valid times 17/19/21Z... which is a window around our own wall clock and has no
    meteorological meaning. It also matched none of IFS's 00Z-anchored 3-hourly times, which
    is the whole reason `_snapped_grid` had to exist: an unsnapped request returns zero rows,
    zero credits and NO error -- a silently empty archive that looks like a successful pull.

    Anchoring on the run removes that class of failure instead of guarding it. Runs are always
    00/06/12/18Z, every one a multiple of 3, so a coarse model lines up by construction. It
    also makes coverage a sentence you can state: the 06Z run to f048 covers a 12Z TAF's 30h
    validity with 12h spare and 6h of hindsight.

    THE PRE-ANCHOR TAIL BECOMES FREE. `back_hours` existed so get_model_verification could
    compare an archived forecast against obs already banked in the DB. A run-anchored ladder
    starts at f000, which is already in the past by the time we fetch at run+5h, so those
    hours arrive anyway -- and from the CORRECT run, instead of as the 9-row stub of an older
    one that made get_point_forecast label its table with a 24h-stale run. `back_hours` is
    still honoured when the run is unknown, so nothing regresses on that path.

    `step_h` is a CEILING, not the step: a caller asking for 2-hourly gets 2-hourly, but a
    caller asking for hourly gets the model's native cadence rather than a promise we cannot
    keep."""
    step = max(step_h, NATIVE_STEP_H.get(model, step_h))
    lvl_step = max(hazard_step_h, NATIVE_STEP_H.get(model, 1))
    if run is None:
        # Caller could not name a run (a bare prefetch outside the archive path). Keep the old
        # clock-anchored behaviour, snapped, so a coarse model still lines up.
        return (_time_grid(anchor, hours, step, back_h=back_hours),
                _snapped_grid(anchor, hours, lvl_step))
    # `hours` is measured FROM THE RUN, so a 48h horizon is f000..f048 -- the model's own
    # forecast ladder, not a 48h window around our clock. That is the whole point of anchoring
    # here: the 06Z run to f048 covers a 12Z TAF's 30h validity with 12h spare and 6h of
    # hindsight, which is a statement about the model rather than about the cron.
    return _time_grid(run, hours, step), _time_grid(run, hours, lvl_step)


def _dedupe(coord_lists: list[list]) -> list[tuple[float, float, str]]:
    """Union several coordinate lists, keeping the first name for a given (lat, lon)."""
    seen: dict = {}
    for lst in coord_lists:
        for la, lo, name in lst:
            seen.setdefault((round(la, 4), round(lo, 4)), (round(la, 4), round(lo, 4), name))
    return list(seen.values())


def model_coords(model: str, coords: list) -> list:
    """The subset of `coords` this model can actually answer for.

    A CONUS-only model asked about Ramstein does not error -- it returns a full set of rows
    with NULL values, and we pay for them and store them. MEASURED 2026-07-29 over the live
    archive: every OCONUS station held 5,715 HRRR rows and 624 NBM rows at **0% non-null**,
    totalling **844,422 null HRRR rows (36.4%) and 612,096 null NBM rows (33.8%)**.

    `_applicable_models` was written for this, but it only drops a CONUS-only model when NO
    coordinate in the batch is in CONUS -- and we union all 71 stations into one batch, so a
    CONUS coordinate is always present and the guard never fires. Its docstring calls the
    OCONUS nulls "the lesser evil"; the cost was simply never measured. It is 952 credits a
    pull, because credits are times x vars x ceil(coords/500) and 2,904 coords is 6 chunks
    where the 2,000 CONUS ones are 4.

    Filtering HERE rather than in `_applicable_models` keeps the model in play for the
    stations it covers instead of dropping it for everyone. It also removes the empty HRRR
    block `get_hazard_scan` prints at OCONUS stations -- that block exists only because the
    null rows are there to be found."""
    if model not in _CONUS_ONLY_MODELS:
        return coords
    return [c for c in coords if _in_conus(c[0], c[1])]


def _fetch_and_insert(
    surface_coords: list, hazard_coords_all: list, *,
    as_of: datetime, anchor: datetime, models: tuple[str, ...],
    hours: int, step_h: int, hazards: bool, hazard_step_h: int, back_hours: int,
    db_path: str | None, use_cache: bool, extra_rows: list[dict] | None = None,
    profiles: bool = True,
) -> tuple[int, int, int, list]:
    """Fetch surface + pressure-level data for the given coordinate unions across `models`,
    chunked to <=500 coords/request, and insert under one write_lock (with any `extra_rows`,
    e.g. the steering-probe columns). Returns (charged, flattened, inserted, notes).

    The pressure-level request is ONE MERGED BUNDLE: the sounding ladder (T/RH/u/v/height at
    every level the model serves) plus the hazard extras the ladder does not cover (cloud
    liquid, omega, CAPE/CIN/helicity). Merging matters because credits are
    valid_times x variables x ceil(coords/500) -- two requests over the same coordinates and
    the same time grid would bill the shared T/RH/u/v twice. Both readers use the same
    aliases, so one set of rows serves the skew-T and the icing/turbulence scan alike.

    COORDINATES AND TIMES ARE BOTH PER MODEL. The requests were always one per (model, chunk);
    what used to be shared was the coordinate union and the valid-time grid, which is how a
    CONUS-only model came to be billed for Japan. See `model_coords` and `_model_times`."""
    fetched_at = _utcnow()
    charged = 0
    to_insert: list[dict] = list(extra_rows or [])
    notes: list[str] = []
    for model in models:
        # Per-model as_of: hourly models are pinned to the synoptic cycle so the archived
        # cycle names a run. One as_of for every model was the old behaviour and it let
        # HRRR/NBM drift to whatever hourly run the cron happened to catch.
        run, model_as_of = archive_run_and_as_of(model, as_of)
        if run is not None:
            notes.append(f"{model} pinned to the {run:%Y-%m-%dT%HZ} synoptic run "
                         "(hourly model archived on the 6-hourly cycle)")
        # The GRID anchors on the cycle being archived, for EVERY model -- `run` above is only
        # non-None for the hourly models that also need as_of pinned.
        sfc_times, haz_times = _model_times(model, anchor, hours, step_h, hazard_step_h,
                                            back_hours, run=archive_cycle(as_of))
        model_surface = model_coords(model, surface_coords)
        model_levels = model_coords(model, hazard_coords_all)
        if not model_surface:
            notes.append(f"{model} skipped: no requested coordinate is inside its domain")
            continue
        if len(model_surface) < len(surface_coords):
            notes.append(f"{model} is CONUS-only: {len(model_surface)} of "
                         f"{len(surface_coords)} coords requested (the rest would be null)")
        for chunk in _chunk(model_surface):
            try:
                ts = gribstream.fetch_points(model, chunk, _surface_vars(model),
                                             times=sfc_times, as_of=model_as_of,
                                             use_cache=use_cache)
                charged += ts.charged
                to_insert += _flatten(model, ts, as_of=model_as_of, fetched_at=fetched_at)
            except ValueError as e:
                notes.append(f"{model} surface: {e}")
        level_vars = (_profile_vars(model) if profiles else [])
        if hazards:
            level_vars += _hazard_vars(model, profiles=profiles)
        if level_vars and model_levels:
            for chunk in _chunk(model_levels):
                try:
                    ts = gribstream.fetch_points(model, chunk, level_vars,
                                                 times=haz_times, as_of=model_as_of,
                                                 use_cache=use_cache)
                    charged += ts.charged
                    to_insert += _flatten(model, ts, as_of=model_as_of, fetched_at=fetched_at)
                except ValueError as e:
                    notes.append(f"{model} levels: {e}")

    with store.write_lock(db_path):
        con = store.connect(db_path or settings.db_path)
        try:
            store.init_model_data_schema(con)
            inserted = store.insert_model_data(con, to_insert)
        finally:
            con.close()
    return charged, len(to_insert), inserted, notes


# --- reading a vertical profile back out of the archive --------------------------------
# The read half of the sounding replacement: archived pressure-level rows -> the typed
# profile `charts.skewt` draws. Pure computation over store rows; no network, no matplotlib.

@dataclass
class FcstProfile:
    """A model forecast sounding for one valid time, surface-first, levels with missing data
    dropped. Plain lists/floats -- no units, no matplotlib (charts.py owns plotting). Field
    names are the ones charts.skewt reads; they outlived the BUFKIT source they were named
    for (drct/sknt/hght are GEMPAK spellings)."""
    station: str
    model: str
    run: datetime                 # cycle, naive UTC
    fhr: int
    valid: str                    # display stamp, e.g. '260728/1800'
    lat: float
    lon: float
    elev_m: float
    pres: list[float]             # hPa
    tmpc: list[float]             # C
    dwpc: list[float]             # C
    drct: list[float]             # deg
    sknt: list[float]             # kt
    hght: list[float]             # m
    indices: dict                 # CAPE/CINS/... where the model supplies them
    url: str


def _dewpoint_c(t_c: float, rh_pct: float) -> float:
    """Dewpoint from temperature + relative humidity, Magnus-Tetens. Done here rather than
    with MetPy because charts.py is the only module allowed to import MetPy, and doing it in
    one place keeps GFS (no DPT field) and HRRR (has one) directly comparable."""
    rh = min(max(rh_pct, 0.1), 100.0)
    a, b = 17.625, 243.04
    g = math.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return (b * g) / (a - g)


def _uv_to_dir_speed(u_ms: float, v_ms: float) -> tuple[float, float]:
    """u/v in m/s -> (meteorological wind-FROM direction in degrees, speed in knots)."""
    return (math.degrees(math.atan2(-u_ms, -v_ms)) % 360.0,
            math.hypot(u_ms, v_ms) * 1.943844)


_PROFILE_IDX = {"cape": "CAPE", "cin": "CINS", "hlcy": "HLCY"}


def build_profile(con, station: str, model: str, valid_time: datetime, *,
                  lat: float | None = None, lon: float | None = None) -> FcstProfile:
    """Assemble one archived vertical profile into a FcstProfile.

    Reads the pressure-level rows for `valid_time` at the station coordinate, keeps the
    freshest run present, and converts to the plotting units (K->C, RH->dewpoint, u/v->
    direction/speed). Raises ValueError -- which the tool layer turns into feedback, not a
    crash -- when the model carries no profile or the archive has no rows for that hour."""
    if model not in PROFILE_MODELS:
        raise ValueError(
            f"{model} carries no vertical profile (pressure levels exist for "
            f"{', '.join(PROFILE_MODELS)}; NBM is surface-only by construction)")
    if lat is None or lon is None:
        lat, lon, _ = site_coord(station)
    levels = profile_levels(model)
    wanted = [f"{p}{_lvl_key(hpa)}" for hpa in levels for p in ("t", "rh", "u", "v", "hgt")]
    wanted += list(_PROFILE_IDX)
    rows = store.model_data_series(con, model, lat, lon, start=valid_time, end=valid_time,
                                   variables=wanted)
    if not rows:
        raise ValueError(
            f"no archived {model.upper()} profile for {station.upper()} at "
            f"{valid_time:%Y-%m-%dT%H}Z (the prefetch may not cover this hour)")
    run = max(r["run"] for r in rows)
    vals = {r["variable"]: r["value"] for r in rows if r["run"] == run and r["value"] is not None}

    pres, tmpc, dwpc, drct, sknt, hght = [], [], [], [], [], []
    for hpa in levels:                       # PROFILE_LEVELS is already surface-first
        k = _lvl_key(hpa)
        t, rh, u, v, z = (vals.get(f"t{k}"), vals.get(f"rh{k}"),
                          vals.get(f"u{k}"), vals.get(f"v{k}"), vals.get(f"hgt{k}"))
        if None in (t, rh, u, v, z):         # a partial level would distort the plot
            continue
        t_c = t - 273.15 if t > 100.0 else t            # archived K for GFS/HRRR and IFS alike
        d, s = _uv_to_dir_speed(u, v)
        pres.append(float(hpa))
        tmpc.append(t_c)
        dwpc.append(min(_dewpoint_c(t_c, rh), t_c))     # Td can never exceed T
        drct.append(d)
        sknt.append(s)
        hght.append(z)
    if len(pres) < 3:
        raise ValueError(
            f"archived {model.upper()} profile for {station.upper()} at "
            f"{valid_time:%Y-%m-%dT%H}Z has only {len(pres)} complete level(s)")
    idx = {label: vals[a] for a, label in _PROFILE_IDX.items() if a in vals}
    return FcstProfile(
        station=station.upper(), model=model, run=run,
        fhr=max(int((valid_time - run).total_seconds() // 3600), 0),
        valid=f"{valid_time:%y%m%d/%H%M}", lat=lat, lon=lon,
        elev_m=float("nan"), pres=pres, tmpc=tmpc, dwpc=dwpc, drct=drct,
        sknt=sknt, hght=hght, indices=idx,
        url=f"gribstream:{model}@{lat:.4f},{lon:.4f}",
    )


def profile_valid_times(con, station: str, model: str, *, lat=None, lon=None) -> list[datetime]:
    """Valid times that actually have profile rows archived, for snapping a request and for
    telling the model what it can ask for when it misses."""
    if lat is None or lon is None:
        lat, lon, _ = site_coord(station)
    levels = profile_levels(model)
    if not levels:
        return []
    probe = f"t{_lvl_key(levels[len(levels) // 2])}"     # a mid-level temperature
    return store.model_data_valid_times(con, model, lat, lon, variables=[probe])


# --- verification pulls -----------------------------------------------------------------
# Verification is a DIFFERENT query shape from the forecast archive, and the difference is
# not a parameter. Given a valid-time list plus asOf, GRIBStream returns, for each valid
# time, the forecast from the FRESHEST qualifying run -- exactly right for "what is the
# guidance now", and useless for "is the fresher run closer", because then every valid time
# carries exactly ONE run and no hour is ever covered twice. Widening the window does not
# help: while a 12Z run exists in the same request, 00Z will never be returned for 16Z.
# So verification fetches ONE REQUEST PER RUN, each with asOf pinned inside that run's own
# window (after it, before its successor) so that run is the only one available. Every hour
# then carries one forecast per run and the comparison is real.
# Native run cadence per model -- how often a new cycle posts. Two jobs: (a) SNAP the newest
# verification run to a real cycle (GFS/IFS only exist at 00/06/12/18Z; HRRR/NBM every hour),
# and (b) PIN each request's asOf to just before the successor at the SAME cadence, so exactly
# the target run qualifies. Using a fixed 6h pin on an hourly model was the old bug: targeting
# the 06Z HRRR run at asOf=11:59 returned the 11Z run, collapsing every run to a late, near-
# zero-coverage hourly cycle.
_MODEL_CYCLE_H = {"gfs": 6, "ifsoper": 6, "hrrr": 1, "nbm": 1}


def _model_cycle_h(model: str) -> int:
    return _MODEL_CYCLE_H.get(model, 6)


# --- archive cadence: every archived cycle must NAME a run ------------------------------
# HRRR and NBM update hourly, but the archive job fires ~4x/day, so "the freshest run at fire
# time" made the guidance behind a forecast an accident of cron timing -- two runs at the same
# station could sit on different HRRR cycles with nothing recording why. Decision 2026-07-28:
# archive them on the SAME 00/06/12/18Z synoptic cycles as GFS and IFS. Zero credit impact
# (identical requests, identical time grid); it trades up to ~6h of freshness for a cycle that
# is deterministic and therefore comparable across runs -- the property a replayable archive
# needs and a live cron does not.
_ARCHIVE_CYCLE_H = 6
# Wait this long after a cycle before selecting it. HRRR/NBM post ~50 min out, so an hour is
# enough, and it stops a call made ON a synoptic hour from pinning a run that is not up yet
# (which would return nothing rather than falling back to the previous cycle).
_ARCHIVE_POST_LAG_H = 1


def archive_run_and_as_of(model: str, as_of: datetime) -> tuple[datetime | None, datetime]:
    """Pin `model` to a deterministic synoptic run for archiving.

    Returns (run, as_of_to_send). `run` is None for a model already on a 6-hourly cycle
    (GFS, IFS) -- its freshest run IS a synoptic run, so it is left alone and nothing about
    its cost or coverage changes. An hourly model (HRRR, NBM) is snapped BACK to the newest
    00/06/12/18Z cycle that has had time to post, then pinned just before that cycle's
    HOURLY successor so exactly it qualifies -- the same pin `prefetch_verification` uses,
    and for the same reason: a 6h pin on an hourly model selects the wrong run.

    The result is never later than the caller's `as_of`, so the leakage guard still holds."""
    cycle = _model_cycle_h(model)
    if cycle >= _ARCHIVE_CYCLE_H:
        return None, as_of
    t = (as_of - timedelta(hours=_ARCHIVE_POST_LAG_H)).replace(minute=0, second=0, microsecond=0)
    run = t - timedelta(hours=t.hour % _ARCHIVE_CYCLE_H)
    return run, min(run + timedelta(hours=cycle) - timedelta(minutes=1), as_of)


def _ver_spacing_h(model: str) -> int:
    """Hours between the runs verification compares. A 6-hourly model uses its native 6h. A
    rapid-refresh (hourly) model at 1h spacing would give three near-identical columns that
    each cover almost nothing -- observations end at the issue time, so only OLDER runs have
    many hours to verify against -- so hourly models are spaced 3h: distinct recent runs with
    real coverage that still show whether the fresher run is closer."""
    return 6 if _model_cycle_h(model) >= 6 else 3


def _verify_vars(model: str) -> list[gribstream.Var]:
    """The surface subset verification actually renders. Cheaper per hour than the full
    surface set (credits scale with variables), which is what buys the hourly grid."""
    keep = ("t2m", "td2m", "u10", "v10", "wind", "wdir", "gust", "mslp")
    return [v for v in _surface_vars(model) if v.alias in keep]


def verification_runs(as_of: datetime, *, n_runs: int = 3, cycle_h: int = 6,
                      spacing_h: int | None = None) -> list[datetime]:
    """The newest `n_runs` runs at/before `as_of`, newest first. The newest is snapped to a
    real `cycle_h` cadence (so a 6-hourly model never targets a cycle that does not exist);
    successive runs step back by `spacing_h` (defaults to `cycle_h`)."""
    spacing_h = spacing_h or cycle_h
    newest = as_of.replace(minute=0, second=0, microsecond=0)
    newest -= timedelta(hours=newest.hour % cycle_h)
    return [newest - timedelta(hours=spacing_h * k) for k in range(n_runs)]


def prefetch_verification(
    station: str,
    *,
    as_of: datetime | None = None,
    models: tuple[str, ...] = MODELS,
    hours_back: int = 24,
    n_runs: int = 3,
    step_h: int = 1,
    db_path: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Archive the last `n_runs` runs of each model over the SAME hourly grid, so the
    verification table can show one hour forecast by several runs.

    One request per (model, run). `as_of` for each request is pinned just before the NEXT
    run so only that run qualifies -- leakage-safe by construction, since every run used is
    older than the issue time. Each run is asked only for the hours it can actually cover
    (a run cannot forecast before it starts), so the credits land at roughly
    sum over runs of (covered hours x variables), not n_runs x window x variables."""
    as_of = (as_of or _utcnow()).replace(minute=0, second=0, microsecond=0)
    window_start = as_of - timedelta(hours=hours_back)
    lat, lon, name = site_coord(station)
    coords = [(lat, lon, name)]
    use_models, dropped = _applicable_models(coords, models)
    fetched_at = _utcnow()
    charged, to_insert, notes = 0, [], []
    if dropped:
        notes.append(f"skipped CONUS-only model(s) {','.join(dropped)} for {station}")

    model_runs: dict = {}
    for model in use_models:
        variables = _verify_vars(model)
        cycle = _model_cycle_h(model)
        runs = verification_runs(as_of, n_runs=n_runs, cycle_h=cycle,
                                 spacing_h=_ver_spacing_h(model))
        model_runs[model] = [f"{r:%Y-%m-%dT%HZ}" for r in runs]
        for run in runs:
            start = max(run, window_start)
            if start > as_of:
                continue
            n = int((as_of - start).total_seconds() // 3600) // step_h
            times = [start + timedelta(hours=step_h * k) for k in range(n + 1)]
            # Pin asOf just before this run's successor at the MODEL's own cadence, so exactly
            # this run qualifies (a fixed 6h pin on an hourly model selected the wrong run).
            run_as_of = run + timedelta(hours=cycle) - timedelta(minutes=1)
            try:
                ts = gribstream.fetch_points(model, coords, variables, times=times,
                                             as_of=run_as_of, use_cache=use_cache)
                charged += ts.charged
                to_insert += _flatten(model, ts, as_of=run_as_of, fetched_at=fetched_at)
            except ValueError as e:
                notes.append(f"{model} run {run:%Y-%m-%dT%HZ}: {e}")

    with store.write_lock(db_path):
        con = store.connect(db_path or settings.db_path)
        try:
            store.init_model_data_schema(con)
            inserted = store.insert_model_data(con, to_insert)
        finally:
            con.close()
    return {"station": station.upper(), "models": list(use_models),
            "runs": model_runs,
            "window": f"{window_start:%Y-%m-%dT%HZ}..{as_of:%Y-%m-%dT%HZ}",
            "credits_charged": charged, "rows": len(to_insert), "inserted": inserted,
            "notes": notes}


# --- GEFS ensemble product --------------------------------------------------------------
GEFS_MODEL = "gefsatmos"
GEFS_N_MEMBERS = 31                         # control + 30 perturbations
GEFS_STEP_H = 3                             # native cadence; the grid is anchored at 00Z
# A thinned default: members are billed linearly, and ~10 members recover most of the spread
# for a probability read, so the roster-wide default is a subset. Override for a full pull.
GEFS_DEFAULT_MEMBERS = tuple(range(0, GEFS_N_MEMBERS, 3))   # 0,3,6,...,30 -> 11 members
GEFS_FULL_MEMBERS = tuple(range(GEFS_N_MEMBERS))            # the archive default (see below)


def estimate_ensemble(stations: list[str], *, hours: int = 30, step_h: int = GEFS_STEP_H,
                      members: tuple[int, ...] = GEFS_DEFAULT_MEMBERS) -> dict:
    """Credits `prefetch_ensemble_many` would charge. No network.

    Note what is NOT in this formula: the station count. The ensemble pulls ONE point per
    station and points are free below 500, so the bill is identical for 1 station or 400 --
    the whole reason the batched form exists."""
    times = _snapped_grid(_utcnow(), hours, max(int(step_h), GEFS_STEP_H))
    chunks = max(1, math.ceil(len(stations) / 500))
    return {"credits": len(times) * len(_gefs_vars()) * len(members) * chunks,
            "valid_times": len(times), "vars": len(_gefs_vars()), "members": len(members),
            "chunks": chunks}


def prefetch_ensemble_many(
    stations: list[str],
    *,
    as_of: datetime | None = None,
    hours: int = 30,
    step_h: int = GEFS_STEP_H,
    members: tuple[int, ...] = GEFS_DEFAULT_MEMBERS,
    db_path: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Archive the GEFS ensemble for SEVERAL stations in ONE bundle.

    The batching asymmetry here is sharper than for the deterministic tier. Members bill
    LINEARLY (credits = valid_times x variables x members) but coordinates are free below
    500, and the ensemble pulls a SINGLE point per station -- probabilities are for the
    aerodrome, not its neighbourhood. So N separate `prefetch_ensemble` calls cost N times
    the bundle for data one request would have returned. Whole network, one request, one bill.

    `as_of` pins the run cutoff (leakage-safe like the deterministic prefetch). The valid-time
    grid is snapped to GEFS's own 00Z-anchored 3-hourly cadence -- see `_snapped_grid`; an
    unsnapped grid returns nothing at all and says so nowhere."""
    as_of = (as_of or _utcnow()).replace(minute=0, second=0, microsecond=0)
    coords = _dedupe([[site_coord(s)] for s in stations])
    times = _snapped_grid(as_of, hours, max(int(step_h), GEFS_STEP_H))
    fetched_at = _utcnow()
    charged, to_insert, notes = 0, [], []
    for chunk in _chunk(coords):
        try:
            ts = gribstream.fetch_points(GEFS_MODEL, chunk, _gefs_vars(), times=times,
                                         as_of=as_of, members=list(members),
                                         use_cache=use_cache)
            charged += ts.charged
            to_insert += _flatten(GEFS_MODEL, ts, as_of=as_of, fetched_at=fetched_at)
        except ValueError as e:
            notes.append(f"{GEFS_MODEL}: {e}")
    # An empty return is the failure mode this product actually has -- no exception, no
    # credits, an archive that looks pulled. Say so rather than reporting a clean zero.
    if not to_insert and not notes:
        notes.append(f"{GEFS_MODEL} returned NO rows for {len(coords)} coord(s) over "
                     f"{times[0]:%Y-%m-%dT%HZ}..{times[-1]:%Y-%m-%dT%HZ}: the run may not be "
                     "posted at this as_of, or no member carried these fields. Nothing was "
                     "archived -- do not treat this cycle as captured.")

    with store.write_lock(db_path):
        con = store.connect(db_path or settings.db_path)
        try:
            store.init_model_data_schema(con)
            inserted = store.insert_model_data(con, to_insert)
        finally:
            con.close()
    return {"stations": [s.upper() for s in stations], "model": GEFS_MODEL,
            "members": list(members), "as_of": as_of, "coords": len(coords),
            "valid_times": len(times),
            "window": f"{times[0]:%Y-%m-%dT%HZ}..{times[-1]:%Y-%m-%dT%HZ}",
            "credits_charged": charged, "rows": len(to_insert), "inserted": inserted,
            "notes": notes}


def prefetch_ensemble(
    station: str,
    *,
    as_of: datetime | None = None,
    hours: int = 30,
    step_h: int = 3,
    members: tuple[int, ...] = GEFS_DEFAULT_MEMBERS,
    db_path: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Archive the GEFS ensemble for ONE station so `get_ensemble_prob` can turn the member
    spread into hourly probabilities. Distinct from `prefetch`: members are billed LINEARLY,
    so this is deliberately NOT part of the default surface archive.

    A thin wrapper over `prefetch_ensemble_many` rather than a parallel implementation --
    this function previously built its own unsnapped valid-time grid and, at an odd issue
    hour, matched none of GEFS's 00Z-anchored 3-hourly times: zero rows, zero credits, no
    error. Sharing the batched path means that cannot be fixed in one place and left broken
    in the other. `members` defaults to a thinned subset; pass GEFS_FULL_MEMBERS for all 31."""
    r = prefetch_ensemble_many([station], as_of=as_of, hours=hours, step_h=step_h,
                               members=members, db_path=db_path, use_cache=use_cache)
    return {"station": station.upper(), **{k: v for k, v in r.items() if k != "stations"}}


def prefetch(
    station: str,
    *,
    as_of: datetime | None = None,
    models: tuple[str, ...] = MODELS,
    hours: int = 30,
    step_h: int = 1,
    hazards: bool = True,
    hazard_step_h: int = 3,
    back_hours: int = 6,
    flow_relative: bool | None = None,
    db_path: str | None = None,
    use_cache: bool = True,
    profiles: bool = True,
) -> dict:
    """Pre-fetch ONE station's model neighborhood into the model_data archive for a cycle.

    `as_of` (default now) pins the run cutoff: only forecasts issued at/before it are pulled,
    so passing the TAF issue time makes the archive leakage-safe by construction. Surface
    fields are pulled for the FULL coordinate set at a `step_h` grid; the merged pressure-level
    bundle (sounding ladder + hazard extras) at a coarser `hazard_step_h` grid. `profiles`
    adds the sounding ladder that backs get_fcst_sounding. `flow_relative` (default from
    settings) densifies upstream via climo. HRRR is CONUS-only, so an OCONUS site simply
    yields no HRRR rows (caught, not fatal). A thin wrapper over prefetch_many."""
    r = prefetch_many([station], as_of=as_of, models=models, hours=hours, step_h=step_h,
                      hazards=hazards, hazard_step_h=hazard_step_h, back_hours=back_hours,
                      flow_relative=flow_relative, db_path=db_path, use_cache=use_cache,
                      profiles=profiles)
    return {"station": station.upper(), **{k: v for k, v in r.items() if k != "stations"}}


def estimate_prefetch_many(
    stations: list[str],
    *,
    as_of: datetime | None = None,
    models: tuple[str, ...] = MODELS,
    hours: int = 30,
    step_h: int = 1,
    hazards: bool = True,
    hazard_step_h: int = 3,
    back_hours: int = 6,
    flow_relative: bool | None = None,
    profiles: bool = True,
) -> dict:
    """What `prefetch_many` WOULD charge, with a per-model breakdown. No network, no DB.

    Deliberately built from the SAME helpers the fetch uses -- `_time_grid`,
    `_applicable_models`, `_surface_vars`/`_profile_vars`/`_hazard_vars`, `_chunk` -- rather
    than reimplementing the cost model. The previous estimator was a simplified restatement
    and under-reported by roughly 3x, in four separate ways, every one of which is the kind
    of drift a parallel implementation invites:
      - it omitted `_profile_vars` entirely (5 variables at EVERY pressure level), which
        became the dominant cost when the sounding tier moved off BUFKIT onto GRIBStream;
      - it used `hours // step_h + 1`, ignoring the pre-anchor tail `back_hours` adds;
      - it called `_hazard_vars(model)` without `profiles=`, counting the T/RH the profile
        bundle already pays for -- the merged-bundle saving, double-counted;
      - it iterated `MODELS` directly, billing HRRR and NBM on an all-OCONUS batch that
        `_applicable_models` drops.
    Signature mirrors `prefetch_many` so the two cannot be called with different assumptions.

    One irreducible approximation: the flow-relative grid's ROTATION needs a live steering
    probe. Rotation does not change the point COUNT and points are free below 500, so the
    figure is exact unless a chunk boundary is straddled -- flagged in `notes` when close."""
    as_of = as_of or _utcnow()
    anchor = as_of.replace(minute=0, second=0, microsecond=0)
    if flow_relative is None:
        flow_relative = settings.model_data_flow_relative
    # Any bearing gives the right CARDINALITY; only the orientation would differ live.
    flow_from = 0.0 if flow_relative else None

    surface_coords = _dedupe([coords_for(s, flow_from=flow_from) for s in stations])
    want_levels = hazards or profiles
    hazard_coords_all = (_dedupe([hazard_coords(s, flow_from=flow_from) for s in stations])
                         if want_levels else [])

    models_eff, dropped = _applicable_models(surface_coords, models)
    cycle = archive_cycle(as_of)

    per_model, total = {}, 0
    for model in models_eff:
        # PER MODEL, through the same helpers the fetch uses: its own run-anchored grids at its
        # own native cadence (`_model_times`) over its own coordinate subset (`model_coords`).
        # Restating any of that here is exactly how the previous estimator came to under-report
        # by 5.4x -- it is not enough to share the variable bundles.
        sfc_times, haz_times = _model_times(model, anchor, hours, step_h, hazard_step_h,
                                            back_hours, run=cycle)
        m_sfc = model_coords(model, surface_coords)
        m_lvl = model_coords(model, hazard_coords_all)
        n_sfc_chunks = len(list(_chunk(m_sfc))) if m_sfc else 0
        n_lvl_chunks = len(list(_chunk(m_lvl))) if m_lvl else 0
        sfc = len(sfc_times) * len(_surface_vars(model)) * n_sfc_chunks
        level_vars = _profile_vars(model) if profiles else []
        if hazards:
            level_vars = level_vars + _hazard_vars(model, profiles=profiles)
        lvl = len(haz_times) * len(level_vars) * n_lvl_chunks if level_vars else 0
        per_model[model] = {"surface": sfc, "levels": lvl, "level_vars": len(level_vars),
                            "sfc_coords": len(m_sfc), "sfc_times": len(sfc_times)}
        total += sfc + lvl
    # The steering probe is one request per station: one valid time, the deep-layer u/v set.
    probe = len(stations) * len(_steer_vars()) if flow_relative else 0

    notes = []
    if dropped:
        notes.append(f"{', '.join(dropped)} skipped: no coordinate in CONUS")
    for model in models_eff:
        n = per_model[model]["sfc_coords"]
        if n and n < len(surface_coords):
            notes.append(f"{model} is CONUS-only: billed over {n} of {len(surface_coords)} "
                         "surface coords (the rest would return null)")
    for label, n in (("surface", len(surface_coords)), ("level", len(hazard_coords_all))):
        if n and n % 500 > 450:
            notes.append(f"{label} coords ({n}) are near a 500-point chunk boundary; a live "
                         "flow-relative rotation could add a chunk and raise the true cost")
    return {
        "credits": total + probe,
        "per_model": per_model,
        "steering_probe": probe,
        "models": list(models_eff),
        "coords": len(surface_coords),
        "hazard_coords": len(hazard_coords_all),
        # There is no single surface grid any more -- each model runs on its own cadence, so
        # these are the WIDEST across models and exist only for a summary line. Reading them
        # as "the grid" is what made the dry-run header print IFS's 17 times for every model:
        # they used to be the leaked loop variable. Per-model truth is in `per_model[m]`.
        "max_surface_times": max((p["sfc_times"] for p in per_model.values()), default=0),
        "level_times": len(haz_times) if models_eff else 0,
        "notes": notes,
    }


def prefetch_many(
    stations: list[str],
    *,
    as_of: datetime | None = None,
    models: tuple[str, ...] = MODELS,
    hours: int = 30,
    step_h: int = 1,
    hazards: bool = True,
    hazard_step_h: int = 3,
    back_hours: int = 6,
    flow_relative: bool | None = None,
    db_path: str | None = None,
    use_cache: bool = True,
    profiles: bool = True,
) -> dict:
    """Pre-fetch SEVERAL stations that share one issue time (`as_of`) in as few requests as
    possible -- the batched roster-wide optimization. Because coordinates are free up to 500,
    the union of all stations' neighborhoods costs the SAME per request as one station: N due
    stations that share a cycle collapse from N requests to ceil(total_coords/500) (~1). The
    archive is coordinate-indexed, so collect.py's per-station copy_model_data still demuxes
    each station by its own coords. All stations MUST share `as_of` (same cycle hour)."""
    as_of = as_of or _utcnow()
    anchor = as_of.replace(minute=0, second=0, microsecond=0)
    if flow_relative is None:
        flow_relative = settings.model_data_flow_relative

    # PASS 1 per station: probe the steering flow (live), orient that station's grid. The probe
    # rows are archived so collect.py's copy recomputes the identical bearing offline.
    surf_lists, haz_lists, probe_rows = [], [], []
    probe_charged, oriented = 0, 0
    for s in stations:
        site = site_coord(s)
        flow_from = None
        if flow_relative:
            flow_from, rows, ch = _steering_probe(site[0], site[1], s, as_of, use_cache=use_cache)
            probe_charged += ch
            probe_rows += rows
            if flow_from is None:                       # no current steering -> climo fallback
                flow_from = _flow_from_climo(s, as_of.month, as_of.hour, db_path)
            if flow_from is not None:
                oriented += 1
        surf_lists.append(coords_for(s, flow_from=flow_from))
        want_levels = hazards or profiles
        haz_lists.append(hazard_coords(s, flow_from=flow_from) if want_levels else [])
    surface_coords = _dedupe(surf_lists)
    hazard_coords_all = _dedupe(haz_lists) if (hazards or profiles) else []

    # Drop CONUS-only models for a wholly-OCONUS request (they return only billable all-null).
    models_eff, dropped_models = _applicable_models(surface_coords, models)

    # PASS 2: fetch the oriented grid (+ archive the probe columns).
    charged, flattened, inserted, notes = _fetch_and_insert(
        surface_coords, hazard_coords_all, as_of=as_of, anchor=anchor, models=models_eff,
        hours=hours, step_h=step_h, hazards=hazards, hazard_step_h=hazard_step_h,
        back_hours=back_hours, db_path=db_path, use_cache=use_cache, extra_rows=probe_rows,
        profiles=profiles)
    if dropped_models:
        notes.insert(0, f"{', '.join(dropped_models)} skipped: no coordinate in CONUS "
                        f"(GFS-only OCONUS; HRRR/NBM are CONUS-domain on GRIBStream)")

    return {
        "stations": [s.upper() for s in stations],
        "as_of": as_of,
        "models": list(models_eff),
        "flow_relative": bool(flow_relative),
        "oriented_stations": oriented,
        "coords": len(surface_coords),
        "hazard_coords": len(hazard_coords_all),
        "profiles": bool(profiles),
        "requests": len(models_eff) * (len(list(_chunk(surface_coords)))
                                       + (len(list(_chunk(hazard_coords_all)))
                                          if (hazards or profiles) else 0))
                    + (len(stations) if flow_relative else 0),   # + steering probes
        "rows_flattened": flattened,
        "rows_inserted": inserted,
        "credits_charged": charged + probe_charged,
        "notes": notes,
    }
