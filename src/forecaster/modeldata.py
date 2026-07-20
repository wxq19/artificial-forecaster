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
# 'pl 850'), NOT the GFS GRIB2 style. These are now VERIFIED from the official model page
# (gribstream.com/models/ifsoper), so the spec below is accurate -- BUT ifsoper stays OUT of
# the default MODELS until one live pull confirms it end to end, because two unit/format
# quirks bite at enable time:
#   - IFS `tcc` (total cloud) is a FRACTION 0-1, whereas GFS/NBM TCDC is PERCENT 0-100; the
#     model-state formatter prints tcdc as-is, so a per-model *100 is needed before enabling.
#   - IFS has NO surface visibility/gust/ceiling fields, and NO CAPE/CIN/HLCY/CLMR (so no
#     convective-turbulence or cloud-liquid-icing signal); a future IFS hazard bundle could
#     add icing (t/r) + shear (u/v/w) at 'pl <hPa>' levels only.
# Global 0.25deg, runs 00/06/12/18Z, out to 360 h -> works OCONUS too (unlike HRRR).
_IFS_ENABLED = False


def _surface_vars(model: str) -> list[gribstream.Var]:
    V = gribstream.Var
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


def _hazard_vars(model: str) -> list[gribstream.Var]:
    # NBM is surface-only; IFS pressure-level names are unverified -> no hazard bundle until
    # probed (icing/turbulence stays a GFS+HRRR product, as designed).
    if model in ("nbm", "ifsoper"):
        return []
    V = gribstream.Var
    vs: list[gribstream.Var] = []
    for lv in ICE_LEVELS:
        p = lv[:3]
        vs += [V("TMP", lv, f"t{p}"), V("RH", lv, f"rh{p}")]
        if model == "gfs":
            vs.append(V("CLMR", lv, f"clw{p}"))
    for lv in SHEAR_LEVELS:
        p = lv[:3]
        vs += [V("UGRD", lv, f"u{p}"), V("VGRD", lv, f"v{p}")]
    for lv in VVEL_LEVELS:
        vs.append(V("VVEL", lv, f"w{lv[:3]}"))
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


def hazard_coords(station: str, *, flow_from: float | None = None) -> list[tuple[float, float, str]]:
    """Coordinates the pressure-level hazard bundle is pulled for: the site + the ring grid
    (NOT the neighbor airfields -- those exist for surface obs collocation). Free points, so
    this is a focus/clarity choice, not a cost one."""
    site = site_coord(station)
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
                "value": r.get(v), "member": member,
                "as_of": as_of, "fetched_at": fetched_at,
            })
    return rows


def _chunk(coords: list, size: int = 500):
    """Split a coordinate list into <=size batches (the free-coordinate ceiling per request)."""
    for i in range(0, len(coords), size):
        yield coords[i:i + size]


def _dedupe(coord_lists: list[list]) -> list[tuple[float, float, str]]:
    """Union several coordinate lists, keeping the first name for a given (lat, lon)."""
    seen: dict = {}
    for lst in coord_lists:
        for la, lo, name in lst:
            seen.setdefault((round(la, 4), round(lo, 4)), (round(la, 4), round(lo, 4), name))
    return list(seen.values())


def _fetch_and_insert(
    surface_coords: list, hazard_coords_all: list, *,
    as_of: datetime, anchor: datetime, models: tuple[str, ...],
    hours: int, step_h: int, hazards: bool, hazard_step_h: int, back_hours: int,
    db_path: str | None, use_cache: bool, extra_rows: list[dict] | None = None,
) -> tuple[int, int, int, list]:
    """Fetch surface (+ hazard) for the given coordinate unions across `models`, chunked to
    <=500 coords/request, and insert under one write_lock (with any `extra_rows`, e.g. the
    steering-probe columns). Returns (charged, flattened, inserted, notes)."""
    sfc_times = _time_grid(anchor, hours, step_h, back_h=back_hours)
    haz_times = _time_grid(anchor, hours, hazard_step_h)
    fetched_at = _utcnow()
    charged = 0
    to_insert: list[dict] = list(extra_rows or [])
    notes: list[str] = []
    for model in models:
        for chunk in _chunk(surface_coords):
            try:
                ts = gribstream.fetch_points(model, chunk, _surface_vars(model),
                                             times=sfc_times, as_of=as_of, use_cache=use_cache)
                charged += ts.charged
                to_insert += _flatten(model, ts, as_of=as_of, fetched_at=fetched_at)
            except ValueError as e:
                notes.append(f"{model} surface: {e}")
        if hazards and _hazard_vars(model) and hazard_coords_all:
            for chunk in _chunk(hazard_coords_all):
                try:
                    ts = gribstream.fetch_points(model, chunk, _hazard_vars(model),
                                                 times=haz_times, as_of=as_of, use_cache=use_cache)
                    charged += ts.charged
                    to_insert += _flatten(model, ts, as_of=as_of, fetched_at=fetched_at)
                except ValueError as e:
                    notes.append(f"{model} hazard: {e}")

    with store.write_lock(db_path):
        con = store.connect(db_path or settings.db_path)
        try:
            store.init_model_data_schema(con)
            inserted = store.insert_model_data(con, to_insert)
        finally:
            con.close()
    return charged, len(to_insert), inserted, notes


def prefetch(
    station: str,
    *,
    as_of: datetime | None = None,
    models: tuple[str, ...] = MODELS,
    hours: int = 30,
    step_h: int = 2,
    hazards: bool = True,
    hazard_step_h: int = 3,
    back_hours: int = 6,
    flow_relative: bool | None = None,
    db_path: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Pre-fetch ONE station's model neighborhood into the model_data archive for a cycle.

    `as_of` (default now) pins the run cutoff: only forecasts issued at/before it are pulled,
    so passing the TAF issue time makes the archive leakage-safe by construction. Surface
    fields are pulled for the FULL coordinate set at a `step_h` grid; the pressure-level
    hazard bundle (GFS/HRRR) at a coarser `hazard_step_h` grid. `flow_relative` (default from
    settings) densifies upstream via climo. HRRR is CONUS-only, so an OCONUS site simply
    yields no HRRR rows (caught, not fatal). A thin wrapper over prefetch_many."""
    r = prefetch_many([station], as_of=as_of, models=models, hours=hours, step_h=step_h,
                      hazards=hazards, hazard_step_h=hazard_step_h, back_hours=back_hours,
                      flow_relative=flow_relative, db_path=db_path, use_cache=use_cache)
    return {"station": station.upper(), **{k: v for k, v in r.items() if k != "stations"}}


def prefetch_many(
    stations: list[str],
    *,
    as_of: datetime | None = None,
    models: tuple[str, ...] = MODELS,
    hours: int = 30,
    step_h: int = 2,
    hazards: bool = True,
    hazard_step_h: int = 3,
    back_hours: int = 6,
    flow_relative: bool | None = None,
    db_path: str | None = None,
    use_cache: bool = True,
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
        haz_lists.append(hazard_coords(s, flow_from=flow_from) if hazards else [])
    surface_coords = _dedupe(surf_lists)
    hazard_coords_all = _dedupe(haz_lists) if hazards else []

    # Drop CONUS-only models for a wholly-OCONUS request (they return only billable all-null).
    models_eff, dropped_models = _applicable_models(surface_coords, models)

    # PASS 2: fetch the oriented grid (+ archive the probe columns).
    charged, flattened, inserted, notes = _fetch_and_insert(
        surface_coords, hazard_coords_all, as_of=as_of, anchor=anchor, models=models_eff,
        hours=hours, step_h=step_h, hazards=hazards, hazard_step_h=hazard_step_h,
        back_hours=back_hours, db_path=db_path, use_cache=use_cache, extra_rows=probe_rows)
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
        "requests": len(models_eff) * (len(list(_chunk(surface_coords)))
                                       + (len(list(_chunk(hazard_coords_all))) if hazards else 0))
                    + (len(stations) if flow_relative else 0),   # + steering probes
        "rows_flattened": flattened,
        "rows_inserted": inserted,
        "credits_charged": charged + probe_charged,
        "notes": notes,
    }
