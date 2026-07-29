"""Freeze one (station, cycle) worth of LIVE-FETCHED agent inputs into the artifact store.

    uv run python scripts/archive_run.py --dry-run                  # plan only, no network
    uv run python scripts/archive_run.py --stations KWRI            # one station, this hour
    uv run python scripts/archive_run.py --at 2026-07-28T11:00      # a named cycle
    uv run python scripts/archive_run.py                            # every station issuing now

WHAT THIS IS FOR. Round 2 serves tools ONLY from the archive, never live (owner, 2026-07-28).
Six tools fetch from the network -- get_imagery, get_loop, get_map, get_sounding, get_terrain,
get_current_taf -- and their products CANNOT be re-fetched for a past instant: GOES STAR holds
~10 days, SLIDER ~17 hours, and the analysis charts have no time parameter at all. So this job
is the irreversible half of the collection. Model data is the opposite case and is NOT captured
here: GRIBStream serves past runs via `asOf`, so archive_model_data.py can always rebuild it.

WHAT IS FROZEN, AND WHAT IS NOT. This stores PROVIDER BYTES plus provenance, not tool receipts.
A receipt is a pure function of the bytes and the index row, so re-generating it at serve time
lets a wording improvement reach an already-collected round -- the same argument that made
loops archive FRAMES rather than composed filmstrips (contract rule 3). What that buys is
concrete: any (frames, step_min) a model asks for is subsampled from one capture.

KEYED ON RESOLVED PRODUCT, NEVER ON STATION. 71 stations collapse to 16 satellite regions, 5
water-vapour scopes and 7 chart sets, so the 22 stations issuing at 11Z mostly share bytes.
`run_manifest` records what each station was ENTITLED to see, written at capture time -- see
store.insert_manifest for why deriving it at replay is not defensible.

READ `docs/artifact_store.md` before changing the identity strings. They are the archive's
addresses; changing one orphans every artifact already captured under it.
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPException
from typing import Callable
from urllib.error import HTTPError

from forecaster import (
    artifacts,
    awc,
    charts,
    imagery,
    neighbors,
    soundings,
    stations,
    store,
    terrain,
    tools,
    upper_air_sites,
    wxmaps,
)

# Static artifacts have no time dimension, so they key on a sentinel rather than on the cycle
# (contract rule 4). Replay ignores it. A real timestamp here would re-capture the same terrain
# map every hour and defeat the dedup that makes the archive affordable.
STATIC_UTC = datetime(1970, 1, 1)

# GFS panels a forecaster actually reaches for on a 30-hour TAF. Deliberately NOT the full
# 0..384 ladder: every fhr is a separate fetch on every domain, and nothing in a TAF turns on
# day 7. Override with --fhrs.
#
# f018 and f030 added 2026-07-29 (owner). Without them the ladder ran 12 -> 24 -> 36, leaving
# TWELVE-HOUR HOLES either side of f024 -- and the f012-f024 gap is exactly the peak diurnal
# convection window a 30-hour TAF turns on. Panels are free (Tropical Tidbits is not metered)
# and dedup across every station sharing a domain, so the cost is ~10-15 MB a sweep before
# dedup, against a hole no amount of point data fills: a chart shows the SHAPE of a trough that
# a column of numbers at one point cannot.
DEFAULT_FHRS = (0, 6, 12, 18, 24, 30, 36)

# LOOP CADENCE -- capture the finest STEP, and let SPAN come from accumulation.
#
# One capture cannot cover the whole schema (2-10 frames, 10-120 min): 10 frames at 120 min
# reaches 20 hours back and only EUMETSAT holds that. But frames key on their REAL time, not
# on the cycle, so consecutive hourly captures concatenate into one continuous series per
# (region, product) -- and a 10x120min request is then answered by selecting from the series.
#
# So the only thing a capture must get right is the STEP: a 30-minute capture can never answer
# a 10-minute request, while a 10-minute capture answers every coarser one by subsampling.
# The frame count only has to bridge to the next hourly capture: span = (frames-1) x step, so
# 7 x 10 min = 60 min exactly closes the gap. 6 would leave a 10-minute hole every hour.
#
# This is CHEAPER than the 10x30 it replaces -- 7 frames instead of 10, and loop frames are
# ~60% of the archive -- while strictly increasing what can be served. Verified 2026-07-28:
# a 10-minute step returns DISTINCT frames on all three providers (STAR, SLIDER, EUMETSAT).
DEFAULT_LOOP_FRAMES = 7
DEFAULT_LOOP_STEP_MIN = 10

# Fetch concurrency. The sweep is network-bound, not CPU-bound: measured 2026-07-29 on the Pi,
# a serial sweep ran 56-66 MINUTES at load 0.00, which is long enough that `flock -n` skipped
# the next hour outright (04Z ran to 06:14 and killed 05Z and 06Z; 14Z ran to 15:08 and killed
# 15Z). A missed hour cannot be re-fetched, so sweep duration is a DATA-LOSS term, not a
# comfort one. It is also what makes the cycle label drift from the bytes: at 14:02 the first
# station got a 13:56Z image and the last got 14:40Z, so shortening the sweep tightens the
# label. Kept modest because these are shared public endpoints (NOAA STAR, IEM radmap.php,
# EUMETSAT WMS) and a rate-limited provider returns FEWER artifacts, not more.
DEFAULT_WORKERS = 6

# One retry, then give up. A `plan` or `expand` failure is not one lost picture -- it loses the
# whole station or the whole loop for an hour that never comes back: the 2026-07-29 08Z sweep
# hit a DNS blip and 18 stations recorded NOTHING. The delay is deliberately longer than a DNS
# timeout so the second attempt is not simply the first one again.
RETRY_ATTEMPTS = 2
RETRY_DELAY_S = 4.0

# How long to WAIT for the index instead of dying on it. DuckDB takes an exclusive file lock,
# so any other process holding it -- a replay serving from the archive, an inspection query,
# a QC pass -- makes `connect_archive` raise. That happened for real on 2026-07-29 at the 23Z
# cycle: a read-only reporting script held the index, the sweep's FIFTH station could not open
# it, the exception escaped `archive_station`, and the whole run died having captured 4 of 71
# stations. **67 stations lost an hour of imagery that cannot be re-fetched.**
#
# Waiting is the right answer because the collision is transient BY CONSTRUCTION now: the
# sweep's own windows are milliseconds (see the three-phase split), so anything blocking us is
# either another short window or a reader that will finish. 60 s is far longer than any
# legitimate holder and still far short of the hourly cadence.
LOCK_WAIT_S = 60.0
LOCK_POLL_S = 1.0


def connect_index(index_path: str, *, read_only: bool = False):
    """`store.connect_archive`, but wait out a transient file lock rather than raise.

    Only a LOCK error is retried. A corrupt file or a bad path still fails immediately -- those
    do not clear by waiting, and pretending otherwise would turn a real fault into a 60 s stall
    followed by the same error."""
    deadline = time.monotonic() + LOCK_WAIT_S
    while True:
        try:
            return store.connect_archive(index_path, read_only=read_only)
        except Exception as e:  # noqa: BLE001 -- classified by message, then re-raised
            if "lock" not in str(e).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(LOCK_POLL_S)

# Held around every call into charts.py. Everything else a fetcher does is a download, which
# threads fine; drawing does not, because pyplot's figure manager is process-global state.
# See _fetch_sounding for what a race here would actually produce.
_RENDER_LOCK = threading.Lock()


@dataclass
class Item:
    """One planned artifact. `fetch` is deferred so --dry-run costs no network."""

    kind: str
    identity: str
    requested_utc: datetime
    provider: str
    fetch: Callable[[], tuple[bytes, str | None, datetime | None, str | None]] | None = None
    note: str | None = None
    # Loop frames are enumerated only by fetching the provider's index, so one Item can
    # expand into many. `expand` returns Items in place of this one.
    expand: Callable[[], list["Item"]] | None = None


@dataclass
class Fetched:
    """One item's fetch outcome, carried from the PARALLEL phase into the SERIAL write.

    This type exists so the network never runs while the index lock is held -- see
    `archive_station`. `error` set means the item is lost for this cycle; every other field
    is then meaningless."""

    item: "Item"
    fetched_utc: datetime | None = None
    data: bytes | None = None
    url: str | None = None
    served: datetime | None = None
    note: str | None = None
    error: str | None = None


def _transient(exc: BaseException) -> bool:
    """Is this failure worth one more attempt?

    YES for a dropped network: DNS, timeouts, resets, 5xx. Those are what cleared on their own
    in the 08Z sweep. NO for a 4xx or a ValueError -- those are the provider telling us the
    product is not there, and 27 of the 29 failures that day were `AWC returned no current TAF`,
    a station between validity periods. Retrying those spends a request to be told the same
    thing, and on a sweep this size that cost is real."""
    if isinstance(exc, HTTPError):
        return exc.code >= 500
    return isinstance(exc, (OSError, HTTPException))     # URLError/socket errors subclass OSError


def _with_retry(fn: Callable):
    """Run `fn`, once more after a pause if it failed for a transient reason."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- classified by _transient, then re-raised
            if attempt == RETRY_ATTEMPTS - 1 or not _transient(e):
                raise
            time.sleep(RETRY_DELAY_S)
    raise AssertionError("unreachable")


@dataclass
class Result:
    captured: int = 0
    reused: int = 0          # key already present -- no fetch, no bytes
    deduped: int = 0         # bytes already on disk under another key
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    bytes_new: int = 0
    manifest: list[tuple[str, str, datetime]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planning. Every resolver here is the SAME function the tool calls, so the archive
# cannot drift from what the agent would have been served. Where a tool applies a
# presentation rule on top (OSPO has no geocolor; Meteosat publishes no water vapour;
# Meteosat crops to the station), that rule is mirrored and named in the comment.
# ---------------------------------------------------------------------------

def _sat_region(icao: str, lat: float, lon: float) -> str | None:
    return imagery.satellite_region_for_latlon(lat, lon)


def _satellite_items(icao: str, lat: float, lon: float, cycle: datetime) -> list[Item]:
    """One still per product at the station's resolved scope."""
    region = _sat_region(icao, lat, lon)
    if region is None:
        return []                                  # mid-ocean/polar: the tool refuses too
    out: list[Item] = []
    for product in imagery.SAT_PRODUCTS:
        prod = product
        spec = imagery.SAT_REGIONS[region]
        # OSPO Japan has no geocolor; the tool relabels its day/night default as infrared, so
        # capturing "geocolor" there would store the infrared bytes under a name they are not.
        if spec.provider == "himawari_ospo" and prod == "geocolor":
            continue
        if spec.provider == "meteosat_eumetsat_wms":
            if not imagery.meteosat_has_product(prod):
                continue                           # MTG publishes none; the tool returns text
            # Meteosat takes an arbitrary bbox, so the tool serves a station-CENTERED crop.
            # That is per-station by construction and cannot dedup -- 8 stations, not 1 image.
            out.append(Item(
                kind="satellite", identity=f"meteosat_point/{icao}/{prod}",
                requested_utc=cycle, provider="eumetsat",
                fetch=lambda p=prod: _fetch_meteosat_point(lat, lon, p, cycle)))
            continue
        # water_vapor widens to a synoptic scope INSIDE fetch_satellite, so resolve it here
        # too -- else the identity names a sector the bytes are not (the 07-28c receipt bug).
        served_region = imagery.synoptic_region(region, prod)
        out.append(Item(
            kind="satellite", identity=f"{served_region}/{prod}", requested_utc=cycle,
            provider=imagery.SAT_REGIONS[served_region].provider,
            fetch=lambda r=served_region, p=prod: _fetch_satellite(r, p)))
    return out


def all_region_items(cycle: datetime) -> list[Item]:
    """Every satellite region x product, independent of any station (--all-regions).

    WHY IT IS OPTIONAL. `get_imagery` accepts an explicit `region`, so a model can name any of
    the catalogue's regions rather than its own. Only 16 of 23 are reachable from the 71
    stations; the other 7 -- the two full disks, conus_west, puerto_rico, caribbean,
    middle_east, africa -- would have no bytes under an archive-only round, and a cycle that
    passes cannot be re-fetched. Measured: ~16 extra stills an hour, roughly 9% on the round.

    These get NO manifest row on purpose. `run_manifest` answers "what was this station
    entitled to see"; an explicitly named far region is by definition outside that, so the
    serve side resolves it through `artifact_keys` directly."""
    out: list[Item] = []
    for region, spec in imagery.SAT_REGIONS.items():
        for product in imagery.SAT_PRODUCTS:
            if spec.provider == "himawari_ospo" and product == "geocolor":
                continue
            if (spec.provider == "meteosat_eumetsat_wms"
                    and not imagery.meteosat_has_product(product)):
                continue
            served = imagery.synoptic_region(region, product)
            out.append(Item(kind="satellite", identity=f"{served}/{product}",
                            requested_utc=cycle,
                            provider=imagery.SAT_REGIONS[served].provider,
                            fetch=lambda r=served, p=product: _fetch_satellite(r, p)))
    return out


def _fetch_satellite(region: str, product: str):
    data, url = imagery.fetch_satellite(region, product)
    # SLIDER names the exact scan in the tile path, so its stills get a true served_utc.
    # GOES STAR and OSPO serve a fixed "latest" path with no time in it at all -- those stay
    # NULL and are bounded by fetched_utc, which is the honest record. Do NOT guess their
    # scan time from a frame index: "newest indexed" is not provably the same image the
    # latest.jpg returned, and a confident wrong timestamp is worse than an absent one.
    return data, url, imagery.slider_time_from_url(url), None


def _fetch_meteosat_point(lat: float, lon: float, product: str, cycle: datetime):
    data, url = imagery.fetch_meteosat_point(lat, lon, product, at=cycle)
    # EUMETSAT snaps to the nearest held scan, bounded by one 10-minute step. The bound
    # belongs in the row, not in someone's memory.
    return data, url, imagery.eumetsat_time(cycle), None


def _loop_items(icao: str, lat: float, lon: float, cycle: datetime, *, frames: int,
                step_min: int) -> list[Item]:
    """Loop FRAMES, never a composed filmstrip (contract rule 3).

    Enumerating them needs the provider's index, so each product is one `expand` Item that
    fetches the series and yields a frame Item per real frame time."""
    region = _sat_region(icao, lat, lon)
    if region is None:
        return []
    spec = imagery.SAT_REGIONS[region]
    out: list[Item] = []
    for product in imagery.SAT_PRODUCTS:
        if spec.provider == "himawari_ospo" and product == "geocolor":
            continue
        if spec.provider == "meteosat_eumetsat_wms" and not imagery.meteosat_has_product(product):
            continue
        # Meteosat loops are station-centered like its stills; the others are sector-wide and
        # dedup across every station in the sector.
        base = (f"meteosat_point/{icao}/{product}"
                if spec.provider == "meteosat_eumetsat_wms" else f"{region}/{product}")
        out.append(Item(
            kind="loop_frame", identity=base, requested_utc=cycle, provider=spec.provider,
            expand=lambda b=base, p=product, pr=spec.provider: _expand_loop(
                b, p, pr, lat, lon, cycle, frames, step_min)))
    return out


_loop_expand_cache: dict[tuple, list["Item"]] = {}
_loop_expand_lock = threading.Lock()


def _expand_loop(identity: str, product: str, provider: str, lat: float, lon: float,
                 cycle: datetime, frames: int, step_min: int) -> list[Item]:
    # MEMOIZED PER SWEEP, and this is the single biggest cost in the job.
    #
    # `satellite_loop` DOWNLOADS every frame's bytes -- `f.data` below is the image, not a
    # promise of one. But loop frames key on the SECTOR (`{region}/{product}`), which the 71
    # stations collapse onto just 56 identities, and this runs once per STATION. The dedup that
    # makes the rest of the sweep cheap lives in `select`, which cannot help here because it
    # only runs AFTER the expand has already pulled the bytes. So every station after the first
    # in a sector re-downloaded seven frames and then counted them "reused" -- 3,148 reused
    # keys a sweep, nearly all loop frames, every one fetched and discarded.
    #
    # Measured: an expand is 9-12 s, 213 of them run per sweep, and only 56 are distinct --
    # roughly 35 minutes of a 56-66 minute sweep spent re-fetching bytes we already had. That
    # is what was pushing sweeps past the hour and making `flock -n` drop whole cycles.
    #
    # Keyed on identity + cycle + cadence, so Meteosat (whose identity carries the ICAO because
    # its loops are station-CENTERED crops) still fetches per station, exactly as it must.
    key = (identity, cycle, frames, step_min)
    with _loop_expand_lock:
        if (hit := _loop_expand_cache.get(key)) is not None:
            return hit
    fr, _source, _coverage = imagery.satellite_loop(
        lat, lon, product, frames=frames, step_min=step_min, at=cycle)
    # `f.data` is what the model saw -- for SLIDER that is the tile with the map layers
    # already composited, which is NOT what re-fetching f.url returns. Store the bytes.
    out = [Item(kind="loop_frame", identity=identity, requested_utc=f.time,
                provider=provider, fetch=lambda fr=f: (fr.data, fr.url, fr.time, None))
           for f in fr]
    with _loop_expand_lock:
        _loop_expand_cache[key] = out
    return out


def reset_loop_expand_cache() -> None:
    """Drop every per-sweep memo. One sweep is one process today, so this is for tests and for
    any caller that runs two cycles in one process.

    ALL THREE are cleared together because they do NOT fail the same way. The loop and
    resolve-source memos key on the cycle, so a second cycle simply misses and re-fetches --
    the only cost is unbounded growth. `_taf_batch` keys on ICAO ALONE, so a second cycle in
    the same process would be served the FIRST cycle's bulletins. Nothing does that today;
    this is what stops it becoming true quietly."""
    with _loop_expand_lock:
        _loop_expand_cache.clear()
    with _resolve_source_lock:
        _resolve_source_cache.clear()
    with _taf_batch_lock:
        _taf_batch.clear()


def _radar_items(icao: str, lat: float, lon: float, cycle: datetime) -> list[Item]:
    """The radar products this station can actually be served.

    MIRRORS `tools._radar_for_station` exactly, including its in-network test. Gating the
    whole group on the regional bbox instead would archive NO radar for KMIB and KRCA, which
    sit in a gap between the curated regions while having a WSR-88D 35 km and 23 km away --
    the tool serves them a station view, so the archive must hold one."""
    near = imagery.nearest_radar(lat, lon)
    region = imagery.radar_region_for_latlon(lat, lon)
    # Mirrors tools._radar_for_station INCLUDING the composite-coverage test: a real WSR-88D
    # outside IEM's US mosaic (Kadena 0 km, Kunsan 3 km, Andersen 20 km) passes the distance
    # guard but returns a picture with no reflectivity in it. Archiving those would freeze an
    # empty raster that reads as "clear". See imagery.iem_composite_covers.
    local = (bool(near and near[1] <= imagery.RADAR_STATION_GUARD_KM)
             and imagery.iem_composite_covers(lat, lon))
    if not region and not local:
        return []                                  # out of network: the tool returns text only
    out: list[Item] = []
    if local:
        out.append(Item(kind="radar", identity=f"station/{icao}", requested_utc=cycle,
                        provider="iem",
                        fetch=lambda: (imagery.fetch_radar("station", center=(lat, lon)),
                                       imagery.radar_url("station", center=(lat, lon)),
                                       None, f"nearest {near[0]['id']} {near[1]:.0f} km")))
    if region:
        out.append(Item(kind="radar", identity=f"regional/{region}", requested_utc=cycle,
                        provider="iem",
                        fetch=lambda r=region: (imagery.fetch_radar("regional", region=r),
                                               imagery.radar_url("regional", region=r),
                                               None, None)))
    out.append(Item(kind="radar", identity="national", requested_utc=cycle, provider="iem",
                    fetch=lambda: (imagery.fetch_radar("national"),
                                   imagery.radar_url("national"), None, None)))
    return out


def _map_items(icao: str, lat: float, lon: float, cycle: datetime,
               fhrs: tuple[int, ...]) -> list[Item]:
    """Every chart the station's region actually depicts, at the forecast hours a TAF uses.

    The gate is `wxmaps.charts_for_latlon`, the same one `_get_map` applies -- so the archive
    holds exactly the set the tool would serve and no station can be handed a chart of the
    wrong hemisphere."""
    allowed, domain, _label = wxmaps.charts_for_latlon(lat, lon)
    if not allowed:
        return []
    domain = domain or "us"
    run = wxmaps.latest_gfs_run()
    out: list[Item] = []
    for name in allowed:
        spec = wxmaps.CATALOG[name]
        if spec.source != "tt":
            # Analysis charts carry no time in the URL, so the cycle IS the only key. They are
            # nationally shared products: identity omits the domain deliberately.
            out.append(Item(kind="map", identity=name, requested_utc=cycle,
                            provider=spec.source,
                            fetch=lambda n=name: (wxmaps.fetch_map(n),
                                                  wxmaps.map_url(n), None, None)))
            continue
        f0 = spec.params.get("f0", 0)
        for fhr in sorted({max(f, f0) - (max(f, f0) % wxmaps.GFS_STEP_H) for f in fhrs}):
            out.append(Item(
                kind="map", identity=f"{domain}/{name}/f{fhr:03d}", requested_utc=cycle,
                provider="tt", note=f"gfs run {run:%Y-%m-%dT%HZ}",
                fetch=lambda n=name, f=fhr, d=domain: (
                    wxmaps.fetch_map(n, fhr=f, run=run, domain=d),
                    wxmaps.map_url(n, fhr=f, run=run, domain=d), run, None)))
    return out


def _sounding_items(icao: str, cycle: datetime) -> list[Item]:
    """The nearest radiosonde that ACTUALLY FLEW at or before the cycle.

    Deferred behind `expand` because finding the launch time reads the provider's inventory,
    and a --dry-run must not touch a provider. Rendered at capture rather than stored raw,
    unlike the imagery: the inventory holds a long history, so an ascent stays re-fetchable
    and the render is not the irreversible part.

    Walks the nearest THREE sites rather than only the first. Measured 2026-07-28: RJTY's
    nearest site (47646 TATENO, 78 km) has NO record at all in this provider, so a
    nearest-only rule silently archives no sounding for Japan."""
    if not upper_air_sites.sites_for(icao):
        return []
    return [Item(kind="sounding", identity=f"{icao}/bufr", requested_utc=cycle,
                 provider="wyoming", expand=lambda: _expand_sounding(icao, cycle))]


_resolve_source_cache: dict[tuple, tuple | None] = {}
_resolve_source_lock = threading.Lock()


def _resolve_source_memo(wmo: str, cycle: datetime):
    """`soundings.resolve_source`, asked once per (site, cycle) instead of once per station.

    Same shape as the loop-expand problem, one order of magnitude smaller. The 71 stations
    share only ~50 radiosonde sites, and several CONUS fields resolve to the same one, so the
    inventory lookup was repeated for sites already resolved this sweep. Measured ~2.1 s per
    station, so ~2.5 min a sweep. The PROFILE download is not affected -- that hangs off the
    returned Item's `fetch`, whose identity is `{wmo}/{src}`, so `select` already deduped it."""
    key = (wmo, cycle)
    with _resolve_source_lock:
        if key in _resolve_source_cache:
            return _resolve_source_cache[key]
    got = soundings.resolve_source(wmo, cycle)
    with _resolve_source_lock:
        _resolve_source_cache[key] = got
    return got


def _expand_sounding(icao: str, cycle: datetime) -> list[Item]:
    for wmo, name, dist, brg, _la, _lo in upper_air_sites.sites_for(icao):
        try:
            got = _resolve_source_memo(wmo, cycle)
        except Exception:  # noqa: BLE001 -- a dead inventory: try the next site out
            continue
        if got is None:
            continue
        src, launched = got
        # The FEED is part of the identity: BUFR and FM35 are different level sets for the
        # same ascent, so one cannot silently stand in for the other at replay.
        # requested_utc is the LAUNCH time, not the cycle: rule 1, key on what was returned.
        # NEVER snapped to 00/12Z -- an off-cycle ascent is released BECAUSE something is
        # happening, so a synoptic snap discards exactly the informative ones.
        return [Item(kind="sounding", identity=f"{wmo}/{src.lower()}", requested_utc=launched,
                     provider="wyoming", note=f"{name}, {dist:.0f} km {brg} of {icao}",
                     fetch=lambda w=wmo, t=launched, s=src: _fetch_sounding(w, t, s))]
    return []


def _fetch_sounding(wmo: str, launched: datetime, src: str = "BUFR"):
    prof = soundings.fetch_profile(wmo, launched, src=src)
    # No `title=` kwarg: charts.skewt reads `profile.title`, which ObsProfile supplies, so an
    # observed ascent is not captioned "forecast". Same call `_sounding_bufr` makes.
    #
    # RENDERED UNDER A LOCK, and this is not optional. charts.skewt goes through pyplot, whose
    # figure manager is GLOBAL and not thread-safe -- and since the fetches became concurrent,
    # this is the one fetcher in the archiver that draws rather than downloads. Two threads in
    # pyplot at once can interleave into one figure, which would hand a replay a skew-T whose
    # caption belongs to a different station. That is the confident-label-over-wrong-content
    # class this file has already been bitten by three times (the Meteosat loop, the widened
    # water-vapour receipt, the EUMETSAT frame labels). The lock costs nothing: soundings are
    # ONE item per station out of ~60, so nothing meaningful ever waits on it.
    with _RENDER_LOCK:
        png = charts.skewt(prof)
    return png, prof.url, launched, None


def _static_items(icao: str, lat: float, lon: float) -> list[Item]:
    """Terrain: captured ONCE per station and keyed on a sentinel time (rule 4)."""
    neigh = neighbors.neighbors_of(icao)
    return [Item(kind="terrain", identity=icao, requested_utc=STATIC_UTC, provider="esri",
                 fetch=lambda: (terrain.relief_map(
                     lat, lon,
                     markers=[(ic, la, lo) for ic, _d, _b, _e, la, lo in neigh],
                     context=neighbors.area_of(icao),
                     radius_mi=tools._map_radius_mi(neigh)), None, None, None))]


def _taf_items(icao: str, cycle: datetime) -> list[Item]:
    """The live official TAF. Text, not pixels -- but it is on the live list, and AWC serves
    only the CURRENT bulletin, so an hour not captured is permanently gone."""
    return [Item(kind="taf", identity=icao, requested_utc=cycle, provider="awc",
                 fetch=lambda: _fetch_taf(icao))]


_taf_batch: dict[str, list] = {}
_taf_batch_lock = threading.Lock()


def prefetch_tafs(icaos: list[str]) -> None:
    """Pull every station's current TAF in ONE request.

    NOT a dedup -- 71 stations need 71 different bulletins. It is 71 REQUESTS collapsing to
    one, and `awc._get` spaces requests 1 s apart, so the old way spent ~71 s a sweep sleeping
    between calls that the API was always willing to batch (`fetch_taf` has taken a list since
    it was written).

    It also makes the capture MORE honest: every station's TAF now comes from a single instant
    instead of being smeared across the 40 minutes a sweep takes, which is the same cycle-label
    drift that `fetched_utc` exists to record."""
    if not icaos:
        return
    got = awc.fetch_taf(list(icaos))
    batch: dict[str, list] = {}
    for issue, raw in got:
        # The bulletin names its own station; trusting the request order would mis-file every
        # TAF after the first station that had none.
        parts = raw.split()
        ident = next((p for p in parts[:3] if len(p) == 4 and p.isalpha()), None)
        if ident:
            batch.setdefault(ident.upper(), []).append((issue, raw))
    with _taf_batch_lock:
        _taf_batch.clear()
        _taf_batch.update(batch)


def _fetch_taf(icao: str):
    with _taf_batch_lock:
        got = _taf_batch.get(icao.upper())
    if got is None:
        # Covers BOTH "no batch ran" and "the batch had nothing for this station". The second
        # costs one redundant request per TAF-less station (1-5 a sweep) to be told the same
        # thing -- kept deliberately, because it also means a partially-successful batch
        # degrades to the old per-station behaviour instead of inventing a missing bulletin.
        got = awc.fetch_taf(icao)
    if not got:
        raise ValueError(f"AWC returned no current TAF for {icao}")
    issue, raw = got[0]
    return raw.encode("utf-8"), None, issue, None


def plan(icao: str, cycle: datetime, *, fhrs: tuple[int, ...], loop_frames: int,
         loop_step_min: int, include: set[str]) -> list[Item]:
    """Everything one (station, cycle) is entitled to, resolved but not yet fetched."""
    lat, lon = awc.station_latlon(icao)
    items: list[Item] = []
    if "terrain" in include:
        items += _static_items(icao, lat, lon)
    if "taf" in include:
        items += _taf_items(icao, cycle)
    if "satellite" in include:
        items += _satellite_items(icao, lat, lon, cycle)
    if "loop" in include:
        items += _loop_items(icao, lat, lon, cycle, frames=loop_frames, step_min=loop_step_min)
    if "radar" in include:
        items += _radar_items(icao, lat, lon, cycle)
    if "map" in include:
        items += _map_items(icao, lat, lon, cycle, fhrs)
    if "sounding" in include:
        items += _sounding_items(icao, cycle)
    return items


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

# Capture runs in three phases, and the split is the whole point: SELECT and STORE touch the
# index, FETCH touches the network, and they must never overlap. DuckDB takes an exclusive file
# lock, so for as long as a connection is open NOTHING else can read the index -- not another
# writer, not even `read_only=True` (verified on the Pi, 2026-07-29: "Could not set lock on
# file ... Conflicting lock is held"). A single connection wrapped around the fetches held that
# lock for 56-66 of every 60 minutes, which leaves no window for the serve side to read the
# archive it is meant to serve from. Keeping the network outside the lock is what makes capture
# and replay able to coexist.

def select(con, item: Item, res: Result, *, refresh: bool) -> bool:
    """Record the manifest row; say whether this item still needs fetching.

    The skip is what makes an hourly sweep affordable: the 22 stations issuing at 11Z share
    one CONUS water-vapour image and one GFS panel set, so the first station pays and the
    other 21 write a manifest row each."""
    res.manifest.append((item.kind, item.identity, item.requested_utc))
    if item.fetch is None:
        res.skipped.append(f"{item.kind} {item.identity} ({item.note or 'no fetcher'})")
        return False
    if not refresh and store.artifact_key(con, item.kind, item.identity,
                                          item.requested_utc) is not None:
        res.reused += 1
        return False
    return True


def select_all(con, items: list[Item], res: Result, *, refresh: bool) -> list[Item]:
    """Every item that still needs fetching, each one ONCE.

    The in-batch dedup is load-bearing now that fetching is parallel. Two items can resolve to
    the same key -- `all_region_items` is full of them, because water vapour widens many
    regions onto one synoptic scope -- and the old serial loop absorbed that for free: the
    first fetched, and the rest saw the key in the index and counted as reused. Selecting the
    whole batch BEFORE any fetch removes that guard, so without this the duplicates would be
    fetched concurrently, race on one cache path, and inflate the captured count."""
    seen: set[tuple[str, str, datetime]] = set()
    todo: list[Item] = []
    for item in items:
        if not select(con, item, res, refresh=refresh):
            continue
        if (key := (item.kind, item.identity, item.requested_utc)) in seen:
            res.reused += 1                        # same key, already in this batch
            continue
        seen.add(key)
        todo.append(item)
    return todo


def fetch_one(item: Item) -> Fetched:
    """Fetch one item. Pure network and no DB handle, so this is what the thread pool runs."""
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        # Stamped per ATTEMPT, so it records the GET that actually returned the bytes. A sweep
        # spans many minutes, so an artifact under a 14:00Z cycle label may really have been
        # pulled at 14:40Z. For GOES STAR, whose served_utc is NULL by design, this is the only
        # record of that skew outside the timestamp burned into the image.
        t0 = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            data, url, served, note = item.fetch()
        except Exception as e:  # noqa: BLE001 -- one dead product must not lose the whole cycle
            last = e
            if attempt == RETRY_ATTEMPTS - 1 or not _transient(e):
                break
            time.sleep(RETRY_DELAY_S)
            continue
        if not data:
            # Not retried: a 200 with an empty body is the provider answering, not failing.
            return Fetched(item, error=f"{item.kind} {item.identity} (provider returned 0 bytes)")
        return Fetched(item, fetched_utc=t0, data=data, url=url, served=served, note=note)
    # Contract rule 2: a silent miss is not a capture. Record it loudly and move on.
    return Fetched(item, error=f"{item.kind} {item.identity} ({type(last).__name__}: {last})")


def store_one(con, got: Fetched, res: Result, *, root) -> None:
    """Index one fetched item. Serial by construction -- artifacts.put writes the blob file,
    and two threads racing on one content-addressed path is not worth the microseconds."""
    if got.error:
        res.failed.append(got.error)
        return
    item = got.item
    sha, mime, n, is_new = artifacts.put(got.data, root=root)
    store.insert_artifact(con, sha256=sha, kind=item.kind, mime=mime, n_bytes=n,
                          first_seen_utc=datetime.now(timezone.utc))
    store.insert_artifact_key(con, kind=item.kind, identity=item.identity,
                              requested_utc=item.requested_utc, sha256=sha,
                              served_utc=got.served, fetched_utc=got.fetched_utc, source_url=got.url,
                              provider=item.provider, note=got.note or item.note)
    res.captured += 1
    if is_new:
        res.bytes_new += n
    else:
        res.deduped += 1


def fetch_all(todo: list[Item], *, workers: int) -> list[Fetched]:
    """Fetch every outstanding item concurrently. Order is preserved so the log reads the
    same as the serial version did."""
    if not todo:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        return list(pool.map(fetch_one, todo))


def archive_station(icao: str, cycle: datetime, *, index_path: str, root, refresh: bool,
                    workers: int, **plan_kw) -> Result:
    """One station's whole entitlement, as resolve -> select -> fetch -> store.

    Takes an index PATH rather than a connection so it can open and close the index twice
    around the fetches, instead of pinning it for the length of the sweep."""
    res = Result()
    try:
        # Retried: `plan` resolves the station's lat/lon over the network, so a blip here
        # loses all ~60 of its artifacts, not one.
        items = _with_retry(lambda: plan(icao, cycle, **plan_kw))
    except Exception as e:  # noqa: BLE001 -- an unresolvable station is one station, not the run
        res.failed.append(f"plan {icao} ({type(e).__name__}: {e})")
        return res

    # Resolve the deferred groups: a loop Item stands for `frames` artifacts and a sounding
    # Item for at most one, and neither can be enumerated without asking the provider.
    #
    # RUN CONCURRENTLY, because an expand is not a lookup -- `satellite_loop` DOWNLOADS its
    # frames, so this phase is 9-12 s of network per loop product and a station has three of
    # them plus a sounding. Serially that is ~30 s per station before a single artifact is
    # fetched, and after the memo it is the largest remaining term in the sweep. The pool is
    # the same size as the fetch pool for the same reason: these are shared public endpoints.
    #
    # Order is preserved by `pool.map`, so the resulting item list -- and therefore the log and
    # the manifest -- reads exactly as it did serially.
    deferred = [i for i in items if i.expand is not None]
    resolved: list[Item] = [i for i in items if i.expand is None]

    def _run_expand(item: Item):
        try:
            return item, _with_retry(item.expand), None    # retried: loses a whole loop
        except Exception as e:  # noqa: BLE001
            return item, None, f"{item.kind} {item.identity} expand ({type(e).__name__}: {e})"

    if deferred:
        with ThreadPoolExecutor(max_workers=min(workers, len(deferred))) as pool:
            outcomes = list(pool.map(_run_expand, deferred))
        for item, expanded, err in outcomes:
            if err:
                res.failed.append(err)
                continue
            if not expanded:
                # Legitimately nothing to capture (no site flew, no frames indexed) -- a SKIP,
                # not a failure, but never silent: rule 2 only forbids recording a miss as a
                # capture.
                res.skipped.append(f"{item.kind} {item.identity} (nothing available to capture)")
                continue
            resolved += expanded

    con = connect_index(index_path)         # lock window 1: reads only
    try:
        todo = select_all(con, resolved, res, refresh=refresh)
    finally:
        con.close()

    got = fetch_all(todo, workers=workers)          # no lock held: this is the slow part

    con = connect_index(index_path)         # lock window 2: writes only
    try:
        for g in got:
            store_one(con, g, res, root=root)
        store.insert_manifest(con, icao, cycle, res.manifest)
    finally:
        con.close()
    return res


def _cycle_from(arg: str | None) -> datetime:
    """The capture instant, floored to the hour. All 24 hours are some station's issue hour,
    so the archiver runs hourly and an hour is the natural cycle key."""
    if arg:
        t = datetime.fromisoformat(arg)
        return t.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    return datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", help="comma-separated ICAOs (default: the whole archive roster)")
    ap.add_argument("--at", help="cycle instant, ISO (default: this hour, UTC)")
    ap.add_argument("--root", help=f"archive directory (default: {artifacts.archive_root()})")
    ap.add_argument("--fhrs", default=",".join(str(f) for f in DEFAULT_FHRS),
                    help="GFS forecast hours for the tt panels")
    ap.add_argument("--loop-frames", type=int, default=DEFAULT_LOOP_FRAMES)
    ap.add_argument("--loop-step-min", type=int, default=DEFAULT_LOOP_STEP_MIN)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"concurrent fetches per station (default {DEFAULT_WORKERS}); "
                         "1 restores the serial behaviour")
    ap.add_argument("--include", default="terrain,taf,satellite,loop,radar,map,sounding",
                    help="comma-separated product groups to capture")
    ap.add_argument("--all-regions", action="store_true",
                    help="also capture every satellite region, not just the ones stations "
                         "resolve to -- covers an explicit `region` argument on get_imagery")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even when the archive already holds the key")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the plan; no fetch, no write")
    args = ap.parse_args()

    icaos = ([s.strip().upper() for s in args.stations.split(",") if s.strip()]
             if args.stations else stations.poll_icaos())
    cycle = _cycle_from(args.at)
    root = artifacts.archive_root(args.root)
    plan_kw = dict(fhrs=tuple(int(f) for f in args.fhrs.split(",") if f.strip()),
                   loop_frames=args.loop_frames, loop_step_min=args.loop_step_min,
                   include={s.strip() for s in args.include.split(",") if s.strip()})

    print(f"=== ARCHIVE run: {len(icaos)} station(s), cycle {cycle:%Y-%m-%dT%HZ} ===")
    print(f"archive root: {root}")

    if args.dry_run:
        # Loop frames cannot be enumerated without the provider index, so they are counted
        # as planned rather than listed. Everything else is exact.
        total = 0
        for icao in icaos:
            try:
                items = plan(icao, cycle, **plan_kw)
            except Exception as e:  # noqa: BLE001
                print(f"{icao}: PLAN FAILED ({type(e).__name__}: {e})")
                continue
            # A loop Item stands for --loop-frames artifacts; every other Item is one, INCLUDING
            # the deferred sounding (it expands to at most one ascent).
            def _weight(i: Item) -> int:
                return args.loop_frames if i.kind == "loop_frame" else 1

            n = sum(_weight(i) for i in items)
            total += n
            groups: dict[str, int] = {}
            for i in items:
                groups[i.kind] = groups.get(i.kind, 0) + _weight(i)
            print(f"{icao}: {n:4d} artifacts  " +
                  "  ".join(f"{k}={v}" for k, v in sorted(groups.items())))
        print(f"\nplanned: ~{total} artifacts across {len(icaos)} station(s) "
              "(loop frames estimated at --loop-frames each; dedup is applied at capture)")
        return 0

    index_path = str(root / "index.duckdb")
    # Schema first, in its own short connection. Every later connection assumes the tables and
    # the fetched_utc migration are already there.
    con = connect_index(index_path)
    try:
        store.init_archive_schema(con)
    finally:
        con.close()

    # One batched TAF request for the whole roster, before the per-station loop. See
    # prefetch_tafs: 71 rate-limited calls become one, and every bulletin is then pinned to the
    # same instant. A failure here is not fatal -- _fetch_taf falls back to asking per station.
    if "taf" in plan_kw["include"]:
        try:
            prefetch_tafs(icaos)
        except Exception as e:  # noqa: BLE001
            print(f"taf prefetch failed, falling back to per-station ({type(e).__name__}: {e})")

    totals = Result()
    if args.all_regions:
        # Before the stations, so a region a station also wants is fetched once here and
        # merely reused below rather than the other way round.
        extra = Result()
        con = connect_index(index_path)
        try:
            todo = select_all(con, all_region_items(cycle), extra, refresh=args.refresh)
        finally:
            con.close()
        got = fetch_all(todo, workers=args.workers)
        con = connect_index(index_path)
        try:
            for g in got:
                store_one(con, g, extra, root=root)
        finally:
            con.close()
        extra.manifest.clear()                      # deliberately no manifest rows -- see above
        totals.captured += extra.captured
        totals.reused += extra.reused
        totals.deduped += extra.deduped
        totals.bytes_new += extra.bytes_new
        totals.failed += extra.failed
        print(f"all-regions: {extra.captured:3d} captured  {extra.reused:3d} reused  "
              f"{len(extra.failed):2d} failed  {extra.bytes_new / 1e6:6.1f} MB new")

    for icao in icaos:
        # ONE STATION MUST NEVER TAKE THE SWEEP DOWN. `archive_station` already turns a dead
        # product or a failed plan into a recorded failure, but anything it did NOT anticipate
        # -- an index lock it could not wait out, an unexpected provider type -- escaped here
        # and aborted the run mid-roster. That is exactly what cost 67 of 71 stations at the
        # 2026-07-29 23Z cycle. An hour of imagery cannot be re-fetched, so the remaining
        # stations are worth far more than a clean stack trace.
        try:
            res = archive_station(icao, cycle, index_path=index_path, root=root,
                                  refresh=args.refresh, workers=args.workers, **plan_kw)
        except Exception as e:  # noqa: BLE001 -- log it loudly, keep sweeping
            totals.failed.append(f"station {icao} ({type(e).__name__}: {e})")
            print(f"{icao}: ABORTED ({type(e).__name__}: {e})")
            continue
        totals.captured += res.captured
        totals.reused += res.reused
        totals.deduped += res.deduped
        totals.bytes_new += res.bytes_new
        totals.skipped += res.skipped
        totals.failed += res.failed
        print(f"{icao}: {res.captured:3d} captured  {res.reused:3d} reused  "
              f"{res.deduped:3d} deduped  {len(res.skipped):2d} skipped  "
              f"{len(res.failed):2d} failed  {res.bytes_new / 1e6:6.1f} MB new")

    # The closing summary is a NICETY. Every artifact is already on disk and indexed by here, so
    # a lock collision reading the totals must not turn a successful sweep into a failed exit.
    stats = None
    try:
        con = connect_index(index_path, read_only=True)
        try:
            stats = store.archive_stats(con)
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001
        print(f"(archive stats unavailable: {type(e).__name__}: {e})")

    print(f"\ncaptured {totals.captured}, reused {totals.reused}, deduped {totals.deduped}, "
          f"{totals.bytes_new / 1e6:.1f} MB new bytes")
    if stats:
        print(f"archive now: {stats['blobs']} blobs, {stats['bytes'] / 1e9:.2f} GB, "
              f"{stats['keys']} keys ({stats['dedup']}x dedup), "
              f"{stats['manifest_rows']} manifest rows")
    for s in totals.skipped:
        print(f"  SKIP {s}")
    for f in totals.failed:
        print(f"  FAIL {f}")
    # A failure is data, not an abort: a dead provider on one product must not stop the sweep,
    # and the exit code is what a cron watches. Non-zero only when EVERYTHING failed.
    return 1 if totals.captured == 0 and totals.failed else 0


if __name__ == "__main__":
    sys.exit(main())
