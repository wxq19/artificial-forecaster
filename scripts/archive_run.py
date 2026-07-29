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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

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
DEFAULT_FHRS = (0, 6, 12, 24, 36)

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
    return data, url, None, None                   # STAR/SLIDER served time is not returned


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


def _expand_loop(identity: str, product: str, provider: str, lat: float, lon: float,
                 cycle: datetime, frames: int, step_min: int) -> list[Item]:
    fr, _source, _coverage = imagery.satellite_loop(
        lat, lon, product, frames=frames, step_min=step_min, at=cycle)
    # `f.data` is what the model saw -- for SLIDER that is the tile with the map layers
    # already composited, which is NOT what re-fetching f.url returns. Store the bytes.
    return [Item(kind="loop_frame", identity=identity, requested_utc=f.time,
                 provider=provider, fetch=lambda fr=f: (fr.data, fr.url, fr.time, None))
            for f in fr]


def _radar_items(icao: str, lat: float, lon: float, cycle: datetime) -> list[Item]:
    """The radar products this station can actually be served.

    MIRRORS `tools._radar_for_station` exactly, including its in-network test. Gating the
    whole group on the regional bbox instead would archive NO radar for KMIB and KRCA, which
    sit in a gap between the curated regions while having a WSR-88D 35 km and 23 km away --
    the tool serves them a station view, so the archive must hold one."""
    near = imagery.nearest_radar(lat, lon)
    region = imagery.radar_region_for_latlon(lat, lon)
    local = bool(near and near[1] <= imagery.RADAR_STATION_GUARD_KM)
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


def _expand_sounding(icao: str, cycle: datetime) -> list[Item]:
    for wmo, name, dist, brg, _la, _lo in upper_air_sites.sites_for(icao):
        try:
            got = soundings.resolve_source(wmo, cycle)
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
    return charts.skewt(prof), prof.url, launched, None


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


def _fetch_taf(icao: str):
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

def capture(con, item: Item, res: Result, *, root, refresh: bool) -> None:
    """Fetch one item unless the archive already holds it, then index it.

    The skip is what makes an hourly sweep affordable: the 22 stations issuing at 11Z share
    one CONUS water-vapour image and one GFS panel set, so the first station pays and the
    other 21 write a manifest row each."""
    res.manifest.append((item.kind, item.identity, item.requested_utc))
    if item.fetch is None:
        res.skipped.append(f"{item.kind} {item.identity} ({item.note or 'no fetcher'})")
        return
    if not refresh and store.artifact_key(con, item.kind, item.identity,
                                          item.requested_utc) is not None:
        res.reused += 1
        return
    try:
        data, url, served, note = item.fetch()
    except Exception as e:  # noqa: BLE001 -- one dead product must not lose the whole cycle
        # Contract rule 2: a silent miss is not a capture. Record it loudly and move on.
        res.failed.append(f"{item.kind} {item.identity} ({type(e).__name__}: {e})")
        return
    if not data:
        res.failed.append(f"{item.kind} {item.identity} (provider returned 0 bytes)")
        return
    sha, mime, n, is_new = artifacts.put(data, root=root)
    store.insert_artifact(con, sha256=sha, kind=item.kind, mime=mime, n_bytes=n,
                          first_seen_utc=datetime.now(timezone.utc))
    store.insert_artifact_key(con, kind=item.kind, identity=item.identity,
                              requested_utc=item.requested_utc, sha256=sha,
                              served_utc=served, source_url=url, provider=item.provider,
                              note=note or item.note)
    res.captured += 1
    if is_new:
        res.bytes_new += n
    else:
        res.deduped += 1


def archive_station(con, icao: str, cycle: datetime, *, root, refresh: bool,
                    **plan_kw) -> Result:
    res = Result()
    try:
        items = plan(icao, cycle, **plan_kw)
    except Exception as e:  # noqa: BLE001 -- an unresolvable station is one station, not the run
        res.failed.append(f"plan {icao} ({type(e).__name__}: {e})")
        return res
    for item in items:
        if item.expand is None:
            capture(con, item, res, root=root, refresh=refresh)
            continue
        try:
            expanded = item.expand()
        except Exception as e:  # noqa: BLE001
            res.failed.append(f"{item.kind} {item.identity} expand ({type(e).__name__}: {e})")
            continue
        if not expanded:
            # Legitimately nothing to capture (no site flew, no frames indexed) -- a SKIP, not
            # a failure, but never silent: rule 2 only forbids recording a miss as a capture.
            res.skipped.append(f"{item.kind} {item.identity} (nothing available to capture)")
            continue
        for sub in expanded:
            capture(con, sub, res, root=root, refresh=refresh)
    store.insert_manifest(con, icao, cycle, res.manifest)
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
    con = store.connect_archive(index_path)
    try:
        store.init_archive_schema(con)
        totals = Result()
        if args.all_regions:
            # Before the stations, so a region a station also wants is fetched once here and
            # merely reused below rather than the other way round.
            extra = Result()
            for it in all_region_items(cycle):
                capture(con, it, extra, root=root, refresh=args.refresh)
            extra.manifest.clear()                  # deliberately no manifest rows -- see above
            totals.captured += extra.captured
            totals.reused += extra.reused
            totals.deduped += extra.deduped
            totals.bytes_new += extra.bytes_new
            totals.failed += extra.failed
            print(f"all-regions: {extra.captured:3d} captured  {extra.reused:3d} reused  "
                  f"{len(extra.failed):2d} failed  {extra.bytes_new / 1e6:6.1f} MB new")
        for icao in icaos:
            res = archive_station(con, icao, cycle, root=root, refresh=args.refresh, **plan_kw)
            totals.captured += res.captured
            totals.reused += res.reused
            totals.deduped += res.deduped
            totals.bytes_new += res.bytes_new
            totals.skipped += res.skipped
            totals.failed += res.failed
            print(f"{icao}: {res.captured:3d} captured  {res.reused:3d} reused  "
                  f"{res.deduped:3d} deduped  {len(res.skipped):2d} skipped  "
                  f"{len(res.failed):2d} failed  {res.bytes_new / 1e6:6.1f} MB new")
        stats = store.archive_stats(con)
    finally:
        con.close()

    print(f"\ncaptured {totals.captured}, reused {totals.reused}, deduped {totals.deduped}, "
          f"{totals.bytes_new / 1e6:.1f} MB new bytes")
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
