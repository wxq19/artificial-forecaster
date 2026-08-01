"""Upper-air sounding image client -- live skew-T fetch seam.

Sibling to awc.py / iem.py: a network data-source client that fetches PRE-RENDERED
skew-T images from public providers and hands back raw bytes. It owns no matplotlib
(we fetch pixels, we do not draw them -- charts.py stays the only matplotlib file),
no SQL, and no DuckDB. A forecaster reads these exact products, so feeding the model
the same image keeps the human-vs-model comparison honest.

Two observed-sounding providers, both radiosonde (RAOB) sites at 00Z/12Z ONLY:
  - SPC     (spc.noaa.gov/exper/soundings) -- SHARPpy-analyzed GIF, richer annotation.
  - Wyoming (weather.uwyo.edu)             -- classic skew-T PNG.
Each provider names stations in its OWN id space (SPC: a 3-letter site like MPX/OUN,
or a WMO number; Wyoming: a WMO number like 72649), so the caller passes the id that
matches `source`. A cross-source id map can come later.

Air-gap note (SuperCloud compute nodes have no internet): every fetch is
cache-aware so a pre-staged image can replay offline. The cache is OPT-IN
(use_cache) -- prototyping is live-first; we archive deliberately once we know what
we actually want to keep.
"""

import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Anchor the cache at the repo root (like config.py), NOT the cwd, so a job whose
# cwd is elsewhere still finds pre-staged images instead of writing a stray dir.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "soundings"

# Be polite to free public providers: space requests so a multi-site loop can't
# fire back-to-back. Module-level on purpose, like iem.py / awc.py.
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0

# A radiosonde posts ~60-90 min after its 00Z/12Z launch. Back off this much before
# snapping so a call just after the hour doesn't target a sounding not yet on the server
# (a 404 that reads as a bad site id). Same pattern as the wxmaps post lag.
_POST_LAG_H = 2.0

# A descriptive agent -- some providers reject the default urllib user-agent.
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"


def synoptic_time(when: datetime | None = None) -> datetime:
    """Most recent radiosonde synoptic hour (00Z or 12Z) that has had time to POST, at
    or before `when` (default: now), as naive UTC to match the store's tz contract.
    Soundings exist only at 00/12Z, so we snap down to one of them -- but only after
    backing off _POST_LAG_H, so a 12:30Z call still targets the prior 00Z (the 12Z
    image is not up yet) instead of 404ing.

    IDEMPOTENT: a time that IS already a synoptic hour is returned unchanged, so applying
    this twice cannot walk the answer backwards. That is not defensive tidying -- it was a
    live defect. `fetch_skewt` snapped, then handed the snapped time to `skewt_url` and
    `cache_path`, which each snapped AGAIN; with the old lag-then-snap the second pass
    stepped back a further slot, so the receipt, the cited URL and the delivered image
    were three different soundings, 24h apart at the ends. Nothing downstream would have
    caught it, and a frozen archive preserves that mislabel permanently.

    The trade this makes deliberately: passing exactly 12:00Z now MEANS 12Z, even moments
    after the hour when the image may not be posted (the caller gets an honest 404 rather
    than yesterday's sounding under today's label). That is the behaviour an archiver
    wants -- name the sounding you meant."""
    t = when or datetime.now(timezone.utc).replace(tzinfo=None)
    if t.hour in (0, 12) and (t.minute, t.second, t.microsecond) == (0, 0, 0):
        return t
    t -= timedelta(hours=_POST_LAG_H)
    return t.replace(hour=12 if t.hour >= 12 else 0, minute=0, second=0, microsecond=0)


def _spc_url(site: str, t: datetime) -> str:
    # SPC directory is 2-digit-year YYMMDDHH; per-station file is <SITE>.gif.
    return f"https://www.spc.noaa.gov/exper/soundings/{t:%y%m%d%H}_OBS/{site.upper()}.gif"


def _wyoming_url(wmo: str, t: datetime) -> str:
    # The wsgi page is an HTML wrapper; the image itself is a stable path keyed by
    # YYYYMMDDHH.<WMO>.skewt.png -- fetch that directly (one request, no HTML parse).
    return f"https://weather.uwyo.edu/upperair/imgs/{t:%Y%m%d%H}.{wmo}.skewt.png"


# source -> (url builder, file extension). The extension keeps the cache filename
# truthful; the caller sniffs the real mime from the bytes (SPC=GIF, Wyoming=PNG).
_SOURCES = {
    "spc": (_spc_url, "gif"),
    "wyoming": (_wyoming_url, "png"),
}


def skewt_url(site: str, when: datetime | None = None, *, source: str = "spc") -> str:
    """The exact provider image URL for (site, synoptic time). Exposed so a caller
    can cite provenance in a log/receipt without re-fetching the image."""
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(_SOURCES)}")
    builder, _ = _SOURCES[source]
    return builder(site, synoptic_time(when))


def cache_path(site: str, when: datetime | None = None, *, source: str = "spc") -> Path:
    """Where fetch_skewt(..., use_cache=True) stores this image. Exposed so a caller
    can point a reviewer at the exact file the model will read."""
    _, ext = _SOURCES[source]
    return _CACHE_DIR / f"{source}_{site.upper()}_{synoptic_time(when):%Y%m%d%H}.{ext}"


# --- BUFR source: availability-driven, not synoptic-snapped -------------------------
#
# WHY THIS EXISTS. The two image providers above serve only the sites they pre-render, and
# `synoptic_time()` assumes every ascent is at 00Z or 12Z. Both assumptions fail outside
# CONUS: the legacy Wyoming image index has no South American stations at all, and real
# stations launch OFF-CYCLE. Measured 2026-07-28: 87155 Resistencia launched at 15Z, and its
# yearly inventory shows 1500 recurring through the record. **An off-cycle ascent is
# released BECAUSE something is happening** -- a special ascent ahead of severe weather, the
# same reason CONUS sites add 06Z/18Z. Snapping to the nearest synoptic hour therefore does
# not just miss a profile, it systematically discards the most informative ones.
#
# So this path never snaps. It asks the provider what exists (`inventory`, ONE request per
# station-year, which is why enumerating is affordable) and takes the newest launch at or
# before the cutoff. Note the endpoint also MOVED: the legacy `cgi-bin/sounding` now 404s.
_BUFR_URL = "https://weather.uwyo.edu/wsgi/sounding"

# BUFR serves TEXT only -- PNG:SKEWT/GIF:SKEWT return a 2.6 KB HTML wrapper with no image
# (re-confirmed 2026-07-28), so we pull the CSV and render through charts.skewt, the same
# text-in/chart-out path the BUFKIT profiles used before they were retired.
_BUFR_CSV_COLS = {"pressure_hPa": "pres", "geopotential height_m": "hght",
                  "temperature_C": "tmpc", "dew point temperature_C": "dwpc",
                  "wind direction_degree": "drct", "wind speed_m/s": "wspd_ms"}
_MS_TO_KT = 1.9438444924406


@dataclass(frozen=True)
class ObsProfile:
    """An OBSERVED radiosonde ascent, duck-typed for charts.skewt (same field names as
    modeldata.FcstProfile). `title` is set so the chart cannot be captioned 'forecast'."""
    station: str
    launched: datetime            # the ACTUAL launch time, naive UTC
    lat: float
    lon: float
    pres: list[float]             # hPa
    tmpc: list[float]             # C
    dwpc: list[float]             # C
    drct: list[float]             # deg
    sknt: list[float]             # kt (CSV is m/s -- converted here)
    hght: list[float]             # m
    n_raw: int                    # levels before thinning (ascent data is ~1 s resolution)
    indices: dict
    url: str
    model: str = "RAOB"
    fhr: int = 0

    @property
    def title(self) -> str:
        return (f"OBSERVED radiosonde skew-T  |  {self.station}  "
                f"launched {self.launched:%Y-%m-%d %H:%MZ}  "
                f"({len(self.pres)} of {self.n_raw} levels)")

    @property
    def valid(self) -> str:
        return f"{self.launched:%y%m%d/%H%M}"

    @property
    def run(self) -> datetime:
        return self.launched


# The provider carries the SAME station under different upstream feeds, and a site present in
# one can be absent from the other. `src=BUFR` is the high-resolution ~1 s ascent; `src=FM35`
# is the traditional TEMP bulletin (mandatory + significant levels, ~100-200 of them). Both
# answer TEXT:CSV with the same column names, so one parser reads either.
#
# THIS IS NOT A NICETY. Measured 2026-07-28: 47646 TATENO -- the nearest radiosonde to RJTY --
# returns HTTP 400 under BUFR and 462 launches for 2026 under FM35. A BUFR-only client reports
# Japan as having no upper-air data at all, which is false and looks exactly like a dead site.
UWYO_SRCS = ("BUFR", "FM35")


def _uwyo_url(wmo: str, when: datetime | None, kind: str, src: str = "BUFR") -> str:
    q = {"id": str(wmo), "type": kind, "src": src}
    q["datetime"] = (when or datetime.now(timezone.utc).replace(tzinfo=None)).strftime("%Y-%m-%d %H:00:00")
    return f"{_BUFR_URL}?{urllib.parse.urlencode(q)}"


def inventory(wmo: str, year: int, *, src: str = "BUFR") -> list[datetime]:
    """Every launch time this station has on record for `year` under `src`, newest last.

    ONE request per station-year. The inventory page embeds a full `datetime=` in every
    cell's link, so this reads the provider's own index rather than probing a time grid --
    which is what makes off-cycle ascents discoverable at all (probing 00/12Z can only ever
    confirm 00/12Z)."""
    html = _get(_uwyo_url(wmo, datetime(year, 1, 1), "INVENTORY", src)).decode("utf-8", "replace")
    seen = {datetime.strptime(m, "%Y-%m-%d %H:%M:%S")
            for m in re.findall(r"datetime=(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", html)}
    return sorted(t for t in seen if t.year == year)


def available_times(wmo: str, before: datetime | None = None, *,
                    back_h: float = 48.0, src: str = "BUFR") -> list[datetime]:
    """Launch times in (before - back_h, before], oldest first. Spans the new-year
    boundary by reading the previous year's inventory too when the window crosses it."""
    end = before or datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(hours=back_h)
    times = inventory(wmo, end.year, src=src)
    if start.year != end.year:
        times = inventory(wmo, start.year, src=src) + times
    return [t for t in times if start < t <= end]


def latest_time(wmo: str, before: datetime | None = None, *,
                back_h: float = 48.0, src: str = "BUFR") -> datetime | None:
    """The newest ascent at or before `before` (default now), or None inside the window.

    Replaces `synoptic_time()` for this source. It applies NO post-lag: a launch only
    appears in the inventory once the provider has it, so presence IS availability --
    whereas the image providers need a lag precisely because nothing tells us."""
    got = available_times(wmo, before, back_h=back_h, src=src)
    return got[-1] if got else None


def resolve_source(wmo: str, before: datetime | None = None, *,
                   back_h: float = 48.0) -> tuple[str, datetime] | None:
    """(src, launch time) for the newest ascent this site actually has, or None.

    Tries the feeds in UWYO_SRCS order and returns the FIRST that answers. A site absent from
    BUFR is not a site with no soundings -- 47646 TATENO 400s under BUFR and has 462 launches
    under FM35 -- so every caller that wants 'the latest ascent here' must ask this, not
    latest_time() alone. Costs one extra request only when the first feed is empty."""
    for src in UWYO_SRCS:
        try:
            t = latest_time(wmo, before, back_h=back_h, src=src)
        except Exception:  # noqa: BLE001 -- a 400/500 on one feed just means try the next
            continue
        if t is not None:
            return src, t
    return None


def last_known_time(wmo: str, *, src: str | None = None) -> datetime | None:
    """The newest launch ON RECORD for this site, ignoring the 48 h window (None = nothing).

    Exists to tell 'this id is not a radiosonde site' apart from 'this site is quiet right
    now'. `inventory()` scrapes the page and returns an empty list for BOTH, so `latest_time`
    alone cannot distinguish them -- and reporting an unknown id as a real site that is not
    reporting states a fact that is false. Reads last year too, so the first days of January
    do not look like a dead site, and ALL feeds unless `src` pins one -- otherwise a site that
    lives only in FM35 would be declared nonexistent."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for source in ((src,) if src else UWYO_SRCS):
        for year in (now.year, now.year - 1):
            try:
                if got := inventory(wmo, year, src=source):
                    return got[-1]
            except Exception:  # noqa: BLE001 -- one dead feed must not mask a live one
                continue
    return None


def _thin(rows: list[dict], *, sfc_step: float = 5.0, upper_step: float = 20.0,
          boundary_depth: float = 150.0) -> list[dict]:
    """Reduce ~1 s ascent data (3,000-4,500 levels) to a plottable profile.

    Pressure-binned rather than every-Nth so the result does not depend on ascent rate.
    The lowest `boundary_depth` hPa is kept ~4x finer: inversions, the fog/stratus layer
    and the LLJ all live there, and they are exactly what a TAF turns on."""
    if not rows:
        return []
    out = [rows[0]]
    p0 = rows[0]["pres"]
    for r in rows[1:]:
        step = sfc_step if (p0 - r["pres"]) <= boundary_depth else upper_step
        if out[-1]["pres"] - r["pres"] >= step:
            out.append(r)
    return out


def fetch_profile(wmo: str, when: datetime, *, use_cache: bool = False,
                  src: str = "BUFR") -> ObsProfile:
    """Fetch + parse ONE observed ascent as a plottable profile. `when` must be an actual
    launch time (see latest_time/resolve_source) -- passed through verbatim, never snapped.

    `src` must be the feed the time came FROM: the two feeds carry different launch sets, so
    a BUFR fetch at an FM35-only time returns no CSV. The cache key carries it for the same
    reason."""
    url = _uwyo_url(wmo, when, "TEXT:CSV", src)
    cache_file = _CACHE_DIR / f"{src.lower()}_{wmo}_{when:%Y%m%d%H%M}.csv"
    if use_cache and cache_file.exists():
        text = cache_file.read_text()
    else:
        text = re.sub(r"<[^>]+>", "", _get(url).decode("utf-8", "replace"))
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    hdr_i = next((i for i, ln in enumerate(lines) if "pressure_hPa" in ln), None)
    if hdr_i is None:
        raise ValueError(f"no CSV profile for {wmo} at {when:%Y-%m-%dT%H:%MZ}")
    cols = [c.strip() for c in lines[hdr_i].split(",")]
    idx = {name: cols.index(col) for col, name in _BUFR_CSV_COLS.items() if col in cols}
    missing = set(_BUFR_CSV_COLS.values()) - set(idx)
    if missing:
        raise ValueError(f"BUFR CSV for {wmo} is missing {sorted(missing)}")
    lat_i, lon_i = cols.index("latitude"), cols.index("longitude")

    rows: list[dict] = []
    lat = lon = None
    for ln in lines[hdr_i + 1:]:
        f = ln.split(",")
        if len(f) <= max(idx.values()):
            continue
        # PER FIELD, not all-or-nothing. The old form built the whole dict in one
        # comprehension inside a try, so ONE blank column discarded the entire level --
        # and radiosonde humidity sensors stop reporting in the dry stratosphere, so the
        # blank column is the DEWPOINT on most ascents. Measured on 47646 (2026-07-29 12Z):
        # 120 rows reaching 6 hPa, 52 with a dewpoint (top 250 hPa) and 68 without
        # (239 -> 6 hPa). We were throwing away good temperature, wind and height for 68 of
        # 120 levels, so every observed sounding was truncated at the humidity ceiling and
        # lost its tropopause and jet level.
        #
        # Missing becomes NaN rather than None: numpy and metpy take it natively, and
        # matplotlib BREAKS a line at NaN instead of interpolating across the gap -- so the
        # dewpoint trace now ends where the sonde stopped measuring humidity, which is what
        # a real skew-T looks like, while temperature and wind carry on up.
        r = {}
        for k, i in idx.items():
            try:
                r[k] = float(f[i])
            except ValueError:
                r[k] = float("nan")
        # A level with no PRESSURE has nowhere to go on the plot -- that one is still fatal
        # to the row. Everything else is a gap, not a reason to drop the level.
        if r["pres"] != r["pres"]:      # NaN
            continue
        if lat is None and len(f) > max(lat_i, lon_i):
            # Length-checked separately from the profile columns: latitude/longitude can sit
            # at a HIGHER index than any profile column, and an IndexError here escapes the
            # ValueError guard and aborts the whole ascent over a position we can live without.
            try:
                lat, lon = float(f[lat_i]), float(f[lon_i])
            except ValueError:
                pass
        r["sknt"] = r.pop("wspd_ms") * _MS_TO_KT
        rows.append(r)
    if not rows:
        raise ValueError(f"BUFR CSV for {wmo} at {when:%Y-%m-%dT%H:%MZ} had no data rows")
    rows.sort(key=lambda r: -r["pres"])
    kept = _thin(rows)
    return ObsProfile(
        station=str(wmo), launched=when, lat=lat or 0.0, lon=lon or 0.0,
        pres=[r["pres"] for r in kept], tmpc=[r["tmpc"] for r in kept],
        dwpc=[r["dwpc"] for r in kept], drct=[r["drct"] for r in kept],
        sknt=[r["sknt"] for r in kept], hght=[r["hght"] for r in kept],
        n_raw=len(rows), indices={}, url=url)


# Where a real ascent ends. Set from the full archive, not a sample: classifying all 293
# archived soundings on 2026-07-31 put their top pressures in three groups -- 4-85 hPa
# (real burst, 280 records), a tight 101-109 hPa cluster (the staged release / the FM35
# convention, 9), and 173/284/524 hPa (caught genuinely mid-flight, 3).
#
# AN EARLIER VALUE OF 30 hPa WAS WRONG and is recorded here so it is not reintroduced. It
# came from an 18-site sample that happened to burst at 4.0-11.0 hPa, and it misclassified
# 19 real ascents -- everything between 30 and 85 hPa -- as truncated. The error fell almost
# entirely on the FM35 feed: 49% of FM35 records failed it against 5.4% of BUFR, because
# FM35 is the traditional TEMP bulletin whose parts A and B terminate at 100 hPa by
# convention, so a 100 hPa FM35 record is frequently the WHOLE bulletin and not a fragment.
# 95 hPa sits above every observed real burst and below the staging cluster.
COMPLETE_ASCENT_MAX_HPA = 95.0

# The Wyoming feed PUBLISHES IN STAGES, and this is the fact that matters: the truncated
# captures cluster at 100-109 hPa rather than scattering -- a staging point, not a balloon
# still climbing. That is why elapsed time does NOT predict completeness (one capture 82 min
# after launch was complete; one at 129 min was still the staged version), and why a
# clock-based gate cannot work at any value. Test the DATA, not the time.
_STAGED_RELEASE_HPA = 100.0
_STAGED_RELEASE_BAND_HPA = (95.0, 115.0)


def ascent_is_complete(prof: "ObsProfile") -> bool:
    """True when this ascent reached burst rather than a staged intermediate release.

    Call this before FREEZING an ascent into the archive: `ON CONFLICT DO NOTHING` makes the
    first copy permanent, so accepting a staged release keeps half an ascent forever, and the
    half that is lost is the upper troposphere and stratosphere -- the tropopause and jet
    level a TAF's turbulence and icing reasoning depends on."""
    tops = [p for p in prof.pres if p is not None and p == p]     # p == p drops NaN
    return bool(tops) and min(tops) <= COMPLETE_ASCENT_MAX_HPA


def ascent_stage_note(prof: "ObsProfile") -> str | None:
    """Short reason an ascent looks incomplete, or None when it is complete."""
    if ascent_is_complete(prof):
        return None
    tops = [p for p in prof.pres if p is not None and p == p]
    if not tops:
        return "no usable pressure levels"
    top = min(tops)
    lo, hi = _STAGED_RELEASE_BAND_HPA
    staged = lo <= top <= hi
    return (f"stops at {top:.0f} hPa"
            + (f" (the provider's {_STAGED_RELEASE_HPA:.0f} hPa staged release, not burst)"
               if staged else " (caught mid-flight, far below any burst altitude)"))


# How long to keep retrying before accepting an ascent that never reached burst. The
# completeness test alone must NOT be the whole rule: a balloon really can pop early, and a
# hard pressure gate would refuse that ascent at every sweep forever -- discarding data the
# human forecaster genuinely had, which is a worse failure than storing a short flight.
# So the rule is COMPLETE **OR** OLD ENOUGH, and the age clause guarantees termination.
STAGED_RETRY_WINDOW_H = 4.0


def ascent_is_final(prof: "ObsProfile", *, now: datetime | None = None) -> bool:
    """True when this is worth freezing: it reached burst, OR it is old enough that this is
    all the provider is ever going to publish.

    Two failure modes, and they pull opposite ways. Freezing too EARLY keeps a staged
    release forever (the provider publishes an intermediate at 100 hPa, and one sampled
    capture was still staged 129 minutes after launch). Refusing on pressure ALONE never
    terminates for a genuine early burst. The disjunction bounds both: a complete record is
    taken at once, and anything else is retried only until the window closes."""
    if ascent_is_complete(prof):
        return True
    ref = now or datetime.now(timezone.utc).replace(tzinfo=None)
    launched = prof.launched
    if launched.tzinfo is not None:
        launched = launched.astimezone(timezone.utc).replace(tzinfo=None)
    return (ref - launched).total_seconds() / 3600.0 >= STAGED_RETRY_WINDOW_H


def _get(url: str) -> bytes:
    """GET raw bytes, spacing requests politely (module-level throttle)."""
    global _last_request
    if (wait := _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)) > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_skewt(
    site: str,
    when: datetime | None = None,
    *,
    source: str = "spc",
    use_cache: bool = False,
) -> bytes:
    """Fetch a pre-rendered observed skew-T image and return raw bytes.

    `source` selects the provider ('spc' default -- richer analysis -- or
    'wyoming'); `site` must be an id in THAT provider's namespace. `when` snaps to
    the latest 00/12Z synoptic hour. With use_cache, a hit replays from disk (the
    air-gap / reproducibility path) and a miss is written after fetching."""
    t = synoptic_time(when)
    url = skewt_url(site, t, source=source)   # validates source (raises on unknown)
    cache_file = cache_path(site, t, source=source)

    if use_cache and cache_file.exists():
        return cache_file.read_bytes()
    data = _get(url)
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
    return data
