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

import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Anchor the cache at the repo root (like config.py), NOT the cwd, so a job whose
# cwd is elsewhere still finds pre-staged images instead of writing a stray dir.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "soundings"

# Be polite to free public providers: space requests so a multi-site loop can't
# fire back-to-back. Module-level on purpose, like iem.py / awc.py.
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request = 0.0

# A descriptive agent -- some providers reject the default urllib user-agent.
_UA = "artificial-forecaster/0.1 (research; contact wquinten@proton.me)"


def synoptic_time(when: datetime | None = None) -> datetime:
    """Most recent radiosonde synoptic hour (00Z or 12Z) at or before `when`
    (default: now), as naive UTC to match the store's tz contract. Soundings only
    exist at 00/12Z, so any wall-clock time must snap down to one of them."""
    now = when or datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(hour=12 if now.hour >= 12 else 0, minute=0, second=0, microsecond=0)


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
