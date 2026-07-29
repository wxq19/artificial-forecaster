"""Self-test for the artifact store and the archive planner.

    uv run python scripts/test_archive_run.py

OFFLINE by design: every provider fetch is stubbed and `awc.station_latlon` is pinned to
known positions, so this runs on the air-gapped node and in CI. It tests the parts that a
live capture cannot check cheaply -- deduplication, idempotence, the manifest join, and the
resolution rules that decide WHAT a station is entitled to.

The planner checks are the anti-drift guard. `archive_run` mirrors presentation rules that
live in `tools.py` (OSPO has no geocolor, Meteosat publishes no water vapour, the radar
cascade's in-network test). If one of those changes in tools.py and not here, the archive
silently stops matching what the agent would be served -- and under serve-from-archive that
is a hole nobody notices until a replay is short an image.
"""

import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecaster import artifacts, awc, imagery, store  # noqa: E402

ar = None       # the script under test, loaded by path in __main__ (scripts/ is not a package)

PASS = 0
FAIL = 0
CYCLE = datetime(2026, 7, 29, 1)

# Real positions, so the resolvers under test see the geography they were written for.
POS = {
    "KWRI": (40.0155, -74.5917),     # CONUS, northeast sector, WSR-88D 17 km
    "KMIB": (48.4156, -101.3581),    # northern Rockies; radar gap, WSR-88D 35 km
    "RJTY": (35.7486, 139.3486),     # Himawari via OSPO -- no geocolor
    "ETAR": (49.4367, 7.6003),       # Meteosat -- no water vapour
    "SAWC": (-45.7853, -67.4655),    # Patagonia -- no radar at all
    # ON a real WSR-88D but OUTSIDE IEM's composite: the distance guard passes, the picture is
    # empty. Kadena 0 km, Kunsan 3 km, Osan 26 km, Andersen 20 km, Humphreys 35 km.
    "RKSO": (37.091, 127.03),
    "PGUA": (13.583, 144.918),
    "RODN": (26.356, 127.768),
    "RKJK": (35.9, 126.618),
    "RKSG": (36.962, 127.031),
}


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name}\n     got  {got!r}\n     want {want!r}")


def check_true(name: str, cond, detail: str = "") -> None:
    check(name, bool(cond) or detail or False, True)


# --- 1. content-addressed blobs -------------------------------------------------------

def test_blobs(root: Path) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"body"
    sha1, mime1, n1, new1 = artifacts.put(png, root=root)
    sha2, _mime2, _n2, new2 = artifacts.put(png, root=root)
    check("blob: identical bytes give one address", sha1, sha2)
    check("blob: first write is new", new1, True)
    check("blob: second write is a no-op", new2, False)
    check("blob: mime sniffed from magic, not the URL", mime1, "image/png")
    check("blob: size recorded", n1, len(png))
    check("blob: round-trips", artifacts.get(sha1, "png", root=root), png)
    check("blob: fans out on the first two hex chars",
          artifacts.blob_path(sha1, "png", root=root).parent.name, sha1[:2])
    check("sniff: jpeg", artifacts.sniff(b"\xff\xd8\xff\xe0junk")[1], "jpg")
    check("sniff: gif (IEM national degrades PNG -> GIF)",
          artifacts.sniff(b"GIF89a...")[0], "image/gif")
    check("sniff: unknown falls back to text (TAF bulletins, BUFR CSV)",
          artifacts.sniff(b"TAF KWRI 290200Z")[0], "text/plain")
    # No .part file may survive a successful write -- a short file at a full-sha name would
    # be trusted forever by every later run.
    check("blob: no partial files left behind",
          list((root / "blobs").rglob("*.part")), [])


# --- 2. the index ---------------------------------------------------------------------

def test_index(db: str) -> None:
    con = store.connect_archive(db)
    store.init_archive_schema(con)
    store.init_archive_schema(con)                       # idempotent
    t = datetime(2026, 7, 29, 1)

    new1 = store.insert_artifact(con, sha256="a" * 64, kind="satellite", mime="image/png",
                                 n_bytes=10, first_seen_utc=t)
    new2 = store.insert_artifact(con, sha256="a" * 64, kind="satellite", mime="image/png",
                                 n_bytes=10, first_seen_utc=t)
    check("index: artifact insert is idempotent", (new1, new2), (True, False))

    k1 = store.insert_artifact_key(con, kind="satellite", identity="conus_east/water_vapor",
                                   requested_utc=t, sha256="a" * 64, provider="goes_star",
                                   source_url="http://example/wv.png")
    k2 = store.insert_artifact_key(con, kind="satellite", identity="conus_east/water_vapor",
                                   requested_utc=t, sha256="b" * 64, provider="goes_star")
    check("index: key insert is idempotent", (k1, k2), (True, False))
    row = store.artifact_key(con, "satellite", "conus_east/water_vapor", t)
    # Rule 5: a re-capture that returned DIFFERENT bytes must NOT overwrite. What the model
    # saw is the first capture.
    check("index: re-capture never overwrites the first blob", row["sha256"], "a" * 64)
    check("index: key joins its blob record", row["mime"], "image/png")

    # Two stations, same cycle, sharing one image: the whole economy of the archive.
    store.insert_manifest(con, "KWRI", t, [("satellite", "conus_east/water_vapor", t)])
    n = store.insert_manifest(con, "KDOV", t, [("satellite", "conus_east/water_vapor", t)])
    check("index: a second station shares the artifact", n, 1)
    check("index: manifest resolves to bytes",
          [(r["identity"], r["sha256"]) for r in store.manifest_for(con, "KDOV", t)],
          [("conus_east/water_vapor", "a" * 64)])
    check("index: manifest is per station",
          len(store.manifest_for(con, "KMIB", t)), 0)
    stats = store.archive_stats(con)
    check("index: dedup counts keys per blob", (stats["blobs"], stats["keys"]), (1, 1))

    # THREE distinct times per row. fetched_utc is our wall clock at the GET and exists
    # because a sweep runs 30-40 min: an artifact labelled 11:00Z may be pulled at 11:38Z,
    # and GOES STAR's served_utc is NULL by design, so nothing else records that skew.
    store.insert_artifact_key(con, kind="satellite", identity="northeast/geocolor",
                              requested_utc=t, sha256="a" * 64, provider="goes_star",
                              fetched_utc=datetime(2026, 7, 29, 1, 38))
    row = store.artifact_key(con, "satellite", "northeast/geocolor", t)
    check("index: fetched_utc records the real GET time, apart from the cycle label",
          (row["requested_utc"], row["fetched_utc"], row["served_utc"]),
          (t, datetime(2026, 7, 29, 1, 38), None))

    # Serve-time snapping: a loop cadence that was never captured on that exact grid.
    for m in (0, 20, 40):
        store.insert_artifact_key(con, kind="loop_frame", identity="northeast/geocolor",
                                  requested_utc=datetime(2026, 7, 29, 1, m),
                                  sha256="a" * 64, provider="goes_star")
    near = store.nearest_artifact_key(con, "loop_frame", "northeast/geocolor",
                                      datetime(2026, 7, 29, 1, 25))
    check("index: nearest returns the closest frame", near["requested_utc"],
          datetime(2026, 7, 29, 1, 20))
    check("index: nearest reports the snap distance so a receipt can say so",
          round(near["snap_minutes"], 1), 5.0)
    check("index: nearest respects a max gap",
          store.nearest_artifact_key(con, "loop_frame", "northeast/geocolor",
                                     datetime(2026, 7, 29, 6, 0), max_minutes=30), None)
    check("index: exact miss is a miss, not a silent snap",
          store.artifact_key(con, "loop_frame", "northeast/geocolor",
                             datetime(2026, 7, 29, 1, 25)), None)
    con.close()


# --- 3. the planner: what a station is entitled to -------------------------------------

def _plan_kinds(icao: str) -> dict[str, list[str]]:
    lat, lon = POS[icao]
    out: dict[str, list[str]] = {}
    for items in (ar._satellite_items(icao, lat, lon, CYCLE),
                  ar._loop_items(icao, lat, lon, CYCLE, frames=6, step_min=30),
                  ar._radar_items(icao, lat, lon, CYCLE),
                  ar._map_items(icao, lat, lon, CYCLE, (0, 24))):
        for i in items:
            out.setdefault(i.kind, []).append(i.identity)
    return out


def test_planner() -> None:
    awc.station_latlon = lambda s: POS[str(s).upper()]   # no network in a self-test

    kwri = _plan_kinds("KWRI")
    check("plan: water vapour widens to the synoptic scope, stills stay on the sector",
          sorted(kwri["satellite"]),
          ["conus_east/water_vapor", "northeast/geocolor", "northeast/infrared"])
    check("plan: KWRI gets a station radar view", "station/KWRI" in kwri["radar"], True)
    check("plan: analysis charts key without a domain, tt panels with one",
          ("surface_analysis" in kwri["map"], "us/gfs_500mb/f024" in kwri["map"]), (True, True))

    # KMIB sits in a gap between the curated radar regions but has a WSR-88D 35 km out. The
    # tool serves it a station view, so gating the group on the regional bbox would archive
    # nothing -- the defect this test exists to pin.
    kmib = _plan_kinds("KMIB")
    check("plan: KMIB (radar-region gap) still gets station + national radar",
          sorted(kmib["radar"]), ["national", "station/KMIB"])
    check("plan: KMIB resolves to northern_rockies, NOT the PACUS conus_west",
          "northern_rockies/geocolor" in kmib["satellite"], True)

    # OSPO has no geocolor; capturing it would store infrared bytes under a name they are not.
    rjty = _plan_kinds("RJTY")
    check("plan: OSPO Japan has no geocolor product",
          any("geocolor" in i for i in rjty["satellite"]), False)
    check("plan: RJTY gets NO radar (WSR-88D is US-only)", rjty.get("radar", []), [])

    # MTG publishes no water vapour at all; serving another channel under the name is the
    # mislabelling class the imagery honesty pass removed.
    etar = _plan_kinds("ETAR")
    check("plan: Meteosat has no water vapour",
          any("water_vapor" in i for i in etar["satellite"]), False)
    check("plan: Meteosat stills are station-centered, so they key per station",
          all(i.startswith("meteosat_point/ETAR/") for i in etar["satellite"]), True)

    # A NEARBY RADAR IS NOT COVERAGE. Found by QC 2026-07-29: RKSO/PGUA/RODN/RKJK/RKSG all sit
    # essentially ON a real WSR-88D (Kadena 0 km, Kunsan 3 km, Osan 26 km, Andersen 20 km,
    # Humphreys 35 km), so the 150 km guard passed and they were archived a correctly-framed
    # image with ZERO reflectivity -- captioned "NEXRAD Base Reflectivity" and therefore read as
    # "no convection". RJTY was safe only because its nearest radar is 1,090 km away, i.e. by
    # accident. Both the planner and tools._radar_for_station must apply the coverage test, or
    # the archive drifts from what the agent is served.
    check("imagery: the IEM composite covers CONUS/AK/HI/PR",
          [imagery.iem_composite_covers(*ll)
           for ll in ((40.0, -74.6), (61.2, -149.8), (21.5, -158.0), (18.4, -66.0))],
          [True, True, True, True])
    check("imagery: it does NOT cover Korea, Guam, Okinawa or Patagonia",
          [imagery.iem_composite_covers(*ll)
           for ll in ((37.1, 127.0), (13.6, 144.9), (26.4, 127.8), (-53.0, -70.8))],
          [False, False, False, False])
    check("plan: a station on an out-of-mosaic WSR-88D gets NO radar",
          [_plan_kinds(i).get("radar", []) for i in ("RKSO", "PGUA")], [[], []])

    sawc = _plan_kinds("SAWC")
    check("plan: Patagonia gets no radar", sawc.get("radar", []), [])
    check("plan: Patagonia resolves to the South America sector",
          "south_america_south/geocolor" in sawc["satellite"], True)
    check("plan: Patagonia charts are the samer domain",
          any(i.startswith("samer/") for i in sawc["map"]), True)

    # Loop identity must match the still's region so frames dedup across a sector's stations.
    check("plan: loop frames key on the sector, not the station",
          sorted(kwri["loop_frame"]),
          ["northeast/geocolor", "northeast/infrared", "northeast/water_vapor"])

    # Static artifacts key on a sentinel: a real timestamp would re-capture terrain hourly.
    check("plan: terrain is static-keyed",
          ar._static_items("KWRI", *POS["KWRI"])[0].requested_utc, ar.STATIC_UTC)

    # Deferred lookups must not touch a provider at plan time (--dry-run must stay offline).
    check("plan: sounding is deferred behind expand",
          (ar._sounding_items("KWRI", CYCLE)[0].fetch,
           ar._sounding_items("KWRI", CYCLE)[0].expand is not None), (None, True))


# --- 4. capture bookkeeping (stubbed fetchers) -----------------------------------------

def _capture(con, item, res, *, root, refresh: bool = False) -> None:
    """The real three-phase sequence, in one call. Mirrors archive_station so these checks
    stay honest about the order select -> fetch -> store actually runs in."""
    for got in ar.fetch_all(ar.select_all(con, [item], res, refresh=refresh), workers=1):
        ar.store_one(con, got, res, root=root)


def test_select_dedup(root: Path, db: str) -> None:
    """Two items on one key must be fetched ONCE.

    The serial loop got this free -- the first capture wrote the key and the rest read it back
    as reused. Selecting the whole batch before any fetch removes that guard, so the dedup has
    to be explicit or `--all-regions` (where many regions widen onto one water-vapour scope)
    would fetch the same image several times at once, racing on one cache path."""
    con = store.connect_archive(db)
    store.init_archive_schema(con)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return b"wv-bytes", "http://x", None, None

    dupes = [ar.Item(kind="satellite", identity="conus_east/water_vapor", requested_utc=CYCLE,
                     provider="goes_star", fetch=fetch) for _ in range(4)]
    res = ar.Result()
    todo = ar.select_all(con, dupes, res, refresh=False)
    check("select: four items on one key select once", len(todo), 1)
    check("select: the other three count as reused, not dropped", res.reused, 3)
    check("select: all four still reach the manifest", len(res.manifest), 4)
    for got in ar.fetch_all(todo, workers=4):
        ar.store_one(con, got, res, root=root)
    check("select: the provider is hit exactly once", calls["n"], 1)
    check("select: one key, one capture", (res.captured, res.deduped), (1, 0))
    con.close()


def test_capture(root: Path, db: str) -> None:
    con = store.connect_archive(db)
    store.init_archive_schema(con)
    res = ar.Result()
    shared = b"\x89PNG\r\n\x1a\nshared-bytes"

    def item(identity: str, data: bytes) -> ar.Item:
        return ar.Item(kind="satellite", identity=identity, requested_utc=CYCLE,
                       provider="goes_star", fetch=lambda: (data, "http://x", None, None))

    _capture(con, item("a/geocolor", shared), res, root=root)
    _capture(con, item("b/geocolor", shared), res, root=root)
    _capture(con, item("a/geocolor", shared), res, root=root)
    check("capture: two identities, one blob -> the second is deduped",
          (res.captured, res.deduped, res.reused), (2, 1, 1))
    check("capture: every item reaches the manifest even when reused",
          len(res.manifest), 3)

    bad = ar.Item(kind="map", identity="boom", requested_utc=CYCLE, provider="tt",
                  fetch=lambda: (_ for _ in ()).throw(ValueError("no such panel")))
    _capture(con, bad, res, root=root)
    check("capture: a dead product is recorded, not raised", len(res.failed), 1)

    empty = ar.Item(kind="map", identity="empty", requested_utc=CYCLE, provider="tt",
                    fetch=lambda: (b"", "http://x", None, None))
    _capture(con, empty, res, root=root)
    check("capture: zero bytes is a failure, not a capture (rule 2)", len(res.failed), 2)
    check("capture: a failed item writes no key",
          store.artifact_key(con, "map", "boom", CYCLE), None)

    # fetched_utc is stamped around the GET, not at the top of the sweep. It is the only record
    # of cycle-label drift for GOES STAR, whose served_utc is NULL by design.
    row = con.execute("SELECT served_utc, fetched_utc FROM artifact_keys "
                      "WHERE identity = 'a/geocolor'").fetchone()
    check("capture: fetched_utc is recorded even when served_utc is NULL",
          (row[0] is None, row[1] is not None), (True, True))
    con.close()


# --- 3b. the loop expand must not re-download per station -------------------------------

def test_loop_expand_memo() -> None:
    """The expand DOWNLOADS the frames, and it runs per STATION while the frames key per
    SECTOR. `select`'s dedup cannot save it -- that runs after the bytes are already pulled.
    Measured before the memo: 213 expands a sweep for 56 distinct identities, ~35 min of a
    56-66 min sweep spent re-fetching frames the archive already held."""
    import forecaster.imagery as I

    calls = {"n": 0}

    class _F:
        def __init__(self, t):
            self.time, self.label, self.url, self.data = t, "l", "http://x", b"frame"

    def fake_loop(lat, lon, product, *, frames, step_min, at):
        calls["n"] += 1
        return ([_F(CYCLE - timedelta(minutes=step_min * i)) for i in range(frames)],
                "src", "cov")

    real = I.satellite_loop
    I.satellite_loop = fake_loop
    ar.reset_loop_expand_cache()
    try:
        # Three CONUS stations in one sector: same identity, so ONE download between them.
        for _ in range(3):
            got = ar._expand_loop("northeast/geocolor", "geocolor", "goes_star",
                                  40.0, -74.6, CYCLE, 7, 10)
        check("loop memo: one sector expand serves every station in it", calls["n"], 1)
        check("loop memo: the cached expand still yields every frame", len(got), 7)

        # Meteosat identities carry the ICAO because its loops are station-CENTERED crops,
        # so they must STILL fetch per station.
        for icao in ("ETAR", "ETAD", "EGUN"):
            ar._expand_loop(f"meteosat_point/{icao}/geocolor", "geocolor",
                            "meteosat_eumetsat_wms", 49.4, 7.6, CYCLE, 7, 10)
        check("loop memo: station-centred Meteosat loops are NOT collapsed", calls["n"], 4)

        # A different cadence is a different product and must not be served from the memo.
        ar._expand_loop("northeast/geocolor", "geocolor", "goes_star",
                        40.0, -74.6, CYCLE, 10, 30)
        check("loop memo: a different (frames, step) re-expands", calls["n"], 5)
    finally:
        I.satellite_loop = real
        ar.reset_loop_expand_cache()


# --- 4a. rendering must not run concurrently --------------------------------------------

def test_render_is_serialized() -> None:
    """charts.py is pyplot, and pyplot's figure manager is process-global.

    `_fetch_sounding` is the one fetcher that DRAWS instead of downloading, so once fetches
    became concurrent it became the one that can corrupt state. Two threads inside pyplot can
    interleave into a single figure, which would hand a replay a skew-T captioned for another
    station -- the confident-label-over-wrong-content failure this project keeps meeting."""
    import threading

    depth = {"now": 0, "max": 0}
    guard = threading.Lock()

    def fake_skewt(_profile):
        with guard:
            depth["now"] += 1
            depth["max"] = max(depth["max"], depth["now"])
        time.sleep(0.05)                       # long enough for a real overlap to show
        with guard:
            depth["now"] -= 1
        return b"\x89PNG\r\n\x1a\nskewt"

    class _Prof:
        url = "http://sounding"

    real_skewt, real_profile = ar.charts.skewt, ar.soundings.fetch_profile
    ar.charts.skewt = fake_skewt
    ar.soundings.fetch_profile = lambda *a, **k: _Prof()
    try:
        items = [ar.Item(kind="sounding", identity=f"7250{i}/bufr", requested_utc=CYCLE,
                         provider="wyoming",
                         fetch=lambda w=f"7250{i}": ar._fetch_sounding(w, CYCLE))
                 for i in range(6)]
        ar.fetch_all(items, workers=6)
        check("render: never two threads inside pyplot at once", depth["max"], 1)
    finally:
        ar.charts.skewt, ar.soundings.fetch_profile = real_skewt, real_profile


# --- 4b. retry: a transient blip must not cost the artifact -----------------------------

def test_retry(root: Path, db: str) -> None:
    """The 2026-07-29 08Z sweep lost 18 stations to one DNS blip. A retry is the difference
    between a hiccup and an hour of data that never comes back."""
    con = store.connect_archive(db)
    store.init_archive_schema(con)
    delay = ar.RETRY_DELAY_S
    ar.RETRY_DELAY_S = 0.0                      # the pause is real; waiting for it in a test is not
    try:
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("[Errno -3] Temporary failure in name resolution")
            return b"recovered-bytes", "http://x", None, None

        res = ar.Result()
        _capture(con, ar.Item(kind="radar", identity="station/KDYS", requested_utc=CYCLE,
                              provider="iem", fetch=flaky), res, root=root)
        check("retry: a DNS blip is retried and the artifact survives",
              (calls["n"], res.captured, len(res.failed)), (2, 1, 0))

        # A 4xx and a ValueError are the provider saying the product is ABSENT. Retrying spends
        # a request to be told the same thing -- 27 of 29 failures that day were exactly this.
        check("retry: a 404 is not transient", ar._transient(HTTPError("u", 404, "x", None, None)),
              False)
        check("retry: a 503 is transient", ar._transient(HTTPError("u", 503, "x", None, None)),
              True)
        check("retry: 'no current TAF' is not transient", ar._transient(ValueError("no TAF")),
              False)

        seen = {"n": 0}

        def dead():
            seen["n"] += 1
            raise ValueError("AWC returned no current TAF for KVBG")

        res2 = ar.Result()
        _capture(con, ar.Item(kind="taf", identity="KVBG", requested_utc=CYCLE,
                              provider="awc", fetch=dead), res2, root=root)
        check("retry: an absent product is attempted exactly once",
              (seen["n"], len(res2.failed)), (1, 1))
    finally:
        ar.RETRY_DELAY_S = delay
        con.close()


# --- 4c. the index lock is released around the fetches ----------------------------------

def test_lock_is_waited_out(db: str) -> None:
    """A held index must make the sweep WAIT, not die.

    On 2026-07-29 at 23Z a read-only reporting script held index.duckdb, the sweep's fifth
    station could not open it, the exception escaped `archive_station`, and the run aborted
    having captured 4 of 71 stations -- 67 stations lost an hour of imagery that cannot be
    re-fetched. Releasing the lock quickly (test_lock_released_during_fetch) is not enough on
    its own: something else can still hold it while we need it."""
    import multiprocessing as mp

    con = store.connect_archive(db)
    store.init_archive_schema(con)
    con.close()

    def _holder(q):
        c = store.connect_archive(db)
        q.put(1)
        time.sleep(3)
        c.close()

    q = mp.Queue()
    p = mp.Process(target=_holder, args=(q,))
    p.start()
    q.get()
    time.sleep(0.3)
    try:
        store.connect_archive(db).close()
        check("lock: a bare connect really is blocked (guard is meaningful)", True, False)
    except Exception:  # noqa: BLE001
        check("lock: a bare connect really is blocked (guard is meaningful)", True, True)
    t0 = time.time()
    got = ar.connect_index(db)
    waited = time.time() - t0
    got.close()
    p.join()
    check("lock: connect_index waits for the holder instead of raising", waited > 1.0,
          True)
    # A non-lock failure must NOT be retried -- waiting 60 s to report the same bad path is
    # worse than failing at once.
    t0 = time.time()
    try:
        ar.connect_index("/nonexistent-dir-xyz/i.duckdb")
        check("lock: a non-lock error is not retried", "opened", "raised")
    except Exception:  # noqa: BLE001
        check("lock: a non-lock error is not retried", time.time() - t0 < 5.0, True)


def test_lock_released_during_fetch(root: Path, db: str) -> None:
    """DuckDB takes an EXCLUSIVE file lock -- while it is held, nothing else can open the
    index, not even read_only=True (verified on the Pi, 2026-07-29). A connection wrapped
    around the fetches therefore locks out the serve side for the whole sweep. This asserts
    the index is openable at the moment a fetch runs, which is the property that lets capture
    and replay coexist."""
    con = store.connect_archive(db)
    store.init_archive_schema(con)
    con.close()

    opened: list[bool] = []

    def probe():
        try:
            other = store.connect_archive(db, read_only=True)
            other.close()
            opened.append(True)
        except Exception:  # noqa: BLE001
            opened.append(False)
        return b"bytes", "http://x", None, None

    res = ar.Result()
    got = ar.fetch_all([ar.Item(kind="map", identity="probe", requested_utc=CYCLE,
                                provider="tt", fetch=probe)], workers=2)
    check("lock: the index is readable while a fetch is in flight", opened, [True])
    con = store.connect_archive(db)
    try:
        ar.store_one(con, got[0], res, root=root)
    finally:
        con.close()
    check("lock: the fetched item still indexes after the lock is retaken", res.captured, 1)


# --- 5. sounding feed fallback + loop cadence + region coverage -------------------------

def test_sounding_feeds() -> None:
    """BUFR-or-FM35. Stubbed: the real check is that a site absent from one feed is found in
    the other, which is what makes Japan visible at all (47646 400s under BUFR, 462 launches
    under FM35)."""
    import forecaster.soundings as S

    real = S.inventory
    fm35_only = {datetime(2026, 7, 29, 0)}

    def stub(wmo, year, *, src="BUFR"):
        if wmo in ("BUFRONLY", "72501"):            # 72501 = KWRI's real nearest site
            return sorted(fm35_only) if src == "BUFR" else []
        if wmo == "FM35ONLY":
            if src == "BUFR":
                raise OSError("HTTP Error 400: Bad Request")   # what 47646 really does
            return sorted(fm35_only)
        return []

    S.inventory = stub
    try:
        when = datetime(2026, 7, 29, 2)
        check("sounding: BUFR is preferred when both answer",
              S.resolve_source("BUFRONLY", when), ("BUFR", datetime(2026, 7, 29, 0)))
        check("sounding: a BUFR 400 falls through to FM35 (the RJTY case)",
              S.resolve_source("FM35ONLY", when), ("FM35", datetime(2026, 7, 29, 0)))
        check("sounding: a site in neither feed resolves to nothing",
              S.resolve_source("NOWHERE", when), None)
        check("sounding: last_known_time searches every feed, not just BUFR",
              S.last_known_time("FM35ONLY"), datetime(2026, 7, 29, 0))
        check("sounding: an unknown id still has no record at all",
              S.last_known_time("NOWHERE"), None)
        # The archiver must carry the feed in the identity: the two are different level sets
        # for one ascent, so neither may silently stand in for the other at replay.
        items = ar._expand_sounding("KWRI", when)
        check("sounding: the planner keys on the feed it actually resolved",
              items[0].identity if items else None, "72501/bufr")
    finally:
        S.inventory = real


def test_loop_cadence() -> None:
    """A capture must bridge to the next hourly one, or the archive has a hole every hour."""
    span_min = (ar.DEFAULT_LOOP_FRAMES - 1) * ar.DEFAULT_LOOP_STEP_MIN
    check("loop: the default capture spans a full hour", span_min >= 60, True)
    check("loop: the step is the finest the tool allows, so any coarser request subsamples",
          ar.DEFAULT_LOOP_STEP_MIN, 10)


def test_all_regions() -> None:
    """--all-regions must cover what a station-driven sweep cannot reach."""
    import forecaster.imagery as I

    items = ar.all_region_items(CYCLE)
    ids = {i.identity for i in items}
    reachable = {"conus_east", "northeast", "northern_rockies", "europe", "himawari_japan",
                 "south_america_south"}
    check("regions: covers regions no station resolves to",
          {"full_disk_east/geocolor", "caribbean/geocolor", "africa/geocolor"} <= ids, True)
    check("regions: still covers the station-reachable ones",
          all(any(i.startswith(r + "/") for i in ids) for r in reachable), True)
    check("regions: OSPO geocolor is excluded here too",
          any(i.startswith("himawari_japan/geocolor") for i in ids), False)
    check("regions: Meteosat water vapour is excluded here too",
          any(i.startswith("europe/water_vapor") for i in ids), False)
    check("regions: every identity names a real region",
          all(i.split("/")[0] in I.SAT_REGIONS for i in ids), True)


def test_migration(db: str) -> None:
    """An index created BEFORE fetched_utc existed must upgrade in place, keeping its rows.

    Not hypothetical: the Pi started capturing at 04:19Z on 2026-07-28 and the column landed
    after. Simulated by creating the old shape by hand, then running init_archive_schema."""
    import duckdb
    con = duckdb.connect(db)
    con.execute("CREATE TABLE artifacts (sha256 VARCHAR PRIMARY KEY, kind VARCHAR, "
                "mime VARCHAR, n_bytes BIGINT, first_seen_utc TIMESTAMP)")
    con.execute("CREATE TABLE artifact_keys (kind VARCHAR, identity VARCHAR, "
                "requested_utc TIMESTAMP, served_utc TIMESTAMP, sha256 VARCHAR, "
                "source_url VARCHAR, provider VARCHAR, note VARCHAR, "
                "PRIMARY KEY (kind, identity, requested_utc))")
    con.execute("CREATE TABLE run_manifest (station VARCHAR, cycle_utc TIMESTAMP, "
                "kind VARCHAR, identity VARCHAR, requested_utc TIMESTAMP, "
                "PRIMARY KEY (station, cycle_utc, kind, identity, requested_utc))")
    sha = "c" * 64
    con.execute("INSERT INTO artifacts VALUES (?, 'satellite', 'image/png', 5, now())", [sha])
    con.execute("INSERT INTO artifact_keys VALUES ('satellite', 'old/geocolor', "
                "'2026-07-29 04:00:00', NULL, ?, NULL, 'goes_star', NULL)", [sha])

    store.init_archive_schema(con)                        # the migration under test

    check("migration: the pre-existing row survives",
          con.execute("SELECT count(*) FROM artifact_keys").fetchone()[0], 1)
    check("migration: the old row's fetched_utc is NULL, not fabricated",
          store.artifact_key(con, "satellite", "old/geocolor",
                             datetime(2026, 7, 29, 4))["fetched_utc"], None)
    store.insert_artifact_key(con, kind="satellite", identity="new/geocolor",
                              requested_utc=datetime(2026, 7, 29, 5), sha256="c" * 64,
                              provider="goes_star",
                              fetched_utc=datetime(2026, 7, 29, 5, 33))
    check("migration: new rows record fetched_utc after the upgrade",
          store.artifact_key(con, "satellite", "new/geocolor",
                             datetime(2026, 7, 29, 5))["fetched_utc"],
          datetime(2026, 7, 29, 5, 33))
    store.init_archive_schema(con)                        # twice is a no-op
    check("migration: re-running the migration is a no-op",
          con.execute("SELECT count(*) FROM artifact_keys").fetchone()[0], 2)
    con.close()


def _load():
    """Import the script under test by path -- scripts/ is not a package."""
    import importlib.util
    p = Path(__file__).resolve().parent / "archive_run.py"
    spec = importlib.util.spec_from_file_location("archive_run_mod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["archive_run_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="archive_selftest_") as tmp:
        root = Path(tmp)
        db = str(root / "index.duckdb")
        test_blobs(root)
        test_index(db)
        test_planner()
        test_loop_expand_memo()
        test_render_is_serialized()
        test_capture(root, str(root / "capture.duckdb"))
        test_select_dedup(root, str(root / "dedup.duckdb"))
        test_retry(root, str(root / "retry.duckdb"))
        test_lock_is_waited_out(str(root / "lockwait.duckdb"))
        test_lock_released_during_fetch(root, str(root / "lock.duckdb"))
        test_sounding_feeds()
        test_loop_cadence()
        test_all_regions()
        test_migration(str(root / "legacy.duckdb"))
    print(f"\n{PASS}/{PASS + FAIL} passed.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    ar = _load()
    sys.exit(main())
