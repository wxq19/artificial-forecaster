"""Cycle-stamp self-test (T6 part 2): every network data tool whose product has a source
cycle / valid / fetch time must carry that stamp on its receipt's FIRST line -- that line is
what res.evidence[*].receipt_text and rec.calls[*].result truncate to, so a stamp anywhere
else is invisible to per-run drift analysis.

Pure offline: the fetchers are monkeypatched to return canned bytes/objects, so this asserts
the RECEIPT WIRING, not the network. Restores every patched symbol in a finally.
"""

from datetime import datetime

from forecaster import charts, imagery, soundings, tools, wxmaps
from forecaster.tools import _get_imagery, _get_loop, _get_map, _get_sounding

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))


PNG = b"\x89PNG\r\n\x1a\n_fake"
RUN = datetime(2026, 7, 17, 12, 0)      # a fixed model cycle
SYN = datetime(2026, 7, 17, 12, 0)      # a fixed synoptic time

_saved: dict = {}


def _patch(mod, name, fn):
    _saved[(mod, name)] = getattr(mod, name)
    setattr(mod, name, fn)


def _first(res) -> str:
    return res.text.splitlines()[0]


try:
    # get_sounding: observed synoptic time on line 1.
    _patch(soundings, "synoptic_time", lambda: SYN)
    _patch(soundings, "skewt_url", lambda *a, **k: "http://spc/x.gif")
    _patch(soundings, "fetch_skewt", lambda *a, **k: PNG)
    r = _get_sounding({"site": "OUN", "source": "spc"})
    check("get_sounding: synoptic time on first line", "2026-07-17T12:00Z" in _first(r), _first(r))

    # get_map (TT forecast): GFS run + fhr on line 1.
    _patch(wxmaps, "latest_gfs_run", lambda: RUN)
    _patch(wxmaps, "map_url", lambda *a, **k: "http://tt/x.png")
    _patch(wxmaps, "fetch_map", lambda *a, **k: PNG)
    r = _get_map({"chart": "gfs_500mb", "fhr": 6})
    check("get_map (forecast): run + fhr on first line",
          "run 2026-07-17T12:00Z" in _first(r) and "f006" in _first(r), _first(r))

    # get_fcst_sounding / get_point_forecast USED to be checked here. Since 2026-07-28 they
    # read the model_data archive instead of fetching BUFKIT, so they are no longer network
    # tools and carry no fetch stamp -- their receipts cite the archived model RUN, which is
    # covered by scripts/test_modeldata.py against a real DB rather than a monkeypatch.

    # get_imagery satellite: fetch wall-clock stamp on line 1.
    _patch(imagery, "fetch_satellite", lambda region, product: (PNG, "http://star/x.jpg"))
    r = _get_imagery({"kind": "satellite", "region": "conus_east"})
    check("get_imagery (satellite): fetched-stamp on first line",
          "fetched " in _first(r) and "Z (source" in _first(r), _first(r))

    # get_imagery radar (regional): fetch wall-clock stamp on line 1.
    _patch(imagery, "radar_url", lambda *a, **k: "http://iem/x.png")
    _patch(imagery, "fetch_radar", lambda *a, **k: PNG)
    r = _get_imagery({"kind": "radar", "region": "northwest"})
    check("get_imagery (radar): fetched-stamp on first line", "fetched " in _first(r), _first(r))

    # get_loop: frame time span on line 1 (the observed-time marker).
    frames = [imagery.LoopFrame(datetime(2026, 7, 17, h), f"2026-07-17T{h}:00Z",
                                f"http://cdn/{h}.jpg", PNG) for h in (10, 11, 12)]
    _patch(imagery, "satellite_loop", lambda *a, **k: (frames, "GOES19 CONUS", "CONUS"))
    _patch(charts, "filmstrip", lambda *a, **k: PNG)
    _patch(charts, "loop_mp4", lambda *a, **k: b"\x00mp4")
    _patch(tools.awc, "station_latlon", lambda icao: (44.9, -93.2))
    r = _get_loop({"station": "KMSP"})
    check("get_loop: frame span on first line",
          "2026-07-17T10:00Z -> 2026-07-17T12:00Z" in _first(r), _first(r))
finally:
    for (mod, name), val in _saved.items():
        setattr(mod, name, val)

npass = sum(p for _, p, _ in checks)
print("=== CYCLE-STAMP SELF-TEST (tool receipts) ===")
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if not passed else ""))
print(f"\n{npass}/{len(checks)} passed.")
