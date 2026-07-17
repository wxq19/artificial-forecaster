"""Fallback self-test (T7 + T8): a network data tool degrades to a backup provider on
failure instead of losing the product entirely. No network -- the fetchers are
monkeypatched to raise/succeed on cue.

  T7 get_sounding: SPC fails -> Wyoming (only for a numeric WMO id); the receipt carries the
     note + the Wyoming provenance. A healthy SPC does exactly one fetch. A 3-letter SPC
     site with no WMO gets honest feedback, not a silent second fetch.
  T8 get_map: a TT forecast panel fails -> the closest SPC mesoanalysis analysis, with a
     receipt that says CURRENT ANALYSIS (not the requested fhr). Healthy TT does no SPC fetch.
"""

from datetime import datetime

from forecaster import soundings, wxmaps
from forecaster.tools import _get_map, _get_sounding

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))


PNG = b"\x89PNG\r\n\x1a\n_fake"
RUN = datetime(2026, 7, 17, 12, 0)
_saved: dict = {}


def _patch(mod, name, fn):
    _saved.setdefault((mod, name), getattr(mod, name))
    setattr(mod, name, fn)


def _first(res) -> str:
    return res.text.splitlines()[0]


try:
    _patch(soundings, "synoptic_time", lambda: RUN)
    _patch(soundings, "skewt_url", lambda site, t, *, source="spc": f"http://{source}/x")

    # T7a: SPC fails on a numeric WMO id -> Wyoming served, note + provenance.
    def _fetch_spc_fails(site, t, *, source="spc"):
        if source == "spc":
            raise RuntimeError("HTTP 404")
        return PNG
    _patch(soundings, "fetch_skewt", _fetch_spc_fails)
    r = _get_sounding({"site": "72649", "source": "spc"})
    check("T7: SPC fail on WMO id -> Wyoming image + note",
          bool(r.images) and r.text.startswith("note: SPC unavailable")
          and "source: wyoming" in r.text, _first(r))

    # T7b: SPC fails on a 3-letter site (no WMO) -> honest feedback, no image.
    r = _get_sounding({"site": "OUN", "source": "spc"})
    check("T7: SPC fail on 3-letter site -> feedback, no fallback",
          not r.images and r.text.startswith("error:") and "no WMO number is known" in r.text,
          _first(r))

    # T7c: healthy SPC -> exactly one fetch, no note.
    calls = {"n": 0}
    def _fetch_ok(site, t, *, source="spc"):
        calls["n"] += 1
        return PNG
    _patch(soundings, "fetch_skewt", _fetch_ok)
    r = _get_sounding({"site": "OUN", "source": "spc"})
    check("T7: healthy SPC -> one fetch, no note",
          calls["n"] == 1 and bool(r.images) and not r.text.startswith("note:"), _first(r))

    # T8a: TT panel fails -> SPC mesoanalysis, degradation note names the requested fhr.
    _patch(wxmaps, "latest_gfs_run", lambda: RUN)
    def _map_url(name, *, fhr=0, run=None):
        return f"http://{wxmaps.CATALOG[name].source}/{name}"
    _patch(wxmaps, "map_url", _map_url)
    def _fetch_map_tt_fails(name, *, fhr=0, run=None):
        if wxmaps.CATALOG[name].source == "tt":
            raise RuntimeError("HTTP 403")
        return PNG
    _patch(wxmaps, "fetch_map", _fetch_map_tt_fails)
    r = _get_map({"chart": "gfs_500mb", "fhr": 24})
    check("T8: TT fail -> SPC mesoanalysis image + degradation note",
          bool(r.images) and r.text.startswith("note: forecast panel unavailable")
          and "CURRENT ANALYSIS" in r.text and "f024" in r.text
          and "meso_500mb" in r.text, _first(r))

    # T8b: healthy TT -> no SPC fetch (the map served is the TT panel).
    fetched: list[str] = []
    def _fetch_map_ok(name, *, fhr=0, run=None):
        fetched.append(name)
        return PNG
    _patch(wxmaps, "fetch_map", _fetch_map_ok)
    r = _get_map({"chart": "gfs_500mb", "fhr": 24})
    check("T8: healthy TT -> served the TT panel, no SPC fallback",
          fetched == ["gfs_500mb"] and "[gfs_500mb]" in r.text, _first(r))
finally:
    for (mod, name), val in _saved.items():
        setattr(mod, name, val)

npass = sum(p for _, p, _ in checks)
print("=== FALLBACK SELF-TEST (T7 + T8) ===")
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if not passed else ""))
print(f"\n{npass}/{len(checks)} passed.")
