"""Fetch the approved surface/upper-air chart set into data/charts/temp/ for review.

A scripted, repeatable version of the manual chart download: walks the wxmaps CATALOG,
pulls each analysis chart once and each GFS forecast chart across the horizon (default
f000-f036 every 6h), and writes code-prefixed files (A1.., B1.., C1..) so they line up
with the review manifest. Refresh anytime by re-running. Uses the wxmaps seam only.
"""

import argparse
from pathlib import Path

from forecaster import wxmaps

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--max-fhr", type=int, default=36, help="max GFS forecast hour (default: 36)")
_ap.add_argument("--step", type=int, default=6, help="forecast-hour step, multiple of 6 (default: 6)")
_ap.add_argument("--outdir", default="data/charts/temp", help="output dir (default: data/charts/temp)")
_args = _ap.parse_args()

OUT = Path(_args.outdir)
OUT.mkdir(parents=True, exist_ok=True)
HORIZON = list(range(0, _args.max_fhr + 1, _args.step))
RUN = wxmaps.latest_gfs_run()

ok = fail = 0
for name, spec in wxmaps.CATALOG.items():
    # Forecast (TT) charts span the horizon; analysis charts are a single "now" image.
    fhrs = HORIZON if spec.source == "tt" else [None]
    for fhr in fhrs:
        if fhr is not None and fhr < spec.params.get("f0", 0):
            continue                    # field has no frame this early (e.g. precip has no f000)
        suffix = f"_f{fhr:03d}" if fhr is not None else ""
        out = OUT / f"{spec.code}_{spec.name}{suffix}.{spec.ext}"
        try:
            img = wxmaps.fetch_map(name, fhr=fhr or 0, run=RUN if spec.source == "tt" else None)
            out.write_bytes(img)
            ok += 1
            print(f"  OK   {out.name:<42} ({len(img)//1024} KB)")
        except Exception as e:  # noqa: BLE001 -- record and continue so one bad chart doesn't sink the run
            fail += 1
            print(f"  FAIL {out.name:<42} {type(e).__name__}: {e}")

print(f"\nGFS run: {RUN:%Y-%m-%dT%H:%MZ}; horizon f{HORIZON[0]:03d}..f{HORIZON[-1]:03d} "
      f"every {_args.step}h")
print(f"{ok} fetched, {fail} failed -> {OUT}")
