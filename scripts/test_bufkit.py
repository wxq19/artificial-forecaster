"""Spike: GFS BUFKIT forecast sounding -> enriched skew-T, next to SPC + Wyoming.

Proves the path-1 approach for forecast soundings AND lets us eyeball the rendered
look against the observed products we already fetch. Fetches a GFS BUFKIT file (ISU
mtarchive), parses one forecast-hour profile, and renders an ENRICHED skew-T with
MetPy (parcel path, CAPE/CIN shading, LCL, hodograph inset, BUFKIT indices box) --
all in the uv/PyPI tier, no conda. Then pulls the SPC and Wyoming OBSERVED skew-Ts
for the same station and drops all three in data/charts/temp/ for comparison.
Throwaway spike -- graduates into a fcstsounding seam + a charts.py renderer.
"""

import argparse
import re
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import metpy.calc as mpcalc      # noqa: E402
from metpy.plots import Hodograph, SkewT  # noqa: E402
from metpy.units import units    # noqa: E402

from forecaster import soundings  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--station", default="kmsp", help="lowercase ICAO for BUFKIT (default: kmsp)")
_ap.add_argument("--date", default="2026/07/07", help="run date YYYY/MM/DD")
_ap.add_argument("--run", default="12", help="run cycle HH (default: 12)")
_ap.add_argument("--fhr", type=int, default=24, help="forecast hour to plot (default: 24)")
_ap.add_argument("--spc-site", default="MPX", help="SPC observed site (default: MPX)")
_ap.add_argument("--wyo-wmo", default="72649", help="Wyoming WMO id (default: 72649=MPX)")
_args = _ap.parse_args()

OUT = Path("data/charts/temp")
OUT.mkdir(parents=True, exist_ok=True)
URL = (f"https://mtarchive.geol.iastate.edu/{_args.date}/bufkit/{_args.run}/gfs/"
       f"gfs3_{_args.station}.buf")

# SNPARM order for profile rows: PRES TMPC TMWC DWPC THTE DRCT SKNT OMEG HGHT
_COLS = {"PRES": 0, "TMPC": 1, "DWPC": 3, "DRCT": 5, "SKNT": 6, "HGHT": 8}
# STNPRM stability/moisture indices we surface in the corner box.
_IDX = ["CAPE", "CINS", "LIFT", "SHOW", "KINX", "TOTL", "PWAT", "LCLP"]
_FLOAT = re.compile(r"-?\d+\.\d+")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "artificial-forecaster/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("latin-1")


def parse_block(text: str, fhr: int) -> tuple[dict, dict, str]:
    """Return (level arrays, index dict, valid-time) for forecast hour `fhr`."""
    for blk in re.split(r"(?m)^STID = ", text)[1:]:
        m = re.search(r"STIM = (\d+)", blk)
        if not m or int(m.group(1)) != fhr:
            continue
        valid = re.search(r"TIME = (\S+)", blk).group(1)
        idx = {n: float(v) for n in _IDX
               if (v := (re.search(rf"{n} = (-?\d+\.\d+)", blk) or [None, None])[1])}
        data = blk[blk.find("HGHT") + 4:]
        vals: list[float] = []
        for ln in data.splitlines():
            toks = ln.split()
            if toks and all(_FLOAT.fullmatch(t) for t in toks):
                vals += [float(t) for t in toks]
            elif toks and vals:
                break
        levels = [vals[i:i + 9] for i in range(0, len(vals) - 8, 9)]
        arr = {name: np.array([lv[i] for lv in levels]) for name, i in _COLS.items()}
        return arr, idx, valid
    raise SystemExit(f"forecast hour f{fhr:03d} not found in {URL}")


# --- BUFKIT forecast skew-T (enriched) ---
prof, idx, valid = parse_block(fetch(URL), _args.fhr)
# Drop fill/missing rows: BUFKIT tops the profile with Td = -9999 in the near-vacuum
# upper levels, and plotting that draws a flat dewpoint jog to the axis edge. Stop at
# the end of valid data instead.
good = (prof["DWPC"] > -9000) & (prof["TMPC"] > -9000) & (prof["PRES"] > 0)
prof = {k: val[good] for k, val in prof.items()}
p = prof["PRES"] * units.hPa
T = prof["TMPC"] * units.degC
Td = prof["DWPC"] * units.degC
u, v = mpcalc.wind_components(prof["SKNT"] * units.knots, prof["DRCT"] * units.degrees)
hght = prof["HGHT"] * units.m

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])
skew = SkewT(fig, rotation=45, subplot=gs[0, 0])
skew.plot(p, T, "r", linewidth=2, label="Temperature")
skew.plot(p, Td, "g", linewidth=2, label="Dewpoint")
skew.plot_barbs(p[::3], u[::3], v[::3])
skew.ax.set_ylim(1000, 150)
skew.ax.set_xlim(-40, 50)
skew.ax.set_xlabel("Temperature (°C)")
skew.ax.set_ylabel("Pressure (hPa)")
skew.plot_dry_adiabats(alpha=0.25)
skew.plot_moist_adiabats(alpha=0.25)
skew.plot_mixing_lines(alpha=0.25)

# surface parcel path + CAPE/CIN shading + LCL (MetPy does the thermo)
try:
    parcel = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
    skew.plot(p, parcel, "k", linewidth=1.2, linestyle="--", label="Parcel")
    skew.shade_cape(p, T, parcel)
    skew.shade_cin(p, T, parcel, Td)
    lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])
    skew.plot(lcl_p, lcl_t, "ko", markerfacecolor="black")
    skew.ax.text(lcl_t.m + 1, lcl_p.m, "LCL", fontsize=8, va="center")
except Exception as e:  # noqa: BLE001 -- thermo edge cases shouldn't kill the plot
    print(f"parcel/CAPE skipped: {type(e).__name__}: {e}")

skew.ax.legend(loc="upper center", fontsize=8, ncol=4, frameon=False)

# Right column split so the hodograph sits at the TOP (aligned with the chart top) and
# the indices box directly beneath it -- not floating mid-figure.
gs_r = gs[0, 1].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.08)
ax_h = fig.add_subplot(gs_r[0])
ax_txt = fig.add_subplot(gs_r[1])
ax_txt.axis("off")

# hodograph colored by height, range auto-scaled to the tropospheric winds
top = p >= 200 * units.hPa            # keep the hodograph to the troposphere
spd = np.hypot(u[top].m, v[top].m)
rng = max(30, int(np.ceil((spd.max() if spd.size else 30) / 10) * 10) + 10)
hod = Hodograph(ax_h, component_range=rng)
hod.add_grid(increment=10 if rng <= 40 else 20)
hod.plot_colormapped(u[top], v[top], hght[top])
ax_h.set_title("Hodograph (kt, by height)", fontsize=9)

# indices box beneath the hodograph, top-aligned in its cell
lines = [f"{n:<5}{idx[n]:>8.0f}" if n in ("CAPE", "CINS")
         else f"{n:<5}{idx.get(n, float('nan')):>8.1f}"
         for n in _IDX if n in idx]
ax_txt.text(0.0, 0.0, "GFS BUFKIT indices\n" + "\n".join(lines),
            family="monospace", fontsize=9, va="bottom", ha="left", transform=ax_txt.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#bbb"))

fig.suptitle(
    f"GFS forecast skew-T  |  {_args.station.upper()}  run {_args.date} {_args.run}Z  "
    f"f{_args.fhr:03d}  valid {valid}", fontsize=11)

# SkewT under-fills its gridspec cell, so gridspec alone leaves the right column riding
# above the chart. Pin the hodograph TOP to the chart top and the indices BOTTOM to the
# chart bottom by reading the skew-T's real plot-box extent after a draw.
fig.canvas.draw()
sp = skew.ax.get_position()
fw, fh = fig.get_size_inches()
hp = ax_h.get_position()
hh = hp.width * fw / fh                        # square hodograph (equal aspect fills it)
ax_h.set_position([hp.x0, sp.y1 - hh, hp.width, hh])
tp = ax_txt.get_position()
ax_txt.set_position([tp.x0, sp.y0, tp.width, (sp.y1 - hh - 0.03) - sp.y0])

buf_out = OUT / f"cmp_bufkit_gfs_{_args.station}_f{_args.fhr:03d}.png"
fig.savefig(buf_out, dpi=110, bbox_inches="tight")   # uniform crop -> trims margins, keeps alignment
print(f"BUFKIT: {URL}\n  parsed {len(p)} levels, valid {valid} -> {buf_out}")

# --- SPC + Wyoming observed, same station, for side-by-side comparison ---
spc_out = OUT / f"cmp_spc_{_args.spc_site}.gif"
spc_out.write_bytes(soundings.fetch_skewt(_args.spc_site, source="spc"))
print(f"SPC:    {soundings.skewt_url(_args.spc_site, source='spc')} -> {spc_out}")

wyo_out = OUT / f"cmp_wyoming_{_args.wyo_wmo}.png"
wyo_out.write_bytes(soundings.fetch_skewt(_args.wyo_wmo, source="wyoming"))
print(f"Wyo:    {soundings.skewt_url(_args.wyo_wmo, source='wyoming')} -> {wyo_out}")
